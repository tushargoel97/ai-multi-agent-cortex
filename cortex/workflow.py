"""LangGraph workflow — multi-agent assistant with capability-based routing."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

import logging
import re
from datetime import datetime

from cortex.config import get_settings
from cortex.db.services.llm_registry import (
    FINE_TUNED_PREFIX,
    build_client_from_resolved,
    resolve_fine_tuned_model,
    resolve_with_session,
)
from cortex.declarative import get_agent_spec
from cortex.enums import Agents
from cortex.model_client import get_chat_client
from cortex.observability import setup_tracing

# Initialise tracing before any LangChain objects are constructed.
setup_tracing()

_persist_logger = logging.getLogger("cortex.workflow")


# ── Router Schema ────────────────────────────────────────────────────────────


class Intent(StrEnum):
    GENERAL_CHAT = "general_chat"
    KNOWLEDGE_QUERY = "knowledge_query"
    REASONING_TASK = "reasoning_task"
    PROMPT_CACHING = "prompt_caching"
    PRODUCT_SPECS = "product_specs"
    IMAGE_GENERATION = "image_generation"
    CODING_TASK = "coding_task"
    SHOPPING = "shopping"
    BOOKING = "booking"


class RouterIntent(BaseModel):
    """Structured output from the intent-detection router."""

    intent: Intent = Field(description="Classified capability needed for the user's message.")
    reasoning: str = Field(description="One-sentence justification for the chosen intent.")


# ── Intent → Node mapping ───────────────────────────────────────────────────

_INTENT_TO_NODE: dict[Intent, str] = {
    Intent.GENERAL_CHAT: "generalist",
    Intent.KNOWLEDGE_QUERY: "researcher",
    Intent.REASONING_TASK: "reasoner",
    Intent.PROMPT_CACHING: "prompt_cacher",
    Intent.PRODUCT_SPECS: "specialist",
    Intent.IMAGE_GENERATION: "imagegen",
    Intent.CODING_TASK: "coder",
    Intent.SHOPPING: "shopping",
    Intent.BOOKING: "booking",
}


def _assistant_name() -> str:
    return get_settings().assistant_name


# ── Memory: short-term summary + long-term store recall ─────────────────────

WINDOW_KEEP = 12      # newest messages passed verbatim to agents
SUMMARY_TRIGGER = 20  # start summarizing once the thread outgrows this
SUMMARY_REFRESH = 8   # fold-in cadence: new msgs beyond covered point


class ChatState(MessagesState):
    """Graph state = full message history + rolling summary bookkeeping."""

    summary: str
    summary_upto: int  # number of leading messages already folded into summary


def _is_router_marker(m: object) -> bool:
    return isinstance(m, AIMessage) and "routing" in (m.additional_kwargs or {})


def _has_image(m: object) -> bool:
    content = getattr(m, "content", None)
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict) and "image" in str(b.get("type", ""))
        for b in content
    )


def _text_content(m: object) -> str:
    """Plain text of a message whose content may be a block list."""
    content = getattr(m, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return str(content or "")


def _last_human(messages: list) -> HumanMessage | None:
    return next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)


def _window(messages: list) -> list:
    """Newest WINDOW_KEEP messages, cleaned for model consumption.

    Drops router intent markers and never starts on an orphan tool result.
    The full history stays in graph state (the UI renders it) — this only
    bounds what agents re-read each turn.
    """
    msgs = [m for m in messages if not _is_router_marker(m)]
    w = msgs[-WINDOW_KEEP:]
    while w and isinstance(w[0], ToolMessage):
        w.pop(0)
    return w


def _transcript(messages: list) -> str:
    lines = []
    for m in messages:
        if _is_router_marker(m) or getattr(m, "tool_calls", None):
            continue
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage) and isinstance(m.content, str) and m.content:
            lines.append(f"Assistant: {m.content}")
    return "\n".join(lines)


async def _update_summary(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Fold older messages into the rolling summary (short-term memory).

    Returns state updates ({} when nothing to do). Full history is preserved;
    the summary lets agents keep context without re-reading everything.
    """
    msgs = state["messages"]
    upto = state.get("summary_upto", 0)
    if len(msgs) <= SUMMARY_TRIGGER or len(msgs) - WINDOW_KEEP - upto < SUMMARY_REFRESH:
        return {}

    to_fold = _transcript(msgs[upto : len(msgs) - WINDOW_KEEP])
    if not to_fold:
        return {"summary_upto": len(msgs) - WINDOW_KEEP}

    existing = state.get("summary", "")
    prompt = (
        "Maintain a compact running summary of a conversation. Preserve concrete "
        "facts, names, numbers, decisions, and open questions; drop pleasantries.\n\n"
        f"Current summary (may be empty):\n{existing or '(none)'}\n\n"
        f"New messages to fold in:\n{to_fold}\n\n"
        "Return only the updated summary, max ~200 words."
    )
    try:
        result = await get_chat_client(config=config).ainvoke(
            prompt, config={"tags": ["langsmith:nostream"]}
        )
        summary = result.content if isinstance(result.content, str) else str(result.content)
        return {"summary": summary.strip(), "summary_upto": len(msgs) - WINDOW_KEEP}
    except Exception:  # noqa: BLE001
        _persist_logger.exception("Summary refresh failed — keeping previous summary")
        return {}


async def _recall_memories(messages: list) -> str:
    """Semantic recall from the long-term store for the latest user message."""
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    if last_human is None:
        return ""
    try:
        from cortex.memory import MEMORY_NAMESPACE

        store = get_store()
        hits = await store.asearch(
            MEMORY_NAMESPACE, query=str(last_human.content), limit=4
        )
        return "\n".join(f"- {h.value.get('content', '')}" for h in hits)
    except Exception:  # noqa: BLE001 — store unavailable outside the runtime
        return ""


async def _memory_context(
    state: ChatState, config: RunnableConfig
) -> tuple[str, dict[str, Any]]:
    """Build the system-prompt memory suffix and any summary state updates."""
    updates = await _update_summary(state, config)
    summary = updates.get("summary", state.get("summary", ""))
    memories = await _recall_memories(state["messages"])

    parts = []
    if summary:
        parts.append(f"## Conversation summary (older context)\n{summary}")
    if memories:
        parts.append(
            "## Long-term memories about the user (from previous conversations)\n"
            f"{memories}"
        )
    return "\n\n".join(parts), updates


# ── Nodes ────────────────────────────────────────────────────────────────────

# Safety net when the routing model is unavailable or emits garbage (e.g. a
# small local model that can't do structured output): classify by keywords
# instead of failing the whole run.
_HARDWARE_RE = re.compile(
    r"ps5|playstation|xbox|steam ?deck|nintendo|switch 2|rtx|geforce|radeon|"
    r"ryzen|intel core|i9-|i7-|core ultra|tflops|gpu|cpu|graphics card|nvidia|amd|h100|b200",
    re.IGNORECASE,
)
# Buy/price intent beats a bare hardware keyword: "where to buy a PS5 Pro" is
# shopping, not a spec question. Booking wins over both.
_SHOPPING_RE = re.compile(
    r"\b(buy|purchase|order|shop|price|prices|pricing|cost|cheap|cheapest|"
    r"deal|deals|discount|coupon|best price|where to buy|how much)\b",
    re.IGNORECASE,
)
_BOOKING_RE = re.compile(
    r"\b(book|booking|reserve|reservation|flight|flights|hotel|hotels|ticket|"
    r"tickets|concert|movie|movies|show|shows|event|events)\b",
    re.IGNORECASE,
)


def _heuristic_intent(messages: list) -> Intent:
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    text = str(last_human.content) if last_human is not None else ""
    if _BOOKING_RE.search(text):
        return Intent.BOOKING
    if _SHOPPING_RE.search(text):
        return Intent.SHOPPING
    if _HARDWARE_RE.search(text):
        return Intent.PRODUCT_SPECS
    return Intent.GENERAL_CHAT


def route_from_start(
    state: ChatState, config: RunnableConfig
) -> Literal["router", "specialist"]:
    """Send chats straight to the specialist when a fine-tuned model is the
    effective chat model (dropdown selection OR registry default) — small
    fine-tuned models cannot run the router's structured-output
    classification, and picking them IS the intent."""
    configurable = (config or {}).get("configurable") or {}
    if configurable.get("local_base_url"):
        return "router"  # "Use local LLM" toggle: user points at their own endpoint
    from cortex.db.services.auto_mode import is_auto

    if is_auto(configurable.get("model_id")):
        return "router"  # auto mode: intent decides the model, never the bypass
    last = _last_human(state["messages"])
    if last is not None and _has_image(last):
        return "router"  # images need a vision model, never the 1B specialist
    try:
        resolved = resolve_with_session(configurable.get("model_id"))
        if resolved and resolved.model_id.startswith(FINE_TUNED_PREFIX):
            return "specialist"
    except Exception:  # noqa: BLE001 — registry hiccup: fall through to router
        _persist_logger.exception("route_from_start model resolution failed")
    return "router"


async def router(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Classify user intent using structured output (ProviderStrategy).

    Never fails the run: if the routing model errors (bad key, local model
    without structured-output support, connection loss), fall back to a
    keyword heuristic so the turn still gets answered.
    """
    spec = get_agent_spec(Agents.ROUTER)
    # Recent window only, filtered to human/AI messages — tool call/response
    # pairs from prior agent runs would cause OpenAI to reject the request.
    chat_messages = [
        m
        for m in _window(state["messages"])
        if isinstance(m, (HumanMessage, AIMessage))
        and not getattr(m, "tool_calls", None)
    ]
    try:
        agent = create_agent(
            model=get_chat_client(config=config),
            tools=[],
            system_prompt=spec.render_system_prompt(assistant_name=_assistant_name()),
            response_format=ProviderStrategy(RouterIntent),
        )
        result = await agent.ainvoke({"messages": chat_messages})
        intent: RouterIntent = result["structured_response"]
        routing = intent.model_dump()
        intent_value = intent.intent.value
    except Exception as e:  # noqa: BLE001
        fallback = _heuristic_intent(chat_messages)
        _persist_logger.warning(
            "Router model failed (%s: %s) — heuristic fallback to %r",
            type(e).__name__,
            e,
            fallback.value,
        )
        routing = {"intent": fallback.value, "reasoning": f"heuristic fallback ({type(e).__name__})"}
        intent_value = fallback.value
    # In auto mode, record which model this intent resolves to so the UI's
    # routing chip can show it. Skip image_generation: its real model is only
    # known after generation (the imagegen node reports result.model_used), so
    # a pre-guess (the first image candidate) would be misleading.
    configurable = (config or {}).get("configurable") or {}
    from cortex.db.services.auto_mode import is_auto, resolve_auto_model

    if (
        is_auto(configurable.get("model_id"))
        and intent_value != Intent.IMAGE_GENERATION.value
    ):
        try:
            resolved = resolve_auto_model(intent_value)
            if resolved is not None:
                routing["model"] = resolved.model_id
        except Exception:  # noqa: BLE001 — chip model is cosmetic, never fatal
            _persist_logger.exception("auto-mode chip model resolution failed")
    return {
        "messages": [
            AIMessage(
                content=intent_value,
                additional_kwargs={"routing": routing},
            )
        ]
    }


def route_by_intent(
    state: ChatState,
) -> Literal[
    "generalist",
    "researcher",
    "reasoner",
    "coder",
    "prompt_cacher",
    "specialist",
    "imagegen",
    "shopping",
    "booking",
]:
    """Read the router's structured classification and pick the next node."""
    last_msg = state["messages"][-1]
    routing = last_msg.additional_kwargs.get("routing", {})
    intent_value = routing.get("intent", last_msg.content.strip().lower())

    try:
        intent = Intent(intent_value)
    except ValueError:
        intent = Intent.GENERAL_CHAT  # safe fallback

    node = _INTENT_TO_NODE[intent]
    # The specialist is a text-only 1B model — hardware questions that come
    # with an image (e.g. a screenshot of a spec table) go to the researcher,
    # whose vision-capable model can actually read the image.
    last = _last_human(state["messages"])
    if node == "specialist" and last is not None and _has_image(last):
        return "researcher"
    return node


def _system_prompt_for(model: Any, static: str, dynamic: str | None):
    """Anthropic models get a cache_control breakpoint after the static agent
    prompt — the cached prefix covers tool definitions + system, and dynamic
    context (memories, summary) stays after the breakpoint so it never breaks
    the cache. Other providers cache automatically; they get a plain string."""
    try:
        from langchain_anthropic import ChatAnthropic

        if isinstance(model, ChatAnthropic):
            blocks: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": static,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            if dynamic:
                blocks.append({"type": "text", "text": dynamic})
            return SystemMessage(content=blocks)
    except Exception:  # noqa: BLE001 — caching is an optimization, never a blocker
        pass
    return f"{static}\n\n{dynamic}" if dynamic else static


def _build_agent(
    agent_id: Agents,
    *,
    config: RunnableConfig | None = None,
    with_pii: bool = True,
    extra_system: str | None = None,
    auto_intent: str | None = None,
):
    spec = get_agent_spec(agent_id)
    middleware: list[Any] = []
    if with_pii:
        # Redact common PII categories in model output.
        middleware.append(
            PIIMiddleware("credit_card", strategy="redact", apply_to_output=True)
        )
        middleware.append(
            PIIMiddleware("email", strategy="redact", apply_to_output=True)
        )
    model = get_chat_client(config=config, auto_intent=auto_intent)
    static = spec.render_system_prompt(assistant_name=_assistant_name())
    # Inject today's date + the user's browser region so agents resolve
    # "9th July" to the right year and default shopping/booking to the user's
    # country — kept in the dynamic (post-cache) segment.
    from cortex.tools.commerce import region_from_browser

    now = datetime.now()
    cfg = (config or {}).get("configurable") or {}
    region = region_from_browser(
        str(cfg.get("locale") or ""), str(cfg.get("timezone") or "")
    )
    context_line = (
        f"Today's date is {now.strftime('%A, %B')} {now.day}, {now.year}. "
        f"The user appears to be in region {region} (from their browser); use "
        f"it as the default country for shopping, booking, prices, and local "
        f"results unless they say otherwise."
    )
    dynamic = f"{context_line}\n\n{extra_system}" if extra_system else context_line
    return create_agent(
        model=model,
        tools=spec.get_tools(),
        system_prompt=_system_prompt_for(model, static, dynamic),
        middleware=middleware,
    )


async def _run_agent(
    agent_id: Agents,
    state: ChatState,
    config: RunnableConfig,
    auto_intent: str | None = None,
) -> dict[str, Any]:
    """Shared node body: memory context + windowed history + agent invoke."""
    memory_suffix, updates = await _memory_context(state, config)
    agent = _build_agent(
        agent_id,
        config=config,
        extra_system=memory_suffix or None,
        auto_intent=auto_intent,
    )
    result = await agent.ainvoke({"messages": _window(state["messages"])})
    return {"messages": result["messages"], **updates}


async def generalist(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Default chat agent — friendly conversation + identity."""
    return await _run_agent(Agents.GENERALIST, state, config, Intent.GENERAL_CHAT.value)


async def researcher(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Knowledge agent — local KB + Wikipedia."""
    return await _run_agent(Agents.RESEARCHER, state, config, Intent.KNOWLEDGE_QUERY.value)


async def reasoner(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Math, logic, and step-by-step problem solving."""
    return await _run_agent(Agents.REASONER, state, config, Intent.REASONING_TASK.value)


async def coder(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Coding specialist — writes, explains, reviews, refactors, and debugs code."""
    return await _run_agent(Agents.CODER, state, config, Intent.CODING_TASK.value)


async def shopping(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Shopping specialist — region-aware product prices and buying advice."""
    return await _run_agent(Agents.SHOPPING, state, config, Intent.SHOPPING.value)


async def booking(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Booking specialist — flights, hotels, movies, concerts, events, and shows."""
    return await _run_agent(Agents.BOOKING, state, config, Intent.BOOKING.value)


async def prompt_cacher(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """LLM-prompt-caching expert (large system prompt, demonstrates caching)."""
    return await _run_agent(Agents.PROMPT_CACHER, state, config, Intent.PROMPT_CACHING.value)


async def imagegen(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Image Artist — Google image models behind a two-layer safety gate."""
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    prompt = str(last_human.content) if last_human is not None else ""
    configurable = (config or {}).get("configurable") or {}
    thread_id = str(configurable.get("thread_id") or "thread")

    from cortex.imagegen import generate_image

    result = await generate_image(prompt, thread_id)
    if result.status == "ok":
        caption = result.detail or "Here you go:"
        content = f"{caption}\n\n![Generated image](/api/images/{result.filename})"
        message = AIMessage(
            content=content,
            response_metadata={"model_name": result.model_used},
        )
    else:
        message = AIMessage(content=result.detail)
    return {"messages": [message]}


_GAP_NOTE_PREFIX = "*I've logged this as a knowledge gap"

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _numbers_preserved(draft: str, synthesized: str, reference: str = "") -> bool:
    """Fact guard for the synthesizer: no invented numbers, and (without an
    authoritative reference) every draft number must survive. With a
    reference, the synthesizer corrects drifted values against it — so
    numbers may come from either the draft or the reference, nowhere else."""
    draft_nums = set(_NUMBER_RE.findall(draft))
    allowed_extra = {str(n) for n in range(1, 21)}  # step numbers / list indices
    synth_nums = set(_NUMBER_RE.findall(synthesized))
    if reference:
        allowed = draft_nums | set(_NUMBER_RE.findall(reference)) | allowed_extra
        return synth_nums <= allowed
    if len(draft_nums) < 4:
        # Not a spec sheet (e.g. a short math result) — the synthesizer is
        # allowed to expand working steps, which adds intermediate numbers.
        return True
    return draft_nums <= synth_nums and (synth_nums - draft_nums) <= allowed_extra


def _routed_intent(messages: list) -> str | None:
    """Intent recorded by the router for this turn (newest marker wins)."""
    for m in reversed(messages):
        if _is_router_marker(m):
            return (m.additional_kwargs.get("routing") or {}).get("intent")
    return None


_CODE_BLOCK_RE = re.compile(r"```([\w+-]*)\n(.*?)```", re.DOTALL)


def _code_syntax_notes(text: str) -> list[str]:
    """Parse-only syntax sanity checks on fenced code — never executes anything.

    Conservative on purpose: only flags substantial, complete-looking Python
    and any JSON, so intentional snippets / pseudocode don't false-positive.
    True validation (running the code) is the sandbox roadmap item.
    """
    notes: list[str] = []
    for lang, body in _CODE_BLOCK_RE.findall(text):
        lang = lang.strip().lower()
        if lang in ("python", "py"):
            lines = [ln for ln in body.splitlines() if ln.strip()]
            if len(lines) < 4 or not re.search(
                r"^\s*(?:async\s+def|def|class|import|from)\b", body, re.M
            ):
                continue  # a snippet, not a full unit — skip
            import ast

            try:
                ast.parse(body)
            except SyntaxError as e:
                notes.append(f"Python block near line {e.lineno}: {e.msg}")
        elif lang == "json":
            import json as _json

            try:
                _json.loads(body)
            except ValueError:
                notes.append("JSON block: not valid JSON")
    return notes


async def synthesize(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Formatting pass over factual answers (tables, worked math, structure).

    Rewrites the final AI message in place (same id) so the transcript keeps
    the answering model's usage/name. Presentation only — the prompt forbids
    fact changes — and ANY failure passes the original answer through.
    """
    msgs = state["messages"]
    final = msgs[-1] if msgs else None
    if not (
        isinstance(final, AIMessage)
        and isinstance(final.content, str)
        and final.content.strip()
    ):
        return {}
    # Coding answers reuse this node but deterministically: never hand code to
    # the fast-tier reformatter (it can silently corrupt it) — just run a
    # parse-only syntax check and append a heads-up when a full block is broken.
    if _routed_intent(msgs) == Intent.CODING_TASK.value:
        notes = _code_syntax_notes(final.content)
        if not notes:
            return {}
        note = "\n\n> ⚠️ **Syntax check:** " + "; ".join(notes[:3])
        return {
            "messages": [
                AIMessage(
                    id=final.id,
                    content=final.content + note,
                    additional_kwargs=final.additional_kwargs,
                    response_metadata=final.response_metadata,
                    usage_metadata=final.usage_metadata,
                )
            ]
        }
    last_human = _last_human(msgs)
    question = _text_content(last_human) if last_human is not None else ""
    try:
        from cortex.db.services.auto_mode import FAST_TIER, resolve_auto_model

        # Hardware/spec answers get a stronger formatter than the fast utility
        # tier so the spec-sheet table is produced reliably (a tiny fast model
        # often leaves the 1B specialist's prose as-is). Detect them by the
        # router intent or a fine-tuned answering model (specialist bypass).
        # Everything else stays on the fast tier for speed/cost.
        model_name = str((final.response_metadata or {}).get("model_name", ""))
        is_spec = (
            _routed_intent(msgs) == Intent.PRODUCT_SPECS.value
            or model_name.startswith(FINE_TUNED_PREFIX)
        )
        resolved = (
            resolve_auto_model(Intent.KNOWLEDGE_QUERY.value) if is_spec else None
        )
        if resolved is None:
            resolved = resolve_auto_model(FAST_TIER)
        if resolved is None:
            return {}
        # Ground drifted numbers: authoritative specs for products named in
        # the question (same YAMLs the specialist was trained on — no web).
        reference = ""
        try:
            from cortex.facts import match_products, reference_block

            reference = reference_block(match_products(question))
        except Exception:  # noqa: BLE001 — facts mount optional
            pass
        prompt = f"Question:\n{question}\n\nDraft answer:\n{final.content}"
        if reference:
            prompt += (
                "\n\nAuthoritative spec reference (ground truth — where the "
                "draft's values for these products disagree, silently use "
                f"these instead):\n{reference}"
            )
        spec = get_agent_spec(Agents.SYNTHESIZER)
        model = build_client_from_resolved(resolved)
        result = await model.ainvoke(
            [
                SystemMessage(
                    spec.render_system_prompt(assistant_name=_assistant_name())
                ),
                HumanMessage(prompt),
            ]
        )
        text = (
            result.content
            if isinstance(result.content, str)
            else "".join(
                b.get("text", "") for b in result.content if isinstance(b, dict)
            )
        ).strip()
        if not text:
            return {}
        if not _numbers_preserved(final.content, text, reference):
            _persist_logger.warning(
                "Synthesizer altered numbers — keeping the raw answer"
            )
            return {}
        if _GAP_NOTE_PREFIX in final.content and _GAP_NOTE_PREFIX not in text:
            note = final.content[final.content.index(_GAP_NOTE_PREFIX):]
            text = f"{text}\n\n{note}"
        replacement = AIMessage(
            id=final.id,
            content=text,
            additional_kwargs=final.additional_kwargs,
            response_metadata=final.response_metadata,
            usage_metadata=final.usage_metadata,
        )
        return {"messages": [replacement]}
    except Exception:  # noqa: BLE001 — synthesis must never lose an answer
        _persist_logger.exception("Synthesizer pass failed — keeping raw answer")
        return {}


async def specialist(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Hardware expert — answers from the self-trained model's weights.

    Always uses the fine-tuned model from the registry (model_id prefixed
    `finetuned-` under the local provider), regardless of the model selected
    in the chat dropdown. No tools: no RAG, no web search.
    """
    resolved = resolve_fine_tuned_model()
    if resolved is None:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "No fine-tuned hardware model is registered yet. "
                        "Train one in Admin → Fine-Tuning, then click "
                        "'Convert & Register'."
                    )
                )
            ]
        }

    model = build_client_from_resolved(resolved)
    # Spec recall wants exactness — greedy decoding so memorized numbers
    # (prices, TFLOPS) come out digit-perfect every time.
    model.temperature = 0.0
    # NO system prompt: the LoRA training data is bare user/assistant pairs,
    # and prepending one knocks the 1B model off its memorized answers
    # (verified: it duplicates comparison columns with a system prompt).
    agent = create_agent(model=model, tools=[], system_prompt=None)
    # Send ONLY the latest user question, as plain text. Small fine-tuned
    # models anchor hard on prior turns and will parrot an earlier answer
    # instead of addressing the new question; they also reject tool-role
    # remnants and image blocks. Spec lookup is stateless — one question in,
    # one answer out.
    last_human = _last_human(state["messages"])
    question_text = _text_content(last_human) if last_human is not None else ""
    chat_messages = [HumanMessage(question_text)] if question_text else []
    try:
        result = await agent.ainvoke({"messages": chat_messages})
    except Exception:  # noqa: BLE001 — local service may be down/unloaded
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"The fine-tuned model '{resolved.model_id}' could not be "
                        "reached on the local LLM service. Check that the ai "
                        "service is running and the model is loaded "
                        "(Admin → Local Models)."
                    )
                )
            ]
        }

    # Self-improvement loop: if the model refused or silently ignored a
    # product it wasn't trained on, log the question as a knowledge gap.
    # Admin → Fine-Tuning researches gaps (web) and retrains — the model
    # itself never uses the web at answer time.
    messages = result["messages"]
    final = messages[-1] if messages else None
    if (
        final is not None
        and isinstance(final, AIMessage)
        and isinstance(final.content, str)
        and last_human is not None
    ):
        from cortex.db.services.knowledge_gaps import detect_gap, log_gap

        reason = detect_gap(question_text, final.content)
        if reason and log_gap(question_text, final.content, reason):
            final.content += (
                "\n\n*I've logged this as a knowledge gap — research & retrain "
                "from Admin → Fine-Tuning to teach me this hardware.*"
            )
    return {"messages": messages}


# ── Graph ────────────────────────────────────────────────────────────────────


def build_workflow(*, checkpointer: Any = None, store: Any = None):
    """Construct and compile the multi-agent Cortex workflow.

    Flow:
        START → (fine-tuned model selected/default? → specialist)
              → router → (conditional)
                          → generalist / prompt_cacher / imagegen      → END
                          → researcher / reasoner / specialist / coder → synthesize → END

    Auto mode: configurable.model_id == "auto" always goes through the router
    and each node resolves its model per intent (declarative/auto_mode.yaml).

    Guardrails:
        - PIIMiddleware: redacts credit-card numbers and emails in model output.

    Memory:
        - Short-term: rolling conversation summary in state (``summary``),
          refreshed as threads outgrow the message window.
        - Long-term: LangGraph store (namespace ``memories``) with semantic
          index (langgraph.json → cortex/memory.py); agents recall relevant
          facts automatically and can save new ones via the save_memory tool.

    Persistence:
        Durable. The custom server (cortex/server) compiles this graph with an
        ``AsyncPostgresSaver`` checkpointer and ``AsyncPostgresStore``, so thread
        state and long-term memory live in Postgres and survive restarts,
        rebuilds, and upgrades. Pass ``checkpointer``/``store`` in; when omitted
        (e.g. a bare ``langgraph dev`` for debugging) the runtime supplies its own.
    """
    builder = StateGraph(ChatState)

    # Nodes
    builder.add_node("router", router)
    builder.add_node("generalist", generalist)
    builder.add_node("researcher", researcher)
    builder.add_node("reasoner", reasoner)
    builder.add_node("coder", coder)
    builder.add_node("shopping", shopping)
    builder.add_node("booking", booking)
    builder.add_node("prompt_cacher", prompt_cacher)
    builder.add_node("specialist", specialist)
    builder.add_node("imagegen", imagegen)
    builder.add_node("synthesize", synthesize)

    # Edges
    # Fine-tuned model selected in the chat dropdown → bypass the router
    # (deterministic, no structured output on small local models).
    builder.add_conditional_edges(START, route_from_start)
    builder.add_conditional_edges("router", route_by_intent)
    builder.add_edge("generalist", END)
    builder.add_edge("prompt_cacher", END)
    builder.add_edge("imagegen", END)
    builder.add_edge("shopping", END)
    builder.add_edge("booking", END)
    # Factual agents get a formatting pass (tables / worked math / structure).
    # The coder shares this node too, but synthesize handles code
    # deterministically (a parse-only syntax check) — it never lets the fast
    # model rewrite code.
    builder.add_edge("researcher", "synthesize")
    builder.add_edge("reasoner", "synthesize")
    builder.add_edge("specialist", "synthesize")
    builder.add_edge("coder", "synthesize")
    builder.add_edge("synthesize", END)

    # Mirror auto-mode defaults into app_settings so the admin editor (which
    # talks only to Postgres) can show the shipped candidate lists. Best-effort.
    try:
        from cortex.db.services.auto_mode import publish_defaults

        publish_defaults()
    except Exception:  # noqa: BLE001 — UI convenience, never blocks graph build
        _persist_logger.exception("publish_defaults at graph build failed")

    return builder.compile(checkpointer=checkpointer, store=store)


# Module-level compiled graph for langgraph.json
graph = build_workflow()
