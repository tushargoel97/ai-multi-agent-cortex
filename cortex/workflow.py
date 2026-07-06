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
    agent: str | None = Field(
        default=None,
        description=(
            "Exact name of a custom specialized agent to handle this message, "
            "if one clearly fits better than the standard capabilities; "
            "otherwise null."
        ),
    )


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
    r"ps5|playstation|xbox|steam ?deck|nintendo|switch\s?2?|"
    r"rtx|gtx|geforce|radeon|\brx\s?\d|\barc\b|"
    r"ryzen|threadripper|epyc|xeon|intel\s+core|core\s+ultra|i[3579]-\d|"
    r"snapdragon|exynos|mediatek|dimensity|apple\s+silicone?|bionic|"
    r"\bm[1-9]\b|\ba1[0-9]\b|\bchips?\b|chipset|\bsoc\b|processor|"
    r"tflops|\bgpu\b|\bcpu\b|graphics\s+card|nvidia|\bamd\b|h100|h200|b200",
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
    # Teach the router about admin-created custom agents so it can route to
    # them by description (they register live — no restart / graph rebuild).
    custom = _custom_agents_for_routing()
    system_prompt = _agent_static_prompt(Agents.ROUTER.value, spec)
    if custom:
        listing = "\n".join(f"- {a['name']}: {a['description']}" for a in custom)
        system_prompt += (
            "\n\n## Custom specialized agents\n"
            "If one of these user-defined agents is clearly the best fit for "
            "the latest message, set `agent` to its EXACT name (still pick the "
            "closest `intent` too). Otherwise leave `agent` null.\n" + listing
        )
    valid_agents = {a["name"] for a in custom}
    try:
        agent = create_agent(
            model=get_chat_client(config=config),
            tools=[],
            system_prompt=system_prompt,
            response_format=ProviderStrategy(RouterIntent),
        )
        result = await agent.ainvoke({"messages": chat_messages})
        intent: RouterIntent = result["structured_response"]
        routing = intent.model_dump()
        intent_value = intent.intent.value
        picked = (routing.get("agent") or "").strip()
        routing["agent"] = picked if picked in valid_agents else None
    except Exception as e:  # noqa: BLE001
        fallback = _heuristic_intent(chat_messages)
        _persist_logger.warning(
            "Router model failed (%s: %s) — heuristic fallback to %r",
            type(e).__name__,
            e,
            fallback.value,
        )
        routing = {
            "intent": fallback.value,
            "reasoning": f"heuristic fallback ({type(e).__name__})",
            "agent": None,
        }
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
    "custom_agent",
]:
    """Read the router's structured classification and pick the next node."""
    last_msg = state["messages"][-1]
    routing = last_msg.additional_kwargs.get("routing", {})
    if routing.get("agent"):
        return "custom_agent"
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


def _agent_static_prompt(name: str, spec: Any) -> str:
    """Rendered system prompt, honoring an Admin → Agents edit (DB) over YAML."""
    try:
        from cortex.db.services.agents import agent_prompt

        override = agent_prompt(name)
    except Exception:  # noqa: BLE001
        override = None
    if override:
        from jinja2 import Template

        try:
            return Template(override).render(assistant_name=_assistant_name())
        except Exception:  # noqa: BLE001 — bad template: fall back to YAML
            _persist_logger.exception("Agent prompt override render failed")
    return spec.render_system_prompt(assistant_name=_assistant_name())


def _custom_agents_for_routing() -> list[dict]:
    try:
        from cortex.db.services.agents import custom_agents_for_routing

        return custom_agents_for_routing()
    except Exception:  # noqa: BLE001
        return []


def _effective_agent_tools(spec: Any) -> list[Any]:
    """Agent tools with admin overrides applied.

    DB grants (Admin → Tools) replace the YAML whitelist when present and
    globally-disabled tools are dropped; names resolve to built-in, LangChain
    catalog, and MCP instances. Falls back to the YAML/registry tools on any
    error so a bad tool config never breaks a run.
    """
    try:
        from cortex.db.services.tool_catalog import (
            effective_tool_names,
            resolve_tool_instances,
        )

        names = effective_tool_names(spec.name, spec.whitelisted_tools)
        return resolve_tool_instances(names)
    except Exception:  # noqa: BLE001 — never break a run over tool config
        _persist_logger.exception("Agent tool resolution fell back to YAML")
        return spec.get_tools()


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
    static = _agent_static_prompt(agent_id.value, spec)
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
        tools=_effective_agent_tools(spec),
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


async def custom_agent(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Run an admin-created custom agent the router selected by name.

    Custom agents are defined entirely in Admin → Agents (system prompt +
    granted tools) and route live via the router — no graph rebuild needed.
    """
    routing: dict | None = None
    for m in reversed(state["messages"]):
        if _is_router_marker(m):
            routing = m.additional_kwargs.get("routing") or {}
            break
    name = (routing or {}).get("agent")

    from cortex.db.services.agents import load_custom_agent

    spec = load_custom_agent(name) if name else None
    if spec is None:
        # Selected agent vanished / was disabled mid-turn — answer generally.
        return await generalist(state, config)

    memory_suffix, updates = await _memory_context(state, config)
    model = get_chat_client(config=config)

    from jinja2 import Template

    try:
        static = Template(spec["system_prompt"]).render(
            assistant_name=_assistant_name()
        )
    except Exception:  # noqa: BLE001 — bad template: use it raw
        static = spec["system_prompt"]

    now = datetime.now()
    context_line = f"Today's date is {now.strftime('%A, %B')} {now.day}, {now.year}."
    dynamic = f"{context_line}\n\n{memory_suffix}" if memory_suffix else context_line

    from cortex.db.services.tool_catalog import (
        effective_tool_names,
        resolve_tool_instances,
    )

    try:
        tools = resolve_tool_instances(effective_tool_names(name, []))
    except Exception:  # noqa: BLE001
        tools = []

    middleware: list[Any] = [
        PIIMiddleware("credit_card", strategy="redact", apply_to_output=True),
        PIIMiddleware("email", strategy="redact", apply_to_output=True),
    ]
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=_system_prompt_for(model, static, dynamic),
        middleware=middleware,
    )
    result = await agent.ainvoke({"messages": _window(state["messages"])})
    return {"messages": result["messages"], **updates}


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
_FALLBACK_NOTE_PREFIX = "*Answered from live web sources"
_NOTE_PREFIXES = (_GAP_NOTE_PREFIX, _FALLBACK_NOTE_PREFIX)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

_TABLE_RE = re.compile(r"^\s*\|.+\|\s*\n\s*\|[\s:|-]+\|", re.MULTILINE)


def _has_markdown_table(text: str) -> bool:
    """True when the text already contains a markdown table (header + separator)."""
    return bool(_TABLE_RE.search(text))


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
        # A spec answer that is already a markdown table needs no reformat —
        # skip the extra LLM round-trip (it was adding ~10-20s before the
        # table appeared). The web fallback and researcher emit tables directly.
        if is_spec and _has_markdown_table(final.content):
            return {}
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
        for _pfx in _NOTE_PREFIXES:
            if _pfx in final.content and _pfx not in text:
                note = final.content[final.content.index(_pfx):]
                text = f"{text}\n\n{note}"
                break
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

    # Critique + fallback runs in the spec_review node (next in the graph):
    # it logs a knowledge gap and, when the product isn't in the fine-tune's
    # training, answers accurately via a frontier model + web search.
    return {"messages": result["messages"]}


# Web tools the frontier fallback uses to ground an answer (researcher's set).
_FALLBACK_TOOLS = ("web_search", "fetch_url", "search_knowledge_base")


def _untrained_product_reason(question: str) -> str | None:
    """Gap reason when the user names a concrete hardware product the fine-tune
    was never trained on — so any specifics it emitted are ungrounded.

    The facts index (facts.yaml + learned_facts.yaml) is the same ground truth
    the specialist trained on, so a named product missing from it is untrained.
    Returns None when no concrete product is named or the facts mount is absent
    (never force a fallback on uncertainty).
    """
    from cortex.db.services.knowledge_gaps import _PRODUCT_RE

    if not _PRODUCT_RE.search(question):
        return None
    try:
        from cortex.facts import match_products

        if match_products(question):
            return None  # product is in the training facts → trust the specialist
    except Exception:  # noqa: BLE001 — facts mount optional; don't force a fallback
        return None
    return "product_not_addressed"


async def _frontier_spec_answer(
    question: str, config: RunnableConfig
) -> AIMessage | None:
    """Accurate hardware answer from a capable frontier model + web RAG.

    Uses the auto-mode knowledge tier (a strong, non-fine-tuned model) with the
    researcher's web tools so the user still gets a correct, current answer when
    the local specialist couldn't. Best-effort: returns None if no suitable
    model is configured or the tool loop fails.
    """
    from cortex.db.services.auto_mode import resolve_auto_model

    try:
        resolved = resolve_auto_model(Intent.KNOWLEDGE_QUERY.value)
    except Exception:  # noqa: BLE001
        resolved = None
    if resolved is None or resolved.model_id.startswith(FINE_TUNED_PREFIX):
        return None  # no capable non-fine-tuned model available

    from cortex.tools.registry import get_tools

    model = build_client_from_resolved(resolved)
    system_prompt = (
        "You are a hardware specifications expert. The product in the user's "
        "question is NOT in any local knowledge base, so you MUST call the "
        "web_search and fetch_url tools to find authoritative, current "
        "specifications before answering — never guess. Present the specs as a "
        "concise markdown table (Spec | Value) followed by brief source "
        "attribution; if you still cannot verify after searching, say so plainly."
    )
    # Honor admin tool controls: a disabled/removed tool drops out here, so
    # the fallback degrades to the remaining enabled web tools instead of
    # calling it.
    try:
        from cortex.db.services.tool_catalog import (
            filter_enabled,
            resolve_tool_instances,
        )

        fallback_tools = resolve_tool_instances(filter_enabled(list(_FALLBACK_TOOLS)))
    except Exception:  # noqa: BLE001 — tool controls optional; use the raw set
        fallback_tools = get_tools(list(_FALLBACK_TOOLS))
    try:
        agent = create_agent(
            model=model,
            tools=fallback_tools,
            system_prompt=system_prompt,
        )
        result = await agent.ainvoke({"messages": [HumanMessage(question)]})
    except Exception:  # noqa: BLE001 — fallback is best-effort, never fatal
        _persist_logger.exception("Frontier spec fallback failed")
        return None
    fmsgs = result.get("messages", [])
    final = fmsgs[-1] if fmsgs else None
    if (
        isinstance(final, AIMessage)
        and isinstance(final.content, str)
        and final.content.strip()
    ):
        return final
    return None


class _SpecCritique(BaseModel):
    """Structured verdict from the specialist fact-checker."""

    has_error: bool = Field(
        description="True only if the draft has a clear, objective factual error."
    )
    reason: str = Field(
        description="Short description of the worst error, or 'ok' when correct."
    )


async def _critique_spec_draft(question: str, draft: str) -> str | None:
    """The real self-critique: an LLM fact-check of the specialist's draft.

    The keyword heuristics can't tell a confidently-wrong answer (an Apple
    A-series chip labelled 'NVIDIA', a Snapdragon described with AMD 'Zen 4'
    cores, a $40,000 phone SoC) from a good one — the products ARE named, so
    they pass. This asks a capable non-fine-tuned model to judge the draft's
    facts and figures, returning a gap reason on a clear error, else None.
    Best-effort: needs a frontier critic; skips silently otherwise.
    """
    from cortex.db.services.auto_mode import resolve_auto_model

    try:
        resolved = resolve_auto_model(Intent.KNOWLEDGE_QUERY.value)
    except Exception:  # noqa: BLE001
        resolved = None
    if resolved is None or resolved.model_id.startswith(FINE_TUNED_PREFIX):
        return None  # no capable critic available — keep the heuristics' verdict

    system = (
        "You are a strict hardware fact-checker. You receive a user question "
        "and a DRAFT answer written by a small, error-prone model. Set "
        "has_error=true when the draft has ANY clear, objective error:\n"
        "- a product attributed to the wrong maker/brand (e.g. Apple's A-series "
        "or M-series called 'NVIDIA'; Qualcomm's Snapdragon called 'AMD');\n"
        "- an architecture or feature that cannot belong to the named product "
        "(e.g. 'CUDA cores' on a non-NVIDIA chip; AMD 'Zen' / 'Radeon' on a "
        "Snapdragon; an Apple Neural Engine on an Intel CPU);\n"
        "- a physically implausible figure (e.g. a phone/laptop SoC priced at "
        "$40000, drawing 10000 W, or with an absurd memory bandwidth).\n"
        "Do NOT flag rounding, minor spec drift, or wording. Judge only the "
        "draft you are given; do not rewrite it."
    )
    try:
        model = build_client_from_resolved(resolved)
        agent = create_agent(
            model=model,
            tools=[],
            system_prompt=system,
            response_format=ProviderStrategy(_SpecCritique),
        )
        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(f"Question:\n{question}\n\nDraft answer:\n{draft}")
                ]
            }
        )
        critique: _SpecCritique = result["structured_response"]
    except Exception:  # noqa: BLE001 — critic is best-effort, never fatal
        _persist_logger.exception("Spec draft critique failed")
        return None
    if critique.has_error:
        return f"fact_error: {critique.reason[:200]}".strip()
    return None


async def spec_review(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Self-critique the specialist's draft; on a knowledge gap, answer
    accurately via a frontier model + web RAG.

    The gap is always logged for future fine-tuning (the self-improvement
    loop). When the fine-tune wasn't trained on the product — it refused,
    silently answered about something else, or the product isn't in its
    training facts — the user still gets a correct answer from a capable
    frontier model grounded on live web sources.
    """
    msgs = state["messages"]
    final = msgs[-1] if msgs else None
    last_human = _last_human(msgs)
    if not (
        isinstance(final, AIMessage)
        and isinstance(final.content, str)
        and last_human is not None
    ):
        return {}
    question = _text_content(last_human)
    draft = final.content

    from cortex.db.services.knowledge_gaps import detect_gap, log_gap

    # Cheap heuristics first: refusal phrases, a named product missing from the
    # answer, or a product not in the training facts. Then the real
    # self-critique — an LLM fact-check that catches confidently-wrong answers
    # the heuristics can't (wrong brand, impossible architecture, absurd specs).
    reason = detect_gap(question, draft) or _untrained_product_reason(question)
    if reason is None:
        reason = await _critique_spec_draft(question, draft)
    if reason is None:
        return {}  # specialist answer looks grounded → synthesize formats it

    log_gap(question, draft, reason)  # keep the self-improvement signal

    fallback = await _frontier_spec_answer(question, config)
    if fallback is not None:
        if reason.startswith("fact_error"):
            note = (
                "\n\n*Answered from live web sources — my fine-tuned model had "
                "some of these specs wrong, so I've corrected them and logged "
                "it for retraining.*"
            )
        else:
            note = (
                "\n\n*Answered from live web sources — this hardware isn't in my "
                "fine-tuned knowledge yet, so I've logged it for future training.*"
            )
        return {
            "messages": [
                AIMessage(
                    id=final.id,
                    content=fallback.content + note,
                    additional_kwargs=final.additional_kwargs,
                    response_metadata=fallback.response_metadata
                    or final.response_metadata,
                    usage_metadata=fallback.usage_metadata,
                )
            ]
        }

    # No frontier model available — keep the draft with the classic gap note.
    if _GAP_NOTE_PREFIX not in draft:
        return {
            "messages": [
                AIMessage(
                    id=final.id,
                    content=draft
                    + "\n\n*I've logged this as a knowledge gap — research & "
                    "retrain from Admin → Fine-Tuning to teach me this hardware.*",
                    additional_kwargs=final.additional_kwargs,
                    response_metadata=final.response_metadata,
                    usage_metadata=final.usage_metadata,
                )
            ]
        }
    return {}


# ── Graph ────────────────────────────────────────────────────────────────────


def build_workflow(*, checkpointer: Any = None, store: Any = None):
    """Construct and compile the multi-agent Cortex workflow.

    Flow:
        START → (fine-tuned model selected/default? → specialist)
              → router → (conditional)
                          → generalist / prompt_cacher / imagegen      → END
                          → researcher / reasoner / coder → synthesize  → END
                          → specialist → spec_review (self-critique +
                            frontier web-RAG fallback) → synthesize      → END

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
    builder.add_node("spec_review", spec_review)
    builder.add_node("imagegen", imagegen)
    builder.add_node("custom_agent", custom_agent)
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
    builder.add_edge("custom_agent", END)
    # Factual agents get a formatting pass (tables / worked math / structure).
    # The coder shares this node too, but synthesize handles code
    # deterministically (a parse-only syntax check) — it never lets the fast
    # model rewrite code.
    builder.add_edge("researcher", "synthesize")
    builder.add_edge("reasoner", "synthesize")
    # The specialist gets a self-critique pass first: on a knowledge gap it
    # logs the gap and answers accurately via a frontier model + web RAG.
    builder.add_edge("specialist", "spec_review")
    builder.add_edge("spec_review", "synthesize")
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
