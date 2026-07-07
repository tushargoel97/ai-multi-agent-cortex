"""LangGraph workflow, multi-agent assistant with capability-based routing."""

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

import asyncio
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

# Reverse map: built-in agent/node name → its auto-mode intent tier. Used when a
# built-in runs as a subagent so it still resolves its correct model tier.
_NODE_TO_INTENT: dict[str, str] = {
    node: intent.value for intent, node in _INTENT_TO_NODE.items()
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
    # The specialist's raw draft, stashed here instead of emitted as a visible
    # message so the user never sees the 1B prose flash before it is replaced
    # by the spec table. spec_review consumes it and emits the final answer.
    spec_draft: str
    # Set by spec_review to the gap reason when the specialist's answer can't be
    # trusted (refusal / untrained product / fact-check failure), routes the
    # query to the researcher (web-RAG) instead of emitting the bad draft.
    spec_gap: str


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
    The full history stays in graph state (the UI renders it), this only
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
        _persist_logger.exception("Summary refresh failed, keeping previous summary")
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
    except Exception:  # noqa: BLE001, store unavailable outside the runtime
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
    try:
        from cortex.facts import match_products

        if text and match_products(text):
            return Intent.PRODUCT_SPECS  # any trained-domain entity → specialist
    except Exception:  # noqa: BLE001, facts mount optional
        pass
    return Intent.GENERAL_CHAT


def route_from_start(
    state: ChatState, config: RunnableConfig
) -> Literal["router", "specialist"]:
    """Send chats straight to the specialist when a fine-tuned model is the
    effective chat model (dropdown selection OR registry default), small
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
    except Exception:  # noqa: BLE001, registry hiccup: fall through to router
        _persist_logger.exception("route_from_start model resolution failed")
    return "router"


def _strip_notes(text: str) -> str:
    """Drop trailing gap / fact-check notes appended by spec_review / researcher.

    A resumed thread otherwise carries a stale "this hardware isn't in my
    fine-tuned knowledge yet" note that keeps biasing the router toward web-RAG
    even after the model has since been retrained on the product.
    """
    cut = len(text)
    for pfx in _NOTE_PREFIXES:
        idx = text.find(pfx)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].rstrip()


async def router(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Classify user intent using structured output (ProviderStrategy).

    Never fails the run: if the routing model errors (bad key, local model
    without structured-output support, connection loss), fall back to a
    keyword heuristic so the turn still gets answered.
    """
    spec = get_agent_spec(Agents.ROUTER)
    # Recent window only, filtered to human/AI messages, tool call/response
    # pairs from prior agent runs would cause OpenAI to reject the request.
    # Strip trailing gap/fact-check notes from prior answers so a resumed
    # thread's stale "not in my fine-tuned knowledge yet" note can't keep the
    # classifier on web-RAG after the model was retrained on the product.
    chat_messages: list[Any] = []
    for m in _window(state["messages"]):
        if not isinstance(m, (HumanMessage, AIMessage)) or getattr(
            m, "tool_calls", None
        ):
            continue
        if isinstance(m, AIMessage) and isinstance(m.content, str):
            chat_messages.append(AIMessage(content=_strip_notes(m.content)))
        else:
            chat_messages.append(m)
    # Teach the router about admin-created custom agents so it can route to
    # them by description (they register live, no restart / graph rebuild).
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
            "Router model failed (%s: %s), heuristic fallback to %r",
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
    # Deterministic override: if the user names a product that IS in our
    # training facts, the fine-tuned specialist knows it, force product_specs
    # so it answers from weights instead of the router's guess sending a
    # trained product to web-RAG (e.g. "tell me about snapdragon 8 elite gen5"
    # was misread as knowledge_query). Ground truth beats the LLM classifier.
    # A custom-agent pick still wins (it was an explicit, higher-priority match).
    if not routing.get("agent") and intent_value != Intent.PRODUCT_SPECS.value:
        try:
            last_human = _last_human(chat_messages)
            question = _text_content(last_human) if last_human is not None else ""
            from cortex.facts import match_products

            if question and match_products(question):
                intent_value = Intent.PRODUCT_SPECS.value
                routing["intent"] = intent_value
                routing["reasoning"] = (
                    "override: named product is in the fine-tuned training facts"
                )
        except Exception:  # noqa: BLE001, facts mount optional; keep LLM verdict
            pass
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
        except Exception:  # noqa: BLE001, chip model is cosmetic, never fatal
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
    # The specialist is a text-only 1B model, hardware questions that come
    # with an image (e.g. a screenshot of a spec table) go to the researcher,
    # whose vision-capable model can actually read the image.
    last = _last_human(state["messages"])
    if node == "specialist" and last is not None and _has_image(last):
        return "researcher"
    return node


def _system_prompt_for(model: Any, static: str, dynamic: str | None):
    """Anthropic models get a cache_control breakpoint after the static agent
    prompt, the cached prefix covers tool definitions + system, and dynamic
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
    except Exception:  # noqa: BLE001, caching is an optimization, never a blocker
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
        except Exception:  # noqa: BLE001, bad template: fall back to YAML
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
    except Exception:  # noqa: BLE001, never break a run over tool config
        _persist_logger.exception("Agent tool resolution fell back to YAML")
        return spec.get_tools()


def _load_agent_runtime(name: str):
    """(system_prompt, tools, auto_intent) for any agent by name, built-in or
    custom, so it can run as a subagent. None if it doesn't exist or is a
    disabled custom agent. Never includes its OWN subagents, so delegation is a
    single level (no recursion).
    """
    try:
        agent_id = Agents(name)
    except ValueError:
        agent_id = None
    if agent_id is not None:
        try:
            spec = get_agent_spec(agent_id)
        except Exception:  # noqa: BLE001
            return None
        return (
            _agent_static_prompt(name, spec),
            _effective_agent_tools(spec),
            _NODE_TO_INTENT.get(name),
        )

    from cortex.db.services.agents import load_custom_agent
    from cortex.db.services.tool_catalog import (
        effective_tool_names,
        resolve_tool_instances,
    )

    spec = load_custom_agent(name)
    if spec is None:
        return None
    from jinja2 import Template

    try:
        static = Template(spec["system_prompt"]).render(
            assistant_name=_assistant_name()
        )
    except Exception:  # noqa: BLE001, bad template: use it raw
        static = spec.get("system_prompt") or ""
    try:
        tools = resolve_tool_instances(effective_tool_names(name, []))
    except Exception:  # noqa: BLE001
        tools = []
    return static, tools, None


def _subagent_tool(
    subagent_name: str, description: str, config: RunnableConfig | None
):
    """Wrap an agent as a tool the parent can delegate a focused subtask to.

    Isolated context (task in → result out) + read-only shared memory: the
    subagent recalls the shared long-term store but never persists to it (the
    ``save_memory`` write tool is stripped). Only the main agent writes memory.
    """
    from langchain_core.tools import StructuredTool

    async def _delegate(task: str) -> str:
        runtime = _load_agent_runtime(subagent_name)
        if runtime is None:
            return f"The '{subagent_name}' subagent is unavailable."
        static, tools, intent = runtime
        # Read-only memory: never let a subagent persist long-term memories.
        tools = [t for t in tools if getattr(t, "name", "") != "save_memory"]
        recalled = await _recall_memories([HumanMessage(task)])
        system = static
        if recalled:
            system += (
                "\n\n## Shared long-term memory (read-only, recall only, do not "
                f"try to save)\n{recalled}"
            )
        try:
            sub = create_agent(
                model=get_chat_client(config=config, auto_intent=intent),
                tools=tools,
                system_prompt=system,
            )
            result = await sub.ainvoke({"messages": [HumanMessage(task)]})
        except Exception:  # noqa: BLE001, a subagent must never crash the parent
            _persist_logger.exception("Subagent %r failed", subagent_name)
            return f"The '{subagent_name}' subagent could not complete the task."
        msgs = result.get("messages", [])
        final = msgs[-1] if msgs else None
        if (
            isinstance(final, AIMessage)
            and isinstance(final.content, str)
            and final.content.strip()
        ):
            return final.content
        return f"The '{subagent_name}' subagent returned no output."

    safe = re.sub(r"[^a-z0-9_]+", "_", subagent_name.lower()).strip("_") or "subagent"
    tool_desc = (
        f"Delegate a focused subtask to the '{subagent_name}' subagent"
        + (f": {description}" if description else "")
        + ". Give it a clear, self-contained instruction; it works in isolation "
        "and returns its findings for you to use."
    )
    return StructuredTool.from_function(
        coroutine=_delegate, name=f"ask_{safe}", description=tool_desc
    )


def _subagent_tools(agent_name: str, config: RunnableConfig | None) -> list[Any]:
    """Subagent-delegation tools granted to a parent agent (Admin → Agents)."""
    try:
        from cortex.db.services.agents import subagents_for

        subs = subagents_for(agent_name)
    except Exception:  # noqa: BLE001
        return []
    return [
        _subagent_tool(s["name"], s.get("description", ""), config) for s in subs
    ]


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
    # country, kept in the dynamic (post-cache) segment.
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
    tools = _effective_agent_tools(spec) + _subagent_tools(agent_id.value, config)
    return create_agent(
        model=model,
        tools=tools,
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
    """Default chat agent, friendly conversation + identity."""
    return await _run_agent(Agents.GENERALIST, state, config, Intent.GENERAL_CHAT.value)


async def researcher(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Knowledge agent, web search + local KB + Wikipedia.

    Also handles specialist knowledge gaps: when the fine-tuned model refused,
    answered about an untrained product, or was fact-checked wrong, spec_review
    routes here (``spec_gap`` set) so the user gets a web-grounded answer.
    """
    out = await _run_agent(
        Agents.RESEARCHER, state, config, Intent.KNOWLEDGE_QUERY.value
    )
    gap = (state.get("spec_gap") or "").strip()
    if gap:
        out["spec_gap"] = ""  # clear so it doesn't re-trigger
        msgs = out.get("messages") or []
        final = msgs[-1] if msgs else None
        if (
            isinstance(final, AIMessage)
            and isinstance(final.content, str)
            and final.content.strip()
        ):
            note = (
                "\n\n*Fact-checked, my fine-tuned model had some of these specs "
                "wrong, so I've corrected them against verified web sources and "
                "logged it for retraining.*"
                if gap.startswith("fact_error")
                else "\n\n*This hardware isn't in my fine-tuned knowledge yet, so "
                "I answered from verified web sources and logged it for future "
                "training.*"
            )
            final.content += note
    return out


async def reasoner(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Math, logic, and step-by-step problem solving."""
    return await _run_agent(Agents.REASONER, state, config, Intent.REASONING_TASK.value)


async def coder(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Coding specialist, writes, explains, reviews, refactors, and debugs code."""
    return await _run_agent(Agents.CODER, state, config, Intent.CODING_TASK.value)


async def shopping(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Shopping specialist, region-aware product prices and buying advice."""
    return await _run_agent(Agents.SHOPPING, state, config, Intent.SHOPPING.value)


async def booking(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Booking specialist, flights, hotels, movies, concerts, events, and shows."""
    return await _run_agent(Agents.BOOKING, state, config, Intent.BOOKING.value)


async def custom_agent(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Run an admin-created custom agent the router selected by name.

    Custom agents are defined entirely in Admin → Agents (system prompt +
    granted tools) and route live via the router, no graph rebuild needed.
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
        # Selected agent vanished / was disabled mid-turn, answer generally.
        return await generalist(state, config)

    memory_suffix, updates = await _memory_context(state, config)
    model = get_chat_client(config=config)

    from jinja2 import Template

    try:
        static = Template(spec["system_prompt"]).render(
            assistant_name=_assistant_name()
        )
    except Exception:  # noqa: BLE001, bad template: use it raw
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
    tools = tools + _subagent_tools(name, config)

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
    """Image Artist, Google image models behind a two-layer safety gate."""
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
_FALLBACK_NOTE_PREFIX = "*Fact-checked"
_NOTE_PREFIXES = (_GAP_NOTE_PREFIX, _FALLBACK_NOTE_PREFIX)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

_TABLE_RE = re.compile(r"^\s*\|.+\|\s*\n\s*\|[\s:|-]+\|", re.MULTILINE)


def _has_markdown_table(text: str) -> bool:
    """True when the text already contains a markdown table (header + separator)."""
    return bool(_TABLE_RE.search(text))


# Spec / comparison queries (products, hardware, software): these should always
# render as a table, whichever agent answered them.
_SPEC_QUERY_RE = re.compile(
    r"\b(spec|specs|specification|specifications|compare|comparison|compared|"
    r"versus|vs\.?|difference between|which is (?:better|faster|best)|"
    r"pros and cons|benchmarks?)\b",
    re.IGNORECASE,
)


def _numbers_preserved(draft: str, synthesized: str, reference: str = "") -> bool:
    """Fact guard for the synthesizer: no invented numbers, and (without an
    authoritative reference) every draft number must survive. With a
    reference, the synthesizer corrects drifted values against it, so
    numbers may come from either the draft or the reference, nowhere else."""
    draft_nums = set(_NUMBER_RE.findall(draft))
    allowed_extra = {str(n) for n in range(1, 21)}  # step numbers / list indices
    synth_nums = set(_NUMBER_RE.findall(synthesized))
    if reference:
        allowed = draft_nums | set(_NUMBER_RE.findall(reference)) | allowed_extra
        return synth_nums <= allowed
    if len(draft_nums) < 4:
        # Not a spec sheet (e.g. a short math result), the synthesizer is
        # allowed to expand working steps, which adds intermediate numbers.
        return True
    return draft_nums <= synth_nums and (synth_nums - draft_nums) <= allowed_extra


def _no_invented_numbers(source: str, rendered: str, reference: str = "") -> bool:
    """Looser guard for the deterministic table path: the table may reorganize
    or drop values, but must not INVENT a number absent from the draft and the
    authoritative reference (keeps the spec answer honest)."""
    allowed = (
        set(_NUMBER_RE.findall(source))
        | set(_NUMBER_RE.findall(reference))
        | {str(n) for n in range(1, 21)}
    )
    return set(_NUMBER_RE.findall(rendered)) <= allowed


def _carry_notes(source: str, text: str) -> str:
    """Re-append a gap / fact-check note from ``source`` if ``text`` dropped it."""
    for _pfx in _NOTE_PREFIXES:
        if _pfx in source and _pfx not in text:
            return f"{text}\n\n{source[source.index(_pfx):]}"
    return text


def _routed_intent(messages: list) -> str | None:
    """Intent recorded by the router for this turn (newest marker wins)."""
    for m in reversed(messages):
        if _is_router_marker(m):
            return (m.additional_kwargs.get("routing") or {}).get("intent")
    return None


_CODE_BLOCK_RE = re.compile(r"```([\w+-]*)\n(.*?)```", re.DOTALL)


def _code_syntax_notes(text: str) -> list[str]:
    """Parse-only syntax sanity checks on fenced code, never executes anything.

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
                continue  # a snippet, not a full unit, skip
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


async def _render_spec_answer(
    question: str, draft: str, reference: str, resolved: Any
) -> str | None:
    """Turn a spec / comparison draft into a table, rendered deterministically.

    The model only EXTRACTS structured data (columns + rows, copied verbatim);
    ``render_spec_markdown`` renders the markdown, so the result always renders
    as a valid table. Returns None when nothing tabular could be extracted (the
    caller then falls back to the prose formatter).
    """
    from cortex.tools.spec_table import SpecTable, render_spec_markdown

    system = (
        "You convert a draft answer about product, hardware, or software specs "
        "into a structured comparison table. Identify the items being compared "
        "(the columns) and one row per spec. Copy values VERBATIM from the "
        "draft, never invent, round, or drop a value; leave a cell blank when "
        "the draft doesn't give it. Use a single column for a single item. When "
        "an authoritative reference is provided, prefer its values where they "
        "conflict with the draft. Add a one-line verdict only if the draft "
        "states one."
    )
    prompt = f"Question:\n{question}\n\nDraft:\n{draft}"
    if reference:
        prompt += f"\n\nAuthoritative reference (ground truth):\n{reference}"
    try:
        agent = create_agent(
            model=build_client_from_resolved(resolved),
            tools=[],
            system_prompt=system,
            response_format=ProviderStrategy(SpecTable),
        )
        result = await agent.ainvoke({"messages": [HumanMessage(prompt)]})
        table: SpecTable = result["structured_response"]
    except Exception:  # noqa: BLE001, extraction best-effort; fall back to prose
        _persist_logger.exception("Spec table extraction failed")
        return None
    rows = [{"spec": r.spec, "values": list(r.values)} for r in table.rows]
    md = render_spec_markdown(list(table.products), rows, table.verdict or "")
    if not md:
        return None
    if not _no_invented_numbers(draft, md, reference):
        _persist_logger.info("Spec table extraction invented numbers, using prose")
        return None
    return md


async def synthesize(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Formatting pass over factual answers (tables, worked math, structure).

    Rewrites the final AI message in place (same id) so the transcript keeps
    the answering model's usage/name. Presentation only, the prompt forbids
    fact changes, and ANY failure passes the original answer through.
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
    # the fast-tier reformatter (it can silently corrupt it), just run a
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
            or bool(_SPEC_QUERY_RE.search(question))
        )
        # A spec answer that is already a markdown table needs no reformat, 
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
        # the question (same YAMLs the specialist was trained on, no web).
        reference = ""
        prose_domain = False
        try:
            from cortex.facts import (
                is_prose_products,
                match_products,
                reference_block,
            )

            matched = match_products(question)
            reference = reference_block(matched)
            prose_domain = is_prose_products(matched)
        except Exception:  # noqa: BLE001, facts mount optional
            pass

        # Spec / comparison answers: render the table DETERMINISTICALLY. The
        # model only extracts structured data (columns + rows, copied verbatim);
        # the markdown is rendered by code, so it always renders as a valid
        # table instead of relying on the model to format one correctly.
        # Non-hardware domains (games, …) answer in prose instead of a table.
        if is_spec and not prose_domain:
            rendered = await _render_spec_answer(
                question, final.content, reference, resolved
            )
            if rendered:
                return {
                    "messages": [
                        AIMessage(
                            id=final.id,
                            content=_carry_notes(final.content, rendered),
                            additional_kwargs=final.additional_kwargs,
                            response_metadata=final.response_metadata,
                            usage_metadata=final.usage_metadata,
                        )
                    ]
                }

        prompt = f"Question:\n{question}\n\nDraft answer:\n{final.content}"
        if reference:
            prompt += (
                "\n\nAuthoritative reference (ground truth, where the "
                "draft's values for these items disagree, silently use "
                f"these instead):\n{reference}"
            )
        spec = get_agent_spec(Agents.SYNTHESIZER)
        model = build_client_from_resolved(resolved)

        async def _reformat(extra: str = "") -> str:
            res = await model.ainvoke(
                [
                    SystemMessage(
                        spec.render_system_prompt(assistant_name=_assistant_name())
                    ),
                    HumanMessage(prompt + extra),
                ]
            )
            return (
                res.content
                if isinstance(res.content, str)
                else "".join(
                    b.get("text", "") for b in res.content if isinstance(b, dict)
                )
            ).strip()

        text = await _reformat()
        # Enforce the spec table: a fast reformatter sometimes echoes the 1B
        # specialist's prose/bullets, or drifts a number and trips the guard.
        # Retry once, forcing a strict table with every number preserved.
        if is_spec and (
            not _has_markdown_table(text)
            or not _numbers_preserved(final.content, text, reference)
        ):
            forced = await _reformat(
                "\n\nIMPORTANT: Output ONLY a GitHub-flavored markdown table, a "
                "header row, a `| --- |` separator, then one row per spec, with no "
                "prose before or after. Copy EVERY number from the draft verbatim; "
                "never add, drop, round, or invent a number."
            )
            if (
                forced
                and _has_markdown_table(forced)
                and _numbers_preserved(final.content, forced, reference)
            ):
                text = forced
        if not text:
            return {}
        if not _numbers_preserved(final.content, text, reference):
            _persist_logger.warning(
                "Synthesizer altered numbers, keeping the raw answer"
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
    except Exception:  # noqa: BLE001, synthesis must never lose an answer
        _persist_logger.exception("Synthesizer pass failed, keeping raw answer")
        return {}


async def specialist(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Hardware expert, answers from the self-trained model's weights.

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
    # Spec recall wants exactness, greedy decoding so memorized numbers
    # (prices, TFLOPS) come out digit-perfect every time.
    model.temperature = 0.0
    # Send ONLY the latest user question, as plain text. Small fine-tuned
    # models anchor hard on prior turns and will parrot an earlier answer
    # instead of addressing the new question; they also reject tool-role
    # remnants and image blocks. Spec lookup is stateless, one question in,
    # one answer out. NO system prompt: the LoRA training data is bare
    # user/assistant pairs, and prepending one knocks the 1B model off its
    # memorized answers (verified: it duplicates comparison columns).
    last_human = _last_human(state["messages"])
    question_text = _text_content(last_human) if last_human is not None else ""
    if not question_text:
        return {"spec_draft": ""}
    try:
        # ``nostream`` + no message emitted: the raw 1B prose never reaches the
        # UI. spec_review reformats it into the spec table (or replaces it with
        # a frontier answer) and emits the single visible message. This is what
        # removes the ~20s "prose, then table" swap the user saw.
        result = await model.ainvoke(
            [HumanMessage(question_text)],
            config={"tags": ["nostream"]},
        )
        draft = (
            result.content
            if isinstance(result.content, str)
            else "".join(
                b.get("text", "") for b in result.content if isinstance(b, dict)
            )
        ).strip()
    except Exception:  # noqa: BLE001, local service may be down/unloaded
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

    # Hand the draft to spec_review (next node) via state, not as a visible
    # message. It runs the fact-check and the table extraction in parallel and
    # emits the final answer, a spec table, or a frontier web-RAG answer when
    # the draft is wrong / the product isn't in the fine-tune's training.
    return {"spec_draft": draft}


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
    cores, a $40,000 phone SoC) from a good one, the products ARE named, so
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
        return None  # no capable critic available, keep the heuristics' verdict

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
    except Exception:  # noqa: BLE001, critic is best-effort, never fatal
        _persist_logger.exception("Spec draft critique failed")
        return None
    if critique.has_error:
        # Return a short, stable reason code, the KnowledgeGap.reason column is
        # narrow (String(40)); the detail is only useful for the debug log.
        _persist_logger.info(
            "Spec fact-check flagged draft: %s", critique.reason[:200]
        )
        return "fact_error"
    return None


def _untrained_product_reason(question: str) -> str | None:
    """Gap reason when the user names a concrete hardware product the fine-tune
    was never trained on, so any specifics it emitted are ungrounded.

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
    except Exception:  # noqa: BLE001, facts mount optional; don't force a fallback
        return None
    return "product_not_addressed"


def _route_after_spec_review(
    state: ChatState,
) -> Literal["researcher", "__end__"]:
    """Grounded specialist answer ends the run; a gap goes to the researcher
    (web-RAG) for an accurate, sourced answer."""
    return "researcher" if (state.get("spec_gap") or "").strip() else "__end__"


def _specialist_metadata() -> dict[str, Any]:
    """response_metadata carrying the fine-tuned model name, so the UI's model
    badge still attributes the answer to the specialist (it no longer emits
    its own message)."""
    try:
        resolved = resolve_fine_tuned_model()
        if resolved is not None:
            return {"model_name": resolved.model_id}
    except Exception:  # noqa: BLE001
        pass
    return {}


async def spec_review(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Terminal node for the specialist path: emits the single visible answer.

    Runs the fact-check and the spec-table extraction CONCURRENTLY on the
    specialist's (silent) draft, so the table appears in one wall-clock LLM
    round-trip instead of two sequential ones. Then:
      - draft is grounded  → emit the extracted spec table (or the draft as-is
        when nothing tabular could be pulled);
      - draft is wrong / untrained → emit an accurate frontier web-RAG answer
        (already a table) and log the knowledge gap for future fine-tuning.
    """
    draft = (state.get("spec_draft") or "").strip()
    last_human = _last_human(state["messages"])
    if not draft or last_human is None:
        return {}
    question = _text_content(last_human)
    meta = _specialist_metadata()

    from cortex.db.services.knowledge_gaps import detect_gap, log_gap

    # Cheap heuristics decide up front whether the draft is even worth
    # fact-checking / tabulating: an outright refusal or a product not in the
    # training facts skips straight to the frontier fallback.
    heuristic_reason = detect_gap(question, draft) or _untrained_product_reason(
        question
    )

    async def _fact_check() -> str | None:
        # Heuristic gap already found → no need to spend the critic call.
        if heuristic_reason is not None:
            return heuristic_reason
        return await _critique_spec_draft(question, draft)

    async def _table() -> str | None:
        # A clear gap means the draft is untrusted, don't tabulate it; the
        # frontier answer will replace it.
        if heuristic_reason is not None:
            return None
        reference = ""
        prose_domain = False
        try:
            from cortex.facts import (
                is_prose_products,
                match_products,
                reference_block,
            )

            matched = match_products(question)
            reference = reference_block(matched)
            prose_domain = is_prose_products(matched)
        except Exception:  # noqa: BLE001, facts mount optional
            pass
        # Non-hardware domains (games, …) answer in prose, no spec table.
        if prose_domain:
            return None
        resolved = None
        try:
            from cortex.db.services.auto_mode import (
                FAST_TIER,
                resolve_auto_model,
            )

            resolved = resolve_auto_model(
                Intent.KNOWLEDGE_QUERY.value
            ) or resolve_auto_model(FAST_TIER)
        except Exception:  # noqa: BLE001
            resolved = None
        if resolved is None:
            return None
        return await _render_spec_answer(question, draft, reference, resolved)

    # Fact-check and table extraction are independent transforms of the same
    # draft → run them at the same time.
    reason, table_md = await asyncio.gather(_fact_check(), _table())

    if reason is None:
        # Draft is grounded: show the spec table (or the draft if no table
        # could be extracted). Carry any gap/fact-check note the draft holds.
        content = _carry_notes(draft, table_md) if table_md else draft
        return {
            "messages": [AIMessage(content=content, response_metadata=meta)],
            "spec_gap": "",
        }

    # Gap: the draft is a refusal, is about an untrained product, or the
    # fact-check flagged it. A frontier model alone can't be trusted for exact
    # specs, hand the query to the researcher (web-RAG). ``spec_gap`` flips
    # the conditional edge to the researcher node; it web-searches, answers,
    # and synthesize tabulates the result.
    log_gap(question, draft, reason)  # keep the self-improvement signal
    return {"spec_gap": reason}


# ── Graph ────────────────────────────────────────────────────────────────────


def build_workflow(*, checkpointer: Any = None, store: Any = None):
    """Construct and compile the multi-agent Cortex workflow.

    Flow:
        START → (fine-tuned model selected/default? → specialist)
              → router → (conditional)
                          → generalist / prompt_cacher / imagegen      → END
                          → researcher / reasoner / coder → synthesize  → END
                          → specialist (silent draft) → spec_review
                            (parallel fact-check ‖ table extract; emits the
                            table, or a frontier web-RAG answer on a gap) → END

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
    # deterministically (a parse-only syntax check), it never lets the fast
    # model rewrite code.
    builder.add_edge("researcher", "synthesize")
    builder.add_edge("reasoner", "synthesize")
    # The specialist answers silently (draft in state, no prose flash);
    # spec_review fact-checks and tabulates it IN PARALLEL. A grounded answer
    # ends here; a gap (refusal / untrained / fact-check fail) routes to the
    # researcher for a web-grounded answer, then synthesize tabulates it.
    builder.add_edge("specialist", "spec_review")
    builder.add_conditional_edges(
        "spec_review",
        _route_after_spec_review,
        {"researcher": "researcher", "__end__": END},
    )
    builder.add_edge("coder", "synthesize")
    builder.add_edge("synthesize", END)

    # Mirror auto-mode defaults into app_settings so the admin editor (which
    # talks only to Postgres) can show the shipped candidate lists. Best-effort.
    try:
        from cortex.db.services.auto_mode import publish_defaults

        publish_defaults()
    except Exception:  # noqa: BLE001, UI convenience, never blocks graph build
        _persist_logger.exception("publish_defaults at graph build failed")

    return builder.compile(checkpointer=checkpointer, store=store)


# Module-level compiled graph for langgraph.json
graph = build_workflow()
