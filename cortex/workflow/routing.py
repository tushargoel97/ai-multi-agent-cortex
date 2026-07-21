"""Intent classification and workflow routing."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from cortex.declarative import get_agent_spec
from cortex.enums import Agents
from cortex.errors import retryable_model_exceptions
from cortex.model_client import auto_fallback_clients
from cortex.workflow.context import has_image, last_human, message_window
from cortex.workflow.runtime import (
    agent_static_prompt,
    custom_agents_for_routing,
    local_specialists,
    router_classifier_client,
)
from cortex.workflow.synthesis import NOTE_PREFIXES
from cortex.workflow.types import INTENT_TO_NODE, ChatState, Intent, RouterIntent

logger = logging.getLogger("cortex.workflow")

_SPEC_RE = re.compile(
    r"\b(spec|specs|specification|specifications|technical details|capabilities|"
    r"features|compare|comparison|versus|vs\.?|difference between|benchmark|benchmarks)\b",
    re.IGNORECASE,
)
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


def heuristic_intent(messages: list) -> Intent:
    last = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    text = str(last.content) if last is not None else ""
    if _BOOKING_RE.search(text):
        return Intent.BOOKING
    if _SHOPPING_RE.search(text):
        return Intent.SHOPPING
    if _SPEC_RE.search(text):
        return Intent.PRODUCT_SPECS
    return Intent.GENERAL_CHAT


def route_from_start(
    state: ChatState, config: RunnableConfig
) -> Literal["router"]:
    return "router"


def strip_notes(text: str) -> str:
    cut = len(text)
    for prefix in NOTE_PREFIXES:
        index = text.find(prefix)
        if index != -1:
            cut = min(cut, index)
    return text[:cut].rstrip()


async def router(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    configurable = (config or {}).get("configurable") or {}
    mode = str(configurable.get("mode") or "").lower()
    if mode in ("thinking", "research", "engineer"):
        forced = {
            "thinking": Intent.REASONING_TASK,
            "research": Intent.KNOWLEDGE_QUERY,
            "engineer": Intent.CODING_TASK,
        }[mode]
        routing: dict[str, Any] = {
            "intent": forced.value,
            "reasoning": f"{mode.capitalize()} mode",
            "agent": None,
            "local_model": None,
            "complexity": "standard",
        }
        try:
            from cortex.db.services.auto_mode import resolve_auto_model

            resolved = resolve_auto_model(
                "engineer" if mode == "engineer" else forced.value,
                profile="quality" if mode == "thinking" else None,
            )
            if resolved is not None:
                routing["model"] = resolved.model_id
        except Exception:  # noqa: BLE001
            pass
        return {
            "messages": [
                AIMessage(content=forced.value, additional_kwargs={"routing": routing})
            ]
        }

    spec = get_agent_spec(Agents.ROUTER)
    chat_messages: list[Any] = []
    for message in message_window(state["messages"]):
        if not isinstance(message, (HumanMessage, AIMessage)) or getattr(
            message, "tool_calls", None
        ):
            continue
        if isinstance(message, AIMessage) and isinstance(message.content, str):
            chat_messages.append(AIMessage(content=strip_notes(message.content)))
        else:
            chat_messages.append(message)

    custom = custom_agents_for_routing()
    system_prompt = agent_static_prompt(Agents.ROUTER.value, spec)
    if custom:
        listing = "\n".join(f"- {agent['name']}: {agent['description']}" for agent in custom)
        system_prompt += (
            "\n\n## Custom specialized agents\n"
            "If one of these user-defined agents is clearly the best fit for "
            "the latest message, set `agent` to its EXACT name (still pick the "
            "closest `intent` too). Otherwise leave `agent` null.\n" + listing
        )

    from cortex.db.services.auto_mode import is_auto

    specialists = local_specialists() if is_auto(configurable.get("model_id")) else []
    if specialists:
        listing = "\n".join(
            f"- {specialist.model_id}: {specialist.description}"
            for specialist in specialists
        )
        system_prompt += (
            "\n\n## Local specialist models (self-hosted, zero cost)\n"
            "When the latest message falls squarely within one of these models' "
            "described capabilities, set `intent` to `local_specialist` and "
            "`local_model` to the EXACT model name so the query is answered "
            "on-device instead of a paid cloud model. Never send them "
            "out-of-domain queries; when none fits, leave `local_model` null "
            "and never pick `local_specialist`.\n" + listing
        )
    valid_agents = {agent["name"] for agent in custom}
    valid_specialists = {specialist.model_id for specialist in specialists}
    try:

        def make_router(model: Any):
            return create_agent(
                model=model,
                tools=[],
                system_prompt=system_prompt,
                response_format=ProviderStrategy(RouterIntent),
            )

        agent = make_router(router_classifier_client(config))
        fallbacks = auto_fallback_clients(config)
        if fallbacks:
            agent = agent.with_fallbacks(
                [make_router(model) for model in fallbacks],
                exceptions_to_handle=retryable_model_exceptions(),
            )
        result = await agent.ainvoke({"messages": chat_messages})
        intent: RouterIntent = result["structured_response"]
        routing = intent.model_dump(mode="json")
        intent_value = intent.intent.value
        picked_agent = (routing.get("agent") or "").strip()
        routing["agent"] = picked_agent if picked_agent in valid_agents else None
        picked_local = (routing.get("local_model") or "").strip()
        routing["local_model"] = (
            picked_local if picked_local in valid_specialists else None
        )
        if intent_value == Intent.LOCAL_SPECIALIST.value and not routing["local_model"]:
            intent_value = Intent.GENERAL_CHAT.value
            routing["intent"] = intent_value
    except Exception as exc:  # noqa: BLE001
        fallback = heuristic_intent(chat_messages)
        logger.warning(
            "Router model failed (%s: %s), heuristic fallback to %r",
            type(exc).__name__,
            exc,
            fallback.value,
        )
        routing = {
            "intent": fallback.value,
            "reasoning": f"heuristic fallback ({type(exc).__name__})",
            "agent": None,
            "local_model": None,
            "complexity": "standard",
        }
        intent_value = fallback.value
        try:
            from cortex.db.services.auto_mode import FAST_TIER, resolve_auto_model
            from cortex.model_client.model_health import report_model_failure

            if isinstance(exc, retryable_model_exceptions()):
                primary = resolve_auto_model(FAST_TIER)
                if primary is not None:
                    report_model_failure(primary.model_id)
        except Exception:  # noqa: BLE001
            pass

    from cortex.db.services.auto_mode import profile_for_complexity, resolve_auto_model

    if intent_value == Intent.LOCAL_SPECIALIST.value and routing.get("local_model"):
        routing["model"] = routing["local_model"]
    elif is_auto(configurable.get("model_id")) and intent_value != Intent.IMAGE_GENERATION.value:
        try:
            resolved = resolve_auto_model(
                intent_value,
                profile=profile_for_complexity(routing.get("complexity")),
            )
            if resolved is not None:
                routing["model"] = resolved.model_id
        except Exception:  # noqa: BLE001
            logger.exception("auto-mode chip model resolution failed")
    return {
        "messages": [
            AIMessage(content=intent_value, additional_kwargs={"routing": routing})
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
    last_message = state["messages"][-1]
    routing = last_message.additional_kwargs.get("routing", {})
    if routing.get("agent"):
        return "custom_agent"
    intent_value = routing.get("intent", last_message.content.strip().lower())
    try:
        intent = Intent(intent_value)
    except ValueError:
        intent = Intent.GENERAL_CHAT
    node = INTENT_TO_NODE[intent]
    last = last_human(state["messages"])
    return "researcher" if node == "specialist" and last and has_image(last) else node
