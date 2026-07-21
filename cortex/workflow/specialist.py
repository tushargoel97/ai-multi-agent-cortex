from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from cortex.local_grounding import answer_with_local_specialist
from cortex.workflow.context import agent_context, is_router_marker, last_human, text_content
from cortex.workflow.planning import plan_from_messages
from cortex.workflow.types import ChatState

logger = logging.getLogger("cortex.workflow")


def _error(content: str) -> dict[str, Any]:
    return {"messages": [AIMessage(content=content)]}


async def run_local_specialist(
    state: ChatState, config: RunnableConfig, model_id: str
) -> dict[str, Any]:
    latest = last_human(state["messages"])
    if latest is None or not text_content(latest).strip():
        return _error(f"The local specialist '{model_id}' received no question.")
    plan = plan_from_messages(state["messages"], "local_specialist")
    try:
        reply = await answer_with_local_specialist(
            model_id,
            state["messages"],
            agent_context(config, complexity=plan.complexity),
        )
    except Exception:  # noqa: BLE001
        logger.exception("Local specialist %r failed", model_id)
        reply = None
    if not isinstance(reply, AIMessage) or not text_content(reply).strip():
        return _error(
            f"The local specialist '{model_id}' is unavailable or returned no answer."
        )
    answer = text_content(reply).strip()
    from cortex.db.services.knowledge_gaps import detect_gap, log_gap

    if reason := detect_gap(text_content(latest), answer):
        log_gap(text_content(latest), answer, reason)
    reply.response_metadata = {
        **(reply.response_metadata or {}),
        "model_name": model_id,
    }
    reply.additional_kwargs = {
        **(reply.additional_kwargs or {}),
        "execution_tier": plan.tier,
    }
    return {"messages": [reply]}


async def specialist(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    route = next(
        (
            message.additional_kwargs.get("routing", {})
            for message in reversed(state["messages"])
            if is_router_marker(message)
        ),
        {},
    )
    model_id = str(route.get("local_model") or "").strip()
    return (
        await run_local_specialist(state, config, model_id)
        if model_id
        else _error("No local specialist model was selected for this request.")
    )
