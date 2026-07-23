from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store

from cortex.model_client import get_chat_client
from cortex.workflow.context import WINDOW_KEEP, transcript
from cortex.workflow.types import ChatState

SUMMARY_TRIGGER = 20
SUMMARY_REFRESH = 8
logger = logging.getLogger("cortex.workflow")


async def update_summary(
    state: ChatState, config: RunnableConfig
) -> dict[str, Any]:
    messages = state["messages"]
    covered = state.get("summary_upto", 0)
    if (
        len(messages) <= SUMMARY_TRIGGER
        or len(messages) - WINDOW_KEEP - covered < SUMMARY_REFRESH
    ):
        return {}
    folded = transcript(messages[covered : len(messages) - WINDOW_KEEP])
    if not folded:
        return {"summary_upto": len(messages) - WINDOW_KEEP}
    folded = folded[-12_000:]
    current = state.get("summary", "")[-4_000:]
    prompt = (
        "Maintain a compact running summary of a conversation. Preserve concrete "
        "facts, names, numbers, decisions, and open questions; drop pleasantries.\n\n"
        f"Current summary (may be empty):\n{current or '(none)'}\n\n"
        f"New messages to fold in:\n{folded}\n\n"
        "Return only the updated summary, max ~200 words."
    )
    try:
        result = await get_chat_client(
            config=config,
            effort="low",
            max_output_tokens=350,
        ).ainvoke(
            prompt,
            config={"tags": ["langsmith:nostream"]},
        )
        summary = result.content if isinstance(result.content, str) else str(result.content)
        return {"summary": summary.strip(), "summary_upto": len(messages) - WINDOW_KEEP}
    except Exception:  # noqa: BLE001
        logger.exception("Summary refresh failed, keeping previous summary")
        return {}


async def recall_memories(messages: list) -> str:
    latest = next(
        (message for message in reversed(messages) if isinstance(message, HumanMessage)),
        None,
    )
    if latest is None:
        return ""
    try:
        from cortex.memory import MEMORY_NAMESPACE

        hits = await get_store().asearch(
            MEMORY_NAMESPACE,
            query=str(latest.content),
            limit=4,
        )
        return "\n".join(
            f"- {str(hit.value.get('content', ''))[:500]}" for hit in hits
        )
    except Exception:  # noqa: BLE001
        return ""


async def memory_context(
    state: ChatState, config: RunnableConfig
) -> tuple[str, dict[str, Any]]:
    updates, memories = await asyncio.gather(
        update_summary(state, config),
        recall_memories(state["messages"]),
    )
    summary = updates.get("summary", state.get("summary", ""))
    parts = []
    if summary:
        parts.append(f"## Conversation summary (older context)\n{summary}")
    if memories:
        parts.append(
            "## Long-term memories about the user (from previous conversations)\n"
            f"{memories}"
        )
    return "\n\n".join(parts), updates
