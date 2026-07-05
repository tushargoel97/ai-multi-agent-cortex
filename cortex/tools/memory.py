"""Long-term memory tools — save/recall durable facts across chat threads.

Backed by the LangGraph runtime store (semantic index configured in
langgraph.json). Memories survive restarts together with thread state.
"""

from __future__ import annotations

import logging
import uuid

from langgraph.config import get_store
from pydantic import BaseModel, Field

from cortex.memory import MEMORY_NAMESPACE
from cortex.tools.registry import register_tool

logger = logging.getLogger(__name__)


class SaveMemoryInput(BaseModel):
    """Input for saving a long-term memory."""

    fact: str = Field(
        description=(
            "A single durable fact about the user or their world, phrased in "
            "third person (e.g. 'The user's name is Tushar', 'Prefers concise "
            "answers', 'Main rig has an RTX 4090')."
        )
    )


@register_tool(args_schema=SaveMemoryInput)
async def save_memory(fact: str) -> str:
    """Persist a lasting fact about the user (name, preferences, projects,
    hardware, goals) so future conversations — in any thread — can use it.
    Call this when the user shares something worth remembering long-term."""
    try:
        store = get_store()
        await store.aput(MEMORY_NAMESPACE, str(uuid.uuid4()), {"content": fact})
    except Exception:  # noqa: BLE001 — long-term store down; never break the agent turn
        logger.warning("save_memory failed; long-term store unavailable", exc_info=True)
        return "Long-term memory is temporarily unavailable, so I couldn't save that."
    return f"Saved to long-term memory: {fact}"


class SearchMemoriesInput(BaseModel):
    """Input for searching long-term memories."""

    query: str = Field(description="What to look for in stored memories")


@register_tool(args_schema=SearchMemoriesInput)
async def search_memories(query: str) -> str:
    """Search long-term memories about the user from previous conversations.
    Use when past context (preferences, earlier projects, personal details)
    would improve the answer."""
    try:
        store = get_store()
        hits = await store.asearch(MEMORY_NAMESPACE, query=query, limit=6)
    except Exception:  # noqa: BLE001 — long-term store down; degrade instead of failing
        logger.warning("search_memories failed; long-term store unavailable", exc_info=True)
        return "No stored memories available right now."
    if not hits:
        return "No stored memories match."
    return "\n".join(f"- {h.value.get('content', '')}" for h in hits)
