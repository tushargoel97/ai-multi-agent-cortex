"""Memory support, embeddings for the LangGraph store index + shared constants.

Long-term memory lives in the LangGraph runtime store (namespace ``memories``),
semantically indexed via the ``store.index`` config in langgraph.json which
points at :func:`aembed_texts` below. Under `langgraph dev` the store (and its
vectors) is persisted to the same volume-backed pickles as thread state.

Short-term memory is a rolling conversation summary kept in graph state
(``ChatState.summary`` in ``cortex/workflow/types.py``).
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

EMBED_DIMS = 1536

# Single-user app: one global namespace. Switch to ("memories", user_id)
# if/when the UI grows user identities.
MEMORY_NAMESPACE = ("memories",)


async def aembed_texts(texts: list[str]) -> list[list[float]]:
    """Embedding hook for the LangGraph store index (see langgraph.json).

    Uses the registry-aware embedding client (OpenAI key from the DB registry
    or env). On failure it returns zero vectors so memory writes still succeed
, semantic ranking degrades until embeddings are configured.
    """
    texts = list(texts)
    try:
        from cortex.model_client import get_embedding_client

        client = get_embedding_client()
        return await asyncio.to_thread(client.embed_documents, texts)
    except Exception:  # noqa: BLE001
        logger.exception(
            "Embedding failed, storing zero vectors (semantic memory search degraded)"
        )
        return [[0.0] * EMBED_DIMS for _ in texts]
