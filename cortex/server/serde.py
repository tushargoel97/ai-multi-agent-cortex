"""Wire-format helpers, speak the LangGraph Platform serialization the SDK expects.

The ``@langchain/langgraph-sdk`` client deserializes messages as plain dicts
keyed by ``type`` (``"ai"``/``"human"``/``"tool"``/``"system"``), NOT the
LangChain "lc-constructor" envelope. Message *chunks* are first coerced to full
messages so their ``type`` is the base type (``"ai"``) rather than
``"AIMessageChunk"``, matching the reference server (langgraph-api/stream.mts).
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import (
    BaseMessage,
    BaseMessageChunk,
    message_chunk_to_message,
)

# Tags that mark internal LLM calls whose tokens must never surface in the UI
# stream (guardrail screen_prompt, summary refresh: see cortex/workflow.py).
NOSTREAM_TAGS = {"langsmith:nostream", "nostream"}


def message_to_dict(message: Any) -> Any:
    """Serialize a LangChain message (or chunk) to the SDK's plain-dict shape."""
    if isinstance(message, BaseMessageChunk):
        message = message_chunk_to_message(message)
    if isinstance(message, BaseMessage):
        return message.model_dump()
    return message


def jsonable(obj: Any) -> Any:
    """Recursively convert graph state (messages → dicts) to JSON-native data."""
    if isinstance(obj, BaseMessage):
        return message_to_dict(obj)
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def is_nostream(metadata: Any) -> bool:
    """True when a streamed message chunk is tagged as internal (do not surface)."""
    if not isinstance(metadata, dict):
        return False
    tags = metadata.get("tags") or []
    return any(tag in NOSTREAM_TAGS for tag in tags)


def sse(event: str, data: Any) -> str:
    """Format a single Server-Sent Events frame."""
    payload = json.dumps(jsonable(data), default=str, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"
