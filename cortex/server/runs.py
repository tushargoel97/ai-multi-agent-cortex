"""Run streaming — the core SSE endpoint the chat UI consumes.

Maps ``graph.astream(stream_mode=["values","messages","updates","custom"])`` onto
the LangGraph Platform SSE event protocol that ``@langchain/langgraph-sdk``'s
``useStream`` expects. The message-accumulation logic mirrors the reference
server (``langchain-ai/langgraphjs`` · ``libs/langgraph-api/src/stream.mts``):
per-id chunks are folded and re-emitted as ``messages/partial`` for smooth
token streaming, with an initial ``messages/metadata`` frame per message.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from cortex.server.runtime import ASSISTANT_ID, runtime
from cortex.server.serde import is_nostream, message_to_dict, sse
from cortex.server.threads import ensure_thread, set_thread_status

logger = logging.getLogger("cortex.server.runs")

router = APIRouter()

# The UI reconstructs from `values` (full state) and streams tokens from
# `messages`; `updates`/`custom` drive activity indicators and generative UI.
STREAM_MODES = ["values", "messages", "updates", "custom"]

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _build_config(thread_id: str | None, body: dict) -> dict:
    cfg = body.get("config") or {}
    configurable = dict(cfg.get("configurable") or {})
    if thread_id is not None:
        configurable["thread_id"] = str(thread_id)
    checkpoint = body.get("checkpoint") or {}
    if checkpoint.get("checkpoint_id"):
        configurable["checkpoint_id"] = checkpoint["checkpoint_id"]
    if checkpoint.get("checkpoint_ns") is not None:
        configurable["checkpoint_ns"] = checkpoint["checkpoint_ns"]
    config: dict[str, Any] = {"configurable": configurable}
    if cfg.get("tags"):
        config["tags"] = cfg["tags"]
    if cfg.get("recursion_limit"):
        config["recursion_limit"] = cfg["recursion_limit"]
    return config


def _input_or_command(body: dict):
    """Return (input, command). A ``command`` (HITL resume) supersedes input."""
    command = body.get("command")
    if command:
        from langgraph.types import Command

        kwargs: dict[str, Any] = {}
        if "resume" in command:
            kwargs["resume"] = command["resume"]
        if command.get("update") is not None:
            kwargs["update"] = command["update"]
        if command.get("goto") is not None:
            kwargs["goto"] = command["goto"]
        return None, Command(**kwargs)
    return body.get("input"), None


def _message_frames(chunk: Any, seen: dict[str, Any]) -> list[str]:
    message_chunk, metadata = chunk
    if is_nostream(metadata):
        return []
    mid = getattr(message_chunk, "id", None)
    if not mid:
        return []
    frames: list[str] = []
    if mid not in seen:
        seen[mid] = message_chunk
        frames.append(sse("messages/metadata", {mid: {"metadata": metadata}}))
    else:
        seen[mid] = seen[mid] + message_chunk  # BaseMessageChunk accumulates
    frames.append(sse("messages/partial", [message_to_dict(seen[mid])]))
    return frames


async def _stream(thread_id: str | None, body: dict) -> AsyncIterator[str]:
    run_id = str(uuid4())
    yield sse("metadata", {"run_id": run_id, "thread_id": str(thread_id) if thread_id else None})

    config = _build_config(thread_id, body)
    graph_input, command = _input_or_command(body)
    seen: dict[str, Any] = {}

    if thread_id is not None:
        try:
            await set_thread_status(thread_id, "busy")
        except Exception:  # noqa: BLE001
            pass

    try:
        async for mode, chunk in runtime.graph.astream(
            command if command is not None else graph_input,
            config,
            stream_mode=STREAM_MODES,
        ):
            if mode == "messages":
                for frame in _message_frames(chunk, seen):
                    yield frame
            elif mode == "values":
                yield sse("values", chunk)
            elif mode == "updates":
                yield sse("updates", chunk)
            elif mode == "custom":
                yield sse("custom", chunk)
    except Exception as exc:  # noqa: BLE001 — surface as a stream error, don't 500
        logger.exception("Run failed for thread %s", thread_id)
        yield sse("error", {"error": type(exc).__name__, "message": str(exc)})
    finally:
        if thread_id is not None:
            try:
                await set_thread_status(thread_id, "idle")
            except Exception:  # noqa: BLE001
                pass


@router.post("/threads/{thread_id}/runs/stream")
async def stream_run(thread_id: str, request: Request):
    body = await _json(request)
    await ensure_thread(
        thread_id,
        {"graph_id": ASSISTANT_ID, "assistant_id": body.get("assistant_id", ASSISTANT_ID)},
    )
    return StreamingResponse(
        _stream(thread_id, body), media_type="text/event-stream", headers=SSE_HEADERS
    )


@router.post("/runs/stream")
async def stream_run_stateless(request: Request):
    body = await _json(request)
    return StreamingResponse(
        _stream(None, body), media_type="text/event-stream", headers=SSE_HEADERS
    )


async def _json(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return {}
    return body if isinstance(body, dict) else {}
