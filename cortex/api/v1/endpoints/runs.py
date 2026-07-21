"""Durable LangGraph run streaming endpoints."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from cortex.api.dependencies.runtime import ASSISTANT_ID, runtime
from cortex.api.v1.serialization import is_nostream, message_to_dict, sse
from cortex.api.v1.endpoints.threads import ensure_thread, set_thread_status

logger = logging.getLogger("cortex.api.v1.endpoints.runs")

router = APIRouter()

STREAM_MODES = ["values", "messages", "updates", "custom"]

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

_IDEM_TTL = 30.0
_recent_keys: dict[str, float] = {}


def _evict_keys(now: float) -> None:
    for k in [k for k, ts in _recent_keys.items() if now - ts > _IDEM_TTL]:
        _recent_keys.pop(k, None)


def _idempotency_key(thread_id: str, body: dict) -> str | None:
    """An explicit key, else the last human message id (stable per submit)."""
    meta = body.get("metadata")
    if isinstance(meta, dict) and meta.get("idempotency_key"):
        return f"{thread_id}:{meta['idempotency_key']}"
    inp = body.get("input")
    msgs = inp.get("messages") if isinstance(inp, dict) else None
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("type") == "human" and m.get("id"):
                return f"{thread_id}:{m['id']}"
    return None


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


_SENTINEL = object()  # end-of-run marker pushed to subscriber queues
_RUN_TTL = 30.0  # keep a finished run this long so a late reconnect can replay
_bg_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> asyncio.Task:
    """``create_task`` + hold a strong ref so it isn't GC'd mid-flight."""
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return task


class _Run:
    __slots__ = ("run_id", "key", "buffer", "subscribers", "done", "task")

    def __init__(self, run_id: str, key: str | None) -> None:
        self.run_id = run_id
        self.key = key
        self.buffer: list[str] = []
        self.subscribers: set[asyncio.Queue] = set()
        self.done = False
        self.task: asyncio.Task | None = None

    def publish(self, frame: str) -> None:
        self.buffer.append(frame)
        for q in self.subscribers:
            q.put_nowait(frame)

    def finish(self) -> None:
        self.done = True
        for q in self.subscribers:
            q.put_nowait(_SENTINEL)

    def attach(self) -> tuple[asyncio.Queue, list[str]]:
        q: asyncio.Queue = asyncio.Queue()
        replay = list(self.buffer)
        if not self.done:
            self.subscribers.add(q)
        return q, replay


_active: dict[str, _Run] = {}


async def drain_active_runs(timeout: float = 170) -> int:
    tasks = {
        run.task
        for run in _active.values()
        if run.task is not None and not run.task.done()
    }
    if not tasks:
        return 0
    _, pending = await asyncio.wait(tasks, timeout=timeout)
    return len(pending)


async def _subscribe(run: _Run) -> AsyncIterator[str]:
    q, replay = run.attach()
    for frame in replay:
        yield frame
    if run.done:
        return
    try:
        while True:
            frame = await q.get()
            if frame is _SENTINEL:
                break
            yield frame
    finally:
        run.subscribers.discard(q)


async def _run_graph(thread_id: str, body: dict, run: _Run) -> None:
    """Run the graph to completion, detached from any HTTP connection."""
    run.publish(sse("metadata", {"run_id": run.run_id, "thread_id": thread_id}))
    config = _build_config(thread_id, body)
    graph_input, command = _input_or_command(body)
    seen: dict[str, Any] = {}
    try:
        async for mode, chunk in runtime.graph.astream(
            command if command is not None else graph_input,
            config,
            stream_mode=STREAM_MODES,
        ):
            if mode == "messages":
                for frame in _message_frames(chunk, seen):
                    run.publish(frame)
            elif mode == "values":
                run.publish(sse("values", chunk))
            elif mode == "updates":
                run.publish(sse("updates", chunk))
            elif mode == "custom":
                run.publish(sse("custom", chunk))
    except asyncio.CancelledError:  # server shutdown: let it propagate
        raise
    except Exception as exc:  # noqa: BLE001, surface as a stream error, don't 500
        logger.exception("Run failed for thread %s", thread_id)
        run.publish(sse("error", {"error": type(exc).__name__, "message": str(exc)}))
    finally:
        run.finish()
        _spawn(_release(thread_id, run))  # detached: runs even on cancel


async def _release(thread_id: str, run: _Run) -> None:
    try:
        await set_thread_status(thread_id, "idle")
    except Exception:  # noqa: BLE001
        pass
    await asyncio.sleep(_RUN_TTL)
    if _active.get(thread_id) is run:
        _active.pop(thread_id, None)


def _cancel_active(thread_id: str, run_id: str | None = None) -> bool:
    """Cancel the thread's active run task. Returns False if there's nothing to
    cancel (or the run_id doesn't match)."""
    run = _active.get(thread_id)
    if run is None or (run_id is not None and run.run_id != run_id):
        return False
    task = run.task
    if task is not None and not task.done():
        task.cancel()
    return True


async def _stream(body: dict) -> AsyncIterator[str]:
    """Inline stream for the stateless (thread-less) run endpoint."""
    yield sse("metadata", {"run_id": str(uuid4()), "thread_id": None})
    config = _build_config(None, body)
    graph_input, command = _input_or_command(body)
    seen: dict[str, Any] = {}
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
    except Exception as exc:  # noqa: BLE001, surface as a stream error, don't 500
        logger.exception("Stateless run failed")
        yield sse("error", {"error": type(exc).__name__, "message": str(exc)})


@router.post("/threads/{thread_id}/runs/stream")
async def stream_run(thread_id: str, request: Request):
    body = await _json(request)
    await ensure_thread(
        thread_id,
        {"graph_id": ASSISTANT_ID, "assistant_id": body.get("assistant_id", ASSISTANT_ID)},
    )
    tid = str(thread_id)
    now = time.monotonic()
    _evict_keys(now)
    idem = _idempotency_key(tid, body)

    active = _active.get(tid)
    if active is not None and not active.done:
        if idem and active.key == idem:
            return StreamingResponse(
                _subscribe(active),
                media_type="text/event-stream",
                headers=SSE_HEADERS,
            )
        return JSONResponse(
            {
                "error": "run_in_progress",
                "message": "A run is already in progress for this conversation.",
            },
            status_code=409,
        )

    if idem and idem in _recent_keys:
        return JSONResponse(
            {"error": "duplicate_run", "message": "Duplicate request ignored."},
            status_code=409,
        )
    if idem:
        _recent_keys[idem] = now

    run = _Run(str(uuid4()), idem)
    _active[tid] = run
    try:
        await set_thread_status(tid, "busy")
    except Exception:  # noqa: BLE001
        pass
    run.task = _spawn(_run_graph(tid, body, run))
    return StreamingResponse(
        _subscribe(run), media_type="text/event-stream", headers=SSE_HEADERS
    )


@router.get("/threads/{thread_id}/runs/{run_id}/stream")
async def join_run_stream(thread_id: str, run_id: str):
    """Re-attach to a still-running run's stream (resume on navigating back)."""
    run = _active.get(str(thread_id))
    if run is None or run.run_id != run_id:
        return JSONResponse({"error": "not_found"}, status_code=404)
    return StreamingResponse(
        _subscribe(run), media_type="text/event-stream", headers=SSE_HEADERS
    )


@router.post("/threads/{thread_id}/runs/{run_id}/cancel")
async def cancel_run(thread_id: str, run_id: str):
    """Cancel a specific in-flight run (the SDK's stop() path)."""
    ok = _cancel_active(str(thread_id), run_id)
    return JSONResponse({"ok": ok}, status_code=200 if ok else 404)


@router.post("/threads/{thread_id}/runs/cancel")
async def cancel_thread_run(thread_id: str):
    """Cancel the thread's active run (the UI Cancel button's path)."""
    return JSONResponse({"ok": _cancel_active(str(thread_id))})


@router.post("/runs/stream")
async def stream_run_stateless(request: Request):
    body = await _json(request)
    return StreamingResponse(
        _stream(body), media_type="text/event-stream", headers=SSE_HEADERS
    )


async def _json(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return {}
    return body if isinstance(body, dict) else {}
