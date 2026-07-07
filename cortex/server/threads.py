"""Thread endpoints + Postgres data-access.

Thread *metadata* lives in a small ``threads`` table; thread *state* (messages,
summary) comes from the durable checkpointer via ``graph.aget_state``. This
mirrors the subset of the LangGraph Platform thread API the UI calls:
create / get / search / patch / delete / state / history.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from psycopg.types.json import Jsonb

from cortex.server.runtime import ASSISTANT_ID, runtime
from cortex.server.serde import jsonable

logger = logging.getLogger("cortex.server.threads")

router = APIRouter()

THREADS_DDL = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id  UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata   JSONB NOT NULL DEFAULT '{}'::jsonb,
    status     TEXT  NOT NULL DEFAULT 'idle'
);
CREATE INDEX IF NOT EXISTS threads_metadata_gin ON threads USING gin (metadata);
CREATE INDEX IF NOT EXISTS threads_updated_at_idx ON threads (updated_at DESC);
"""


# ── Data access ──────────────────────────────────────────────────────────────
async def ensure_thread(thread_id: str, metadata: dict | None = None) -> None:
    """Insert the thread if new, else merge extra metadata (idempotent)."""
    async with runtime.pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO threads (thread_id, metadata) VALUES (%s, %s)
            ON CONFLICT (thread_id) DO UPDATE
                SET metadata = threads.metadata || EXCLUDED.metadata,
                    updated_at = now()
            """,
            (str(thread_id), Jsonb(metadata or {})),
        )


async def get_thread_row(thread_id: str) -> dict | None:
    async with runtime.pool.connection() as conn:
        cur = await conn.execute(
            "SELECT * FROM threads WHERE thread_id = %s", (str(thread_id),)
        )
        return await cur.fetchone()


async def search_thread_rows(
    metadata: dict | None, limit: int, offset: int
) -> list[dict]:
    async with runtime.pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT * FROM threads WHERE metadata @> %s
            ORDER BY updated_at DESC LIMIT %s OFFSET %s
            """,
            (Jsonb(metadata or {}), limit, offset),
        )
        return await cur.fetchall()


async def patch_thread_metadata(thread_id: str, metadata: dict) -> None:
    async with runtime.pool.connection() as conn:
        await conn.execute(
            "UPDATE threads SET metadata = metadata || %s, updated_at = now() "
            "WHERE thread_id = %s",
            (Jsonb(metadata or {}), str(thread_id)),
        )


async def set_thread_status(thread_id: str, status: str) -> None:
    async with runtime.pool.connection() as conn:
        await conn.execute(
            "UPDATE threads SET status = %s, updated_at = now() WHERE thread_id = %s",
            (status, str(thread_id)),
        )


async def _thread_values(thread_id: str) -> dict:
    """Latest committed state values for a thread (empty if none yet)."""
    try:
        snap = await runtime.graph.aget_state(
            {"configurable": {"thread_id": str(thread_id)}}
        )
    except Exception:  # noqa: BLE001, a thread with no checkpoint yet
        return {}
    return jsonable(snap.values) if snap and snap.values else {}


async def row_to_thread(row: dict, *, include_values: bool = True) -> dict:
    return {
        "thread_id": str(row["thread_id"]),
        "created_at": _iso(row["created_at"]),
        "updated_at": _iso(row["updated_at"]),
        "metadata": row["metadata"],
        "status": row["status"],
        "values": await _thread_values(row["thread_id"]) if include_values else {},
    }


def _iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


# ── Checkpoint/state serialization ───────────────────────────────────────────
def _checkpoint_ref(config: dict | None) -> dict | None:
    if not config:
        return None
    c = config.get("configurable", {})
    if not c.get("checkpoint_id"):
        return None
    return {
        "thread_id": c.get("thread_id"),
        "checkpoint_ns": c.get("checkpoint_ns", ""),
        "checkpoint_id": c.get("checkpoint_id"),
    }


def _interrupt(it: Any) -> dict:
    if isinstance(it, dict):
        return it
    return {
        "value": jsonable(getattr(it, "value", None)),
        "resumable": getattr(it, "resumable", True),
        "ns": getattr(it, "ns", None),
        "when": getattr(it, "when", "during"),
    }


def _tasks(tasks: Any) -> list[dict]:
    out = []
    for t in tasks or ():
        out.append(
            {
                "id": getattr(t, "id", None),
                "name": getattr(t, "name", None),
                "path": list(getattr(t, "path", ()) or ()),
                "error": getattr(t, "error", None),
                "interrupts": [_interrupt(i) for i in getattr(t, "interrupts", ()) or ()],
                "state": None,
                "result": jsonable(getattr(t, "result", None)),
            }
        )
    return out


def snapshot_to_state(snap: Any) -> dict:
    if snap is None:
        return {
            "values": {},
            "next": [],
            "tasks": [],
            "metadata": {},
            "created_at": None,
            "checkpoint": None,
            "parent_checkpoint": None,
        }
    return {
        "values": jsonable(snap.values),
        "next": list(snap.next),
        "tasks": _tasks(snap.tasks),
        "metadata": jsonable(snap.metadata or {}),
        "created_at": snap.created_at,
        "checkpoint": _checkpoint_ref(snap.config),
        "parent_checkpoint": _checkpoint_ref(snap.parent_config),
    }


# ── Routes ───────────────────────────────────────────────────────────────────
async def _json(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001, empty body is valid for some calls
        return {}
    return body if isinstance(body, dict) else {}


@router.post("/threads")
async def create_thread(request: Request):
    body = await _json(request)
    thread_id = body.get("thread_id") or str(uuid4())
    await ensure_thread(thread_id, body.get("metadata") or {})
    row = await get_thread_row(thread_id)
    return await row_to_thread(row)


@router.post("/threads/search")
async def search_threads(request: Request):
    body = await _json(request)
    rows = await search_thread_rows(
        body.get("metadata") or {},
        int(body.get("limit", 100)),
        int(body.get("offset", 0)),
    )
    return [await row_to_thread(r, include_values=False) for r in rows]


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    row = await get_thread_row(thread_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return await row_to_thread(row)


@router.patch("/threads/{thread_id}")
async def patch_thread(thread_id: str, request: Request):
    body = await _json(request)
    await patch_thread_metadata(thread_id, body.get("metadata") or {})
    row = await get_thread_row(thread_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return await row_to_thread(row)


@router.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    try:
        await runtime.checkpointer.adelete_thread(str(thread_id))
    except Exception:  # noqa: BLE001, older savers may lack adelete_thread
        logger.warning("adelete_thread unavailable; leaving checkpoints for %s", thread_id)
    async with runtime.pool.connection() as conn:
        await conn.execute("DELETE FROM threads WHERE thread_id = %s", (str(thread_id),))
    return {"thread_id": str(thread_id)}


@router.get("/threads/{thread_id}/state")
async def get_thread_state(thread_id: str):
    snap = await runtime.graph.aget_state(
        {"configurable": {"thread_id": str(thread_id)}}
    )
    return snapshot_to_state(snap)


@router.post("/threads/{thread_id}/state")
async def update_thread_state(thread_id: str, request: Request):
    body = await _json(request)
    configurable = {"thread_id": str(thread_id)}
    checkpoint = body.get("checkpoint") or {}
    if checkpoint.get("checkpoint_id"):
        configurable["checkpoint_id"] = checkpoint["checkpoint_id"]
    new_config = await runtime.graph.aupdate_state(
        {"configurable": configurable},
        body.get("values"),
        as_node=body.get("as_node"),
    )
    return {"checkpoint": _checkpoint_ref(new_config)}


@router.post("/threads/{thread_id}/history")
async def get_thread_history(thread_id: str, request: Request):
    body = await _json(request)
    return await _history(thread_id, int(body.get("limit", 10)), body.get("before"))


@router.get("/threads/{thread_id}/history")
async def get_thread_history_get(thread_id: str, limit: int = 10):
    return await _history(thread_id, limit, None)


async def _history(thread_id: str, limit: int, before: Any) -> list[dict]:
    config = {"configurable": {"thread_id": str(thread_id)}}
    before_config = None
    if isinstance(before, dict) and before.get("checkpoint_id"):
        before_config = {
            "configurable": {
                "thread_id": str(thread_id),
                "checkpoint_id": before["checkpoint_id"],
            }
        }
    out: list[dict] = []
    async for snap in runtime.graph.aget_state_history(
        config, limit=limit, before=before_config
    ):
        out.append(snapshot_to_state(snap))
    return out
