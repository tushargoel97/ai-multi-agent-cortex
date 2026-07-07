"""FastAPI application — durable, self-hosted LangGraph runtime (Path 2B).

Startup wires a shared psycopg3 pool to an ``AsyncPostgresSaver`` (checkpoints /
threads) and an ``AsyncPostgresStore`` (semantic long-term memory), then compiles
the Cortex graph against them. All state is durable in Postgres — surviving
restarts, rebuilds, and upgrades — with no LangSmith license and no Redis.

Served by uvicorn: ``uvicorn cortex.server.app:app --host 0.0.0.0 --port 8000``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from cortex.memory import EMBED_DIMS, aembed_texts
from cortex.observability import setup_tracing
from cortex.server import assistants, runs, threads
from cortex.server.runtime import db_uri, runtime
from cortex.server.threads import THREADS_DDL
from cortex.workflow import build_workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortex.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_tracing()

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.store.postgres.aio import AsyncPostgresStore

    uri = db_uri()
    if not uri:
        raise RuntimeError("POSTGRES_URI / DATABASE_URL is not set")

    # SEPARATE pools for the checkpointer, the store, and threads-table access.
    # A single shared pool caused psycopg "another command is already in
    # progress" errors: within one graph step the checkpointer write and the
    # store's semantic recall can run concurrently (more so now that
    # spec_review fans out with asyncio.gather), and sharing a pinned
    # connection interleaves two commands on it. Dedicated pools isolate them.
    def _make_pool(max_size: int) -> AsyncConnectionPool:
        return AsyncConnectionPool(
            conninfo=uri,
            max_size=max_size,
            open=False,
            kwargs={"autocommit": True, "row_factory": dict_row},
        )

    cp_pool = _make_pool(20)
    store_pool = _make_pool(10)
    data_pool = _make_pool(5)
    for p in (cp_pool, store_pool, data_pool):
        await p.open()

    checkpointer = AsyncPostgresSaver(cp_pool)
    await checkpointer.setup()

    # pgvector must exist before the store's vector migrations run; init.sql only
    # covers freshly-initialized volumes, so guarantee it here too (idempotent).
    async with store_pool.connection() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    store = AsyncPostgresStore(
        store_pool, index={"dims": EMBED_DIMS, "embed": aembed_texts}
    )
    await store.setup()

    async with data_pool.connection() as conn:
        await conn.execute(THREADS_DDL)

    runtime.pool = data_pool
    runtime.checkpointer = checkpointer
    runtime.store = store
    runtime.graph = build_workflow(checkpointer=checkpointer, store=store)

    # Tool control: mirror built-in tools into the DB and load external tools
    # (LangChain catalog + MCP servers) so admin-granted tools are available.
    try:
        from cortex.db.services.tool_catalog import (
            publish_tool_catalog,
            refresh_dynamic_tools,
        )

        publish_tool_catalog()
        await refresh_dynamic_tools()
    except Exception:  # noqa: BLE001 — tool catalog is additive, never fatal
        logger.exception("Tool catalog init failed")

    try:
        from cortex.db.services.agents import publish_agents

        publish_agents()
    except Exception:  # noqa: BLE001 — agent mirror is additive, never fatal
        logger.exception("Agent init failed")

    logger.info("Cortex durable server ready — Postgres persistence, no license/Redis.")
    try:
        yield
    finally:
        for p in (cp_pool, store_pool, data_pool):
            await p.close()


app = FastAPI(title="Cortex LangGraph Server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assistants.router)
app.include_router(threads.router)
app.include_router(runs.router)


@app.get("/ok")
async def ok():
    return {"ok": True}


@app.get("/info")
async def info():
    return {"version": "0.1.0", "flags": {"assistants": True, "crons": False}}
