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

    # Shared pool: autocommit + dict rows are required by the Postgres saver
    # (pipeline writes) and used by the threads-table data access.
    pool = AsyncConnectionPool(
        conninfo=uri,
        max_size=20,
        open=False,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    await pool.open()

    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()

    # pgvector must exist before the store's vector migrations run; init.sql only
    # covers freshly-initialized volumes, so guarantee it here too (idempotent).
    async with pool.connection() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    store = AsyncPostgresStore(
        pool, index={"dims": EMBED_DIMS, "embed": aembed_texts}
    )
    await store.setup()

    async with pool.connection() as conn:
        await conn.execute(THREADS_DDL)

    runtime.pool = pool
    runtime.checkpointer = checkpointer
    runtime.store = store
    runtime.graph = build_workflow(checkpointer=checkpointer, store=store)

    logger.info("Cortex durable server ready — Postgres persistence, no license/Redis.")
    try:
        yield
    finally:
        await pool.close()


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
