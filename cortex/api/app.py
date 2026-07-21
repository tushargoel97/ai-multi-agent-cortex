"""Versioned Cortex API application with durable Postgres state."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from psycopg.errors import UndefinedTable
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from cortex.memory import EMBED_DIMS, aembed_texts
from cortex.observability import setup_tracing
from cortex.api.dependencies.runtime import db_uri, runtime
from cortex.api.middleware.cors import configure_cors
from cortex.api.v1.endpoints import runs
from cortex.api.v1.endpoints.threads import THREADS_DDL
from cortex.api.v1.router import router as api_v1_router
from cortex.workflow import build_workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortex.api")


def checkpoint_serializer():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("cortex.workflow", "Intent"),
            ("cortex.workflow.types", "Intent"),
        ]
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_tracing()

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.store.postgres.aio import AsyncPostgresStore

    uri = db_uri()
    if not uri:
        raise RuntimeError("POSTGRES_URI / DATABASE_URL is not set")

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

    checkpointer = AsyncPostgresSaver(cp_pool, serde=checkpoint_serializer())
    await checkpointer.setup()

    async with store_pool.connection() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    store = AsyncPostgresStore(
        store_pool, index={"dims": EMBED_DIMS, "embed": aembed_texts}
    )
    await store.setup()

    async with data_pool.connection() as conn:
        await conn.execute(THREADS_DDL)
        await conn.execute(
            "ALTER TABLE IF EXISTS llm_models "
            "ADD COLUMN IF NOT EXISTS description TEXT"
        )
        try:
            await conn.execute(
                "UPDATE llm_providers SET base_url = "
                "regexp_replace(base_url, '/v1/?$', '/api/v1') "
                "WHERE kind = 'local' AND base_url ~ "
                "'^http://(ai|localhost|host\\.docker\\.internal):8100/v1/?$'"
            )
        except UndefinedTable:
            logger.debug("Local provider URL migration skipped", exc_info=True)

    runtime.pool = data_pool
    runtime.checkpointer = checkpointer
    runtime.store = store
    runtime.graph = build_workflow(checkpointer=checkpointer, store=store)

    try:
        from cortex.api.v1.endpoints.threads import reset_stale_runs

        freed = await reset_stale_runs()
        if freed:
            logger.info("Reset %d stale busy thread(s) on startup", freed)
    except Exception:  # noqa: BLE001, recovery is best-effort
        logger.exception("reset_stale_runs failed")

    try:
        from cortex.db.services.tool_catalog import (
            publish_tool_catalog,
            refresh_dynamic_tools,
        )

        publish_tool_catalog()
        await refresh_dynamic_tools()
    except Exception:  # noqa: BLE001, tool catalog is additive, never fatal
        logger.exception("Tool catalog init failed")

    try:
        from cortex.db.services.agents import publish_agents
        from cortex.db.services.auto_mode import publish_defaults

        publish_agents()
        publish_defaults()
    except Exception:  # noqa: BLE001, agent mirror is additive, never fatal
        logger.exception("Agent init failed")

    try:
        from sqlalchemy import text as _text

        from cortex.db.engine import engine as _engine

        _EM = "\u2014"
        _targets = [
            ("tools", "description"),
            ("agents", "description"),
            ("agents", "system_prompt"),
            ("llm_models", "display_name"),
            ("llm_providers", "name"),
            ("knowledge_articles", "title"),
            ("knowledge_articles", "content"),
        ]
        for _t, _c in _targets:
            stmt = _text(
                f"UPDATE {_t} SET {_c} = "
                f"regexp_replace({_c}, :pat, ', ', 'g') WHERE {_c} LIKE :like"
            )
            try:
                with _engine.begin() as _conn:
                    _conn.execute(
                        stmt, {"pat": r"\s*" + _EM + r"\s*", "like": f"%{_EM}%"}
                    )
            except Exception:  # noqa: BLE001, a missing table/column is harmless
                pass
    except Exception:  # noqa: BLE001, cleanup is best-effort, never fatal
        logger.exception("em-dash DB cleanup failed")

    logger.info("Cortex durable server ready, Postgres persistence, no license/Redis.")
    try:
        yield
    finally:
        pending = await runs.drain_active_runs()
        if pending:
            logger.warning("Shutdown timed out with %d active run(s)", pending)
        for p in (cp_pool, store_pool, data_pool):
            await p.close()


app = FastAPI(title="Cortex LangGraph Server", lifespan=lifespan)
configure_cors(app)
app.include_router(api_v1_router, prefix="/api/v1")
