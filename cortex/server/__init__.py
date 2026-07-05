"""Custom durable LangGraph server (Path 2B).

A minimal, self-hosted reimplementation of the subset of the LangGraph Platform
REST + SSE API that the agent-chat-ui (``@langchain/langgraph-sdk`` ``useStream``)
depends on, backed by ``AsyncPostgresSaver`` + ``AsyncPostgresStore`` for fully
native, durable persistence in Postgres — no LangSmith license and no Redis.

Entry point: ``cortex.server.app:app`` (served by uvicorn).
"""
