"""Shared durable server runtime."""

from __future__ import annotations

import os
from typing import Any

ASSISTANT_ID = "cortex"


def db_uri() -> str:
    """Return a normalized psycopg3 connection URI."""
    uri = os.environ.get("POSTGRES_URI") or os.environ.get("DATABASE_URL", "")
    return (
        uri.replace("postgresql+psycopg2://", "postgresql://")
        .replace("postgresql+psycopg://", "postgresql://")
        .replace("postgres+psycopg2://", "postgresql://")
    )


class Runtime:
    """Process-wide handles set up in the FastAPI lifespan."""

    pool: Any = None
    checkpointer: Any = None
    store: Any = None
    graph: Any = None


runtime = Runtime()
