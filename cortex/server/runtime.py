"""Shared server runtime — durable checkpointer/store + the compiled graph.

Populated once at application startup (see ``app.lifespan``) and read by the
route handlers. Kept in its own module to avoid circular imports between
``app`` and the routers.
"""

from __future__ import annotations

import os
from typing import Any

ASSISTANT_ID = "cortex"  # graph id the UI targets (NEXT_PUBLIC_ASSISTANT_ID)


def db_uri() -> str:
    """Plain psycopg3 connection URI for the checkpointer/store/threads table.

    Accepts either ``POSTGRES_URI`` or the app's SQLAlchemy-style ``DATABASE_URL``
    and normalizes the driver suffix psycopg3 does not understand.
    """
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
