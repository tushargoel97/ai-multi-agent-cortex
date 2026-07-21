"""Durable chat history: mirror threads to Postgres, restore after wipes.

`langgraph dev` keeps threads in in-memory pickles that don't survive
upgrades or graph-shape changes. This sidecar loops over the public
LangGraph API and upserts every thread's state (messages, summary, title
metadata) into the `thread_backups` table as plain JSON, format-proof by
construction. When the server comes up empty (fresh volume, incompatible
pickle) and backups exist, they are restored through the same API.

Runs as the `thread-backup` compose service using the langgraph image.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

import httpx
import psycopg2
import psycopg2.extras

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("thread-backup")

LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://langgraph:2024/api/v1")
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://cortex:cortex@db:5432/cortex"
).replace("postgresql+psycopg2://", "postgresql://")
INTERVAL = int(os.getenv("BACKUP_INTERVAL_SECONDS", "30"))

DDL = """
CREATE TABLE IF NOT EXISTS thread_backups (
  thread_id uuid PRIMARY KEY,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  state_values jsonb,
  updated_at timestamptz NOT NULL DEFAULT now()
)
"""


def _connect():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    return conn


async def _fetch_threads(client: httpx.AsyncClient) -> list[dict]:
    resp = await client.post(
        f"{LANGGRAPH_URL}/threads/search", json={"limit": 1000}
    )
    resp.raise_for_status()
    return resp.json()


def _backup(conn, threads: list[dict]) -> int:
    live_ids = []
    saved = 0
    with conn.cursor() as cur:
        for t in threads:
            live_ids.append(t["thread_id"])
            values = t.get("values") or {}
            if not values.get("messages"):
                continue  # nothing worth restoring
            cur.execute(
                """
                INSERT INTO thread_backups (thread_id, metadata, state_values, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (thread_id) DO UPDATE
                  SET metadata = EXCLUDED.metadata,
                      state_values = EXCLUDED.state_values,
                      updated_at = now()
                """,
                (
                    t["thread_id"],
                    json.dumps(t.get("metadata") or {}),
                    json.dumps(values),
                ),
            )
            saved += 1
        # Reconcile deletions, but only while the server actually has
        # threads. An empty server means a wipe, and the backups are then
        # exactly what we must NOT delete.
        if live_ids:
            cur.execute(
                "DELETE FROM thread_backups WHERE NOT (thread_id = ANY(%s::uuid[]))",
                (live_ids,),
            )
    return saved


async def _restore(client: httpx.AsyncClient, conn) -> int:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT thread_id, metadata, state_values FROM thread_backups"
        )
        rows = cur.fetchall()
    restored = 0
    for row in rows:
        try:
            resp = await client.post(
                f"{LANGGRAPH_URL}/threads",
                json={
                    "thread_id": str(row["thread_id"]),
                    "metadata": row["metadata"] or {},
                    "if_exists": "do_nothing",
                },
            )
            resp.raise_for_status()
            resp = await client.post(
                f"{LANGGRAPH_URL}/threads/{row['thread_id']}/state",
                json={"values": row["state_values"]},
            )
            resp.raise_for_status()
            restored += 1
        except Exception:  # noqa: BLE001, keep restoring the rest
            logger.exception("Restore failed for thread %s", row["thread_id"])
    return restored


async def main() -> None:
    conn = _connect()
    with conn.cursor() as cur:
        cur.execute(DDL)
    logger.info(
        "Thread backup running: %s -> %s every %ss",
        LANGGRAPH_URL,
        DATABASE_URL.split("@")[-1],
        INTERVAL,
    )
    # Distinguishes a wiped server (fresh boot, restore backups) from the
    # user deleting every thread mid-session (clear backups): once this
    # process has seen the server non-empty, empty means deletion.
    seen_nonempty = False
    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            try:
                threads = await _fetch_threads(client)
                if not threads:
                    with conn.cursor() as cur:
                        cur.execute("SELECT count(*) FROM thread_backups")
                        (backup_count,) = cur.fetchone()
                    if backup_count and not seen_nonempty:
                        n = await _restore(client, conn)
                        logger.info(
                            "Server was empty, restored %s/%s thread(s) from backup",
                            n,
                            backup_count,
                        )
                    elif backup_count:
                        with conn.cursor() as cur:
                            cur.execute("DELETE FROM thread_backups")
                        logger.info(
                            "All threads deleted by user, cleared %s backup(s)",
                            backup_count,
                        )
                else:
                    seen_nonempty = True
                    _backup(conn, threads)
            except (httpx.HTTPError, psycopg2.Error) as e:
                logger.warning("Backup tick failed (%s), retrying", e)
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
                conn = _connect()
            except Exception:  # noqa: BLE001
                logger.exception("Backup tick failed unexpectedly")
            await asyncio.sleep(INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
