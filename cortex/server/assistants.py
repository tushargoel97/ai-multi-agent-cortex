"""Assistant endpoints — minimal single-graph implementation.

This app serves exactly one graph (``cortex``). The UI resolves it by graph id
(``NEXT_PUBLIC_ASSISTANT_ID=cortex``) and passes that as the ``assistant_id`` on
runs, so a light stub is all the SDK needs here.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from cortex.server.runtime import ASSISTANT_ID

router = APIRouter()

_EPOCH = "1970-01-01T00:00:00+00:00"


def _assistant() -> dict:
    return {
        "assistant_id": ASSISTANT_ID,
        "graph_id": ASSISTANT_ID,
        "name": ASSISTANT_ID,
        "config": {},
        "context": {},
        "metadata": {"created_by": "system"},
        "created_at": _EPOCH,
        "updated_at": _EPOCH,
        "version": 1,
    }


@router.post("/assistants/search")
async def search_assistants(request: Request):
    return [_assistant()]


@router.get("/assistants/{assistant_id}")
async def get_assistant(assistant_id: str):
    return _assistant()


@router.get("/assistants/{assistant_id}/graph")
async def get_assistant_graph(assistant_id: str):
    return {"nodes": [], "edges": []}


@router.get("/assistants/{assistant_id}/schemas")
async def get_assistant_schemas(assistant_id: str):
    return {
        "graph_id": ASSISTANT_ID,
        "input_schema": {},
        "output_schema": {},
        "state_schema": {},
        "config_schema": {},
    }
