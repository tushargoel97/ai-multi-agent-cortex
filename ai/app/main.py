"""Cortex local-LLM service.

- Admin API (/admin/*): catalog, HuggingFace search, download/load/delete.
- OpenAI-compatible inference API (/v1/chat/completions, /v1/models) so
  the LangGraph backend can route to this service via base_url=http://ai:8100/v1.
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.services import model_manager as mm

logging.basicConfig(level=logging.INFO if not settings.debug else logging.DEBUG)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.default_local_model:
        try:
            await mm.load_model(settings.default_local_model)
        except Exception as e:
            logger.warning("Could not preload default model: %s", e)
    yield


app = FastAPI(title="Cortex Local LLM", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────── health ────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "cortex-ai",
        "loaded_model": mm.loaded_model(),
        "downloaded": [m["name"] for m in mm.list_downloaded()],
    }


# ─────────────────────────────────── admin ────────────────────────────────
class DownloadRequest(BaseModel):
    name: str
    repo_id: str | None = None
    filename: str | None = None


@app.get("/admin/catalog")
async def admin_catalog():
    return {
        "models": mm.list_catalog(),
        "loaded": mm.loaded_model(),
    }


@app.get("/admin/search")
async def admin_search(q: str = Query(..., min_length=1), limit: int = 20):
    try:
        return {"results": await mm.search_huggingface(q, limit=limit)}
    except Exception:
        logger.exception("HF search failed")
        raise HTTPException(502, "HuggingFace search failed")


@app.get("/admin/progress")
async def admin_progress():
    return mm.download_progress()


@app.post("/admin/download")
async def admin_download(body: DownloadRequest):
    try:
        # Run download in a background task so we return immediately
        asyncio.create_task(
            mm.download_model(body.name, repo_id=body.repo_id, filename=body.filename)
        )
        return {"name": body.name, "status": "started"}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/admin/load")
async def admin_load(body: DownloadRequest):
    try:
        await mm.load_model(body.name)
        return {"status": "loaded", "model": body.name}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception:
        logger.exception("Load failed")
        raise HTTPException(500, "Failed to load model")


class ImportRequest(BaseModel):
    name: str
    filename: str
    description: str | None = None
    context_length: int = 4096


@app.post("/admin/import-local")
async def admin_import_local(body: ImportRequest):
    """Register a GGUF already present in the models dir (e.g. host-trained),
    then load it so it's immediately servable."""
    try:
        info = mm.import_local_model(
            body.name,
            body.filename,
            description=body.description,
            context_length=body.context_length,
        )
        await mm.load_model(body.name)
        return {"status": "imported", "model": info}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception:
        logger.exception("Import failed")
        raise HTTPException(500, "Failed to import model")


@app.delete("/admin/models/{name}")
async def admin_delete(name: str):
    if not await mm.delete_model(name):
        raise HTTPException(404, "Model not downloaded")
    return {"status": "deleted", "model": name}


# ─────────────────────────── OpenAI-compatible API ────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str | list


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False
    top_p: float | None = None


@app.get("/v1/models")
async def openai_models():
    items = []
    for m in mm.list_downloaded():
        items.append(
            {
                "id": m["name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "cortex-local",
            }
        )
    return {"object": "list", "data": items}


def _msg_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content)


@app.post("/v1/chat/completions")
async def openai_chat_completions(body: ChatRequest):
    requested = body.model
    loaded = mm.loaded_model()

    # Auto-load if a different downloaded model was requested
    if requested and requested != loaded:
        if not mm.is_downloaded(requested):
            raise HTTPException(
                404,
                f"Model '{requested}' not downloaded. Download it via /admin/download first.",
            )
        await mm.load_model(requested)

    llm = mm.get_llm()
    if llm is None:
        raise HTTPException(
            503,
            "No model loaded. Load one via POST /admin/load { name: '...' }.",
        )

    msgs = [{"role": m.role, "content": _msg_to_str(m.content)} for m in body.messages]
    kwargs = dict(
        messages=msgs,
        temperature=body.temperature,
        max_tokens=body.max_tokens or 1024,
    )
    if body.top_p is not None:
        kwargs["top_p"] = body.top_p

    model_name = requested or loaded or "local"

    if not body.stream:
        result = await asyncio.to_thread(lambda: llm.create_chat_completion(**kwargs))
        # Normalize to OpenAI shape (llama-cpp already mostly matches)
        result.setdefault("id", f"chatcmpl-{uuid.uuid4().hex}")
        result.setdefault("object", "chat.completion")
        result.setdefault("created", int(time.time()))
        result["model"] = model_name
        return result

    async def event_stream():
        kwargs_stream = {**kwargs, "stream": True}
        cid = f"chatcmpl-{uuid.uuid4().hex}"

        def _iter():
            return llm.create_chat_completion(**kwargs_stream)

        # llama-cpp streaming is synchronous; wrap in a thread
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def _producer():
            try:
                for chunk in _iter():
                    chunk.setdefault("id", cid)
                    chunk.setdefault("object", "chat.completion.chunk")
                    chunk.setdefault("created", int(time.time()))
                    chunk["model"] = model_name
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        loop.run_in_executor(None, _producer)
        while True:
            item = await queue.get()
            if item is sentinel:
                yield "data: [DONE]\n\n"
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
