"""Cortex local-LLM service.

- Admin API (/admin/*): catalog, HuggingFace search, download/load/delete.
- OpenAI-compatible inference API (/v1/chat/completions, /v1/models) so
  the LangGraph backend can route to this service via base_url=http://ai:8100/v1.
"""

import asyncio
import json
import logging
import threading
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
_background_tasks: set[asyncio.Task] = set()


def _start_background_task(coro, *, label: str) -> None:
    """Keep a strong task reference and surface background failures in logs."""
    task = asyncio.create_task(coro, name=label)
    _background_tasks.add(task)

    def _finished(done: asyncio.Task) -> None:
        _background_tasks.discard(done)
        if done.cancelled():
            return
        error = done.exception()
        if error is not None:
            logger.error("Background task %s failed: %s", label, error)

    task.add_done_callback(_finished)


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
        mm.validate_download_request(
            body.name,
            repo_id=body.repo_id,
            filename=body.filename,
        )
        _start_background_task(
            mm.download_model(
                body.name,
                repo_id=body.repo_id,
                filename=body.filename,
            ),
            label=f"download-model:{body.name}",
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
        info = await mm.import_and_load_model(
            body.name,
            body.filename,
            description=body.description,
            context_length=body.context_length,
        )
        return {"status": "imported", "model": info}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
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
    if requested and not mm.is_downloaded(requested):
        raise HTTPException(
            404,
            f"Model '{requested}' not downloaded. Download it via /admin/download first.",
        )
    if requested is None and mm.get_llm() is None:
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

    if not body.stream:
        try:
            async with mm.model_session(requested) as (llm, model_name):
                if llm is None:
                    raise HTTPException(503, "No model is currently loaded.")
                result = await asyncio.to_thread(
                    lambda: llm.create_chat_completion(**kwargs)
                )
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        # Normalize to OpenAI shape (llama-cpp already mostly matches)
        result.setdefault("id", f"chatcmpl-{uuid.uuid4().hex}")
        result.setdefault("object", "chat.completion")
        result.setdefault("created", int(time.time()))
        result["model"] = model_name or "local"
        return result

    async def event_stream():
        try:
            async with mm.model_session(requested) as (llm, model_name):
                if llm is None:
                    error = {"error": {"message": "No model is currently loaded."}}
                    yield f"data: {json.dumps(error)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                kwargs_stream = {**kwargs, "stream": True}
                cid = f"chatcmpl-{uuid.uuid4().hex}"
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue = asyncio.Queue()
                sentinel = object()
                stop = threading.Event()

                def _producer():
                    iterator = None
                    try:
                        iterator = llm.create_chat_completion(**kwargs_stream)
                        for chunk in iterator:
                            if stop.is_set():
                                break
                            chunk.setdefault("id", cid)
                            chunk.setdefault("object", "chat.completion.chunk")
                            chunk.setdefault("created", int(time.time()))
                            chunk["model"] = model_name or "local"
                            loop.call_soon_threadsafe(queue.put_nowait, chunk)
                    except Exception as error:
                        loop.call_soon_threadsafe(queue.put_nowait, error)
                    finally:
                        if iterator is not None and hasattr(iterator, "close"):
                            iterator.close()
                        loop.call_soon_threadsafe(queue.put_nowait, sentinel)

                producer = asyncio.create_task(
                    asyncio.to_thread(_producer),
                    name=f"local-inference:{model_name or 'local'}",
                )
                try:
                    while True:
                        item = await queue.get()
                        if item is sentinel:
                            yield "data: [DONE]\n\n"
                            break
                        if isinstance(item, Exception):
                            logger.error("Streaming inference failed: %s", item)
                            error = {"error": {"message": "Local inference failed."}}
                            yield f"data: {json.dumps(error)}\n\n"
                            continue
                        yield f"data: {json.dumps(item)}\n\n"
                finally:
                    stop.set()
                    await asyncio.shield(producer)
        except FileNotFoundError as error:
            payload = {"error": {"message": str(error)}}
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
