import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.api.dependencies.tasks import start
from app.services import model_manager as mm

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin")


class DownloadRequest(BaseModel):
    name: str
    repo_id: str | None = None
    filename: str | None = None


class CatalogPatch(BaseModel):
    description: str | None = None
    context_length: int | None = None


class ImportRequest(BaseModel):
    name: str
    filename: str
    description: str | None = None
    context_length: int = 4096


@router.get("/catalog")
async def catalog():
    return {
        "models": mm.list_catalog(),
        "loaded": mm.loaded_model(),
        "memory": mm.host_memory(),
        "idle": mm.idle_state(),
    }


@router.get("/local-files")
async def local_files():
    return {"files": mm.list_untracked_gguf_files()}


@router.patch("/models/{name}")
async def update_model(name: str, body: CatalogPatch):
    try:
        model = mm.update_catalog_entry(
            name,
            description=body.description,
            context_length=body.context_length,
        )
        return {"status": "updated", "model": model}
    except KeyError:
        raise HTTPException(404, f"Unknown model {name!r}")


@router.get("/search")
async def search(q: str = Query(..., min_length=1), limit: int = 20):
    try:
        return {"results": await mm.search_huggingface(q, limit=limit)}
    except Exception as error:
        logger.exception("HuggingFace search failed")
        raise HTTPException(502, "HuggingFace search failed") from error


@router.get("/progress")
async def progress():
    return mm.download_progress()


@router.get("/hf-details")
async def hf_details(repo_id: str = Query(..., min_length=3)):
    try:
        return await mm.hf_model_details(repo_id)
    except Exception as error:
        logger.exception("HuggingFace details failed for %s", repo_id)
        raise HTTPException(502, "Could not load HuggingFace details") from error


@router.get("/models/{name}")
async def model_detail(name: str):
    entry = mm.catalog_entry(name)
    if entry is None:
        raise HTTPException(404, f"Unknown model {name!r}")
    return entry


@router.post("/download/pause")
async def pause_download(body: DownloadRequest):
    if not mm.request_download_control(body.name, "pause"):
        raise HTTPException(409, "No active download to pause")
    return {"name": body.name, "status": "pausing"}


@router.post("/download/cancel")
async def cancel_download(body: DownloadRequest):
    if not mm.request_download_control(body.name, "cancel"):
        raise HTTPException(409, "No active download to cancel")
    return {"name": body.name, "status": "cancelling"}


@router.post("/download")
async def download(body: DownloadRequest):
    try:
        mm.validate_download_request(
            body.name,
            repo_id=body.repo_id,
            filename=body.filename,
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    start(
        mm.download_model(
            body.name,
            repo_id=body.repo_id,
            filename=body.filename,
        ),
        label=f"download-model:{body.name}",
    )
    return {"name": body.name, "status": "started"}


@router.post("/load")
async def load(body: DownloadRequest):
    try:
        await mm.load_model(body.name)
        return {"status": "loaded", "model": body.name}
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    except Exception as error:
        logger.exception("Model load failed")
        raise HTTPException(500, "Failed to load model") from error


@router.post("/import-local")
async def import_local(body: ImportRequest):
    try:
        model = await mm.import_and_load_model(
            body.name,
            body.filename,
            description=body.description,
            context_length=body.context_length,
        )
        return {"status": "imported", "model": model}
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        logger.exception("Model import failed")
        raise HTTPException(500, "Failed to import model") from error


@router.delete("/models/{name}")
async def delete(name: str):
    if not await mm.delete_model(name):
        raise HTTPException(404, "Model not downloaded")
    return {"status": "deleted", "model": name}
