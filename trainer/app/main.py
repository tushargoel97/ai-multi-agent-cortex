"""Cortex Trainer — host-side MLX LoRA fine-tuning service (Apple Silicon).

Runs OUTSIDE Docker (MLX needs the Apple GPU). The agent-chat-ui admin panel
reaches it through the Next.js proxy at /api/admin/trainer/* which forwards to
http://host.docker.internal:8200/admin/*.

Run:  cd trainer && uv run uvicorn app.main:app --host 0.0.0.0 --port 8200
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import pipeline
from . import sources as src
from .config import settings

# generate_dataset.py lives at trainer/ root, one level above this package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_dataset  # noqa: E402

app = FastAPI(title="Cortex Trainer", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    iters: int = 600
    batch_size: int = 4
    learning_rate: float = 1e-4
    # e.g. "Qwen/Qwen2.5-0.5B-Instruct" (default) or "google/gemma-4-e2b-it"
    base_model: str | None = None


class DatasetRequest(BaseModel):
    include_builtin: bool = True
    use_sources: bool = True
    max_pairs: int = 500


class UrlSourceRequest(BaseModel):
    url: str


class PromptSourceRequest(BaseModel):
    text: str
    name: str | None = None


class ConvertRequest(BaseModel):
    output_name: str = "finetuned-gemma3-1b-hardware"


class GapItem(BaseModel):
    id: str
    question: str


class GapResearchRequest(BaseModel):
    gaps: list[GapItem]


class ScrapeRequest(BaseModel):
    # URLs (any vendor page) and/or uploaded source ids. TechPowerUp is NOT a
    # default — it usually 403s bots; add it explicitly to try it.
    sources: list[str] = [
        "https://www.amd.com/en/products/specifications/processors.html",
    ]
    max_products: int = 30


@app.get("/health")
@app.get("/admin/health")
def health() -> dict:
    return {
        "status": "ok",
        "phase": pipeline.get_status().get("phase", "idle"),
        "base_model": settings.base_model,
        "converter_ready": settings.convert_script.exists(),
    }


@app.post("/admin/dataset/generate")
def dataset_generate(req: DatasetRequest | None = None) -> dict:
    req = req or DatasetRequest()
    # Pick up edits to generate_dataset.py without a service restart.
    importlib.reload(generate_dataset)
    if req.use_sources and src.list_sources():
        # Source-backed generation is a background job (LLM per chunk) —
        # progress arrives via GET /admin/progress.
        try:
            pipeline.start_dataset_generation(req.include_builtin, req.max_pairs)
        except pipeline.JobConflictError as e:
            raise HTTPException(status_code=409, detail=str(e))
        return {"status": "started"}
    # Built-in facts.yaml expansion is instant — keep the synchronous shape.
    return generate_dataset.generate()


@app.get("/admin/dataset/status")
def dataset_status() -> dict:
    out: dict = {}
    for split in ("train", "valid"):
        path = settings.data_dir / f"{split}.jsonl"
        out[split] = {
            "exists": path.exists(),
            "count": sum(1 for _ in open(path)) if path.exists() else 0,
            "modified_at": path.stat().st_mtime if path.exists() else None,
        }
    custom = settings.data_dir / "custom.jsonl"
    out["custom_pairs"] = sum(1 for _ in open(custom)) if custom.exists() else 0
    out["sources_count"] = len(src.list_sources())
    return out


# ── Training-data sources (PDF / Excel / URL / prompt) ─────────────────────


@app.get("/admin/sources")
def sources_list() -> dict:
    return {"sources": src.list_sources()}


@app.post("/admin/sources/upload")
async def sources_upload(file: UploadFile = File(...)) -> dict:
    try:
        return src.add_file_source(file.filename or "upload", await file.read())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/admin/sources/url")
def sources_add_url(req: UrlSourceRequest) -> dict:
    try:
        return src.add_url_source(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/admin/sources/prompt")
def sources_add_prompt(req: PromptSourceRequest) -> dict:
    try:
        return src.add_prompt_source(req.text, req.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/admin/sources/{source_id}")
def sources_delete(source_id: str) -> dict:
    if not src.delete_source(source_id):
        raise HTTPException(status_code=404, detail="source not found")
    return {"status": "deleted", "id": source_id}


# ── Knowledge-gap research (self-improvement loop) ──────────────────────────


@app.post("/admin/scrape")
def scrape_specs(req: ScrapeRequest) -> dict:
    """Dynamic spec import: URLs (any brand) and/or uploaded source ids."""
    from app.sources import list_sources
    from app.config import settings as cfg

    by_id = {s["id"]: s for s in list_sources()}
    resolved: list[str] = []
    for item in req.sources:
        if item in by_id:  # uploaded document — resolve to its stored path
            resolved.append(str(cfg.data_dir / "sources" / by_id[item]["path"]))
        else:
            resolved.append(item)
    try:
        pipeline.start_scrape(resolved, req.max_products)
    except pipeline.JobConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "started", "sources": resolved, "max_products": req.max_products}


@app.post("/admin/gaps/research")
def gaps_research(req: GapResearchRequest) -> dict:
    try:
        pipeline.start_gap_research([g.model_dump() for g in req.gaps])
    except pipeline.JobConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "started", "gaps": len(req.gaps)}


@app.get("/admin/gaps/learned")
def gaps_learned() -> dict:
    from . import research

    entries = research.load_learned()
    return {
        "count": len(entries),
        "products": [
            {"name": e.get("name"), "exists": e.get("exists", True)} for e in entries
        ],
    }


@app.post("/admin/train")
def train(req: TrainRequest) -> dict:
    if not (settings.data_dir / "train.jsonl").exists():
        raise HTTPException(status_code=400, detail="dataset missing — generate it first")
    try:
        pipeline.start_training(
            req.iters, req.batch_size, req.learning_rate, base_model=req.base_model
        )
    except pipeline.JobConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {
        "status": "started",
        "iters": req.iters,
        "base_model": req.base_model or settings.base_model,
    }


@app.post("/admin/train/stop")
def train_stop() -> dict:
    return {"stopped": pipeline.stop()}


@app.get("/admin/progress")
def progress() -> dict:
    return pipeline.get_status()


@app.post("/admin/convert")
def convert(req: ConvertRequest) -> dict:
    try:
        pipeline.start_convert(req.output_name)
    except pipeline.JobConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "started", "output_name": req.output_name}
