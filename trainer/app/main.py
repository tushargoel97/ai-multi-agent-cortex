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
    # Quick top-up: warm-start from the existing adapters (fewer iters) instead
    # of a full retrain. The base is taken from base_model.txt, not base_model.
    resume: bool = False


class DatasetRequest(BaseModel):
    # Kept for request compatibility; dataset generation is always the
    # deterministic facts.yaml + learned_facts.yaml expansion.
    include_builtin: bool = True


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
    # Intelligent scrape-agent crawl budget (applies to generic, non-AMD URLs).
    max_pages: int = 20
    max_depth: int = 2
    delay_s: float = 2.5


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
    """Expand facts.yaml + learned_facts.yaml into spec / overview /
    comparison / buying / refusal training examples.

    This is deterministic and instant. Sources (URLs/PDFs/images) become
    structured spec sheets in learned_facts.yaml via the scrape agent
    (POST /admin/scrape) BEFORE this step — not by inventing Q&A from raw
    text. Run 'Import specs from sources' first, then 'Generate dataset'.
    """
    req = req or DatasetRequest()
    # Pick up edits to generate_dataset.py without a service restart.
    importlib.reload(generate_dataset)
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
    out["sources_count"] = len(src.list_sources())
    # Whether a prior full train left adapters to warm-start a quick top-up.
    out["adapters_exist"] = (settings.adapters_dir / "adapters.safetensors").exists()
    return out


@app.get("/admin/dataset/preview")
def dataset_preview(split: str = "train", limit: int = 300) -> dict:
    """Return the generated training examples for a split so the admin can
    eyeball validity before training. ``split`` ∈ {train, valid}.
    """
    import json as _json

    fname = {
        "train": "train.jsonl",
        "valid": "valid.jsonl",
    }.get(split, "train.jsonl")
    path = settings.data_dir / fname
    if not path.exists():
        return {
            "split": split,
            "exists": False,
            "total": 0,
            "shown": 0,
            "pairs": [],
            "modified_at": None,
        }
    cap = max(1, min(limit, 1000))
    pairs: list[dict] = []
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            if len(pairs) >= cap:
                continue
            try:
                msgs = _json.loads(line).get("messages", [])
                q = next(
                    (m.get("content", "") for m in msgs if m.get("role") == "user"), ""
                )
                a = next(
                    (m.get("content", "") for m in msgs if m.get("role") == "assistant"),
                    "",
                )
                pairs.append({"q": q, "a": a})
            except Exception:  # noqa: BLE001 — flag the bad line, keep going
                pairs.append({"q": "", "a": f"[unparseable] {line[:200]}"})
    return {
        "split": split,
        "exists": True,
        "total": total,
        "shown": len(pairs),
        "pairs": pairs,
        "modified_at": path.stat().st_mtime,
    }


# ── Training-data sources (PDF / Excel / image / URL / prompt) ──────────────────────


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

    by_id = {s["id"]: s for s in list_sources()}
    resolved: list = []
    for item in req.sources:
        if item in by_id:  # uploaded source — pass its (type-aware) entry dict
            resolved.append(by_id[item])
        else:
            resolved.append(item)  # a raw URL string
    try:
        pipeline.start_scrape(
            resolved,
            req.max_products,
            max_pages=req.max_pages,
            max_depth=req.max_depth,
            delay_s=req.delay_s,
        )
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
            req.iters,
            req.batch_size,
            req.learning_rate,
            base_model=req.base_model,
            resume=req.resume,
        )
    except pipeline.JobConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "status": "started",
        "iters": req.iters,
        "resume": req.resume,
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


# Bytes per parameter for common safetensors dtypes (rough download-size guess).
_DTYPE_BYTES = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8": 1, "I8": 1, "U8": 1,
    "I4": 0.5, "U4": 0.5,
}


def _hf_model_size(st) -> tuple[int | None, int | None]:
    """(total_params, estimated_bytes) from a ModelInfo.safetensors block."""
    if st is None:
        return None, None
    total = getattr(st, "total", None)
    params = getattr(st, "parameters", None) or {}
    if params:
        approx = sum(
            int(c) * _DTYPE_BYTES.get(str(dt).upper(), 2) for dt, c in params.items()
        )
        return total, int(approx) or (int(total) * 2 if total else None)
    return total, (int(total) * 2 if total else None)


@app.get("/admin/hf/search")
def hf_search(q: str, limit: int = 20) -> dict:
    """Search Hugging Face for trainable base models (text-generation repos).

    mlx-lm downloads the chosen repo (safetensors) at train time; gated repos
    (e.g. google/gemma-*) need an HF_TOKEN in the trainer's environment.
    """
    q = (q or "").strip()
    if not q:
        return {"results": []}
    try:
        from huggingface_hub import HfApi
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"huggingface_hub unavailable: {e}")
    try:
        models = HfApi().list_models(
            search=q,
            pipeline_tag="text-generation",
            sort="downloads",
            limit=max(1, min(limit, 40)),
            expand=["safetensors", "downloads", "likes", "gated"],
        )
        results = []
        for m in models:
            params, size_bytes = _hf_model_size(getattr(m, "safetensors", None))
            results.append(
                {
                    "id": m.id,
                    "downloads": int(getattr(m, "downloads", 0) or 0),
                    "likes": int(getattr(m, "likes", 0) or 0),
                    "gated": bool(getattr(m, "gated", False)),
                    "params": params,
                    "size_bytes": size_bytes,
                }
            )
        return {"results": results}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"HuggingFace search failed: {e}")


@app.delete("/admin/artifacts")
def clear_artifacts() -> dict:
    """Delete the LoRA adapters + fused working dirs. They are base-specific
    (tied to base_model.txt) and can't be reused to train a different base —
    the next full train recreates them. The registered GGUF is removed
    separately via the ai service. Refuses while a job is running.
    """
    if pipeline._busy():
        raise HTTPException(status_code=409, detail="a training/convert job is running")
    import shutil

    removed: list[str] = []
    for d in (settings.adapters_dir, settings.fused_dir):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name)
    return {"status": "cleared", "removed": removed}
