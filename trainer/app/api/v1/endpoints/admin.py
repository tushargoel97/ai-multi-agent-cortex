"""Host-side Cortex trainer API."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app import domains as dom
from app import pipeline
from app import sources as src
from app.backends import adapter_backend_id, capabilities, get_backend
from app.config import settings
from app.runs import estimate_seconds, get as get_run, list_all

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import generate_dataset  # noqa: E402

router = APIRouter()


class TrainRequest(BaseModel):
    iters: int = Field(default=600, ge=10, le=100_000)
    batch_size: int = Field(default=4, ge=1, le=64)
    learning_rate: float = Field(default=1e-4, gt=0, le=0.1)
    backend_id: str = Field(default=settings.default_backend, min_length=1, max_length=80)
    base_model: str | None = None
    resume: bool = False
    early_stopping_patience: int = Field(default=5, ge=0, le=100)
    early_stopping_min_delta: float = Field(default=0.001, ge=0, le=1)


class EstimateRequest(BaseModel):
    backend_id: str = Field(default=settings.default_backend, min_length=1, max_length=80)
    iters: int = Field(default=600, ge=10, le=100_000)
    batch_size: int = Field(default=4, ge=1, le=64)
    base_model: str | None = None


class DatasetRequest(BaseModel):
    subdomains: list[str] | None = None
    domains: list[str] | None = None
    include_builtin: bool = True


class UrlSourceRequest(BaseModel):
    url: str


class PromptSourceRequest(BaseModel):
    text: str
    name: str | None = None


class ConvertRequest(BaseModel):
    output_name: str = Field(
        default="finetuned-gemma3-1b-hardware",
        pattern=r"^finetuned-[a-z0-9][a-z0-9._-]{0,180}$",
    )


class EvaluateRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)
    cases: int = Field(default=12, ge=8, le=40)


class GapItem(BaseModel):
    id: str
    question: str


class GapResearchRequest(BaseModel):
    gaps: list[GapItem]


class ScrapeRequest(BaseModel):
    sources: list[str] = [
        "https://www.amd.com/en/products/specifications/processors.html",
    ]
    max_products: int = 30
    max_pages: int = 20
    max_depth: int = 2
    delay_s: float = 2.5


@router.get("/health")
@router.get("/admin/health")
def health() -> dict:
    return {
        "status": "ok",
        "phase": pipeline.get_status().get("phase", "idle"),
        "base_model": settings.base_model,
        "converter_ready": settings.convert_script.exists(),
    }


@router.get("/admin/capabilities")
def trainer_capabilities() -> dict:
    return capabilities(
        data_dir=settings.data_dir,
        artifacts_dir=settings.artifacts_dir,
        host_id=settings.host_id,
        host_label=settings.host_label,
        default_backend=settings.default_backend,
    )


@router.post("/admin/estimate")
def training_estimate(req: EstimateRequest) -> dict:
    try:
        backend = get_backend(req.backend_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    available, reason = backend.available()
    source_model = req.base_model or settings.base_model
    training_model = backend.training_model(source_model, settings.artifacts_dir)
    needs_prepare = (
        backend.description.algorithm == "qlora_4bit"
        and not (Path(training_model) / "config.json").exists()
    )
    calibrated, samples = estimate_seconds(
        settings.runs_dir,
        backend.description.id,
        source_model,
        req.iters,
        req.batch_size,
    )
    return {
        "backend_id": backend.description.id,
        "available": available,
        "reason": reason,
        "estimated_seconds": calibrated
        or backend.estimate_seconds(
            iters=req.iters, batch_size=req.batch_size, needs_prepare=needs_prepare
        ),
        "estimate_source": "history" if calibrated else "baseline",
        "estimate_samples": samples,
        "needs_prepare": needs_prepare,
    }


@router.post("/admin/dataset/generate")
def dataset_generate(req: DatasetRequest | None = None) -> dict:
    """Generate training examples from selected domain facts."""
    req = req or DatasetRequest()
    importlib.reload(generate_dataset)
    subdomains = req.subdomains
    domains = req.domains
    if subdomains is None and domains is None:
        domains = ["hardware"] if req.include_builtin else []
    return generate_dataset.generate(subdomains=subdomains, domains=domains)


@router.get("/admin/domains")
def dataset_domains() -> dict:
    """Return the selectable domain hierarchy."""
    importlib.reload(generate_dataset)
    return {"domains": generate_dataset.available_domains()}


class DomainRequest(BaseModel):
    name: str
    description: str = ""


class FieldModel(BaseModel):
    key: str = ""
    label: str = ""
    questions: list[str] | None = None
    answer: str | None = None


class SubdomainRequest(BaseModel):
    name: str
    description: str = ""
    render: str = "prose"
    fields: list[FieldModel] = []
    overview: list[str] | None = None


class EntitiesRequest(BaseModel):
    entities: list[dict] = []


class SchemaProposeRequest(BaseModel):
    description: str = ""
    sample_text: str = ""


class TemplateProposeRequest(BaseModel):
    fields: list[FieldModel] = []


@router.post("/admin/domains/propose-schema")
def domain_propose_schema(req: SchemaProposeRequest) -> dict:
    """Smart schema proposal (fields + render mode) for the user to review."""
    try:
        return dom.propose_schema(req.description, req.sample_text)
    except Exception as e:  # noqa: BLE001, LLM/endpoint best-effort
        raise HTTPException(502, f"Schema proposal failed: {e}")


@router.post("/admin/domains/propose-templates")
def domain_propose_templates(req: TemplateProposeRequest) -> dict:
    """Smart question/answer template proposal for the user to review."""
    try:
        return dom.propose_templates([f.model_dump() for f in req.fields])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Template proposal failed: {e}")


@router.post("/admin/domains")
def domain_create(req: DomainRequest) -> dict:
    try:
        return dom.create_domain(req.name, req.description)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/admin/domains/{domain}")
def domain_delete(domain: str) -> dict:
    try:
        dom.delete_domain(domain)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@router.post("/admin/domains/{domain}/subdomains")
def subdomain_save(domain: str, req: SubdomainRequest) -> dict:
    try:
        return dom.save_subdomain(
            domain,
            req.name,
            description=req.description,
            render=req.render,
            fields=[f.model_dump() for f in req.fields],
            overview=req.overview,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/admin/domains/{domain}/subdomains/{sub}")
def subdomain_get(domain: str, sub: str) -> dict:
    try:
        return dom.get_subdomain(domain, sub)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.delete("/admin/domains/{domain}/subdomains/{sub}")
def subdomain_delete(domain: str, sub: str) -> dict:
    dom.delete_subdomain(domain, sub)
    return {"ok": True}


@router.post("/admin/domains/{domain}/subdomains/{sub}/entities")
def subdomain_entities(domain: str, sub: str, req: EntitiesRequest) -> dict:
    try:
        rows = dom.set_entities(domain, sub, req.entities)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"entities": rows, "count": len(rows)}


@router.get("/admin/dataset/status")
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
    out["adapters_exist"] = (settings.adapters_dir / "adapters.safetensors").exists()
    out["adapters_backend_id"] = adapter_backend_id(settings.adapters_dir)
    return out


@router.get("/admin/dataset/preview")
def dataset_preview(split: str = "train", limit: int = 300) -> dict:
    """Return a bounded dataset preview."""
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
            except Exception:  # noqa: BLE001, flag the bad line, keep going
                pairs.append({"q": "", "a": f"[unparseable] {line[:200]}"})
    return {
        "split": split,
        "exists": True,
        "total": total,
        "shown": len(pairs),
        "pairs": pairs,
        "modified_at": path.stat().st_mtime,
    }


@router.get("/admin/sources")
def sources_list() -> dict:
    return {"sources": src.list_sources()}


@router.post("/admin/sources/upload")
async def sources_upload(file: UploadFile = File(...)) -> dict:
    try:
        return src.add_file_source(file.filename or "upload", await file.read())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/admin/sources/url")
def sources_add_url(req: UrlSourceRequest) -> dict:
    try:
        return src.add_url_source(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/admin/sources/prompt")
def sources_add_prompt(req: PromptSourceRequest) -> dict:
    try:
        return src.add_prompt_source(req.text, req.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/admin/sources/{source_id}")
def sources_delete(source_id: str) -> dict:
    if not src.delete_source(source_id):
        raise HTTPException(status_code=404, detail="source not found")
    return {"status": "deleted", "id": source_id}


@router.post("/admin/scrape")
def scrape_specs(req: ScrapeRequest) -> dict:
    """Dynamic spec import: URLs (any brand) and/or uploaded source ids."""
    from app.sources import list_sources

    by_id = {s["id"]: s for s in list_sources()}
    resolved: list = []
    for item in req.sources:
        if item in by_id:  # uploaded source, pass its (type-aware) entry dict
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


class SmartImportRequest(BaseModel):
    sources: list[str] = []
    target: str = "auto"  # "auto" or "domain/subdomain"


@router.post("/admin/import/propose")
def import_propose(req: SmartImportRequest) -> dict:
    """Domain-aware import: read the selected sources and propose which
    domain/subdomain + schema + entities to add (reviewed before writing)."""
    from app.sources import list_sources

    by_id = {s["id"]: s for s in list_sources()}
    resolved: list = [by_id.get(item, item) for item in req.sources]
    try:
        pipeline.start_smart_import(resolved, req.target or "auto")
    except pipeline.JobConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "started"}


class ImportApplyRequest(BaseModel):
    domain: str
    subdomain: str
    render: str = "prose"
    fields: list[dict] = []
    entities: list[dict] = []


@router.post("/admin/import/apply")
def import_apply(req: ImportApplyRequest) -> dict:
    """Persist an approved import proposal (creates the subdomain if new)."""
    from app import research

    try:
        return research.apply_import(req.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/admin/gaps/research")
def gaps_research(req: GapResearchRequest) -> dict:
    try:
        pipeline.start_gap_research([g.model_dump() for g in req.gaps])
    except pipeline.JobConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "started", "gaps": len(req.gaps)}


@router.get("/admin/gaps/learned")
def gaps_learned() -> dict:
    from app import research

    entries = research.load_learned()
    return {
        "count": len(entries),
        "products": [
            {"name": e.get("name"), "exists": e.get("exists", True)} for e in entries
        ],
    }


@router.post("/admin/train")
def train(req: TrainRequest) -> dict:
    if not (settings.data_dir / "train.jsonl").exists():
        raise HTTPException(status_code=400, detail="dataset missing, generate it first")
    try:
        run_id = pipeline.start_training(
            req.iters,
            req.batch_size,
            req.learning_rate,
            base_model=req.base_model,
            resume=req.resume,
            backend_id=req.backend_id,
            early_stopping_patience=req.early_stopping_patience,
            early_stopping_min_delta=req.early_stopping_min_delta,
        )
    except pipeline.JobConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "status": "started",
        "iters": req.iters,
        "resume": req.resume,
        "base_model": req.base_model or settings.base_model,
        "backend_id": req.backend_id,
        "run_id": run_id,
    }


@router.post("/admin/train/stop")
def train_stop() -> dict:
    return {"stopped": pipeline.stop()}


@router.get("/admin/progress")
def progress() -> dict:
    return pipeline.get_status()


@router.get("/admin/runs")
def training_runs() -> dict:
    return {"runs": list_all(settings.runs_dir)}


@router.get("/admin/runs/{run_id}")
def training_run(run_id: str) -> dict:
    if not (record := get_run(settings.runs_dir, run_id)):
        raise HTTPException(status_code=404, detail="run not found")
    return record


@router.post("/admin/runs/{run_id}/evaluate")
def evaluate_run(run_id: str, req: EvaluateRequest) -> dict:
    try:
        pipeline.start_evaluation(run_id, req.model_id, req.cases)
    except pipeline.JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "started", "run_id": run_id, "model_id": req.model_id}


@router.post("/admin/convert")
def convert(req: ConvertRequest) -> dict:
    try:
        pipeline.start_convert(req.output_name)
    except pipeline.JobConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "started", "output_name": req.output_name}


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


@router.get("/admin/hf/search")
def hf_search(q: str, limit: int = 20) -> dict:
    """Search Hugging Face for trainable text-generation models."""
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


@router.delete("/admin/artifacts")
def clear_artifacts() -> dict:
    """Delete inactive adapter and fused-model artifacts."""
    if pipeline._busy():
        raise HTTPException(status_code=409, detail="a training/convert job is running")
    import shutil

    removed: list[str] = []
    for d in (settings.adapters_dir, settings.fused_dir):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name)
    return {"status": "cleared", "removed": removed}
