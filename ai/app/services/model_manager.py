"""Local model catalog + HuggingFace discovery + download/load/delete."""

import asyncio
import logging
import os
import shutil

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_llm = None
_loaded_model_name: str | None = None
_download_progress: dict[str, dict] = {}

MODELS_DIR = settings.models_dir

# Curated catalog (CPU-friendly, broadly capable)
MODEL_CATALOG: dict[str, dict] = {
    "qwen2.5-1.5b": {
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "description": "Qwen 2.5 1.5B — fastest, minimal RAM.",
        "size_mb": 1024,
        "context_length": 4096,
        "parameters": "1.5B",
        "tags": ["fast", "json"],
    },
    "llama-3.2-3b": {
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "description": "Meta Llama 3.2 3B — well-rounded small model.",
        "size_mb": 2048,
        "context_length": 4096,
        "parameters": "3B",
        "tags": ["general"],
    },
    "qwen2.5-3b": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "description": "Qwen 2.5 3B — strong structured-output small model.",
        "size_mb": 2048,
        "context_length": 4096,
        "parameters": "3B",
        "tags": ["json", "coding"],
    },
    "phi-4-mini": {
        "repo_id": "bartowski/phi-4-mini-instruct-GGUF",
        "filename": "phi-4-mini-instruct-Q4_K_M.gguf",
        "description": "Microsoft Phi-4 Mini — top reasoning at 3.8B.",
        "size_mb": 2400,
        "context_length": 8192,
        "parameters": "3.8B",
        "tags": ["reasoning", "math", "new"],
    },
    "llama-3.1-8b": {
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "description": "Meta Llama 3.1 8B — flagship 8B instruct.",
        "size_mb": 4920,
        "context_length": 8192,
        "parameters": "8B",
        "tags": ["general", "coding"],
    },
}


def _model_path(name: str) -> str | None:
    info = MODEL_CATALOG.get(name)
    if not info:
        return None
    p = os.path.join(MODELS_DIR, info["filename"])
    return p if os.path.exists(p) else None


def is_downloaded(name: str) -> bool:
    return _model_path(name) is not None


def loaded_model() -> str | None:
    return _loaded_model_name


def list_catalog() -> list[dict]:
    return [
        {
            "name": name,
            **info,
            "downloaded": is_downloaded(name),
            "active": name == _loaded_model_name,
        }
        for name, info in MODEL_CATALOG.items()
    ]


def list_downloaded() -> list[dict]:
    return [m for m in list_catalog() if m["downloaded"]]


# ── HuggingFace search (LM-Studio-style discovery of latest GGUF models) ──

_HF_API = "https://huggingface.co/api"
_QUANT_PREFERENCE = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q3_K_M", "Q8_0", "Q6_K", "Q5_K_S"]


def _pick_best_gguf(filenames: list[str]) -> str | None:
    gguf = [
        f for f in filenames
        if f.endswith(".gguf") and "/" not in f and "mmproj" not in f.lower()
    ]
    if not gguf:
        return None
    for q in _QUANT_PREFERENCE:
        for f in gguf:
            if q in f:
                return f
    return gguf[0]


async def search_huggingface(query: str, limit: int = 20) -> list[dict]:
    out: list[dict] = []
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{_HF_API}/models",
            params={
                "search": query,
                "filter": "gguf",
                "sort": "downloads",
                "direction": "-1",
                "limit": str(limit),
            },
        )
        resp.raise_for_status()
        repos = resp.json()
        details = await asyncio.gather(
            *[client.get(f"{_HF_API}/models/{r['id']}") for r in repos[:limit]],
            return_exceptions=True,
        )
        for repo, d in zip(repos[:limit], details):
            if isinstance(d, Exception) or d.status_code != 200:
                continue
            data = d.json()
            files = [s["rfilename"] for s in data.get("siblings", [])]
            best = _pick_best_gguf(files)
            if not best:
                continue
            already = any(info["repo_id"] == repo["id"] for info in MODEL_CATALOG.values())
            out.append({
                "repo_id": repo["id"],
                "filename": best,
                "downloads": repo.get("downloads", 0),
                "likes": repo.get("likes", 0),
                "tags": [t for t in repo.get("tags", []) if ":" not in t and t != "gguf"],
                "in_catalog": already,
            })
    return out


def download_progress() -> dict[str, dict]:
    return dict(_download_progress)


async def download_model(
    name: str,
    *,
    repo_id: str | None = None,
    filename: str | None = None,
) -> dict:
    if name not in MODEL_CATALOG:
        if not repo_id or not filename:
            raise ValueError(
                f"Unknown model {name!r}. Provide repo_id and filename for non-catalog models."
            )
        MODEL_CATALOG[name] = {
            "repo_id": repo_id,
            "filename": filename,
            "description": f"Community model from {repo_id}",
            "size_mb": 0,
            "context_length": 4096,
            "parameters": "",
            "tags": ["community"],
        }
    info = MODEL_CATALOG[name]

    if is_downloaded(name):
        return {"name": name, "status": "already_downloaded"}

    os.makedirs(MODELS_DIR, exist_ok=True)
    _download_progress[name] = {
        "progress": 0,
        "downloaded_mb": 0,
        "total_mb": info["size_mb"],
        "status": "starting",
    }

    def _update(current: int, total: int) -> None:
        total_mb = round(total / (1024 * 1024), 1) if total else info["size_mb"]
        _download_progress[name] = {
            "progress": round(current / total * 100, 1) if total else 0,
            "downloaded_mb": round(current / (1024 * 1024), 1),
            "total_mb": total_mb,
            "status": "downloading",
        }

    def _do_download():
        from huggingface_hub import hf_hub_download
        from tqdm import tqdm

        class _Tqdm(tqdm):
            def update(self, n=1):
                super().update(n)
                if self.total:
                    _update(self.n, self.total)

        return hf_hub_download(
            repo_id=info["repo_id"],
            filename=info["filename"],
            local_dir=MODELS_DIR,
            tqdm_class=_Tqdm,
        )

    try:
        path = await asyncio.to_thread(_do_download)
        _download_progress[name] = {
            **_download_progress.get(name, {}),
            "progress": 100,
            "status": "complete",
        }
        return {"name": name, "status": "downloaded", "path": path}
    except Exception:
        _download_progress[name]["status"] = "error"
        raise
    finally:
        async def _cleanup():
            await asyncio.sleep(10)
            _download_progress.pop(name, None)
        asyncio.create_task(_cleanup())


async def delete_model(name: str) -> bool:
    global _llm, _loaded_model_name
    if name == _loaded_model_name:
        _llm = None
        _loaded_model_name = None
    p = _model_path(name)
    if not p:
        return False
    os.remove(p)
    cache = os.path.join(MODELS_DIR, ".cache")
    if os.path.exists(cache):
        shutil.rmtree(cache, ignore_errors=True)
    return True


def _load_sync(name: str):
    global _llm, _loaded_model_name
    if _loaded_model_name == name and _llm is not None:
        return _llm
    p = _model_path(name)
    if not p:
        raise FileNotFoundError(f"Model {name!r} not downloaded.")
    if _llm is not None:
        del _llm
        _llm = None
    from llama_cpp import Llama

    ctx = MODEL_CATALOG.get(name, {}).get("context_length", settings.n_ctx)
    logger.info("Loading %s (ctx=%d)…", name, ctx)
    _llm = Llama(
        model_path=p,
        n_ctx=ctx,
        n_threads=settings.local_llm_threads,
        verbose=False,
    )
    _loaded_model_name = name
    return _llm


async def load_model(name: str) -> None:
    if not is_downloaded(name):
        await download_model(name)
    await asyncio.to_thread(_load_sync, name)


def get_llm():
    """Return the currently loaded llama-cpp instance (or None)."""
    return _llm
