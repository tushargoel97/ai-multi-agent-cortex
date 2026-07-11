"""Local model catalog + HuggingFace discovery + download/load/delete."""

import asyncio
import json
import logging
import os
import re
import shutil
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_llm = None
_loaded_model_name: str | None = None
_download_progress: dict[str, dict] = {}
_model_lock = asyncio.Lock()
_download_locks: dict[str, asyncio.Lock] = {}

MODELS_DIR = settings.models_dir

# Curated catalog (CPU-friendly, broadly capable)
MODEL_CATALOG: dict[str, dict] = {
    "gemma-3-1b": {
        "repo_id": "ggml-org/gemma-3-1b-it-GGUF",
        "filename": "gemma-3-1b-it-Q8_0.gguf",
        "description": "Google Gemma 3 1B, fast, minimal RAM.",
        "size_mb": 1020,
        "context_length": 8192,
        "parameters": "1B",
        "tags": ["fast", "gemma"],
    },
    "llama-3.2-3b": {
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "description": "Meta Llama 3.2 3B, well-rounded small model.",
        "size_mb": 2048,
        "context_length": 4096,
        "parameters": "3B",
        "tags": ["general"],
    },
    "phi-4-mini": {
        "repo_id": "bartowski/phi-4-mini-instruct-GGUF",
        "filename": "phi-4-mini-instruct-Q4_K_M.gguf",
        "description": "Microsoft Phi-4 Mini, top reasoning at 3.8B.",
        "size_mb": 2400,
        "context_length": 8192,
        "parameters": "3.8B",
        "tags": ["reasoning", "math", "new"],
    },
    "llama-3.1-8b": {
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "description": "Meta Llama 3.1 8B, flagship 8B instruct.",
        "size_mb": 4920,
        "context_length": 8192,
        "parameters": "8B",
        "tags": ["general", "coding"],
    },
}


# Non-builtin entries (community downloads, fine-tuned imports) are persisted
# to /models/catalog.json so they survive container restarts.
_CUSTOM_CATALOG_PATH = os.path.join(MODELS_DIR, "catalog.json")


def _load_custom_catalog() -> None:
    if not os.path.exists(_CUSTOM_CATALOG_PATH):
        return
    try:
        with open(_CUSTOM_CATALOG_PATH) as f:
            MODEL_CATALOG.update(json.load(f))
    except Exception:
        logger.exception("Could not read %s, ignoring", _CUSTOM_CATALOG_PATH)


def _persist_custom_entry(name: str, info: dict) -> None:
    custom: dict = {}
    if os.path.exists(_CUSTOM_CATALOG_PATH):
        try:
            with open(_CUSTOM_CATALOG_PATH) as f:
                custom = json.load(f)
        except Exception:
            logger.exception("Could not read %s, rewriting", _CUSTOM_CATALOG_PATH)
    custom[name] = info
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(_CUSTOM_CATALOG_PATH, "w") as f:
        json.dump(custom, f, indent=2)


def _remove_custom_entry(name: str) -> None:
    if not os.path.exists(_CUSTOM_CATALOG_PATH):
        return
    try:
        with open(_CUSTOM_CATALOG_PATH) as f:
            custom = json.load(f)
        if name in custom:
            del custom[name]
            with open(_CUSTOM_CATALOG_PATH, "w") as f:
                json.dump(custom, f, indent=2)
    except Exception:
        logger.exception("Could not update %s", _CUSTOM_CATALOG_PATH)


_load_custom_catalog()


def _path_in_models_dir(filename: str) -> str:
    """Resolve a catalog filename without allowing it to escape MODELS_DIR."""
    root = os.path.realpath(MODELS_DIR)
    path = os.path.realpath(os.path.join(root, filename))
    if os.path.commonpath((root, path)) != root:
        raise ValueError(
            "Model filename must resolve inside the configured models directory."
        )
    return path


def import_local_model(
    name: str,
    filename: str,
    description: str | None = None,
    context_length: int = 4096,
) -> dict:
    """Register a GGUF already on disk. Call through import_and_load_model."""
    global _llm, _loaded_model_name
    path = _path_in_models_dir(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file {filename!r} in {MODELS_DIR}")
    # The file may be a retrained replacement of an already-loaded model,
    # drop the in-memory instance so the next load reads the new weights.
    if _loaded_model_name == name:
        _llm = None
        _loaded_model_name = None
    info = {
        "repo_id": "",
        "filename": filename,
        "description": description or f"Locally imported GGUF ({filename})",
        "size_mb": round(os.path.getsize(path) / (1024 * 1024), 1),
        "context_length": context_length,
        "parameters": "",
        "tags": ["fine-tuned", "local-file"],
    }
    MODEL_CATALOG[name] = info
    _persist_custom_entry(name, info)
    return {"name": name, **info}


def _model_path(name: str) -> str | None:
    info = MODEL_CATALOG.get(name)
    if not info:
        return None
    try:
        p = _path_in_models_dir(info["filename"])
    except ValueError:
        logger.error("Ignoring unsafe model path for catalog entry %s", name)
        return None
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
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def _is_primary_shard(f: str) -> bool:
    """True for single-file GGUFs and the first shard of a multi-part set."""
    m = _SHARD_RE.search(f)
    return m is None or m.group(1) == "00001"


def _pick_best_gguf(filenames: list[str]) -> str | None:
    # Modern repos keep quants in sub-folders and split large models into
    # multi-part shards, accept both (only the first shard is the entry point).
    gguf = [
        f for f in filenames
        if f.endswith(".gguf") and "mmproj" not in f.lower()
    ]
    if not gguf:
        return None
    pool = [f for f in gguf if _is_primary_shard(f)] or gguf
    pool.sort(key=lambda f: ("/" in f, f))  # prefer flat files, then stable order
    for q in _QUANT_PREFERENCE:
        for f in pool:
            if q in f:
                return f
    return pool[0]


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


def validate_download_request(
    name: str,
    *,
    repo_id: str | None = None,
    filename: str | None = None,
) -> None:
    """Validate before a background download task is accepted by the API."""
    if name not in MODEL_CATALOG and (not repo_id or not filename):
        raise ValueError(
            f"Unknown model {name!r}. Provide repo_id and filename for non-catalog models."
        )
    if filename:
        _path_in_models_dir(filename)


async def download_model(
    name: str,
    *,
    repo_id: str | None = None,
    filename: str | None = None,
) -> dict:
    validate_download_request(name, repo_id=repo_id, filename=filename)
    download_lock = _download_locks.setdefault(name, asyncio.Lock())
    async with download_lock:
        if name not in MODEL_CATALOG:
            MODEL_CATALOG[name] = {
                "repo_id": repo_id,
                "filename": filename,
                "description": f"Community model from {repo_id}",
                "size_mb": 0,
                "context_length": 4096,
                "parameters": "",
                "tags": ["community"],
            }
            _persist_custom_entry(name, MODEL_CATALOG[name])
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

            fname = info["filename"]
            shard = _SHARD_RE.search(fname)
            if shard:
                # Multi-part GGUF: fetch every shard into the same directory so
                # llama.cpp can load the set from the first shard.
                total_parts = int(shard.group(2))
                prefix = fname[: shard.start()]
                first_path = None
                for i in range(1, total_parts + 1):
                    part = f"{prefix}-{i:05d}-of-{total_parts:05d}.gguf"
                    path = hf_hub_download(
                        repo_id=info["repo_id"],
                        filename=part,
                        local_dir=MODELS_DIR,
                        tqdm_class=_Tqdm,
                    )
                    if i == 1:
                        first_path = path
                return first_path
            return hf_hub_download(
                repo_id=info["repo_id"],
                filename=fname,
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
    async with _model_lock:
        download_lock = _download_locks.setdefault(name, asyncio.Lock())
        async with download_lock:
            return _delete_model_unlocked(name)


def _delete_model_unlocked(name: str) -> bool:
    global _llm, _loaded_model_name
    if name == _loaded_model_name:
        _llm = None
        _loaded_model_name = None
    p = _model_path(name)
    if not p:
        return False
    os.remove(p)
    # Multi-part GGUFs sit next to their sibling shards, remove the whole set.
    shard = _SHARD_RE.search(p)
    if shard:
        import glob

        for f in glob.glob(f"{p[: shard.start()]}-*-of-*.gguf"):
            try:
                os.remove(f)
            except OSError:
                pass
    _remove_custom_entry(name)
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
    async with _model_lock:
        await _load_model_unlocked(name)


async def _load_model_unlocked(name: str) -> None:
    if not is_downloaded(name):
        await download_model(name)
    await asyncio.to_thread(_load_sync, name)


async def import_and_load_model(
    name: str,
    filename: str,
    description: str | None = None,
    context_length: int = 4096,
) -> dict:
    """Atomically replace a catalog entry and make it the active model."""
    async with _model_lock:
        download_lock = _download_locks.setdefault(name, asyncio.Lock())
        async with download_lock:
            info = import_local_model(
                name,
                filename,
                description=description,
                context_length=context_length,
            )
            await asyncio.to_thread(_load_sync, name)
            return info


@asynccontextmanager
async def model_session(
    requested_model: str | None = None,
) -> AsyncIterator[tuple[object | None, str | None]]:
    """Serialize llama.cpp inference with model swaps and destructive admin work."""
    async with _model_lock:
        if requested_model and requested_model != _loaded_model_name:
            if not is_downloaded(requested_model):
                raise FileNotFoundError(f"Model {requested_model!r} not downloaded.")
            await asyncio.to_thread(_load_sync, requested_model)
        yield _llm, requested_model or _loaded_model_name


def get_llm():
    """Return the currently loaded llama-cpp instance (or None)."""
    return _llm
