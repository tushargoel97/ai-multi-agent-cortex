"""Local model catalog + HuggingFace discovery + download/load/delete."""

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator
from urllib.parse import quote

import httpx

from app.api.dependencies.tasks import start
from app.config import settings
from app.services.download_manager import DownloadAborted, download_file, remove_partial

logger = logging.getLogger(__name__)

_llm = None
_loaded_model_name: str | None = None
_last_used_ts: float = 0.0
_download_progress: dict[str, dict] = {}
_download_control: dict[str, str] = {}
_download_generations: dict[str, object] = {}
_model_lock = asyncio.Lock()
_download_locks: dict[str, asyncio.Lock] = {}
_catalog_lock = threading.Lock()

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


_CUSTOM_CATALOG_PATH = os.path.join(MODELS_DIR, "catalog.json")


def _load_custom_catalog() -> None:
    if not os.path.exists(_CUSTOM_CATALOG_PATH):
        return
    try:
        with open(_CUSTOM_CATALOG_PATH) as f:
            MODEL_CATALOG.update(json.load(f))
    except Exception:
        logger.exception("Could not read %s, ignoring", _CUSTOM_CATALOG_PATH)


def _read_custom_catalog() -> dict:
    custom: dict = {}
    if os.path.exists(_CUSTOM_CATALOG_PATH):
        try:
            with open(_CUSTOM_CATALOG_PATH) as f:
                custom = json.load(f)
        except Exception:
            logger.exception("Could not read %s, rewriting", _CUSTOM_CATALOG_PATH)
    return custom


def _write_custom_catalog(custom: dict) -> None:
    directory = os.path.dirname(_CUSTOM_CATALOG_PATH) or MODELS_DIR
    os.makedirs(directory, exist_ok=True)
    temporary = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=directory, prefix=".catalog-", suffix=".tmp", delete=False
        ) as f:
            temporary = f.name
            json.dump(custom, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, _CUSTOM_CATALOG_PATH)
    finally:
        if temporary and os.path.exists(temporary):
            os.remove(temporary)


def _persist_custom_entry(name: str, info: dict) -> None:
    with _catalog_lock:
        custom = _read_custom_catalog()
        custom[name] = dict(info)
        _write_custom_catalog(custom)


def _remove_custom_entry(name: str) -> None:
    with _catalog_lock:
        if not os.path.exists(_CUSTOM_CATALOG_PATH):
            return
        try:
            custom = _read_custom_catalog()
            if name in custom:
                del custom[name]
                _write_custom_catalog(custom)
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
    enrich_entry_from_gguf(name)
    return {"name": name, **MODEL_CATALOG[name]}


def _model_path(name: str) -> str | None:
    info = MODEL_CATALOG.get(name)
    if not info:
        return None
    try:
        p = _path_in_models_dir(info["filename"])
    except ValueError:
        logger.error("Ignoring unsafe model path for catalog entry %s", name)
        return None
    shard = _SHARD_RE.search(p)
    if shard and not all(os.path.exists(path) for path in _shard_paths(p, shard)):
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


def _sniff_gguf_metadata(path: str) -> dict:
    """Read capability metadata from a GGUF header."""
    from llama_cpp import Llama

    reader = Llama(model_path=path, vocab_only=True, verbose=False)
    meta = getattr(reader, "metadata", None) or {}
    del reader
    arch = str(meta.get("general.architecture") or "")
    native_ctx = 0
    if arch:
        try:
            native_ctx = int(meta.get(f"{arch}.context_length") or 0)
        except (TypeError, ValueError):
            native_ctx = 0
    template = str(meta.get("tokenizer.chat_template") or "").lower()
    tool_use = any(k in template for k in ("tool_call", '"tools"', "tools ", ".tools"))
    out: dict = {}
    if arch:
        out["architecture"] = arch
    if native_ctx:
        out["native_context_length"] = native_ctx
    if template:
        out["tool_use"] = tool_use
    return out


def enrich_entry_from_gguf(name: str) -> None:
    """Add GGUF capability metadata to a catalog entry."""
    info = MODEL_CATALOG.get(name)
    path = _model_path(name)
    if not info or not path:
        return
    try:
        sniffed = _sniff_gguf_metadata(path)
    except Exception:  # noqa: BLE001, a bad header must never break the flow
        logger.exception("GGUF metadata sniff failed for %s", name)
        return
    if not sniffed:
        return
    info.update(sniffed)
    if not info.get("size_mb"):
        info["size_mb"] = round(os.path.getsize(path) / (1024 * 1024), 1)
    _persist_custom_entry(name, info)


def list_untracked_gguf_files() -> list[dict]:
    """List GGUF files not referenced by the catalog."""
    if not os.path.isdir(MODELS_DIR):
        return []
    known = {info.get("filename") for info in MODEL_CATALOG.values()}
    out = []
    for f in sorted(os.listdir(MODELS_DIR)):
        if not f.endswith(".gguf") or "mmproj" in f.lower() or f in known:
            continue
        if not _is_primary_shard(f):
            continue
        try:
            size_mb = round(os.path.getsize(os.path.join(MODELS_DIR, f)) / (1024 * 1024), 1)
        except OSError:
            continue
        out.append({"filename": f, "size_mb": size_mb})
    return out


def update_catalog_entry(
    name: str,
    *,
    description: str | None = None,
    context_length: int | None = None,
) -> dict:
    """Persist editable catalog fields."""
    info = MODEL_CATALOG.get(name)
    if info is None:
        raise KeyError(name)
    if description is not None:
        info["description"] = description.strip()
    if context_length is not None and context_length > 0:
        info["context_length"] = int(context_length)
    _persist_custom_entry(name, info)
    return {"name": name, **info}


def host_memory() -> dict:
    """Return host memory in MB."""
    try:
        fields = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                fields[key] = rest.strip()
        total_kb = int((fields.get("MemTotal") or "0 kB").split()[0])
        avail_kb = int((fields.get("MemAvailable") or "0 kB").split()[0])
        return {
            "total_mb": round(total_kb / 1024),
            "available_mb": round(avail_kb / 1024),
        }
    except Exception:  # noqa: BLE001, non-Linux host or unreadable meminfo
        return {}


def idle_state() -> dict:
    ttl = settings.idle_ttl_minutes
    return {
        "idle_ttl_minutes": ttl,
        "idle_seconds": round(time.monotonic() - _last_used_ts) if _loaded_model_name else 0,
    }


async def maybe_unload_idle() -> str | None:
    """Unload the model after the idle timeout."""
    global _llm, _loaded_model_name
    ttl = settings.idle_ttl_minutes
    if ttl <= 0 or _loaded_model_name is None:
        return None
    if time.monotonic() - _last_used_ts < ttl * 60:
        return None
    async with _model_lock:
        if _loaded_model_name is None or time.monotonic() - _last_used_ts < ttl * 60:
            return None
        name = _loaded_model_name
        _llm = None
        _loaded_model_name = None
        logger.info("Idle TTL (%dm) reached, unloaded %s", ttl, name)
        return name


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


def _download_size_mb(items: list[dict], best: str) -> float:
    """Return the total download size for a GGUF set."""
    shard = _SHARD_RE.search(best)
    if shard:
        prefix = best[: shard.start()]
        total = sum(
            it.get("size", 0)
            for it in items
            if it.get("path", "").startswith(prefix) and it["path"].endswith(".gguf")
        )
    else:
        total = next((it.get("size", 0) for it in items if it.get("path") == best), 0)
    return round(total / (1024 * 1024), 1) if total else 0


async def search_huggingface(query: str, limit: int = 20) -> list[dict]:
    out: list[dict] = []
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            f"{_HF_API}/models",
            params={
                "search": query,
                "filter": "gguf",
                "sort": "downloads",
                "direction": "-1",
                "limit": str(limit),
                "full": "true",
            },
        )
        resp.raise_for_status()
        repos = resp.json()[:limit]
        trees = await asyncio.gather(
            *[client.get(f"{_HF_API}/models/{r['id']}/tree/main?recursive=true") for r in repos],
            return_exceptions=True,
        )
        for repo, t in zip(repos, trees):
            if isinstance(t, Exception) or t.status_code != 200:
                continue
            items = t.json()
            best = _pick_best_gguf([it["path"] for it in items if it.get("type") == "file"])
            if not best:
                continue
            already = any(info["repo_id"] == repo["id"] for info in MODEL_CATALOG.values())
            out.append({
                "repo_id": repo["id"],
                "filename": best,
                "size_mb": _download_size_mb(items, best),
                "downloads": repo.get("downloads", 0),
                "likes": repo.get("likes", 0),
                "last_modified": repo.get("lastModified") or repo.get("createdAt") or "",
                "tags": [t for t in repo.get("tags", []) if ":" not in t and t != "gguf"],
                "in_catalog": already,
            })
    return out


async def hf_model_details(repo_id: str) -> dict:
    """Return HuggingFace model metadata and file sizes."""
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        resp = await client.get(f"{_HF_API}/models/{repo_id}")
        resp.raise_for_status()
        data = resp.json()
        try:
            tree = await client.get(f"{_HF_API}/models/{repo_id}/tree/main?recursive=true")
            if tree.status_code == 200:
                sizes = {
                    item["path"]: item.get("size", 0)
                    for item in tree.json()
                    if item.get("type") == "file"
                }
                data["files"] = [
                    {"path": p, "size": sizes.get(p, 0), "is_gguf": p.endswith(".gguf")}
                    for p in sorted(sizes)
                ]
        except Exception:  # noqa: BLE001, tree is a nicety, not required
            logger.debug("tree fetch failed for %s", repo_id)
    return data


def catalog_entry(name: str) -> dict | None:
    """Return one catalog entry with computed state."""
    info = MODEL_CATALOG.get(name)
    if info is None:
        return None
    return {
        "name": name,
        **info,
        "downloaded": is_downloaded(name),
        "active": name == _loaded_model_name,
    }


def download_progress() -> dict[str, dict]:
    progress = dict(_download_progress)
    for name, info in MODEL_CATALOG.items():
        if name in progress or is_downloaded(name):
            continue
        try:
            downloaded = _downloaded_bytes(_download_targets(info))
        except ValueError:
            continue
        if not downloaded:
            continue
        total = round(float(info.get("size_mb") or 0) * 1024 * 1024)
        progress[name] = {
            "progress": round(downloaded / total * 100, 1) if total else 0,
            "downloaded_mb": round(downloaded / (1024 * 1024), 1),
            "total_mb": info.get("size_mb") or 0,
            "status": "paused",
        }
    return progress


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
        targets = _download_targets(info)
        resumed = _downloaded_bytes(targets)
        expected = round(float(info["size_mb"]) * 1024 * 1024)
        generation = object()
        _download_generations[name] = generation
        _download_progress[name] = {
            "progress": round(resumed / expected * 100, 1) if expected else 0,
            "downloaded_mb": round(resumed / (1024 * 1024), 1),
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
            completed = 0
            for filename, target in zip(_download_filenames(info["filename"]), targets):
                if os.path.exists(target):
                    completed += os.path.getsize(target)
                    continue

                def _file_progress(current: int, total: int) -> None:
                    aggregate_total = max(expected, completed + total)
                    _update(completed + current, aggregate_total)

                download_file(
                    _hf_download_url(info["repo_id"], filename),
                    target,
                    control=lambda: _download_control.get(name),
                    progress=_file_progress,
                    headers=_hf_headers(),
                )
                completed += os.path.getsize(target)
            return targets[0]

        try:
            path = await asyncio.to_thread(_do_download)
            await asyncio.to_thread(enrich_entry_from_gguf, name)
            _download_progress[name] = {
                **_download_progress.get(name, {}),
                "progress": 100,
                "status": "complete",
            }
            _schedule_progress_cleanup(name, 10, generation)
            return {"name": name, "status": "downloaded", "path": path}
        except DownloadAborted as ab:
            if ab.mode == "cancel":
                await asyncio.to_thread(_purge_partial_download, name)
                _download_progress[name] = {
                    **_download_progress.get(name, {}),
                    "status": "cancelled",
                }
                _schedule_progress_cleanup(name, 4, generation)
            else:
                _download_progress[name] = {
                    **_download_progress.get(name, {}),
                    "status": "paused",
                }
            return {"name": name, "status": ab.mode}
        except Exception:
            # Keep the partial bytes: once the error entry expires the download
            # reappears as resumable "paused" (reconstructed from disk), so no
            # progress is lost. Only explicit cancel or the orphan sweep deletes.
            _download_progress[name]["status"] = "error"
            _schedule_progress_cleanup(name, 30, generation)
            raise
        finally:
            _download_control.pop(name, None)


def _schedule_progress_cleanup(name: str, delay: float, generation: object) -> None:
    async def _cleanup():
        await asyncio.sleep(delay)
        if _download_generations.get(name) is generation:
            _download_progress.pop(name, None)
            _download_generations.pop(name, None)

    start(_cleanup(), label=f"download-progress-cleanup:{name}")


def request_download_control(name: str, mode: str) -> bool:
    entry = _download_progress.get(name) or download_progress().get(name)
    if not entry or mode not in ("pause", "cancel"):
        return False
    _download_progress[name] = entry
    if entry.get("status") in ("paused", "error") and mode == "cancel":
        generation = _download_generations.setdefault(name, object())
        _purge_partial_download(name)
        _download_progress[name] = {**entry, "status": "cancelled"}
        _schedule_progress_cleanup(name, 4, generation)
        return True
    if entry.get("status") not in ("starting", "downloading"):
        return False
    _download_control[name] = mode
    _download_progress[name] = {
        **entry,
        "status": "pausing" if mode == "pause" else "cancelling",
    }
    return True


def _purge_partial_download(name: str) -> None:
    """Remove a cancelled model and its partial files."""
    info = MODEL_CATALOG.get(name, {})
    fname = info.get("filename")
    if fname:
        try:
            base = _path_in_models_dir(fname)
        except ValueError:
            base = None
        if base:
            for p in _download_targets(info):
                try:
                    os.remove(p)
                except OSError:
                    pass
                remove_partial(p)
    if "community" in (info.get("tags") or []):
        _remove_custom_entry(name)
        MODEL_CATALOG.pop(name, None)


def sweep_stale_downloads() -> list[str]:
    """Startup sweep for download debris nothing tracks anymore: the legacy
    HF blob cache dir and ``.part``/``.incomplete`` files that belong to no
    catalog entry (catalog-owned partials are kept — they surface as resumable
    paused downloads). Returns what was removed, for the boot log."""
    removed: list[str] = []
    if not os.path.isdir(MODELS_DIR):
        return removed
    legacy_cache = os.path.join(MODELS_DIR, ".cache")
    if os.path.isdir(legacy_cache):
        shutil.rmtree(legacy_cache, ignore_errors=True)
        removed.append(".cache/")
    owned: set[str] = set()
    for info in MODEL_CATALOG.values():
        try:
            owned.update(_download_targets(info))
        except (KeyError, ValueError):
            continue
    for entry in sorted(os.listdir(MODELS_DIR)):
        path = os.path.join(MODELS_DIR, entry)
        if not os.path.isfile(path):
            continue
        is_partial = entry.endswith(".part") or entry.endswith(".incomplete")
        if is_partial and path.removesuffix(".part") not in owned:
            try:
                os.remove(path)
                removed.append(entry)
            except OSError:
                pass
    if removed:
        logger.info("Swept stale download debris: %s", ", ".join(removed))
    return removed


def _download_filenames(filename: str) -> list[str]:
    shard = _SHARD_RE.search(filename)
    if shard is None:
        return [filename]
    prefix, total = filename[: shard.start()], int(shard.group(2))
    return [f"{prefix}-{index:05d}-of-{total:05d}.gguf" for index in range(1, total + 1)]


def _download_targets(info: dict) -> list[str]:
    return [_path_in_models_dir(name) for name in _download_filenames(info["filename"])]


def _downloaded_bytes(targets: list[str]) -> int:
    return sum(
        os.path.getsize(path if os.path.exists(path) else f"{path}.part")
        for path in targets
        if os.path.exists(path) or os.path.exists(f"{path}.part")
    )


def _shard_paths(base: str, shard: re.Match) -> list[str]:
    total = int(shard.group(2))
    prefix = base[: shard.start()]
    return [f"{prefix}-{index:05d}-of-{total:05d}.gguf" for index in range(1, total + 1)]


def _hf_download_url(repo_id: str, filename: str) -> str:
    return (
        f"https://huggingface.co/{quote(repo_id, safe='/')}/resolve/main/"
        f"{quote(filename, safe='/')}"
    )


def _hf_headers() -> dict[str, str]:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


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
    info = MODEL_CATALOG.get(name)
    if info is None or not _model_path(name):
        return False
    for path in _download_targets(info):
        try:
            os.remove(path)
        except OSError:
            pass
    _remove_custom_entry(name)
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
    global _last_used_ts
    _last_used_ts = time.monotonic()
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
    global _last_used_ts
    async with _model_lock:
        if requested_model and requested_model != _loaded_model_name:
            if not is_downloaded(requested_model):
                raise FileNotFoundError(f"Model {requested_model!r} not downloaded.")
            await asyncio.to_thread(_load_sync, requested_model)
        _last_used_ts = time.monotonic()
        yield _llm, requested_model or _loaded_model_name
        _last_used_ts = time.monotonic()


def get_llm():
    """Return the currently loaded llama-cpp instance (or None)."""
    return _llm
