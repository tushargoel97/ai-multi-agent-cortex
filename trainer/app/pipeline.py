"""Single-job training/conversion pipeline.

Runs `mlx_lm lora` / `mlx_lm fuse` / llama.cpp's convert_hf_to_gguf.py as
subprocesses, parsing stdout into an in-memory status dict that the API
exposes for polling. Only one job (training OR conversion) may run at a time.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from .backends import AdapterMetadata, get_backend
from .backends.base import FusionConfig, TrainingBackend, TrainingConfig
from .config import settings
from .exporters import gguf_conversion_step, sanitize_fused_tokenizer

_TRAIN_RE = re.compile(r"Iter (\d+): Train loss ([\d.]+)")
_VAL_RE = re.compile(r"Iter (\d+): Val loss ([\d.]+)")

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_stop_requested = False

_status: dict[str, Any] = {"phase": "idle"}


class JobConflictError(RuntimeError):
    """Another job is already running."""


def get_status() -> dict[str, Any]:
    with _lock:
        snap = dict(_status)
    if "log_tail" in snap:
        snap["log_tail"] = list(snap["log_tail"])
    if "history" in snap:
        snap["history"] = list(snap["history"])
    return snap


def _busy() -> bool:
    return _status["phase"] in {
        "training",
        "preparing",
        "fusing",
        "converting",
        "researching",
        "scraping",
        "importing",
    }


def _reset(phase: str, **extra: Any) -> None:
    global _stop_requested
    _stop_requested = False
    _status.clear()
    _status.update(
        {
            "phase": phase,
            "started_at": time.time(),
            "log_tail": deque(maxlen=30),
            **extra,
        }
    )


def _log(line: str) -> None:
    _status["log_tail"].append(line.rstrip())


def _snapshot_logs() -> None:
    # deque isn't JSON-serializable; keep it as a list in reads
    _status["log_tail"] = list(_status["log_tail"])


def start_training(
    iters: int = 600,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    base_model: str | None = None,
    resume: bool = False,
    backend_id: str | None = None,
) -> None:
    selected_backend = backend_id or settings.default_backend
    backend = get_backend(selected_backend)
    available, reason = backend.available()
    if not available:
        raise ValueError(reason or f"backend {selected_backend!r} is unavailable")
    if resume and not backend.description.resume_supported:
        raise ValueError(f"backend {selected_backend!r} does not support resume")

    adapter_file = settings.adapters_dir / "adapters.safetensors"
    if resume:
        # Quick top-up: continue from the existing adapters instead of starting
        # fresh. Resuming against a different base than the adapters were
        # trained on silently produces a broken model, so always reuse the
        # base recorded in base_model.txt and ignore any requested override.
        if not adapter_file.exists():
            raise FileNotFoundError(
                f"no existing adapters at {adapter_file}, run a full train "
                "before a quick top-up"
            )
        metadata = AdapterMetadata.load(
            settings.adapters_dir, base_model or settings.base_model
        )
        if metadata.backend_id != selected_backend:
            raise ValueError(
                f"existing adapters use backend {metadata.backend_id!r}; "
                f"select it instead of {selected_backend!r} or run a full train"
            )
        training_model = metadata.training_model
        source_model = metadata.source_model
    else:
        source_model = base_model or settings.base_model
        training_model = backend.training_model(source_model, settings.artifacts_dir)

    cfg = TrainingConfig(
        python=sys.executable,
        source_model=source_model,
        training_model=training_model,
        data_dir=settings.data_dir,
        adapters_dir=settings.adapters_dir,
        artifacts_dir=settings.artifacts_dir,
        iters=iters,
        batch_size=batch_size,
        learning_rate=learning_rate,
        resume=resume,
    )
    with _lock:
        if _busy():
            raise JobConflictError(f"a job is already running (phase={_status['phase']})")
        _reset(
            "training",
            job="train",
            iter=0,
            total_iters=iters,
            train_loss=None,
            val_loss=None,
            history=[],
            base_model=source_model,
            training_model=training_model,
            backend_id=selected_backend,
            algorithm=backend.description.algorithm,
            resume=resume,
            estimated_seconds=backend.estimate_seconds(
                iters=iters,
                batch_size=batch_size,
                needs_prepare=any(
                    step.skip_if_exists and not step.skip_if_exists.exists()
                    for step in backend.command_steps(cfg)
                ),
            ),
        )

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.adapters_dir.mkdir(parents=True, exist_ok=True)
    # Fusion must use the actual training checkpoint. QLoRA stores a quantized
    # local checkpoint while preserving the original source for display/reuse.
    AdapterMetadata(
        backend_id=selected_backend,
        source_model=source_model,
        training_model=training_model,
    ).write(settings.adapters_dir)
    threading.Thread(
        target=_run_training,
        args=(backend, cfg),
        daemon=True,
    ).start()


def _run_training(backend: TrainingBackend, cfg: TrainingConfig) -> None:
    global _proc
    try:
        for step in backend.command_steps(cfg):
            if step.skip_if_exists is not None and step.skip_if_exists.exists():
                with _lock:
                    _log(f"Reusing prepared model at {step.skip_if_exists.parent}")
                continue
            if step.phase == "preparing":
                Path(cfg.training_model).parent.mkdir(parents=True, exist_ok=True)
            with _lock:
                if _stop_requested:
                    _snapshot_logs()
                    _status["phase"] = "stopped"
                    return
                _status["phase"] = step.phase
                _log(f"Starting {step.phase}: {' '.join(step.argv[:6])} …")
            proc = subprocess.Popen(
                step.argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(settings.artifacts_dir),
            )
            with _lock:
                _proc = proc
                stop_now = _stop_requested
            if stop_now:
                proc.terminate()
            assert proc.stdout is not None
            for line in proc.stdout:
                with _lock:
                    _log(line)
                    if m := _TRAIN_RE.search(line):
                        _status["iter"] = int(m.group(1))
                        _status["train_loss"] = float(m.group(2))
                        _status["history"].append(
                            {"iter": int(m.group(1)), "train_loss": float(m.group(2))}
                        )
                        _status["history"] = _status["history"][-200:]
                    elif m := _VAL_RE.search(line):
                        _status["val_loss"] = float(m.group(2))
            code = proc.wait()
            with _lock:
                _proc = None
                if _stop_requested:
                    _snapshot_logs()
                    _status["phase"] = "stopped"
                    return
                if code != 0:
                    _snapshot_logs()
                    _status["phase"] = "error"
                    _status["error"] = (
                        f"{step.phase} exited with code {code}, see log_tail"
                    )
                    return
        with _lock:
            _snapshot_logs()
            _status["phase"] = "trained"
            _status["elapsed_seconds"] = round(time.time() - _status["started_at"], 1)
    except Exception as exc:  # noqa: BLE001, persist background failure for polling
        with _lock:
            _proc = None
            _snapshot_logs()
            _status["phase"] = "error"
            _status["error"] = f"training failed: {type(exc).__name__}: {exc}"


def stop() -> bool:
    global _stop_requested
    with _lock:
        if _status.get("job") != "train" or not _busy():
            return False
        _stop_requested = True
        proc = _proc if _proc is not None and _proc.poll() is None else None
    # A request can arrive before Popen publishes the process or between QLoRA
    # preparation and training. The worker observes _stop_requested in both
    # windows; terminate immediately when a child is already running.
    if proc is not None:
        proc.terminate()
    return True


def start_scrape(
    index_urls: list[str],
    max_products: int,
    max_pages: int = 20,
    max_depth: int = 2,
    delay_s: float = 2.5,
) -> None:
    """Dynamic spec import → learned_facts.yaml (background).

    AMD's DB and uploaded documents use deterministic parsers; every other URL
    is crawled by the intelligent scrape agent within the given budget.
    """
    with _lock:
        if _busy():
            raise JobConflictError(f"a job is already running (phase={_status['phase']})")
        if not index_urls:
            raise ValueError("no index URLs to scrape")
        _reset(
            "scraping",
            job="scrape",
            scrape_total=0,
            scrape_done=0,
            scrape_current="starting…",
            products_learned=0,
        )
    threading.Thread(
        target=_run_scrape,
        args=(index_urls, max_products, max_pages, max_depth, delay_s),
        daemon=True,
    ).start()


def _run_scrape(
    index_urls: list[str],
    max_products: int,
    max_pages: int = 20,
    max_depth: int = 2,
    delay_s: float = 2.5,
) -> None:
    from . import scraper

    def on_progress(done: int, total: int, label: str) -> None:
        with _lock:
            _status["scrape_done"] = done
            _status["scrape_total"] = total
            _status["scrape_current"] = label
            _log(f"[{done}/{total}] {label}")

    def on_log(msg: str) -> None:
        with _lock:
            _log(msg)

    try:
        summary = scraper.scrape(
            index_urls,
            max_products,
            on_progress,
            max_pages=max_pages,
            max_depth=max_depth,
            delay_s=delay_s,
            on_log=on_log,
        )
        with _lock:
            _status["products_learned"] = len(summary["saved"])
            _status["scrape_saved"] = summary["saved"]
            _status["scrape_skipped"] = summary["skipped"][:20]
            _status["scrape_errors"] = summary["errors"][:10]
            _status["scrape_outcomes"] = summary.get("outcomes", [])[:40]
            _snapshot_logs()
            _status["phase"] = "scrape_done"
    except Exception as e:  # noqa: BLE001
        with _lock:
            _snapshot_logs()
            _status["phase"] = "error"
            _status["error"] = f"scrape failed: {e}"


def start_smart_import(source_items: list, target: str) -> None:
    """Domain-aware import of the selected sources into a reviewable proposal
    (background). ``target`` is 'auto' or 'domain/subdomain'."""
    with _lock:
        if _busy():
            raise JobConflictError(f"a job is already running (phase={_status['phase']})")
        if not source_items:
            raise ValueError("no sources selected")
        _reset(
            "importing",
            job="import",
            import_target=target,
            import_proposal=None,
            scrape_current="reading sources\u2026",
        )
    threading.Thread(
        target=_run_smart_import, args=(source_items, target), daemon=True
    ).start()


def _run_smart_import(source_items: list, target: str) -> None:
    from . import research

    def on_log(msg: str) -> None:
        with _lock:
            _log(msg)
            _status["scrape_current"] = msg

    try:
        proposal = research.propose(source_items, target, on_log=on_log)
        with _lock:
            _snapshot_logs()
            _status["import_proposal"] = proposal
            _status["phase"] = "import_proposed"
    except Exception as e:  # noqa: BLE001
        with _lock:
            _snapshot_logs()
            _status["phase"] = "error"
            _status["error"] = f"smart import failed: {e}"


def start_gap_research(gaps: list[dict]) -> None:
    """Research knowledge gaps on the web → learned_facts.yaml (background)."""
    with _lock:
        if _busy():
            raise JobConflictError(f"a job is already running (phase={_status['phase']})")
        if not gaps:
            raise ValueError("no gaps to research")
        _reset(
            "researching",
            job="research",
            gaps_total=len(gaps),
            gaps_done=0,
            products_learned=0,
            research_results=[],
        )
    threading.Thread(target=_run_gap_research, args=(gaps,), daemon=True).start()


def _run_gap_research(gaps: list[dict]) -> None:
    from . import research

    def log(msg: str) -> None:
        with _lock:
            _log(msg)

    try:
        for i, gap in enumerate(gaps):
            question = gap.get("question", "")
            log(f"Gap: {question[:90]}")
            summaries: list[str] = []
            try:
                names = research.extract_product_names(question)
                log(f"  products: {names}")
                for name in names:
                    entry = research.research_product(name, on_log=log)
                    if entry is None:
                        summaries.append(f"{name}: research failed")
                        continue
                    research.save_learned_entry(entry)
                    with _lock:
                        _status["products_learned"] += 1
                    if entry.get("exists", True):
                        summaries.append(f"{name}: specs learned")
                    else:
                        summaries.append(f"{name}: not a real product, {entry.get('notes', '')[:80]}")
                status = "researched" if summaries else "failed"
            except Exception as e:  # noqa: BLE001
                log(f"  !! {type(e).__name__}: {e}")
                status = "failed"
            with _lock:
                _status["gaps_done"] = i + 1
                _status["research_results"].append(
                    {"id": gap.get("id"), "status": status, "summary": "; ".join(summaries)}
                )
        with _lock:
            _snapshot_logs()
            _status["phase"] = "research_done"
    except Exception as e:  # noqa: BLE001
        with _lock:
            _snapshot_logs()
            _status["phase"] = "error"
            _status["error"] = f"gap research failed: {e}"


def start_convert(output_name: str) -> None:
    with _lock:
        if _busy():
            raise JobConflictError(f"a job is already running (phase={_status['phase']})")
        if not (settings.adapters_dir / "adapters.safetensors").exists():
            raise FileNotFoundError(
                f"no trained adapters at {settings.adapters_dir}, train first"
            )
        if not settings.convert_script.exists():
            raise FileNotFoundError(
                f"{settings.convert_script} missing, run `bash trainer/setup.sh` first"
            )
        metadata = AdapterMetadata.load(settings.adapters_dir, settings.base_model)
        get_backend(metadata.backend_id)
        _reset("fusing", job="convert", output_name=output_name)
    threading.Thread(target=_run_convert, args=(output_name,), daemon=True).start()


def _run_convert(output_name: str) -> None:
    try:
        gguf_path = settings.models_dir / f"{output_name}.gguf"
        # Write to a temp file and atomically rename: the ai service may have
        # the old GGUF mmap'd, truncating that inode in place crashes it.
        tmp_path = settings.models_dir / f"{output_name}.gguf.tmp"
        metadata = AdapterMetadata.load(settings.adapters_dir, settings.base_model)
        backend = get_backend(metadata.backend_id)
        steps = [
            *backend.fusion_steps(
                FusionConfig(
                    python=sys.executable,
                    training_model=metadata.training_model,
                    adapters_dir=settings.adapters_dir,
                    fused_dir=settings.fused_dir,
                )
            ),
            gguf_conversion_step(
                python=sys.executable,
                convert_script=settings.convert_script,
                fused_dir=settings.fused_dir,
                output_path=tmp_path,
            ),
        ]

        shutil.rmtree(settings.fused_dir, ignore_errors=True)
        settings.models_dir.mkdir(parents=True, exist_ok=True)

        for step in steps:
            if step.phase == "converting":
                dropped = sanitize_fused_tokenizer(settings.fused_dir)
                if dropped:
                    with _lock:
                        _log(
                            f"Dropped {len(dropped)} out-of-vocab tokenizer token(s): "
                            f"{dropped}"
                        )
            with _lock:
                _status["phase"] = step.phase
            result = subprocess.run(step.argv, capture_output=True, text=True)
            with _lock:
                for line in (result.stdout + result.stderr).splitlines()[-30:]:
                    _log(line)
                if result.returncode != 0:
                    _snapshot_logs()
                    _status["phase"] = "error"
                    _status["error"] = (
                        f"{step.phase} failed (exit {result.returncode}), see log_tail"
                    )
                    return

        tmp_path.replace(gguf_path)

        with _lock:
            _snapshot_logs()
            _status["phase"] = "done"
            _status["gguf_filename"] = gguf_path.name
    except Exception as exc:  # noqa: BLE001, persist background failure for polling
        with _lock:
            _snapshot_logs()
            _status["phase"] = "error"
            _status["error"] = f"conversion failed: {type(exc).__name__}: {exc}"
