"""Single-job trainer pipeline."""

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
from uuid import uuid4

from . import runs
from .backends import AdapterMetadata, get_backend
from .backends.base import FusionConfig, TrainingBackend, TrainingConfig
from .config import settings
from .evaluator import evaluate
from .exporters import gguf_conversion_step, sanitize_fused_tokenizer

_TRAIN_RE = re.compile(r"Iter (\d+): Train loss ([\d.]+)")
_VAL_RE = re.compile(r"Iter (\d+): Val loss ([\d.]+)")

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_stop_requested = False

_status: dict[str, Any] = {"phase": "idle"}
runs.recover(settings.runs_dir)


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
        "evaluating",
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
    _status["log_tail"] = list(_status["log_tail"])


def _run_record() -> dict[str, Any] | None:
    if not (run_id := _status.get("run_id")):
        return None
    record = {
        key: value
        for key, value in _status.items()
        if key not in {"history", "log_tail"}
    } | {"run_id": run_id}
    if (job := record.get("job")) != "train":
        record[f"{job}_started_at"] = record.pop("started_at")
    return record


def _save_run() -> None:
    with _lock:
        record = _run_record()
    if record:
        runs.save(settings.runs_dir, record)


def _finish_training(phase: str, error: str | None = None) -> None:
    global _proc
    with _lock:
        _proc = None
        _snapshot_logs()
        _status["phase"] = phase
        _status["elapsed_seconds"] = round(time.time() - _status["started_at"], 1)
        if error:
            _status["error"] = error
    _save_run()


def _select_best_checkpoint(adapters_dir: Path, iteration: int | None) -> str:
    target = adapters_dir / "adapters.safetensors"
    best = adapters_dir / f"{iteration:07d}_adapters.safetensors" if iteration else target
    if best != target and best.exists():
        shutil.copy2(best, target)
    return best.name if best.exists() else target.name


def _record_validation(iteration: int, loss: float) -> bool:
    _status["val_loss"] = loss
    _status["val_history"].append({"iter": iteration, "val_loss": loss})
    best = _status["best_val_loss"]
    if best is None or loss < best - _status["early_stopping_min_delta"]:
        _status.update(best_val_loss=loss, best_iter=iteration, stale_evals=0)
    else:
        _status["stale_evals"] += 1
    patience = _status["early_stopping_patience"]
    return bool(patience and _status["stale_evals"] >= patience)


def start_training(
    iters: int = 600,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    base_model: str | None = None,
    resume: bool = False,
    backend_id: str | None = None,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 0.001,
) -> str:
    selected_backend = backend_id or settings.default_backend
    backend = get_backend(selected_backend)
    available, reason = backend.available()
    if not available:
        raise ValueError(reason or f"backend {selected_backend!r} is unavailable")
    if resume and not backend.description.resume_supported:
        raise ValueError(f"backend {selected_backend!r} does not support resume")

    adapter_file = settings.adapters_dir / "adapters.safetensors"
    if resume:
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
    run_id = str(uuid4())
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
            val_history=[],
            best_val_loss=None,
            best_iter=None,
            stale_evals=0,
            base_model=source_model,
            training_model=training_model,
            backend_id=selected_backend,
            algorithm=backend.description.algorithm,
            resume=resume,
            run_id=run_id,
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
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
    AdapterMetadata(
        backend_id=selected_backend,
        source_model=source_model,
        training_model=training_model,
        run_id=run_id,
    ).write(settings.adapters_dir)
    _save_run()
    threading.Thread(
        target=_run_training,
        args=(backend, cfg),
        daemon=True,
    ).start()
    return run_id


def _run_training(backend: TrainingBackend, cfg: TrainingConfig) -> None:
    global _proc
    early_stopped = False
    try:
        for step in backend.command_steps(cfg):
            if step.skip_if_exists is not None and step.skip_if_exists.exists():
                with _lock:
                    _log(f"Reusing prepared model at {step.skip_if_exists.parent}")
                _save_run()
                continue
            if step.phase == "preparing":
                Path(cfg.training_model).parent.mkdir(parents=True, exist_ok=True)
            with _lock:
                stop_now = _stop_requested
                if not stop_now:
                    _status["phase"] = step.phase
                    _log(f"Starting {step.phase}: {' '.join(step.argv[:6])} …")
            if stop_now:
                _finish_training("stopped")
                return
            _save_run()
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
                persist = False
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
                        iteration, loss = int(m.group(1)), float(m.group(2))
                        persist = True
                        if _record_validation(iteration, loss):
                            early_stopped = True
                            _status["early_stopped"] = True
                            _log(f"Early stopping at iteration {iteration}")
                            proc.terminate()
                if persist:
                    _save_run()
            code = proc.wait()
            with _lock:
                _proc = None
                if _stop_requested:
                    stopped = True
                else:
                    stopped = False
            if stopped:
                _finish_training("stopped")
                return
            if code != 0 and not early_stopped:
                _finish_training("error", f"{step.phase} exited with code {code}, see log_tail")
                return
            if early_stopped:
                break
        with _lock:
            best_iter = _status.get("best_iter")
        selected = _select_best_checkpoint(cfg.adapters_dir, best_iter)
        with _lock:
            _status["selected_checkpoint"] = selected
        _finish_training("trained")
    except Exception as exc:  # noqa: BLE001, persist background failure for polling
        _finish_training("error", f"training failed: {type(exc).__name__}: {exc}")


def stop() -> bool:
    global _stop_requested
    with _lock:
        if _status.get("job") != "train" or not _busy():
            return False
        _stop_requested = True
        proc = _proc if _proc is not None and _proc.poll() is None else None
    if proc is not None:
        proc.terminate()
    return True


def start_evaluation(run_id: str, model_id: str, cases: int = 12) -> None:
    record = runs.get(settings.runs_dir, run_id)
    if not record:
        raise ValueError(f"unknown run {run_id!r}")
    if Path(record.get("gguf_filename", "")).stem != model_id:
        raise ValueError("model_id does not match the run artifact")
    with _lock:
        if _busy():
            raise JobConflictError(f"a job is already running (phase={_status['phase']})")
        _reset("evaluating", job="evaluate", run_id=run_id, model_id=model_id, cases=cases)
    _save_run()
    threading.Thread(target=_run_evaluation, args=(model_id, cases), daemon=True).start()


def _run_evaluation(model_id: str, cases: int) -> None:
    try:
        result = evaluate(
            settings.data_dir / "valid.jsonl",
            model_id,
            settings.eval_base_url,
            settings.eval_api_key,
            cases,
        )
        with _lock:
            _status.update(phase="evaluated", evaluation=result)
    except Exception as exc:  # noqa: BLE001, persist background failure for polling
        with _lock:
            _status.update(
                phase="evaluation_error",
                error=f"evaluation failed: {type(exc).__name__}: {exc}",
            )
    _save_run()


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
        _reset(
            "fusing",
            job="convert",
            output_name=output_name,
            run_id=metadata.run_id,
        )
    _save_run()
    threading.Thread(target=_run_convert, args=(output_name,), daemon=True).start()


def _run_convert(output_name: str) -> None:
    try:
        gguf_path = settings.models_dir / f"{output_name}.gguf"
        # Preserve the inode of any loaded GGUF until replacement is complete.
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
            _save_run()
            result = subprocess.run(step.argv, capture_output=True, text=True)
            failed = result.returncode != 0
            with _lock:
                for line in (result.stdout + result.stderr).splitlines()[-30:]:
                    _log(line)
                if failed:
                    _snapshot_logs()
                    _status["phase"] = "error"
                    _status["error"] = (
                        f"{step.phase} failed (exit {result.returncode}), see log_tail"
                    )
            if failed:
                _save_run()
                return

        tmp_path.replace(gguf_path)

        with _lock:
            _snapshot_logs()
            _status["phase"] = "done"
            _status["gguf_filename"] = gguf_path.name
        _save_run()
    except Exception as exc:  # noqa: BLE001, persist background failure for polling
        with _lock:
            _snapshot_logs()
            _status["phase"] = "error"
            _status["error"] = f"conversion failed: {type(exc).__name__}: {exc}"
        _save_run()
