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

from .config import settings

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
        "fusing",
        "converting",
        "researching",
        "scraping",
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


def _base_model_marker() -> "Path":
    return settings.adapters_dir / "base_model.txt"


def start_training(
    iters: int = 600,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    base_model: str | None = None,
    resume: bool = False,
) -> None:
    global _proc
    adapter_file = settings.adapters_dir / "adapters.safetensors"
    if resume:
        # Quick top-up: continue from the existing adapters instead of starting
        # fresh. Resuming against a different base than the adapters were
        # trained on silently produces a broken model, so always reuse the
        # base recorded in base_model.txt and ignore any requested override.
        if not adapter_file.exists():
            raise FileNotFoundError(
                f"no existing adapters at {adapter_file} — run a full train "
                "before a quick top-up"
            )
        marker = _base_model_marker()
        base = (
            marker.read_text().strip()
            if marker.exists()
            else (base_model or settings.base_model)
        )
    else:
        base = base_model or settings.base_model
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
            base_model=base,
            resume=resume,
        )

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", base,
        "--train",
        "--data", str(settings.data_dir),
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--num-layers", "16",
        "--adapter-path", str(settings.adapters_dir),
        "--steps-per-report", "10",
        "--steps-per-eval", "50",
        "--save-every", "100",
    ]
    if resume:
        # Warm-start from the current adapters (saved back to the same path).
        cmd += ["--resume-adapter-file", str(adapter_file)]

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Remember which base these adapters belong to — fusing against a
    # different base than trained silently produces a broken model.
    settings.adapters_dir.mkdir(parents=True, exist_ok=True)
    _base_model_marker().write_text(base)
    _proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(settings.artifacts_dir),
    )
    threading.Thread(target=_watch_training, args=(_proc,), daemon=True).start()


def _watch_training(proc: subprocess.Popen) -> None:
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
        _snapshot_logs()
        if _stop_requested:
            _status["phase"] = "stopped"
        elif code == 0:
            _status["phase"] = "trained"
        else:
            _status["phase"] = "error"
            _status["error"] = f"training exited with code {code} — see log_tail"


def stop() -> bool:
    global _stop_requested
    with _lock:
        if _proc is None or _proc.poll() is not None:
            return False
        _stop_requested = True
    _proc.terminate()
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
                        summaries.append(f"{name}: not a real product — {entry.get('notes', '')[:80]}")
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
                f"no trained adapters at {settings.adapters_dir} — train first"
            )
        if not settings.convert_script.exists():
            raise FileNotFoundError(
                f"{settings.convert_script} missing — run `bash trainer/setup.sh` first"
            )
        _reset("fusing", job="convert", output_name=output_name)
    threading.Thread(target=_run_convert, args=(output_name,), daemon=True).start()


def _sanitize_fused_tokenizer() -> None:
    """Drop tokenizer tokens whose ids fall outside the model's vocab.

    Gemma 3 text checkpoints ship the family-shared tokenizer, which includes
    multimodal specials (e.g. <image_soft_token> at id 262144) beyond the text
    model's embedding rows — llama.cpp's converter asserts on those.
    """
    import json

    cfg_path = settings.fused_dir / "config.json"
    tok_path = settings.fused_dir / "tokenizer.json"
    if not (cfg_path.exists() and tok_path.exists()):
        return
    vocab_size = json.loads(cfg_path.read_text()).get("vocab_size")
    if not vocab_size:
        return

    tok = json.loads(tok_path.read_text())
    added = tok.get("added_tokens", [])
    over = [t for t in added if t.get("id", 0) >= vocab_size]
    if not over:
        return
    dropped = {t["content"] for t in over}
    tok["added_tokens"] = [t for t in added if t.get("id", 0) < vocab_size]
    tok_path.write_text(json.dumps(tok, ensure_ascii=False))
    with _lock:
        _log(f"Dropped {len(over)} out-of-vocab tokenizer token(s): {sorted(dropped)}")

    # Scrub every reference in tokenizer_config — transformers re-registers
    # named special tokens at runtime (assigning fresh out-of-range ids) if
    # they are referenced but missing from the vocab.
    tc_path = settings.fused_dir / "tokenizer_config.json"
    if tc_path.exists():
        tc = json.loads(tc_path.read_text())
        decoder = tc.get("added_tokens_decoder")
        if isinstance(decoder, dict):
            tc["added_tokens_decoder"] = {
                k: v for k, v in decoder.items() if int(k) < vocab_size
            }
        for key in list(tc.keys()):
            value = tc[key]
            if isinstance(value, str) and value in dropped:
                del tc[key]
            elif isinstance(value, list):
                tc[key] = [v for v in value if v not in dropped]
            elif isinstance(value, dict) and key != "added_tokens_decoder":
                tc[key] = {k: v for k, v in value.items() if v not in dropped}
        tc_path.write_text(json.dumps(tc, ensure_ascii=False))


def _run_convert(output_name: str) -> None:
    gguf_path = settings.models_dir / f"{output_name}.gguf"
    # Write to a temp file and atomically rename: the ai service may have the
    # old GGUF mmap'd (llama.cpp) — truncating that inode in place crashes it.
    tmp_path = settings.models_dir / f"{output_name}.gguf.tmp"
    steps = [
        (
            "fusing",
            [
                sys.executable, "-m", "mlx_lm", "fuse",
                "--model", (
                    _base_model_marker().read_text().strip()
                    if _base_model_marker().exists()
                    else settings.base_model
                ),
                "--adapter-path", str(settings.adapters_dir),
                "--save-path", str(settings.fused_dir),
            ],
        ),
        (
            "converting",
            [
                sys.executable, str(settings.convert_script),
                str(settings.fused_dir),
                "--outfile", str(tmp_path),
                "--outtype", "q8_0",
            ],
        ),
    ]

    shutil.rmtree(settings.fused_dir, ignore_errors=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    for phase, cmd in steps:
        if phase == "converting":
            _sanitize_fused_tokenizer()
        with _lock:
            _status["phase"] = phase
        result = subprocess.run(cmd, capture_output=True, text=True)
        with _lock:
            for line in (result.stdout + result.stderr).splitlines()[-30:]:
                _log(line)
            if result.returncode != 0:
                _snapshot_logs()
                _status["phase"] = "error"
                _status["error"] = f"{phase} failed (exit {result.returncode}) — see log_tail"
                return

    tmp_path.replace(gguf_path)

    with _lock:
        _snapshot_logs()
        _status["phase"] = "done"
        _status["gguf_filename"] = gguf_path.name
