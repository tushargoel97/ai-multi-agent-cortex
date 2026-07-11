from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from statistics import median
from typing import Any

_lock = threading.Lock()
_ACTIVE_PHASES = {"preparing", "training", "fusing", "converting", "evaluating"}


def save(directory: Path, record: dict[str, Any]) -> dict[str, Any]:
    run_id = record["run_id"]
    path = directory / f"{run_id}.json"
    with _lock:
        directory.mkdir(parents=True, exist_ok=True)
        current = _read(path) or {}
        current.update(record)
        current["updated_at"] = time.time()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(current, separators=(",", ":"), sort_keys=True))
        tmp.replace(path)
    return current


def list_all(directory: Path) -> list[dict[str, Any]]:
    with _lock:
        records = (
            [_read(path) for path in directory.glob("*.json")]
            if directory.exists()
            else []
        )
    return sorted(
        (record for record in records if record),
        key=lambda item: item.get("started_at", 0),
        reverse=True,
    )


def get(directory: Path, run_id: str) -> dict[str, Any] | None:
    return next(
        (record for record in list_all(directory) if record["run_id"] == run_id),
        None,
    )


def estimate_seconds(
    directory: Path,
    backend_id: str,
    base_model: str,
    iters: int,
    batch_size: int,
) -> tuple[int | None, int]:
    rates = [
        run["elapsed_seconds"] / (run["iter"] * run["batch_size"])
        for run in list_all(directory)
        if run.get("backend_id") == backend_id
        and run.get("base_model") == base_model
        and run.get("selected_checkpoint")
        and run.get("elapsed_seconds")
        and run.get("iter")
        and run.get("batch_size")
    ]
    return (
        (round(median(rates) * iters * batch_size * 1.1), len(rates))
        if rates
        else (None, 0)
    )


def recover(directory: Path) -> None:
    for record in list_all(directory):
        if record.get("phase") in _ACTIVE_PHASES:
            save(directory, {"run_id": record["run_id"], "phase": "interrupted"})


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None
