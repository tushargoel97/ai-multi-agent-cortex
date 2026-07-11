"""Host resource discovery exposed by the trainer capabilities API."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from .registry import registered_backends


def _ram_gb() -> float | None:
    try:
        if platform.system() == "Darwin":
            raw = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return round(int(raw.strip()) / 1024**3, 1)
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / 1024**3, 1)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


@lru_cache(maxsize=1)
def _gpu_name() -> str:
    if platform.system() != "Darwin":
        return ""
    try:
        raw = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            text=True,
            timeout=15,
        )
        displays = json.loads(raw).get("SPDisplaysDataType") or []
        return str(displays[0].get("sppci_model") or "") if displays else ""
    except (OSError, ValueError, subprocess.SubprocessError):
        return ""


def capabilities(
    *,
    data_dir: Path,
    artifacts_dir: Path,
    host_id: str,
    host_label: str,
    default_backend: str,
) -> dict:
    train_file = data_dir / "train.jsonl"
    dataset_n = sum(1 for _ in train_file.open()) if train_file.exists() else 0
    backends = []
    for backend in registered_backends():
        available, reason = backend.available()
        desc = backend.description
        backends.append(
            {
                **desc.__dict__,
                "available": available,
                "reason": reason,
                "estimated_seconds": backend.estimate_seconds(
                    iters=600,
                    batch_size=4,
                    needs_prepare=desc.algorithm == "qlora_4bit",
                ),
                "dataset_examples": dataset_n,
            }
        )
    disk = shutil.disk_usage(artifacts_dir.parent)
    return {
        "host_id": host_id,
        "label": host_label,
        "os": platform.system().lower(),
        "arch": platform.machine(),
        "gpu": {"name": _gpu_name(), "unified_memory": platform.system() == "Darwin"},
        "ram_gb": _ram_gb(),
        "free_disk_gb": round(disk.free / 1024**3, 1),
        "default_backend": default_backend,
        "backends": backends,
        "base_models": [
            {
                "id": "unsloth/gemma-3-1b-it",
                "label": "Gemma 3 1B, recommended (~2 GB, fast train + serve, no HF login)",
                "output": "finetuned-gemma3-1b-hardware",
            },
            {
                "id": "google/gemma-4-e2b-it",
                "label": "Gemma 4 E2B, highest quality, 9.5 GB, slow on CPU",
                "output": "finetuned-gemma4-e2b-hardware",
            },
        ],
    }
