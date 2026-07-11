"""Backward-compatible metadata for adapters produced by a backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LEGACY_BACKEND_ID = "mlx-lora"


def _read_marker(path: Path, fallback: str) -> str:
    if not path.exists():
        return fallback
    value = path.read_text().strip()
    return value or fallback


@dataclass(frozen=True)
class AdapterMetadata:
    backend_id: str
    source_model: str
    training_model: str
    run_id: str = ""

    @classmethod
    def load(cls, adapters_dir: Path, fallback_model: str) -> "AdapterMetadata":
        training_model = _read_marker(adapters_dir / "base_model.txt", fallback_model)
        return cls(
            backend_id=_read_marker(
                adapters_dir / "backend_id.txt", LEGACY_BACKEND_ID
            ),
            source_model=_read_marker(
                adapters_dir / "source_model.txt", training_model
            ),
            training_model=training_model,
            run_id=_read_marker(adapters_dir / "run_id.txt", ""),
        )

    def write(self, adapters_dir: Path) -> None:
        adapters_dir.mkdir(parents=True, exist_ok=True)
        (adapters_dir / "backend_id.txt").write_text(self.backend_id)
        (adapters_dir / "source_model.txt").write_text(self.source_model)
        (adapters_dir / "base_model.txt").write_text(self.training_model)
        (adapters_dir / "run_id.txt").write_text(self.run_id)


def adapter_backend_id(adapters_dir: Path) -> str:
    return _read_marker(adapters_dir / "backend_id.txt", LEGACY_BACKEND_ID)
