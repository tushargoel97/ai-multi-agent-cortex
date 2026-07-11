"""Training backend contracts shared by the trainer pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class TrainingConfig:
    python: str
    source_model: str
    training_model: str
    data_dir: Path
    adapters_dir: Path
    artifacts_dir: Path
    iters: int
    batch_size: int
    learning_rate: float
    resume: bool = False


@dataclass(frozen=True)
class FusionConfig:
    python: str
    training_model: str
    adapters_dir: Path
    fused_dir: Path


@dataclass(frozen=True)
class CommandStep:
    phase: str
    argv: list[str]
    skip_if_exists: Path | None = None


@dataclass(frozen=True)
class BackendDescription:
    id: str
    platform: str
    algorithm: str
    label: str
    quality_tier: str
    min_memory_gb: float
    resume_supported: bool
    description: str


class TrainingBackend(Protocol):
    description: BackendDescription

    def available(self) -> tuple[bool, str]: ...

    def training_model(self, source_model: str, artifacts_dir: Path) -> str: ...

    def command_steps(self, config: TrainingConfig) -> list[CommandStep]: ...

    def fusion_steps(self, config: FusionConfig) -> list[CommandStep]: ...

    def estimate_seconds(
        self, *, iters: int, batch_size: int, needs_prepare: bool
    ) -> int: ...
