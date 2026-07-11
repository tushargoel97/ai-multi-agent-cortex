"""Training backend registration and lookup."""

from __future__ import annotations

from .base import TrainingBackend
from .mlx import MlxLoraBackend, MlxQLoraBackend

_BACKENDS: dict[str, TrainingBackend] = {
    backend.description.id: backend
    for backend in (MlxLoraBackend(), MlxQLoraBackend())
}


def get_backend(backend_id: str) -> TrainingBackend:
    try:
        return _BACKENDS[backend_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown backend {backend_id!r}; choose one of {', '.join(_BACKENDS)}"
        ) from exc


def registered_backends() -> tuple[TrainingBackend, ...]:
    return tuple(_BACKENDS.values())
