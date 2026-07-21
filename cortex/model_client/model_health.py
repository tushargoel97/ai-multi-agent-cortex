"""Short-lived auto-mode cooldowns for retryable model failures."""

from __future__ import annotations

import time

COOLDOWN_SECONDS = 300.0
_cooldowns: dict[str, float] = {}


def report_model_failure(model_id: str | None) -> None:
    if model_id:
        _cooldowns[model_id] = time.monotonic() + COOLDOWN_SECONDS


def report_model_success(model_id: str | None) -> None:
    if model_id:
        _cooldowns.pop(model_id, None)


def in_cooldown(model_id: str) -> bool:
    until = _cooldowns.get(model_id)
    if until is None:
        return False
    if time.monotonic() >= until:
        _cooldowns.pop(model_id, None)
        return False
    return True
