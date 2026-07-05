"""Auto mode — resolve the model for an intent from the active profile.

The chat UI sends the sentinel ``model_id: "auto"``; the router classifies the
message and each agent node asks this module for the model matching its
intent. Candidates come from ``cortex/declarative/auto_mode.yaml`` and only
registry-enabled models are eligible, so Admin → Models stays in control.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

from cortex.db.services.app_settings import get_setting
from cortex.db.services.llm_registry import (
    ResolvedModel,
    resolve_by_model_id,
    resolve_fine_tuned_model,
)

logger = logging.getLogger(__name__)

AUTO_MODEL_ID = "auto"
PROFILE_SETTING_KEY = "auto_profile"
DEFAULT_PROFILE = "balanced"
FAST_TIER = "fast"

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "declarative" / "auto_mode.yaml"


def is_auto(model_id: object) -> bool:
    return isinstance(model_id, str) and model_id.strip().lower() == AUTO_MODEL_ID


@lru_cache(maxsize=1)
def _profiles() -> dict:
    return (yaml.safe_load(_CONFIG_PATH.read_text()) or {}).get("profiles", {})


def active_profile() -> str:
    name = get_setting(PROFILE_SETTING_KEY, DEFAULT_PROFILE)
    return name if name in _profiles() else DEFAULT_PROFILE


def resolve_auto_model(intent: str) -> ResolvedModel | None:
    """First enabled candidate for the intent; falls back to the fast tier."""
    profile = _profiles().get(active_profile(), {})
    candidates = profile.get(intent) or profile.get(FAST_TIER) or []
    for model_id in candidates:
        try:
            if model_id == "finetuned":
                resolved = resolve_fine_tuned_model()
            else:
                resolved = resolve_by_model_id(model_id)
        except Exception:  # noqa: BLE001 — registry hiccup: try next candidate
            logger.exception("auto-mode candidate %r failed to resolve", model_id)
            continue
        if resolved is not None:
            return resolved
    return None


def image_model_candidates() -> list[str]:
    profile = _profiles().get(active_profile(), {})
    return list(profile.get("image_generation") or [])
