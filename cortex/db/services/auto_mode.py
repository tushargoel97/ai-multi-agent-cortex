"""Auto mode, resolve the model for an intent from the active profile.

The chat UI sends the sentinel ``model_id: "auto"``; the router classifies the
message and each agent node asks this module for the model matching its
intent. Candidates come from ``cortex/declarative/auto_mode.yaml`` and only
registry-enabled models are eligible, so Admin → Models stays in control.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import yaml

from cortex.db.services.app_settings import get_setting, set_setting
from cortex.db.services.llm_registry import (
    ResolvedModel,
    resolve_by_model_id,
)

logger = logging.getLogger(__name__)

AUTO_MODEL_ID = "auto"
PROFILE_SETTING_KEY = "auto_profile"
# The packaged YAML is mirrored here so the admin UI (Postgres-only) can read
# the shipped defaults; admins layer their edits into the overrides key.
DEFAULTS_SETTING_KEY = "auto_mode_defaults"
OVERRIDES_SETTING_KEY = "auto_mode_overrides"
DEFAULT_PROFILE = "balanced"
FAST_TIER = "fast"

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "declarative" / "auto_mode.yaml"

_defaults_published = False


def is_auto(model_id: object) -> bool:
    return isinstance(model_id, str) and model_id.strip().lower() == AUTO_MODEL_ID


@lru_cache(maxsize=1)
def _yaml_profiles() -> dict:
    return (yaml.safe_load(_CONFIG_PATH.read_text()) or {}).get("profiles", {})


def _load_overrides() -> dict:
    """Admin-edited per-intent candidate overrides (app_settings JSON)."""
    raw = get_setting(OVERRIDES_SETTING_KEY, "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning(
            "%s is not valid JSON, ignoring overrides", OVERRIDES_SETTING_KEY
        )
        return {}
    return data if isinstance(data, dict) else {}


def effective_profiles() -> dict:
    """YAML defaults deep-merged with admin overrides.

    An override list for a (profile, intent) replaces that intent's candidate
    list; any intent or profile the admin hasn't touched falls back to the
    YAML shipped with the package.
    """
    profiles = {name: dict(intents) for name, intents in _yaml_profiles().items()}
    for pname, intents in _load_overrides().items():
        if not isinstance(intents, dict):
            continue
        target = profiles.setdefault(pname, {})
        for intent, candidates in intents.items():
            if isinstance(candidates, list):
                target[intent] = [str(c) for c in candidates]
    return profiles


def publish_defaults() -> None:
    """Mirror the packaged YAML defaults into app_settings (once per process).

    The admin UI talks only to Postgres, so this is how it learns the shipped
    candidate lists. Best-effort, the graph must never fail over a UI mirror.
    """
    global _defaults_published
    if _defaults_published:
        return
    _defaults_published = True
    try:
        set_setting(DEFAULTS_SETTING_KEY, json.dumps(_yaml_profiles()))
    except Exception:  # noqa: BLE001, cosmetic mirror for the UI, never fatal
        logger.exception("Could not publish auto-mode defaults to app_settings")


def active_profile() -> str:
    name = get_setting(PROFILE_SETTING_KEY, DEFAULT_PROFILE)
    return name if name in effective_profiles() else DEFAULT_PROFILE


def resolve_auto_candidates(
    intent: str, profile: str | None = None
) -> list[ResolvedModel]:
    """Every enabled candidate for the intent, in priority order (first = best).

    Same candidate list as :func:`resolve_auto_model`, but returns all that
    resolve so callers can build a quota/rate-limit fallback chain. Falls back
    to the fast tier when the intent has no list, and de-duplicates by
    ``model_id`` so the chain never retries the identical model.

    ``profile`` overrides the active profile (e.g. Thinking forces ``quality``).
    """
    profiles = effective_profiles()
    name = (
        profile
        if profile and profile in profiles
        else get_setting(PROFILE_SETTING_KEY, DEFAULT_PROFILE)
    )
    profile_map = profiles.get(name if name in profiles else DEFAULT_PROFILE, {})
    candidates = profile_map.get(intent) or profile_map.get(FAST_TIER) or []
    resolved_list: list[ResolvedModel] = []
    seen: set[str] = set()
    for model_id in candidates:
        try:
            resolved = resolve_by_model_id(model_id)
        except Exception:  # noqa: BLE001, registry hiccup: try next candidate
            logger.exception("auto-mode candidate %r failed to resolve", model_id)
            continue
        if resolved is not None and resolved.model_id not in seen:
            # Skip cloud candidates whose provider has no API key: the client
            # would silently fall back to a possibly-stale env var and 401.
            # Local (and Azure, which may authenticate via AAD) are exempt.
            if resolved.kind.value not in ("local", "azure_openai") and not (
                resolved.api_key or ""
            ).strip():
                logger.warning(
                    "auto-mode candidate %r has no API key on its provider; skipping",
                    model_id,
                )
                continue
            seen.add(resolved.model_id)
            resolved_list.append(resolved)
    try:
        from cortex.model_client.model_health import in_cooldown

        healthy, cooling = [], []
        for resolved in resolved_list:
            (cooling if in_cooldown(resolved.model_id) else healthy).append(resolved)
        return healthy + cooling
    except Exception:  # noqa: BLE001, health ordering is best-effort
        pass
    return resolved_list


_COMPLEXITY_PROFILE = {"simple": "cost", "complex": "quality"}


def profile_for_complexity(complexity: str | None) -> str | None:
    return _COMPLEXITY_PROFILE.get((complexity or "").strip().lower())


def resolve_auto_model(
    intent: str, profile: str | None = None
) -> ResolvedModel | None:
    """First enabled candidate for the intent; falls back to the fast tier.

    ``profile`` overrides the active profile for this lookup (e.g. Thinking
    mode forces the ``quality`` tier).
    """
    candidates = resolve_auto_candidates(intent, profile)
    return candidates[0] if candidates else None



def image_model_candidates() -> list[str]:
    profiles = effective_profiles()
    name = get_setting(PROFILE_SETTING_KEY, DEFAULT_PROFILE)
    profile = profiles.get(name if name in profiles else DEFAULT_PROFILE, {})
    return list(profile.get("image_generation") or [])
