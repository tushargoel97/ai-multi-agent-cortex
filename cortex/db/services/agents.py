"""DB-backed agent management.

Mirrors built-in agents into the ``agents`` table (so their prompts are
editable), keeps unedited built-ins in sync with the packaged YAML, and exposes
custom agents to the router (by description) and to the ``custom_agent`` node.
"""

from __future__ import annotations

import logging

from cortex.db.engine import engine, get_session
from cortex.db.models import Base
from cortex.db.models.agent import Agent, AgentKind

logger = logging.getLogger(__name__)

_tables_ready = False

# Short labels for the built-ins (UI display; built-ins route via intent, not
# description).
_BUILTIN_DESCRIPTIONS = {
    "router": "Classifies each message to the right capability.",
    "generalist": "Friendly general conversation and identity.",
    "researcher": "Factual questions, grounded with web + knowledge base.",
    "reasoner": "Math, logic, and step-by-step problem solving.",
    "coder": "Writes, explains, reviews, and debugs code.",
    "prompt_cacher": "LLM prompt-caching expertise.",
    "specialist": "Self-trained hardware-spec specialist.",
    "synthesizer": "Formats factual answers into clean tables / structure.",
    "shopping": "Region-aware product prices and buying advice.",
    "booking": "Flights, hotels, movies, concerts, and events.",
}


def _ensure_tables() -> None:
    global _tables_ready
    if _tables_ready:
        return
    Base.metadata.create_all(engine, tables=[Agent.__table__])
    _tables_ready = True


def _yaml_prompt(spec) -> str:
    p = spec.system_prompt
    return p if isinstance(p, str) else "\n".join(p)


def publish_agents() -> None:
    """Seed built-in agents into the table + keep unedited ones in sync with
    the packaged YAML (never clobbers an admin edit)."""
    try:
        _ensure_tables()
        from cortex.declarative import AGENT_SPECS

        with get_session() as s:
            rows = {r.name: r for r in s.query(Agent).all()}
            for name, spec in AGENT_SPECS.items():
                prompt = _yaml_prompt(spec)
                row = rows.get(name)
                if row is None:
                    s.add(
                        Agent(
                            name=name,
                            kind=AgentKind.BUILTIN.value,
                            description=_BUILTIN_DESCRIPTIONS.get(name, ""),
                            system_prompt=prompt,
                            enabled=True,
                            edited=False,
                        )
                    )
                elif row.kind == AgentKind.BUILTIN.value and not row.edited:
                    # keep unedited built-ins in sync with code
                    row.system_prompt = prompt
                    if not row.description:
                        row.description = _BUILTIN_DESCRIPTIONS.get(name, "")
        _publish_agent_defaults()
    except Exception:  # noqa: BLE001 — agent mirror is best-effort
        logger.exception("publish_agents failed")


def _publish_agent_defaults() -> None:
    """Mirror the packaged built-in prompts/descriptions to app_settings so the
    admin UI can show and reset-to-default without importing cortex."""
    try:
        import json

        from cortex.db.services.app_settings import set_setting
        from cortex.declarative import AGENT_SPECS

        defaults = {
            name: {
                "description": _BUILTIN_DESCRIPTIONS.get(name, ""),
                "system_prompt": _yaml_prompt(spec),
            }
            for name, spec in AGENT_SPECS.items()
        }
        set_setting("agent_defaults", json.dumps(defaults))
    except Exception:  # noqa: BLE001
        logger.exception("agent defaults mirror failed")


def agent_prompt(name: str) -> str | None:
    """Effective system-prompt template for an agent (None ⇒ use YAML)."""
    try:
        _ensure_tables()
        with get_session() as s:
            row = s.query(Agent).filter(Agent.name == name).first()
            if row is None or not (row.system_prompt or "").strip():
                return None
            return row.system_prompt
    except Exception:  # noqa: BLE001
        return None


def custom_agents_for_routing() -> list[dict]:
    """Enabled custom agents with a description — the router picks among these."""
    try:
        _ensure_tables()
        with get_session() as s:
            rows = (
                s.query(Agent)
                .filter(Agent.kind == AgentKind.CUSTOM.value, Agent.enabled.is_(True))
                .all()
            )
            return [
                {"name": r.name, "description": r.description}
                for r in rows
                if (r.description or "").strip()
            ]
    except Exception:  # noqa: BLE001
        return []


def load_custom_agent(name: str) -> dict | None:
    """A single enabled custom agent's spec, or None."""
    try:
        _ensure_tables()
        with get_session() as s:
            r = (
                s.query(Agent)
                .filter(
                    Agent.name == name,
                    Agent.kind == AgentKind.CUSTOM.value,
                    Agent.enabled.is_(True),
                )
                .first()
            )
            if r is None:
                return None
            return {
                "name": r.name,
                "system_prompt": r.system_prompt,
                "description": r.description,
            }
    except Exception:  # noqa: BLE001
        return None
