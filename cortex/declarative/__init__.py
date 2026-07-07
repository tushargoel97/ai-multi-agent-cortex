"""Agent spec registry, declarative YAML loader.

Loads all agent specs from ``agents.yaml`` (one ``---`` document per agent)
into a flat  name → AgentSpec  mapping.  The registry is populated eagerly at
import time so every other module can simply do::

    from cortex.declarative import AGENT_SPECS, get_agent_spec
"""

import logging
from pathlib import Path

from cortex.declarative.models import AgentSpec, SpecRegistry
from cortex.declarative.yaml_utils import load_yaml_documents
from cortex.enums import Agents
from cortex.errors import AgentSpecNotFoundError

logger = logging.getLogger(__name__)

# All agent specs live in one multi-document YAML file next to this module.
AGENT_SPECS_FILE = Path(__file__).parent / "agents.yaml"

# Module-level registry (populated at the bottom of this file)
AGENT_SPECS: SpecRegistry


def load_agent_specs() -> SpecRegistry:
    """Load every agent spec from ``agents.yaml`` (one ``---`` document each).

    Documents that fail validation are skipped with a warning.

    Returns:
        Flat ``name → AgentSpec`` mapping.
    """
    registry: SpecRegistry = {}

    for spec_data in load_yaml_documents(AGENT_SPECS_FILE):
        registry[spec_data["name"]] = AgentSpec.model_validate(spec_data)

    logger.info("Loaded %d agent spec(s): %s", len(registry), sorted(registry))
    return registry


def get_agent_spec(agent_name: Agents) -> AgentSpec:
    """Look up an agent specification by enum value.

    Args:
        agent_name: Member of the :class:`~cortex.enums.Agents` enum.

    Returns:
        The matching :class:`~cortex.declarative.models.AgentSpec`.

    Raises:
        AgentSpecNotFoundError: When the agent name is absent from the registry.
    """
    try:
        return AGENT_SPECS[agent_name.value]
    except KeyError as exc:
        raise AgentSpecNotFoundError("cortex", agent_name.value) from exc


# ── Eagerly populate the registry on import ──────────────────────────────────
AGENT_SPECS: SpecRegistry = load_agent_specs()
