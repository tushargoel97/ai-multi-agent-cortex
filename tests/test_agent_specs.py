from cortex.declarative import AGENT_SPECS, get_agent_spec
from cortex.enums import Agents


def test_every_builtin_agent_has_a_valid_declarative_spec():
    missing = [agent.value for agent in Agents if agent.value not in AGENT_SPECS]
    assert missing == []


def test_debugger_spec_is_available_for_engineer_delegation():
    spec = get_agent_spec(Agents.DEBUGGER)
    assert spec.name == "debugger"
    assert "root-cause" in str(spec.system_prompt).lower()
    assert "{{ user_message }}" in str(spec.user_prompt)
