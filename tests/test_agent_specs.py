from cortex.declarative import AGENT_SPECS, get_agent_spec
from cortex.db.services import auto_mode
from cortex.enums import Agents


def test_every_builtin_agent_has_a_valid_declarative_spec():
    missing = [agent.value for agent in Agents if agent.value not in AGENT_SPECS]
    assert missing == []


def test_debugger_spec_is_available_for_engineer_delegation():
    spec = get_agent_spec(Agents.DEBUGGER)
    assert spec.name == "debugger"
    assert "root-cause" in str(spec.system_prompt).lower()
    assert "{{ user_message }}" in str(spec.user_prompt)


def test_specialist_prompt_is_domain_neutral():
    prompt = str(get_agent_spec(Agents.SPECIALIST).system_prompt).lower()
    assert "hardware" not in prompt
    assert "capability description" in prompt


def test_auto_mode_does_not_use_a_global_finetuned_model_sentinel():
    candidates = [
        model
        for profile in auto_mode._yaml_profiles().values()
        for models in profile.values()
        for model in models
    ]
    assert "finetuned" not in candidates
