import importlib.util

import pytest

from cortex.workflow.planning import ExecutionPlan
from cortex.db.models import ProviderKind
from cortex.db.services.llm_registry import (
    ResolvedModel,
    _anthropic_adaptive_thinking,
    build_client_from_resolved,
)


def test_effort_policy_module_exists():
    assert importlib.util.find_spec("cortex.workflow.effort") is not None


def plan(tier: str) -> ExecutionPlan:
    return ExecutionPlan(tier, "complex" if tier == "research" else "standard")


def test_adaptive_effort_uses_the_smallest_sufficient_level():
    from cortex.workflow.effort import effort_budget

    assert effort_budget({}, plan("direct")).level == "low"
    assert effort_budget({}, plan("grounded")).level == "medium"
    assert effort_budget({}, plan("deliberate")).level == "high"
    assert effort_budget({}, plan("research")).level == "high"


def test_explicit_effort_overrides_adaptive_selection():
    from cortex.workflow.effort import effort_budget

    config = {"configurable": {"effort": "xhigh"}}

    assert effort_budget(config, plan("direct")).level == "xhigh"


def test_effort_budgets_grow_monotonically_but_remain_bounded():
    from cortex.workflow.effort import EFFORT_LEVELS, effort_budget

    budgets = [
        effort_budget({"configurable": {"effort": level}}, plan("research"))
        for level in EFFORT_LEVELS[1:]
    ]

    assert [budget.level for budget in budgets] == list(EFFORT_LEVELS[1:])
    assert [budget.history_tokens for budget in budgets] == sorted(
        budget.history_tokens for budget in budgets
    )
    assert [budget.max_tool_calls for budget in budgets] == sorted(
        budget.max_tool_calls for budget in budgets
    )
    assert budgets[-1].history_tokens <= 8_000
    assert budgets[-1].tool_result_chars <= 20_000


def test_invalid_effort_falls_back_to_adaptive():
    from cortex.workflow.effort import effort_budget

    config = {"configurable": {"effort": "unlimited"}}

    assert effort_budget(config, plan("direct")).level == "low"


def resolved(kind: ProviderKind, model: str) -> ResolvedModel:
    return ResolvedModel(kind, model, "test-key", None, None, None)


def test_openai_client_receives_effort_and_output_budget():
    client = build_client_from_resolved(
        resolved(ProviderKind.OPENAI, "gpt-5.5"),
        effort="medium",
        max_output_tokens=1_200,
    )

    assert client.reasoning_effort == "medium"
    assert client.max_tokens == 1_200


@pytest.mark.parametrize(
    ("model", "requested", "expected"),
    [
        ("gpt-5-pro", "low", "high"),
        ("gpt-5.5", "xhigh", "xhigh"),
        ("gpt-5.5", "max", "xhigh"),
    ],
)
def test_openai_effort_uses_provider_supported_levels(model, requested, expected):
    client = build_client_from_resolved(
        resolved(ProviderKind.OPENAI, model),
        effort=requested,
        max_output_tokens=1_200,
    )

    assert client.reasoning_effort == expected


def test_anthropic_client_receives_effort_and_output_budget():
    client = build_client_from_resolved(
        resolved(ProviderKind.ANTHROPIC, "claude-opus-4-8"),
        effort="xhigh",
        max_output_tokens=4_000,
    )

    assert client.effort == "xhigh"
    assert client.max_tokens == 4_000


def test_anthropic_effort_supports_current_model_families():
    client = build_client_from_resolved(
        resolved(ProviderKind.ANTHROPIC, "claude-fable-5"),
        effort="xhigh",
        max_output_tokens=4_000,
    )

    assert client.effort == "xhigh"


def test_anthropic_effort_clamps_levels_unsupported_by_the_model():
    client = build_client_from_resolved(
        resolved(ProviderKind.ANTHROPIC, "claude-sonnet-4-6"),
        effort="xhigh",
        max_output_tokens=4_000,
    )

    assert client.effort == "high"


def test_current_anthropic_models_use_adaptive_thinking():
    assert _anthropic_adaptive_thinking("claude-opus-4-8")


def test_unsupported_anthropic_models_do_not_receive_effort():
    client = build_client_from_resolved(
        resolved(ProviderKind.ANTHROPIC, "claude-haiku-4-5"),
        effort="low",
        max_output_tokens=600,
    )

    assert client.effort is None
    assert client.max_tokens == 600


def test_unsupported_google_models_do_not_receive_thinking_level():
    client = build_client_from_resolved(
        resolved(ProviderKind.GOOGLE, "gemini-1.5-flash"),
        effort="high",
        max_output_tokens=600,
    )

    assert client.thinking_level is None
    assert client.max_output_tokens == 600
