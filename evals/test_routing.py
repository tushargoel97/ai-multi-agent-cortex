"""Routing accuracy tests for the Cortex multi-agent workflow.

Verifies the router classifies user messages into the correct intent
(general_chat, knowledge_query, reasoning_task, prompt_caching).

Run:
    pytest evals/test_routing.py -v
"""

import pytest


def _routed_intent(result: dict) -> str | None:
    for msg in result["all_messages"]:
        if msg.get("type") == "ai":
            routing = msg.get("additional_kwargs", {}).get("routing", {})
            if routing:
                return routing.get("intent")
    return None


@pytest.mark.parametrize(
    "case_id",
    ["ROUTE-001", "ROUTE-002", "ROUTE-003", "ROUTE-004"],
)
def test_router_classifies_intent(agent_runner, golden_dataset, case_id):
    """Each input must be routed to the expected capability."""
    tc = golden_dataset[case_id]
    result = agent_runner(tc["input"])
    intent = _routed_intent(result)
    assert intent == tc["expected_intent"], (
        f"{case_id}: expected '{tc['expected_intent']}' got '{intent}' "
        f"for input: {tc['input']!r}"
    )
