import asyncio
import importlib
import inspect
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from cortex import local_grounding
from cortex.db.services import knowledge_gaps
from cortex.db.services import auto_mode
from cortex.workflow import routing as workflow_routing
from cortex.workflow.types import INTENT_TO_NODE, Intent, RouterIntent

workflow_specialist = importlib.import_module("cortex.workflow.specialist")


def test_explicit_finetuned_model_still_uses_capability_router():
    node = workflow_routing.route_from_start(
        {"messages": [HumanMessage(content="Review this contract clause")]},
        {"configurable": {"model_id": "model-row-id"}},
    )

    assert node == "router"


def test_start_router_keeps_langgraph_injection_parameter_names():
    assert list(inspect.signature(workflow_routing.route_from_start).parameters) == [
        "state",
        "config",
    ]


def test_named_trained_entity_does_not_override_shopping_intent(monkeypatch):
    class Agent:
        async def ainvoke(self, _input):
            return {
                "structured_response": RouterIntent(
                    intent=Intent.SHOPPING,
                    reasoning="Current retailer prices requested",
                )
            }

    monkeypatch.setattr(workflow_routing, "create_agent", lambda **_: Agent())
    monkeypatch.setattr(workflow_routing, "router_classifier_client", lambda _: object())
    monkeypatch.setattr(workflow_routing, "auto_fallback_clients", lambda _: [])
    monkeypatch.setattr(workflow_routing, "agent_static_prompt", lambda *_: "")
    monkeypatch.setattr(workflow_routing, "custom_agents_for_routing", lambda: [])
    monkeypatch.setattr(workflow_routing, "local_specialists", lambda: [])
    result = asyncio.run(
        workflow_routing.router(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Compare Model A and Model B prices in Dubai and convert "
                            "them to Indian rupees"
                        )
                    )
                ]
            },
            {"configurable": {"model_id": "manual-model"}},
        )
    )

    route = result["messages"][0].additional_kwargs["routing"]
    assert route["intent"] == Intent.SHOPPING.value
    assert route["reasoning"] == "Current retailer prices requested"


def test_compound_shopping_request_is_promoted_to_research(monkeypatch):
    resolved = []

    class Agent:
        async def ainvoke(self, _input):
            return {
                "structured_response": RouterIntent(
                    intent=Intent.SHOPPING,
                    reasoning="Retail prices requested",
                )
            }

    monkeypatch.setattr(workflow_routing, "create_agent", lambda **_: Agent())
    monkeypatch.setattr(workflow_routing, "router_classifier_client", lambda _: object())
    monkeypatch.setattr(workflow_routing, "auto_fallback_clients", lambda _: [])
    monkeypatch.setattr(workflow_routing, "agent_static_prompt", lambda *_: "")
    monkeypatch.setattr(workflow_routing, "custom_agents_for_routing", lambda: [])
    monkeypatch.setattr(workflow_routing, "local_specialists", lambda: [])
    monkeypatch.setattr(auto_mode, "is_auto", lambda _: True)
    monkeypatch.setattr(
        auto_mode,
        "resolve_auto_model",
        lambda intent, profile=None: resolved.append((intent, profile))
        or SimpleNamespace(model_id="quality-model"),
    )

    result = asyncio.run(
        workflow_routing.router(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Compare Alpha Pro and Beta Slim prices in Dubai for "
                            "July 2026 and July 2025 and convert them to INR"
                        )
                    )
                ]
            },
            {"configurable": {"model_id": "auto"}},
        )
    )

    route = result["messages"][0].additional_kwargs["routing"]
    assert route["complexity"] == "complex"
    assert route["execution_tier"] == "research"
    assert route["evidence_dimensions"] == [
        "current",
        "historical",
        "comparison",
        "conversion",
    ]
    assert route["required_tools"] == [
        "product_prices",
        "web_search",
        "fiat_exchange_rate",
    ]
    assert resolved[-1] == (Intent.SHOPPING.value, "quality")


def test_single_current_price_stays_on_grounded_tier(monkeypatch):
    class Agent:
        async def ainvoke(self, _input):
            return {
                "structured_response": RouterIntent(
                    intent=Intent.SHOPPING,
                    reasoning="One current price requested",
                )
            }

    monkeypatch.setattr(workflow_routing, "create_agent", lambda **_: Agent())
    monkeypatch.setattr(workflow_routing, "router_classifier_client", lambda _: object())
    monkeypatch.setattr(workflow_routing, "auto_fallback_clients", lambda _: [])
    monkeypatch.setattr(workflow_routing, "agent_static_prompt", lambda *_: "")
    monkeypatch.setattr(workflow_routing, "custom_agents_for_routing", lambda: [])
    monkeypatch.setattr(workflow_routing, "local_specialists", lambda: [])

    result = asyncio.run(
        workflow_routing.router(
            {"messages": [HumanMessage(content="Current price of Alpha Pro in India")]},
            {"configurable": {"model_id": "manual-model"}},
        )
    )

    route = result["messages"][0].additional_kwargs["routing"]
    assert route["complexity"] == "standard"
    assert route["execution_tier"] == "grounded"
    assert route["evidence_dimensions"] == ["current"]
    assert route["required_tools"] == ["product_prices"]


@pytest.mark.parametrize(
    ("intent", "complexity", "question", "tier", "required_tools"),
    [
        (
            Intent.KNOWLEDGE_QUERY,
            "standard",
            "Compare the 2024 and 2025 results using current sources",
            "research",
            ["web_search"],
        ),
        (
            Intent.KNOWLEDGE_QUERY,
            "simple",
            "Who won the 2025 Belgian Grand Prix?",
            "grounded",
            ["web_search"],
        ),
        (
            Intent.BOOKING,
            "standard",
            "Compare flights on two dates under my budget",
            "research",
            ["find_bookings", "web_search"],
        ),
        (
            Intent.CODING_TASK,
            "complex",
            "Refactor this service architecture and verify the edge cases",
            "deliberate",
            [],
        ),
        (
            Intent.REASONING_TASK,
            "complex",
            "Work through this multi-step logic problem",
            "deliberate",
            [],
        ),
        (
            Intent.REASONING_TASK,
            "simple",
            "Calculate 18 * 27",
            "direct",
            ["calculator"],
        ),
        (
            Intent.CODING_TASK,
            "standard",
            "Check the latest FastAPI documentation for this API",
            "grounded",
            ["web_search"],
        ),
        (
            Intent.GENERAL_CHAT,
            "simple",
            "What time is it now?",
            "direct",
            ["get_current_time"],
        ),
        (
            Intent.GENERAL_CHAT,
            "simple",
            "Write a short birthday message",
            "direct",
            [],
        ),
    ],
)
def test_execution_policy_applies_across_capabilities(
    monkeypatch, intent, complexity, question, tier, required_tools
):
    class Agent:
        async def ainvoke(self, _input):
            return {
                "structured_response": RouterIntent(
                    intent=intent,
                    reasoning="Test route",
                    complexity=complexity,
                )
            }

    monkeypatch.setattr(workflow_routing, "create_agent", lambda **_: Agent())
    monkeypatch.setattr(workflow_routing, "router_classifier_client", lambda _: object())
    monkeypatch.setattr(workflow_routing, "auto_fallback_clients", lambda _: [])
    monkeypatch.setattr(workflow_routing, "agent_static_prompt", lambda *_: "")
    monkeypatch.setattr(workflow_routing, "custom_agents_for_routing", lambda: [])
    monkeypatch.setattr(workflow_routing, "local_specialists", lambda: [])

    result = asyncio.run(
        workflow_routing.router(
            {"messages": [HumanMessage(content=question)]},
            {"configurable": {"model_id": "manual-model"}},
        )
    )

    route = result["messages"][0].additional_kwargs["routing"]
    assert route["execution_tier"] == tier
    assert route["required_tools"] == required_tools


def test_auto_routes_described_local_model_without_domain_rules(monkeypatch):
    captured = {}

    class Agent:
        async def ainvoke(self, _input):
            return {
                "structured_response": RouterIntent(
                    intent=Intent.LOCAL_SPECIALIST,
                    reasoning="Contract review matches the model description",
                    local_model="finetuned-contract-reviewer",
                )
            }

    def create(**kwargs):
        captured["prompt"] = kwargs["system_prompt"]
        return Agent()

    monkeypatch.setattr(workflow_routing, "create_agent", create)
    monkeypatch.setattr(workflow_routing, "router_classifier_client", lambda _: object())
    monkeypatch.setattr(workflow_routing, "auto_fallback_clients", lambda _: [])
    monkeypatch.setattr(workflow_routing, "agent_static_prompt", lambda *_: "router")
    monkeypatch.setattr(workflow_routing, "custom_agents_for_routing", lambda: [])
    monkeypatch.setattr(
        workflow_routing,
        "local_specialists",
        lambda: [
            SimpleNamespace(
                model_id="finetuned-contract-reviewer",
                description="Reviews commercial contracts and termination clauses.",
            )
        ],
    )

    result = asyncio.run(
        workflow_routing.router(
            {"messages": [HumanMessage(content="Review this termination clause")]},
            {"configurable": {"model_id": "auto"}},
        )
    )

    route = result["messages"][0].additional_kwargs["routing"]
    assert route["intent"] == Intent.LOCAL_SPECIALIST.value
    assert route["local_model"] == "finetuned-contract-reviewer"
    assert "Reviews commercial contracts" in captured["prompt"]


def test_generic_product_specs_fall_back_to_research():
    assert INTENT_TO_NODE[Intent.PRODUCT_SPECS] == "researcher"
    assert INTENT_TO_NODE[Intent.LOCAL_SPECIALIST] == "specialist"


def test_heuristic_fallback_has_no_domain_specific_product_names():
    assert workflow_routing.heuristic_intent(
        [HumanMessage(content="Compare the specifications of Model A and Model B")]
    ) == Intent.PRODUCT_SPECS
    assert workflow_routing.heuristic_intent(
        [HumanMessage(content="Tell me about Snapdragon 8 Gen 1")]
    ) == Intent.GENERAL_CHAT


def test_local_specialist_receives_recent_conversation(monkeypatch):
    captured = []

    class Client:
        def bind(self, **_kwargs):
            return self

        async def ainvoke(self, messages, config=None):
            captured.extend(messages)
            return AIMessage(content="answer")

    monkeypatch.setattr(
        local_grounding,
        "local_specialist_profile",
        lambda _: SimpleNamespace(
            display_name="Local model", description="General chat"
        ),
    )
    monkeypatch.setattr(
        local_grounding,
        "resolve_local_specialist",
        lambda _: SimpleNamespace(model_id="local-model"),
    )
    monkeypatch.setattr(
        local_grounding,
        "build_client_from_resolved",
        lambda _: Client(),
    )

    state = {
        "messages": [
            HumanMessage(content="My project uses PostgreSQL."),
            AIMessage(content="Understood."),
            HumanMessage(content="Which database did I mention?"),
            AIMessage(
                content="local_specialist",
                additional_kwargs={
                    "routing": {
                        "intent": "local_specialist",
                        "local_model": "local-model",
                        "complexity": "complex",
                        "execution_tier": "deliberate",
                    }
                },
            ),
        ]
    }

    result = asyncio.run(
        workflow_specialist.run_local_specialist(state, {}, "local-model")
    )

    assert [message.content for message in captured[1:]] == [
        "My project uses PostgreSQL.",
        "Understood.",
        "Which database did I mention?",
    ]
    assert result["messages"][0].content == "answer"
    assert result["messages"][0].additional_kwargs["execution_tier"] == "deliberate"


def test_specialist_without_a_routed_model_is_domain_neutral():
    result = asyncio.run(
        workflow_specialist.specialist(
            {
                "messages": [
                    HumanMessage(content="Help with this topic"),
                    AIMessage(
                        content="local_specialist",
                        additional_kwargs={
                            "routing": {"intent": "local_specialist", "local_model": None}
                        },
                    ),
                ]
            },
            {},
        )
    )

    message = result["messages"][0].content
    assert "No local specialist model was selected" in message
    assert "hardware" not in message.lower()


def test_local_specialist_logs_domain_neutral_refusals(monkeypatch):
    logged = []

    async def answer(*_args, **_kwargs):
        return AIMessage(content="That topic is outside my trained capabilities.")

    monkeypatch.setattr(workflow_specialist, "answer_with_local_specialist", answer)
    monkeypatch.setattr(
        knowledge_gaps,
        "log_gap",
        lambda question, response, reason: logged.append((question, response, reason)),
    )

    asyncio.run(
        workflow_specialist.run_local_specialist(
            {"messages": [HumanMessage(content="Explain this unfamiliar topic")]},
            {},
            "finetuned-specialist",
        )
    )

    assert logged == [
        (
            "Explain this unfamiliar topic",
            "That topic is outside my trained capabilities.",
            "refusal",
        )
    ]
