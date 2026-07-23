import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cortex import local_grounding
from cortex import workflow
from cortex.workflow import nodes as workflow_nodes
from cortex.workflow import research as workflow_research
from cortex.workflow.context import agent_context


def test_registered_tool_uses_runnable_interface():
    calls = []

    class Tool:
        def __call__(self, **_kwargs):
            raise AssertionError("registered tools are not plain callables")

        async def ainvoke(self, arguments):
            calls.append(arguments)
            return "evidence"

    result = asyncio.run(
        local_grounding._invoke_tool(Tool(), {"query": "latest result"})
    )

    assert result == "evidence"
    assert calls == [{"query": "latest result"}]


@pytest.mark.parametrize(
    ("intent", "expected_tool"),
    [
        ("knowledge_query", "web_search"),
        ("shopping", "product_prices"),
        ("booking", "find_bookings"),
    ],
)
def test_collect_evidence_uses_one_deterministic_tool(intent, expected_tool, monkeypatch):
    calls = []

    monkeypatch.setattr(
        local_grounding,
        "_TOOL_RUNNERS",
        {
            expected_tool: lambda question, region: calls.append((question, region)) or "evidence"
        },
    )

    evidence = asyncio.run(
        local_grounding.collect_evidence(
            intent,
            "latest result",
            {"configurable": {"locale": "en-IN", "timezone": "Asia/Kolkata"}},
        )
    )

    assert evidence.tool_name == expected_tool
    assert evidence.content == "evidence"
    assert calls == [("latest result", "IN")]


def test_grounded_messages_fit_budget_and_remove_router_markers():
    messages = [HumanMessage(content="old " * 3000), AIMessage(content="old answer " * 1000)]
    messages += [
        HumanMessage(content="who won the race?"),
        AIMessage(
            content="knowledge_query",
            additional_kwargs={"routing": {"intent": "knowledge_query"}},
        ),
    ]

    prompt = local_grounding.local_messages(
        messages,
        "Today is July 21, 2026.",
        evidence=local_grounding.Evidence("web_search", "result " * 3000),
        max_chars=10_000,
    )

    assert sum(len(str(message.content)) for message in prompt) <= 10_000
    assert "knowledge_query" not in [message.content for message in prompt]
    assert "web_search" in str(prompt[-1].content)
    assert str(prompt[-1].content).endswith("Question:\nwho won the race?")


def test_run_grounded_local_uses_evidence_without_binding_tools(monkeypatch):
    captured = []
    options = []

    class Client:
        def bind(self, **kwargs):
            options.append(kwargs)
            return self

        async def ainvoke(self, messages, config=None):
            captured.extend(messages)
            return AIMessage(content="grounded answer")

    monkeypatch.setattr(
        local_grounding,
        "resolve_with_session",
        lambda _: SimpleNamespace(kind=SimpleNamespace(value="local"), model_id="local-model"),
    )
    monkeypatch.setattr(local_grounding, "build_client_from_resolved", lambda _: Client())

    async def evidence(*_):
        return local_grounding.Evidence("web_search", "verified result")

    monkeypatch.setattr(local_grounding, "collect_evidence", evidence)

    result = asyncio.run(
        local_grounding.run_grounded_local(
            "knowledge_query",
            [HumanMessage(content="who won?")],
            {"configurable": {"model_id": "row-id"}},
            request_context="Today is July 21, 2026.",
        )
    )

    assert result.content == "grounded answer"
    assert options == [{"max_tokens": 768}]
    assert any("verified result" in str(message.content) for message in captured)
    assert all(not getattr(message, "tool_calls", None) for message in captured)


def test_explicit_low_effort_reduces_local_model_context_and_output(monkeypatch):
    captured = []
    options = []

    class Client:
        def bind(self, **kwargs):
            options.append(kwargs)
            return self

        async def ainvoke(self, messages, config=None):
            captured.extend(messages)
            return AIMessage(content="answer")

    monkeypatch.setattr(
        local_grounding,
        "selected_local_model",
        lambda _: SimpleNamespace(model_id="local"),
    )
    monkeypatch.setattr(local_grounding, "build_client_from_resolved", lambda _: Client())

    asyncio.run(
        local_grounding.run_local_answer(
            "general_chat",
            [HumanMessage(content="Answer this")],
            {"configurable": {"model_id": "row-id", "effort": "low"}},
            request_context="context",
        )
    )

    assert options == [{"max_tokens": 600}]
    assert sum(len(str(message.content)) for message in captured) <= 6_000


def test_grounded_followup_search_uses_previous_user_context(monkeypatch):
    queries = []
    captured = []

    async def evidence(_intent, question, _config):
        queries.append(question)
        return local_grounding.Evidence("web_search", "verified result")

    class Client:
        def bind(self, **_kwargs):
            return self

        async def ainvoke(self, messages, config=None):
            captured.extend(messages)
            return AIMessage(content="answer")

    monkeypatch.setattr(
        local_grounding,
        "resolve_with_session",
        lambda _: SimpleNamespace(kind=SimpleNamespace(value="local"), model_id="local"),
    )
    monkeypatch.setattr(local_grounding, "build_client_from_resolved", lambda _: Client())
    monkeypatch.setattr(local_grounding, "collect_evidence", evidence)

    asyncio.run(
        local_grounding.run_grounded_local(
            "knowledge_query",
            [
                HumanMessage(content="Who won the 2025 Belgian Grand Prix?"),
                AIMessage(
                    content="Oscar Piastri won for McLaren.",
                    response_metadata={"grounded_by": "web_search"},
                ),
                HumanMessage(content="How many wins does he have?"),
            ],
            {"configurable": {"model_id": "row-id"}},
            request_context="Today is July 21, 2026.",
        )
    )

    assert "Oscar Piastri" in queries[0]
    assert "How many wins does he have?" in queries[0]
    assert "Oscar Piastri won for McLaren" in str(captured[-1].content)


def test_general_followup_reuses_previous_grounded_answer(monkeypatch):
    captured = []

    class Client:
        def bind(self, **_kwargs):
            return self

        async def ainvoke(self, messages, config=None):
            captured.extend(messages)
            return AIMessage(content="McLaren")

    monkeypatch.setattr(
        local_grounding,
        "resolve_with_session",
        lambda _: SimpleNamespace(kind=SimpleNamespace(value="local"), model_id="local"),
    )
    monkeypatch.setattr(local_grounding, "build_client_from_resolved", lambda _: Client())
    previous = AIMessage(
        content="Oscar Piastri won for McLaren.",
        response_metadata={"grounded_by": "web_search"},
    )

    result = asyncio.run(
        local_grounding.run_local_answer(
            "general_chat",
            [
                HumanMessage(content="Who won the 2025 Belgian Grand Prix?"),
                previous,
                HumanMessage(content="Which team was he driving for?"),
            ],
            {"configurable": {"model_id": "row-id"}},
            request_context="Today is July 21, 2026.",
        )
    )

    assert "Evidence from web_search" in str(captured[-1].content)
    assert "McLaren" in str(captured[-1].content)
    assert result.response_metadata["grounded_by"] == "web_search"


def test_explicit_local_model_receives_its_capability_description(monkeypatch):
    captured = []

    class Client:
        def bind(self, **_kwargs):
            return self

        async def ainvoke(self, messages, config=None):
            captured.extend(messages)
            return AIMessage(content="reviewed")

    resolved = SimpleNamespace(model_id="finetuned-contract-reviewer")
    monkeypatch.setattr(local_grounding, "selected_local_model", lambda _: resolved)
    monkeypatch.setattr(
        local_grounding,
        "local_specialist_profile",
        lambda _: SimpleNamespace(
            display_name="Contract Reviewer",
            description="Reviews commercial contracts, obligations, and termination clauses.",
        ),
    )
    monkeypatch.setattr(local_grounding, "build_client_from_resolved", lambda _: Client())

    asyncio.run(
        local_grounding.run_local_answer(
            "general_chat",
            [HumanMessage(content="Review this termination clause")],
            {"configurable": {"model_id": "model-row"}},
            request_context="Today is July 21, 2026.",
        )
    )

    assert "Reviews commercial contracts" in str(captured[0].content)


@pytest.mark.parametrize(
    ("node", "intent", "module"),
    [
        (workflow.researcher, "knowledge_query", workflow_research),
        (workflow.shopping, "shopping", workflow_nodes),
        (workflow.booking, "booking", workflow_nodes),
    ],
)
def test_tool_intents_use_grounded_local_path(node, intent, module, monkeypatch):
    calls = []

    async def grounded(actual_intent, messages, config, *, request_context):
        calls.append((actual_intent, messages, config, request_context))
        return AIMessage(content="grounded local answer")

    async def cloud_agent(*_args, **_kwargs):
        raise AssertionError("cloud agent should not run for an explicit local model")

    monkeypatch.setattr(workflow_nodes, "run_grounded_local", grounded)
    monkeypatch.setattr(module, "run_agent", cloud_agent)
    state = {"messages": [HumanMessage(content="latest result")]}
    config = {"configurable": {"model_id": "local-row"}}

    result = asyncio.run(node(state, config))

    assert result["messages"][-1].content == "grounded local answer"
    assert result["messages"][-1].additional_kwargs["execution_tier"] == "grounded"
    assert calls[0][0] == intent
    assert calls[0][1] is state["messages"]
    assert calls[0][2] is config


def test_grounded_local_error_is_not_masked_as_capability_decline(monkeypatch):
    async def grounded(*_args, **_kwargs):
        raise ValueError("Requested tokens exceed context window")

    monkeypatch.setattr(workflow_nodes, "run_grounded_local", grounded)
    state = {"messages": [HumanMessage(content="latest result")]}
    result = asyncio.run(
        workflow.researcher(state, {"configurable": {"model_id": "local-row"}})
    )

    message = result["messages"][-1]
    assert message.additional_kwargs.get("model_error") is True
    assert "tool-capable" not in str(message.content)


def test_complex_shopping_uses_bounded_research_execution(monkeypatch):
    captured = {}

    class Agent:
        async def ainvoke(self, payload, config=None):
            captured["invoke_config"] = config
            return {
                "messages": [
                    *payload["messages"],
                    ToolMessage(content="current", name="product_prices", tool_call_id="1"),
                    ToolMessage(content="history", name="web_search", tool_call_id="2"),
                    ToolMessage(content="converted", name="fiat_exchange_rate", tool_call_id="3"),
                    AIMessage(content="answer"),
                ]
            }

        async def astream(self, payload, config=None, **_):
            yield "values", await self.ainvoke(payload, config)

    async def memory(*_args, **_kwargs):
        return "", {}

    def build(*_args, **kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr(workflow_nodes, "memory_context", memory)
    monkeypatch.setattr(workflow_nodes, "build_agent", build)
    state = {
        "messages": [
            HumanMessage(content="Compare prices now and in 2025 and convert them"),
            AIMessage(
                content="shopping",
                additional_kwargs={
                    "routing": {
                        "intent": "shopping",
                        "complexity": "complex",
                        "execution_tier": "research",
                        "evidence_dimensions": [
                            "current",
                            "historical",
                            "comparison",
                            "conversion",
                        ],
                        "required_tools": [
                            "product_prices",
                            "web_search",
                            "fiat_exchange_rate",
                        ],
                    }
                },
            ),
        ]
    }

    asyncio.run(
        workflow_nodes.run_agent(
            workflow_nodes.Agents.SHOPPING,
            state,
            {},
            "shopping",
        )
    )

    assert captured["complexity"] == "complex"
    assert captured["plan"].tier == "research"
    assert "max_tool_calls" not in captured
    assert captured["invoke_config"]["recursion_limit"] == 60
    assert "independent evidence tasks" in captured["extra_system"]


def test_complex_execution_is_not_limited_by_instant_mode():
    context = agent_context(
        {"configurable": {"mode": "general"}},
        complexity="complex",
    )

    assert "SPEED:" not in context


def test_custom_agent_uses_shared_execution_plan(monkeypatch):
    captured = {}

    class Client:
        pass

    class Agent:
        async def ainvoke(self, payload, config=None):
            captured["invoke_config"] = config
            return {"messages": [*payload["messages"], AIMessage(content="answer")]}

        async def astream(self, payload, config=None, **_):
            yield "values", await self.ainvoke(payload, config)

    async def memory(*_args, **_kwargs):
        return "", {}

    def client(**kwargs):
        captured["client"] = kwargs
        return Client()

    def create(**kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        return Agent()

    monkeypatch.setattr(
        workflow_nodes,
        "load_custom_agent",
        lambda _: {
            "system_prompt": "Custom agent",
        },
    )
    monkeypatch.setattr(workflow_nodes, "memory_context", memory)
    monkeypatch.setattr(workflow_nodes, "get_chat_client", client)
    monkeypatch.setattr(workflow_nodes, "auto_fallback_clients", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(workflow_nodes, "effective_tool_names", lambda *_: [])
    monkeypatch.setattr(workflow_nodes, "resolve_tool_instances", lambda _: [])
    monkeypatch.setattr(workflow_nodes, "subagent_tools", lambda *_: [])
    monkeypatch.setattr(workflow_nodes, "create_agent", create)
    state = {
        "messages": [
            HumanMessage(content="Perform a detailed architecture review"),
            AIMessage(
                content="coding_task",
                additional_kwargs={
                    "routing": {
                        "intent": "coding_task",
                        "agent": "Architecture Reviewer",
                        "complexity": "complex",
                        "execution_tier": "deliberate",
                        "evidence_dimensions": [],
                        "required_tools": [],
                    }
                },
            ),
        ]
    }

    result = asyncio.run(workflow_nodes.custom_agent(state, {}))

    assert captured["client"]["auto_intent"] == "coding_task"
    assert captured["client"]["complexity"] == "complex"
    assert captured["invoke_config"]["recursion_limit"] == 40
    assert result["messages"][-1].additional_kwargs["execution_tier"] == "deliberate"


def test_run_agent_retries_when_required_tools_are_missing(monkeypatch):
    class Agent:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, payload, config=None):
            self.calls += 1
            if self.calls == 1:
                return {
                    "messages": [
                        *payload["messages"],
                        ToolMessage(
                            content='{"offers": [{"price": "AED 1"}]}',
                            name="product_prices",
                            tool_call_id="prices",
                        ),
                        AIMessage(content="unverified conversion"),
                    ]
                }
            return {
                "messages": [
                    *payload["messages"],
                    ToolMessage(
                        content='{"converted": 96029.74}',
                        name="fiat_exchange_rate",
                        tool_call_id="fx",
                    ),
                    AIMessage(content="verified conversion"),
                ]
            }

        async def astream(self, payload, config=None, **_):
            yield "values", await self.ainvoke(payload, config)

    agent = Agent()

    async def memory(*_args, **_kwargs):
        return "", {}

    monkeypatch.setattr(workflow_nodes, "memory_context", memory)
    monkeypatch.setattr(workflow_nodes, "build_agent", lambda *_args, **_kwargs: agent)

    result = asyncio.run(
        workflow_nodes.run_agent(
            workflow_nodes.Agents.SHOPPING,
            {
                "messages": [
                    HumanMessage(content="Convert this price to INR"),
                    AIMessage(
                        content="shopping",
                        additional_kwargs={
                            "routing": {
                                "intent": "shopping",
                                "complexity": "standard",
                                "execution_tier": "grounded",
                                "evidence_dimensions": ["conversion"],
                                "required_tools": [
                                    "product_prices",
                                    "fiat_exchange_rate",
                                ],
                            }
                        },
                    ),
                ]
            },
            {},
            "shopping",
        )
    )

    assert agent.calls == 2
    assert result["messages"][-1].content == "verified conversion"
    assert "unverified conversion" not in [
        message.content
        for message in result["messages"]
        if isinstance(message, AIMessage)
    ]
    assert {
        message.name
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    } == {"product_prices", "fiat_exchange_rate"}
    assert all("Required tools" not in str(message.content) for message in result["messages"])
    assert result["messages"][-1].additional_kwargs["tool_execution"] == {
        "status": "complete",
        "required": ["product_prices", "fiat_exchange_rate"],
        "used": ["product_prices", "fiat_exchange_rate"],
    }


@pytest.mark.parametrize(
    ("node", "intent"),
    [
        (workflow.generalist, "general_chat"),
        (workflow.reasoner, "reasoning_task"),
        (workflow.coder, "coding_task"),
        (workflow.prompt_cacher, "prompt_caching"),
    ],
)
def test_other_local_intents_use_compact_direct_path(node, intent, monkeypatch):
    calls = []

    async def local_answer(actual_intent, messages, config, *, request_context):
        calls.append((actual_intent, request_context))
        return AIMessage(content="local answer")

    async def cloud_agent(*_args, **_kwargs):
        raise AssertionError("agent tool schema should not be sent to a local model")

    monkeypatch.setattr(workflow_nodes, "run_local_answer", local_answer)
    monkeypatch.setattr(workflow_nodes, "run_agent", cloud_agent)
    state = {"messages": [HumanMessage(content="help")], "summary": "prior context"}

    result = asyncio.run(node(state, {"configurable": {"model_id": "local-row"}}))

    assert result["messages"][-1].content == "local answer"
    assert calls[0][0] == intent
    assert "prior context" in calls[0][1]
