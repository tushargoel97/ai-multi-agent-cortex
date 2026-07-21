import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from cortex import local_grounding
from cortex import workflow
from cortex.workflow import nodes as workflow_nodes
from cortex.workflow import research as workflow_research


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
    state = {"messages": [HumanMessage(content="latest result")], "spec_gap": ""}
    config = {"configurable": {"model_id": "local-row"}}

    result = asyncio.run(node(state, config))

    assert result["messages"][-1].content == "grounded local answer"
    assert calls[0][0] == intent
    assert calls[0][1] is state["messages"]
    assert calls[0][2] is config


def test_grounded_local_error_is_not_masked_as_capability_decline(monkeypatch):
    async def grounded(*_args, **_kwargs):
        raise ValueError("Requested tokens exceed context window")

    monkeypatch.setattr(workflow_nodes, "run_grounded_local", grounded)
    state = {"messages": [HumanMessage(content="latest result")], "spec_gap": ""}
    result = asyncio.run(
        workflow.researcher(state, {"configurable": {"model_id": "local-row"}})
    )

    message = result["messages"][-1]
    assert message.additional_kwargs.get("model_error") is True
    assert "tool-capable" not in str(message.content)


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
