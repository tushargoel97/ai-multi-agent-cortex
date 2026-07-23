import importlib
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cortex.enums import Agents
from cortex.workflow import routing as workflow_routing
from cortex.workflow import runtime as workflow_runtime
from cortex.workflow import synthesis as workflow_synthesis


def progress_module():
    try:
        return importlib.import_module("cortex.workflow.progress")
    except ModuleNotFoundError:
        pytest.fail("workflow progress events are not implemented")


def test_progress_event_uses_custom_stream_protocol(monkeypatch):
    progress = progress_module()
    emitted = []
    monkeypatch.setattr(progress, "get_stream_writer", lambda: emitted.append)

    progress.emit_progress("researching", tool="web_search")

    assert emitted == [
        {
            "type": "agent_progress",
            "phase": "researching",
            "tool": "web_search",
        }
    ]


def test_progress_emission_is_safe_outside_a_run(monkeypatch):
    progress = progress_module()

    def unavailable():
        raise RuntimeError("outside runnable context")

    monkeypatch.setattr(progress, "get_stream_writer", unavailable)
    assert progress.emit_progress("thinking") is None


@pytest.mark.asyncio
async def test_agent_middleware_tracks_model_and_tool_phases(monkeypatch):
    progress = progress_module()
    emitted = []
    monkeypatch.setattr(
        progress,
        "emit_progress",
        lambda phase, **details: emitted.append((phase, details)),
    )
    middleware = progress.ProgressMiddleware()

    middleware.before_model({"messages": [HumanMessage("question")]}, None)
    middleware.before_model(
        {
            "messages": [
                HumanMessage("question"),
                ToolMessage("result", tool_call_id="call-1", name="web_search"),
            ]
        },
        None,
    )

    async def run_tool(_):
        return ToolMessage("result", tool_call_id="call-2", name="web_search")

    await middleware.awrap_tool_call(
        SimpleNamespace(tool_call={"name": "web_search"}),
        run_tool,
    )

    assert emitted == [
        ("thinking", {}),
        ("refining", {}),
        ("researching", {"tool": "web_search"}),
        ("collating", {}),
    ]


@pytest.mark.asyncio
async def test_grounded_local_evidence_moves_from_research_to_collation(monkeypatch):
    from cortex import local_grounding

    emitted = []
    monkeypatch.setattr(
        local_grounding,
        "emit_progress",
        lambda phase, **details: emitted.append((phase, details)),
    )
    monkeypatch.setitem(
        local_grounding._TOOL_RUNNERS,
        "product_prices",
        lambda *_: "current offers",
    )

    evidence = await local_grounding.collect_evidence("shopping", "Find it", {})

    assert evidence.content == "current offers"
    assert emitted == [
        ("researching", {"tool": "product_prices"}),
        ("collating", {}),
    ]


def test_shared_agent_builder_installs_progress_middleware(monkeypatch):
    progress = progress_module()
    captured = {}
    monkeypatch.setattr(workflow_runtime, "get_agent_spec", lambda _: object())
    monkeypatch.setattr(workflow_runtime, "get_chat_client", lambda **_: object())
    monkeypatch.setattr(workflow_runtime, "auto_fallback_clients", lambda *_, **__: [])
    monkeypatch.setattr(workflow_runtime, "agent_static_prompt", lambda *_: "")
    monkeypatch.setattr(workflow_runtime, "agent_context", lambda *_, **__: "")
    monkeypatch.setattr(workflow_runtime, "effective_agent_tools", lambda _: [])
    monkeypatch.setattr(workflow_runtime, "subagent_tools", lambda *_: [])
    monkeypatch.setattr(
        workflow_runtime,
        "create_agent",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    workflow_runtime.build_agent(Agents.GENERALIST)

    assert any(
        isinstance(item, progress.ProgressMiddleware)
        for item in captured["middleware"]
    )


@pytest.mark.asyncio
async def test_nested_agent_progress_is_forwarded(monkeypatch):
    emitted = []

    class Agent:
        async def astream(self, *_args, **_kwargs):
            yield "custom", {"type": "agent_progress", "phase": "researching"}
            yield "values", {"messages": [AIMessage("Done")]}

    monkeypatch.setattr(
        workflow_runtime,
        "get_stream_writer",
        lambda: emitted.append,
        raising=False,
    )

    result = await workflow_runtime.invoke_agent(Agent(), {"messages": []})

    assert emitted == [{"type": "agent_progress", "phase": "researching"}]
    assert result["messages"][-1].content == "Done"


@pytest.mark.asyncio
async def test_real_agent_tool_phases_reach_parent_stream(monkeypatch):
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.tools import tool

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *_args, **_kwargs):
            return self

    @tool
    def lookup(query: str) -> str:
        """Look up a value."""
        return f"Found {query}"

    model = Model(
        responses=[
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {"query": "value"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage("Done"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[lookup],
        middleware=[progress_module().ProgressMiddleware()],
    )
    emitted = []
    monkeypatch.setattr(workflow_runtime, "get_stream_writer", lambda: emitted.append)

    result = await workflow_runtime.invoke_agent(
        agent,
        {"messages": [HumanMessage("Find the value")]},
    )

    phases = [event["phase"] for event in emitted]
    assert phases == ["thinking", "researching", "collating", "refining"]
    assert result["messages"][-1].content == "Done"


@pytest.mark.asyncio
async def test_router_emits_routing_phase(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        workflow_routing,
        "emit_progress",
        lambda phase, **_: emitted.append(phase),
        raising=False,
    )

    await workflow_routing.router(
        {"messages": [HumanMessage("Work through this carefully")]},
        {"configurable": {"mode": "thinking"}},
    )

    assert emitted == ["routing"]


@pytest.mark.asyncio
async def test_synthesizer_emits_refining_while_model_pass_runs(monkeypatch):
    emitted = []

    class Model:
        async def ainvoke(self, _):
            return AIMessage("Refined answer")

    class Spec:
        def render_system_prompt(self, **_):
            return "Refine the answer."

    monkeypatch.setattr(workflow_synthesis, "selected_local_model", lambda _: None)
    monkeypatch.setattr(workflow_synthesis, "_format_model", lambda *_: Model())
    monkeypatch.setattr(workflow_synthesis, "get_agent_spec", lambda _: Spec())
    monkeypatch.setattr(
        workflow_synthesis,
        "emit_progress",
        lambda phase, **_: emitted.append(phase),
    )
    state = {
        "messages": [
            HumanMessage("Explain the result"),
            AIMessage(
                "knowledge_query",
                additional_kwargs={"routing": {"intent": "knowledge_query"}},
            ),
            AIMessage("Draft answer"),
        ]
    }

    await workflow_synthesis.synthesize(state, {})

    assert emitted == ["refining"]
