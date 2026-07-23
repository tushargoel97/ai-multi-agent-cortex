import pytest
import json
from langchain.agents.middleware import ContextEditingMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from types import SimpleNamespace

from cortex.declarative import AGENT_SPECS
from cortex.workflow import routing
from cortex.workflow import runtime
from cortex.workflow import nodes
from cortex.workflow import research
from cortex.workflow import synthesis
from cortex.workflow import memory
from cortex.enums import Agents
from cortex.workflow.planning import ExecutionPlan, plan_execution
from cortex.workflow.context import message_window
from cortex.workflow.types import Intent
from cortex.workflow.types import RouterIntent
from cortex.tools import web


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_intent", "expected_tools"),
    [
        ("Hello", "general_chat", []),
        ("What is the current time in India?", "general_chat", ["get_current_time"]),
        ("Calculate 17 * 43", "reasoning_task", ["calculator"]),
    ],
)
async def test_high_confidence_queries_skip_the_model_router(
    monkeypatch,
    query,
    expected_intent,
    expected_tools,
):
    monkeypatch.setattr(routing, "custom_agents_for_routing", lambda: [])
    monkeypatch.setattr(routing, "local_specialists", lambda: [])
    monkeypatch.setattr(
        routing,
        "router_classifier_client",
        lambda _: pytest.fail("model router should not run"),
    )

    result = await routing.router(
        {"messages": [HumanMessage(query)]},
        {"configurable": {"model_id": "manual", "effort": "adaptive"}},
    )
    route = result["messages"][0].additional_kwargs["routing"]

    assert route["intent"] == expected_intent
    assert route["required_tools"] == expected_tools
    assert route["reasoning"] == "high-confidence local route"


def test_direct_turn_exposes_only_tools_required_by_the_plan():
    from cortex.workflow.effort import select_tools

    tools = [
        SimpleNamespace(name="get_current_time"),
        SimpleNamespace(name="save_memory"),
        SimpleNamespace(name="search_memories"),
    ]
    plan = ExecutionPlan("direct", "simple", required_tools=("get_current_time",))

    assert [tool.name for tool in select_tools(tools, plan)] == ["get_current_time"]


def test_web_grounding_keeps_search_and_targeted_page_fetch():
    from cortex.workflow.effort import select_tools

    tools = [
        SimpleNamespace(name="web_search"),
        SimpleNamespace(name="fetch_url"),
        SimpleNamespace(name="wikipedia_search"),
        SimpleNamespace(name="search_memories"),
    ]
    plan = ExecutionPlan("grounded", "standard", required_tools=("web_search",))

    assert [tool.name for tool in select_tools(tools, plan)] == [
        "web_search",
        "fetch_url",
    ]


def test_agent_builder_exposes_only_plan_relevant_tools(monkeypatch):
    captured = {}
    model_args = {}
    plan = ExecutionPlan("direct", "simple", required_tools=("get_current_time",))
    tools = [
        SimpleNamespace(name="get_current_time"),
        SimpleNamespace(name="save_memory"),
    ]
    monkeypatch.setattr(runtime, "get_agent_spec", lambda _: object())
    monkeypatch.setattr(
        runtime,
        "get_chat_client",
        lambda **kwargs: model_args.update(kwargs) or object(),
    )
    monkeypatch.setattr(runtime, "auto_fallback_clients", lambda *_, **__: [])
    monkeypatch.setattr(runtime, "agent_static_prompt", lambda *_: "")
    monkeypatch.setattr(runtime, "agent_context", lambda *_, **__: "")
    monkeypatch.setattr(runtime, "effective_agent_tools", lambda _: tools)
    monkeypatch.setattr(runtime, "subagent_tools", lambda *_: [])
    monkeypatch.setattr(
        runtime,
        "create_agent",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    runtime.build_agent(
        Agents.GENERALIST,
        config={"configurable": {"effort": "low"}},
        plan=plan,
    )

    assert [tool.name for tool in captured["tools"]] == ["get_current_time"]
    assert model_args["effort"] == "low"
    assert model_args["max_output_tokens"] == 600


@pytest.mark.asyncio
async def test_run_agent_passes_the_turn_plan_to_the_builder(monkeypatch):
    captured = {}
    state = {
        "messages": [
            HumanMessage("Hello"),
            AIMessage(
                "general_chat",
                additional_kwargs={
                    "routing": {
                        "intent": "general_chat",
                        "complexity": "simple",
                        "execution_tier": "direct",
                        "required_tools": [],
                    }
                },
            ),
        ]
    }
    monkeypatch.setattr(nodes, "memory_context", lambda *_: async_result(("", {})))
    monkeypatch.setattr(
        nodes,
        "build_agent",
        lambda *_, **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        nodes,
        "invoke_agent",
        lambda *_args, **_kwargs: async_result({"messages": [AIMessage("Hi")]}),
    )

    result = await nodes.run_agent(Agents.GENERALIST, state, {}, "general_chat")

    assert captured["plan"].tier == "direct"
    assert result["messages"][-1].additional_kwargs["effort"] == "low"


@pytest.mark.asyncio
async def test_run_agent_applies_the_selected_history_budget(monkeypatch):
    captured = {}
    state = {
        "messages": [
            HumanMessage("old question"),
            AIMessage("x" * 10_000),
            HumanMessage("new question"),
            AIMessage(
                "general_chat",
                additional_kwargs={
                    "routing": {
                        "intent": "general_chat",
                        "complexity": "simple",
                        "execution_tier": "direct",
                        "required_tools": [],
                    }
                },
            ),
        ]
    }
    monkeypatch.setattr(nodes, "memory_context", lambda *_: async_result(("", {})))
    monkeypatch.setattr(nodes, "build_agent", lambda *_, **__: object())

    async def capture(_agent, payload, **_):
        captured.update(payload)
        return {"messages": [AIMessage("answer")]}

    monkeypatch.setattr(nodes, "invoke_agent", capture)

    await nodes.run_agent(
        Agents.GENERALIST,
        state,
        {"configurable": {"effort": "low"}},
        "general_chat",
    )

    assert [message.content for message in captured["messages"]] == ["new question"]


async def async_result(value):
    return value


@pytest.mark.parametrize(
    ("query", "tool"),
    [
        ("Remember that I prefer aisle seats", "save_memory"),
        ("I use Windows 11 with an NVIDIA RTX 4090", "save_memory"),
        ("What do you remember about my travel preferences?", "search_memories"),
    ],
)
def test_memory_requests_keep_only_the_needed_memory_tool(query, tool):
    plan = plan_execution("general_chat", query, "simple")

    assert plan.required_tools == (tool,)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What is the current Bitcoin price?", ("crypto_price",)),
        ("Who developed the theory of relativity?", ("search_knowledge_base",)),
        (
            "Summarize https://example.com/report",
            ("fetch_url",),
        ),
        (
            "Show the latest model price in USD and INR",
            ("web_search", "fiat_exchange_rate"),
        ),
    ],
)
def test_knowledge_queries_choose_the_smallest_grounding_toolset(query, expected):
    plan = plan_execution("knowledge_query", query, "simple")

    assert plan.required_tools == expected


def test_message_window_drops_old_tool_noise_and_respects_the_token_budget():
    messages = [
        HumanMessage("old question"),
        AIMessage("", tool_calls=[{"name": "web_search", "args": {}, "id": "call-1"}]),
        ToolMessage("x" * 8_000, name="web_search", tool_call_id="call-1"),
        AIMessage("old answer"),
        HumanMessage("new question"),
    ]

    window = message_window(messages, token_budget=20)

    assert window[-1].content == "new question"
    assert not any(isinstance(message, ToolMessage) for message in window)
    assert not any(getattr(message, "tool_calls", None) for message in window)
    assert sum(len(str(message.content)) for message in window) <= 80


def test_agent_middleware_clears_stale_large_tool_results(monkeypatch):
    plan = ExecutionPlan("research", "complex", required_tools=("web_search",))
    middleware = runtime.agent_middleware(
        {"configurable": {"effort": "high", "unrestricted": True}},
        plan=plan,
    )

    editing = next(item for item in middleware if isinstance(item, ContextEditingMiddleware))
    clear = editing.edits[0]

    assert clear.trigger <= 7_000
    assert clear.keep == 3
    assert "fiat_exchange_rate" in clear.exclude_tools
    assert "product_prices" in clear.exclude_tools


def test_web_search_defaults_to_snippets_without_fetching_pages(monkeypatch):
    monkeypatch.setattr(
        web,
        "_provider_search",
        lambda *_: [
            {
                "title": "Result",
                "url": "https://example.com",
                "snippet": "x" * 5_000,
            }
        ],
    )
    monkeypatch.setattr(
        web,
        "_fetch_text",
        lambda *_args, **_kwargs: pytest.fail("pages should not be fetched by default"),
    )

    result = json.loads(web.web_search.invoke({"query": "query"}))

    assert len(result["results"][0]["snippet"]) <= 800
    assert "content" not in result["results"][0]


@pytest.mark.parametrize(
    ("query", "context", "expected"),
    [
        (
            "Compare the current PS5 Pro prices in Dubai and India in INR",
            "",
            False,
        ),
        ("Research this", "", True),
        ("Tell me more about it", "User: Explain the PS5 Pro launch", False),
    ],
)
def test_research_clarification_is_reserved_for_genuinely_ambiguous_requests(
    query,
    context,
    expected,
):
    assert research.needs_research_clarification(query, context) is expected


@pytest.mark.asyncio
async def test_deep_research_uses_the_selected_effort_budget(monkeypatch):
    captured = {}
    state = {
        "messages": [
            HumanMessage("old question"),
            AIMessage("x" * 10_000),
            HumanMessage(
                "Research and compare current PS5 Pro prices in Dubai and India"
            ),
        ]
    }
    monkeypatch.setattr(
        research,
        "memory_context",
        lambda *_: async_result(("", {})),
    )
    monkeypatch.setattr(
        research,
        "research_clarify_questions",
        lambda *_: pytest.fail("specific research should not call the clarifier"),
    )
    monkeypatch.setattr(
        research,
        "build_agent",
        lambda *_, **kwargs: captured.setdefault("build", kwargs) or object(),
    )

    async def invoke(_agent, payload, **_):
        captured["payload"] = payload
        return {"messages": [AIMessage("answer")]}

    monkeypatch.setattr(research, "invoke_agent", invoke)

    await research.deep_research(
        state,
        {"configurable": {"effort": "low"}},
    )

    assert captured["build"]["plan"].tier == "research"
    assert "max_tool_calls" not in captured["build"]
    assert [message.content for message in captured["payload"]["messages"]] == [
        "Research and compare current PS5 Pro prices in Dubai and India"
    ]


@pytest.mark.parametrize(
    ("name", "max_chars"),
    [
        ("generalist", 1_900),
        ("researcher", 2_700),
        ("router", 3_600),
    ],
)
def test_hot_path_system_prompts_stay_compact(name, max_chars):
    prompt = AGENT_SPECS[name].render_system_prompt(assistant_name="Cortex")

    assert len(prompt) <= max_chars


def test_compact_router_prompt_keeps_every_route():
    prompt = AGENT_SPECS["router"].render_system_prompt(assistant_name="Cortex")

    assert all(intent.value in prompt for intent in Intent)


def test_model_router_uses_a_small_low_effort_response_budget(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        runtime,
        "get_chat_client",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    runtime.router_classifier_client({"configurable": {"model_id": "auto"}})

    assert captured["effort"] == "low"
    assert captured["max_output_tokens"] == 300


def test_required_tool_retry_does_not_replay_the_draft_or_full_tool_dump():
    latest = HumanMessage("Find the current price")
    messages = [
        HumanMessage("Find the current price"),
        AIMessage("old answer"),
        latest,
        AIMessage(
            "",
            tool_calls=[
                {"name": "web_search", "args": {}, "id": "search-1"},
            ],
        ),
        ToolMessage("x" * 10_000, name="web_search", tool_call_id="search-1"),
        AIMessage("y" * 10_000),
    ]
    correction = HumanMessage("Call fiat_exchange_rate.")

    retry = nodes.required_tool_retry_messages(
        messages,
        correction,
        tool_result_chars=1_000,
    )

    assert retry[0] is latest
    assert retry[0].content == "Find the current price"
    assert retry[-1] is correction
    assert not any(message.content == "y" * 10_000 for message in retry)
    assert sum(
        len(message.content)
        for message in retry
        if isinstance(message, ToolMessage)
    ) <= 1_000


@pytest.mark.asyncio
async def test_custom_agents_receive_the_same_effort_and_context_budgets(monkeypatch):
    captured = {}
    state = {
        "messages": [
            HumanMessage("old question"),
            AIMessage("x" * 10_000),
            HumanMessage("new question"),
            AIMessage(
                "general_chat",
                additional_kwargs={
                    "routing": {
                        "intent": "general_chat",
                        "agent": "custom",
                        "complexity": "simple",
                        "execution_tier": "direct",
                        "required_tools": [],
                    }
                },
            ),
        ]
    }
    monkeypatch.setattr(
        nodes,
        "load_custom_agent",
        lambda _: {"system_prompt": "Custom"},
    )
    monkeypatch.setattr(nodes, "memory_context", lambda *_: async_result(("", {})))
    monkeypatch.setattr(
        nodes,
        "get_chat_client",
        lambda **kwargs: captured.setdefault("model", kwargs) or object(),
    )
    monkeypatch.setattr(nodes, "resolve_tool_instances", lambda *_: [])
    monkeypatch.setattr(nodes, "effective_tool_names", lambda *_: [])
    monkeypatch.setattr(nodes, "subagent_tools", lambda *_: [])
    monkeypatch.setattr(nodes, "auto_fallback_clients", lambda *_, **__: [])
    monkeypatch.setattr(nodes, "create_agent", lambda **_: object())

    async def invoke(_agent, payload, **_):
        captured["payload"] = payload
        return {"messages": [AIMessage("answer")]}

    monkeypatch.setattr(nodes, "invoke_agent", invoke)

    await nodes.custom_agent(
        state,
        {"configurable": {"effort": "low"}},
    )

    assert captured["model"]["effort"] == "low"
    assert captured["model"]["max_output_tokens"] == 600
    assert [message.content for message in captured["payload"]["messages"]] == [
        "new question"
    ]


@pytest.mark.asyncio
async def test_normal_answers_skip_the_duplicate_synthesis_model(monkeypatch):
    monkeypatch.setattr(synthesis, "selected_local_model", lambda _: None)
    monkeypatch.setattr(
        synthesis,
        "_format_model",
        lambda *_: pytest.fail("normal answers should not be reformatted"),
    )
    state = {
        "messages": [
            HumanMessage("Explain photosynthesis"),
            AIMessage(
                "general_chat",
                additional_kwargs={"routing": {"intent": "general_chat"}},
            ),
            AIMessage("Plants turn light into chemical energy."),
        ]
    }

    assert await synthesis.synthesize(state, {}) == {}


def test_table_formatter_uses_a_bounded_low_effort_model(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        synthesis,
        "get_chat_client",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    synthesis._format_model({"configurable": {"model_id": "manual"}})

    assert captured["effort"] == "low"
    assert captured["max_output_tokens"] == 2_500


@pytest.mark.asyncio
async def test_conversation_summary_uses_a_small_bounded_model_call(monkeypatch):
    captured = {}

    class Model:
        async def ainvoke(self, prompt, **_):
            captured["prompt"] = prompt
            return AIMessage("summary")

    monkeypatch.setattr(
        memory,
        "get_chat_client",
        lambda **kwargs: captured.update(model=kwargs) or Model(),
    )
    state = {
        "messages": [
            message
            for index in range(12)
            for message in (
                HumanMessage(f"question {index} " + "x" * 2_000),
                AIMessage(f"answer {index} " + "y" * 2_000),
            )
        ],
        "summary": "",
        "summary_upto": 0,
    }

    result = await memory.update_summary(state, {})

    assert result["summary"] == "summary"
    assert captured["model"]["effort"] == "low"
    assert captured["model"]["max_output_tokens"] == 350
    assert len(captured["prompt"]) <= 14_000


@pytest.mark.asyncio
async def test_delegated_subagents_inherit_the_turn_effort(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        runtime,
        "load_agent_runtime",
        lambda _: ("Subagent", [], "general_chat"),
    )
    monkeypatch.setattr(runtime, "recall_memories", lambda *_: async_result(""))
    monkeypatch.setattr(
        runtime,
        "get_chat_client",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(runtime, "create_agent", lambda **_: object())
    monkeypatch.setattr(
        runtime,
        "invoke_agent",
        lambda *_args, **_kwargs: async_result(
            {"messages": [AIMessage("delegated answer")]}
        ),
    )

    tool = runtime.subagent_tool(
        "helper",
        "",
        {"configurable": {"effort": "low"}},
    )
    result = await tool.ainvoke({"task": "Answer this"})

    assert result == "delegated answer"
    assert captured["effort"] == "low"
    assert captured["max_output_tokens"] == 600


def test_tool_arguments_are_clamped_to_the_effort_budget():
    from cortex.workflow.effort import budget_tool_call, effort_budget

    budget = effort_budget(
        {"configurable": {"effort": "low"}},
        ExecutionPlan("research", "complex"),
    )

    search = budget_tool_call(
        {
            "name": "web_search",
            "args": {"query": "query", "max_results": 20, "fetch_pages": True},
        },
        budget,
    )
    fetch = budget_tool_call(
        {
            "name": "fetch_url",
            "args": {"url": "https://example.com", "max_chars": 50_000},
        },
        budget,
    )

    assert search["args"]["max_results"] == 3
    assert search["args"]["fetch_pages"] is False
    assert fetch["args"]["max_chars"] == 4_500


def test_explicit_high_effort_is_not_overridden_by_the_instant_directive():
    from cortex.workflow.context import agent_context

    low = agent_context(
        {"configurable": {"mode": "general", "effort": "low"}},
        complexity="standard",
    )
    high = agent_context(
        {"configurable": {"mode": "general", "effort": "high"}},
        complexity="standard",
    )

    assert "at most one tool call" in low
    assert "at most one tool call" not in high


def test_provider_usage_can_be_aggregated_across_internal_model_calls():
    from cortex.workflow.context import aggregate_usage

    usage = aggregate_usage(
        [
            AIMessage(
                "route",
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "total_tokens": 110,
                    "input_token_details": {"cache_read": 20},
                },
            ),
            AIMessage(
                "answer",
                usage_metadata={
                    "input_tokens": 200,
                    "output_tokens": 30,
                    "total_tokens": 230,
                    "input_token_details": {"cache_read": 50},
                },
            ),
        ]
    )

    assert usage == {
        "input_tokens": 300,
        "output_tokens": 40,
        "total_tokens": 340,
        "input_token_details": {"cache_read": 70},
    }


def test_synthesis_replacement_includes_formatter_usage():
    source = AIMessage(
        "draft",
        usage_metadata={
            "input_tokens": 200,
            "output_tokens": 30,
            "total_tokens": 230,
        },
    )
    formatter = AIMessage(
        "formatted",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
        },
    )

    result = synthesis._replacement(source, "table", formatter)

    assert result["messages"][0].usage_metadata == {
        "input_tokens": 300,
        "output_tokens": 50,
        "total_tokens": 350,
    }


@pytest.mark.asyncio
async def test_model_router_usage_is_preserved_on_the_route_marker(monkeypatch):
    class Agent:
        async def ainvoke(self, _):
            return {
                "messages": [
                    AIMessage(
                        "route",
                        usage_metadata={
                            "input_tokens": 100,
                            "output_tokens": 10,
                            "total_tokens": 110,
                        },
                    )
                ],
                "structured_response": RouterIntent(
                    intent=Intent.KNOWLEDGE_QUERY,
                    reasoning="Needs current sources",
                ),
            }

    monkeypatch.setattr(routing, "create_agent", lambda **_: Agent())
    monkeypatch.setattr(routing, "router_classifier_client", lambda _: object())
    monkeypatch.setattr(routing, "auto_fallback_clients", lambda *_, **__: [])
    monkeypatch.setattr(routing, "agent_static_prompt", lambda *_: "")
    monkeypatch.setattr(routing, "custom_agents_for_routing", lambda: [])
    monkeypatch.setattr(routing, "local_specialists", lambda: [])

    result = await routing.router(
        {"messages": [HumanMessage("What changed in the latest release?")]},
        {"configurable": {"model_id": "manual"}},
    )

    assert result["messages"][0].usage_metadata["total_tokens"] == 110
