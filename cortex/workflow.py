"""LangGraph workflow — multi-agent assistant with capability-based routing."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from cortex.config import get_settings
from cortex.declarative import get_agent_spec
from cortex.enums import Agents
from cortex.model_client import get_chat_client
from cortex.observability import setup_tracing

# Initialise tracing before any LangChain objects are constructed.
setup_tracing()

# Make in-memory langgraph runtime flush thread state to disk more often so
# threads aren't lost on container restart (default is 10s; we lower to 2s).
try:
    from langgraph_runtime_inmem import _persistence as _lg_persist  # type: ignore

    _lg_persist._flush_interval = 2
except Exception:  # noqa: BLE001
    pass


# ── Router Schema ────────────────────────────────────────────────────────────


class Intent(StrEnum):
    GENERAL_CHAT = "general_chat"
    KNOWLEDGE_QUERY = "knowledge_query"
    REASONING_TASK = "reasoning_task"
    PROMPT_CACHING = "prompt_caching"


class RouterIntent(BaseModel):
    """Structured output from the intent-detection router."""

    intent: Intent = Field(description="Classified capability needed for the user's message.")
    reasoning: str = Field(description="One-sentence justification for the chosen intent.")


# ── Intent → Node mapping ───────────────────────────────────────────────────

_INTENT_TO_NODE: dict[Intent, str] = {
    Intent.GENERAL_CHAT: "generalist",
    Intent.KNOWLEDGE_QUERY: "researcher",
    Intent.REASONING_TASK: "reasoner",
    Intent.PROMPT_CACHING: "prompt_cacher",
}


def _assistant_name() -> str:
    return get_settings().assistant_name


# ── Nodes ────────────────────────────────────────────────────────────────────


async def router(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    """Classify user intent using structured output (ProviderStrategy)."""
    spec = get_agent_spec(Agents.ROUTER)
    agent = create_agent(
        model=get_chat_client(config=config),
        tools=[],
        system_prompt=spec.render_system_prompt(assistant_name=_assistant_name()),
        response_format=ProviderStrategy(RouterIntent),
    )
    # Filter to only human/AI messages — tool call/response pairs from prior
    # agent runs would cause OpenAI to reject the request.
    chat_messages = [
        m
        for m in state["messages"]
        if isinstance(m, (HumanMessage, AIMessage))
        and not getattr(m, "tool_calls", None)
    ]
    result = await agent.ainvoke({"messages": chat_messages})
    intent: RouterIntent = result["structured_response"]
    return {
        "messages": [
            AIMessage(
                content=intent.intent.value,
                additional_kwargs={"routing": intent.model_dump()},
            )
        ]
    }


def route_by_intent(
    state: MessagesState,
) -> Literal["generalist", "researcher", "reasoner", "prompt_cacher"]:
    """Read the router's structured classification and pick the next node."""
    last_msg = state["messages"][-1]
    routing = last_msg.additional_kwargs.get("routing", {})
    intent_value = routing.get("intent", last_msg.content.strip().lower())

    try:
        intent = Intent(intent_value)
    except ValueError:
        intent = Intent.GENERAL_CHAT  # safe fallback

    return _INTENT_TO_NODE[intent]


def _build_agent(
    agent_id: Agents,
    *,
    config: RunnableConfig | None = None,
    with_pii: bool = True,
):
    spec = get_agent_spec(agent_id)
    middleware: list[Any] = []
    if with_pii:
        # Redact common PII categories in model output.
        middleware.append(
            PIIMiddleware("credit_card", strategy="redact", apply_to_output=True)
        )
        middleware.append(
            PIIMiddleware("email", strategy="redact", apply_to_output=True)
        )
    return create_agent(
        model=get_chat_client(config=config),
        tools=spec.get_tools(),
        system_prompt=spec.render_system_prompt(assistant_name=_assistant_name()),
        middleware=middleware,
    )


async def generalist(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    """Default chat agent — friendly conversation + identity."""
    agent = _build_agent(Agents.GENERALIST, config=config)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": result["messages"]}


async def researcher(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    """Knowledge agent — local KB + Wikipedia."""
    agent = _build_agent(Agents.RESEARCHER, config=config)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": result["messages"]}


async def reasoner(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    """Math, logic, and step-by-step problem solving."""
    agent = _build_agent(Agents.REASONER, config=config)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": result["messages"]}


async def prompt_cacher(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    """LLM-prompt-caching expert (large system prompt, demonstrates caching)."""
    agent = _build_agent(Agents.PROMPT_CACHER, config=config)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": result["messages"]}


# ── Graph ────────────────────────────────────────────────────────────────────


def build_workflow() -> StateGraph:
    """Construct and compile the multi-agent Cortex workflow.

    Flow:
        START → router → (conditional)
                          → generalist
                          → researcher
                          → reasoner
                          → prompt_cacher
                        → END

    Guardrails:
        - PIIMiddleware: redacts credit-card numbers and emails in model output.

    Persistence:
        The LangGraph API runtime automatically provides a PostgreSQL
        checkpointer — no manual checkpointer is needed.
    """
    builder = StateGraph(MessagesState)

    # Nodes
    builder.add_node("router", router)
    builder.add_node("generalist", generalist)
    builder.add_node("researcher", researcher)
    builder.add_node("reasoner", reasoner)
    builder.add_node("prompt_cacher", prompt_cacher)

    # Edges
    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", route_by_intent)
    builder.add_edge("generalist", END)
    builder.add_edge("researcher", END)
    builder.add_edge("reasoner", END)
    builder.add_edge("prompt_cacher", END)

    return builder.compile()


# Module-level compiled graph for langgraph.json
graph = build_workflow()
