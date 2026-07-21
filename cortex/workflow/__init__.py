"""LangGraph workflow for the multi-agent assistant."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from cortex.workflow.nodes import (
    booking,
    coder,
    custom_agent,
    generalist,
    imagegen,
    prompt_cacher,
    reasoner,
    shopping,
)
from cortex.workflow.research import researcher
from cortex.workflow.routing import route_by_intent, route_from_start, router
from cortex.workflow.specialist import specialist
from cortex.workflow.synthesis import synthesize
from cortex.workflow.types import ChatState, Intent, RouterIntent

def build_workflow(*, checkpointer: Any = None, store: Any = None):
    """Compile the workflow with optional durable persistence."""
    builder = StateGraph(ChatState)
    builder.add_node("router", router)
    builder.add_node("generalist", generalist)
    builder.add_node("researcher", researcher)
    builder.add_node("reasoner", reasoner)
    builder.add_node("coder", coder)
    builder.add_node("shopping", shopping)
    builder.add_node("booking", booking)
    builder.add_node("prompt_cacher", prompt_cacher)
    builder.add_node("specialist", specialist)
    builder.add_node("imagegen", imagegen)
    builder.add_node("custom_agent", custom_agent)
    builder.add_node("synthesize", synthesize)
    builder.add_conditional_edges(START, route_from_start)
    builder.add_conditional_edges("router", route_by_intent)
    for node in (
        "generalist",
        "prompt_cacher",
        "imagegen",
        "shopping",
        "booking",
        "custom_agent",
    ):
        builder.add_edge(node, END)
    for node in ("researcher", "reasoner", "coder"):
        builder.add_edge(node, "synthesize")
    builder.add_edge("specialist", END)
    builder.add_edge("synthesize", END)
    return builder.compile(checkpointer=checkpointer, store=store)


graph = build_workflow()
