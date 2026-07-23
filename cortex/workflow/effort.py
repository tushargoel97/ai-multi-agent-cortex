from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from cortex.workflow.planning import ExecutionPlan

EffortLevel = Literal["adaptive", "low", "medium", "high", "xhigh", "max"]
ResolvedEffort = Literal["low", "medium", "high", "xhigh", "max"]
EFFORT_LEVELS: tuple[EffortLevel, ...] = (
    "adaptive",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)


@dataclass(frozen=True)
class EffortBudget:
    level: ResolvedEffort
    history_tokens: int
    max_tool_calls: int
    max_search_results: int
    tool_result_chars: int
    max_output_tokens: int


_BUDGETS: dict[ResolvedEffort, EffortBudget] = {
    "low": EffortBudget("low", 1_500, 2, 3, 4_500, 600),
    "medium": EffortBudget("medium", 2_500, 5, 4, 7_000, 1_200),
    "high": EffortBudget("high", 4_000, 8, 5, 10_000, 2_500),
    "xhigh": EffortBudget("xhigh", 6_000, 12, 6, 14_000, 4_000),
    "max": EffortBudget("max", 8_000, 16, 8, 20_000, 6_000),
}
_ADAPTIVE: dict[str, ResolvedEffort] = {
    "direct": "low",
    "grounded": "medium",
    "deliberate": "high",
    "research": "high",
}


def effort_budget(
    config: dict | None,
    plan: ExecutionPlan,
) -> EffortBudget:
    configurable = (config or {}).get("configurable") or {}
    requested = str(configurable.get("effort") or "adaptive").lower()
    level = requested if requested in _BUDGETS else _ADAPTIVE[plan.tier]
    return _BUDGETS[level]


def select_tools(tools: list, plan: ExecutionPlan) -> list:
    selected = set(plan.required_tools)
    if "web_search" in selected:
        selected.add("fetch_url")
    if "search_knowledge_base" in selected:
        selected.add("wikipedia_search")
    return [tool for tool in tools if getattr(tool, "name", "") in selected]


def budget_tool_call(
    tool_call: dict[str, Any],
    budget: EffortBudget,
) -> dict[str, Any]:
    name = str(tool_call.get("name") or "")
    args = dict(tool_call.get("args") or {})
    if name == "web_search":
        requested = args.get("max_results", budget.max_search_results)
        args["max_results"] = min(
            requested if isinstance(requested, int) else budget.max_search_results,
            budget.max_search_results,
        )
        args["fetch_pages"] = False
    elif name == "fetch_url":
        requested = args.get("max_chars", budget.tool_result_chars)
        args["max_chars"] = min(
            requested if isinstance(requested, int) else budget.tool_result_chars,
            budget.tool_result_chars,
        )
    return {**tool_call, "args": args}


def tool_budget_middleware(budget: EffortBudget):
    from langchain.agents.middleware import wrap_tool_call

    @wrap_tool_call
    async def enforce_tool_budget(request, handler):
        return await handler(
            request.override(tool_call=budget_tool_call(request.tool_call, budget))
        )

    return enforce_tool_budget
