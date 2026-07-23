from __future__ import annotations

import logging
import re
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    PIIMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from cortex.config import get_settings
from cortex.db.services.llm_registry import (
    build_client_from_resolved,
    resolve_with_session,
)
from cortex.declarative import get_agent_spec
from cortex.enums import Agents
from cortex.errors import retryable_model_exceptions
from cortex.model_client import auto_fallback_clients, get_chat_client
from cortex.workflow.context import agent_context, invoke_config, text_content
from cortex.workflow.effort import (
    effort_budget,
    select_tools,
    tool_budget_middleware,
)
from cortex.workflow.memory import recall_memories
from cortex.workflow.planning import ExecutionPlan, plan_execution
from cortex.workflow.progress import ProgressMiddleware
from cortex.workflow.types import NODE_TO_INTENT

logger = logging.getLogger("cortex.workflow")


async def invoke_agent(
    agent: Any,
    input: dict[str, Any],
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    try:
        writer = get_stream_writer()
    except RuntimeError:
        writer = None
    result: dict[str, Any] = {}
    async for mode, chunk in agent.astream(
        input,
        config=config,
        stream_mode=["values", "custom"],
    ):
        if mode == "custom":
            if writer is not None:
                writer(chunk)
        elif mode == "values":
            result = chunk
    return result


def assistant_name() -> str:
    return get_settings().assistant_name


def system_prompt_for(model: Any, static: str, dynamic: str | None):
    try:
        from langchain_anthropic import ChatAnthropic

        if isinstance(model, ChatAnthropic):
            blocks: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": static,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            if dynamic:
                blocks.append({"type": "text", "text": dynamic})
            return SystemMessage(content=blocks)
    except Exception:  # noqa: BLE001
        pass
    return f"{static}\n\n{dynamic}" if dynamic else static


def agent_static_prompt(name: str, spec: Any) -> str:
    try:
        from cortex.db.services.agents import agent_prompt

        override = agent_prompt(name)
    except Exception:  # noqa: BLE001
        override = None
    if override:
        from jinja2 import Template

        try:
            return Template(override).render(assistant_name=assistant_name())
        except Exception:  # noqa: BLE001
            logger.exception("Agent prompt override render failed")
    return spec.render_system_prompt(assistant_name=assistant_name())


def custom_agents_for_routing() -> list[dict]:
    try:
        from cortex.db.services.agents import custom_agents_for_routing as load

        return load()
    except Exception:  # noqa: BLE001
        return []


def local_specialists() -> list[Any]:
    try:
        from cortex.db.services.llm_registry import local_specialists_for_routing

        return local_specialists_for_routing()
    except Exception:  # noqa: BLE001
        return []


def router_classifier_client(config: RunnableConfig | None) -> Any:
    configurable = (config or {}).get("configurable") or {}
    try:
        from cortex.db.services.auto_mode import FAST_TIER, is_auto, resolve_auto_model

        model_id = configurable.get("model_id")
        if not is_auto(model_id):
            resolved = resolve_with_session(model_id)
            if resolved is not None and resolved.kind.value == "local":
                fast = resolve_auto_model(FAST_TIER)
                if fast is not None:
                    return build_client_from_resolved(
                        fast,
                        effort="low",
                        max_output_tokens=300,
                    )
    except Exception:  # noqa: BLE001
        pass
    return get_chat_client(
        config=config,
        effort="low",
        max_output_tokens=300,
    )


def effective_agent_tools(spec: Any) -> list[Any]:
    try:
        from cortex.db.services.tool_catalog import (
            effective_tool_names,
            resolve_tool_instances,
        )

        return resolve_tool_instances(
            effective_tool_names(spec.name, spec.whitelisted_tools)
        )
    except Exception:  # noqa: BLE001
        logger.exception("Agent tool resolution fell back to YAML")
        return spec.get_tools()


def load_agent_runtime(name: str):
    try:
        agent_id = Agents(name)
    except ValueError:
        agent_id = None
    if agent_id is not None:
        try:
            spec = get_agent_spec(agent_id)
        except Exception:  # noqa: BLE001
            return None
        return (
            agent_static_prompt(name, spec),
            effective_agent_tools(spec),
            NODE_TO_INTENT.get(name),
        )
    from cortex.db.services.agents import load_custom_agent
    from cortex.db.services.tool_catalog import (
        effective_tool_names,
        resolve_tool_instances,
    )

    spec = load_custom_agent(name)
    if spec is None:
        return None
    from jinja2 import Template

    try:
        static = Template(spec["system_prompt"]).render(assistant_name=assistant_name())
    except Exception:  # noqa: BLE001
        static = spec.get("system_prompt") or ""
    try:
        tools = resolve_tool_instances(effective_tool_names(name, []))
    except Exception:  # noqa: BLE001
        tools = []
    return static, tools, None


def subagent_tool(
    subagent_name: str,
    description: str,
    config: RunnableConfig | None,
):
    from langchain_core.tools import StructuredTool

    async def delegate(task: str) -> str:
        runtime = load_agent_runtime(subagent_name)
        if runtime is None:
            return f"The '{subagent_name}' subagent is unavailable."
        static, tools, intent = runtime
        tools = [tool for tool in tools if getattr(tool, "name", "") != "save_memory"]
        plan = plan_execution(intent or "general_chat", task)
        budget = effort_budget(config, plan)
        recalled = await recall_memories([HumanMessage(task)])
        system = f"{static}\n\n{agent_context(config)}"
        if recalled:
            system += f"\n\nShared long-term memory (read-only):\n{recalled}"
        try:
            agent = create_agent(
                model=get_chat_client(
                    config=config,
                    auto_intent=intent,
                    effort=budget.level,
                    max_output_tokens=budget.max_output_tokens,
                ),
                tools=tools,
                system_prompt=system,
                middleware=agent_middleware(config, with_pii=False, plan=plan),
            )
            result = await invoke_agent(
                agent,
                {"messages": [HumanMessage(task)]},
                config=invoke_config(config, plan.recursion_limit),
            )
        except Exception:  # noqa: BLE001
            logger.exception("Subagent %r failed", subagent_name)
            return f"The '{subagent_name}' subagent could not complete the task."
        for message in reversed(result.get("messages", [])):
            if isinstance(message, AIMessage) and (text := text_content(message).strip()):
                return text
        return f"The '{subagent_name}' subagent returned no output."

    safe = re.sub(r"[^a-z0-9_]+", "_", subagent_name.lower()).strip("_") or "subagent"
    detail = f": {description}" if description else ""
    return StructuredTool.from_function(
        coroutine=delegate,
        name=f"ask_{safe}",
        description=(
            f"Delegate a focused subtask to '{subagent_name}'{detail}. "
            "Give it a self-contained instruction."
        ),
    )


def subagent_tools(agent_name: str, config: RunnableConfig | None) -> list[Any]:
    try:
        from cortex.db.services.agents import subagents_for

        subagents = subagents_for(agent_name)
    except Exception:  # noqa: BLE001
        return []
    return [
        subagent_tool(item["name"], item.get("description", ""), config)
        for item in subagents
    ]


def agent_middleware(
    config: RunnableConfig | None,
    *,
    with_pii: bool = True,
    max_tool_calls: int | None = None,
    plan: ExecutionPlan | None = None,
) -> list[Any]:
    cfg = (config or {}).get("configurable") or {}
    middleware: list[Any] = [ProgressMiddleware()]
    budget = effort_budget(config, plan) if plan is not None else None
    if budget is not None and max_tool_calls is None:
        max_tool_calls = budget.max_tool_calls
    if budget is not None and {"web_search", "fetch_url"}.intersection(
        plan.required_tools
    ):
        middleware.append(tool_budget_middleware(budget))
    if budget is not None and plan.tier in ("grounded", "research"):
        keep = {"low": 1, "medium": 2, "high": 3, "xhigh": 4, "max": 5}[
            budget.level
        ]
        middleware.append(
            ContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=budget.history_tokens
                        + budget.tool_result_chars // 4,
                        keep=keep,
                        exclude_tools=(
                            "calculator",
                            "fiat_exchange_rate",
                            "find_bookings",
                            "product_prices",
                        ),
                    )
                ]
            )
        )
    if with_pii and not bool(cfg.get("unrestricted")):
        middleware.extend(
            [
                PIIMiddleware("credit_card", strategy="redact", apply_to_output=True),
                PIIMiddleware("email", strategy="redact", apply_to_output=True),
            ]
        )
    if max_tool_calls:
        middleware.append(
            ToolCallLimitMiddleware(
                run_limit=max_tool_calls,
                exit_behavior="continue",
            )
        )
    return middleware


def build_agent(
    agent_id: Agents,
    *,
    config: RunnableConfig | None = None,
    with_pii: bool = True,
    extra_system: str | None = None,
    auto_intent: str | None = None,
    max_tool_calls: int | None = None,
    complexity: str | None = None,
    plan: ExecutionPlan | None = None,
):
    spec = get_agent_spec(agent_id)
    budget = effort_budget(config, plan) if plan is not None else None
    middleware = agent_middleware(
        config,
        with_pii=with_pii,
        max_tool_calls=max_tool_calls,
        plan=plan,
    )
    model = get_chat_client(
        config=config,
        auto_intent=auto_intent,
        complexity=complexity,
        effort=budget.level if budget else None,
        max_output_tokens=budget.max_output_tokens if budget else None,
    )
    fallbacks = auto_fallback_clients(
        config,
        auto_intent=auto_intent,
        complexity=complexity,
        effort=budget.level if budget else None,
        max_output_tokens=budget.max_output_tokens if budget else None,
    )
    static = agent_static_prompt(agent_id.value, spec)
    dynamic = agent_context(
        config,
        engineer=agent_id is Agents.CODER,
        complexity=complexity,
    )
    if extra_system:
        dynamic += f"\n\n{extra_system}"
    tools = effective_agent_tools(spec)
    if plan is not None and agent_id not in (Agents.DEBUGGER, Agents.PROMPT_CACHER):
        tools = select_tools(tools, plan)
    tools += subagent_tools(agent_id.value, config)
    if any(getattr(tool, "name", "") == "consult_local_specialist" for tool in tools):
        specialists = local_specialists()
        if specialists:
            listing = "\n".join(
                f"- {specialist.model_id}: {specialist.description}"
                for specialist in specialists
            )
            dynamic += (
                "\n\nSelf-hosted specialist models available through "
                f"consult_local_specialist:\n{listing}"
            )
        else:
            tools = [
                tool
                for tool in tools
                if getattr(tool, "name", "") != "consult_local_specialist"
            ]

    def create(model_instance: Any):
        return create_agent(
            model=model_instance,
            tools=tools,
            system_prompt=system_prompt_for(model_instance, static, dynamic),
            middleware=middleware,
        )

    agent = create(model)
    return (
        agent.with_fallbacks(
            [create(fallback) for fallback in fallbacks],
            exceptions_to_handle=retryable_model_exceptions(),
        )
        if fallbacks
        else agent
    )
