"""Reusable workflow nodes for agent execution."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from cortex.db.services.agents import load_custom_agent
from cortex.db.services.tool_catalog import effective_tool_names, resolve_tool_instances
from cortex.enums import Agents
from cortex.errors import model_error_reply, retryable_model_exceptions
from cortex.local_grounding import run_grounded_local, run_local_answer
from cortex.model_client import auto_fallback_clients, get_chat_client
from cortex.workflow.context import (
    agent_context,
    invoke_config,
    is_router_marker,
    last_human,
    message_window,
    request_context,
    text_content,
)
from cortex.workflow.memory import memory_context, update_summary
from cortex.workflow.planning import ExecutionPlan, plan_from_messages
from cortex.workflow.runtime import assistant_name, build_agent, subagent_tools, system_prompt_for
from cortex.workflow.types import ChatState, Intent

logger = logging.getLogger("cortex.workflow")

def _missing_tools(messages: list, required: tuple[str, ...]) -> tuple[str, ...]:
    used = {
        str(message.name)
        for message in messages
        if isinstance(message, ToolMessage) and message.name
    }
    return tuple(name for name in required if name not in used)


def _tool_metadata(messages: list, required: tuple[str, ...]) -> dict[str, Any]:
    used = []
    failed = []
    for message in messages:
        if not isinstance(message, ToolMessage) or not message.name:
            continue
        name = str(message.name)
        if name not in used:
            used.append(name)
        content = text_content(message)
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            payload = None
        empty = isinstance(payload, dict) and any(
            key in payload and not payload[key] for key in ("results", "offers", "options")
        )
        if (isinstance(payload, dict) and payload.get("error")) or empty:
            failed.append(name)
    missing = [name for name in required if name not in used]
    status = "missing" if missing else "partial" if failed else "complete"
    metadata: dict[str, Any] = {
        "status": status,
        "required": list(required),
        "used": used,
    }
    if failed:
        metadata["failed"] = list(dict.fromkeys(failed))
    if missing:
        metadata["missing"] = missing
    return metadata


def _mark_execution(messages: list, plan: ExecutionPlan) -> AIMessage | None:
    final = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None)
        ),
        None,
    )
    if final is None:
        return None
    final.additional_kwargs = {
        **(final.additional_kwargs or {}),
        "execution_tier": plan.tier,
        **({"deep_research": True} if plan.tier == "research" else {}),
        **(
            {"tool_execution": _tool_metadata(messages, plan.required_tools)}
            if plan.required_tools
            else {}
        ),
    }
    return final


def model_error_message(exc: BaseException, config: RunnableConfig) -> AIMessage:
    from cortex.db.services.auto_mode import is_auto

    auto = is_auto(((config or {}).get("configurable") or {}).get("model_id"))
    logger.warning("Model call failed (%s: %s)", type(exc).__name__, exc)
    return AIMessage(
        content=model_error_reply(exc, auto=auto),
        additional_kwargs={"model_error": True},
    )


async def selected_local_response(
    intent: str,
    state: ChatState,
    config: RunnableConfig,
    *,
    grounded: bool = False,
) -> dict[str, Any] | None:
    plan = plan_from_messages(state["messages"], intent)
    context = request_context(config)
    if summary := (state.get("summary") or "").strip():
        context += f"\n\nConversation summary:\n{summary}"
    try:
        runner = run_grounded_local if grounded else run_local_answer
        message = await runner(
            intent,
            state["messages"],
            config,
            request_context=context,
        )
    except Exception as exc:  # noqa: BLE001
        return {"messages": [model_error_message(exc, config)]}
    if message is None:
        return None
    message.additional_kwargs = {
        **(message.additional_kwargs or {}),
        "execution_tier": plan.tier,
    }
    return {"messages": [message], **(await update_summary(state, config))}


async def run_agent(
    agent_id: Agents,
    state: ChatState,
    config: RunnableConfig,
    auto_intent: str | None = None,
) -> dict[str, Any]:
    memory_suffix, updates = await memory_context(state, config)
    plan = plan_from_messages(state["messages"], auto_intent or agent_id.value)
    complexity = plan.complexity
    required_tools = plan.required_tools
    extra_system = plan.directive()
    if memory_suffix:
        extra_system = f"{extra_system}\n\n{memory_suffix}" if extra_system else memory_suffix
    try:
        agent = build_agent(
            agent_id,
            config=config,
            extra_system=extra_system,
            auto_intent=auto_intent,
            complexity=complexity,
            max_tool_calls=plan.max_tool_calls,
        )
        result = await agent.ainvoke(
            {"messages": message_window(state["messages"])},
            config=invoke_config(config, plan.recursion_limit),
        )
        missing = _missing_tools(result.get("messages", []), required_tools)
        if missing:
            interim = next(
                (
                    message
                    for message in reversed(result.get("messages", []))
                    if isinstance(message, AIMessage)
                    and not getattr(message, "tool_calls", None)
                ),
                None,
            )
            if interim is not None and not interim.id:
                interim.id = "required-tools-interim"
            correction = HumanMessage(
                content=(
                    "Required tools before answering: "
                    + ", ".join(missing)
                    + ". Call them and use their returned values verbatim."
                ),
                id="required-tools-retry",
            )
            result = await agent.ainvoke(
                {"messages": [*result.get("messages", []), correction]},
                config=invoke_config(config),
            )
            result["messages"] = [
                message
                for message in result.get("messages", [])
                if getattr(message, "id", None)
                not in {correction.id, getattr(interim, "id", None)}
            ]
            missing = _missing_tools(result["messages"], required_tools)
            if missing:
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "I couldn't verify the requested live data, so I won't "
                                "provide an unverified answer. Check that the required "
                                "tools are enabled and try again."
                            ),
                            additional_kwargs={
                                "grounding_error": True,
                                "execution_tier": plan.tier,
                                "tool_execution": _tool_metadata(
                                    result["messages"], required_tools
                                ),
                            },
                        )
                    ],
                    **updates,
                }
    except Exception as exc:  # noqa: BLE001
        try:
            from cortex.db.services.auto_mode import (
                is_auto,
                profile_for_complexity,
                resolve_auto_model,
            )
            from cortex.model_client.model_health import report_model_failure

            if isinstance(exc, retryable_model_exceptions()) and auto_intent and is_auto(
                ((config or {}).get("configurable") or {}).get("model_id")
            ):
                primary = resolve_auto_model(
                    auto_intent,
                    profile=profile_for_complexity(complexity),
                )
                if primary is not None:
                    report_model_failure(primary.model_id)
        except Exception:  # noqa: BLE001
            pass
        return {"messages": [model_error_message(exc, config)], **updates}
    final = _mark_execution(result["messages"], plan)
    try:
        from cortex.model_client.model_health import report_model_success

        report_model_success(
            str((final.response_metadata or {}).get("model_name") or "")
            if final
            else None
        )
    except Exception:  # noqa: BLE001
        pass
    return {"messages": result["messages"], **updates}


async def generalist(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    local = await selected_local_response(Intent.GENERAL_CHAT.value, state, config)
    return local or await run_agent(
        Agents.GENERALIST, state, config, Intent.GENERAL_CHAT.value
    )


async def reasoner(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    local = await selected_local_response(Intent.REASONING_TASK.value, state, config)
    return local or await run_agent(
        Agents.REASONER, state, config, Intent.REASONING_TASK.value
    )


async def coder(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    configurable = (config or {}).get("configurable") or {}
    intent = (
        "engineer"
        if str(configurable.get("mode") or "").lower() == "engineer"
        else Intent.CODING_TASK.value
    )
    local = await selected_local_response(intent, state, config)
    return local or await run_agent(Agents.CODER, state, config, intent)


async def shopping(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    local = await selected_local_response(
        Intent.SHOPPING.value, state, config, grounded=True
    )
    return local or await run_agent(
        Agents.SHOPPING,
        state,
        config,
        Intent.SHOPPING.value,
    )


async def booking(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    local = await selected_local_response(
        Intent.BOOKING.value, state, config, grounded=True
    )
    return local or await run_agent(
        Agents.BOOKING, state, config, Intent.BOOKING.value
    )


async def custom_agent(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    routing: dict | None = None
    for message in reversed(state["messages"]):
        if is_router_marker(message):
            routing = message.additional_kwargs.get("routing") or {}
            break
    name = (routing or {}).get("agent")
    spec = load_custom_agent(name) if name else None
    if spec is None:
        return await generalist(state, config)

    memory_suffix, updates = await memory_context(state, config)
    plan = plan_from_messages(
        state["messages"],
        str((routing or {}).get("intent") or Intent.GENERAL_CHAT.value),
    )
    intent = str((routing or {}).get("intent") or Intent.GENERAL_CHAT.value)
    model = get_chat_client(
        config=config,
        auto_intent=intent,
        complexity=plan.complexity,
    )
    from jinja2 import Template

    try:
        static = Template(spec["system_prompt"]).render(assistant_name=assistant_name())
    except Exception:  # noqa: BLE001
        static = spec["system_prompt"]
    context = agent_context(config, complexity=plan.complexity)
    if directive := plan.directive():
        context += f"\n\n{directive}"
    dynamic = f"{context}\n\n{memory_suffix}" if memory_suffix else context
    try:
        tools = resolve_tool_instances(effective_tool_names(name, []))
    except Exception:  # noqa: BLE001
        tools = []
    tools += subagent_tools(name, config)
    middleware: list[Any] = []
    if not bool(((config or {}).get("configurable") or {}).get("unrestricted")):
        middleware = [
            PIIMiddleware("credit_card", strategy="redact", apply_to_output=True),
            PIIMiddleware("email", strategy="redact", apply_to_output=True),
        ]
    if plan.max_tool_calls:
        middleware.append(
            ToolCallLimitMiddleware(
                run_limit=plan.max_tool_calls,
                exit_behavior="continue",
            )
        )

    def make_agent(client: Any):
        return create_agent(
            model=client,
            tools=tools,
            system_prompt=system_prompt_for(client, static, dynamic),
            middleware=middleware,
        )

    agent = make_agent(model)
    fallbacks = auto_fallback_clients(
        config,
        auto_intent=intent,
        complexity=plan.complexity,
    )
    if fallbacks:
        agent = agent.with_fallbacks(
            [make_agent(client) for client in fallbacks],
            exceptions_to_handle=retryable_model_exceptions(),
        )
    try:
        result = await agent.ainvoke(
            {"messages": message_window(state["messages"])},
            config=invoke_config(config, plan.recursion_limit),
        )
    except Exception as exc:  # noqa: BLE001
        return {"messages": [model_error_message(exc, config)], **updates}
    _mark_execution(result["messages"], plan)
    return {"messages": result["messages"], **updates}


async def prompt_cacher(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    local = await selected_local_response(Intent.PROMPT_CACHING.value, state, config)
    return local or await run_agent(
        Agents.PROMPT_CACHER, state, config, Intent.PROMPT_CACHING.value
    )


async def imagegen(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    last = next(
        (message for message in reversed(state["messages"]) if isinstance(message, HumanMessage)),
        None,
    )
    prompt = str(last.content) if last is not None else ""
    configurable = (config or {}).get("configurable") or {}
    from cortex.imagegen import generate_image

    result = await generate_image(
        prompt,
        str(configurable.get("thread_id") or "thread"),
        unrestricted=bool(configurable.get("unrestricted")),
    )
    if result.status == "ok":
        message = AIMessage(
            content=(
                f"{result.detail or 'Here you go:'}\n\n"
                f"![Generated image](/api/v1/images/{result.filename})"
            ),
            response_metadata={"model_name": result.model_used},
        )
    else:
        message = AIMessage(content=result.detail)
    return {"messages": [message]}
