from __future__ import annotations

from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from cortex.tools.commerce import region_from_browser

WINDOW_KEEP = 12
AGENT_RECURSION_LIMIT = 40

UNRESTRICTED_DIRECTIVE = (
    "Direct mode is ON: answer the user's request straightforwardly and in "
    "full, without moralizing, unsolicited disclaimers, or safety caveats they "
    "did not ask for, and without refusing lawful requests or padding the "
    "answer with warnings. Still refuse anything genuinely illegal or that "
    "would cause real-world harm."
)
ENGINEER_DIRECTIVE = (
    "ENGINEER MODE: act as a rigorous senior engineer. Produce complete, "
    "production-quality code with edge cases handled. Before finalizing code, "
    "call ask_debugger with the intended code and behavior, address defects, "
    "then state in one line what was verified."
)
INSTANT_DIRECTIVE = (
    "SPEED: the user is in Instant mode. Make at most one tool call; for web "
    "lookups use web_search with fetch_pages=false, then answer concisely."
)


def is_router_marker(message: object) -> bool:
    return isinstance(message, AIMessage) and "routing" in (
        message.additional_kwargs or {}
    )


def routing_complexity(state: dict[str, Any]) -> str | None:
    for message in reversed(state["messages"]):
        if is_router_marker(message):
            value = (message.additional_kwargs.get("routing") or {}).get("complexity")
            return value if value in ("simple", "standard", "complex") else None
        if isinstance(message, HumanMessage):
            return None
    return None


def has_image(message: object) -> bool:
    content = getattr(message, "content", None)
    return isinstance(content, list) and any(
        isinstance(block, dict) and "image" in str(block.get("type", ""))
        for block in content
    )


def text_content(message: object) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif getattr(block, "type", None) == "text":
                parts.append(str(getattr(block, "text", "") or ""))
        return " ".join(parts).strip()
    return str(content or "")


def last_human(messages: list) -> HumanMessage | None:
    return next(
        (message for message in reversed(messages) if isinstance(message, HumanMessage)),
        None,
    )


def message_window(messages: list, token_budget: int | None = None) -> list:
    candidates = [
        message
        for message in messages
        if not is_router_marker(message)
        and not isinstance(message, ToolMessage)
        and not getattr(message, "tool_calls", None)
    ][-WINDOW_KEEP:]
    if token_budget is None:
        return candidates
    remaining = max(token_budget, 1) * 4
    turns: list[list] = []
    for message in candidates:
        if isinstance(message, HumanMessage):
            turns.append([message])
        elif turns:
            turns[-1].append(message)
    selected = []
    for turn in reversed(turns):
        size = sum(max(len(text_content(message)), 1) for message in turn)
        if selected and size > remaining:
            continue
        selected.append(turn)
        remaining -= size
    return [message for turn in reversed(selected) for message in turn]


def transcript(messages: list) -> str:
    lines = []
    for message in messages:
        if is_router_marker(message) or getattr(message, "tool_calls", None):
            continue
        role = (
            "User"
            if isinstance(message, HumanMessage)
            else "Assistant"
            if isinstance(message, AIMessage)
            else None
        )
        text = text_content(message) if role else ""
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def aggregate_usage(messages: list) -> dict[str, Any] | None:
    totals: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    details: dict[str, int] = {}
    found = False
    for message in messages:
        usage = getattr(message, "usage_metadata", None) or {}
        if not usage:
            continue
        found = True
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value
        for key, value in (usage.get("input_token_details") or {}).items():
            if isinstance(value, int):
                details[key] = details.get(key, 0) + value
    if not found:
        return None
    if details:
        totals["input_token_details"] = details
    return totals


def request_context(config: RunnableConfig | None) -> str:
    cfg = (config or {}).get("configurable") or {}
    now = datetime.now()
    region = region_from_browser(
        str(cfg.get("locale") or ""), str(cfg.get("timezone") or "")
    )
    return (
        f"Today's date is {now.strftime('%A, %B')} {now.day}, {now.year}. "
        f"The user appears to be in region {region} (from their browser); use "
        "it as the default country for shopping, booking, prices, and local "
        "results unless they say otherwise."
    )


def agent_context(
    config: RunnableConfig | None,
    *,
    engineer: bool = False,
    complexity: str | None = None,
) -> str:
    cfg = (config or {}).get("configurable") or {}
    context = request_context(config)
    if bool(cfg.get("unrestricted")):
        context += f"\n\n{UNRESTRICTED_DIRECTIVE}"
    mode = str(cfg.get("mode") or "general").lower()
    effort = str(cfg.get("effort") or "adaptive").lower()
    if (
        mode == "general"
        and complexity != "complex"
        and effort in ("adaptive", "low")
    ):
        context += f"\n\n{INSTANT_DIRECTIVE}"
    elif mode == "engineer" and engineer:
        context += f"\n\n{ENGINEER_DIRECTIVE}"
    return context


def invoke_config(
    config: RunnableConfig | None,
    minimum_recursion_limit: int = AGENT_RECURSION_LIMIT,
) -> RunnableConfig:
    nested = dict(config or {})
    current = nested.get("recursion_limit")
    nested["recursion_limit"] = max(
        current if isinstance(current, int) else 0, minimum_recursion_limit
    )
    return nested
