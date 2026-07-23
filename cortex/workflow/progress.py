from __future__ import annotations

from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.config import get_stream_writer

ProgressPhase = Literal[
    "routing",
    "thinking",
    "researching",
    "collating",
    "refining",
    "generating_image",
]


def emit_progress(phase: ProgressPhase, **details: str) -> None:
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(
        {
            "type": "agent_progress",
            "phase": phase,
            **{key: value for key, value in details.items() if value},
        }
    )


class ProgressMiddleware(AgentMiddleware):
    def before_model(self, state: dict[str, Any], runtime: Any) -> None:
        messages = state.get("messages", [])
        latest_human = max(
            (
                index
                for index, message in enumerate(messages)
                if isinstance(message, HumanMessage)
            ),
            default=-1,
        )
        phase = (
            "refining"
            if any(
                isinstance(message, ToolMessage)
                for message in messages[latest_human + 1 :]
            )
            else "thinking"
        )
        emit_progress(phase)

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        tool = str(request.tool_call.get("name") or "")
        emit_progress("researching", tool=tool)
        result = await handler(request)
        emit_progress("collating")
        return result
