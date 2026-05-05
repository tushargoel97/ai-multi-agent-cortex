"""Custom guardrail middleware for Cortex agents.

The built-in :class:`langchain.agents.middleware.PIIMiddleware` already
covers credit-card and email redaction; the workflow installs that one
on every agent.

This module provides additional, opt-in middleware that is not enabled by
default but is available for users to add to ``_build_agent`` in
``cortex/workflow.py`` if they need it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage


class ToolAllowlistMiddleware(AgentMiddleware):
    """Hard-blocks any tool call whose name is not in the allowlist.

    Useful as a defensive layer above the LLM's tool selection: even if
    the model hallucinates a tool name, this middleware prevents the
    runtime from invoking anything outside the explicitly approved set.
    """

    def __init__(self, allowed_tools: Iterable[str]):
        self.allowed = set(allowed_tools)

    def _denied_message(self, tool_call: dict) -> ToolMessage:
        return ToolMessage(
            content=json.dumps(
                {
                    "success": False,
                    "error": (
                        f"Tool '{tool_call['name']}' is not in the "
                        "allowlist for this agent and has been blocked."
                    ),
                }
            ),
            tool_call_id=tool_call["id"],
        )

    def wrap_tool_call(self, request, handler):
        tool_call = request.tool_call
        if tool_call["name"] not in self.allowed:
            return self._denied_message(tool_call)
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        tool_call = request.tool_call
        if tool_call["name"] not in self.allowed:
            return self._denied_message(tool_call)
        return await handler(request)
