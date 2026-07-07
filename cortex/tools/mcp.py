"""FastMCP server, exposes Cortex's stateless tools over the Model Context
Protocol for external MCP clients (Claude Desktop, IDEs, other agents).

Additive and decoupled by design: the chat graph uses the **same** tools
in-process (see :mod:`cortex.tools.registry`), so the app never depends on this
server and pays no per-call latency, if this server is down, the assistant is
unaffected.

Stateful tools are intentionally NOT exposed here because they need process-local
context the graph provides:

* ``save_memory`` / ``search_memories``, require the LangGraph runtime store.
* ``search_knowledge_base``, requires the Postgres/pgvector session + embeddings.

Everything else (web search, page fetch, Wikipedia, crypto, product prices,
booking search, time, calculator) is pure network/compute and is
bridged to MCP via ``to_fastmcp``.

Run it:  ``python -m cortex.tools.mcp``  (streamable-HTTP on ``$MCP_PORT``)
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP
from langchain_mcp_adapters.tools import to_fastmcp

import cortex.tools  # noqa: F401, importing the package registers every tool
from cortex.tools.registry import registry

# Stateful tools that need process-local context (runtime store / DB session).
# They stay in-process only and are never exposed over MCP.
INPROCESS_ONLY: frozenset[str] = frozenset(
    {"save_memory", "search_memories", "search_knowledge_base"}
)


def exposed_tool_names() -> list[str]:
    """Names of the stateless tools that are safe to serve over MCP."""
    return sorted(name for name in registry if name not in INPROCESS_ONLY)


def build_mcp() -> FastMCP:
    """Assemble a FastMCP server from the stateless registry tools."""
    tools = [to_fastmcp(registry[name]) for name in exposed_tool_names()]
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8811"))
    return FastMCP(name="cortex-tools", tools=tools, host=host, port=port)


mcp = build_mcp()


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
