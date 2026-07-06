"""DB-backed tool catalog.

Mirrors built-in tools into the ``tools`` table, resolves each agent's effective
tool set (DB grants override the YAML whitelist), and loads external tools
(LangChain catalog + MCP servers) into a cached pool the agents draw from.
"""

from __future__ import annotations

import logging

from langchain_core.tools import BaseTool

from cortex.db.engine import engine, get_session
from cortex.db.models import Base
from cortex.db.models.tool import AgentTool, MCPServer, Tool, ToolKind

logger = logging.getLogger(__name__)

_tables_ready = False
_dynamic_pool: dict[str, BaseTool] = {}


def _ensure_tables() -> None:
    global _tables_ready
    if _tables_ready:
        return
    Base.metadata.create_all(
        engine, tables=[MCPServer.__table__, Tool.__table__, AgentTool.__table__]
    )
    _tables_ready = True


def publish_tool_catalog() -> None:
    """Mirror built-in tools into the tools table (once per process).

    Only inserts rows that don't exist yet — never clobbers admin edits.
    """
    try:
        _ensure_tables()
        import cortex.tools  # noqa: F401 — ensure @register_tool decorators ran
        from cortex.tools.registry import registry

        with get_session() as s:
            known = {row[0] for row in s.query(Tool.name).all()}
            for name, tool in registry.items():
                if name in known:
                    continue
                s.add(
                    Tool(
                        name=name,
                        kind=ToolKind.BUILTIN.value,
                        description=(getattr(tool, "description", "") or "")[:500],
                        enabled=True,
                        config={},
                    )
                )
        _publish_ui_mirror()
    except Exception:  # noqa: BLE001 — catalog mirror is best-effort
        logger.exception("publish_tool_catalog failed")


def _publish_ui_mirror() -> None:
    """Mirror the LangChain catalog + agent tool defaults into app_settings so
    the admin UI (Postgres-only) can render them without importing cortex."""
    import json

    from cortex.db.services.app_settings import set_setting

    try:
        from cortex.tools.catalog import catalog_listing

        set_setting("tool_catalog", json.dumps(catalog_listing()))
    except Exception:  # noqa: BLE001
        logger.exception("catalog mirror failed")
    try:
        from cortex.declarative import AGENT_SPECS

        defaults = {
            name: list(spec.whitelisted_tools) for name, spec in AGENT_SPECS.items()
        }
        set_setting("agent_tool_defaults", json.dumps(defaults))
    except Exception:  # noqa: BLE001
        logger.exception("agent defaults mirror failed")


def effective_tool_names(agent_name: str, yaml_default: list[str]) -> list[str]:
    """Tool names an agent may use.

    DB grants (``agent_tools``) replace the YAML whitelist when present, then
    globally-disabled tools are removed.
    """
    _ensure_tables()
    with get_session() as s:
        grants = [
            row[0]
            for row in s.query(AgentTool.tool_name)
            .filter(AgentTool.agent_name == agent_name)
            .all()
        ]
        disabled = {
            row[0] for row in s.query(Tool.name).filter(Tool.enabled.is_(False)).all()
        }
    names = grants if grants else list(yaml_default)
    return [n for n in names if n not in disabled]


def resolve_tool_instances(names: list[str]) -> list[BaseTool]:
    """Resolve names to instances from the built-in registry + cached pool."""
    from cortex.tools.registry import registry

    out: list[BaseTool] = []
    for n in names:
        tool = registry.get(n) or _dynamic_pool.get(n)
        if tool is not None:
            out.append(tool)
    return out


def dynamic_pool() -> dict[str, BaseTool]:
    return dict(_dynamic_pool)


def _load_catalog_tools() -> dict[str, BaseTool]:
    try:
        from cortex.tools.catalog import build_catalog_tool
    except Exception:  # noqa: BLE001
        return {}
    with get_session() as s:
        specs = [
            (t.name, dict(t.config or {}))
            for t in s.query(Tool)
            .filter(Tool.kind == ToolKind.LANGCHAIN.value, Tool.enabled.is_(True))
            .all()
        ]
    out: dict[str, BaseTool] = {}
    for name, config in specs:
        catalog_id = config.get("catalog") or name
        try:
            out[name] = build_catalog_tool(catalog_id, config.get("config") or {})
        except Exception as e:  # noqa: BLE001
            logger.warning("catalog tool %r unavailable: %s", name, e)
    return out


async def _load_mcp_tools() -> list[BaseTool]:
    _ensure_tables()
    with get_session() as s:
        servers = [
            {
                "name": sv.name,
                "transport": sv.transport,
                "url": sv.url,
                "command": sv.command,
                "args": list(sv.args or []),
                "env": dict(sv.env or {}),
            }
            for sv in s.query(MCPServer).filter(MCPServer.enabled.is_(True)).all()
        ]
    connections: dict[str, dict] = {}
    for sv in servers:
        if sv["transport"] in ("streamable_http", "sse") and sv["url"]:
            conn: dict = {"url": sv["url"], "transport": sv["transport"]}
            if sv["env"]:
                conn["headers"] = sv["env"]
            connections[sv["name"]] = conn
        elif sv["transport"] == "stdio" and sv["command"]:
            connections[sv["name"]] = {
                "command": sv["command"],
                "args": sv["args"],
                "transport": "stdio",
                "env": sv["env"],
            }
    if not connections:
        return []
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        # Prefix tool names with their server so multiple MCP servers (and the
        # built-ins) never collide.
        client = MultiServerMCPClient(connections, tool_name_prefix=True)
        return await client.get_tools()
    except Exception:  # noqa: BLE001 — a bad server must not break tool loading
        logger.exception("MCP tool load failed")
        return []


def _sync_mcp_tool_rows(tools: list[BaseTool]) -> None:
    """Upsert discovered MCP tools so the admin UI can grant them to agents."""
    try:
        with get_session() as s:
            known = {
                row[0]
                for row in s.query(Tool.name)
                .filter(Tool.kind == ToolKind.MCP.value)
                .all()
            }
            for t in tools:
                if t.name in known:
                    continue
                s.add(
                    Tool(
                        name=t.name,
                        kind=ToolKind.MCP.value,
                        description=(getattr(t, "description", "") or "")[:500],
                        enabled=True,
                        config={},
                    )
                )
    except Exception:  # noqa: BLE001
        logger.exception("MCP tool row sync failed")


async def refresh_dynamic_tools() -> None:
    """Rebuild the cached pool of external tools (LangChain catalog + MCP)."""
    global _dynamic_pool
    pool: dict[str, BaseTool] = {}
    try:
        pool.update(_load_catalog_tools())
    except Exception:  # noqa: BLE001
        logger.exception("catalog tool load failed")
    mcp_tools = await _load_mcp_tools()
    if mcp_tools:
        _sync_mcp_tool_rows(mcp_tools)
        for t in mcp_tools:
            pool[t.name] = t
    _dynamic_pool = pool
    logger.info("Dynamic tool pool refreshed: %d external tool(s)", len(pool))
