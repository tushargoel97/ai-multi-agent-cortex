"""Tool control — available tools, MCP servers, and per-agent tool grants.

Admin-UI managed. Built-in tools (``cortex.tools.registry``) are mirrored here
so they can be enabled/disabled; admins can also add prebuilt LangChain tools
(``cortex.tools.catalog``) and external MCP servers, then grant any of them to
individual agents. Per-agent grants override the YAML ``whitelisted_tools``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from cortex.db.models.base import Base


class ToolKind(StrEnum):
    BUILTIN = "builtin"      # registered in cortex.tools.registry
    LANGCHAIN = "langchain"  # prebuilt LangChain integration (cortex.tools.catalog)
    MCP = "mcp"              # provided by an external MCP server


class MCPServer(Base):
    """An external MCP server whose tools become available to grant to agents."""

    __tablename__ = "mcp_servers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    # streamable_http | sse | stdio
    transport: Mapped[str] = mapped_column(
        String(20), nullable=False, default="streamable_http"
    )
    url: Mapped[str | None] = mapped_column(Text, nullable=True)      # http transports
    command: Mapped[str | None] = mapped_column(Text, nullable=True)  # stdio
    args: Mapped[list] = mapped_column(JSONB, default=list)           # stdio args
    env: Mapped[dict] = mapped_column(JSONB, default=dict)            # env / http headers
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Tool(Base):
    """A tool the agents can be granted — built-in, LangChain catalog, or MCP."""

    __tablename__ = "tools"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    kind: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ToolKind.BUILTIN.value
    )
    description: Mapped[str] = mapped_column(Text, default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # For langchain tools: {"catalog": "<id>", "config": {...}}. For MCP: {}.
    config: Mapped[dict] = mapped_column(JSONB, default=dict)
    mcp_server_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("mcp_servers.id", ondelete="CASCADE"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AgentTool(Base):
    """Per-agent tool grant. When any row exists for an agent, its grants
    replace that agent's YAML ``whitelisted_tools`` (admin edits win)."""

    __tablename__ = "agent_tools"
    __table_args__ = (
        UniqueConstraint("agent_name", "tool_name", name="uq_agent_tool"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    agent_name: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    tool_name: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
