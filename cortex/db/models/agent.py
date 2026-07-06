"""Agents — DB-editable system prompts for built-ins and admin-created agents.

Built-in agents (from ``cortex/declarative/agents.yaml``) are mirrored here so
their system prompts can be edited from the admin UI; custom agents are created
entirely in the UI and auto-route via the intent router (by ``description``).
Per-agent tool access reuses the ``agent_tools`` grants.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import Boolean, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from cortex.db.models.base import Base


class AgentKind(StrEnum):
    BUILTIN = "builtin"  # a graph node backed by agents.yaml
    CUSTOM = "custom"    # admin-created, routed by description via custom_agent


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(60), nullable=False, unique=True)
    kind: Mapped[str] = mapped_column(
        String(20), nullable=False, default=AgentKind.CUSTOM.value
    )
    # Routing hint for the intent router (custom agents) + UI label.
    description: Mapped[str] = mapped_column(Text, default="")
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # True once an admin edits a built-in's prompt — stops the startup sync
    # from overwriting their edit with the packaged YAML.
    edited: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
