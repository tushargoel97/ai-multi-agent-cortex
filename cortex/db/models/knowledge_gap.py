"""Knowledge gaps, specialist questions the fine-tuned model couldn't answer.

Feeds the self-improvement loop: gaps are researched (web) by the trainer
service between runs, folded into the training data, and eliminated by the
next fine-tune. The model never uses the web at answer time.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from cortex.db.models.base import Base


class KnowledgeGap(Base):
    __tablename__ = "knowledge_gaps"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=True)
    reason: Mapped[str] = mapped_column(String(40), nullable=False, default="refusal")
    # new → researched → trained (or dismissed)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="new")
    researched_summary: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
