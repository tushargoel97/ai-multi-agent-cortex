"""Gap capture for the specialist's self-improvement loop."""

from __future__ import annotations

import logging

from cortex.db.engine import engine, get_session
from cortex.db.models import Base
from cortex.db.models.knowledge_gap import KnowledgeGap

logger = logging.getLogger(__name__)

_table_ready = False

# Phrases the fine-tuned refusal training emits when it knows it doesn't know.
REFUSAL_MARKERS = (
    "isn't in my fine-tuned dataset",
    "rather not guess",
    "outside my fine-tuned knowledge",
    "outside my trained capabilities",
    "outside my training data",
    "not covered by my training",
    "only give you one side",
)

def detect_gap(_question: str, answer: str) -> str | None:
    """Return the gap reason, or None when the answer looks adequate."""
    low = answer.lower()
    if any(m in low for m in REFUSAL_MARKERS):
        return "refusal"
    return None


def _ensure_table() -> None:
    global _table_ready
    if _table_ready:
        return
    Base.metadata.create_all(engine, tables=[KnowledgeGap.__table__])
    _table_ready = True


def log_gap(question: str, answer: str, reason: str) -> bool:
    """Record a gap; never raises (chat must not break on logging)."""
    try:
        _ensure_table()
        with get_session() as session:
            exists = (
                session.query(KnowledgeGap)
                .filter(KnowledgeGap.question == question)
                .filter(KnowledgeGap.status.in_(["new", "researched"]))
                .first()
            )
            if exists is None:
                session.add(
                    KnowledgeGap(
                        question=question, answer=answer, reason=(reason or "")[:40]
                    )
                )
            return exists is None
    except Exception:  # noqa: BLE001
        logger.exception("Could not log knowledge gap")
        return False
