"""Gap capture for the specialist's self-improvement loop."""

from __future__ import annotations

import logging
import re

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
    "only give you one side",
)

# Product-looking phrases: if the user names one and the answer never mentions
# it, the model silently substituted something else: that's a gap too.
_PRODUCT_RE = re.compile(
    r"\b(?:"
    r"ps5(?:\s+\w+)?|playstation\s*5(?:\s+\w+)?|xbox(?:\s+series)?(?:\s+\w+)?|"
    r"steam\s*deck(?:\s+\w+)?|switch(?:\s+\d+)?|"
    r"rtx\s*\d{3,4}(?:\s*(?:ti|super))?|gtx\s*\d{3,4}(?:\s*ti)?|"
    r"rx\s*\d{3,4}(?:\s*xtx?)?|ryzen\s*\d?\s*\d{4}\w*|"
    r"i[3579]-\d{4,5}\w*|core\s+ultra\s*\d*\s*\d{3}\w*|"
    r"h100|b200"
    r")\b",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    """Collapse whitespace and canonicalize alias forms so 'PlayStation 5 Pro'
    matches 'ps5 pro' and 'Ryzen 7 3700X' matches 'Ryzen 3700X'."""
    text = re.sub(r"[\s\-]+", "", text.lower())
    text = text.replace("playstation5", "ps5")
    return re.sub(r"ryzen[3579](?=\d{4})", "ryzen", text)


def detect_gap(question: str, answer: str) -> str | None:
    """Return the gap reason, or None when the answer looks adequate."""
    low = answer.lower()
    if any(m in low for m in REFUSAL_MARKERS):
        return "refusal"
    norm_answer = _norm(answer)
    for match in _PRODUCT_RE.findall(question):
        if _norm(match) not in norm_answer:
            return "product_not_addressed"
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
                # reason is a short code ("refusal" | "product_not_addressed" |
                # "fact_error"); guard the narrow String(40) column against any
                # overflow so the insert can never silently fail.
                session.add(
                    KnowledgeGap(
                        question=question, answer=answer, reason=(reason or "")[:40]
                    )
                )
            return exists is None
    except Exception:  # noqa: BLE001
        logger.exception("Could not log knowledge gap")
        return False
