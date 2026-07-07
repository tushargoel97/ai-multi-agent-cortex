"""Key-value app settings backed by the `app_settings` table."""

from __future__ import annotations

import logging

from cortex.db.engine import engine, get_session
from cortex.db.models import AppSetting, Base

logger = logging.getLogger(__name__)

_table_ready = False


def _ensure_table() -> None:
    global _table_ready
    if _table_ready:
        return
    Base.metadata.create_all(engine, tables=[AppSetting.__table__])
    _table_ready = True


def get_setting(key: str, default: str = "") -> str:
    try:
        _ensure_table()
        with get_session() as session:
            row = session.get(AppSetting, key)
            return row.value if row is not None else default
    except Exception:  # noqa: BLE001, settings are never worth failing a run
        logger.exception("get_setting(%r) failed", key)
        return default


def set_setting(key: str, value: str) -> None:
    _ensure_table()
    with get_session() as session:
        row = session.get(AppSetting, key)
        if row is None:
            session.add(AppSetting(key=key, value=value))
        else:
            row.value = value
