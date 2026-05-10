from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.analytics.schema import CREATE_TABLE_STATEMENTS
from src.config import ROOT_DIR

logger = logging.getLogger(__name__)

_ANALYTICS_READY = False
_ANALYTICS_DISABLED_REASON: str | None = None


def _get_config_value(name: str, default: str | None = None) -> str | None:
    try:
        if name in st.secrets:
            value = st.secrets.get(name)
            return str(value) if value is not None else default
    except Exception:
        pass
    load_dotenv(ROOT_DIR / ".env")
    return os.getenv(name, default)


def is_analytics_enabled() -> bool:
    enabled = str(_get_config_value("ENABLE_ANALYTICS", "true")).strip().lower()
    return enabled not in {"0", "false", "no", "off"}


def get_database_url() -> str | None:
    return _get_config_value("DATABASE_URL")


def get_sqlite_db_path() -> Path:
    configured = str(_get_config_value("ANALYTICS_DB_PATH", "analytics.db")).strip() or "analytics.db"
    path = Path(configured)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def analytics_status() -> tuple[bool, str | None]:
    return _ANALYTICS_READY and is_analytics_enabled(), _ANALYTICS_DISABLED_REASON


def init_analytics_db() -> bool:
    global _ANALYTICS_READY, _ANALYTICS_DISABLED_REASON

    if not is_analytics_enabled():
        _ANALYTICS_READY = False
        _ANALYTICS_DISABLED_REASON = "ENABLE_ANALYTICS=false"
        return False

    database_url = get_database_url()
    if database_url and not database_url.startswith("sqlite:///"):
        _ANALYTICS_READY = False
        _ANALYTICS_DISABLED_REASON = "SQLite MVP에서는 sqlite:/// 형식만 지원합니다."
        logger.warning("Analytics DATABASE_URL is set but non-sqlite URLs are not supported in this MVP.")
        return False

    try:
        ensure_tables()
        _ANALYTICS_READY = True
        _ANALYTICS_DISABLED_REASON = None
        return True
    except Exception as exc:
        logger.exception("Failed to initialize analytics DB: %s", exc)
        _ANALYTICS_READY = False
        _ANALYTICS_DISABLED_REASON = str(exc)
        return False


def get_db_connection() -> sqlite3.Connection:
    database_url = get_database_url()
    if database_url and database_url.startswith("sqlite:///"):
        db_path = Path(database_url.replace("sqlite:///", "", 1))
    else:
        db_path = get_sqlite_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def ensure_tables() -> None:
    with get_db_connection() as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        for statement in CREATE_TABLE_STATEMENTS:
            connection.execute(statement)
        connection.commit()
