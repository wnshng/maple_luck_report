from __future__ import annotations

import hashlib
import json
import logging
import traceback
import uuid
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import streamlit as st

from src.analytics.posthog_client import track_posthog_event
from src.analytics.storage import analytics_status, get_db_connection
from src.config import ROOT_DIR

logger = logging.getLogger(__name__)

SENSITIVE_KEYS = {
    "api_key",
    "x-nxopen-api-key",
    "authorization",
    "character_name",
    "ocid",
    "raw_payload",
    "response",
    "raw_response",
    "raw_records",
    "request_headers",
    "token",
    "password",
    "secret",
    "key",
    "nexon_api_key",
}


def now_kst_iso() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def hash_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps({"unserializable": True}, ensure_ascii=False)


def sanitize_error_message(message: Any) -> str:
    text = str(message or "")
    for bad in ["api_key", "ocid", "character_name", "x-nxopen-api-key", "authorization"]:
        text = text.replace(bad, "[redacted-key]")
    return text[:2000]


def sanitize_log_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            normalized = str(key).strip().lower()
            if normalized in SENSITIVE_KEYS:
                continue
            if normalized == "selected_character_name":
                continue
            if normalized == "character_hash":
                cleaned[key] = value
                continue
            cleaned[key] = sanitize_log_payload(value)
        return cleaned
    if isinstance(payload, list):
        return [sanitize_log_payload(item) for item in payload[:100]]
    if isinstance(payload, tuple):
        return [sanitize_log_payload(item) for item in payload[:100]]
    return payload


def get_or_create_analytics_identity() -> tuple[str, str]:
    if "anonymous_user_id" not in st.session_state:
        st.session_state["anonymous_user_id"] = str(uuid.uuid4())
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["anonymous_user_id"], st.session_state["session_id"]


def _get_context_metadata() -> dict[str, Any]:
    anonymous_user_id, session_id = get_or_create_analytics_identity()
    return {
        "anonymous_user_id": anonymous_user_id,
        "session_id": session_id,
        "timestamp": now_kst_iso(),
        "app_version": st.session_state.get("app_version", "local"),
    }


def _upsert_session_row() -> None:
    enabled, _ = analytics_status()
    if not enabled:
        return

    metadata = _get_context_metadata()
    anonymous_user_id = metadata["anonymous_user_id"]
    session_id = metadata["session_id"]
    timestamp = metadata["timestamp"]
    app_version = metadata["app_version"]
    user_agent_hash = hash_value(st.context.headers.get("User-Agent")) if hasattr(st, "context") else None
    referrer = None
    if hasattr(st, "context"):
        referrer = st.context.headers.get("Referer")

    with get_db_connection() as connection:
        existing = connection.execute(
            "SELECT first_seen_at FROM analytics_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if existing is None:
            connection.execute(
                """
                INSERT INTO analytics_sessions (
                    session_id, anonymous_user_id, first_seen_at, last_seen_at,
                    session_duration_seconds, user_agent_hash, referrer,
                    app_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    anonymous_user_id,
                    timestamp,
                    timestamp,
                    0.0,
                    user_agent_hash,
                    referrer,
                    app_version,
                    timestamp,
                    timestamp,
                ),
            )
        else:
            first_seen = datetime.fromisoformat(existing["first_seen_at"])
            last_seen = datetime.fromisoformat(timestamp)
            duration = max((last_seen - first_seen).total_seconds(), 0.0)
            connection.execute(
                """
                UPDATE analytics_sessions
                SET anonymous_user_id = ?,
                    last_seen_at = ?,
                    session_duration_seconds = ?,
                    user_agent_hash = COALESCE(?, user_agent_hash),
                    referrer = COALESCE(?, referrer),
                    app_version = ?,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (
                    anonymous_user_id,
                    timestamp,
                    duration,
                    user_agent_hash,
                    referrer,
                    app_version,
                    timestamp,
                    session_id,
                ),
            )
        connection.commit()


def log_event(event_name: str, page_name: str | None = None, properties: dict[str, Any] | None = None) -> None:
    try:
        enabled, _ = analytics_status()
        sanitized = sanitize_log_payload(properties or {})
        try:
            track_posthog_event(event_name, sanitized)
        except Exception:
            pass
        if not enabled:
            return
        _upsert_session_row()
        metadata = _get_context_metadata()
        with get_db_connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO analytics_events (
                    event_id, session_id, anonymous_user_id, event_name,
                    page_name, event_properties_json, timestamp, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    metadata["session_id"],
                    metadata["anonymous_user_id"],
                    event_name,
                    page_name,
                    safe_json_dumps(sanitized),
                    metadata["timestamp"],
                    metadata["timestamp"],
                ),
            )
            connection.commit()
    except Exception as exc:
        logger.warning("Analytics log_event failed: %s", exc)


def log_api_call(
    api_name: str,
    endpoint_name: str,
    status: str,
    response_time_ms: float | None = None,
    error_type: str | None = None,
) -> None:
    try:
        enabled, _ = analytics_status()
        if not enabled:
            return
        _upsert_session_row()
        metadata = _get_context_metadata()
        with get_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO analytics_api_calls (
                    session_id, anonymous_user_id, api_name, endpoint_name, status,
                    response_time_ms, error_type, timestamp, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata["session_id"],
                    metadata["anonymous_user_id"],
                    api_name,
                    endpoint_name,
                    status,
                    response_time_ms,
                    error_type,
                    metadata["timestamp"],
                    metadata["timestamp"],
                ),
            )
            connection.commit()
    except Exception as exc:
        logger.warning("Analytics log_api_call failed: %s", exc)


def log_error(
    error_type: str,
    error_message: Any,
    page_name: str | None = None,
    stack_trace: str | None = None,
) -> None:
    try:
        enabled, _ = analytics_status()
        if not enabled:
            return
        _upsert_session_row()
        metadata = _get_context_metadata()
        sanitized_message = sanitize_error_message(error_message)
        sanitized_trace = sanitize_error_message(stack_trace or traceback.format_exc())
        with get_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO analytics_errors (
                    session_id, anonymous_user_id, error_type, error_message_sanitized,
                    stack_trace_sanitized, page_name, timestamp, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata["session_id"],
                    metadata["anonymous_user_id"],
                    error_type,
                    sanitized_message,
                    sanitized_trace,
                    page_name,
                    metadata["timestamp"],
                    metadata["timestamp"],
                ),
            )
            connection.commit()
        try:
            track_posthog_event(
                "error_occurred",
                {
                    "error_type": error_type,
                    "page_name": page_name,
                },
            )
        except Exception:
            pass
        log_event(
            "error_occurred",
            page_name=page_name,
            properties={"error_type": error_type, "error_message": sanitized_message},
        )
    except Exception as exc:
        logger.warning("Analytics log_error failed: %s", exc)
