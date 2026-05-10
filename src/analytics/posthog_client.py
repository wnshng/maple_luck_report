from __future__ import annotations

import os
import uuid
from typing import Any

import streamlit as st

try:
    from posthog import Posthog
except Exception:  # pragma: no cover - optional dependency at runtime
    Posthog = None


SENSITIVE_KEYS = {
    "api_key",
    "nexon_api_key",
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
}


def _get_config_value(name: str, default: str | None = None) -> str | None:
    try:
        if name in st.secrets:
            value = st.secrets.get(name)
            return str(value) if value is not None else default
    except Exception:
        pass
    return os.getenv(name, default)


def _analytics_enabled() -> bool:
    value = _get_config_value("ENABLE_ANALYTICS", "true")
    return str(value).strip().lower() == "true"


@st.cache_resource(show_spinner=False)
def get_posthog_client():
    if not _analytics_enabled() or Posthog is None:
        return None
    api_key = _get_config_value("POSTHOG_API_KEY")
    host = _get_config_value("POSTHOG_HOST", "https://app.posthog.com")
    if not api_key:
        return None
    try:
        return Posthog(project_api_key=api_key, host=host)
    except Exception:
        return None


def get_or_create_anonymous_user_id() -> str:
    if "anonymous_user_id" not in st.session_state:
        st.session_state["anonymous_user_id"] = str(uuid.uuid4())
    return st.session_state["anonymous_user_id"]


def sanitize_properties(properties: dict[str, Any] | None) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    if properties:
        for key, value in properties.items():
            key_text = str(key)
            key_lower = key_text.strip().lower()
            if key_lower in SENSITIVE_KEYS:
                continue
            if any(token in key_lower for token in ["api_key", "token", "secret", "password", "ocid"]):
                continue
            cleaned[key_text] = value
    cleaned["app_version"] = _get_config_value("APP_VERSION", "local")
    return cleaned


def track_posthog_event(event_name: str, properties: dict[str, Any] | None = None) -> None:
    try:
        client = get_posthog_client()
        if client is None:
            return
        client.capture(
            distinct_id=get_or_create_anonymous_user_id(),
            event=event_name,
            properties=sanitize_properties(properties),
        )
    except Exception:
        return
