from __future__ import annotations


CREATE_TABLE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS analytics_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT UNIQUE,
        session_id TEXT,
        anonymous_user_id TEXT,
        event_name TEXT,
        page_name TEXT,
        event_properties_json TEXT,
        timestamp TEXT,
        created_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS analytics_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT UNIQUE,
        anonymous_user_id TEXT,
        first_seen_at TEXT,
        last_seen_at TEXT,
        session_duration_seconds REAL,
        user_agent_hash TEXT,
        referrer TEXT,
        app_version TEXT,
        created_at TEXT,
        updated_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS analytics_errors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        anonymous_user_id TEXT,
        error_type TEXT,
        error_message_sanitized TEXT,
        stack_trace_sanitized TEXT,
        page_name TEXT,
        timestamp TEXT,
        created_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS analytics_api_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        anonymous_user_id TEXT,
        api_name TEXT,
        endpoint_name TEXT,
        status TEXT,
        response_time_ms REAL,
        error_type TEXT,
        timestamp TEXT,
        created_at TEXT
    )
    """,
]
