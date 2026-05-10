from __future__ import annotations

import json
import os
from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics.storage import analytics_status, get_db_connection
from src.config import get_today_kst


def _get_config_value(name: str, default: str | None = None) -> str | None:
    try:
        if name in st.secrets:
            value = st.secrets.get(name)
            return str(value) if value is not None else default
    except Exception:
        pass
    return os.getenv(name, default)


def render_admin_analytics_entry() -> dict[str, bool]:
    admin_password = str(_get_config_value("ADMIN_PASSWORD", "")).strip()
    query_admin = str(st.query_params.get("admin", "0")) == "1"
    state = {"enabled": False, "authorized": False}

    if not query_admin:
        return state

    with st.sidebar:
        with st.expander("관리자 모드", expanded=True):
            if not admin_password:
                st.caption("ADMIN_PASSWORD가 설정되지 않아 관리자 대시보드가 비활성화되어 있습니다.")
                return state

            state["enabled"] = True
            password = st.text_input("관리자 비밀번호", type="password", key="admin_password_input")
            if st.button("관리자 인증", key="admin_auth_button", width="stretch"):
                st.session_state["admin_authenticated"] = password == admin_password
            if st.session_state.get("admin_authenticated", False):
                st.success("관리자 인증이 완료되었습니다.")
                state["authorized"] = True
            elif password:
                st.error("관리자 인증에 실패했습니다.")
    return state


def render_admin_analytics_dashboard() -> None:
    enabled, reason = analytics_status()
    st.subheader("운영 로그 대시보드")
    st.caption("이 화면은 관리자 전용입니다. 로그는 익명화되어 저장되며 API Key, 캐릭터명, ocid 원문은 저장하지 않습니다.")

    if not enabled:
        st.warning(f"운영 로그 기능이 비활성화되어 있습니다. 사유: {reason or '알 수 없음'}")
        return

    today = get_today_kst()
    default_start = today - timedelta(days=6)
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        start_date = st.date_input("로그 시작일", value=default_start, key="admin_log_start")
    with col2:
        end_date = st.date_input("로그 종료일", value=today, key="admin_log_end")
    with col3:
        st.caption("일반 사용자 화면에는 이 로그 데이터가 노출되지 않습니다.")

    if start_date > end_date:
        st.error("관리자 로그 조회 시작일은 종료일보다 늦을 수 없습니다.")
        return

    events_df = _read_sql(
        """
        SELECT * FROM analytics_events
        WHERE date(timestamp) BETWEEN date(?) AND date(?)
        ORDER BY timestamp DESC
        LIMIT 50000
        """,
        (start_date.isoformat(), end_date.isoformat()),
    )
    sessions_df = _read_sql(
        """
        SELECT * FROM analytics_sessions
        WHERE date(last_seen_at) BETWEEN date(?) AND date(?)
        ORDER BY updated_at DESC
        LIMIT 50000
        """,
        (start_date.isoformat(), end_date.isoformat()),
    )
    api_df = _read_sql(
        """
        SELECT * FROM analytics_api_calls
        WHERE date(timestamp) BETWEEN date(?) AND date(?)
        ORDER BY timestamp DESC
        LIMIT 50000
        """,
        (start_date.isoformat(), end_date.isoformat()),
    )
    errors_df = _read_sql(
        """
        SELECT * FROM analytics_errors
        WHERE date(timestamp) BETWEEN date(?) AND date(?)
        ORDER BY timestamp DESC
        LIMIT 50000
        """,
        (start_date.isoformat(), end_date.isoformat()),
    )

    summary_tabs = st.tabs(["요약", "이벤트 로그", "API 로그", "에러 로그"])

    with summary_tabs[0]:
        _render_summary(events_df, sessions_df, api_df, errors_df)
    with summary_tabs[1]:
        st.dataframe(events_df, width="stretch", hide_index=True)
    with summary_tabs[2]:
        st.dataframe(api_df, width="stretch", hide_index=True)
    with summary_tabs[3]:
        st.dataframe(errors_df, width="stretch", hide_index=True)


def _read_sql(query: str, params: tuple) -> pd.DataFrame:
    with get_db_connection() as connection:
        return pd.read_sql_query(query, connection, params=params)


def _render_summary(events_df: pd.DataFrame, sessions_df: pd.DataFrame, api_df: pd.DataFrame, errors_df: pd.DataFrame) -> None:
    total_sessions = len(sessions_df)
    unique_users = sessions_df["anonymous_user_id"].nunique() if not sessions_df.empty else 0
    today_str = get_today_kst().isoformat()
    today_sessions = int((sessions_df["last_seen_at"].fillna("").astype(str).str[:10] == today_str).sum()) if not sessions_df.empty else 0
    today_events = int((events_df["timestamp"].fillna("").astype(str).str[:10] == today_str).sum()) if not events_df.empty else 0
    recent_fetch_success = int((events_df["event_name"] == "data_fetch_success").sum()) if not events_df.empty else 0
    recent_fetch_failed = int((events_df["event_name"] == "data_fetch_failed").sum()) if not events_df.empty else 0
    api_error_count = int((api_df["status"] == "failed").sum()) if not api_df.empty else 0
    avg_api_ms = float(api_df["response_time_ms"].dropna().mean()) if not api_df.empty and api_df["response_time_ms"].dropna().size else 0.0
    avg_session_sec = float(sessions_df["session_duration_seconds"].dropna().mean()) if not sessions_df.empty and sessions_df["session_duration_seconds"].dropna().size else 0.0

    cols = st.columns(5)
    cols[0].metric("총 세션 수", f"{total_sessions:,}")
    cols[1].metric("고유 익명 사용자 수", f"{unique_users:,}")
    cols[2].metric("오늘 이벤트 수", f"{today_events:,}")
    cols[3].metric("오늘 세션 수", f"{today_sessions:,}")
    cols[4].metric("평균 세션 길이", f"{avg_session_sec:.1f}초")

    cols2 = st.columns(4)
    cols2[0].metric("데이터 불러오기 성공", f"{recent_fetch_success:,}")
    cols2[1].metric("데이터 불러오기 실패", f"{recent_fetch_failed:,}")
    cols2[2].metric("API 오류 수", f"{api_error_count:,}")
    cols2[3].metric("평균 API 응답 시간", f"{avg_api_ms:.1f}ms")

    if not sessions_df.empty:
        session_counts = (
            sessions_df.assign(day=sessions_df["last_seen_at"].astype(str).str[:10])
            .groupby("day", as_index=False)
            .size()
            .rename(columns={"size": "session_count"})
        )
        st.plotly_chart(px.bar(session_counts, x="day", y="session_count", title="일자별 세션 수"), width="stretch", key="admin_sessions_by_day")

    if not events_df.empty:
        event_counts = events_df.groupby("event_name", as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False)
        st.plotly_chart(px.bar(event_counts.head(20), x="event_name", y="count", title="이벤트명별 발생 수"), width="stretch", key="admin_events_by_name")

        page_counts = events_df[events_df["event_name"] == "analysis_tab_viewed"].copy()
        if not page_counts.empty:
            tab_counts = page_counts.groupby("page_name", as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False)
            st.plotly_chart(px.bar(tab_counts, x="page_name", y="count", title="탭별 조회 수"), width="stretch", key="admin_tabs_by_count")

        grouping_counts = events_df[events_df["event_name"] == "top5_grouping_changed"]
        if not grouping_counts.empty:
            grouping_props = _expand_event_properties(grouping_counts)
            if "grouping_label" in grouping_props.columns:
                group_summary = (
                    grouping_props.groupby("grouping_label", as_index=False)
                    .size()
                    .rename(columns={"size": "count"})
                    .sort_values("count", ascending=False)
                )
                st.plotly_chart(px.bar(group_summary, x="grouping_label", y="count", title="TOP 5 분석 기준 사용 빈도"), width="stretch", key="admin_grouping_use")
            if "selected_metric" in grouping_props.columns:
                metric_summary = (
                    grouping_props.groupby("selected_metric", as_index=False)
                    .size()
                    .rename(columns={"size": "count"})
                    .sort_values("count", ascending=False)
                )
                st.plotly_chart(px.bar(metric_summary, x="selected_metric", y="count", title="지표 선택 빈도"), width="stretch", key="admin_metric_use")

    if not api_df.empty:
        api_status = api_df.groupby("status", as_index=False).size().rename(columns={"size": "count"})
        st.plotly_chart(px.pie(api_status, names="status", values="count", title="API 성공/실패 비율"), width="stretch", key="admin_api_status")

    if not errors_df.empty:
        error_counts = errors_df.groupby("error_type", as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False)
        st.plotly_chart(px.bar(error_counts, x="error_type", y="count", title="오류 유형별 카운트"), width="stretch", key="admin_error_types")


def _expand_event_properties(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty or "event_properties_json" not in events_df.columns:
        return events_df

    def _load(value: object) -> dict:
        if value is None or value == "":
            return {}
        try:
            parsed = json.loads(str(value))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    props = events_df["event_properties_json"].map(_load)
    props_df = pd.json_normalize(props)
    props_df.index = events_df.index
    return pd.concat([events_df, props_df], axis=1)
