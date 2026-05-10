from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.analytics.admin_dashboard import render_admin_analytics_dashboard, render_admin_analytics_entry
from src.analytics.logger import (
    get_or_create_analytics_identity,
    hash_value,
    log_error,
    log_event,
)
from src.analytics.storage import init_analytics_db, is_analytics_enabled
from src.auth import get_nexon_login_guide
from src.config import (
    PARALLEL_HISTORY_FETCH_WORKERS,
    ROOT_DIR,
    clamp_date_range,
    ensure_data_dirs,
    get_available_date_range,
    get_default_two_year_range,
    get_env_api_key,
    get_today_kst,
    setup_logging,
)
from src.dashboard_analysis import (
    MIN_TOP_ATTEMPTS,
    POTENTIAL_REFERENCE_PATH,
    STARFORCE_REFERENCE_PATH,
    add_confidence,
    attach_event_tags,
    build_group_cols_from_checkboxes,
    build_success_probability_group,
    compare_event_periods,
    confidence_label,
    cube_reference_lookup,
    extract_profile_info,
    format_gap_percent,
    format_percent,
    get_top_conditions_by_grouping,
    group_condition_metrics,
    infer_event_rows_from_records,
    load_reference_csv,
    make_bad_condition_text,
    make_day_of_month_insight_text,
    make_good_condition_text,
    make_hour_insight_text,
    make_weekday_insight_text,
    metric_definition,
    parse_transition_start,
    starforce_destroy_reference_lookup,
    starforce_reference_lookup,
    summarize_cube_by_day_of_month,
    summarize_cube_by_hour,
    summarize_cube_by_hour_band,
    summarize_cube_by_type,
    summarize_cube_by_weekday,
    summarize_starforce_by_day_of_month,
    summarize_starforce_by_hour,
    summarize_starforce_by_hour_band,
    summarize_starforce_by_range,
    summarize_starforce_by_weekday,
    summarize_starforce_by_transition,
)
from src.data_loader import (
    fetch_cube_history_dataframe,
    fetch_potential_history_dataframe,
    fetch_starforce_history_dataframe,
)
from src.effective_options import STAT_OPTIONS, add_effective_option_features, summarize_effective_options
from src.effective_options import (
    get_default_major_options,
    infer_main_stat_by_class,
    normalize_class_name,
)
from src.metrics import summarize_starforce
from src.nexon_client import NexonAPIError, NexonMapleClient
from src.visualizations import (
    plot_cube_type_rate,
    plot_day_of_month_rate,
    plot_hour_band_rate,
    plot_hourly_rate,
    plot_item_rate,
    plot_multi_rate_bar,
    plot_rate_heatmap,
    plot_reference_gap_bar,
    plot_starforce_stage_rate,
    plot_weekday_rate,
    set_plotly_theme_mode,
)


NOTICE_TEXT = """
- 운빨 리포트는 큐브·잠재능력·스타포스 기록을 과거 관측 기준으로 정리한 참고용 분석입니다.
- 날짜별 분석은 특정 하루가 아니라 기준기간 동안 같은 일자를 묶은 분석입니다. 예: 2월 9일, 3월 9일, 4월 9일은 모두 9일로 집계합니다.
- 시간별 분석은 0시~23시 기준, 요일별 분석은 월요일~일요일 기준으로 비교합니다.
- 조건 조합 분석은 일자/요일/시간/큐브 타입/스타포스 구간/전이 구간을 조합한 과거 기록 비교입니다.
- TOP 5는 미래 추천이 아니라 과거 기록상 결과가 좋게 또는 아쉽게 관측된 조건을 정리한 참고용 통계입니다.
- 신뢰도는 표본 수 기반 참고 강도이며, 향후 결과를 보장하지 않습니다.
- API Key는 저장하지 않으며, 사용자의 세션에서만 사용합니다.
- 서비스 개선과 오류 분석을 위해 익명 사용 로그가 수집될 수 있습니다. API Key 원문, 캐릭터명 원문, ocid 원문, 큐브/스타포스 상세 원본 기록은 로그에 저장하지 않습니다.
- Nexon Open API 기반 데이터는 정책상 30일 이내 갱신 의무가 있을 수 있으므로 배포 시 최신 정책을 확인해야 합니다.
"""

STARFORCE_NOTICE = """
- 스타포스 강화 결과는 2023-12-27 이후 데이터부터 조회 가능하며 최대 2년 범위에서 조회됩니다.
- 스타포스 확률 정보는 최대 5분 후 확인 가능하므로 방금 강화한 기록은 즉시 반영되지 않을 수 있습니다.
"""

WEEKDAY_ORDER = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
TIMEBLOCK_ORDER = ["새벽", "오전", "오후", "저녁"]
DAY_OF_MONTH_ORDER = [f"{day}일" for day in range(1, 32)]
HOUR_LABEL_ORDER = [f"{hour}시" for hour in range(24)]
PERSISTED_APP_STATE_PATH = ROOT_DIR / ".app_state" / "last_state.pkl"
PERSISTED_APP_STATE_KEYS = [
    "cube_df",
    "potential_df",
    "starforce_df",
    "characters_df",
    "character_basic_by_ocid",
    "last_sync_at",
    "last_query_range",
    "selected_character_option",
    "selected_major_options",
    "use_auto_major_option",
    "last_major_option_character_key",
    "query_start_date",
    "query_end_date",
    "theme_mode",
]

LIGHT_THEME_PALETTE = {
    "page_bg": "#F8FBFF",
    "sidebar_bg": "#F6F9FD",
    "card_bg": "rgba(255,255,255,0.90)",
    "card_bg_soft": "rgba(255,255,255,0.82)",
    "border": "rgba(49,70,101,0.14)",
    "border_soft": "rgba(49,70,101,0.10)",
    "text_primary": "#102133",
    "text_secondary": "#516173",
    "text_muted": "#6B7C93",
    "accent": "#F97316",
    "accent_hover": "#EA580C",
    "success": "#00A676",
    "warning": "#A65E00",
    "danger": "#D14343",
    "info": "#0EA5E9",
    "chip_bg": "#FFF3E8",
    "chip_border": "#F8D7BB",
    "input_bg": "#FFFFFF",
    "input_text": "#102133",
    "metric_bg": "rgba(255,255,255,0.82)",
}

DARK_THEME_PALETTE = {
    "page_bg": "#0F172A",
    "sidebar_bg": "#111827",
    "card_bg": "#1E293B",
    "card_bg_soft": "#243044",
    "border": "#334155",
    "border_soft": "#475569",
    "text_primary": "#F8FAFC",
    "text_secondary": "#CBD5E1",
    "text_muted": "#94A3B8",
    "accent": "#FB923C",
    "accent_hover": "#F97316",
    "success": "#34D399",
    "warning": "#FBBF24",
    "danger": "#F87171",
    "info": "#FDBA74",
    "chip_bg": "#1B2333",
    "chip_border": "#5B4A33",
    "input_bg": "#1E293B",
    "input_text": "#F8FAFC",
    "metric_bg": "#1E293B",
}


def main() -> None:
    setup_logging()
    ensure_data_dirs()

    st.set_page_config(
        page_title="메이플 운빨 리포트",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()
    _bootstrap_analytics()

    st.title("메이플 운빨 리포트")
    st.caption("최근 2년 큐브·잠재능력·스타포스 기록으로 내 운빨 패턴을 확인해보세요.")

    controls = _render_sidebar()
    set_plotly_theme_mode(controls["resolved_theme_mode"])
    _inject_style(controls["resolved_theme_mode"])
    _inject_external_analytics_scripts()
    _handle_api_load(controls)

    context = _build_context(controls)
    _render_profile_header(context)

    admin_state = render_admin_analytics_entry()

    tab_labels = [
        "종합 요약",
        "일자별",
        "시간별",
        "요일별",
        "조건 TOP 5",
        "원본 데이터",
        "API 디버그",
    ]
    if admin_state.get("authorized"):
        tab_labels.append("운영 로그")

    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _render_overview_tab(context)
    with tabs[1]:
        _render_day_of_month_tab(context)
    with tabs[2]:
        _render_hour_tab(context)
    with tabs[3]:
        _render_weekday_tab(context)
    with tabs[4]:
        _render_condition_tab(context)
    with tabs[5]:
        _render_raw_tab(context)
    with tabs[6]:
        _render_debug_tab(context, controls)
    if admin_state.get("authorized"):
        with tabs[7]:
            render_admin_analytics_dashboard()

    st.divider()
    st.markdown(NOTICE_TEXT)


def _init_state() -> None:
    default_start, default_end = get_default_two_year_range()
    defaults = {
        "cube_df": pd.DataFrame(),
        "potential_df": pd.DataFrame(),
        "starforce_df": pd.DataFrame(),
        "characters_df": pd.DataFrame(),
        "character_basic_by_ocid": {},
        "api_debug": {},
        "last_sync_at": None,
        "last_query_range": None,
        "selected_character_option": None,
        "selected_character_object": None,
        "selected_major_options": [],
        "use_auto_major_option": True,
        "last_major_option_character_key": None,
        "default_start_date": default_start,
        "default_end_date": default_end,
        "manual_character_name": "",
        "theme_mode": "light",
        "app_version": os.getenv("APP_VERSION", "local").strip() or "local",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if not st.session_state.get("_persisted_state_restored"):
        _restore_persisted_app_state()
        st.session_state["_persisted_state_restored"] = True


def _sanitize_df_for_persistence(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    output = df.copy()
    if "raw_payload" in output.columns:
        output = output.drop(columns=["raw_payload"])
    return output


def _sanitize_basic_cache_for_persistence(cache: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    if not isinstance(cache, dict):
        return sanitized
    for ocid, row in cache.items():
        if not isinstance(row, dict):
            continue
        sanitized[ocid] = {key: value for key, value in row.items() if key != "raw_payload"}
    return sanitized


def _restore_persisted_app_state() -> None:
    try:
        if not PERSISTED_APP_STATE_PATH.exists():
            return
        with PERSISTED_APP_STATE_PATH.open("rb") as file:
            payload = pickle.load(file)
        if not isinstance(payload, dict):
            return
        for key in PERSISTED_APP_STATE_KEYS:
            if key in payload:
                st.session_state[key] = payload[key]
        if any(
            isinstance(st.session_state.get(name), pd.DataFrame) and not st.session_state.get(name).empty
            for name in ["cube_df", "potential_df", "starforce_df", "characters_df"]
        ):
            st.session_state["_restored_from_local_state"] = True
    except Exception:
        return


def _persist_app_state() -> None:
    try:
        payload: dict[str, Any] = {}
        for key in PERSISTED_APP_STATE_KEYS:
            value = st.session_state.get(key)
            if key in {"cube_df", "potential_df", "starforce_df", "characters_df"}:
                payload[key] = _sanitize_df_for_persistence(value)
            elif key == "character_basic_by_ocid":
                payload[key] = _sanitize_basic_cache_for_persistence(value or {})
            else:
                payload[key] = value
        PERSISTED_APP_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PERSISTED_APP_STATE_PATH.open("wb") as file:
            pickle.dump(payload, file)
    except Exception:
        return


def _bootstrap_analytics() -> None:
    init_analytics_db()
    get_or_create_analytics_identity()
    if not st.session_state.get("_analytics_session_started_logged"):
        log_event("session_start", page_name="app")
        st.session_state["_analytics_session_started_logged"] = True
    if not st.session_state.get("_analytics_app_loaded_logged"):
        log_event("app_loaded", page_name="app", properties={"app_version": st.session_state.get("app_version", "local")})
        st.session_state["_analytics_app_loaded_logged"] = True


def _inject_external_analytics_scripts() -> None:
    if st.session_state.get("_external_analytics_injected"):
        return

    ga4_measurement_id = os.getenv("GA4_MEASUREMENT_ID", "").strip()
    hotjar_site_id = os.getenv("HOTJAR_SITE_ID", "").strip()
    hotjar_snippet_version = os.getenv("HOTJAR_SNIPPET_VERSION", "6").strip() or "6"

    script_parts: list[str] = []
    if ga4_measurement_id:
        script_parts.append(
            f"""
            <script async src="https://www.googletagmanager.com/gtag/js?id={ga4_measurement_id}"></script>
            <script>
              window.dataLayer = window.dataLayer || [];
              function gtag(){{dataLayer.push(arguments);}}
              gtag('js', new Date());
              gtag('config', '{ga4_measurement_id}');
            </script>
            """
        )
    if hotjar_site_id:
        script_parts.append(
            f"""
            <script>
              (function(h,o,t,j,a,r){{
                  h.hj=h.hj||function(){{(h.hj.q=h.hj.q||[]).push(arguments)}};
                  h._hjSettings={{hjid:{hotjar_site_id},hjsv:{hotjar_snippet_version}}};
                  a=o.getElementsByTagName('head')[0];
                  r=o.createElement('script');r.async=1;
                  r.src=t+h._hjSettings.hjid+j+h._hjSettings.hjsv;
                  a.appendChild(r);
              }})(window,document,'https://static.hotjar.com/c/hotjar-','.js?sv=');
            </script>
            """
        )

    if not script_parts:
        return

    components.html("".join(script_parts), height=0, width=0)
    st.session_state["_external_analytics_injected"] = True


def _character_level_bucket(level: Any) -> str:
    numeric = pd.to_numeric(level, errors="coerce")
    if pd.isna(numeric):
        return "unknown"
    base = int(numeric // 10 * 10)
    return f"{base}-{base + 9}"


def _log_tab_view_once(tab_name: str) -> None:
    viewed = set(st.session_state.get("_analytics_viewed_tabs", []))
    if tab_name in viewed:
        return
    log_event("analysis_tab_viewed", page_name=tab_name, properties={"tab_name": tab_name})
    viewed.add(tab_name)
    st.session_state["_analytics_viewed_tabs"] = sorted(viewed)


def get_theme_palette(theme_mode: str) -> dict[str, str]:
    return DARK_THEME_PALETTE if theme_mode == "dark" else LIGHT_THEME_PALETTE


def _resolve_theme_mode(theme_mode: str) -> str:
    if theme_mode == "system":
        return "dark" if st.get_option("theme.base") == "dark" else "light"
    return theme_mode


def render_theme_toggle() -> tuple[str, str]:
    selected = st.selectbox(
        "화면 테마",
        ["라이트", "다크", "시스템"],
        index={"light": 0, "dark": 1, "system": 2}.get(st.session_state.get("theme_mode", "light"), 0),
        key="theme_mode_select",
    )
    theme_mode = {"라이트": "light", "다크": "dark", "시스템": "system"}[selected]
    st.session_state["theme_mode"] = theme_mode
    return theme_mode, _resolve_theme_mode(theme_mode)


def fetch_character_list(api_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    client = NexonMapleClient(api_key)
    payload = client.get_character_list()
    return payload, dict(client.last_debug_info)


def flatten_account_character_list(raw_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for account in raw_payload.get("account_list", []) or []:
        account_id = account.get("account_id")
        for character in account.get("character_list", []) or []:
            rows.append(
                {
                    "account_id": account_id,
                    "ocid": character.get("ocid"),
                    "character_name": character.get("character_name"),
                    "world_name": character.get("world_name"),
                    "character_class": character.get("character_class"),
                    "character_level": character.get("character_level"),
                    "raw_payload": character,
                }
            )
    return rows


def normalize_character_list(raw_payload: dict[str, Any]) -> pd.DataFrame:
    rows = flatten_account_character_list(raw_payload)
    columns = [
        "account_id",
        "ocid",
        "character_name",
        "world_name",
        "character_class",
        "character_level",
        "raw_payload",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df["character_level"] = pd.to_numeric(df["character_level"], errors="coerce")
    df["character_name"] = df["character_name"].fillna("").astype(str).str.strip()
    df["world_name"] = df["world_name"].fillna("").astype(str).str.strip()
    df["character_class"] = df["character_class"].fillna("").astype(str).str.strip()
    df["option_key"] = df.apply(_character_option_key, axis=1)
    df["option_label"] = df.apply(_character_option_label, axis=1)
    return df


def fetch_character_ocid(api_key: str, character_name: str) -> tuple[str, dict[str, Any]]:
    client = NexonMapleClient(api_key)
    ocid = client.get_character_id(character_name)
    return ocid, dict(client.last_debug_info)


def fetch_character_basic(api_key: str, ocid: str, query_date: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    client = NexonMapleClient(api_key)
    payload = client.get_character_basic(ocid, query_date=query_date)
    return payload, dict(client.last_debug_info)


def normalize_character_basic(raw_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "ocid": raw_payload.get("ocid"),
        "date": raw_payload.get("date"),
        "character_name": raw_payload.get("character_name"),
        "world_name": raw_payload.get("world_name"),
        "character_gender": raw_payload.get("character_gender"),
        "character_class": raw_payload.get("character_class"),
        "character_class_level": raw_payload.get("character_class_level"),
        "character_level": raw_payload.get("character_level"),
        "character_exp": raw_payload.get("character_exp"),
        "character_exp_rate": raw_payload.get("character_exp_rate"),
        "character_guild_name": raw_payload.get("character_guild_name"),
        "character_image": raw_payload.get("character_image"),
        "character_date_create": raw_payload.get("character_date_create"),
        "date_last_login": raw_payload.get("date_last_login"),
        "date_last_logout": raw_payload.get("date_last_logout"),
        "access_flag": raw_payload.get("access_flag"),
        "liberation_quest_clear": raw_payload.get("liberation_quest_clear"),
        "raw_payload": raw_payload,
    }


def sync_character_by_name(api_key: str, character_name: str) -> pd.DataFrame:
    ocid, _ = fetch_character_ocid(api_key, character_name)
    raw_basic, _ = fetch_character_basic(api_key, ocid)
    basic = normalize_character_basic(raw_basic)
    row = {
        "account_id": None,
        "ocid": basic.get("ocid") or ocid,
        "character_name": basic.get("character_name") or character_name,
        "world_name": basic.get("world_name"),
        "character_class": basic.get("character_class"),
        "character_level": basic.get("character_level"),
        "raw_payload": basic.get("raw_payload"),
        **basic,
    }
    df = pd.DataFrame([row])
    df["character_level"] = pd.to_numeric(df["character_level"], errors="coerce")
    df["option_key"] = df.apply(_character_option_key, axis=1)
    df["option_label"] = df.apply(_character_option_label, axis=1)
    return df


def sync_multiple_characters(api_key: str, character_names: list[str]) -> pd.DataFrame:
    frames = []
    for character_name in character_names:
        name = str(character_name).strip()
        if not name:
            continue
        frames.append(sync_character_by_name(api_key, name))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def get_default_character_by_level(characters_df: pd.DataFrame) -> pd.Series | None:
    if characters_df is None or characters_df.empty:
        return None
    working = characters_df.copy()
    if "character_level" in working.columns:
        working["character_level"] = pd.to_numeric(working["character_level"], errors="coerce")
    else:
        working["character_level"] = np.nan
    working["world_name"] = working.get("world_name", pd.Series("", index=working.index)).fillna("").astype(str)
    working["character_name"] = working.get("character_name", pd.Series("", index=working.index)).fillna("").astype(str)
    ranked = working.sort_values(
        ["character_level", "world_name", "character_name"],
        ascending=[False, True, True],
        na_position="last",
    )
    if ranked.empty:
        return None
    return ranked.iloc[0]


def render_character_selector(characters_df: pd.DataFrame) -> dict[str, Any] | None:
    if characters_df is None or characters_df.empty:
        return None
    option_labels = characters_df["option_label"].tolist()
    selected_label = st.session_state.get("selected_character_option")
    if selected_label not in option_labels:
        default_row = get_default_character_by_level(characters_df)
        if default_row is not None:
            selected_label = str(default_row["option_label"])
            st.session_state["selected_character_option"] = selected_label
    selected_label = st.selectbox(
        "분석 캐릭터",
        option_labels,
        index=option_labels.index(selected_label) if selected_label in option_labels else 0,
        key="selected_character_option",
    )
    row = characters_df[characters_df["option_label"] == selected_label]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def _character_option_key(row: pd.Series) -> str:
    ocid = str(row.get("ocid") or "").strip()
    if ocid:
        return ocid
    return f"{row.get('character_name', '')}|{row.get('world_name', '')}|{row.get('character_class', '')}"


def _character_option_label(row: pd.Series) -> str:
    level = row.get("character_level")
    level_text = f"Lv.{int(level)}" if pd.notna(level) else "Lv.?"
    return " | ".join(
        [
            str(row.get("character_name") or "이름 미확인"),
            str(row.get("world_name") or "월드 미확인"),
            str(row.get("character_class") or "직업 미확인"),
            level_text,
        ]
    )


def _merge_character_rows(base_row: dict[str, Any] | None, basic_row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not base_row and not basic_row:
        return None
    merged = dict(base_row or {})
    for key, value in (basic_row or {}).items():
        if value is None or value == "":
            continue
        merged[key] = value
    return merged


def _sync_account_character_list(api_key: str) -> pd.DataFrame:
    payload, debug_info = fetch_character_list(api_key)
    characters_df = normalize_character_list(payload)
    st.session_state["characters_df"] = characters_df
    st.session_state["character_basic_by_ocid"] = {}
    st.session_state["api_debug"]["character_list"] = {
        **debug_info,
        "record_count": len(characters_df),
        "raw_response_preview": payload,
    }
    if characters_df.empty:
        raise RuntimeError("계정 캐릭터 목록이 비어 있습니다. API Key와 연동 계정을 확인해 주세요.")

    default_row = get_default_character_by_level(characters_df)
    if default_row is not None:
        st.session_state["selected_character_option"] = str(default_row["option_label"])
        log_event(
            "auto_character_selected_by_level",
            page_name="sidebar",
            properties={
                "character_class": default_row.get("character_class"),
                "world_name": default_row.get("world_name"),
                "character_level_bucket": _character_level_bucket(default_row.get("character_level")),
            },
        )
    _persist_app_state()
    return characters_df


def _sync_character_from_name(api_key: str, character_name: str) -> pd.DataFrame:
    ocid, id_debug = fetch_character_ocid(api_key, character_name)
    raw_basic, basic_debug = fetch_character_basic(api_key, ocid)
    basic = normalize_character_basic(raw_basic)
    st.session_state["api_debug"]["character_id"] = id_debug
    st.session_state["api_debug"]["character_basic"] = basic_debug
    row = {
        "account_id": None,
        "ocid": basic.get("ocid") or ocid,
        "character_name": basic.get("character_name") or character_name,
        "world_name": basic.get("world_name"),
        "character_class": basic.get("character_class"),
        "character_level": basic.get("character_level"),
        "raw_payload": basic.get("raw_payload"),
        **basic,
    }
    character_df = pd.DataFrame([row])
    character_df["character_level"] = pd.to_numeric(character_df["character_level"], errors="coerce")
    character_df["option_key"] = character_df.apply(_character_option_key, axis=1)
    character_df["option_label"] = character_df.apply(_character_option_label, axis=1)
    if character_df.empty:
        raise RuntimeError("캐릭터명으로 조회된 기본 정보가 없습니다.")

    existing = st.session_state.get("characters_df", pd.DataFrame())
    combined = pd.concat([existing, character_df], ignore_index=True, sort=False) if isinstance(existing, pd.DataFrame) and not existing.empty else character_df
    if "ocid" in combined.columns:
        combined = combined.drop_duplicates(subset=["ocid"], keep="first")
    st.session_state["characters_df"] = combined
    st.session_state["selected_character_option"] = str(character_df.iloc[0]["option_label"])
    _persist_app_state()
    return combined


def _get_selected_character_with_basic(api_key: str, selected_character: dict[str, Any] | None) -> dict[str, Any] | None:
    if not selected_character:
        return None
    ocid = str(selected_character.get("ocid") or "").strip()
    if not ocid or not api_key.strip():
        return selected_character

    basic_cache = dict(st.session_state.get("character_basic_by_ocid", {}))
    if ocid not in basic_cache:
        try:
            raw_payload, debug_info = fetch_character_basic(api_key, ocid)
            basic_cache[ocid] = normalize_character_basic(raw_payload)
            st.session_state["api_debug"]["character_basic"] = {
                **debug_info,
                "selected_ocid_exists": True,
            }
        except Exception as exc:
            st.session_state["api_debug"]["character_basic"] = {
                "error": str(exc),
                "selected_ocid_exists": True,
            }
            log_error("character_basic_fetch_failed", exc, page_name="sidebar")
            st.sidebar.info("캐릭터 기본 정보 조회에 실패했지만, 목록 정보 기준으로 분석을 계속합니다.")
            basic_cache[ocid] = {}
        st.session_state["character_basic_by_ocid"] = basic_cache
        _persist_app_state()

    return _merge_character_rows(selected_character, basic_cache.get(ocid))


def update_major_options_on_character_change(selected_character: dict[str, Any] | None) -> str | None:
    if not selected_character:
        return None
    character_key = str(selected_character.get("option_key") or selected_character.get("ocid") or selected_character.get("character_name") or "")
    character_class = str(selected_character.get("character_class") or "").strip()
    inferred_main_stat = infer_main_stat_by_class(character_class)
    if not st.session_state.get("use_auto_major_option", True):
        st.session_state["last_major_option_character_key"] = character_key
        return inferred_main_stat

    if st.session_state.get("last_major_option_character_key") == character_key:
        return inferred_main_stat

    default_options = get_default_major_options(character_class)
    if default_options:
        st.session_state["selected_major_options"] = default_options
        log_event(
            "auto_major_option_applied",
            page_name="sidebar",
            properties={
                "character_class": character_class,
                "inferred_main_stat": inferred_main_stat,
                "selected_option_count": len(default_options),
            },
        )
    st.session_state["last_major_option_character_key"] = character_key
    return inferred_main_stat


def _render_sidebar_character_profile(selected_character: dict[str, Any] | None) -> None:
    if not selected_character:
        st.caption("캐릭터 목록을 불러오면 레벨이 가장 높은 캐릭터가 기본 분석 캐릭터로 선택됩니다.")
        return

    image_url = selected_character.get("character_image")
    if image_url:
        st.image(image_url, width=84)
    st.markdown(f"**{selected_character.get('character_name') or '이름 미확인'}**")
    st.caption(
        " · ".join(
            [
                str(selected_character.get("world_name") or "월드 미확인"),
                str(selected_character.get("character_class") or "직업 미확인"),
                f"Lv.{int(selected_character['character_level'])}" if pd.notna(selected_character.get("character_level")) else "Lv.?",
            ]
        )
    )
    if selected_character.get("character_guild_name"):
        st.caption(f"길드: {selected_character.get('character_guild_name')}")


def _render_sidebar() -> dict[str, Any]:
    env_api_key = get_env_api_key()
    login_guide = get_nexon_login_guide()
    default_start, default_end = get_default_two_year_range()
    if st.session_state.get("default_start_date") != default_start:
        st.session_state["default_start_date"] = default_start
    if st.session_state.get("default_end_date") != default_end:
        st.session_state["default_end_date"] = default_end

    selected_character: dict[str, Any] | None = None
    inferred_main_stat: str | None = None

    with st.sidebar:
        st.header("데이터 불러오기")
        theme_mode, resolved_theme_mode = render_theme_toggle()
        st.session_state["resolved_theme_mode"] = resolved_theme_mode
        auth_method = st.radio(
            "인증 방식",
            ["API Key 직접 입력", "넥슨 게임 데이터 활용 로그인"],
            index=0,
        )

        api_key = ""
        if auth_method == "API Key 직접 입력":
            api_key = st.text_input(
                "Nexon Open API Key",
                value=env_api_key,
                type="password",
                help=".env의 NEXON_OPEN_API_KEY도 지원하지만 기본은 이 입력값입니다.",
            )
            with st.expander("Open API Key 발급 방법 / 사용법", expanded=False):
                st.markdown(
                    """
1. [넥슨 Open API 사전 준비하기](https://openapi.nexon.com/ko/guide/prepare-in-advance/)에서 넥슨 ID 로그인과 애플리케이션 등록 절차를 확인합니다.
2. 로그인 후 **내 애플리케이션 > 애플리케이션 등록**에서 `메이플스토리`를 선택하고 앱을 등록합니다.
3. 등록이 완료되면 **내 애플리케이션 > 애플리케이션 상세**에서 API Key를 확인할 수 있습니다.
4. 아래 입력칸에 발급받은 Key를 붙여 넣고, `내 캐릭터 목록 불러오기`를 눌러 캐릭터를 불러옵니다.
5. 이후 기간을 확인한 뒤 `전체 기록 불러오기`를 누르면 잠재능력/큐브와 스타포스 기록을 함께 조회합니다.

- 공식 안내: [API 사용하기](https://openapi.nexon.com/ko/guide/request-api/)
- API 요청 시 Header 이름은 `x-nxopen-api-key` 입니다.
- 개발 단계 API Key는 호출량이 제한될 수 있어, 긴 기간 조회에서는 시간이 더 걸릴 수 있습니다.
"""
                )
        else:
            st.info(login_guide.message)

        st.subheader("캐릭터 선택")
        load_characters_clicked = st.button("내 캐릭터 목록 불러오기", width="stretch")
        if load_characters_clicked:
            log_event("character_list_fetch_started", page_name="sidebar")
            if auth_method != "API Key 직접 입력":
                st.warning("현재는 API Key 직접 입력 방식에서만 캐릭터 목록을 불러올 수 있습니다.")
                log_event("character_list_fetch_failed", page_name="sidebar", properties={"error_type": "unsupported_auth_method"})
            elif not api_key.strip():
                st.warning("API Key를 입력한 뒤 캐릭터 목록을 불러와 주세요.")
                log_event("api_key_validation_failed", page_name="sidebar", properties={"reason": "missing_api_key"})
            else:
                try:
                    characters_df = _sync_account_character_list(api_key)
                    st.success(f"계정 캐릭터 {len(characters_df)}명을 불러왔습니다.")
                    log_event(
                        "character_list_fetch_success",
                        page_name="sidebar",
                        properties={"character_count": len(characters_df)},
                    )
                    log_event("api_key_validation_success", page_name="sidebar")
                except Exception as exc:
                    st.error(f"캐릭터 목록을 불러오지 못했습니다: {exc}")
                    log_event(
                        "character_list_fetch_failed",
                        page_name="sidebar",
                        properties={"error_type": type(exc).__name__},
                    )
                    log_error("character_list_fetch_failed", exc, page_name="sidebar")

        with st.expander("캐릭터명 직접 입력 조회", expanded=False):
            manual_character_name = st.text_input("캐릭터명", key="manual_character_name")
            manual_sync_clicked = st.button("캐릭터명으로 추가", key="manual_character_sync", width="stretch")
            if manual_sync_clicked:
                if not api_key.strip():
                    st.warning("API Key를 입력한 뒤 캐릭터명 조회를 실행해 주세요.")
                    log_event("api_key_validation_failed", page_name="manual_character_lookup", properties={"reason": "missing_api_key"})
                elif not manual_character_name.strip():
                    st.warning("캐릭터명을 입력해 주세요.")
                else:
                    try:
                        log_event("character_search_submitted", page_name="manual_character_lookup")
                        _sync_character_from_name(api_key, manual_character_name)
                        st.success("캐릭터 정보를 목록에 추가했습니다.")
                        log_event("character_sync_success", page_name="manual_character_lookup")
                    except Exception as exc:
                        st.error(f"캐릭터명 직접 조회에 실패했습니다: {exc}")
                        log_event("character_sync_failed", page_name="manual_character_lookup", properties={"error_type": type(exc).__name__})
                        log_error("character_sync_failed", exc, page_name="manual_character_lookup")

        characters_df = st.session_state.get("characters_df", pd.DataFrame())
        if isinstance(characters_df, pd.DataFrame) and not characters_df.empty:
            selected_character = render_character_selector(characters_df)
            selected_character = _get_selected_character_with_basic(api_key, selected_character)
        st.session_state["selected_character_object"] = selected_character
        if selected_character:
            selected_key = str(selected_character.get("option_key") or selected_character.get("ocid") or selected_character.get("character_name") or "")
            if st.session_state.get("_analytics_last_selected_character_key") != selected_key:
                log_event(
                    "character_selected",
                    page_name="sidebar",
                    properties={
                        "character_class": selected_character.get("character_class"),
                        "world_name": selected_character.get("world_name"),
                        "character_level_bucket": _character_level_bucket(selected_character.get("character_level")),
                    },
                )
                st.session_state["_analytics_last_selected_character_key"] = selected_key
        _render_sidebar_character_profile(selected_character)

        st.divider()
        st.subheader("옵션 기준")
        st.checkbox("직업 기준 주요옵션 자동 설정", key="use_auto_major_option")
        inferred_main_stat = update_major_options_on_character_change(selected_character)
        if selected_character:
            st.caption(f"직업: {selected_character.get('character_class') or '미확인'}")
        if inferred_main_stat:
            st.caption(f"자동 추론 주스탯: {inferred_main_stat}")
        elif selected_character:
            st.caption("직업별 주요옵션 기본값을 찾지 못해 직접 선택이 필요합니다.")

        selected_stats = st.multiselect(
            "주요옵션 기준",
            STAT_OPTIONS,
            key="selected_major_options",
        )
        current_major_signature = (
            tuple(sorted(selected_stats)),
            bool(st.session_state.get("use_auto_major_option", True)),
            str(inferred_main_stat or ""),
            str(selected_character.get("character_class") if selected_character else ""),
        )
        if st.session_state.get("_analytics_last_major_signature") != current_major_signature:
            log_event(
                "major_option_changed",
                page_name="sidebar",
                properties={
                    "auto_major_option_enabled": bool(st.session_state.get("use_auto_major_option", True)),
                    "selected_option_count": len(selected_stats),
                    "inferred_main_stat": inferred_main_stat,
                    "character_class": selected_character.get("character_class") if selected_character else None,
                },
            )
            st.session_state["_analytics_last_major_signature"] = current_major_signature
        st.caption("주요옵션은 선택한 옵션이 1줄 이상, 유효옵션은 2줄 이상 나온 경우로 계산합니다.")

        st.divider()
        available_start, available_end = get_default_two_year_range()
        if st.session_state.get("_restored_from_local_state"):
            st.caption("최근 불러온 데이터와 선택 상태를 이 브라우저에서 자동 복원했습니다.")
        st.caption("기본값은 오늘 기준 최근 2년입니다.")
        st.caption(f"현재 조회 가능 기간: {available_start} ~ {available_end}")

        current_start = st.session_state.get("query_start_date", default_start)
        current_end = st.session_state.get("query_end_date", default_end)
        current_start = max(current_start, available_start)
        current_end = min(current_end, available_end)
        if current_start > current_end:
            current_start, current_end = available_start, available_end

        start_date = st.date_input(
            "시작일",
            value=current_start,
            min_value=available_start,
            max_value=available_end,
            key="query_start_date",
        )
        end_date = st.date_input(
            "종료일",
            value=current_end,
            min_value=available_start,
            max_value=available_end,
            key="query_end_date",
        )
        auto_clamp = st.checkbox("기간 자동 보정 후 조회", value=True)
        if (end_date - start_date).days + 1 > 730:
            st.warning("기간이 길수록 API 호출이 오래 걸릴 수 있습니다.")
        st.caption("전체 기록을 불러오는 중입니다. 기간이 길수록 시간이 걸릴 수 있습니다.")
        load_clicked = st.button("전체 기록 불러오기", type="primary", width="stretch")

        st.divider()
        st.subheader("TOP 조건 설정")
        top_min_attempts = st.slider("TOP 조건 최소 시도 수", min_value=1, max_value=100, value=MIN_TOP_ATTEMPTS)
        top_dedup_strength = st.selectbox("TOP 조건 중복 제거 강도", ["약함", "보통", "강함"], index=1)
        top_score_basis = st.selectbox("TOP 조건 기준", ["전체 평균 대비", "보정률 기준"], index=1)
        low_sample_display = st.radio("표본 부족 항목 그래프 표시", ["표시", "숨김"], index=0)

        show_raw_response = st.checkbox("원본 응답 보기", value=False)

    _persist_app_state()
    return {
        "api_key": api_key,
        "auth_method": auth_method,
        "start_date": start_date,
        "end_date": end_date,
        "auto_clamp": auto_clamp,
        "selected_character": selected_character,
        "selected_character_name": selected_character.get("character_name") if selected_character else "전체",
        "selected_character_ocid": selected_character.get("ocid") if selected_character else None,
        "selected_character_world": selected_character.get("world_name") if selected_character else None,
        "job_name": selected_character.get("character_class", "") if selected_character else "",
        "selected_stats": selected_stats,
        "top_min_attempts": top_min_attempts,
        "top_dedup_strength": top_dedup_strength,
        "top_score_basis": top_score_basis,
        "show_low_sample": low_sample_display == "표시",
        "show_raw_response": show_raw_response,
        "load_clicked": load_clicked,
        "use_auto_major_option": st.session_state.get("use_auto_major_option", True),
        "inferred_main_stat": inferred_main_stat,
        "theme_mode": theme_mode,
        "resolved_theme_mode": resolved_theme_mode,
    }


def _handle_api_load(controls: dict[str, Any]) -> None:
    if not controls["load_clicked"]:
        return
    if controls["auth_method"] != "API Key 직접 입력":
        st.sidebar.warning("넥슨 게임 데이터 활용 로그인 방식은 추후 OAuth/동의 기반 연동으로 확장 예정입니다.")
        return
    if not controls["api_key"].strip():
        st.sidebar.warning("API Key를 입력해 주세요.")
        log_event("api_key_validation_failed", page_name="data_fetch", properties={"reason": "missing_api_key"})
        return

    fetch_started_at = time.perf_counter()
    selected_types = ["cube", "potential", "starforce"]
    date_range_days = (controls["end_date"] - controls["start_date"]).days + 1
    log_event(
        "data_fetch_started",
        page_name="sidebar",
        properties={"target_type": "all", "date_range_days": date_range_days},
    )

    loading_placeholder = st.empty()
    result_placeholder = st.empty()

    try:
        log_event("api_key_validation_success", page_name="data_fetch")
        total_records = 0
        success_types: list[str] = []
        failed_types: list[str] = []
        labels = {"cube": "큐브", "potential": "잠재능력 재설정", "starforce": "스타포스"}

        with loading_placeholder.container(border=True):
            st.markdown("### 전체 기록을 불러오는 중입니다")
            st.caption("기간이 길수록 시간이 걸릴 수 있습니다. 불러오는 동안 이 화면에서 진행 상태를 확인할 수 있습니다.")
            phase_placeholder = st.empty()
            detail_placeholder = st.empty()
            progress = st.progress(0.02, text="조회 준비 중입니다...")

        phase_placeholder.markdown("**1단계 · 조회 기간을 점검하는 중입니다**")
        detail_placeholder.caption("데이터 종류별 조회 가능 기간을 확인하고 있습니다.")
        fetch_plans: list[tuple[str, str, str]] = []
        for data_type in selected_types:
            start_date, end_date = controls["start_date"], controls["end_date"]
            if controls["auto_clamp"]:
                start_date, end_date, messages = clamp_date_range(start_date, end_date, data_type)
                for message in messages:
                    st.sidebar.info(message)
            elif start_date > end_date:
                st.sidebar.warning("시작일은 종료일보다 늦을 수 없습니다.")
                continue

            if start_date > end_date:
                continue

            fetch_plans.append((data_type, start_date.isoformat(), end_date.isoformat()))

        if not fetch_plans:
            loading_placeholder.empty()
            st.sidebar.warning("조회할 수 있는 기간이 없습니다. 날짜 설정을 확인해 주세요.")
            return

        phase_placeholder.markdown("**2단계 · API 호출을 시작합니다**")
        detail_placeholder.caption("잠재능력/큐브와 스타포스 기록을 병렬로 요청하고 있습니다.")
        progress.progress(0.08, text="기록 조회를 시작합니다...")
        completed = 0
        max_workers = min(PARALLEL_HISTORY_FETCH_WORKERS, len(fetch_plans))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_fetch_history_job, controls["api_key"], data_type, start_str, end_str): (data_type, start_str, end_str)
                for data_type, start_str, end_str in fetch_plans
            }
            for future in as_completed(future_map):
                data_type, start_str, end_str = future_map[future]
                try:
                    result = future.result()
                    total_records += _apply_loaded_history_result(result)
                    success_types.append(data_type)
                    detail_placeholder.caption(
                        f"{labels[data_type]} 데이터를 불러왔습니다. {completed + 1}/{len(fetch_plans)} 단계 반영 중입니다."
                    )
                except Exception as exc:
                    failed_types.append(data_type)
                    st.sidebar.error(f"{labels[data_type]} 데이터를 불러오는 중 오류가 발생했습니다: {exc}")
                    detail_placeholder.caption(
                        f"{labels[data_type]} 데이터는 실패했지만, 다른 기록은 계속 불러오고 있습니다."
                    )
                    log_error(f"{data_type}_fetch_failed", exc, page_name="sidebar")
                completed += 1
                phase_placeholder.markdown(
                    f"**3단계 · 화면 분석용 데이터로 반영 중입니다**  \n{completed}/{len(fetch_plans)} 종류를 처리했습니다."
                )
                progress.progress(
                    min(0.08 + (completed / len(fetch_plans)) * 0.84, 0.95),
                    text=f"기록 불러오기 진행 중... {completed}/{len(fetch_plans)}",
                )
        elapsed_ms = round((time.perf_counter() - fetch_started_at) * 1000, 2)
        phase_placeholder.markdown("**4단계 · 마무리 중입니다**")
        detail_placeholder.caption("불러온 기록을 정리하고 화면에 반영하고 있습니다.")
        progress.progress(1.0, text="전체 기록 불러오기가 완료되었습니다.")
        loading_placeholder.empty()
        if success_types:
            log_event(
                "data_fetch_success",
                page_name="sidebar",
                properties={
                    "target_type": ",".join(success_types),
                    "date_range_days": date_range_days,
                    "record_count": total_records,
                    "elapsed_ms": elapsed_ms,
                    "failed_types": failed_types,
                },
            )
            result_placeholder.success(
                f"전체 기록 불러오기가 완료되었습니다. 성공 {len(success_types)}종류, 실패 {len(failed_types)}종류, 총 {total_records:,}건을 반영했습니다."
            )
        if failed_types and success_types:
            st.sidebar.warning("일부 기록만 불러왔습니다. 성공한 데이터 기준으로 분석을 계속합니다.")
        if not success_types:
            log_event(
                "data_fetch_failed",
                page_name="sidebar",
                properties={"error_type": "all_fetch_failed", "target_type": "all"},
            )
            loading_placeholder.empty()
            st.sidebar.error("전체 기록을 불러오지 못했습니다. 기간 또는 API 상태를 확인해 주세요.")
    except (NexonAPIError, RuntimeError, ValueError) as exc:
        loading_placeholder.empty()
        st.sidebar.error(str(exc))
        if "API Key" in str(exc):
            log_event("api_key_validation_failed", page_name="data_fetch", properties={"reason": type(exc).__name__})
        log_event(
            "data_fetch_failed",
            page_name="sidebar",
            properties={"error_type": type(exc).__name__, "target_type": "all"},
        )
        log_error("data_fetch_failed", exc, page_name="sidebar")
    except Exception as exc:
        loading_placeholder.empty()
        st.sidebar.error(f"예상하지 못한 오류가 발생했습니다: {exc}")
        log_event(
            "data_fetch_failed",
            page_name="sidebar",
            properties={"error_type": type(exc).__name__, "target_type": "all"},
        )
        log_error("data_fetch_failed", exc, page_name="sidebar")


def _fetch_history_job(api_key: str, data_type: str, start_str: str, end_str: str) -> dict[str, Any]:
    client = NexonMapleClient(api_key)
    if data_type == "cube":
        loaded = fetch_cube_history_dataframe(client, start_str, end_str)
    elif data_type == "potential":
        loaded = fetch_potential_history_dataframe(client, start_str, end_str)
    else:
        loaded = fetch_starforce_history_dataframe(client, start_str, end_str)
    return {
        "data_type": data_type,
        "start_str": start_str,
        "end_str": end_str,
        "loaded": loaded,
        "debug_info": dict(client.last_debug_info),
    }


def _apply_loaded_history_result(result: dict[str, Any]) -> int:
    data_type = result["data_type"]
    start_str = result["start_str"]
    end_str = result["end_str"]
    loaded = result["loaded"]
    debug_info = result["debug_info"]
    labels = {"cube": "큐브", "potential": "잠재능력 재설정", "starforce": "스타포스"}
    st.session_state[f"{data_type}_df"] = loaded.dataframe
    st.session_state["api_debug"][data_type] = {
        **debug_info,
        "data_type": data_type,
        "query_start_date": start_str,
        "query_end_date": end_str,
        "record_count": len(loaded.raw_records),
        "raw_records_preview": loaded.raw_records[:3],
    }
    st.session_state["last_sync_at"] = datetime.now()
    st.session_state["last_query_range"] = f"{start_str} ~ {end_str}"
    st.session_state["_restored_from_local_state"] = False
    _persist_app_state()

    for message in loaded.messages:
        st.sidebar.info(message)

    if data_type == "starforce":
        if loaded.raw_records:
            st.sidebar.success(f"스타포스 records 수: {len(loaded.raw_records):,}건")
        else:
            st.sidebar.warning("조회된 스타포스 기록이 없습니다. 해당 기간에 기록이 없거나 확률 정보 반영 전일 수 있습니다.")

    st.sidebar.success(f"{labels[data_type]} 데이터를 화면 분석용으로 불러왔습니다.")
    return len(loaded.raw_records)


def _build_context(controls: dict[str, Any]) -> dict[str, Any]:
    cube_df = _apply_filters(st.session_state["cube_df"], "cube")
    potential_df = _apply_filters(st.session_state["potential_df"], "potential")
    starforce_df = _add_label_columns(_apply_filters(st.session_state["starforce_df"], "starforce"))
    cube_like_df = _combine_cube_like(cube_df, potential_df)
    effective_job_name = controls["job_name"] if controls.get("use_auto_major_option", True) else None
    effective_df = _add_label_columns(add_effective_option_features(cube_like_df, effective_job_name, controls["selected_stats"]))

    inferred_event_df = infer_event_rows_from_records(cube_df, potential_df, starforce_df)
    combined_event_df = inferred_event_df.copy()
    effective_df = attach_event_tags(effective_df, combined_event_df, "cube")
    starforce_df = attach_event_tags(starforce_df, combined_event_df, "starforce")
    cube_df = attach_event_tags(cube_df, combined_event_df, "cube")
    potential_df = attach_event_tags(potential_df, combined_event_df, "potential")

    cube_summary = summarize_effective_options(effective_df)
    star_summary = summarize_starforce(starforce_df)
    profile_info = _build_profile_info(
        controls.get("selected_character"),
        cube_df,
        potential_df,
        starforce_df,
        controls["job_name"],
    )

    cube_by_day_of_month = _label_day_of_month(summarize_cube_by_day_of_month(effective_df))
    cube_by_hour = summarize_cube_by_hour(effective_df)
    cube_by_hour_band = summarize_cube_by_hour_band(effective_df)
    cube_by_weekday = summarize_cube_by_weekday(effective_df)
    cube_by_type = summarize_cube_by_type(effective_df)

    star_by_day_of_month = _label_day_of_month(summarize_starforce_by_day_of_month(starforce_df))
    star_by_hour = summarize_starforce_by_hour(starforce_df)
    star_by_hour_band = summarize_starforce_by_hour_band(starforce_df)
    star_by_weekday = summarize_starforce_by_weekday(starforce_df)
    star_by_range = summarize_starforce_by_range(starforce_df)
    star_by_transition = summarize_starforce_by_transition(starforce_df)

    cube_ref_df = load_reference_csv(POTENTIAL_REFERENCE_PATH)
    star_ref_df = load_reference_csv(STARFORCE_REFERENCE_PATH)
    success_probability_groups = build_success_probability_group(starforce_df, star_ref_df)
    cube_event_compare = compare_event_periods(effective_df, "cube")
    star_event_compare = compare_event_periods(starforce_df, "starforce")

    return {
        "controls": controls,
        "selected_character": controls.get("selected_character"),
        "cube_df": cube_df,
        "potential_df": potential_df,
        "starforce_df": starforce_df,
        "effective_df": effective_df,
        "cube_summary": cube_summary,
        "star_summary": star_summary,
        "profile_info": profile_info,
        "cube_by_day_of_month": cube_by_day_of_month,
        "cube_by_hour": cube_by_hour,
        "cube_by_hour_band": cube_by_hour_band,
        "cube_by_weekday": cube_by_weekday,
        "cube_by_type": cube_by_type,
        "star_by_day_of_month": star_by_day_of_month,
        "star_by_hour": star_by_hour,
        "star_by_hour_band": star_by_hour_band,
        "star_by_weekday": star_by_weekday,
        "star_by_range": star_by_range,
        "star_by_transition": star_by_transition,
        "cube_ref_df": cube_ref_df,
        "star_ref_df": star_ref_df,
        "event_df": combined_event_df,
        "inferred_event_df": inferred_event_df,
        "cube_event_compare": cube_event_compare,
        "star_event_compare": star_event_compare,
        "success_probability_groups": success_probability_groups,
    }


def _apply_filters(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    output = filter_records_by_selected_character(df.copy(), st.session_state.get("selected_character_object"))
    label = {"cube": "큐브", "potential": "잠재능력 재설정", "starforce": "스타포스"}[kind]
    with st.sidebar.expander(f"{label} 필터", expanded=False):
        filter_columns = ["world_name", "item_name", "weekday_kr", "hour_band"]
        if kind in {"cube", "potential"}:
            filter_columns += ["cube_type", "before_potential_grade", "after_potential_grade"]
        if kind == "starforce":
            filter_columns += ["starforce_range", "transition_label"]
        for col in filter_columns:
            if col not in output.columns or not output[col].notna().any():
                continue
            options = sorted(output[col].dropna().astype(str).unique().tolist())
            selected = st.multiselect(_filter_label(col), options, key=f"{kind}_{col}_filter")
            if selected:
                output = output[output[col].astype(str).isin(selected)]
    return output


def filter_records_by_selected_character(df: pd.DataFrame, selected_character: dict[str, Any] | None) -> pd.DataFrame:
    if df is None or df.empty or not selected_character:
        return pd.DataFrame() if df is None else df

    output = df.copy()
    selected_ocid = str(selected_character.get("ocid") or "").strip()
    selected_name = str(selected_character.get("character_name") or "").strip()
    selected_world = str(selected_character.get("world_name") or "").strip()

    if selected_ocid and "ocid" in output.columns and output["ocid"].notna().any():
        matched = output[output["ocid"].astype(str).str.strip() == selected_ocid]
        if not matched.empty:
            return matched

    if selected_name and selected_world and {"character_name", "world_name"}.issubset(output.columns):
        matched = output[
            (output["character_name"].astype(str).str.strip() == selected_name)
            & (output["world_name"].astype(str).str.strip() == selected_world)
        ]
        if not matched.empty:
            return matched

    if selected_name and "character_name" in output.columns:
        return output[output["character_name"].astype(str).str.strip() == selected_name]
    return output


def _combine_cube_like(cube_df: pd.DataFrame, potential_df: pd.DataFrame) -> pd.DataFrame:
    frames = [df for df in [cube_df, potential_df] if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _add_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    output = df.copy()
    if "day_of_month" in output.columns and "day_of_month_label" not in output.columns:
        output["day_of_month_label"] = output["day_of_month"].map(lambda value: f"{int(value)}일" if pd.notna(value) else None)
    return output


def _render_profile_header(context: dict[str, Any]) -> None:
    profile = context["profile_info"]
    sync_time = st.session_state.get("last_sync_at")
    sync_text = sync_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(sync_time, datetime) else "없음"
    period_text = st.session_state.get("last_query_range") or "불러온 기간 없음"
    render_character_profile_card(context.get("selected_character"), profile, sync_text, period_text)


def _build_profile_info(
    selected_character: dict[str, Any] | None,
    cube_df: pd.DataFrame,
    potential_df: pd.DataFrame,
    starforce_df: pd.DataFrame,
    job_name: str,
) -> dict[str, str]:
    if selected_character:
        nickname = str(selected_character.get("character_name") or "-")
        world = str(selected_character.get("world_name") or "-")
        level_value = selected_character.get("character_level")
        level = str(int(level_value)) if pd.notna(level_value) else "API 응답에서 미확인"
        image_text = nickname[:1] if nickname and nickname != "-" else "M"
        return {
            "nickname": nickname,
            "world": world,
            "job": str(selected_character.get("character_class") or job_name or "직업 미확인"),
            "level": level,
            "image_text": image_text,
            "image_url": str(selected_character.get("character_image") or "").strip(),
            "guild_name": str(selected_character.get("character_guild_name") or "").strip(),
        }
    fallback = extract_profile_info(
        cube_df,
        potential_df,
        starforce_df,
        job_name,
        None,
    )
    fallback["image_url"] = ""
    fallback["guild_name"] = ""
    return fallback


def render_character_profile_card(
    selected_character: dict[str, Any] | None,
    profile: dict[str, Any],
    sync_text: str,
    period_text: str,
) -> None:
    image_url = str(profile.get("image_url") or "").strip()
    nickname = str(profile.get("nickname") or "-")
    world = str(profile.get("world") or "-")
    job = str(profile.get("job") or "직업 미확인")
    level = str(profile.get("level") or "미확인")
    guild_name = str(profile.get("guild_name") or "").strip()
    fallback_text = str(profile.get("image_text") or (nickname[:1] if nickname and nickname != "-" else "M"))

    chips = [
        render_metric_chip("월드", world, st.session_state.get("resolved_theme_mode", "light"), "neutral"),
        render_metric_chip("직업", job, st.session_state.get("resolved_theme_mode", "light"), "neutral"),
        render_metric_chip("레벨", f"Lv.{level}" if str(level).isdigit() else level, st.session_state.get("resolved_theme_mode", "light"), "accent"),
        render_metric_chip("데이터 기준 기간", period_text, st.session_state.get("resolved_theme_mode", "light"), "info"),
        render_metric_chip("최근 동기화", sync_text, st.session_state.get("resolved_theme_mode", "light"), "neutral"),
    ]
    if guild_name:
        chips.insert(3, render_metric_chip("길드", guild_name, st.session_state.get("resolved_theme_mode", "light"), "success"))

    avatar_html = (
        f"<img src=\"{image_url}\" alt=\"character image\" class=\"maple-profile-avatar-img\" />"
        if image_url
        else f"<span class=\"maple-profile-avatar-fallback\">{fallback_text}</span>"
    )

    st.markdown(
        f"""
<div class="maple-card maple-profile-card maple-profile-hero">
  <div class="maple-profile-layout">
    <div class="maple-profile-avatar maple-profile-avatar-shell">
      {avatar_html}
    </div>
    <div class="maple-profile-content">
      <div class="maple-title-lg">{nickname}</div>
      <div class="maple-text-secondary">내 큐브·스타포스 기록으로 보는 과거 운빨 분석 리포트</div>
      <div class="maple-chip-row">{''.join(chips)}</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_section_header(title: str, description: str | None = None) -> None:
    description_html = f"<p>{description}</p>" if description else ""
    st.markdown(
        f"""
<div class="maple-section-header">
  <h3>{title}</h3>
  {description_html}
</div>
""",
        unsafe_allow_html=True,
    )


def render_metric_card_grid(metrics: list[dict[str, Any]], columns: int = 4) -> None:
    if not metrics:
        return
    theme_mode = st.session_state.get("resolved_theme_mode", "light")
    column_sets = st.columns(columns)
    for idx, metric in enumerate(metrics):
        variant = metric.get("variant", "accent")
        delta = metric.get("delta")
        subtitle = metric.get("subtitle")
        with column_sets[idx % columns]:
            chips = []
            if delta:
                chip_variant = "success" if str(delta).strip().startswith("+") else "danger"
                chips.append(render_metric_chip("평균 대비", str(delta), theme_mode, chip_variant))
            if subtitle:
                chips.append(render_metric_chip("설명", str(subtitle), theme_mode, "neutral"))
            accent_class = f"maple-metric-card-{variant}"
            st.markdown(
                f"""
<div class="maple-card maple-metric-card {accent_class}">
  <div class="maple-metric-label">{metric.get('label', '')}</div>
  <div class="maple-metric-value">{metric.get('value', '-')}</div>
  <div class="maple-chip-row">{''.join(chips)}</div>
</div>
""",
                unsafe_allow_html=True,
            )


def render_chart_card(title: str, fig, *, key: str, description: str | None = None) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if description:
            st.caption(description)
        st.plotly_chart(fig, width="stretch", key=key)


def _render_overview_tab(context: dict[str, Any]) -> None:
    _log_tab_view_once("종합 요약")
    render_section_header("종합 요약", "선택된 캐릭터의 최근 2년 기록을 기준으로 잠재능력/큐브와 스타포스 결과를 한눈에 정리합니다.")
    cube_summary = context["cube_summary"]
    star_summary = context["star_summary"]
    effective_df = context["effective_df"]
    starforce_df = context["starforce_df"]

    render_metric_card_grid(
        [
            {"label": "큐브/잠재 총 시도 수", "value": f"{cube_summary['total_cube_uses']:,}회", "subtitle": "최근 2년 기준", "variant": "accent"},
            {"label": "주요옵션 출현률", "value": format_percent(cube_summary["major_rate"]), "subtitle": "선택 주요옵션 1줄 이상", "variant": "success"},
            {"label": "유효옵션 출현률", "value": format_percent(cube_summary["effective_rate"]), "subtitle": "선택 주요옵션 2줄 이상", "variant": "accent"},
            {"label": "등급업률", "value": format_percent(_bool_rate(effective_df, "is_grade_up")), "subtitle": "큐브/잠재 전체", "variant": "warning"},
            {"label": "스타포스 총 시도 수", "value": f"{star_summary['total_attempts']:,}회", "subtitle": "최근 2년 기준", "variant": "accent"},
            {"label": "성공률", "value": format_percent(star_summary["success_rate"]), "subtitle": "스타포스 전체", "variant": "success"},
            {"label": "파괴율", "value": format_percent(_bool_rate(starforce_df, "is_destroyed")), "subtitle": "스타포스 전체", "variant": "danger"},
            {"label": "데이터 기준 기간", "value": st.session_state.get("last_query_range") or "없음", "subtitle": "마지막 불러오기 기준", "variant": "neutral"},
        ],
        columns=4,
    )
    with st.expander("보정률은 어떻게 계산되나요?", expanded=False):
        st.markdown(
            """
- 실제률은 단순히 `성공 수 / 시도 수`로 계산합니다.
- 시도 수가 적으면 1회 성공만으로도 100%처럼 보일 수 있어, 전체 평균을 prior로 섞은 보정률을 함께 봅니다.
- 식: `adjusted_rate = (success_count + overall_rate × 30) / (attempts + 30)`
- 시도 수가 많을수록 보정률은 실제률에 가까워지고, 시도 수가 적을수록 전체 평균에 가까워집니다.
- 보정률은 미래 결과를 뜻하는 값이 아니라, 표본 수가 적은 조건의 과대평가를 줄이기 위한 참고 지표입니다.
"""
        )
    cube_rows = _condition_rows_for_selected(
        context,
        target_type="cube",
        grouping=["day_of_month_label", "hour_label", "cube_type"],
        metric_label="유효옵션 출현률",
        direction="good",
    )
    star_rows = _condition_rows_for_selected(
        context,
        target_type="starforce",
        grouping=["day_of_month_label", "hour_label", "starforce_transition"],
        metric_label="스타포스 성공률",
        direction="good",
    )
    insights = []
    if not cube_rows.empty:
        insights.append(make_good_condition_text(cube_rows.iloc[0]))
    if not star_rows.empty:
        insights.append(make_good_condition_text(star_rows.iloc[0]))
    if insights:
        st.info(" ".join(insights[:2]) + " 이 분석은 과거 기록 기반 참고용이며, 향후 결과를 보장하지 않습니다.")
    else:
        st.info("현재 표본 기준을 만족하는 조건이 많지 않아 특정 조건을 두드러지게 해석하기 어렵습니다. 표본 수가 적은 결과는 참고용으로만 해석해야 합니다.")

    left, right = st.columns(2)
    with left:
        render_section_header("잠재능력/큐브에서 유효옵션 출현률이 좋게 관측된 조건 TOP 5")
        _render_condition_cards(cube_rows, tone="good")
    with right:
        render_section_header("스타포스에서 성공률이 좋게 관측된 조건 TOP 5")
        _render_condition_cards(star_rows, tone="good")
    st.caption("위 결과는 과거 기록 기반의 참고용 통계이며, 향후 결과를 보장하지 않습니다.")


def _render_day_of_month_tab(context: dict[str, Any]) -> None:
    _log_tab_view_once("일자별 분석")
    render_section_header("일자별 분석", "기준기간 동안 같은 일자끼리 묶어 비교합니다. 예: 2월 9일, 3월 9일, 4월 9일은 모두 9일로 집계됩니다.")

    cube_tab, star_tab = st.tabs(["잠재능력/큐브", "스타포스"])

    with cube_tab:
        _render_cube_day_of_month_section(context)
    with star_tab:
        _render_star_day_of_month_section(context)


def _render_hour_tab(context: dict[str, Any]) -> None:
    _log_tab_view_once("시간별 분석")
    render_section_header("시간별 분석", "0시~23시 기준으로 비교하며, 시간대 분석은 새벽/오전/오후/저녁 구간으로 함께 봅니다.")

    cube_tab, star_tab = st.tabs(["잠재능력/큐브", "스타포스"])
    with cube_tab:
        _render_cube_hour_section(context)
    with star_tab:
        _render_star_hour_section(context)


def _render_weekday_tab(context: dict[str, Any]) -> None:
    _log_tab_view_once("요일별 분석")
    render_section_header("요일별 분석", "월요일~일요일 기록을 묶어 비교한 결과입니다. 표본 수가 적은 요일은 참고용으로 해석해야 합니다.")

    cube_tab, star_tab = st.tabs(["잠재능력/큐브", "스타포스"])
    with cube_tab:
        _render_cube_weekday_section(context)
    with star_tab:
        _render_star_weekday_section(context)


def _render_condition_tab(context: dict[str, Any]) -> None:
    _log_tab_view_once("조건 조합 TOP 5")
    render_section_header("조건 조합 TOP 5", "사용자가 선택한 조건 조합 하나만 기준으로 그룹화해, 과거 기록상 결과가 좋게 또는 아쉽게 관측된 조건을 비교합니다.")
    with st.expander("보정률은 어떻게 계산되나요?", expanded=False):
        st.markdown(
            """
- 실제률은 `성공 수 / 시도 수`로 계산합니다.
- 다만 시도 수가 적으면 1회 성공만으로도 100%처럼 보일 수 있어, 전체 평균을 함께 섞은 보정률을 사용합니다.
- 사용 식: `adjusted_rate = (success_count + overall_rate × 30) / (attempts + 30)`
- 시도 수가 많을수록 보정률은 실제률에 가까워지고, 시도 수가 적을수록 전체 평균에 가까워집니다.
- 보정률은 미래 확률이 아니라, 표본 수가 적은 조건의 과대평가를 줄이기 위한 참고 지표입니다.
"""
        )

    cube_tab, star_tab = st.tabs(["잠재능력/큐브", "스타포스"])
    with cube_tab:
        grouping_cols, grouping_label = _render_grouping_checkboxes("cube")
        cube_source_df = context["effective_df"]
        cube_type_options = []
        if "cube_type" in cube_source_df.columns and cube_source_df["cube_type"].notna().any():
            cube_type_options = sorted(cube_source_df["cube_type"].dropna().astype(str).unique().tolist())
        selected_cube_types = st.multiselect(
            "큐브 타입 상세 선택",
            options=cube_type_options,
            default=[],
            key="cube_condition_type_filter",
            help="선택한 큐브 타입만 분석 대상에 포함합니다. 비워두면 전체 큐브 타입을 대상으로 계산합니다.",
        )
        if selected_cube_types:
            cube_source_df = cube_source_df[cube_source_df["cube_type"].astype(str).isin(selected_cube_types)].copy()
        selected_cube_type_text = "전체" if not selected_cube_types else ", ".join(selected_cube_types)
        metric_filter = st.selectbox(
            "기준 지표 선택",
            ["주요옵션 출현률", "유효옵션 출현률", "등급업률"],
            index=1,
            key="cube_condition_metric_filter",
        )
        cube_condition_signature = (
            tuple(grouping_cols),
            metric_filter,
            context["controls"]["top_min_attempts"],
            context["controls"]["top_score_basis"],
            tuple(selected_cube_types),
        )
        if st.session_state.get("_analytics_last_cube_condition_signature") != cube_condition_signature:
            log_event(
                "top5_grouping_changed",
                page_name="조건 조합 TOP 5",
                properties={
                    "target_type": "cube",
                    "selected_grouping": grouping_cols,
                    "grouping_label": grouping_label,
                    "selected_metric": metric_filter,
                    "min_attempts": context["controls"]["top_min_attempts"],
                    "score_basis": context["controls"]["top_score_basis"],
                    "selected_cube_type": selected_cube_types,
                },
            )
            st.session_state["_analytics_last_cube_condition_signature"] = cube_condition_signature
        st.caption(f"선택 기준: {grouping_label} · 최소 시도 수 {context['controls']['top_min_attempts']}회 · 정렬 기준 {context['controls']['top_score_basis']}")
        st.caption(f"분석 대상 큐브 타입: {selected_cube_type_text}")
        st.caption("큐브 타입 필터는 분석 대상 데이터를 제한합니다. 분석 기준에서 큐브 타입을 체크하면 조건명에도 큐브 타입이 표시됩니다.")
        st.caption("아래 결과는 선택한 기준으로만 그룹화한 결과입니다.")
        good_rows = _condition_rows_for_selected(context, "cube", grouping_cols, metric_filter, "good", source_df=cube_source_df)
        bad_rows = _condition_rows_for_selected(context, "cube", grouping_cols, metric_filter, "bad", source_df=cube_source_df)
        left, right = st.columns(2)
        with left:
            st.markdown("**과거 기록상 결과가 좋게 관측된 조건 TOP 5**")
            _render_condition_cards(good_rows, tone="good")
        with right:
            st.markdown("**과거 기록상 결과가 아쉬웠던 조건 TOP 5**")
            _render_condition_cards(bad_rows, tone="bad")
        st.caption("위 결과는 과거 기록 기반의 참고용 통계이며, 향후 결과를 보장하지 않습니다.")
        st.markdown("**같은 조건에서 여러 지표가 함께 눈에 띈 통합 카드**")
        _render_grouped_metric_cards(
            _build_grouped_metric_cards(
                context,
                "cube",
                grouping_cols,
                ["주요옵션 출현률", "유효옵션 출현률", "등급업률"],
                source_df=cube_source_df,
            ),
        )
        with st.expander("지표별 상세 TOP 5", expanded=False):
            for label in ["주요옵션 출현률", "유효옵션 출현률", "등급업률"]:
                st.markdown(f"**{label} 기준 좋게 관측된 조건 TOP 5**")
                _render_condition_cards(_condition_rows_for_selected(context, "cube", grouping_cols, label, "good", source_df=cube_source_df), tone="good")
                st.markdown(f"**{label} 기준 아쉬웠던 조건 TOP 5**")
                _render_condition_cards(_condition_rows_for_selected(context, "cube", grouping_cols, label, "bad", source_df=cube_source_df), tone="bad")
            st.caption("위 결과는 과거 기록 기반의 참고용 통계이며, 향후 결과를 보장하지 않습니다.")
        _render_cube_condition_maps(context, metric_filter, source_df=cube_source_df)

    with star_tab:
        grouping_cols, grouping_label = _render_grouping_checkboxes("starforce")
        metric_filter = st.selectbox(
            "기준 지표 선택",
            ["스타포스 성공률", "스타포스 파괴율"],
            index=0,
            key="star_condition_metric_filter",
        )
        star_condition_signature = (
            tuple(grouping_cols),
            metric_filter,
            context["controls"]["top_min_attempts"],
            context["controls"]["top_score_basis"],
        )
        if st.session_state.get("_analytics_last_star_condition_signature") != star_condition_signature:
            log_event(
                "top5_grouping_changed",
                page_name="조건 조합 TOP 5",
                properties={
                    "target_type": "starforce",
                    "selected_grouping": grouping_cols,
                    "grouping_label": grouping_label,
                    "selected_metric": metric_filter,
                    "min_attempts": context["controls"]["top_min_attempts"],
                    "score_basis": context["controls"]["top_score_basis"],
                },
            )
            st.session_state["_analytics_last_star_condition_signature"] = star_condition_signature
        st.caption(f"선택 기준: {grouping_label} · 최소 시도 수 {context['controls']['top_min_attempts']}회 · 정렬 기준 {context['controls']['top_score_basis']}")
        st.caption("아래 결과는 선택한 기준으로만 그룹화한 결과입니다.")
        good_rows = _condition_rows_for_selected(context, "starforce", grouping_cols, metric_filter, "good")
        bad_rows = _condition_rows_for_selected(context, "starforce", grouping_cols, metric_filter, "bad")
        left, right = st.columns(2)
        with left:
            st.markdown("**과거 기록상 결과가 좋게 관측된 조건 TOP 5**")
            _render_condition_cards(good_rows, tone="good")
        with right:
            st.markdown("**과거 기록상 결과가 아쉬웠던 조건 TOP 5**")
            _render_condition_cards(bad_rows, tone="bad")
        st.caption("위 결과는 과거 기록 기반의 참고용 통계이며, 향후 결과를 보장하지 않습니다.")
        st.markdown("**같은 조건에서 여러 지표가 함께 눈에 띈 통합 카드**")
        _render_grouped_metric_cards(
            _build_grouped_metric_cards(context, "starforce", grouping_cols, ["스타포스 성공률", "스타포스 파괴율"]),
        )
        with st.expander("지표별 상세 TOP 5", expanded=False):
            st.markdown("**성공률 기준 좋게 관측된 조건 TOP 5**")
            _render_condition_cards(_condition_rows_for_selected(context, "starforce", grouping_cols, "스타포스 성공률", "good"), tone="good")
            st.markdown("**파괴율 기준 낮게 관측된 조건 TOP 5**")
            _render_condition_cards(_condition_rows_for_selected(context, "starforce", grouping_cols, "스타포스 파괴율", "good"), tone="good")
            st.markdown("**성공률 기준 아쉬웠던 조건 TOP 5**")
            _render_condition_cards(_condition_rows_for_selected(context, "starforce", grouping_cols, "스타포스 성공률", "bad"), tone="bad")
            st.markdown("**파괴율 기준 높게 관측된 조건 TOP 5**")
            _render_condition_cards(_condition_rows_for_selected(context, "starforce", grouping_cols, "스타포스 파괴율", "bad"), tone="bad")
            st.caption("위 결과는 과거 기록 기반의 참고용 통계이며, 향후 결과를 보장하지 않습니다.")
        _render_star_condition_maps(context, metric_filter)


def _render_reference_tab(context: dict[str, Any]) -> None:
    st.subheader("기준 확률 비교")
    st.caption("기준 확률 CSV가 있을 때만 비교하며, 기준 확률 대비 높게 또는 낮게 관측된 차이만 참고용으로 보여줍니다.")

    cube_tab, star_tab = st.tabs(["잠재능력/큐브", "스타포스"])

    with cube_tab:
        grouping = st.selectbox(
            "잠재능력/큐브 기준 확률 비교 기준",
            ["큐브 타입", "일자 + 큐브 타입", "시간 + 큐브 타입", "요일 + 큐브 타입", "일자 + 시간 + 큐브 타입", "요일 + 시간 + 큐브 타입"],
            index=0,
            key="cube_reference_grouping",
        )
        metric_label = st.selectbox(
            "비교 지표",
            ["주요옵션 출현률", "유효옵션 출현률", "등급업률"],
            index=1,
            key="cube_reference_metric",
        )
        cube_rows = _condition_rows_for_selected(context, "cube", grouping, metric_label, "good")
        cube_rows = cube_rows[cube_rows["reference_rate"].notna()].copy()
        if cube_rows.empty:
            st.info("잠재능력/큐브 기준 확률과 매칭되는 행이 없습니다.")
        else:
            _render_reference_rows(cube_rows, "cube_reference_gap_condition_chart")

    with star_tab:
        grouping = st.selectbox(
            "스타포스 기준 확률 비교 기준",
            [
                "스타포스 구간",
                "일자 + 스타포스 구간",
                "시간 + 스타포스 구간",
                "요일 + 스타포스 구간",
                "일자 + 시간 + 스타포스 구간",
                "요일 + 시간 + 스타포스 구간",
            ],
            index=0,
            key="star_reference_grouping",
        )
        metric_label = st.selectbox(
            "비교 지표",
            ["스타포스 성공률", "스타포스 파괴율"],
            index=0,
            key="star_reference_metric",
        )
        star_rows = _condition_rows_for_selected(context, "starforce", grouping, metric_label, "good")
        star_rows = star_rows[star_rows["reference_rate"].notna()].copy()
        if star_rows.empty:
            st.info("스타포스 기준 확률과 매칭되는 행이 없습니다.")
        else:
            _render_reference_rows(star_rows, "star_reference_gap_condition_chart")


def _render_event_tab(context: dict[str, Any]) -> None:
    st.subheader("이벤트 영향 분석")
    st.caption("이벤트 정보는 기록 내 이벤트 필드와 수동 태그를 함께 사용합니다. 자동 추정 이벤트는 참고용으로만 해석해야 합니다.")

    event_df = context["event_df"]
    if event_df is None or event_df.empty:
        st.info("현재 연결된 이벤트 태그가 없습니다. 사이드바에서 수동 이벤트를 추가하면 비교에 반영됩니다.")
    else:
        st.markdown("**이벤트 목록**")
        st.dataframe(
            event_df[["event_name", "event_type", "apply_target", "start_date", "end_date", "source", "confidence", "note"]],
            width="stretch",
            hide_index=True,
        )

    cube_tab, star_tab = st.tabs(["잠재능력/큐브", "스타포스"])
    with cube_tab:
        compare_df = context["cube_event_compare"]
        if compare_df is None or compare_df.empty:
            st.info("잠재능력/큐브 이벤트 비교에 사용할 데이터가 부족합니다.")
        else:
            st.dataframe(compare_df, width="stretch", hide_index=True)
            st.plotly_chart(
                plot_multi_rate_bar(
                    compare_df,
                    "event_group",
                    {
                        "주요옵션 출현률": "major_option_rate",
                        "유효옵션 출현률": "effective_option_rate",
                        "등급업률": "grade_up_rate",
                    },
                    "이벤트 기간 vs 일반 기간 비교",
                ),
                width="stretch",
                key="cube_event_compare_chart",
            )
    with star_tab:
        compare_df = context["star_event_compare"]
        if compare_df is None or compare_df.empty:
            st.info("스타포스 이벤트 비교에 사용할 데이터가 부족합니다.")
        else:
            st.dataframe(compare_df, width="stretch", hide_index=True)
            st.plotly_chart(
                plot_multi_rate_bar(
                    compare_df,
                    "event_group",
                    {
                        "성공률": "success_rate",
                        "파괴율": "destroy_rate",
                    },
                    "이벤트 기간 vs 일반 기간 비교",
                ),
                width="stretch",
                key="star_event_compare_chart",
            )
    st.caption("이벤트 정보는 API 제공 범위와 공지 파싱 정확도에 따라 누락되거나 부정확할 수 있습니다. 자동 추정 이벤트는 참고용으로만 사용하세요.")


def _render_raw_tab(context: dict[str, Any]) -> None:
    _log_tab_view_once("원본 데이터")
    render_section_header("원본 데이터", "선택된 캐릭터 기준으로 필터링된 기록을 화면에서만 간단히 확인합니다.")
    st.caption("원본 기록은 화면에서만 확인하며, 이번 버전에서는 분석 결과 다운로드를 제공하지 않습니다.")
    st.markdown("**큐브 데이터**")
    st.dataframe(context["cube_df"], width="stretch", hide_index=True)
    st.markdown("**잠재능력 재설정 데이터**")
    st.dataframe(context["potential_df"], width="stretch", hide_index=True)
    st.markdown("**스타포스 데이터**")
    st.dataframe(context["starforce_df"], width="stretch", hide_index=True)
    st.markdown("**유효옵션 가공 데이터**")
    st.dataframe(context["effective_df"], width="stretch", hide_index=True)


def _render_debug_tab(context: dict[str, Any], controls: dict[str, Any]) -> None:
    _log_tab_view_once("API 디버그")
    render_section_header("API 디버그", "최근 API 호출 상태와 요약 정보를 확인합니다. API Key 원문과 전체 raw response는 표시하지 않습니다.")
    st.caption("API Key는 표시하지 않으며, 이번 버전은 분석 CSV를 생성하지 않고 화면에서만 결과를 보여줍니다.")
    debug = st.session_state.get("api_debug", {})
    st.write("API Key 설정 여부:", "설정됨" if controls["api_key"].strip() else "미설정")
    st.write("마지막 동기화 시각:", st.session_state.get("last_sync_at") or "없음")
    st.write("수집 기간:", st.session_state.get("last_query_range") or "없음")
    st.write("큐브 기록 수:", len(context["effective_df"]))
    st.write("스타포스 기록 수:", len(context["starforce_df"]))
    st.write("잠재 기준 확률 CSV:", "있음" if context["cube_ref_df"] is not None else "없음")
    st.write("스타포스 기준 확률 CSV:", "있음" if context["star_ref_df"] is not None else "없음")

    if not debug:
        st.info("아직 API 호출 기록이 없습니다.")
        return
    selected = st.selectbox("디버그 대상", list(debug.keys()), index=max(0, len(debug) - 1))
    info = debug[selected]
    st.write("path:", info.get("path"))
    st.write("params:", info.get("params"))
    st.write("status_code:", info.get("status_code"))
    st.write("response_keys:", info.get("response_keys"))
    st.write("count:", info.get("count"))
    st.write("record_key_exists:", info.get("record_key_exists"))
    st.write("record_count:", info.get("record_count"))
    st.write("next_cursor_exists:", info.get("next_cursor_exists"))
    st.write("캐시 재사용 날짜 수:", info.get("cache_hit_count"))
    st.write("실제 API 호출 날짜 수:", info.get("api_call_count"))
    st.write("실패한 날짜 목록:", info.get("error_dates"))
    if controls.get("show_raw_response"):
        st.json(info.get("raw_response_preview") or info.get("raw_records_preview") or {})


def _render_cube_day_of_month_section(context: dict[str, Any]) -> None:
    df = context["effective_df"]
    if df.empty:
        st.info("조회된 잠재능력/큐브 데이터가 없습니다.")
        return
    metric_label = st.selectbox("잠재능력/큐브 지표 선택", ["주요옵션 출현률", "유효옵션 출현률", "등급업률"], index=1, key="cube_day_metric")
    best_row = _best_ranked_row(
        context["cube_by_day_of_month"],
        "day_of_month_label",
        metric_label,
        min_attempts=context["controls"]["top_min_attempts"],
        score_basis=context["controls"]["top_score_basis"],
    )
    _render_summary_conclusion_card(f"과거 기록상 {metric_label}이 가장 높게 관측된 일자", best_row, label_col="day_of_month_label")
    if best_row is not None:
        st.info(make_day_of_month_insight_text(best_row) + " 이는 같은 일자 데이터를 묶어 계산한 결과입니다.")

    top_cols = st.columns(2)
    with top_cols[0]:
        fig = plot_day_of_month_rate(
            context["cube_by_day_of_month"],
            "major_option_rate",
            "일자별 주요옵션 출현률",
            min_attempts=context["controls"]["top_min_attempts"],
            show_low_sample=context["controls"]["show_low_sample"],
        )
        add_average_line(fig, _bool_rate(df, "has_major_option"), "전체 평균")
        render_chart_card("일자별 주요옵션 출현률", fig, key="cube_day_major_chart")
    with top_cols[1]:
        fig = plot_day_of_month_rate(
            context["cube_by_day_of_month"],
            "effective_option_rate",
            "일자별 유효옵션 출현률",
            min_attempts=context["controls"]["top_min_attempts"],
            show_low_sample=context["controls"]["show_low_sample"],
        )
        add_average_line(fig, _bool_rate(df, "has_effective_option"), "전체 평균")
        render_chart_card("일자별 유효옵션 출현률", fig, key="cube_day_effective_chart")
    fig = plot_day_of_month_rate(
        context["cube_by_day_of_month"],
        "grade_up_rate",
        "일자별 등급업률",
        min_attempts=context["controls"]["top_min_attempts"],
        show_low_sample=context["controls"]["show_low_sample"],
    )
    add_average_line(fig, _bool_rate(df, "is_grade_up"), "전체 평균")
    render_chart_card("일자별 등급업률", fig, key="cube_day_gradeup_chart")
    st.caption("TOP 5와 결론 카드는 최소 시도 수 기준을 만족한 항목만 사용합니다. 기준 미만 항목은 그래프에서 연하게 표시됩니다.")
    st.dataframe(
        _rank_dimension_table(
            context["cube_by_day_of_month"],
            "day_of_month_label",
            metric_label,
            min_attempts=context["controls"]["top_min_attempts"],
            score_basis=context["controls"]["top_score_basis"],
        ),
        width="stretch",
        hide_index=True,
    )


def _render_star_day_of_month_section(context: dict[str, Any]) -> None:
    df = context["starforce_df"]
    st.markdown(STARFORCE_NOTICE)
    if df.empty:
        st.info("조회된 스타포스 데이터가 없습니다.")
        return
    metric_label = st.selectbox("스타포스 지표 선택", ["스타포스 성공률", "스타포스 파괴율"], index=0, key="star_day_metric")
    best_row = _best_ranked_row(
        context["star_by_day_of_month"],
        "day_of_month_label",
        metric_label,
        min_attempts=context["controls"]["top_min_attempts"],
        score_basis=context["controls"]["top_score_basis"],
    )
    title = "과거 기록상 성공률이 가장 높게 관측된 일자" if metric_label == "스타포스 성공률" else "과거 기록상 파괴율이 가장 낮게 관측된 일자"
    _render_summary_conclusion_card(title, best_row, label_col="day_of_month_label")
    if best_row is not None:
        st.info(make_day_of_month_insight_text(best_row) + " 이는 같은 일자 데이터를 묶어 계산한 결과입니다.")

    cols = st.columns(2)
    _plot_with_baseline(
        cols[0],
        plot_day_of_month_rate(context["star_by_day_of_month"], "success_rate", "일자별 성공률", min_attempts=context["controls"]["top_min_attempts"], show_low_sample=context["controls"]["show_low_sample"]),
        _bool_rate(df, "is_success"),
        "star_day_success_chart",
    )
    _plot_with_baseline(
        cols[1],
        plot_day_of_month_rate(context["star_by_day_of_month"], "destroy_rate", "일자별 파괴율", min_attempts=context["controls"]["top_min_attempts"], show_low_sample=context["controls"]["show_low_sample"]),
        _bool_rate(df, "is_destroyed"),
        "star_day_destroy_chart",
    )

    day_mode = st.radio("일자별 스타포스 세부 보기", ["구간별", "전이 구간별"], horizontal=True, key="star_day_mode")
    detail_col = "starforce_range" if day_mode == "구간별" else "starforce_transition"
    detail_title = "일자별 스타포스 구간별 성공률" if day_mode == "구간별" else "일자별 스타포스 전이 구간별 성공률"
    range_day_df = add_confidence(
        _group_rate_summary(df, ["day_of_month_label", detail_col], "is_success", "success_rate"),
        "attempts",
    )
    if not range_day_df.empty:
        st.plotly_chart(
            plot_rate_heatmap(
                range_day_df,
                x_col="day_of_month_label",
                y_col=detail_col,
                rate_col="success_rate",
                title=detail_title,
                x_order=DAY_OF_MONTH_ORDER,
            ),
            width="stretch",
            key="star_day_range_heatmap",
        )
    else:
        st.info("일자별 스타포스 세부 성공률 데이터를 그릴 데이터가 부족합니다.")

    st.caption("TOP 5와 결론 카드는 최소 시도 수 기준을 만족한 항목만 사용합니다. 기준 미만 항목은 그래프에서 연하게 표시됩니다.")
    st.dataframe(
        _rank_dimension_table(
            context["star_by_day_of_month"],
            "day_of_month_label",
            metric_label,
            min_attempts=context["controls"]["top_min_attempts"],
            score_basis=context["controls"]["top_score_basis"],
        ),
        width="stretch",
        hide_index=True,
    )


def _render_cube_hour_section(context: dict[str, Any]) -> None:
    df = context["effective_df"]
    if df.empty:
        st.info("조회된 잠재능력/큐브 데이터가 없습니다.")
        return
    metric_label = st.selectbox("잠재능력/큐브 지표 선택", ["주요옵션 출현률", "유효옵션 출현률", "등급업률"], index=1, key="cube_hour_metric")
    best_row = _best_ranked_row(
        context["cube_by_hour"],
        "hour_label",
        metric_label,
        min_attempts=context["controls"]["top_min_attempts"],
        score_basis=context["controls"]["top_score_basis"],
    )
    _render_summary_conclusion_card(f"과거 기록상 {metric_label}이 가장 높게 관측된 시간", best_row, label_col="hour_label")
    if best_row is not None:
        st.info(make_hour_insight_text(best_row) + " 단, 과거 기록 기반 분석이므로 향후 결과를 보장하지 않습니다.")

    top_cols = st.columns(2)
    with top_cols[0]:
        fig = plot_hourly_rate(
            context["cube_by_hour"],
            "major_option_rate",
            "시간별 주요옵션 출현률",
            min_attempts=context["controls"]["top_min_attempts"],
            show_low_sample=context["controls"]["show_low_sample"],
        )
        add_average_line(fig, _bool_rate(df, "has_major_option"), "전체 평균")
        render_chart_card("시간별 주요옵션 출현률", fig, key="cube_hour_major_chart")
    with top_cols[1]:
        fig = plot_hourly_rate(
            context["cube_by_hour"],
            "effective_option_rate",
            "시간별 유효옵션 출현률",
            min_attempts=context["controls"]["top_min_attempts"],
            show_low_sample=context["controls"]["show_low_sample"],
        )
        add_average_line(fig, _bool_rate(df, "has_effective_option"), "전체 평균")
        render_chart_card("시간별 유효옵션 출현률", fig, key="cube_hour_effective_chart")
    fig = plot_hourly_rate(
        context["cube_by_hour"],
        "grade_up_rate",
        "시간별 등급업률",
        min_attempts=context["controls"]["top_min_attempts"],
        show_low_sample=context["controls"]["show_low_sample"],
    )
    add_average_line(fig, _bool_rate(df, "is_grade_up"), "전체 평균")
    render_chart_card("시간별 등급업률", fig, key="cube_hour_gradeup_chart")

    band_cols = st.columns(2)
    with band_cols[0]:
        fig = plot_hour_band_rate(
            context["cube_by_hour_band"],
            "major_option_rate",
            "시간대별 주요옵션 출현률",
            min_attempts=context["controls"]["top_min_attempts"],
            show_low_sample=context["controls"]["show_low_sample"],
        )
        add_average_line(fig, _bool_rate(df, "has_major_option"), "전체 평균")
        render_chart_card("시간대별 주요옵션 출현률", fig, key="cube_timeblock_major_chart")
    with band_cols[1]:
        fig = plot_hour_band_rate(
            context["cube_by_hour_band"],
            "effective_option_rate",
            "시간대별 유효옵션 출현률",
            min_attempts=context["controls"]["top_min_attempts"],
            show_low_sample=context["controls"]["show_low_sample"],
        )
        add_average_line(fig, _bool_rate(df, "has_effective_option"), "전체 평균")
        render_chart_card("시간대별 유효옵션 출현률", fig, key="cube_timeblock_effective_chart")
    fig = plot_hour_band_rate(
        context["cube_by_hour_band"],
        "grade_up_rate",
        "시간대별 등급업률",
        min_attempts=context["controls"]["top_min_attempts"],
        show_low_sample=context["controls"]["show_low_sample"],
    )
    add_average_line(fig, _bool_rate(df, "is_grade_up"), "전체 평균")
    render_chart_card("시간대별 등급업률", fig, key="cube_timeblock_gradeup_chart")

    st.caption("TOP 5와 결론 카드는 최소 시도 수 기준을 만족한 항목만 사용합니다. 기준 미만 항목은 그래프에서 연하게 표시됩니다.")
    st.dataframe(
        _rank_dimension_table(
            context["cube_by_hour"],
            "hour_label",
            metric_label,
            min_attempts=context["controls"]["top_min_attempts"],
            score_basis=context["controls"]["top_score_basis"],
        ),
        width="stretch",
        hide_index=True,
    )


def _render_star_hour_section(context: dict[str, Any]) -> None:
    df = context["starforce_df"]
    if df.empty:
        st.info("조회된 스타포스 데이터가 없습니다.")
        return
    metric_label = st.selectbox("스타포스 지표 선택", ["스타포스 성공률", "스타포스 파괴율"], index=0, key="star_hour_metric")
    best_row = _best_ranked_row(
        context["star_by_hour"],
        "hour_label",
        metric_label,
        min_attempts=context["controls"]["top_min_attempts"],
        score_basis=context["controls"]["top_score_basis"],
    )
    title = "과거 기록상 성공률이 가장 높게 관측된 시간" if metric_label == "스타포스 성공률" else "과거 기록상 파괴율이 가장 낮게 관측된 시간"
    _render_summary_conclusion_card(title, best_row, label_col="hour_label")
    if best_row is not None:
        st.info(make_hour_insight_text(best_row) + " 단, 과거 기록 기반 분석이므로 향후 결과를 보장하지 않습니다.")

    cols = st.columns(2)
    _plot_with_baseline(
        cols[0],
        plot_hourly_rate(context["star_by_hour"], "success_rate", "시간별 성공률", min_attempts=context["controls"]["top_min_attempts"], show_low_sample=context["controls"]["show_low_sample"]),
        _bool_rate(df, "is_success"),
        "star_hour_success_chart",
    )
    _plot_with_baseline(
        cols[1],
        plot_hourly_rate(context["star_by_hour"], "destroy_rate", "시간별 파괴율", min_attempts=context["controls"]["top_min_attempts"], show_low_sample=context["controls"]["show_low_sample"]),
        _bool_rate(df, "is_destroyed"),
        "star_hour_destroy_chart",
    )

    band_cols = st.columns(2)
    _plot_with_baseline(
        band_cols[0],
        plot_hour_band_rate(context["star_by_hour_band"], "success_rate", "시간대별 성공률", min_attempts=context["controls"]["top_min_attempts"], show_low_sample=context["controls"]["show_low_sample"]),
        _bool_rate(df, "is_success"),
        "star_timeblock_success_chart",
    )
    _plot_with_baseline(
        band_cols[1],
        plot_hour_band_rate(context["star_by_hour_band"], "destroy_rate", "시간대별 파괴율", min_attempts=context["controls"]["top_min_attempts"], show_low_sample=context["controls"]["show_low_sample"]),
        _bool_rate(df, "is_destroyed"),
        "star_timeblock_destroy_chart",
    )

    analysis_mode = st.radio("시간별 스타포스 분석 단위", ["구간별", "전이 구간별", "공식 성공률 그룹별"], horizontal=True, key="star_hour_analysis_mode")
    if analysis_mode == "구간별":
        range_hour_df = add_confidence(
            _group_rate_summary(df, ["hour_label", "starforce_range"], "is_success", "success_rate"),
            "attempts",
        )
        if not range_hour_df.empty:
            st.plotly_chart(
                plot_rate_heatmap(
                    range_hour_df,
                    x_col="hour_label",
                    y_col="starforce_range",
                    rate_col="success_rate",
                    title="시간별 스타포스 구간별 성공률",
                    x_order=HOUR_LABEL_ORDER,
                ),
                width="stretch",
                key="star_hour_range_heatmap",
            )
        else:
            st.info("시간별 스타포스 구간별 성공률 데이터를 만들기 어렵습니다.")
    elif analysis_mode == "전이 구간별":
        transition_hour_df = add_confidence(
            _group_rate_summary(df, ["hour_label", "starforce_transition"], "is_success", "success_rate"),
            "attempts",
        )
        if not transition_hour_df.empty:
            transition_hour_df["_transition_sort"] = transition_hour_df["starforce_transition"].map(parse_transition_start)
            transition_hour_df = transition_hour_df.sort_values(["_transition_sort", "starforce_transition", "hour_label"]).drop(columns="_transition_sort")
            st.plotly_chart(
                plot_rate_heatmap(
                    transition_hour_df,
                    x_col="hour_label",
                    y_col="starforce_transition",
                    rate_col="success_rate",
                    title="시간별 스타포스 전이 구간별 성공률",
                    x_order=HOUR_LABEL_ORDER,
                ),
                width="stretch",
                key="star_hour_transition_heatmap",
            )
        else:
            st.info("시간별 스타포스 전이 구간별 성공률 데이터를 만들기 어렵습니다.")
    else:
        probability_df = context["success_probability_groups"]
        if probability_df is None or probability_df.empty:
            st.info("공식 성공률 그룹 분석은 기준 확률 데이터가 필요합니다.")
        else:
            probability_cols = st.columns(2)
            _plot_with_baseline(
                probability_cols[0],
                plot_item_rate(probability_df.rename(columns={"success_probability_group": "item_name"}), "success_rate", "공식 성공률 그룹별 성공률", min_attempts=context["controls"]["top_min_attempts"], show_low_sample=context["controls"]["show_low_sample"]),
                _bool_rate(df, "is_success"),
                "star_hour_probability_group_success_chart",
            )
            _plot_with_baseline(
                probability_cols[1],
                plot_item_rate(probability_df.rename(columns={"success_probability_group": "item_name", "destruction_rate": "destroy_rate"}), "destroy_rate", "공식 성공률 그룹별 파괴율", min_attempts=context["controls"]["top_min_attempts"], show_low_sample=context["controls"]["show_low_sample"]),
                _bool_rate(df, "is_destroyed"),
                "star_hour_probability_group_destroy_chart",
            )

    st.caption("TOP 5와 결론 카드는 최소 시도 수 기준을 만족한 항목만 사용합니다. 기준 미만 항목은 그래프에서 연하게 표시됩니다.")
    st.dataframe(
        _rank_dimension_table(
            context["star_by_hour"],
            "hour_label",
            metric_label,
            min_attempts=context["controls"]["top_min_attempts"],
            score_basis=context["controls"]["top_score_basis"],
        ),
        width="stretch",
        hide_index=True,
    )


def _render_cube_weekday_section(context: dict[str, Any]) -> None:
    df = context["effective_df"]
    if df.empty:
        st.info("조회된 잠재능력/큐브 데이터가 없습니다.")
        return
    metric_label = st.selectbox("잠재능력/큐브 지표 선택", ["주요옵션 출현률", "유효옵션 출현률", "등급업률"], index=1, key="cube_weekday_metric")
    rate_col, success_col = _cube_metric_meta(metric_label)
    best_row = _best_ranked_row(
        context["cube_by_weekday"],
        "weekday_kr",
        metric_label,
        min_attempts=context["controls"]["top_min_attempts"],
        score_basis=context["controls"]["top_score_basis"],
    )
    _render_summary_conclusion_card(f"과거 기록상 {metric_label}이 가장 높게 관측된 요일", best_row, label_col="weekday_kr")
    if best_row is not None:
        st.info(make_weekday_insight_text(best_row) + " 단, 과거 기록 기반 분석이므로 향후 결과를 보장하지 않습니다.")

    st.markdown("**전체 큐브 기준 요일별 비교**")
    top_cols = st.columns(2)
    with top_cols[0]:
        fig = plot_weekday_rate(
            context["cube_by_weekday"],
            "major_option_rate",
            "요일별 주요옵션 출현률",
            min_attempts=context["controls"]["top_min_attempts"],
            show_low_sample=context["controls"]["show_low_sample"],
        )
        add_average_line(fig, _bool_rate(df, "has_major_option"), "전체 평균")
        render_chart_card("요일별 주요옵션 출현률", fig, key="cube_weekday_major_chart")
    with top_cols[1]:
        fig = plot_weekday_rate(
            context["cube_by_weekday"],
            "effective_option_rate",
            "요일별 유효옵션 출현률",
            min_attempts=context["controls"]["top_min_attempts"],
            show_low_sample=context["controls"]["show_low_sample"],
        )
        add_average_line(fig, _bool_rate(df, "has_effective_option"), "전체 평균")
        render_chart_card("요일별 유효옵션 출현률", fig, key="cube_weekday_effective_chart")
    fig = plot_weekday_rate(
        context["cube_by_weekday"],
        "grade_up_rate",
        "요일별 등급업률",
        min_attempts=context["controls"]["top_min_attempts"],
        show_low_sample=context["controls"]["show_low_sample"],
    )
    add_average_line(fig, _bool_rate(df, "is_grade_up"), "전체 평균")
    render_chart_card("요일별 등급업률", fig, key="cube_weekday_gradeup_chart")

    st.markdown("**선택 큐브 타입 기준 요일별 비교**")
    cube_type_values = sorted(df["cube_type"].dropna().astype(str).unique().tolist()) if "cube_type" in df.columns else []
    if cube_type_values:
        weekday_col1, weekday_col2 = st.columns([2, 1])
        selected_cube_type = weekday_col1.selectbox("큐브 타입 선택", cube_type_values, key="cube_weekday_selected_type")
        average_mode = weekday_col2.selectbox(
            "평균선 기준",
            ["큐브 타입별 평균", "전체 큐브 평균", "평균선 없음"],
            index=0,
            key="cube_weekday_average_mode",
        )
        selected_cube_df = df[df["cube_type"].astype(str) == selected_cube_type].copy()
        selected_cube_weekday = summarize_cube_by_weekday(selected_cube_df)
        if not selected_cube_weekday.empty:
            fig = plot_weekday_rate(
                selected_cube_weekday,
                rate_col,
                f"{selected_cube_type} 기준 요일별 {metric_label}",
                min_attempts=context["controls"]["top_min_attempts"],
                show_low_sample=context["controls"]["show_low_sample"],
            )
            if average_mode == "전체 큐브 평균":
                add_average_line(fig, calculate_overall_metric_average(df, success_col), "전체 큐브 평균")
            elif average_mode == "큐브 타입별 평균":
                add_average_line(fig, calculate_overall_metric_average(selected_cube_df, success_col), f"{selected_cube_type} 평균")
            st.plotly_chart(fig, width="stretch", key="cube_selected_type_weekday_chart")
        else:
            st.info("선택한 큐브 타입의 요일별 비교 데이터를 만들기 어렵습니다.")
    else:
        st.info("큐브 타입 정보가 없어 선택 큐브 타입 요일별 비교를 표시하지 않습니다.")

    st.markdown("**큐브 타입 × 요일 관측 결과 맵**")
    if cube_type_values:
        cube_type_weekday = add_confidence(
            _group_rate_summary(df, ["cube_type", "weekday_kr"], success_col, rate_col),
            "attempts",
        )
        if not cube_type_weekday.empty:
            st.plotly_chart(
                plot_rate_heatmap(
                    cube_type_weekday,
                    x_col="weekday_kr",
                    y_col="cube_type",
                    rate_col=rate_col,
                    title=f"큐브 타입 × 요일 {metric_label}",
                    x_order=WEEKDAY_ORDER,
                ),
                width="stretch",
                key="cube_type_weekday_heatmap",
            )
        else:
            st.info("큐브 타입 × 요일 관측 결과 맵을 그릴 데이터가 부족합니다.")

    st.markdown("**큐브 타입별 비교**")
    cube_type_chart_df = context["cube_by_type"].copy()
    count_col = {
        "주요옵션 출현률": "major_option_count",
        "유효옵션 출현률": "effective_option_count",
        "등급업률": "grade_up_count",
    }[metric_label]
    overall_average = calculate_overall_metric_average(df, success_col)
    if not cube_type_chart_df.empty and count_col in cube_type_chart_df.columns and overall_average is not None:
        cube_type_chart_df["adjusted_rate"] = (
            cube_type_chart_df[count_col] + overall_average * 30
        ) / (cube_type_chart_df["attempts"] + 30)
    cube_type_fig = plot_cube_type_rate(
        cube_type_chart_df,
        rate_col,
        f"큐브 타입별 {metric_label}",
        min_attempts=context["controls"]["top_min_attempts"],
        show_low_sample=context["controls"]["show_low_sample"],
    )
    add_vertical_average_line(cube_type_fig, overall_average, "전체 큐브 평균")
    st.plotly_chart(cube_type_fig, width="stretch", key="cube_type_metric_chart")

    st.caption("TOP 5와 결론 카드는 최소 시도 수 기준을 만족한 항목만 사용합니다. 기준 미만 항목은 그래프에서 연하게 표시됩니다.")
    st.caption("시도 수 10회 미만 항목은 표본 부족으로 연하게 표시됩니다.")
    st.dataframe(
        _rank_dimension_table(
            context["cube_by_weekday"],
            "weekday_kr",
            metric_label,
            min_attempts=context["controls"]["top_min_attempts"],
            score_basis=context["controls"]["top_score_basis"],
        ),
        width="stretch",
        hide_index=True,
    )


def _render_star_weekday_section(context: dict[str, Any]) -> None:
    df = context["starforce_df"]
    if df.empty:
        st.info("조회된 스타포스 데이터가 없습니다.")
        return
    metric_label = st.selectbox("스타포스 지표 선택", ["스타포스 성공률", "스타포스 파괴율"], index=0, key="star_weekday_metric")
    best_row = _best_ranked_row(
        context["star_by_weekday"],
        "weekday_kr",
        metric_label,
        min_attempts=context["controls"]["top_min_attempts"],
        score_basis=context["controls"]["top_score_basis"],
    )
    title = "과거 기록상 성공률이 가장 높게 관측된 요일" if metric_label == "스타포스 성공률" else "과거 기록상 파괴율이 가장 낮게 관측된 요일"
    _render_summary_conclusion_card(title, best_row, label_col="weekday_kr")
    if best_row is not None:
        st.info(make_weekday_insight_text(best_row) + " 단, 과거 기록 기반 분석이므로 향후 결과를 보장하지 않습니다.")

    cols = st.columns(2)
    _plot_with_baseline(
        cols[0],
        plot_weekday_rate(context["star_by_weekday"], "success_rate", "요일별 성공률", min_attempts=context["controls"]["top_min_attempts"], show_low_sample=context["controls"]["show_low_sample"]),
        _bool_rate(df, "is_success"),
        "star_weekday_success_chart",
    )
    _plot_with_baseline(
        cols[1],
        plot_weekday_rate(context["star_by_weekday"], "destroy_rate", "요일별 파괴율", min_attempts=context["controls"]["top_min_attempts"], show_low_sample=context["controls"]["show_low_sample"]),
        _bool_rate(df, "is_destroyed"),
        "star_weekday_destroy_chart",
    )

    analysis_mode = st.radio("스타포스 분석 단위", ["구간별", "전이 구간별", "공식 성공률 그룹별"], horizontal=True, key="star_analysis_mode")
    if analysis_mode == "구간별":
        range_cols = st.columns(2)
        _plot_with_baseline(
            range_cols[0],
            plot_starforce_stage_rate(context["star_by_range"].rename(columns={"starforce_range": "before_starforce"}), "스타포스 구간별 성공률", min_attempts=context["controls"]["top_min_attempts"], show_low_sample=context["controls"]["show_low_sample"]),
            _bool_rate(df, "is_success"),
            "star_range_success_chart",
        )
        _plot_with_baseline(
            range_cols[1],
            plot_item_rate(context["star_by_range"].rename(columns={"starforce_range": "item_name"}), "destroy_rate", "스타포스 구간별 파괴율", min_attempts=context["controls"]["top_min_attempts"], show_low_sample=context["controls"]["show_low_sample"]),
            _bool_rate(df, "is_destroyed"),
            "star_range_destroy_chart",
        )
    elif analysis_mode == "전이 구간별":
        transition_success = _sort_transition_frame(context["star_by_transition"], "transition_label")
        transition_destroy = _sort_transition_frame(
            add_confidence(_group_rate_summary(df, ["starforce_transition"], "is_destroyed", "destroy_rate"), "attempts").rename(columns={"starforce_transition": "transition_label"}),
            "transition_label",
        )
        transition_cols = st.columns(2)
        _plot_with_baseline(
            transition_cols[0],
            plot_item_rate(transition_success.rename(columns={"transition_label": "item_name"}), "success_rate", "전이 구간별 성공률", min_attempts=context["controls"]["top_min_attempts"], show_low_sample=context["controls"]["show_low_sample"]),
            _bool_rate(df, "is_success"),
            "star_transition_success_chart",
        )
        _plot_with_baseline(
            transition_cols[1],
            plot_item_rate(transition_destroy.rename(columns={"transition_label": "item_name"}), "destroy_rate", "전이 구간별 파괴율", min_attempts=context["controls"]["top_min_attempts"], show_low_sample=context["controls"]["show_low_sample"]),
            _bool_rate(df, "is_destroyed"),
            "star_transition_destroy_chart",
        )
    else:
        probability_df = context["success_probability_groups"]
        if probability_df is None or probability_df.empty:
            st.info("공식 성공률 그룹 분석은 기준 확률 데이터가 필요합니다.")
        else:
            probability_cols = st.columns(2)
            _plot_with_baseline(
                probability_cols[0],
                plot_item_rate(probability_df.rename(columns={"success_probability_group": "item_name"}), "success_rate", "공식 성공률 그룹별 성공률", min_attempts=context["controls"]["top_min_attempts"], show_low_sample=context["controls"]["show_low_sample"]),
                _bool_rate(df, "is_success"),
                "star_probability_group_success_chart",
            )
            _plot_with_baseline(
                probability_cols[1],
                plot_item_rate(probability_df.rename(columns={"success_probability_group": "item_name", "destruction_rate": "destroy_rate"}), "destroy_rate", "공식 성공률 그룹별 파괴율", min_attempts=context["controls"]["top_min_attempts"], show_low_sample=context["controls"]["show_low_sample"]),
                _bool_rate(df, "is_destroyed"),
                "star_probability_group_destroy_chart",
            )
            st.dataframe(probability_df, width="stretch", hide_index=True)

    st.caption("TOP 5와 결론 카드는 최소 시도 수 기준을 만족한 항목만 사용합니다. 기준 미만 항목은 그래프에서 연하게 표시됩니다.")
    st.dataframe(
        _rank_dimension_table(
            context["star_by_weekday"],
            "weekday_kr",
            metric_label,
            min_attempts=context["controls"]["top_min_attempts"],
            score_basis=context["controls"]["top_score_basis"],
        ),
        width="stretch",
        hide_index=True,
    )


def _render_cube_condition_maps(context: dict[str, Any], metric_label: str, source_df: pd.DataFrame | None = None) -> None:
    df = source_df if source_df is not None else context["effective_df"]
    if df.empty:
        return
    cube_type_options = ["전체"] + sorted(df["cube_type"].dropna().astype(str).unique().tolist()) if "cube_type" in df.columns else ["전체"]
    selected_cube_type = st.selectbox("큐브 타입 필터", cube_type_options, key="cube_condition_map_type")
    map_df = df if selected_cube_type == "전체" else df[df["cube_type"].astype(str) == selected_cube_type]
    st.caption(f"히트맵 지표는 현재 선택한 기준 지표인 `{metric_label}`에 맞춰 표시합니다.")
    success_col, rate_col = {
        "주요옵션 출현률": ("has_major_option", "major_rate"),
        "유효옵션 출현률": ("has_effective_option", "effective_rate"),
        "등급업률": ("is_grade_up", "grade_up_rate"),
    }[metric_label]

    day_hour_df = add_confidence(_group_rate_summary(map_df, ["day_of_month_label", "hour_label"], success_col, rate_col), "attempts")
    weekday_hour_df = add_confidence(_group_rate_summary(map_df, ["weekday_kr", "hour_label"], success_col, rate_col), "attempts")

    left, right = st.columns(2)
    with left:
        if not day_hour_df.empty:
            st.plotly_chart(
                plot_rate_heatmap(
                    day_hour_df,
                    x_col="hour_label",
                    y_col="day_of_month_label",
                    rate_col=rate_col,
                    title="일자 × 시간 관측 결과 맵",
                    x_order=HOUR_LABEL_ORDER,
                    y_order=DAY_OF_MONTH_ORDER,
                ),
                width="stretch",
                key="cube_day_hour_heatmap",
            )
        else:
            st.info("일자 × 시간 관측 결과 맵을 그릴 데이터가 부족합니다.")
    with right:
        if not weekday_hour_df.empty:
            st.plotly_chart(
                plot_rate_heatmap(
                    weekday_hour_df,
                    x_col="hour_label",
                    y_col="weekday_kr",
                    rate_col=rate_col,
                    title="요일 × 시간 관측 결과 맵",
                    x_order=HOUR_LABEL_ORDER,
                    y_order=WEEKDAY_ORDER,
                ),
                width="stretch",
                key="cube_weekday_hour_heatmap",
            )
        else:
            st.info("요일 × 시간 관측 결과 맵을 그릴 데이터가 부족합니다.")


def _render_star_condition_maps(context: dict[str, Any], metric_label: str) -> None:
    df = context["starforce_df"]
    if df.empty:
        return
    range_options = ["전체"] + sorted(df["starforce_range"].dropna().astype(str).unique().tolist()) if "starforce_range" in df.columns else ["전체"]
    selected_range = st.selectbox("스타포스 구간 필터", range_options, key="star_condition_map_range")
    map_df = df if selected_range == "전체" else df[df["starforce_range"].astype(str) == selected_range]
    st.caption(f"히트맵 지표는 현재 선택한 기준 지표인 `{metric_label}`에 맞춰 표시합니다.")
    success_col, rate_col = {
        "성공률": ("is_success", "success_rate"),
        "파괴율": ("is_destroyed", "destroy_rate"),
        "스타포스 성공률": ("is_success", "success_rate"),
        "스타포스 파괴율": ("is_destroyed", "destroy_rate"),
    }[metric_label]

    day_hour_df = add_confidence(_group_rate_summary(map_df, ["day_of_month_label", "hour_label"], success_col, rate_col), "attempts")
    weekday_hour_df = add_confidence(_group_rate_summary(map_df, ["weekday_kr", "hour_label"], success_col, rate_col), "attempts")

    left, right = st.columns(2)
    with left:
        if not day_hour_df.empty:
            st.plotly_chart(
                plot_rate_heatmap(
                    day_hour_df,
                    x_col="hour_label",
                    y_col="day_of_month_label",
                    rate_col=rate_col,
                    title="일자 × 시간 관측 결과 맵",
                    x_order=HOUR_LABEL_ORDER,
                    y_order=DAY_OF_MONTH_ORDER,
                ),
                width="stretch",
                key="star_day_hour_heatmap",
            )
        else:
            st.info("일자 × 시간 관측 결과 맵을 그릴 데이터가 부족합니다.")
    with right:
        if not weekday_hour_df.empty:
            st.plotly_chart(
                plot_rate_heatmap(
                    weekday_hour_df,
                    x_col="hour_label",
                    y_col="weekday_kr",
                    rate_col=rate_col,
                    title="요일 × 시간 관측 결과 맵",
                    x_order=HOUR_LABEL_ORDER,
                    y_order=WEEKDAY_ORDER,
                ),
                width="stretch",
                key="star_weekday_hour_heatmap",
            )
        else:
            st.info("요일 × 시간 관측 결과 맵을 그릴 데이터가 부족합니다.")


def _render_reference_rows(rows: pd.DataFrame, key: str) -> None:
    rows = rows.copy()
    rows["reference_gap_label"] = rows["reference_gap_p"].map(format_gap_percent)
    st.dataframe(
        rows[
            [
                "condition_label",
                "metric_name",
                "actual_rate",
                "reference_rate",
                "reference_gap_label",
                "attempts",
                "confidence",
            ]
        ].rename(
            columns={
                "condition_label": "조건",
                "metric_name": "분석 대상",
                "actual_rate": "실제률",
                "reference_rate": "기준 확률",
                "reference_gap_label": "기준 확률 대비 차이",
                "attempts": "시도 수",
                "confidence": "신뢰도",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.plotly_chart(
        plot_reference_gap_bar(rows, "condition_label", "reference_gap_p", "기준 확률 대비 차이"),
        width="stretch",
        key=key,
    )


def _render_condition_cards(df: pd.DataFrame, tone: str) -> None:
    if df is None or df.empty:
        st.info("조건 랭킹에 표시할 데이터가 아직 부족합니다.")
        return
    theme_mode = st.session_state.get("resolved_theme_mode", "light")
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        title = make_good_condition_text(row) if tone == "good" else make_bad_condition_text(row)
        gap_variant = "success" if float(row.get("overall_gap_p", 0) or 0) >= 0 else "danger"
        if tone == "bad":
            gap_variant = "danger"
        chips = [
            render_metric_chip("보정률", format_percent(row["adjusted_rate"]), theme_mode, "accent"),
            render_metric_chip("전체 평균 대비", format_gap_percent(row["overall_gap_p"]), theme_mode, gap_variant),
        ]
        if pd.notna(row.get("reference_gap_p")):
            chips.append(render_metric_chip("기준 확률 대비", format_gap_percent(row["reference_gap_p"]), theme_mode, "info"))
        chips.append(render_metric_chip("시도 수", f"{int(row['attempts'])}회", theme_mode, "neutral"))
        chips.append(render_confidence_badge(str(row["confidence"]), theme_mode))
        st.markdown(
            f"""
<div class="maple-card maple-rank-card maple-rank-card-{tone}">
<div class="maple-rank-topline">
  <span class="maple-rank-pill">{idx}위</span>
  <div class="maple-rank-title">{row['condition_label']}</div>
</div>
<div class="maple-rank-subtitle">기준 지표 {row['metric_name']}</div>
<div class="maple-rank-rate">{format_percent(row['actual_rate'])}</div>
<div class="maple-chip-row">{''.join(chips)}</div>
<div class="maple-card-caption">{title}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def _label_day_of_month(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "day_of_month" not in df.columns:
        return pd.DataFrame() if df is None else df
    output = df.copy()
    output["day_of_month_label"] = output["day_of_month"].map(lambda value: f"{int(value)}일" if pd.notna(value) else None)
    return output


def _condition_rows_for_selected(
    context: dict[str, Any],
    target_type: str,
    grouping: str | list[str],
    metric_label: str,
    direction: str,
    source_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = source_df if source_df is not None else (context["effective_df"] if target_type == "cube" else context["starforce_df"])
    reference_lookup = _reference_lookup_for_metric(context, target_type, metric_label)
    rows = get_top_conditions_by_grouping(
        df=df,
        selected_grouping=grouping,
        target_type=target_type,
        metric_name=metric_label,
        direction=direction,
        min_attempts=context["controls"]["top_min_attempts"],
        score_basis=context["controls"]["top_score_basis"],
        reference_lookup=reference_lookup,
        dedup_strength=context["controls"]["top_dedup_strength"],
    )
    return rows


def _render_grouping_checkboxes(target_type: str) -> tuple[list[str], str]:
    st.markdown("#### TOP 5 분석 기준")
    if target_type == "cube":
        col1, col2, col3, col4 = st.columns(4)
        selected_flags = {
            "day": col1.checkbox("일자", value=False, key="cube_top_day"),
            "weekday": col2.checkbox("요일", value=False, key="cube_top_weekday"),
            "hour": col3.checkbox("시간", value=True, key="cube_top_hour"),
            "cube_type": col4.checkbox("큐브 타입", value=False, key="cube_top_type"),
        }
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
        selected_flags = {
            "day": col1.checkbox("일자", value=False, key="sf_top_day"),
            "weekday": col2.checkbox("요일", value=False, key="sf_top_weekday"),
            "hour": col3.checkbox("시간", value=True, key="sf_top_hour"),
            "range": col4.checkbox("스타포스 구간", value=False, key="sf_top_range"),
            "transition": col5.checkbox("전이 구간", value=False, key="sf_top_transition"),
        }

    group_cols, grouping_label = build_group_cols_from_checkboxes(selected_flags, target_type)
    if len(group_cols) >= 4:
        st.info("조건을 많이 조합할수록 표본 수가 줄어들 수 있습니다.")
    return group_cols, grouping_label


def _reference_lookup_for_metric(context: dict[str, Any], target_type: str, metric_label: str):
    if target_type == "cube":
        metric_code = {
            "주요옵션 출현률": "major_option_rate",
            "유효옵션 출현률": "effective_option_rate",
            "등급업률": "grade_up_rate",
        }[metric_label]
        return cube_reference_lookup(context["cube_ref_df"], metric_code)
    if metric_label == "스타포스 파괴율":
        return starforce_destroy_reference_lookup(context["star_ref_df"])
    return starforce_reference_lookup(context["star_ref_df"])


def _build_grouped_metric_cards(
    context: dict[str, Any],
    target_type: str,
    grouping: str | list[str],
    metric_labels: list[str],
    source_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    frames = []
    for metric_label in metric_labels:
        rows = _condition_rows_for_selected(context, target_type, grouping, metric_label, "good", source_df=source_df)
        if not rows.empty:
            frames.append(rows)
    if not frames:
        return []
    return group_condition_metrics(pd.concat(frames, ignore_index=True, sort=False))


def _render_grouped_metric_cards(cards: list[dict[str, Any]]) -> None:
    if not cards:
        st.info("같은 조건에서 여러 지표가 함께 눈에 띄는 항목이 아직 많지 않습니다.")
        return
    theme_mode = st.session_state.get("resolved_theme_mode", "light")
    for idx, card in enumerate(cards[:5], start=1):
        lines = []
        for metric in card["metrics"]:
            gap_text = format_gap_percent(metric["overall_gap_p"])
            line = (
                f"{metric['metric_name']}: 실제률 {format_percent(metric['actual_rate'])} / "
                f"보정률 {format_percent(metric.get('adjusted_rate'))} / 전체 평균 대비 {gap_text}"
            )
            if pd.notna(metric.get("reference_gap_p")):
                line += f" / 기준 확률 대비 {format_gap_percent(metric['reference_gap_p'])}"
            lines.append(line)
        chips = [
            render_metric_chip("시도 수", f"{card['attempts']}회", theme_mode, "neutral"),
            render_confidence_badge(str(card["confidence"]), theme_mode),
        ]
        st.markdown(
            f"""
<div class="maple-card maple-rank-card maple-rank-card-grouped">
<div class="maple-rank-topline">
  <span class="maple-rank-pill">{idx}위</span>
  <div class="maple-rank-title">{card['condition_label']}</div>
</div>
<div class="maple-chip-row">{''.join(chips)}</div>
<div class="maple-card-caption">{'<br>'.join(lines)}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    st.caption("위 결과는 과거 기록 기반의 참고용 통계이며, 향후 결과를 보장하지 않습니다.")


def _rank_dimension_table(
    summary_df: pd.DataFrame,
    label_col: str,
    metric_label: str,
    *,
    min_attempts: int,
    score_basis: str,
) -> pd.DataFrame:
    columns = [
        label_col,
        "actual_rate",
        "adjusted_rate",
        "overall_avg_rate",
        "reference_rate",
        "overall_gap_p",
        "reference_gap_p",
        "attempts",
        "success_count",
        "confidence",
        "score",
        "rank",
        "sample_note",
    ]
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(columns=columns)

    metric = metric_definition(metric_label)
    low_is_good = metric["low_is_good"]
    rate_col, count_col, gap_col = {
        "주요옵션 출현률": ("major_option_rate", "major_option_count", "major_overall_gap_p"),
        "유효옵션 출현률": ("effective_option_rate", "effective_option_count", "effective_overall_gap_p"),
        "등급업률": ("grade_up_rate", "grade_up_count", "grade_up_overall_gap_p"),
        "스타포스 성공률": ("success_rate", "success_count", "success_overall_gap_p"),
        "스타포스 파괴율": ("destroy_rate", "destroy_count", "destroy_overall_gap_p"),
    }[metric["label"]]
    if any(col not in summary_df.columns for col in [label_col, rate_col, count_col, "attempts"]):
        return pd.DataFrame(columns=columns)

    output = summary_df.copy()
    output["actual_rate"] = output[rate_col]
    output["success_count"] = output[count_col]
    output["overall_gap_p"] = output.get(gap_col)
    output["overall_avg_rate"] = output["actual_rate"] - output["overall_gap_p"]
    output["reference_rate"] = np.nan
    output["reference_gap_p"] = np.nan
    output["metric_name"] = metric["label"]
    output["condition_label"] = output[label_col]
    output["adjusted_rate"] = (
        output["success_count"] + output["overall_avg_rate"] * 30
    ) / (output["attempts"] + 30)
    output["adjusted_gap_p"] = output["adjusted_rate"] - output["overall_avg_rate"]
    output["confidence"] = output["attempts"].astype(int).map(confidence_label)
    output["sample_note"] = output["attempts"].map(lambda n: "표본 부족" if int(n) < min_attempts else "기준 충족")
    base = output["overall_gap_p"] if score_basis == "전체 평균 대비" else output["adjusted_gap_p"]
    if low_is_good:
        output["score"] = -base * np.log(output["attempts"] + 1)
    else:
        output["score"] = base * np.log(output["attempts"] + 1)
    eligible = output["attempts"] >= min_attempts
    ranked = output.loc[eligible].sort_values(["score", "attempts"], ascending=[False, False]).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    output = output.merge(ranked[[label_col, "rank"]], on=label_col, how="left")
    output["rank"] = output["rank"].fillna("-")
    display = output[columns].copy()
    if label_col == "day_of_month_label":
        display["_sort_key"] = display[label_col].str.replace("일", "", regex=False).astype(float)
    elif label_col == "hour_label":
        display["_sort_key"] = display[label_col].str.replace("시", "", regex=False).astype(float)
    elif label_col == "weekday_kr":
        order_map = {day: idx for idx, day in enumerate(WEEKDAY_ORDER)}
        display["_sort_key"] = display[label_col].map(order_map)
    else:
        display["_sort_key"] = range(len(display))
    display = display.sort_values("_sort_key").drop(columns="_sort_key")
    return display


def _best_ranked_row(
    summary_df: pd.DataFrame,
    label_col: str,
    metric_label: str,
    *,
    min_attempts: int,
    score_basis: str,
) -> pd.Series | None:
    ranked = _rank_dimension_table(summary_df, label_col, metric_label, min_attempts=min_attempts, score_basis=score_basis)
    if ranked.empty:
        return None
    ranked = ranked[ranked["rank"] != "-"].copy()
    if ranked.empty:
        return None
    ranked["_rank_num"] = pd.to_numeric(ranked["rank"], errors="coerce")
    ranked = ranked.sort_values("_rank_num")
    return ranked.iloc[0]


def _render_summary_conclusion_card(
    title: str,
    row: pd.Series | None,
    *,
    label_col: str,
) -> None:
    if row is None:
        st.info(f"{title}: 표본 기준을 만족하는 항목이 없습니다.")
        return
    theme_mode = st.session_state.get("resolved_theme_mode", "light")
    gap_variant = "success" if float(row.get("overall_gap_p", 0) or 0) >= 0 else "danger"
    chips = [
        render_metric_chip("보정률", format_percent(row["adjusted_rate"]), theme_mode, "accent"),
        render_metric_chip("전체 평균 대비", format_gap_percent(row["overall_gap_p"]), theme_mode, gap_variant),
    ]
    if pd.notna(row.get("reference_gap_p")):
        chips.append(render_metric_chip("기준 확률 대비", format_gap_percent(row["reference_gap_p"]), theme_mode, "info"))
    chips.append(render_metric_chip("시도 수", f"{int(row['attempts'])}회", theme_mode, "neutral"))
    chips.append(render_confidence_badge(str(row["confidence"]), theme_mode))
    st.markdown(
        f"""
<div class="maple-card maple-insight-card">
  <div class="maple-card-kicker">{title}</div>
  <div class="maple-title-xl">{row[label_col]}</div>
  <div class="maple-hero-rate">{format_percent(row['actual_rate'])}</div>
  <div class="maple-chip-row">{''.join(chips)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _group_rate_summary(df: pd.DataFrame, group_cols: list[str], success_col: str, rate_col: str) -> pd.DataFrame:
    columns = [*group_cols, "attempts", "success_count", rate_col]
    if df is None or df.empty or any(col not in df.columns for col in [*group_cols, success_col]):
        return pd.DataFrame(columns=columns)
    working = df.dropna(subset=group_cols).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working[success_col] = working[success_col].fillna(False).astype(bool)
    summary = (
        working.groupby(group_cols, as_index=False)
        .agg(attempts=(success_col, "size"), success_count=(success_col, "sum"))
    )
    summary[rate_col] = summary["success_count"] / summary["attempts"]
    return summary


def _sort_transition_frame(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if df is None or df.empty or label_col not in df.columns:
        return pd.DataFrame() if df is None else df
    output = df.copy()
    output["_transition_sort"] = output[label_col].map(parse_transition_start)
    output = output.sort_values(["_transition_sort", label_col], na_position="last").drop(columns="_transition_sort")
    return output


def _cube_metric_meta(metric_label: str) -> tuple[str, str]:
    return {
        "주요옵션 출현률": ("major_option_rate", "has_major_option"),
        "유효옵션 출현률": ("effective_option_rate", "has_effective_option"),
        "등급업률": ("grade_up_rate", "is_grade_up"),
    }[metric_label]


def calculate_overall_metric_average(df: pd.DataFrame, success_col: str) -> float | None:
    return _bool_rate(df, success_col)


def add_average_line(fig, avg_value: float | None, label: str) -> None:
    if avg_value is None or pd.isna(avg_value):
        return
    theme_mode = st.session_state.get("resolved_theme_mode", "light")
    palette = get_theme_palette(theme_mode)
    line_color = palette["warning"]
    if "큐브 타입" in label:
        line_color = "#A78BFA" if theme_mode == "dark" else "#7C6CF2"
    elif "기준 확률" in label:
        line_color = palette["success"]
    fig.add_hline(
        y=avg_value * 100,
        line_dash="dot",
        line_color=line_color,
        annotation_text=f"{label} {avg_value * 100:.1f}%",
        annotation_position="top left",
    )


def add_vertical_average_line(fig, avg_value: float | None, label: str) -> None:
    if avg_value is None or pd.isna(avg_value):
        return
    theme_mode = st.session_state.get("resolved_theme_mode", "light")
    palette = get_theme_palette(theme_mode)
    line_color = palette["warning"]
    if "큐브 타입" in label:
        line_color = "#A78BFA" if theme_mode == "dark" else "#7C6CF2"
    elif "기준 확률" in label:
        line_color = palette["success"]
    fig.add_vline(
        x=avg_value * 100,
        line_dash="dot",
        line_color=line_color,
        annotation_text=f"{label} {avg_value * 100:.1f}%",
        annotation_position="top right",
    )


def _plot_with_baseline(container, fig, baseline_rate: float | None, key: str) -> None:
    if baseline_rate is not None:
        add_average_line(fig, baseline_rate, "전체 평균")
    container.plotly_chart(fig, width="stretch", key=key)


def render_badge(text: str, variant: str, theme_mode: str) -> str:
    palette = get_theme_palette(theme_mode)
    variants = {
        "neutral": (palette["chip_bg"], palette["chip_border"], palette["text_secondary"]),
        "accent": (palette["chip_bg"], palette["accent"], palette["accent"]),
        "success": (palette["chip_bg"], palette["success"], palette["success"]),
        "danger": (palette["chip_bg"], palette["danger"], palette["danger"]),
        "warning": (palette["chip_bg"], palette["warning"], palette["warning"]),
        "info": (palette["chip_bg"], palette["info"], palette["info"]),
        "confidence-none": (palette["border"], palette["border"], palette["text_secondary"]),
        "confidence-low": ("#78350F" if theme_mode == "dark" else "#FFF5E8", "#FBBF24" if theme_mode == "dark" else "#F2C572", "#FBBF24" if theme_mode == "dark" else "#A65E00"),
        "confidence-medium": ("#1E3A8A" if theme_mode == "dark" else "#EEF4FF", "#93C5FD" if theme_mode == "dark" else "#9BC0FF", "#93C5FD" if theme_mode == "dark" else "#2F80ED"),
        "confidence-high": ("#064E3B" if theme_mode == "dark" else "#EAF7F1", "#6EE7B7" if theme_mode == "dark" else "#8CDDBF", "#6EE7B7" if theme_mode == "dark" else "#00A676"),
    }
    bg, border, color = variants.get(variant, variants["neutral"])
    return (
        f"<span class='maple-chip' style='background:{bg};border-color:{border};color:{color};'>"
        f"{text}</span>"
    )


def render_confidence_badge(confidence: str, theme_mode: str) -> str:
    mapping = {
        "참고 불가": "confidence-none",
        "낮음": "confidence-low",
        "보통": "confidence-medium",
        "높음": "confidence-high",
    }
    return render_badge(f"신뢰도 {confidence}", mapping.get(confidence, "neutral"), theme_mode)


def render_metric_chip(label: str, value: str, theme_mode: str, variant: str = "neutral") -> str:
    return render_badge(f"{label} {value}", variant, theme_mode)


def _first_row(df: pd.DataFrame) -> pd.Series | None:
    if df is None or df.empty:
        return None
    return df.iloc[0]


def _best_single_dimension_row(df: pd.DataFrame, allowed_groups: set[str]) -> pd.Series | None:
    if df is None or df.empty or "condition_group" not in df.columns:
        return None
    filtered = df[df["condition_group"].isin(allowed_groups)].copy()
    if filtered.empty:
        return None
    filtered = filtered.sort_values(["adjusted_score", "attempts"], ascending=[False, False])
    return filtered.iloc[0]


def _filter_condition_rows(df: pd.DataFrame, metric_filter: str) -> pd.DataFrame:
    if df is None or df.empty or metric_filter == "전체":
        return df
    return df[df["metric_name"] == metric_filter].copy()


def _filter_label(column: str) -> str:
    return {
        "character_name": "캐릭터",
        "world_name": "월드",
        "item_name": "아이템",
        "cube_type": "큐브 타입",
        "before_potential_grade": "사용 전 잠재 등급",
        "after_potential_grade": "사용 후 잠재 등급",
        "starforce_range": "스타포스 구간",
        "transition_label": "전이 구간",
        "weekday_kr": "요일",
        "hour_band": "시간대",
    }.get(column, column)


def _star_item_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "item_name" not in df.columns:
        return pd.DataFrame(columns=["item_name", "attempts", "success_count", "success_rate"])
    working = df.dropna(subset=["item_name"]).copy()
    if working.empty:
        return pd.DataFrame(columns=["item_name", "attempts", "success_count", "success_rate"])
    summary = (
        working.groupby("item_name", as_index=False)
        .agg(attempts=("item_name", "size"), success_count=("is_success", "sum"))
    )
    summary["success_rate"] = summary["success_count"] / summary["attempts"]
    return summary.sort_values("attempts", ascending=False)


def _bool_rate(df: pd.DataFrame, column: str) -> float | None:
    if df is None or df.empty or column not in df.columns:
        return None
    return float(df[column].fillna(False).astype(bool).mean())


def _inject_style(theme_mode: str) -> None:
    palette = get_theme_palette(theme_mode)
    st.markdown(
        f"""
<style>
    :root {{
        --page-bg: {palette["page_bg"]};
        --sidebar-bg: {palette["sidebar_bg"]};
        --card-bg: {palette["card_bg"]};
        --card-bg-soft: {palette["card_bg_soft"]};
        --border: {palette["border"]};
        --border-soft: {palette["border_soft"]};
        --text-primary: {palette["text_primary"]};
        --text-secondary: {palette["text_secondary"]};
        --text-muted: {palette["text_muted"]};
        --accent: {palette["accent"]};
        --accent-hover: {palette["accent_hover"]};
        --success: {palette["success"]};
        --warning: {palette["warning"]};
        --danger: {palette["danger"]};
        --info: {palette["info"]};
        --chip-bg: {palette["chip_bg"]};
        --chip-border: {palette["chip_border"]};
        --input-bg: {palette["input_bg"]};
        --input-text: {palette["input_text"]};
        --metric-bg: {palette["metric_bg"]};
    }}

    .stApp {{
        background: var(--page-bg);
        color: var(--text-primary);
    }}

    [data-testid="stAppViewContainer"],
    .main,
    .main > div {{
        background: var(--page-bg) !important;
    }}

    header[data-testid="stHeader"],
    [data-testid="stHeader"] {{
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }}

    div[data-testid="stDecoration"],
    [data-testid="stDecoration"],
    div[data-testid="stStatusWidget"],
    [data-testid="stStatusWidget"],
    #MainMenu {{
        display: none !important;
    }}

    div[data-testid="stToolbar"],
    [data-testid="stToolbar"] {{
        background: transparent !important;
        box-shadow: none !important;
        border: 0 !important;
    }}

    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    button[kind="header"][aria-label*="sidebar"],
    button[kind="header"][title*="sidebar"],
    button[kind="header"][aria-label*="Sidebar"],
    button[kind="header"][title*="Sidebar"] {{
        position: fixed !important;
        top: 0.85rem !important;
        left: 0.9rem !important;
        z-index: 1002 !important;
        width: 2.65rem !important;
        height: 2.65rem !important;
        border-radius: 999px !important;
        border: 1px solid var(--border) !important;
        background: var(--card-bg) !important;
        color: var(--text-primary) !important;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18) !important;
    }}

    div[data-testid="stToolbar"] [data-testid="collapsedControl"],
    div[data-testid="stToolbar"] [data-testid="stSidebarCollapsedControl"],
    div[data-testid="stToolbar"] button[aria-label*="sidebar"],
    div[data-testid="stToolbar"] button[title*="sidebar"],
    div[data-testid="stToolbar"] button[aria-label*="Sidebar"],
    div[data-testid="stToolbar"] button[title*="Sidebar"],
    [data-testid="stToolbar"] [data-testid="collapsedControl"],
    [data-testid="stToolbar"] [data-testid="stSidebarCollapsedControl"],
    [data-testid="stToolbar"] button[aria-label*="sidebar"],
    [data-testid="stToolbar"] button[title*="sidebar"],
    [data-testid="stToolbar"] button[aria-label*="Sidebar"],
    [data-testid="stToolbar"] button[title*="Sidebar"] {{
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}

    [data-testid="collapsedControl"]:hover,
    [data-testid="stSidebarCollapsedControl"]:hover,
    button[kind="header"][aria-label*="sidebar"]:hover,
    button[kind="header"][title*="sidebar"]:hover,
    button[kind="header"][aria-label*="Sidebar"]:hover,
    button[kind="header"][title*="Sidebar"]:hover {{
        background: var(--card-bg-soft) !important;
        border-color: var(--accent) !important;
    }}

    div[data-testid="stToolbar"] button,
    [data-testid="stToolbar"] button {{
        background: transparent !important;
        color: var(--text-primary) !important;
        border: 0 !important;
        box-shadow: none !important;
    }}

    div[data-testid="stToolbar"] button[title*="Deploy"],
    div[data-testid="stToolbar"] button[aria-label*="Deploy"],
    div[data-testid="stToolbar"] a[title*="Deploy"],
    div[data-testid="stToolbar"] a[aria-label*="Deploy"],
    div[data-testid="stToolbar"] [data-testid*="deploy"],
    div[data-testid="stToolbar"] [data-testid*="Deploy"],
    div[data-testid="stToolbar"] button[title*="Git"],
    div[data-testid="stToolbar"] button[aria-label*="Git"],
    div[data-testid="stToolbar"] a[title*="Git"],
    div[data-testid="stToolbar"] a[aria-label*="Git"],
    div[data-testid="stToolbar"] button[title*="Share"],
    div[data-testid="stToolbar"] button[aria-label*="Share"],
    div[data-testid="stToolbar"] a[title*="Share"],
    div[data-testid="stToolbar"] a[aria-label*="Share"],
    div[data-testid="stToolbar"] button[title*="share"],
    div[data-testid="stToolbar"] button[aria-label*="share"],
    div[data-testid="stToolbar"] a[title*="share"],
    div[data-testid="stToolbar"] a[aria-label*="share"] {{
        display: none !important;
    }}

    .stApp,
    .stApp p,
    .stApp label,
    .stApp span,
    .stApp div,
    .stMarkdown,
    .stCaption {{
        color: var(--text-primary);
    }}

    .block-container {{
        padding-top: 1.8rem;
        padding-bottom: 3.4rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1320px;
    }}

    section[data-testid="stSidebar"] {{
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border);
    }}

    section[data-testid="stSidebar"] * {{
        color: var(--text-primary) !important;
    }}

    section[data-testid="stSidebar"] .stButton button {{
        min-height: 44px !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {{
        margin-top: 0.9rem !important;
        margin-bottom: 0.6rem !important;
    }}

    section[data-testid="stSidebar"] .stCaption {{
        line-height: 1.5 !important;
    }}

    section[data-testid="stSidebar"] hr {{
        margin: 1.2rem 0 !important;
        border-color: var(--border) !important;
    }}

    div[data-testid="stMetric"],
    div[data-testid="stExpander"],
    div[data-testid="stAlert"],
    div[data-testid="stDataFrame"],
    div[data-testid="stForm"],
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border: 1px solid var(--border) !important;
        border-radius: 20px !important;
        background: var(--metric-bg) !important;
    }}

    div[data-testid="stTabs"] {{
        border: none !important;
        background: transparent !important;
        border-radius: 0 !important;
        padding: 0 !important;
        margin-bottom: 1.5rem !important;
    }}

    [data-testid="stMetric"] {{
        padding: 18px 20px;
        min-height: 168px;
    }}

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div,
    .stDateInput input,
    .stDateInput [data-baseweb="input"] > div,
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {{
        background: var(--input-bg) !important;
        color: var(--input-text) !important;
        border-color: var(--border-soft) !important;
    }}

    input,
    textarea,
    [data-baseweb="select"] input {{
        color: var(--input-text) !important;
    }}

    button[kind],
    .stButton button,
    .stDownloadButton button {{
        background: var(--accent) !important;
        color: #F8FAFC !important;
        border: 1px solid var(--accent) !important;
        border-radius: 12px !important;
    }}

    button[kind]:hover,
    .stButton button:hover,
    .stDownloadButton button:hover {{
        background: var(--accent-hover) !important;
        border-color: var(--accent-hover) !important;
    }}

    .stTabs [data-baseweb="tab-list"],
    div[data-testid="stTabs"] [role="tablist"] {{
        gap: 12px;
        margin-bottom: 1.35rem;
        flex-wrap: wrap;
        padding: 0.15rem 0 0.4rem 0;
    }}

    .stTabs [data-baseweb="tab"],
    div[data-testid="stTabs"] button[data-baseweb="tab"] {{
        background: var(--card-bg-soft);
        border: 1px solid var(--border);
        border-radius: 999px;
        color: var(--text-secondary);
        padding: 0.72rem 1.25rem;
        min-height: 48px;
        font-weight: 700;
        transition: background 0.18s ease, border-color 0.18s ease, color 0.18s ease, transform 0.18s ease;
    }}

    .stTabs [data-baseweb="tab"]:hover,
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {{
        background: var(--card-bg);
        border-color: var(--border-soft);
        color: var(--text-primary);
        transform: translateY(-1px);
    }}

    .stTabs [aria-selected="true"],
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
        background: var(--accent) !important;
        color: #FFFFFF !important;
        border-color: var(--accent) !important;
        box-shadow: 0 10px 22px rgba(37, 99, 235, 0.22) !important;
    }}

    .stTabs [data-baseweb="tab"]::after,
    .stTabs [data-baseweb="tab"]::before,
    div[data-testid="stTabs"] button[data-baseweb="tab"]::after,
    div[data-testid="stTabs"] button[data-baseweb="tab"]::before,
    [data-testid="stTabs"] [data-baseweb="tab-highlight"],
    [data-testid="stTabs"] [role="tablist"] + div {{
        display: none !important;
        background: transparent !important;
        border: 0 !important;
        height: 0 !important;
        box-shadow: none !important;
    }}

    .maple-card {{
        border: 1px solid var(--border-soft);
        border-radius: 24px;
        padding: 26px 30px;
        background: var(--card-bg);
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.10);
        margin-bottom: 26px;
    }}

    .maple-metric-card {{
        position: relative;
        min-height: 168px;
        background:
            linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0) 100%),
            var(--card-bg);
        overflow: hidden;
    }}

    .maple-metric-card::before {{
        content: "";
        position: absolute;
        inset: 0 auto auto 0;
        width: 100%;
        height: 4px;
        background: var(--accent);
        opacity: 0.95;
    }}

    .maple-metric-card-success::before {{
        background: var(--success);
    }}

    .maple-metric-card-warning::before {{
        background: var(--warning);
    }}

    .maple-metric-card-danger::before {{
        background: var(--danger);
    }}

    .maple-metric-card-neutral::before {{
        background: var(--border-soft);
    }}

    .maple-metric-label {{
        font-size: 14px;
        font-weight: 700;
        color: var(--text-secondary);
        margin-bottom: 14px;
    }}

    .maple-metric-value {{
        font-size: 42px;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: var(--text-primary);
        word-break: keep-all;
    }}

    .maple-profile-card {{
        margin-bottom: 26px;
    }}

    .maple-profile-hero {{
        border-top: 3px solid var(--accent);
        padding: 30px 34px;
    }}

    .maple-profile-layout {{
        display: flex;
        align-items: center;
        gap: 30px;
    }}

    .maple-profile-content {{
        flex: 1;
    }}

    .maple-profile-avatar {{
        width: 190px;
        height: 190px;
        border-radius: 34px;
        background: radial-gradient(circle at 50% 28%, rgba(96, 165, 250, 0.92) 0%, rgba(30, 41, 59, 0.96) 72%);
        color: #F8FAFC;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        font-weight: 700;
        overflow: hidden;
        border: 1px solid var(--border);
        flex-shrink: 0;
    }}

    .maple-profile-avatar-shell {{
        background: var(--card-bg-soft);
    }}

    .maple-profile-avatar-img {{
        width: 185px;
        height: 185px;
        object-fit: contain;
        display: block;
        max-width: 185px;
        max-height: 185px;
        padding: 0;
        transform: translateY(6%);
        transform-origin: center center;
        image-rendering: auto;
    }}

    .maple-profile-avatar-fallback {{
        font-size: 52px;
        font-weight: 800;
    }}

    .maple-title-lg {{
        font-size: 40px;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.1;
    }}

    .maple-title-xl {{
        font-size: 26px;
        font-weight: 800;
        margin-top: 6px;
        color: var(--text-primary);
    }}

    .maple-card-kicker {{
        font-size: 14px;
        font-weight: 700;
        color: var(--text-secondary);
        margin-bottom: 8px;
    }}

    .maple-rank-title {{
        font-size: 21px;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.35;
    }}

    .maple-rank-subtitle {{
        font-size: 13px;
        color: var(--text-secondary);
        margin-top: 8px;
    }}

    .maple-rank-rate,
    .maple-hero-rate {{
        font-size: 34px;
        font-weight: 800;
        color: var(--accent);
        margin-top: 12px;
        line-height: 1;
    }}

    .maple-chip-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 14px;
    }}

    .maple-chip {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid var(--chip-border);
        background: var(--chip-bg);
        font-size: 12px;
        line-height: 1.2;
        white-space: nowrap;
    }}

    .maple-card-caption {{
        font-size: 13px;
        color: var(--text-secondary);
        margin-top: 12px;
        line-height: 1.5;
    }}

    .maple-rank-topline {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }}

    .maple-rank-pill {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 48px;
        height: 32px;
        padding: 0 12px;
        border-radius: 999px;
        background: var(--card-bg-soft);
        border: 1px solid var(--border);
        color: var(--text-secondary);
        font-size: 13px;
        font-weight: 800;
    }}

    .maple-rank-card {{
        position: relative;
        background:
            linear-gradient(180deg, rgba(255,255,255,0.035) 0%, rgba(255,255,255,0) 100%),
            var(--card-bg);
        padding-left: 34px;
    }}

    .maple-rank-card::before {{
        content: "";
        position: absolute;
        left: 0;
        top: 20px;
        bottom: 20px;
        width: 5px;
        border-radius: 999px;
        background: var(--accent);
        opacity: 0.95;
    }}

    .maple-rank-card-good::before,
    .maple-rank-card-grouped::before {{
        background: var(--accent);
    }}

    .maple-rank-card-bad::before {{
        background: var(--danger);
    }}

    .maple-section-header {{
        margin: 8px 0 18px 0;
    }}

    .maple-section-header h3 {{
        margin: 0;
        color: var(--text-primary);
        font-size: 1.55rem;
        line-height: 1.2;
    }}

    .maple-section-header p {{
        margin: 8px 0 0 0;
        color: var(--text-secondary);
        font-size: 0.98rem;
        line-height: 1.6;
    }}

    .maple-text-secondary {{
        font-size: 15px;
        color: var(--text-secondary);
        line-height: 1.55;
    }}

    .maple-text-muted {{
        font-size: 12px;
        color: var(--text-muted);
    }}

    [data-testid="stExpander"] {{
        padding: 8px 14px !important;
        margin-bottom: 20px !important;
    }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        padding: 8px 10px !important;
        margin-bottom: 22px !important;
    }}

    .stColumn {{
        gap: 1rem !important;
    }}

    @media (max-width: 960px) {{
        .block-container {{
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
            padding-top: 1.1rem !important;
        }}

        .maple-card,
        .maple-profile-hero {{
            padding: 20px 20px !important;
        }}

        .maple-profile-layout {{
            flex-direction: column;
            align-items: flex-start;
            gap: 18px;
        }}

        .maple-profile-avatar {{
            width: 150px;
            height: 150px;
        }}

        .maple-profile-avatar-img {{
            width: 145px;
            height: 145px;
            max-width: 145px;
            max-height: 145px;
        }}

        .maple-title-lg {{
            font-size: 32px;
        }}
    }}

    [data-testid="stDataFrame"] *,
    [data-testid="stDataFrame"] table,
    [data-testid="stDataFrame"] th,
    [data-testid="stDataFrame"] td {{
        color: var(--text-primary) !important;
    }}
</style>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
