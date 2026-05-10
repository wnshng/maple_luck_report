from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from src.analytics.logger import log_event


def build_analysis_summary_signature(
    *,
    character_class: str | None,
    world_name: str | None,
    last_query_range: str | None,
    cube_attempts: int,
    potential_attempts: int,
    starforce_attempts: int,
) -> str:
    return "|".join(
        [
            str(character_class or ""),
            str(world_name or ""),
            str(last_query_range or ""),
            str(cube_attempts),
            str(potential_attempts),
            str(starforce_attempts),
        ]
    )


def build_analysis_summary_properties(context: dict[str, Any]) -> dict[str, Any] | None:
    cube_df = context.get("cube_df", pd.DataFrame())
    potential_df = context.get("potential_df", pd.DataFrame())
    starforce_df = context.get("starforce_df", pd.DataFrame())
    effective_df = context.get("effective_df", pd.DataFrame())
    controls = context.get("controls", {})
    selected_character = context.get("selected_character") or {}
    cube_summary = context.get("cube_summary") or {}
    star_summary = context.get("star_summary") or {}

    cube_attempts = int(len(cube_df)) if isinstance(cube_df, pd.DataFrame) else 0
    potential_attempts = int(len(potential_df)) if isinstance(potential_df, pd.DataFrame) else 0
    starforce_attempts = int(len(starforce_df)) if isinstance(starforce_df, pd.DataFrame) else 0
    total_record_count = cube_attempts + potential_attempts + starforce_attempts
    if total_record_count <= 0:
        return None

    start_date = controls.get("start_date")
    end_date = controls.get("end_date")
    date_range_days = _date_range_days(start_date, end_date)

    properties: dict[str, Any] = {
        "date_range_days": date_range_days,
        "total_record_count": total_record_count,
        "cube_attempts": cube_attempts,
        "potential_attempts": potential_attempts,
        "starforce_attempts": starforce_attempts,
        "has_cube_data": cube_attempts > 0,
        "has_potential_data": potential_attempts > 0,
        "has_starforce_data": starforce_attempts > 0,
        "major_option_rate": _safe_float(cube_summary.get("major_rate")),
        "effective_option_rate": _safe_float(cube_summary.get("effective_rate")),
        "grade_up_rate": _safe_bool_rate(effective_df, "is_grade_up"),
        "starforce_success_rate": _safe_float(star_summary.get("success_rate")),
        "starforce_destroy_rate": _safe_bool_rate(starforce_df, "is_destroyed"),
        "character_class": selected_character.get("character_class"),
        "world_name": selected_character.get("world_name"),
        "character_level_bucket": _level_bucket(selected_character.get("character_level")),
        "best_cube_day_of_month": _best_label(context.get("cube_by_day_of_month"), "day_of_month", "effective_option_rate", suffix="일"),
        "best_cube_hour": _best_label(context.get("cube_by_hour"), "hour_label", "effective_option_rate"),
        "best_cube_weekday": _best_label(context.get("cube_by_weekday"), "weekday_kr", "effective_option_rate"),
        "best_cube_type": _best_label(context.get("cube_by_type"), "cube_type", "effective_option_rate"),
        "best_starforce_day_of_month": _best_label(context.get("star_by_day_of_month"), "day_of_month", "success_rate", suffix="일"),
        "best_starforce_hour": _best_label(context.get("star_by_hour"), "hour_label", "success_rate"),
        "best_starforce_weekday": _best_label(context.get("star_by_weekday"), "weekday_kr", "success_rate"),
        "best_starforce_transition": _best_label(context.get("star_by_transition"), "transition_label", "success_rate"),
    }
    return properties


def track_analysis_summary(context: dict[str, Any]) -> None:
    try:
        properties = build_analysis_summary_properties(context)
        if not properties:
            return
        signature = build_analysis_summary_signature(
            character_class=properties.get("character_class"),
            world_name=properties.get("world_name"),
            last_query_range=st.session_state.get("last_query_range"),
            cube_attempts=int(properties.get("cube_attempts") or 0),
            potential_attempts=int(properties.get("potential_attempts") or 0),
            starforce_attempts=int(properties.get("starforce_attempts") or 0),
        )
        if st.session_state.get("_last_analysis_summary_signature") == signature:
            return
        st.session_state["_last_analysis_summary_signature"] = signature
        log_event("analysis_summary_generated", page_name="summary", properties=properties)
    except Exception:
        return


def _best_label(
    summary_df: pd.DataFrame | None,
    label_col: str,
    rate_col: str,
    *,
    suffix: str = "",
    min_attempts: int = 10,
) -> str | None:
    if summary_df is None or not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        return None
    if label_col not in summary_df.columns or rate_col not in summary_df.columns:
        return None

    attempt_col = "attempts" if "attempts" in summary_df.columns else "attempt_count" if "attempt_count" in summary_df.columns else None
    working = summary_df.copy()
    working = working[working[rate_col].notna()]
    if attempt_col:
        eligible = working[pd.to_numeric(working[attempt_col], errors="coerce").fillna(0) >= min_attempts]
        if not eligible.empty:
            working = eligible
    if working.empty:
        return None

    sort_attempt_col = attempt_col or label_col
    working = working.sort_values([rate_col, sort_attempt_col], ascending=[False, False])
    value = working.iloc[0].get(label_col)
    if pd.isna(value):
        return None
    if label_col == "day_of_month":
        return f"{int(value)}{suffix}"
    return str(value)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _safe_bool_rate(df: pd.DataFrame | None, column: str) -> float | None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or column not in df.columns:
        return None
    try:
        return float(df[column].fillna(False).astype(bool).mean())
    except Exception:
        return None


def _date_range_days(start_date: Any, end_date: Any) -> int | None:
    if isinstance(start_date, date) and isinstance(end_date, date):
        return (end_date - start_date).days + 1
    return None


def _level_bucket(level: Any) -> str:
    numeric = pd.to_numeric(level, errors="coerce")
    if pd.isna(numeric):
        return "unknown"
    base = int(numeric // 10 * 10)
    return f"{base}-{base + 9}"
