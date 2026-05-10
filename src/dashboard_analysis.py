from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import ROOT_DIR


REFERENCE_DIR = ROOT_DIR / "maplestory_reference"
STARFORCE_REFERENCE_PATH = REFERENCE_DIR / "starforce_probability_reference.csv"
POTENTIAL_REFERENCE_PATH = REFERENCE_DIR / "potential_probability_reference.csv"
MIN_TOP_ATTEMPTS = 10
PRIOR_SAMPLE_SIZE = 30
EVENT_COLUMNS = [
    "event_id",
    "event_name",
    "event_type",
    "start_date",
    "end_date",
    "apply_target",
    "source",
    "confidence",
    "note",
]


def _safe_condition_label(row: pd.Series, fallback_keys: tuple[str, ...] = ()) -> str:
    value = row.get("condition_label")
    if pd.notna(value):
        return str(value)
    for key in fallback_keys:
        fallback = row.get(key)
        if pd.notna(fallback):
            return str(fallback)
    return "조건 미확인"


def _safe_metric_label(row: pd.Series) -> str:
    value = row.get("metric_name")
    if pd.notna(value):
        return str(value)
    return "지표"


def confidence_label(attempts: int) -> str:
    if attempts < 10:
        return "참고 불가"
    if attempts < 30:
        return "낮음"
    if attempts < 100:
        return "보통"
    return "높음"


def add_confidence(df: pd.DataFrame, attempt_col: str = "attempts") -> pd.DataFrame:
    if df is None or df.empty or attempt_col not in df.columns:
        return pd.DataFrame() if df is None else df
    output = df.copy()
    output["confidence"] = output[attempt_col].fillna(0).astype(int).map(confidence_label)
    return output


def format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "기준 없음"
    return f"{value * 100:.1f}%"


def format_gap_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "기준 없음"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.1f}%p"


def load_reference_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def empty_event_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def normalize_event_df(event_df: pd.DataFrame | None) -> pd.DataFrame:
    if event_df is None or event_df.empty:
        return empty_event_df()
    output = event_df.copy()
    for col in EVENT_COLUMNS:
        if col not in output.columns:
            output[col] = None
    for date_col in ["start_date", "end_date"]:
        output[date_col] = pd.to_datetime(output[date_col], errors="coerce").dt.date
    output["event_name"] = output["event_name"].astype(str)
    output["event_type"] = output["event_type"].astype(str)
    output["apply_target"] = output["apply_target"].fillna("전체").astype(str)
    output["source"] = output["source"].fillna("manual").astype(str)
    output["confidence"] = output["confidence"].fillna("medium").astype(str)
    output["note"] = output["note"].fillna("").astype(str)
    return output[EVENT_COLUMNS].dropna(subset=["start_date", "end_date"], how="any").reset_index(drop=True)


def build_group_cols_from_checkboxes(selected_flags: dict[str, bool], target_type: str) -> tuple[list[str], str]:
    if target_type == "cube":
        ordered = [
            ("day", "day_of_month_label", "일자"),
            ("weekday", "weekday_kr", "요일"),
            ("hour", "hour_label", "시간"),
            ("cube_type", "cube_type", "큐브 타입"),
        ]
        default_cols = ["hour_label"]
        default_label = "시간"
    else:
        ordered = [
            ("day", "day_of_month_label", "일자"),
            ("weekday", "weekday_kr", "요일"),
            ("hour", "hour_label", "시간"),
            ("range", "starforce_range", "스타포스 구간"),
            ("transition", "starforce_transition", "스타포스 전이 구간"),
        ]
        default_cols = ["hour_label"]
        default_label = "시간"

    group_cols = [col for key, col, _ in ordered if selected_flags.get(key)]
    label_parts = [label for key, _, label in ordered if selected_flags.get(key)]
    if not group_cols:
        return default_cols, default_label
    return group_cols, " + ".join(label_parts)


def build_condition_label_from_group(row: pd.Series, group_cols: list[str], target_type: str) -> str:
    parts: list[str] = []
    for col in group_cols:
        value = row.get(col)
        if pd.isna(value):
            continue
        if col == "day_of_month":
            parts.append(f"{int(value)}일")
        elif col == "day_of_month_label":
            parts.append(str(value))
        elif col == "hour":
            parts.append(f"{int(value)}시")
        elif col == "hour_label":
            parts.append(str(value))
        elif col == "weekday":
            parts.append(str(value))
        elif col == "weekday_kr":
            parts.append(str(value))
        elif col == "event_tag":
            parts.append("이벤트 기간" if str(value) == "이벤트 기간" else "일반 기간")
        elif col == "event_type":
            parts.append(_pretty_event_type(str(value)))
        elif col == "starforce_transition":
            parts.append(str(value))
        else:
            parts.append(str(value))
    return " / ".join(parts) if parts else ("조건 미확인" if target_type == "cube" else "구간 미확인")


def infer_event_rows_from_records(cube_df: pd.DataFrame, potential_df: pd.DataFrame, starforce_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_row(event_date: Any, event_name: str, event_type: str, apply_target: str, note: str, confidence: str = "medium") -> None:
        if pd.isna(event_date):
            return
        rows.append(
            {
                "event_id": f"inferred-{apply_target}-{event_type}-{event_date}",
                "event_name": event_name,
                "event_type": event_type,
                "start_date": event_date,
                "end_date": event_date,
                "apply_target": apply_target,
                "source": "inferred",
                "confidence": confidence,
                "note": note,
            }
        )

    for frame, target_name in [(cube_df, "큐브"), (potential_df, "잠재능력")]:
        if frame is None or frame.empty:
            continue
        if "miracle_time_flag" in frame.columns and "event_date" in frame.columns:
            flagged = frame[frame["miracle_time_flag"].astype(str).str.lower().isin({"y", "yes", "true", "1"})]
            for event_date in flagged["event_date"].dropna().unique().tolist():
                add_row(event_date, "미라클 타임(추정)", "miracle_time", target_name, "miracle_time_flag 기반 추정")

    if starforce_df is not None and not starforce_df.empty and "event_date" in starforce_df.columns:
        for _, row in starforce_df.dropna(subset=["event_date"]).iterrows():
            event_date = row.get("event_date")
            discount = pd.to_numeric(row.get("event_cost_discount_rate"), errors="coerce")
            event_range = row.get("event_range")
            if pd.notna(discount) and float(discount) > 0:
                add_row(event_date, "스타포스 할인 이벤트(추정)", "starforce_discount", "스타포스", "event_cost_discount_rate 기반 추정")
            elif pd.notna(event_range) or pd.notna(row.get("event_plus_value")) or str(row.get("event_field_flag")).lower() in {"y", "yes", "true", "1"}:
                add_row(event_date, "스타포스 이벤트(추정)", "starforce_event", "스타포스", "event_range/event_plus_value 기반 추정", confidence="low")

    if not rows:
        return empty_event_df()
    return normalize_event_df(pd.DataFrame(rows)).drop_duplicates(
        subset=["event_name", "event_type", "start_date", "end_date", "apply_target"],
        keep="first",
    )


def attach_event_tags(record_df: pd.DataFrame, event_df: pd.DataFrame | None, target_type: str) -> pd.DataFrame:
    if record_df is None or record_df.empty:
        return pd.DataFrame() if record_df is None else record_df
    output = record_df.copy()
    normalized_events = normalize_event_df(event_df)
    output["event_tag"] = "일반 기간"
    output["event_name"] = None
    output["event_type"] = None
    output["event_source"] = None
    output["event_confidence"] = None
    output["is_event_period"] = False

    def safe_series(col: str) -> pd.Series:
        if col in output.columns:
            return output[col]
        return pd.Series([None] * len(output), index=output.index)

    if target_type in {"cube", "potential"} and "miracle_time_flag" in output.columns:
        inferred_mask = output["miracle_time_flag"].astype(str).str.lower().isin({"y", "yes", "true", "1"})
        output.loc[inferred_mask, ["event_tag", "event_name", "event_type", "event_source", "event_confidence", "is_event_period"]] = [
            "이벤트 기간",
            "미라클 타임(추정)",
            "miracle_time",
            "inferred",
            "medium",
            True,
        ]

    if target_type == "starforce":
        discount_mask = pd.to_numeric(safe_series("event_cost_discount_rate"), errors="coerce").fillna(0) > 0
        generic_mask = (
            safe_series("event_range").notna()
            | safe_series("event_plus_value").notna()
            | safe_series("event_field_flag").astype(str).str.lower().isin({"y", "yes", "true", "1"})
        )
        output.loc[generic_mask, ["event_tag", "event_name", "event_type", "event_source", "event_confidence", "is_event_period"]] = [
            "이벤트 기간",
            "스타포스 이벤트(추정)",
            "starforce_event",
            "inferred",
            "low",
            True,
        ]
        output.loc[discount_mask, ["event_tag", "event_name", "event_type", "event_source", "event_confidence", "is_event_period"]] = [
            "이벤트 기간",
            "스타포스 할인 이벤트(추정)",
            "starforce_discount",
            "inferred",
            "medium",
            True,
        ]

    if normalized_events.empty or "event_date" not in output.columns:
        return output

    target_aliases = {
        "cube": {"큐브", "잠재능력", "전체"},
        "potential": {"잠재능력", "전체"},
        "starforce": {"스타포스", "전체"},
    }[target_type]

    event_dates = pd.to_datetime(output["event_date"], errors="coerce").dt.date
    for _, event in normalized_events.iterrows():
        if event["apply_target"] not in target_aliases:
            continue
        mask = event_dates.between(event["start_date"], event["end_date"])
        if not mask.any():
            continue
        names = output.loc[mask, "event_name"].fillna("").astype(str)
        types = output.loc[mask, "event_type"].fillna("").astype(str)
        output.loc[mask, "event_tag"] = "이벤트 기간"
        output.loc[mask, "is_event_period"] = True
        output.loc[mask, "event_source"] = event["source"]
        output.loc[mask, "event_confidence"] = event["confidence"]
        output.loc[mask, "event_name"] = names.map(lambda value: _append_unique_label(value, str(event["event_name"])))
        output.loc[mask, "event_type"] = types.map(lambda value: _append_unique_label(value, str(event["event_type"])))

    output.loc[~output["is_event_period"], "event_name"] = output.loc[~output["is_event_period"], "event_name"].fillna("일반 기간")
    output.loc[~output["is_event_period"], "event_type"] = output.loc[~output["is_event_period"], "event_type"].fillna("normal")
    return output


def compare_event_periods(df: pd.DataFrame, target_type: str) -> pd.DataFrame:
    columns = ["event_group", "attempts", "major_option_rate", "effective_option_rate", "grade_up_rate", "success_rate", "destroy_rate", "confidence"]
    if df is None or df.empty or "is_event_period" not in df.columns:
        return pd.DataFrame(columns=columns)
    working = df.copy()
    working["event_group"] = working["is_event_period"].map(lambda value: "이벤트 기간" if bool(value) else "일반 기간")
    rows: list[dict[str, Any]] = []
    for event_group, group in working.groupby("event_group"):
        row: dict[str, Any] = {"event_group": event_group, "attempts": int(len(group))}
        if target_type == "cube":
            for success_col, rate_col in [("has_major_option", "major_option_rate"), ("has_effective_option", "effective_option_rate"), ("is_grade_up", "grade_up_rate")]:
                row[rate_col] = _bool_rate(group, success_col)
        else:
            row["success_rate"] = _bool_rate(group, "is_success")
            row["destroy_rate"] = _bool_rate(group, "is_destroyed")
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    output = pd.DataFrame(rows)
    return add_confidence(output, "attempts")


def build_success_probability_group(starforce_df: pd.DataFrame, starforce_ref_df: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "success_probability_group",
        "before_star_list",
        "attempts",
        "success_rate",
        "destruction_rate",
        "adjusted_success_rate",
        "adjusted_destruction_rate",
        "confidence",
    ]
    if starforce_df is None or starforce_df.empty or starforce_ref_df is None or starforce_ref_df.empty:
        return pd.DataFrame(columns=columns)
    if not {"before_starforce", "success_rate_ref"}.issubset(starforce_ref_df.columns):
        return pd.DataFrame(columns=columns)

    working = starforce_df.copy()
    working["before_starforce"] = pd.to_numeric(working.get("before_starforce"), errors="coerce")
    ref = starforce_ref_df[["before_starforce", "success_rate_ref"]].copy()
    ref["before_starforce"] = pd.to_numeric(ref["before_starforce"], errors="coerce")
    merged = working.merge(ref, on="before_starforce", how="left")
    merged = merged.dropna(subset=["success_rate_ref"])
    if merged.empty:
        return pd.DataFrame(columns=columns)

    merged["success_probability_group"] = merged["success_rate_ref"].map(lambda value: f"성공확률 {value * 100:.0f}% 구간")
    summary = (
        merged.groupby("success_probability_group", as_index=False)
        .agg(
            attempts=("success_probability_group", "size"),
            success_count=("is_success", "sum"),
            destroy_count=("is_destroyed", "sum"),
            before_star_list=("before_starforce", lambda series: ", ".join(str(int(v)) for v in sorted(series.dropna().unique().tolist()))),
            reference_rate=("success_rate_ref", "first"),
        )
    )
    overall_success = _bool_rate(merged, "is_success") or 0
    overall_destroy = _bool_rate(merged, "is_destroyed") or 0
    summary["success_rate"] = summary["success_count"] / summary["attempts"]
    summary["destruction_rate"] = summary["destroy_count"] / summary["attempts"]
    summary["adjusted_success_rate"] = (summary["success_count"] + overall_success * PRIOR_SAMPLE_SIZE) / (summary["attempts"] + PRIOR_SAMPLE_SIZE)
    summary["adjusted_destruction_rate"] = (summary["destroy_count"] + overall_destroy * PRIOR_SAMPLE_SIZE) / (summary["attempts"] + PRIOR_SAMPLE_SIZE)
    return add_confidence(summary, "attempts")[columns]


def parse_transition_start(label: str | None) -> int | None:
    if label is None or pd.isna(label):
        return None
    text = str(label)
    left = text.split("→", 1)[0].replace("성", "").strip()
    return int(left) if left.isdigit() else None


def _append_unique_label(existing: str, new_label: str) -> str:
    values = [value.strip() for value in str(existing).split(",") if value and value.strip()]
    if new_label not in values:
        values.append(new_label)
    return ", ".join(values)


def _pretty_event_type(value: str) -> str:
    mapping = {
        "miracle_time": "미라클 타임",
        "starforce_discount": "스타포스 할인",
        "starforce_event": "스타포스 이벤트",
        "normal": "일반 기간",
        "cube_event": "큐브 이벤트",
        "potential_event": "잠재능력 이벤트",
    }
    return mapping.get(value, value)


def build_condition_group(df: pd.DataFrame, selected_grouping: str, target_type: str) -> tuple[list[str], str]:
    cube_mapping = {
        "일자": ["day_of_month_label"],
        "시간": ["hour_label"],
        "요일": ["weekday_kr"],
        "큐브 타입": ["cube_type"],
        "일자 + 시간": ["day_of_month_label", "hour_label"],
        "일자 + 큐브 타입": ["day_of_month_label", "cube_type"],
        "시간 + 큐브 타입": ["hour_label", "cube_type"],
        "요일 + 시간": ["weekday_kr", "hour_label"],
        "요일 + 큐브 타입": ["weekday_kr", "cube_type"],
        "일자 + 시간 + 큐브 타입": ["day_of_month_label", "hour_label", "cube_type"],
        "요일 + 시간 + 큐브 타입": ["weekday_kr", "hour_label", "cube_type"],
    }
    star_mapping = {
        "일자": ["day_of_month_label"],
        "시간": ["hour_label"],
        "요일": ["weekday_kr"],
        "스타포스 구간": ["starforce_range"],
        "일자 + 시간": ["day_of_month_label", "hour_label"],
        "일자 + 스타포스 구간": ["day_of_month_label", "starforce_range"],
        "시간 + 스타포스 구간": ["hour_label", "starforce_range"],
        "요일 + 시간": ["weekday_kr", "hour_label"],
        "요일 + 스타포스 구간": ["weekday_kr", "starforce_range"],
        "일자 + 시간 + 스타포스 구간": ["day_of_month_label", "hour_label", "starforce_range"],
        "요일 + 시간 + 스타포스 구간": ["weekday_kr", "hour_label", "starforce_range"],
    }
    mapping = cube_mapping if target_type == "cube" else star_mapping
    group_cols = mapping[selected_grouping]
    return group_cols, " / ".join(group_cols)


def get_top_conditions_by_grouping(
    df: pd.DataFrame,
    selected_grouping: str | list[str] | tuple[str, ...],
    target_type: str,
    metric_name: str,
    direction: str,
    min_attempts: int,
    score_basis: str,
    *,
    reference_lookup: Callable[[pd.Series], float | None] | None = None,
    dedup_strength: str = "보통",
) -> pd.DataFrame:
    columns = [
        "condition_label",
        "target_type",
        "metric_name",
        "actual_rate",
        "adjusted_rate",
        "overall_avg_rate",
        "reference_rate",
        "overall_gap_p",
        "reference_gap_p",
        "adjusted_gap_p",
        "adjusted_reference_gap_p",
        "attempts",
        "success_count",
        "confidence",
        "score",
        "source_record_count",
        "sample_note",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    metric = metric_definition(metric_name)
    success_col = metric["success_col"]
    low_is_good = metric["low_is_good"]
    if success_col not in df.columns:
        return pd.DataFrame(columns=columns)

    if isinstance(selected_grouping, (list, tuple)):
        group_cols = list(selected_grouping)
    else:
        group_cols, _ = build_condition_group(df, selected_grouping, target_type)
    if any(col not in df.columns for col in group_cols):
        return pd.DataFrame(columns=columns)

    working = df.dropna(subset=group_cols).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working[success_col] = working[success_col].fillna(False).astype(bool)
    overall_avg_rate = _bool_rate(working, success_col)

    summary = (
        working.groupby(group_cols, as_index=False)
        .agg(
            attempts=(success_col, "size"),
            success_count=(success_col, "sum"),
        )
    )
    summary["actual_rate"] = summary["success_count"] / summary["attempts"]
    summary["overall_avg_rate"] = overall_avg_rate
    summary["reference_rate"] = (
        summary.apply(reference_lookup, axis=1) if reference_lookup is not None else np.nan
    )
    summary["adjusted_rate"] = (
        summary["success_count"] + overall_avg_rate * PRIOR_SAMPLE_SIZE
    ) / (summary["attempts"] + PRIOR_SAMPLE_SIZE)
    summary["overall_gap_p"] = summary["actual_rate"] - summary["overall_avg_rate"]
    summary["reference_gap_p"] = summary["actual_rate"] - summary["reference_rate"]
    summary["adjusted_gap_p"] = summary["adjusted_rate"] - summary["overall_avg_rate"]
    summary["adjusted_reference_gap_p"] = summary["adjusted_rate"] - summary["reference_rate"]
    summary["confidence"] = summary["attempts"].astype(int).map(confidence_label)
    summary["source_record_count"] = summary["attempts"]
    summary["sample_note"] = summary["attempts"].map(lambda n: "표본 부족" if int(n) < min_attempts else "기준 충족")
    summary["condition_label"] = summary.apply(
        lambda row: build_condition_label_from_group(row, group_cols, target_type),
        axis=1,
    )
    summary["target_type"] = target_type
    summary["metric_name"] = metric_name

    summary = summary[summary["attempts"] >= min_attempts].copy()
    if summary.empty:
        return pd.DataFrame(columns=columns)

    score_col = _score_basis_column(score_basis, has_reference=summary["reference_rate"].notna().any())
    summary["_base_score"] = summary[score_col]
    if low_is_good:
        summary["score"] = (
            -summary["_base_score"] if direction == "good" else summary["_base_score"]
        ) * np.log(summary["attempts"] + 1)
    else:
        summary["score"] = (
            summary["_base_score"] if direction == "good" else -summary["_base_score"]
        ) * np.log(summary["attempts"] + 1)

    summary = summary.sort_values(["score", "attempts"], ascending=[False, False])
    summary = deduplicate_top_conditions(summary[columns], dedup_strength)
    return summary.head(5).reset_index(drop=True)


def deduplicate_top_conditions(df: pd.DataFrame, dedup_strength: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    output = df.copy()
    output["_label_key"] = output["condition_label"].astype(str).str.replace(" ", "", regex=False)
    output["_actual_key"] = output["actual_rate"].round(6)
    output["_attempt_key"] = output["attempts"].astype(int)

    if dedup_strength == "약함":
        return output.drop_duplicates(subset=["target_type", "metric_name", "_label_key"]).drop(
            columns=["_label_key", "_actual_key", "_attempt_key"],
            errors="ignore",
        )
    if dedup_strength == "강함":
        return output.drop_duplicates(
            subset=["target_type", "metric_name", "_actual_key", "_attempt_key"]
        ).drop(columns=["_label_key", "_actual_key", "_attempt_key"], errors="ignore")
    return output.drop_duplicates(
        subset=["target_type", "metric_name", "_label_key", "_actual_key", "_attempt_key"]
    ).drop(columns=["_label_key", "_actual_key", "_attempt_key"], errors="ignore")


def group_condition_metrics(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    grouped_cards: list[dict[str, Any]] = []
    for (target_type, condition_label), group in df.groupby(["target_type", "condition_label"], dropna=False):
        metrics = []
        for _, row in group.iterrows():
            metrics.append(
                {
                    "metric_name": row["metric_name"],
                    "actual_rate": row["actual_rate"],
                    "adjusted_rate": row.get("adjusted_rate"),
                    "overall_gap_p": row["overall_gap_p"],
                    "reference_gap_p": row.get("reference_gap_p"),
                }
            )
        first = group.iloc[0]
        grouped_cards.append(
            {
                "target_type": target_type,
                "condition_label": condition_label,
                "attempts": int(first["attempts"]),
                "confidence": first["confidence"],
                "metrics": metrics,
                "metric_count": len(metrics),
                "strongest_abs_gap": float(group["overall_gap_p"].abs().max()),
            }
        )
    grouped_cards.sort(
        key=lambda card: (
            card["metric_count"],
            card["strongest_abs_gap"],
            card["attempts"],
        ),
        reverse=True,
    )
    return grouped_cards


def metric_definition(metric_name: str) -> dict[str, Any]:
    mapping = {
        "major_option_rate": {"success_col": "has_major_option", "label": "주요옵션 출현률", "low_is_good": False},
        "주요옵션 출현률": {"success_col": "has_major_option", "label": "주요옵션 출현률", "low_is_good": False},
        "effective_option_rate": {"success_col": "has_effective_option", "label": "유효옵션 출현률", "low_is_good": False},
        "유효옵션 출현률": {"success_col": "has_effective_option", "label": "유효옵션 출현률", "low_is_good": False},
        "grade_up_rate": {"success_col": "is_grade_up", "label": "등급업률", "low_is_good": False},
        "등급업률": {"success_col": "is_grade_up", "label": "등급업률", "low_is_good": False},
        "success_rate": {"success_col": "is_success", "label": "스타포스 성공률", "low_is_good": False},
        "스타포스 성공률": {"success_col": "is_success", "label": "스타포스 성공률", "low_is_good": False},
        "destruction_rate": {"success_col": "is_destroyed", "label": "스타포스 파괴율", "low_is_good": True},
        "destroy_rate": {"success_col": "is_destroyed", "label": "스타포스 파괴율", "low_is_good": True},
        "스타포스 파괴율": {"success_col": "is_destroyed", "label": "스타포스 파괴율", "low_is_good": True},
        "파괴율": {"success_col": "is_destroyed", "label": "스타포스 파괴율", "low_is_good": True},
    }
    return mapping[metric_name]


def _score_basis_column(score_basis: str, has_reference: bool) -> str:
    if score_basis == "전체 평균 대비":
        return "overall_gap_p"
    if score_basis == "기준 확률 대비":
        return "reference_gap_p" if has_reference else "overall_gap_p"
    return "adjusted_reference_gap_p" if has_reference else "adjusted_gap_p"


def summarize_cube_by_date(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_date",
        "attempts",
        "major_option_count",
        "major_option_rate",
        "effective_option_count",
        "effective_option_rate",
        "grade_up_count",
        "grade_up_rate",
        "major_overall_gap_p",
        "effective_overall_gap_p",
        "grade_up_overall_gap_p",
        "confidence",
    ]
    if df is None or df.empty or "event_date" not in df.columns:
        return pd.DataFrame(columns=columns)

    working = df.dropna(subset=["event_date"]).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)

    overall_major = _bool_rate(working, "has_major_option")
    overall_effective = _bool_rate(working, "has_effective_option")
    overall_grade = _bool_rate(working, "is_grade_up")

    summary = (
        working.groupby("event_date", as_index=False)
        .agg(
            attempts=("event_date", "size"),
            major_option_count=("has_major_option", "sum"),
            effective_option_count=("has_effective_option", "sum"),
            grade_up_count=("is_grade_up", "sum"),
        )
        .sort_values("event_date")
    )
    summary["major_option_rate"] = summary["major_option_count"] / summary["attempts"]
    summary["effective_option_rate"] = summary["effective_option_count"] / summary["attempts"]
    summary["grade_up_rate"] = summary["grade_up_count"] / summary["attempts"]
    summary["major_overall_gap_p"] = summary["major_option_rate"] - overall_major
    summary["effective_overall_gap_p"] = summary["effective_option_rate"] - overall_effective
    summary["grade_up_overall_gap_p"] = summary["grade_up_rate"] - overall_grade
    return add_confidence(summary, "attempts")[columns]


def summarize_cube_by_day_of_month(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "day_of_month",
        "attempts",
        "major_option_count",
        "major_option_rate",
        "effective_option_count",
        "effective_option_rate",
        "grade_up_count",
        "grade_up_rate",
        "major_overall_gap_p",
        "effective_overall_gap_p",
        "grade_up_overall_gap_p",
        "confidence",
    ]
    if df is None or df.empty or "day_of_month" not in df.columns:
        return pd.DataFrame(columns=columns)

    working = df.dropna(subset=["day_of_month"]).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)

    overall_major = _bool_rate(working, "has_major_option")
    overall_effective = _bool_rate(working, "has_effective_option")
    overall_grade = _bool_rate(working, "is_grade_up")

    summary = (
        working.groupby("day_of_month", as_index=False)
        .agg(
            attempts=("day_of_month", "size"),
            major_option_count=("has_major_option", "sum"),
            effective_option_count=("has_effective_option", "sum"),
            grade_up_count=("is_grade_up", "sum"),
        )
        .sort_values("day_of_month")
    )
    summary["major_option_rate"] = summary["major_option_count"] / summary["attempts"]
    summary["effective_option_rate"] = summary["effective_option_count"] / summary["attempts"]
    summary["grade_up_rate"] = summary["grade_up_count"] / summary["attempts"]
    summary["major_overall_gap_p"] = summary["major_option_rate"] - overall_major
    summary["effective_overall_gap_p"] = summary["effective_option_rate"] - overall_effective
    summary["grade_up_overall_gap_p"] = summary["grade_up_rate"] - overall_grade
    return add_confidence(summary, "attempts")[columns]


def summarize_cube_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    return _cube_rate_summary(df, "weekday_kr")


def summarize_cube_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    summary = _cube_rate_summary(df, "hour")
    if not summary.empty:
        summary["hour_label"] = summary["hour"].map(lambda value: f"{int(value)}시")
    return summary


def summarize_cube_by_hour_band(df: pd.DataFrame) -> pd.DataFrame:
    return _cube_rate_summary(df, "hour_band")


def summarize_cube_by_type(df: pd.DataFrame) -> pd.DataFrame:
    return _cube_rate_summary(df, "cube_type")


def summarize_starforce_by_date(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_date",
        "attempts",
        "success_count",
        "success_rate",
        "destroy_count",
        "destroy_rate",
        "success_overall_gap_p",
        "destroy_overall_gap_p",
        "confidence",
    ]
    if df is None or df.empty or "event_date" not in df.columns:
        return pd.DataFrame(columns=columns)

    working = df.dropna(subset=["event_date"]).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)

    overall_success = _bool_rate(working, "is_success")
    overall_destroy = _bool_rate(working, "is_destroyed")
    summary = (
        working.groupby("event_date", as_index=False)
        .agg(
            attempts=("event_date", "size"),
            success_count=("is_success", "sum"),
            destroy_count=("is_destroyed", "sum"),
        )
        .sort_values("event_date")
    )
    summary["success_rate"] = summary["success_count"] / summary["attempts"]
    summary["destroy_rate"] = summary["destroy_count"] / summary["attempts"]
    summary["success_overall_gap_p"] = summary["success_rate"] - overall_success
    summary["destroy_overall_gap_p"] = summary["destroy_rate"] - overall_destroy
    return add_confidence(summary, "attempts")[columns]


def summarize_starforce_by_day_of_month(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "day_of_month",
        "attempts",
        "success_count",
        "success_rate",
        "destroy_count",
        "destroy_rate",
        "success_overall_gap_p",
        "destroy_overall_gap_p",
        "confidence",
    ]
    if df is None or df.empty or "day_of_month" not in df.columns:
        return pd.DataFrame(columns=columns)

    working = df.dropna(subset=["day_of_month"]).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)

    overall_success = _bool_rate(working, "is_success")
    overall_destroy = _bool_rate(working, "is_destroyed")
    summary = (
        working.groupby("day_of_month", as_index=False)
        .agg(
            attempts=("day_of_month", "size"),
            success_count=("is_success", "sum"),
            destroy_count=("is_destroyed", "sum"),
        )
        .sort_values("day_of_month")
    )
    summary["success_rate"] = summary["success_count"] / summary["attempts"]
    summary["destroy_rate"] = summary["destroy_count"] / summary["attempts"]
    summary["success_overall_gap_p"] = summary["success_rate"] - overall_success
    summary["destroy_overall_gap_p"] = summary["destroy_rate"] - overall_destroy
    return add_confidence(summary, "attempts")[columns]


def summarize_starforce_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    return _star_rate_summary(df, "weekday_kr")


def summarize_starforce_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    summary = _star_rate_summary(df, "hour")
    if not summary.empty:
        summary["hour_label"] = summary["hour"].map(lambda value: f"{int(value)}시")
    return summary


def summarize_starforce_by_hour_band(df: pd.DataFrame) -> pd.DataFrame:
    return _star_rate_summary(df, "hour_band")


def summarize_starforce_by_range(df: pd.DataFrame) -> pd.DataFrame:
    return _star_rate_summary(df, "starforce_range")


def summarize_starforce_by_transition(df: pd.DataFrame) -> pd.DataFrame:
    return _star_rate_summary(df, "transition_label")


def build_condition_scores(
    df: pd.DataFrame,
    *,
    success_col: str,
    metric_name: str,
    target_type: str,
    group_defs: list[tuple[str, list[str]]],
    higher_is_better: bool,
    reference_lookup: Callable[[pd.Series], float | None] | None = None,
) -> pd.DataFrame:
    columns = [
        "condition_label",
        "condition_group",
        "target_type",
        "metric_name",
        "actual_rate",
        "overall_avg_rate",
        "reference_rate",
        "overall_gap_p",
        "reference_gap_p",
        "attempts",
        "success_count",
        "confidence",
        "adjusted_score",
        "direction",
        "interpretation",
    ]
    if df is None or df.empty or success_col not in df.columns:
        return pd.DataFrame(columns=columns)

    overall_avg_rate = _bool_rate(df, success_col)
    outputs: list[pd.DataFrame] = []
    direction = 1 if higher_is_better else -1

    for group_name, group_cols in group_defs:
        if any(col not in df.columns for col in group_cols):
            continue
        grouped = df.dropna(subset=group_cols).copy()
        if grouped.empty:
            continue
        grouped[success_col] = grouped[success_col].fillna(False).astype(bool)
        summary = (
            grouped.groupby(group_cols, as_index=False)
            .agg(
                attempts=(success_col, "size"),
                success_count=(success_col, "sum"),
            )
        )
        summary["actual_rate"] = summary["success_count"] / summary["attempts"]
        summary["overall_avg_rate"] = overall_avg_rate
        if reference_lookup is None:
            summary["reference_rate"] = np.nan
        else:
            summary["reference_rate"] = summary.apply(reference_lookup, axis=1)
        summary["overall_gap_p"] = summary["actual_rate"] - summary["overall_avg_rate"]
        summary["reference_gap_p"] = summary["actual_rate"] - summary["reference_rate"]
        summary["confidence"] = summary["attempts"].astype(int).map(confidence_label)
        base_gap = summary["reference_gap_p"].where(summary["reference_rate"].notna(), summary["overall_gap_p"])
        summary["adjusted_score"] = base_gap * np.log(summary["attempts"] + 1) * direction
        summary["condition_label"] = summary[group_cols].astype(str).agg(" / ".join, axis=1)
        summary["condition_group"] = group_name
        summary["target_type"] = target_type
        summary["metric_name"] = metric_name
        summary["direction"] = "higher_better" if higher_is_better else "lower_better"
        summary["interpretation"] = summary.apply(
            lambda row: _condition_interpretation(row, higher_is_better=higher_is_better),
            axis=1,
        )
        outputs.append(summary[columns])

    if not outputs:
        return pd.DataFrame(columns=columns)
    return pd.concat(outputs, ignore_index=True)


def top_condition_rankings(condition_scores: pd.DataFrame, top_n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(condition_scores.columns) if condition_scores is not None and not condition_scores.empty else [
        "condition_label",
        "condition_group",
        "target_type",
        "metric_name",
        "actual_rate",
        "overall_avg_rate",
        "reference_rate",
        "overall_gap_p",
        "reference_gap_p",
        "attempts",
        "success_count",
        "confidence",
        "adjusted_score",
        "direction",
        "interpretation",
    ]
    if condition_scores is None or condition_scores.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, empty

    filtered = condition_scores[condition_scores["attempts"] >= 10].copy()
    if filtered.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, empty

    good = filtered.sort_values(["adjusted_score", "attempts"], ascending=[False, False]).head(top_n)
    bad = filtered.sort_values(["adjusted_score", "attempts"], ascending=[True, False]).head(top_n)
    return good, bad


def build_date_rankings(summary_df: pd.DataFrame, rate_col: str, top_n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary_df is None or summary_df.empty or rate_col not in summary_df.columns:
        empty = pd.DataFrame()
        return empty, empty
    filtered = summary_df[summary_df["attempts"] >= 10].copy()
    if filtered.empty:
        empty = pd.DataFrame()
        return empty, empty
    gap_col = f"{rate_col.replace('_rate', '')}_overall_gap_p" if f"{rate_col.replace('_rate', '')}_overall_gap_p" in filtered.columns else None
    if gap_col is None:
        gap_col = "success_overall_gap_p" if "success_overall_gap_p" in filtered.columns else "effective_overall_gap_p"
    filtered["adjusted_score"] = filtered[gap_col] * np.log(filtered["attempts"] + 1)
    good = filtered.sort_values(["adjusted_score", "attempts"], ascending=[False, False]).head(top_n)
    bad = filtered.sort_values(["adjusted_score", "attempts"], ascending=[True, False]).head(top_n)
    return good, bad


def make_good_condition_text(row: pd.Series) -> str:
    reference_text = ""
    if pd.notna(row.get("reference_gap_p")):
        reference_text = f" · 기준 확률 대비 {format_gap_percent(row.get('reference_gap_p'))}"
    return (
        f"{row['condition_label']}에서 {row['metric_name']}이 전체 평균보다 "
        f"{format_gap_percent(row['overall_gap_p'])} 높게 관측되었습니다{reference_text}. "
        f"시도 수는 {int(row['attempts'])}회이며 신뢰도는 {row['confidence']}입니다."
    )


def make_bad_condition_text(row: pd.Series) -> str:
    reference_text = ""
    if pd.notna(row.get("reference_gap_p")):
        reference_text = f" · 기준 확률 대비 {format_gap_percent(row.get('reference_gap_p'))}"
    return (
        f"{row['condition_label']}에서 {row['metric_name']}이 전체 평균보다 "
        f"{format_gap_percent(row['overall_gap_p'])} 방향으로 관측되었습니다{reference_text}. "
        f"시도 수는 {int(row['attempts'])}회이며 신뢰도는 {row['confidence']}입니다."
    )


def make_day_of_month_insight_text(row: pd.Series) -> str:
    condition_label = _safe_condition_label(row, ("day_of_month_label",))
    metric_name = _safe_metric_label(row)
    reference_text = ""
    if pd.notna(row.get("reference_gap_p")):
        reference_text = f" · 기준 확률 대비 {format_gap_percent(row.get('reference_gap_p'))}"
    return (
        f"기준기간 내 같은 일자끼리 묶어 비교한 결과, {condition_label}의 {metric_name}이 "
        f"{format_percent(row['actual_rate'])}로 상대적으로 높게 관측되었습니다. "
        f"전체 평균 대비 {format_gap_percent(row['overall_gap_p'])}{reference_text} · 시도 수 {int(row['attempts'])}회 · "
        f"신뢰도는 {row['confidence']}입니다."
    )


def make_hour_insight_text(row: pd.Series) -> str:
    condition_label = _safe_condition_label(row, ("hour_label",))
    metric_name = _safe_metric_label(row)
    reference_text = ""
    if pd.notna(row.get("reference_gap_p")):
        reference_text = f" · 기준 확률 대비 {format_gap_percent(row.get('reference_gap_p'))}"
    return (
        f"기준기간 내 {condition_label} 기록을 모아 비교한 결과, {metric_name}이 "
        f"{format_percent(row['actual_rate'])}로 관측되었습니다. 전체 평균 대비 {format_gap_percent(row['overall_gap_p'])}"
        f"{reference_text} · 시도 수 {int(row['attempts'])}회입니다."
    )


def make_weekday_insight_text(row: pd.Series) -> str:
    condition_label = _safe_condition_label(row, ("weekday_kr",))
    metric_name = _safe_metric_label(row)
    reference_text = ""
    if pd.notna(row.get("reference_gap_p")):
        reference_text = f" · 기준 확률 대비 {format_gap_percent(row.get('reference_gap_p'))}"
    return (
        f"기준기간 내 {condition_label} 기록을 모아 비교한 결과, {metric_name}이 "
        f"{format_percent(row['actual_rate'])}로 관측되었습니다. "
        f"전체 평균 대비 {format_gap_percent(row['overall_gap_p'])}{reference_text} · 시도 수 {int(row['attempts'])}회입니다."
    )


def make_confidence_warning(row: pd.Series) -> str:
    return (
        f"해당 조건의 시도 수는 {int(row['attempts'])}회로 신뢰도는 {row['confidence']}입니다. "
        "이 분석은 과거 기록 기반의 참고용 통계이며, 향후 결과를 보장하지 않습니다."
    )


def _representative_character_row(merged: pd.DataFrame, preferred_character: str | None = None) -> pd.DataFrame:
    if merged.empty or "character_name" not in merged.columns:
        return pd.DataFrame()
    working = merged.dropna(subset=["character_name"]).copy()
    if working.empty:
        return pd.DataFrame()
    if preferred_character and preferred_character != "전체":
        preferred_rows = working[working["character_name"].astype(str) == preferred_character]
        if not preferred_rows.empty:
            return preferred_rows

    level_cols = [col for col in ["character_level", "level"] if col in working.columns]
    if level_cols:
        level_col = level_cols[0]
        working[level_col] = pd.to_numeric(working[level_col], errors="coerce")
        level_summary = (
            working.dropna(subset=[level_col])
            .groupby("character_name", as_index=False)
            .agg(max_level=(level_col, "max"), attempts=("character_name", "size"))
            .sort_values(["max_level", "attempts", "character_name"], ascending=[False, False, True])
        )
        if not level_summary.empty:
            best_name = str(level_summary.iloc[0]["character_name"])
            return working[working["character_name"].astype(str) == best_name]

    attempt_summary = (
        working.groupby("character_name", as_index=False)
        .agg(attempts=("character_name", "size"))
        .sort_values(["attempts", "character_name"], ascending=[False, True])
    )
    if attempt_summary.empty:
        return pd.DataFrame()
    best_name = str(attempt_summary.iloc[0]["character_name"])
    return working[working["character_name"].astype(str) == best_name]


def extract_profile_info(
    cube_df: pd.DataFrame,
    potential_df: pd.DataFrame,
    starforce_df: pd.DataFrame,
    job_name: str,
    preferred_character: str | None = None,
) -> dict[str, str]:
    frames = [frame for frame in [cube_df, potential_df, starforce_df] if frame is not None and not frame.empty]
    nickname = "-"
    world = "-"
    level = "API 응답에서 미확인"
    if frames:
        merged = pd.concat(frames, ignore_index=True, sort=False)
        rep_rows = _representative_character_row(merged, preferred_character)
        if not rep_rows.empty:
            if "character_name" in rep_rows.columns and rep_rows["character_name"].notna().any():
                nickname = str(rep_rows["character_name"].dropna().iloc[0])
            if "world_name" in rep_rows.columns and rep_rows["world_name"].notna().any():
                world = str(rep_rows["world_name"].dropna().iloc[0])
            for level_col in ["character_level", "level"]:
                if level_col in rep_rows.columns:
                    numeric_level = pd.to_numeric(rep_rows[level_col], errors="coerce").dropna()
                    if not numeric_level.empty:
                        level = str(int(numeric_level.max()))
                        break
    return {
        "nickname": nickname,
        "world": world,
        "job": job_name.strip() or "직접 입력 안 됨",
        "level": level,
        "image_text": nickname[:1] if nickname and nickname != "-" else "M",
    }


def cube_reference_lookup(ref_df: pd.DataFrame | None, metric_name: str) -> Callable[[pd.Series], float | None] | None:
    if ref_df is None or ref_df.empty:
        return None
    if not {"cube_type", "metric_name", "reference_rate"}.issubset(ref_df.columns):
        return None
    reference = ref_df.copy()
    reference["cube_type"] = reference["cube_type"].astype(str)
    reference["metric_name"] = reference["metric_name"].astype(str)
    reference_map = {
        (row["cube_type"], row["metric_name"]): row["reference_rate"]
        for _, row in reference.iterrows()
    }
    return lambda row: reference_map.get((str(row.get("cube_type")), metric_name))


def _infer_before_starforce_value(row: pd.Series) -> float | None:
    if pd.notna(row.get("before_starforce")):
        return float(row.get("before_starforce"))

    for key in ("transition_label", "starforce_range", "condition_label"):
        value = row.get(key)
        if value is None or pd.isna(value):
            continue
        text = str(value)
        digits = "".join(ch for ch in text if ch.isdigit() or ch == "~")
        if "~" in digits:
            left = digits.split("~", 1)[0]
            if left.isdigit():
                return float(left)
        numbers = [token for token in text.replace("→", " ").replace("성", " ").split() if token.isdigit()]
        if numbers:
            return float(numbers[0])
        plain_digits = "".join(ch for ch in text if ch.isdigit())
        if plain_digits:
            return float(plain_digits)
    return None


def starforce_reference_lookup(ref_df: pd.DataFrame | None) -> Callable[[pd.Series], float | None] | None:
    if ref_df is None or ref_df.empty:
        return None
    if not {"before_starforce", "success_rate_ref"}.issubset(ref_df.columns):
        return None
    reference = ref_df.copy()
    reference["before_starforce"] = pd.to_numeric(reference["before_starforce"], errors="coerce")
    reference_map = {
        float(row["before_starforce"]): row["success_rate_ref"]
        for _, row in reference.dropna(subset=["before_starforce"]).iterrows()
    }
    return lambda row: reference_map.get(_infer_before_starforce_value(row)) if _infer_before_starforce_value(row) is not None else None


def starforce_destroy_reference_lookup(ref_df: pd.DataFrame | None) -> Callable[[pd.Series], float | None] | None:
    if ref_df is None or ref_df.empty:
        return None
    if not {"before_starforce", "destroy_rate_ref"}.issubset(ref_df.columns):
        return None
    reference = ref_df.copy()
    reference["before_starforce"] = pd.to_numeric(reference["before_starforce"], errors="coerce")
    reference_map = {
        float(row["before_starforce"]): row["destroy_rate_ref"]
        for _, row in reference.dropna(subset=["before_starforce"]).iterrows()
    }
    return lambda row: reference_map.get(_infer_before_starforce_value(row)) if _infer_before_starforce_value(row) is not None else None


def _cube_rate_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    columns = [
        group_col,
        "attempts",
        "major_option_count",
        "major_option_rate",
        "effective_option_count",
        "effective_option_rate",
        "grade_up_count",
        "grade_up_rate",
        "major_overall_gap_p",
        "effective_overall_gap_p",
        "grade_up_overall_gap_p",
        "confidence",
    ]
    if df is None or df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=columns)
    working = df.dropna(subset=[group_col]).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)

    overall_major = _bool_rate(working, "has_major_option")
    overall_effective = _bool_rate(working, "has_effective_option")
    overall_grade = _bool_rate(working, "is_grade_up")
    summary = (
        working.groupby(group_col, as_index=False)
        .agg(
            attempts=(group_col, "size"),
            major_option_count=("has_major_option", "sum"),
            effective_option_count=("has_effective_option", "sum"),
            grade_up_count=("is_grade_up", "sum"),
        )
    )
    summary["major_option_rate"] = summary["major_option_count"] / summary["attempts"]
    summary["effective_option_rate"] = summary["effective_option_count"] / summary["attempts"]
    summary["grade_up_rate"] = summary["grade_up_count"] / summary["attempts"]
    summary["major_overall_gap_p"] = summary["major_option_rate"] - overall_major
    summary["effective_overall_gap_p"] = summary["effective_option_rate"] - overall_effective
    summary["grade_up_overall_gap_p"] = summary["grade_up_rate"] - overall_grade
    return add_confidence(summary, "attempts")[columns]


def _star_rate_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    columns = [
        group_col,
        "attempts",
        "success_count",
        "success_rate",
        "destroy_count",
        "destroy_rate",
        "success_overall_gap_p",
        "destroy_overall_gap_p",
        "confidence",
    ]
    if df is None or df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=columns)
    working = df.dropna(subset=[group_col]).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)

    overall_success = _bool_rate(working, "is_success")
    overall_destroy = _bool_rate(working, "is_destroyed")
    summary = (
        working.groupby(group_col, as_index=False)
        .agg(
            attempts=(group_col, "size"),
            success_count=("is_success", "sum"),
            destroy_count=("is_destroyed", "sum"),
        )
    )
    summary["success_rate"] = summary["success_count"] / summary["attempts"]
    summary["destroy_rate"] = summary["destroy_count"] / summary["attempts"]
    summary["success_overall_gap_p"] = summary["success_rate"] - overall_success
    summary["destroy_overall_gap_p"] = summary["destroy_rate"] - overall_destroy
    return add_confidence(summary, "attempts")[columns]


def _bool_rate(df: pd.DataFrame, column: str) -> float | None:
    if df is None or df.empty or column not in df.columns:
        return None
    return float(df[column].fillna(False).astype(bool).mean())


def _condition_interpretation(row: pd.Series, higher_is_better: bool) -> str:
    confidence = row.get("confidence", "참고 불가")
    if confidence == "참고 불가":
        return "표본 수가 적어 참고용입니다."
    if higher_is_better:
        return "내 기록상 상대적으로 높게 관측된 조건입니다." if row["adjusted_score"] >= 0 else "내 기록상 상대적으로 낮게 관측된 조건입니다."
    return "내 기록상 상대적으로 안정적으로 관측된 조건입니다." if row["adjusted_score"] >= 0 else "내 기록상 파괴율이 아쉽게 관측된 조건입니다."
