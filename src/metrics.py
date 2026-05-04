from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


MIN_SAMPLE_SIZE = 5
CHANNEL_NOTICE = "현재 API 응답에서 채널 정보가 제공되지 않아 채널별 분석은 제외되었습니다."


@dataclass
class LuckReport:
    total_attempts: int = 0
    success_count: int = 0
    success_rate: float | None = None
    destroyed_count: int = 0
    drop_count: int = 0
    starcatch_success_count: int = 0
    chance_time_count: int = 0
    destroy_defence_count: int = 0
    best_weekday: str = "표본 부족"
    best_hour: str = "표본 부족"
    best_hour_band: str = "표본 부족"
    max_consecutive_failure: int = 0
    luck_score: float | None = None
    gain_loss_index: float | None = None
    weekday_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    hour_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    hour_band_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    item_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    type_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    stage_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    channel_summary: pd.DataFrame | None = None
    best_channel: str | None = None
    note: str = "운 점수는 전체 유저 평균이 공개되지 않아 개인 히스토리 내부 평균 대비 상대 점수로 계산합니다."


def compute_cube_luck_metrics(df: pd.DataFrame) -> LuckReport:
    if df.empty:
        return LuckReport(note="큐브 데이터가 없습니다.")

    working = df.copy()
    working["is_grade_up"] = _bool_series(working, "is_grade_up")
    total_attempts = int(len(working))
    grade_up_count = int(working["is_grade_up"].sum())
    grade_up_rate = _safe_rate(grade_up_count, total_attempts)

    weekday_summary = _rate_summary(working, "weekday_kr", "is_grade_up", "grade_up_rate")
    hour_summary = _rate_summary(working, "hour", "is_grade_up", "grade_up_rate")
    hour_band_summary = _rate_summary(working, "hour_band", "is_grade_up", "grade_up_rate")
    item_summary = _rate_summary(working, "item_name", "is_grade_up", "grade_up_rate")
    cube_type_summary = _rate_summary(working, "cube_type", "is_grade_up", "grade_up_rate")
    channel_summary = compute_channel_summary(working, "is_grade_up", "grade_up_rate")

    return LuckReport(
        total_attempts=total_attempts,
        success_count=grade_up_count,
        success_rate=grade_up_rate,
        best_weekday=_best_group(weekday_summary, "weekday_kr", "grade_up_rate"),
        best_hour_band=_best_group(hour_band_summary, "hour_band", "grade_up_rate"),
        max_consecutive_failure=max_consecutive_failure(working, "is_grade_up"),
        luck_score=_overall_luck_score(grade_up_rate),
        gain_loss_index=_gain_loss_index(grade_up_count, total_attempts - grade_up_count),
        weekday_summary=weekday_summary,
        hour_summary=hour_summary,
        hour_band_summary=hour_band_summary,
        item_summary=item_summary,
        type_summary=cube_type_summary,
        channel_summary=channel_summary,
        best_channel=_best_group(channel_summary, "channel", "grade_up_rate") if channel_summary is not None else None,
    )


def compute_starforce_luck_metrics(df: pd.DataFrame) -> LuckReport:
    if df.empty:
        return LuckReport(note="스타포스 데이터가 없습니다.")

    summary = summarize_starforce(df)
    weekday_summary = summarize_starforce_by_weekday(df)
    hour_summary = summarize_starforce_by_hour(df)
    hour_band_summary = summarize_starforce_by_hour_band(df)
    stage_summary = summarize_starforce_by_star_count(df).rename(columns={"attempts": "attempt_count"})
    item_summary = _rate_summary(df.copy(), "item_name", "is_success", "success_rate")

    penalty_count = (summary["total_attempts"] - summary["success_count"]) + summary["drop_count"] + summary["destroy_count"] * 2
    return LuckReport(
        total_attempts=summary["total_attempts"],
        success_count=summary["success_count"],
        success_rate=summary["success_rate"],
        destroyed_count=summary["destroy_count"],
        drop_count=summary["drop_count"],
        starcatch_success_count=summary["starcatch_success_count"],
        chance_time_count=summary["chance_time_count"],
        destroy_defence_count=summary["destroy_defence_count"],
        best_weekday=summary["best_weekday"],
        best_hour=summary["best_hour"],
        best_hour_band=summary["best_hour_band"],
        max_consecutive_failure=summary["max_fail_streak"] or 0,
        luck_score=summary["luck_score"],
        gain_loss_index=_gain_loss_index(summary["success_count"], penalty_count),
        weekday_summary=weekday_summary,
        hour_summary=hour_summary,
        hour_band_summary=hour_band_summary,
        item_summary=item_summary,
        stage_summary=stage_summary,
        channel_summary=None,
        best_channel=None,
    )


def summarize_starforce(df: pd.DataFrame) -> dict[str, Any]:
    warnings: list[str] = []
    if df.empty:
        return {
            "total_attempts": 0,
            "success_count": 0,
            "success_rate": None,
            "destroy_count": 0,
            "drop_count": 0,
            "starcatch_success_count": 0,
            "chance_time_count": 0,
            "destroy_defence_count": 0,
            "best_weekday": "표본 부족",
            "best_hour": "표본 부족",
            "best_hour_band": "표본 부족",
            "max_fail_streak": None,
            "luck_score": None,
            "warnings": warnings,
        }

    working = df.copy()
    working["is_success"] = _bool_series(working, "is_success")
    working["is_destroyed"] = _bool_series(working, "is_destroyed")
    working["is_drop"] = _bool_series(working, "is_drop")
    working["starcatch_positive"] = _flag_like_series(working, "starcatch_result", success_only=True)
    working["chance_time_positive"] = _flag_like_series(working, "chance_time")
    working["destroy_defence_positive"] = _flag_like_series(working, "destroy_defence")

    total_attempts = int(len(working))
    success_count = int(working["is_success"].sum())
    destroy_count = int(working["is_destroyed"].sum())
    drop_count = int(working["is_drop"].sum())
    starcatch_success_count = int(working["starcatch_positive"].sum())
    chance_time_count = int(working["chance_time_positive"].sum())
    destroy_defence_count = int(working["destroy_defence_positive"].sum())
    success_rate = _safe_rate(success_count, total_attempts)
    weekday_summary = summarize_starforce_by_weekday(working)
    hour_summary = summarize_starforce_by_hour(working)
    hour_band_summary = summarize_starforce_by_hour_band(working)

    return {
        "total_attempts": total_attempts,
        "success_count": success_count,
        "success_rate": success_rate,
        "destroy_count": destroy_count,
        "drop_count": drop_count,
        "starcatch_success_count": starcatch_success_count,
        "chance_time_count": chance_time_count,
        "destroy_defence_count": destroy_defence_count,
        "best_weekday": _best_group_label(weekday_summary, "weekday_kr", min_attempts=5),
        "best_hour": _best_group_label(hour_summary, "hour_label", min_attempts=3),
        "best_hour_band": _best_group_label(hour_band_summary, "hour_band", min_attempts=5),
        "max_fail_streak": max_consecutive_failures(working),
        "luck_score": _starforce_luck_score(success_rate, weekday_summary, hour_summary, hour_band_summary),
        "warnings": warnings,
    }


def summarize_starforce_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    return _rate_summary(df.copy(), "weekday_kr", "is_success", "success_rate")


def summarize_starforce_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    return summarize_by_hour(df, success_col="is_success", min_attempts=3)


def summarize_starforce_by_hour_band(df: pd.DataFrame) -> pd.DataFrame:
    return _rate_summary(df.copy(), "hour_band", "is_success", "success_rate")


def summarize_starforce_by_star_count(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "before_starforce",
        "attempts",
        "success_count",
        "success_rate",
        "destroy_count",
        "drop_count",
        "luck_score",
    ]
    if df.empty or "before_starforce" not in df.columns:
        return pd.DataFrame(columns=columns)

    working = df.dropna(subset=["before_starforce"]).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)

    working["before_starforce"] = pd.to_numeric(working["before_starforce"], errors="coerce")
    working["is_success"] = _bool_series(working, "is_success")
    working["is_destroyed"] = _bool_series(working, "is_destroyed")
    working["is_drop"] = _bool_series(working, "is_drop")
    overall_rate = _safe_rate(int(working["is_success"].sum()), int(len(working)))

    summary = (
        working.groupby("before_starforce", as_index=False)
        .agg(
            attempts=("is_success", "size"),
            success_count=("is_success", "sum"),
            destroy_count=("is_destroyed", "sum"),
            drop_count=("is_drop", "sum"),
        )
        .sort_values("before_starforce")
    )
    summary["success_rate"] = summary["success_count"] / summary["attempts"]
    summary["luck_score"] = summary["success_rate"].map(lambda rate: _relative_luck_score(rate, overall_rate))
    return summary[columns]


def get_best_weekday(
    df: pd.DataFrame,
    success_col: str = "is_success",
    min_attempts: int = MIN_SAMPLE_SIZE,
) -> str | None:
    return _best_value_from_df(df, "weekday_kr", success_col, min_attempts)


def get_best_hour_band(
    df: pd.DataFrame,
    success_col: str = "is_success",
    min_attempts: int = MIN_SAMPLE_SIZE,
) -> str | None:
    return _best_value_from_df(df, "hour_band", success_col, min_attempts)


def summarize_by_hour(
    df: pd.DataFrame,
    success_col: str = "is_success",
    min_attempts: int = 3,
) -> pd.DataFrame:
    columns = ["hour", "hour_label", "attempts", "success_count", "success_rate", "lift_vs_overall", "sample_note"]
    if df.empty or "hour" not in df.columns:
        return pd.DataFrame(columns=columns)

    working = df.dropna(subset=["hour"]).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)

    working["hour"] = working["hour"].astype(int)
    working["hour_label"] = working["hour"].map(lambda hour: f"{hour}시")
    working[success_col] = _bool_series(working, success_col)
    overall_rate = _safe_rate(int(working[success_col].sum()), int(len(working)))

    summary = (
        working.groupby(["hour", "hour_label"], as_index=False)
        .agg(attempts=(success_col, "size"), success_count=(success_col, "sum"))
        .sort_values("hour")
    )
    summary["success_rate"] = summary["success_count"] / summary["attempts"]
    summary["lift_vs_overall"] = summary["success_rate"] - (overall_rate or 0)
    summary["sample_note"] = np.where(summary["attempts"] < min_attempts, "표본 적음", "")

    all_hours = pd.DataFrame({"hour": list(range(24)), "hour_label": [f"{hour}시" for hour in range(24)]})
    output = all_hours.merge(summary, on=["hour", "hour_label"], how="left")
    output["attempts"] = output["attempts"].fillna(0).astype(int)
    output["success_count"] = output["success_count"].fillna(0).astype(int)
    output["sample_note"] = output["sample_note"].fillna("표본 없음")
    return output[columns]


def get_best_hour(
    df: pd.DataFrame,
    success_col: str = "is_success",
    min_attempts: int = 3,
) -> dict[str, Any]:
    summary = summarize_by_hour(df, success_col=success_col, min_attempts=min_attempts)
    candidates = summary[(summary["attempts"] >= min_attempts) & summary["success_rate"].notna()].copy()
    if candidates.empty:
        return {"hour": None, "hour_label": "표본 부족", "success_rate": None, "attempts": 0}

    candidates = candidates.sort_values(["success_rate", "attempts"], ascending=[False, False])
    row = candidates.iloc[0]
    return {
        "hour": int(row["hour"]),
        "hour_label": str(row["hour_label"]),
        "success_rate": float(row["success_rate"]),
        "attempts": int(row["attempts"]),
    }


def get_best_channel(
    df: pd.DataFrame,
    success_col: str = "is_success",
    min_attempts: int = MIN_SAMPLE_SIZE,
) -> str | None:
    channel_col = _channel_column(df)
    if channel_col is None or df.empty or not df[channel_col].notna().any():
        return None
    return _best_value_from_df(df.rename(columns={channel_col: "channel"}), "channel", success_col, min_attempts)


def max_consecutive_failures(df: pd.DataFrame, success_col: str = "is_success") -> int | None:
    if df.empty:
        return None
    return max_consecutive_failure(df, success_col)


def max_consecutive_failure(df: pd.DataFrame, success_col: str) -> int:
    if df.empty or success_col not in df.columns:
        return 0

    sort_col = "event_datetime" if "event_datetime" in df.columns else None
    ordered = df.sort_values(sort_col) if sort_col else df.copy()

    max_streak = 0
    current_streak = 0
    for is_success in ordered[success_col].fillna(False).astype(bool):
        if is_success:
            current_streak = 0
            continue
        current_streak += 1
        max_streak = max(max_streak, current_streak)
    return int(max_streak)


def compute_channel_summary(
    df: pd.DataFrame,
    success_col: str,
    rate_col: str,
    min_non_null: int = MIN_SAMPLE_SIZE,
) -> pd.DataFrame | None:
    channel_col = _channel_column(df)
    if channel_col is None:
        return None

    non_null_ratio = df[channel_col].notna().mean() if len(df) else 0
    if df[channel_col].notna().sum() < min_non_null or non_null_ratio < 0.5:
        return None

    summary = _rate_summary(df.rename(columns={channel_col: "channel"}), "channel", success_col, rate_col)
    if summary.empty:
        return None
    return summary


def _best_value_from_df(df: pd.DataFrame, group_col: str, success_col: str, min_attempts: int) -> str:
    if df.empty or group_col not in df.columns:
        return "표본 부족"
    working = df.copy()
    working[success_col] = _bool_series(working, success_col)
    summary = _rate_summary(working, group_col, success_col, "success_rate")
    candidates = summary[summary["attempt_count"] >= min_attempts].copy()
    if candidates.empty:
        return "표본 부족"
    candidates = candidates.sort_values(["success_rate", "attempt_count"], ascending=[False, False])
    value = candidates.iloc[0][group_col]
    if pd.isna(value):
        return "표본 부족"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _channel_column(df: pd.DataFrame) -> str | None:
    for col in ("channel", "channel_name", "channel_no", "ch"):
        if col in df.columns:
            return col
    return None


def _flag_like_series(df: pd.DataFrame, column: str, success_only: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    normalized = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    if success_only:
        return normalized.isin({"Y", "YES", "TRUE", "1", "성공", "적용"})
    return normalized.isin({"Y", "YES", "TRUE", "1", "적용", "성공"})


def _best_group_label(summary: pd.DataFrame, group_col: str, min_attempts: int) -> str:
    if summary.empty:
        return "표본 부족"

    attempt_col = "attempts" if "attempts" in summary.columns else "attempt_count"
    candidates = summary[(summary[attempt_col] >= min_attempts) & summary["success_rate"].notna()].copy()
    if candidates.empty:
        return "표본 부족"
    candidates = candidates.sort_values(["success_rate", attempt_col], ascending=[False, False])
    return str(candidates.iloc[0][group_col])


def _starforce_luck_score(
    overall_rate: float | None,
    weekday_summary: pd.DataFrame,
    hour_summary: pd.DataFrame,
    hour_band_summary: pd.DataFrame,
) -> float | None:
    if overall_rate is None:
        return None

    lifts: list[float] = []
    if not hour_summary.empty:
        lifts.extend(
            hour_summary.loc[hour_summary["attempts"] >= 3, "lift_vs_overall"].dropna().astype(float).tolist()
        )
    if not weekday_summary.empty:
        lifts.extend(
            (
                weekday_summary.loc[weekday_summary["attempt_count"] >= 5, "success_rate"] - overall_rate
            ).dropna().astype(float).tolist()
        )
    if not hour_band_summary.empty:
        lifts.extend(
            (
                hour_band_summary.loc[hour_band_summary["attempt_count"] >= 5, "success_rate"] - overall_rate
            ).dropna().astype(float).tolist()
        )

    best_lift = max(lifts) if lifts else 0.0
    return float(np.clip(50 + best_lift * 100, 0, 100))


def _rate_summary(df: pd.DataFrame, group_col: str, success_col: str, rate_col: str) -> pd.DataFrame:
    columns = [group_col, "attempt_count", "success_count", rate_col, "luck_score"]
    if df.empty or group_col not in df.columns or success_col not in df.columns:
        return pd.DataFrame(columns=columns)

    filtered = df.dropna(subset=[group_col]).copy()
    if filtered.empty:
        return pd.DataFrame(columns=columns)

    filtered[success_col] = filtered[success_col].fillna(False).astype(bool)
    summary = (
        filtered.groupby(group_col, dropna=True)
        .agg(attempt_count=(success_col, "size"), success_count=(success_col, "sum"))
        .reset_index()
    )
    summary[rate_col] = np.where(
        summary["attempt_count"] > 0,
        summary["success_count"] / summary["attempt_count"],
        np.nan,
    )

    baseline = _safe_rate(int(filtered[success_col].sum()), int(len(filtered)))
    summary["luck_score"] = summary[rate_col].map(lambda rate: _relative_luck_score(rate, baseline))
    return summary.sort_values([rate_col, "attempt_count"], ascending=[False, False])


def _best_group(summary: pd.DataFrame | None, group_col: str, rate_col: str) -> str:
    if summary is None or summary.empty:
        return "표본 부족"

    candidates = summary[summary["attempt_count"] >= MIN_SAMPLE_SIZE].copy()
    if candidates.empty:
        return "표본 부족"

    candidates = candidates.sort_values([rate_col, "attempt_count"], ascending=[False, False])
    value: Any = candidates.iloc[0][group_col]
    if pd.isna(value):
        return "표본 부족"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _safe_rate(success_count: int, attempt_count: int) -> float | None:
    if attempt_count <= 0:
        return None
    return success_count / attempt_count


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].fillna(False).astype(bool)


def _overall_luck_score(rate: float | None) -> float | None:
    if rate is None:
        return None
    return 50.0


def _relative_luck_score(rate: float | None, baseline: float | None) -> float | None:
    if rate is None or baseline is None:
        return None
    return float(np.clip(50 + (rate - baseline) * 100, 0, 100))


def _gain_loss_index(positive_count: int, negative_count: int) -> float | None:
    total = positive_count + negative_count
    if total <= 0:
        return None
    return round(((positive_count - negative_count) / total) * 100, 2)
