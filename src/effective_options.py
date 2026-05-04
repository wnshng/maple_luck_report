from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from .metrics import summarize_by_hour


STAT_OPTIONS = [
    "STR",
    "DEX",
    "INT",
    "LUK",
    "최대 HP",
    "공격력",
    "마력",
    "올스탯",
    "보스 데미지",
    "방어율 무시",
    "크리티컬 데미지",
    "데미지",
    "재사용 대기시간 미적용",
    "메소 획득량",
    "아이템 드롭률",
]

JOB_EFFECTIVE_OPTIONS = {
    "궁수": ["DEX", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "메르세데스": ["DEX", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "보우마스터": ["DEX", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "신궁": ["DEX", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "패스파인더": ["DEX", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "카인": ["DEX", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "와일드헌터": ["DEX", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "전사": ["STR", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "히어로": ["STR", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "팔라딘": ["STR", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "다크나이트": ["STR", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "아델": ["STR", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "아란": ["STR", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "소울마스터": ["STR", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "데몬슬레이어": ["STR", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "카이저": ["STR", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "제로": ["STR", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "마법사": ["INT", "마력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "아크메이지": ["INT", "마력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "비숍": ["INT", "마력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "에반": ["INT", "마력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "루미너스": ["INT", "마력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "일리움": ["INT", "마력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "라라": ["INT", "마력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "키네시스": ["INT", "마력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "배틀메이지": ["INT", "마력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "도적": ["LUK", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "나이트로드": ["LUK", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "섀도어": ["LUK", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "듀얼블레이드": ["LUK", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "팬텀": ["LUK", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "카데나": ["LUK", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "호영": ["LUK", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "칼리": ["LUK", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "해적 힘": ["STR", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "바이퍼": ["STR", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "캐논슈터": ["STR", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "스트라이커": ["STR", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "아크": ["STR", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "해적 덱스": ["DEX", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "캡틴": ["DEX", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "메카닉": ["DEX", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "엔젤릭버스터": ["DEX", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "제논": ["STR", "DEX", "LUK", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
    "데몬어벤져": ["최대 HP", "공격력", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
}

OPTION_COLUMNS = (
    "before_potential_options",
    "after_potential_options",
    "before_potential_option",
    "after_potential_option",
    "before_option",
    "after_option",
    "potential_options",
    "additional_potential_options",
    "after_additional_potential_option",
    "potential_option_grade",
)

ALIASES = {
    "STR": ("STR", "힘"),
    "DEX": ("DEX", "덱스", "민첩"),
    "INT": ("INT", "인트", "지력"),
    "LUK": ("LUK", "럭", "운"),
    "최대 HP": ("최대HP", "HP", "MAXHP"),
    "공격력": ("공격력", "공", "ATT", "ATTACK", "공격"),
    "마력": ("마력", "MATT", "MAGICATTACK", "MAGIC"),
    "올스탯": ("올스탯", "올스텟", "ALLSTAT", "ALL STAT"),
    "보스 데미지": ("보스몬스터공격시데미지", "보스데미지", "보공", "BOSS", "BOSSDAMAGE"),
    "방어율 무시": ("몬스터방어율무시", "방어율무시", "방무", "IGNOREDEFENSE", "IED"),
    "크리티컬 데미지": ("크리티컬데미지", "크뎀", "CRITICALDAMAGE", "CRITDAMAGE"),
    "데미지": ("데미지", "DAMAGE"),
    "재사용 대기시간 미적용": ("재사용대기시간미적용", "재사용", "쿨타임", "COOLDOWN"),
    "메소 획득량": ("메소획득량", "메획", "MESO"),
    "아이템 드롭률": ("아이템드롭률", "드롭", "아획", "ITEMDROP", "DROP"),
}


def get_effective_keywords(job_name: str | None, selected_stats: list[str]) -> list[str]:
    keywords: list[str] = []
    normalized_job = (job_name or "").strip().lower()
    for job_key, values in JOB_EFFECTIVE_OPTIONS.items():
        if job_key.lower() in normalized_job:
            keywords.extend(values)
            break
    keywords.extend(selected_stats or [])
    return sorted(set(keywords))


def extract_potential_options(row: pd.Series) -> list[str]:
    options: list[str] = []
    for col in OPTION_COLUMNS:
        if col not in row.index:
            continue
        options.extend(_extract_from_value(row[col]))
    return [option for option in options if option]


def is_effective_option(options: list[str], effective_keywords: list[str]) -> bool:
    return count_effective_lines(options, effective_keywords) > 0


def count_effective_lines(options: list[str], effective_keywords: list[str]) -> int:
    canonical_keywords = [_canonical_keyword(keyword) for keyword in effective_keywords]
    count = 0
    for option in options:
        canonical_option = _canonical_option(option)
        if any(_matches(canonical_option, keyword) for keyword in canonical_keywords):
            count += 1
    return count


def add_effective_option_features(
    df: pd.DataFrame,
    job_name: str | None,
    selected_stats: list[str],
) -> pd.DataFrame:
    output = df.copy()
    if output.empty:
        output["options_after"] = []
        output["effective_keywords"] = []
        output["effective_line_count"] = []
        output["is_effective_option"] = []
        return output

    effective_keywords = get_effective_keywords(job_name, selected_stats)
    output["options_after"] = output.apply(extract_potential_options, axis=1)
    output["effective_keywords"] = [effective_keywords for _ in range(len(output))]
    output["effective_line_count"] = output["options_after"].map(
        lambda options: count_effective_lines(options, effective_keywords)
    )
    output["is_effective_option"] = output["effective_line_count"] > 0
    return output


def summarize_effective_options(df: pd.DataFrame) -> dict:
    total = int(len(df))
    effective_count = int(df.get("is_effective_option", pd.Series(False, index=df.index)).fillna(False).sum()) if total else 0
    avg_lines = float(df["effective_line_count"].fillna(0).mean()) if total and "effective_line_count" in df.columns else 0.0
    return {
        "total_cube_uses": total,
        "effective_count": effective_count,
        "effective_rate": effective_count / total if total else None,
        "avg_effective_lines": avg_lines,
        "best_weekday_for_effective": _best_group_label(summarize_effective_by_weekday(df), "weekday_kr", "effective_rate"),
        "best_hour_for_effective": _best_group_label(summarize_effective_by_hour(df), "hour_label", "success_rate"),
        "best_hour_band_for_effective": _best_group_label(summarize_effective_by_hour_band(df), "hour_band", "effective_rate"),
    }


def summarize_effective_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    return _group_effective(df, "weekday_kr")


def summarize_effective_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    return summarize_by_hour(df, success_col="is_effective_option", min_attempts=3)


def summarize_effective_by_hour_band(df: pd.DataFrame) -> pd.DataFrame:
    return _group_effective(df, "hour_band")


def summarize_effective_by_cube_type(df: pd.DataFrame) -> pd.DataFrame:
    return _group_effective(df, "cube_type")


def summarize_effective_by_item(df: pd.DataFrame) -> pd.DataFrame:
    return _group_effective(df, "item_name").sort_values("attempt_count", ascending=False)


def _group_effective(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    columns = [group_col, "attempt_count", "success_count", "effective_rate", "luck_score"]
    if df.empty or group_col not in df.columns or "is_effective_option" not in df.columns:
        return pd.DataFrame(columns=columns)
    working = df.dropna(subset=[group_col]).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["is_effective_option"] = working["is_effective_option"].fillna(False).astype(bool)
    overall = working["is_effective_option"].mean()
    summary = (
        working.groupby(group_col, as_index=False)
        .agg(attempt_count=("is_effective_option", "size"), success_count=("is_effective_option", "sum"))
    )
    summary["effective_rate"] = summary["success_count"] / summary["attempt_count"]
    summary["luck_score"] = (50 + (summary["effective_rate"] - overall) * 100).clip(0, 100)
    return summary.sort_values(["effective_rate", "attempt_count"], ascending=[False, False])


def _extract_from_value(value: Any) -> list[str]:
    if value is None or (not isinstance(value, (list, dict, tuple, set)) and pd.isna(value)):
        return []
    if isinstance(value, str):
        parsed = _parse_json_like(value)
        if parsed is not value:
            return _extract_from_value(parsed)
        return [part.strip() for part in re.split(r"[\n,;/|]+", value) if part.strip()]
    if isinstance(value, dict):
        option_texts = []
        for key in ("value", "option", "name", "grade"):
            if value.get(key):
                option_texts.append(str(value[key]))
        return [" ".join(option_texts)] if option_texts else [json.dumps(value, ensure_ascii=False)]
    if isinstance(value, (list, tuple, set)):
        options: list[str] = []
        for item in value:
            options.extend(_extract_from_value(item))
        return options
    return [str(value)]


def _parse_json_like(value: str) -> Any:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _canonical_option(value: str) -> str:
    normalized = value.upper()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.replace(":", "").replace("+", "")
    return normalized


def _canonical_keyword(value: str) -> str:
    return str(value).strip()


def _matches(canonical_option: str, keyword: str) -> bool:
    aliases = ALIASES.get(keyword, (keyword,))
    return any(_canonical_option(alias) in canonical_option for alias in aliases)


def _best_group_label(summary: pd.DataFrame, group_col: str, rate_col: str) -> str | None:
    if summary.empty or group_col not in summary.columns or rate_col not in summary.columns:
        return "표본 부족"
    attempt_col = "attempts" if "attempts" in summary.columns else "attempt_count"
    candidates = summary[(summary[attempt_col] >= 3) & summary[rate_col].notna()].copy()
    if candidates.empty:
        return "표본 부족"
    candidates = candidates.sort_values([rate_col, attempt_col], ascending=[False, False])
    return str(candidates.iloc[0][group_col])

