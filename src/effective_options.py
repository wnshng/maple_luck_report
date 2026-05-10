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

TARGET_PRESET_OPTIONS = [
    "주스탯 2줄 이상",
    "공격력/마력 2줄 이상",
    "보공 + 공격력/마력 조합",
    "보공/방무/크뎀 중 2줄 이상",
    "크뎀 포함 2줄 이상",
    "이탈 옵션 포함",
    "사용자 직접 선택",
]

COMMON_PHYSICAL_OPTIONS = ["공격력", "보스 데미지", "방어율 무시", "크리티컬 데미지"]
COMMON_MAGIC_OPTIONS = ["마력", "보스 데미지", "방어율 무시", "크리티컬 데미지"]

CLASS_MAIN_STAT_MAP = {
    "히어로": ("STR", ["STR", *COMMON_PHYSICAL_OPTIONS]),
    "팔라딘": ("STR", ["STR", *COMMON_PHYSICAL_OPTIONS]),
    "다크나이트": ("STR", ["STR", *COMMON_PHYSICAL_OPTIONS]),
    "소울마스터": ("STR", ["STR", *COMMON_PHYSICAL_OPTIONS]),
    "미하일": ("STR", ["STR", *COMMON_PHYSICAL_OPTIONS]),
    "아란": ("STR", ["STR", *COMMON_PHYSICAL_OPTIONS]),
    "카이저": ("STR", ["STR", *COMMON_PHYSICAL_OPTIONS]),
    "아델": ("STR", ["STR", *COMMON_PHYSICAL_OPTIONS]),
    "블래스터": ("STR", ["STR", *COMMON_PHYSICAL_OPTIONS]),
    "데몬슬레이어": ("STR", ["STR", *COMMON_PHYSICAL_OPTIONS]),
    "은월": ("STR", ["STR", *COMMON_PHYSICAL_OPTIONS]),
    "바이퍼": ("STR", ["STR", *COMMON_PHYSICAL_OPTIONS]),
    "캐논슈터": ("STR", ["STR", *COMMON_PHYSICAL_OPTIONS]),
    "아크": ("STR", ["STR", *COMMON_PHYSICAL_OPTIONS]),
    "보우마스터": ("DEX", ["DEX", *COMMON_PHYSICAL_OPTIONS]),
    "신궁": ("DEX", ["DEX", *COMMON_PHYSICAL_OPTIONS]),
    "패스파인더": ("DEX", ["DEX", *COMMON_PHYSICAL_OPTIONS]),
    "윈드브레이커": ("DEX", ["DEX", *COMMON_PHYSICAL_OPTIONS]),
    "와일드헌터": ("DEX", ["DEX", *COMMON_PHYSICAL_OPTIONS]),
    "메르세데스": ("DEX", ["DEX", *COMMON_PHYSICAL_OPTIONS]),
    "카인": ("DEX", ["DEX", *COMMON_PHYSICAL_OPTIONS]),
    "엔젤릭버스터": ("DEX", ["DEX", *COMMON_PHYSICAL_OPTIONS]),
    "캡틴": ("DEX", ["DEX", *COMMON_PHYSICAL_OPTIONS]),
    "메카닉": ("DEX", ["DEX", *COMMON_PHYSICAL_OPTIONS]),
    "아크메이지불독": ("INT", ["INT", *COMMON_MAGIC_OPTIONS]),
    "아크메이지썬콜": ("INT", ["INT", *COMMON_MAGIC_OPTIONS]),
    "비숍": ("INT", ["INT", *COMMON_MAGIC_OPTIONS]),
    "플레임위자드": ("INT", ["INT", *COMMON_MAGIC_OPTIONS]),
    "배틀메이지": ("INT", ["INT", *COMMON_MAGIC_OPTIONS]),
    "에반": ("INT", ["INT", *COMMON_MAGIC_OPTIONS]),
    "루미너스": ("INT", ["INT", *COMMON_MAGIC_OPTIONS]),
    "일리움": ("INT", ["INT", *COMMON_MAGIC_OPTIONS]),
    "라라": ("INT", ["INT", *COMMON_MAGIC_OPTIONS]),
    "키네시스": ("INT", ["INT", *COMMON_MAGIC_OPTIONS]),
    "나이트로드": ("LUK", ["LUK", *COMMON_PHYSICAL_OPTIONS]),
    "섀도어": ("LUK", ["LUK", *COMMON_PHYSICAL_OPTIONS]),
    "듀얼블레이드": ("LUK", ["LUK", *COMMON_PHYSICAL_OPTIONS]),
    "나이트워커": ("LUK", ["LUK", *COMMON_PHYSICAL_OPTIONS]),
    "팬텀": ("LUK", ["LUK", *COMMON_PHYSICAL_OPTIONS]),
    "카데나": ("LUK", ["LUK", *COMMON_PHYSICAL_OPTIONS]),
    "호영": ("LUK", ["LUK", *COMMON_PHYSICAL_OPTIONS]),
    "칼리": ("LUK", ["LUK", *COMMON_PHYSICAL_OPTIONS]),
    "제논": ("복합", ["STR", "DEX", "LUK", "올스탯", "공격력", "보스 데미지", "방어율 무시"]),
    "데몬어벤져": ("최대 HP", ["최대 HP", "올스탯", "공격력", "보스 데미지", "방어율 무시", "크리티컬 데미지"]),
}

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
    "after_potential_options",
    "after_potential_option",
    "after_option",
    "potential_options",
    "additional_potential_options",
    "after_additional_potential_option",
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
    keywords.extend(get_default_major_options(job_name))
    keywords.extend(selected_stats or [])
    return sorted(set(keywords))


def normalize_class_name(character_class: str | None) -> str:
    if not character_class:
        return ""
    return re.sub(r"[^0-9A-Za-z가-힣]", "", str(character_class)).strip()


def infer_main_stat_by_class(character_class: str | None) -> str | None:
    normalized = normalize_class_name(character_class)
    if not normalized:
        return None
    for class_name, (main_stat, _) in CLASS_MAIN_STAT_MAP.items():
        if normalize_class_name(class_name) == normalized:
            return main_stat
    for class_name, (main_stat, _) in CLASS_MAIN_STAT_MAP.items():
        if normalize_class_name(class_name) in normalized or normalized in normalize_class_name(class_name):
            return main_stat
    return None


def get_default_major_options(character_class: str | None) -> list[str]:
    normalized = normalize_class_name(character_class)
    if not normalized:
        return []
    for class_name, (_, options) in CLASS_MAIN_STAT_MAP.items():
        if normalize_class_name(class_name) == normalized:
            return list(dict.fromkeys(options))
    for class_name, (_, options) in CLASS_MAIN_STAT_MAP.items():
        if normalize_class_name(class_name) in normalized or normalized in normalize_class_name(class_name):
            return list(dict.fromkeys(options))
    return []


def extract_potential_options(row: pd.Series) -> list[str]:
    options: list[str] = []
    for col in OPTION_COLUMNS:
        if col not in row.index:
            continue
        options.extend(_extract_from_value(row[col]))
    return [option for option in options if option]


def is_major_option(options: list[str], effective_keywords: list[str]) -> bool:
    return count_effective_lines(options, effective_keywords) >= 1


def is_effective_option(options: list[str], effective_keywords: list[str]) -> bool:
    return count_effective_lines(options, effective_keywords) >= 2


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
    target_rule: dict[str, Any] | None = None,
) -> pd.DataFrame:
    output = df.copy()
    if output.empty:
        output["options_after"] = []
        output["effective_keywords"] = []
        output["effective_line_count"] = []
        output["is_major_option"] = []
        output["is_effective_option"] = []
        output["target_keywords"] = []
        output["target_line_count"] = []
        output["is_target_option"] = []
        return output

    effective_keywords = get_effective_keywords(job_name, selected_stats)
    resolved_target_rule = target_rule or resolve_target_rule(
        preset_name="주스탯 2줄 이상",
        job_name=job_name,
        selected_stats=selected_stats,
        custom_target_stats=selected_stats,
        min_lines=2,
        combo_primary=[],
        combo_secondary=[],
    )
    output["options_after"] = output.apply(extract_potential_options, axis=1)
    output["effective_keywords"] = [effective_keywords for _ in range(len(output))]
    output["effective_line_count"] = output["options_after"].map(
        lambda options: count_effective_lines(options, effective_keywords)
    )
    output["major_option_count"] = output["effective_line_count"]
    output["is_major_option"] = output["effective_line_count"] >= 1
    output["is_effective_option"] = output["effective_line_count"] >= 2
    output["has_major_option"] = output["is_major_option"]
    output["has_effective_option"] = output["is_effective_option"]
    output["target_keywords"] = [resolved_target_rule.get("display_keywords", []) for _ in range(len(output))]
    output["target_line_count"] = output["options_after"].map(
        lambda options: count_target_lines(options, resolved_target_rule)
    )
    output["is_target_option"] = output["options_after"].map(
        lambda options: is_target_option(options, resolved_target_rule)
    )
    output["target_preset_name"] = resolved_target_rule.get("name")
    return output


def summarize_effective_options(df: pd.DataFrame) -> dict:
    total = int(len(df))
    major_count = int(df.get("is_major_option", pd.Series(False, index=df.index)).fillna(False).sum()) if total else 0
    effective_count = int(df.get("is_effective_option", pd.Series(False, index=df.index)).fillna(False).sum()) if total else 0
    target_count = int(df.get("is_target_option", pd.Series(False, index=df.index)).fillna(False).sum()) if total else 0
    avg_lines = float(df["effective_line_count"].fillna(0).mean()) if total and "effective_line_count" in df.columns else 0.0
    avg_target_lines = float(df["target_line_count"].fillna(0).mean()) if total and "target_line_count" in df.columns else 0.0
    return {
        "total_cube_uses": total,
        "major_count": major_count,
        "major_rate": major_count / total if total else None,
        "effective_count": effective_count,
        "effective_rate": effective_count / total if total else None,
        "target_count": target_count,
        "target_rate": target_count / total if total else None,
        "avg_effective_lines": avg_lines,
        "avg_target_lines": avg_target_lines,
        "best_weekday_for_major": _best_group_label(summarize_major_by_weekday(df), "weekday_kr", "major_rate"),
        "best_hour_for_major": _best_group_label(summarize_major_by_hour(df), "hour_label", "success_rate"),
        "best_hour_band_for_major": _best_group_label(summarize_major_by_hour_band(df), "hour_band", "major_rate"),
        "best_weekday_for_effective": _best_group_label(summarize_effective_by_weekday(df), "weekday_kr", "effective_rate"),
        "best_hour_for_effective": _best_group_label(summarize_effective_by_hour(df), "hour_label", "success_rate"),
        "best_hour_band_for_effective": _best_group_label(summarize_effective_by_hour_band(df), "hour_band", "effective_rate"),
        "best_weekday_for_target": _best_group_label(summarize_target_by_weekday(df), "weekday_kr", "target_rate"),
        "best_hour_for_target": _best_group_label(summarize_target_by_hour(df), "hour_label", "success_rate"),
        "best_hour_band_for_target": _best_group_label(summarize_target_by_hour_band(df), "hour_band", "target_rate"),
    }


def summarize_major_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    return _group_boolean_rate(df, "weekday_kr", "is_major_option", "major_rate")


def summarize_effective_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    return _group_boolean_rate(df, "weekday_kr", "is_effective_option", "effective_rate")


def summarize_major_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    return summarize_by_hour(df, success_col="is_major_option", min_attempts=3)


def summarize_effective_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    return summarize_by_hour(df, success_col="is_effective_option", min_attempts=3)


def summarize_major_by_hour_band(df: pd.DataFrame) -> pd.DataFrame:
    return _group_boolean_rate(df, "hour_band", "is_major_option", "major_rate")


def summarize_effective_by_hour_band(df: pd.DataFrame) -> pd.DataFrame:
    return _group_boolean_rate(df, "hour_band", "is_effective_option", "effective_rate")


def summarize_major_by_cube_type(df: pd.DataFrame) -> pd.DataFrame:
    return _group_boolean_rate(df, "cube_type", "is_major_option", "major_rate")


def summarize_effective_by_cube_type(df: pd.DataFrame) -> pd.DataFrame:
    return _group_boolean_rate(df, "cube_type", "is_effective_option", "effective_rate")


def summarize_target_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    return _group_boolean_rate(df, "weekday_kr", "is_target_option", "target_rate")


def summarize_target_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    return summarize_by_hour(df, success_col="is_target_option", min_attempts=3)


def summarize_target_by_hour_band(df: pd.DataFrame) -> pd.DataFrame:
    return _group_boolean_rate(df, "hour_band", "is_target_option", "target_rate")


def summarize_target_by_cube_type(df: pd.DataFrame) -> pd.DataFrame:
    return _group_boolean_rate(df, "cube_type", "is_target_option", "target_rate")


def summarize_target_by_item(df: pd.DataFrame) -> pd.DataFrame:
    return _group_boolean_rate(df, "item_name", "is_target_option", "target_rate")


def summarize_effective_by_item(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "item_name" not in df.columns:
        return pd.DataFrame(
            columns=[
                "item_name",
                "attempt_count",
                "major_count",
                "major_rate",
                "effective_count",
                "effective_rate",
                "target_count",
                "target_rate",
            ]
        )
    major_summary = summarize_major_by_item(df)
    effective_summary = _group_boolean_rate(df, "item_name", "is_effective_option", "effective_rate").rename(
        columns={"success_count": "effective_count"}
    )
    target_summary = _group_boolean_rate(df, "item_name", "is_target_option", "target_rate").rename(
        columns={"success_count": "target_count"}
    )
    merged = major_summary.merge(
        effective_summary[["item_name", "effective_count", "effective_rate"]],
        on="item_name",
        how="outer",
    ).merge(
        target_summary[["item_name", "target_count", "target_rate"]],
        on="item_name",
        how="outer",
    )
    merged["attempt_count"] = merged["attempt_count"].fillna(0).astype(int)
    merged["major_count"] = merged["major_count"].fillna(0).astype(int)
    merged["effective_count"] = merged["effective_count"].fillna(0).astype(int)
    merged["target_count"] = merged["target_count"].fillna(0).astype(int)
    return merged.sort_values("attempt_count", ascending=False)


def summarize_major_by_item(df: pd.DataFrame) -> pd.DataFrame:
    return _group_boolean_rate(df, "item_name", "is_major_option", "major_rate").rename(
        columns={"success_count": "major_count"}
    )


def summarize_option_context_map(
    df: pd.DataFrame,
    target_col: str = "is_effective_option",
    rate_col: str = "effective_rate",
) -> pd.DataFrame:
    columns = ["cube_type", "weekday_kr", "hour_band", "attempt_count", "success_count", rate_col]
    if df.empty or any(col not in df.columns for col in ["cube_type", "weekday_kr", "hour_band", target_col]):
        return pd.DataFrame(columns=columns)

    working = df.dropna(subset=["cube_type", "weekday_kr", "hour_band"]).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)

    working[target_col] = working[target_col].fillna(False).astype(bool)
    summary = (
        working.groupby(["cube_type", "weekday_kr", "hour_band"], as_index=False)
        .agg(attempt_count=(target_col, "size"), success_count=(target_col, "sum"))
    )
    summary[rate_col] = summary["success_count"] / summary["attempt_count"]
    return summary.sort_values(["cube_type", rate_col, "attempt_count"], ascending=[True, False, False])


def resolve_target_rule(
    preset_name: str,
    job_name: str | None,
    selected_stats: list[str],
    custom_target_stats: list[str],
    min_lines: int,
    combo_primary: list[str],
    combo_secondary: list[str],
) -> dict[str, Any]:
    effective_keywords = get_effective_keywords(job_name, selected_stats)
    main_stat_keywords = [keyword for keyword in effective_keywords if keyword in {"STR", "DEX", "INT", "LUK", "최대 HP", "올스탯"}]
    offensive_keywords = ["공격력", "마력"]
    boss_keywords = ["보스 데미지", "방어율 무시", "크리티컬 데미지"]
    outlier_keywords = ["메소 획득량", "아이템 드롭률", "재사용 대기시간 미적용", "최대 HP", "올스탯"]

    if preset_name == "주스탯 2줄 이상":
        groups = [main_stat_keywords or custom_target_stats or selected_stats]
        rule_min_lines = max(min_lines, 2)
    elif preset_name == "공격력/마력 2줄 이상":
        groups = [offensive_keywords]
        rule_min_lines = max(min_lines, 2)
    elif preset_name == "보공 + 공격력/마력 조합":
        groups = [["보스 데미지"], offensive_keywords]
        rule_min_lines = 2
    elif preset_name == "보공/방무/크뎀 중 2줄 이상":
        groups = [boss_keywords]
        rule_min_lines = max(min_lines, 2)
    elif preset_name == "크뎀 포함 2줄 이상":
        groups = [["크리티컬 데미지"], boss_keywords]
        rule_min_lines = max(min_lines, 2)
    elif preset_name == "이탈 옵션 포함":
        groups = [outlier_keywords]
        rule_min_lines = 1
    else:
        direct_keywords = custom_target_stats or selected_stats
        groups = [direct_keywords]
        if combo_primary:
            groups = [combo_primary]
            if combo_secondary:
                groups.append(combo_secondary)
        rule_min_lines = max(1, min_lines)

    display_keywords = sorted({keyword for group in groups for keyword in group if keyword})
    return {
        "name": preset_name,
        "groups": groups,
        "min_lines": rule_min_lines,
        "display_keywords": display_keywords,
        "require_all_groups": len(groups) > 1,
    }


def count_target_lines(options: list[str], target_rule: dict[str, Any] | None) -> int:
    if not target_rule:
        return 0
    canonical_groups = [
        [_canonical_keyword(keyword) for keyword in group]
        for group in target_rule.get("groups", [])
        if group
    ]
    if not canonical_groups:
        return 0

    total_matches = 0
    for option in options:
        canonical_option = _canonical_option(option)
        for group in canonical_groups:
            if any(_matches(canonical_option, keyword) for keyword in group):
                total_matches += 1
                break
    return total_matches


def is_target_option(options: list[str], target_rule: dict[str, Any] | None) -> bool:
    if not target_rule:
        return False

    total_matches = count_target_lines(options, target_rule)
    if total_matches < int(target_rule.get("min_lines", 1)):
        return False

    if not target_rule.get("require_all_groups"):
        return True

    canonical_groups = [
        [_canonical_keyword(keyword) for keyword in group]
        for group in target_rule.get("groups", [])
        if group
    ]
    for group in canonical_groups:
        group_hit = False
        for option in options:
            canonical_option = _canonical_option(option)
            if any(_matches(canonical_option, keyword) for keyword in group):
                group_hit = True
                break
        if not group_hit:
            return False
    return True


def summarize_named_combos(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["combo_name", "attempt_count", "success_count", "rate"]
    if df.empty or "options_after" not in df.columns:
        return pd.DataFrame(columns=columns)

    combo_rules = {
        "주스탯 + 공격력/마력": [["STR", "DEX", "INT", "LUK", "최대 HP"], ["공격력", "마력"]],
        "보공/방무/크뎀 2줄 이상": [["보스 데미지", "방어율 무시", "크리티컬 데미지"]],
    }
    rows: list[dict[str, Any]] = []
    for combo_name, groups in combo_rules.items():
        temp_rule = {
            "groups": groups,
            "min_lines": 2,
            "require_all_groups": len(groups) > 1,
        }
        matched = df["options_after"].map(lambda options: is_target_option(options, temp_rule))
        attempt_count = int(len(df))
        success_count = int(matched.sum())
        rows.append(
            {
                "combo_name": combo_name,
                "attempt_count": attempt_count,
                "success_count": success_count,
                "rate": success_count / attempt_count if attempt_count else None,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _group_boolean_rate(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    rate_col: str,
) -> pd.DataFrame:
    columns = [group_col, "attempt_count", "success_count", rate_col, "luck_score"]
    if df.empty or group_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame(columns=columns)
    working = df.dropna(subset=[group_col]).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working[target_col] = working[target_col].fillna(False).astype(bool)
    overall = working[target_col].mean()
    summary = (
        working.groupby(group_col, as_index=False)
        .agg(attempt_count=(target_col, "size"), success_count=(target_col, "sum"))
    )
    summary[rate_col] = summary["success_count"] / summary["attempt_count"]
    summary["luck_score"] = (50 + (summary[rate_col] - overall) * 100).clip(0, 100)
    return summary.sort_values([rate_col, "attempt_count"], ascending=[False, False])


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
