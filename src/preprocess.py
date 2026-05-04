from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


WEEKDAY_KR = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
GRADE_UP_KEYWORDS = ("등급", "상승", "업", "성공", "SUCCESS", "UPGRADE")
SUCCESS_KEYWORDS = ("성공", "SUCCESS")
DESTROY_KEYWORDS = ("파괴", "DESTROY")
DROP_KEYWORDS = ("하락", "DROP", "DOWN")

DATE_KEYS = ("date_create", "create_date", "event_datetime", "created_at", "date")
CHARACTER_KEYS = ("character_name", "character", "char_name")
WORLD_KEYS = ("world_name", "world")
ITEM_KEYS = ("target_item", "item_name", "item")
RESULT_KEYS = ("item_upgrade_result", "result", "starforce_result", "event_result")
CHANNEL_KEYS = ("channel", "channel_name", "channel_no", "ch", "server_channel")


def normalize_cube_history(raw_records: list[dict[str, Any]]) -> pd.DataFrame:
    raw_df = _raw_dataframe(raw_records)
    rows: list[dict[str, Any]] = []
    for record in raw_records or []:
        before_grade = _option_grade(record.get("before_potential_option")) or _option_grade(
            record.get("before_additional_potential_option")
        )
        after_grade = (
            record.get("potential_option_grade")
            or _option_grade(record.get("after_potential_option"))
            or record.get("additional_potential_option_grade")
            or _option_grade(record.get("after_additional_potential_option"))
        )
        result_type = record.get("item_upgrade_result")

        row = {
            "id": record.get("id"),
            "character_name": _first_present(record, CHARACTER_KEYS),
            "world_name": _first_present(record, WORLD_KEYS),
            "date_create": _first_present(record, DATE_KEYS),
            "item_name": _first_present(record, ITEM_KEYS),
            "cube_type": record.get("cube_type") or record.get("potential_type"),
            "before_potential_grade": before_grade,
            "after_potential_grade": after_grade,
            "result_type": result_type,
            "is_grade_up": _is_grade_up(result_type),
            "miracle_time_flag": record.get("miracle_time_flag"),
            "item_equipment_part": record.get("item_equipment_part"),
            "item_level": record.get("item_level"),
            "upgrade_guarantee": record.get("upgrade_guarantee"),
            "upgrade_guarantee_count": record.get("upgrade_guarantee_count"),
            "before_potential_option": _json_or_none(record.get("before_potential_option")),
            "after_potential_option": _json_or_none(record.get("after_potential_option")),
            "before_additional_potential_option": _json_or_none(
                record.get("before_additional_potential_option")
            ),
            "after_additional_potential_option": _json_or_none(
                record.get("after_additional_potential_option")
            ),
            "channel": _channel_value(record),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return _empty_cube_dataframe()
    return add_time_features(_merge_raw_columns(df, raw_df), "date_create")


def normalize_potential_history(raw_records: list[dict[str, Any]]) -> pd.DataFrame:
    df = normalize_cube_history(raw_records)
    if not df.empty and "cube_type" in df.columns:
        df["cube_type"] = df["cube_type"].fillna("잠재능력 재설정")
    return df


def normalize_starforce_history(raw_records: list[dict[str, Any]]) -> pd.DataFrame:
    raw_df = _raw_dataframe(raw_records)
    rows: list[dict[str, Any]] = []
    for record in raw_records or []:
        before_starforce_count = _to_number(record.get("before_starforce_count"))
        after_starforce_count = _to_number(record.get("after_starforce_count"))
        item_upgrade_result = record.get("item_upgrade_result")
        result_type = item_upgrade_result
        first_event = _first_starforce_event(record.get("starforce_event_list"))
        is_success = _starforce_is_success(item_upgrade_result, before_starforce_count, after_starforce_count)
        is_drop = _starforce_is_drop(item_upgrade_result, before_starforce_count, after_starforce_count)
        is_destroyed = _starforce_is_destroyed(item_upgrade_result)

        row = {
            "id": record.get("id"),
            "character_name": record.get("character_name"),
            "world_name": record.get("world_name"),
            "target_item": record.get("target_item"),
            "item_name": record.get("target_item"),
            "date_create": record.get("date_create"),
            "item_upgrade_result": item_upgrade_result,
            "result_type": result_type,
            "before_starforce_count": before_starforce_count,
            "after_starforce_count": after_starforce_count,
            "before_starforce": before_starforce_count,
            "after_starforce": after_starforce_count,
            "target_starforce": before_starforce_count + 1 if pd.notna(before_starforce_count) else np.nan,
            "is_success": is_success,
            "is_drop": is_drop,
            "is_destroyed": is_destroyed,
            "is_fail": not is_success,
            "starcatch_result": record.get("starcatch_result"),
            "superior_item_flag": record.get("superior_item_flag"),
            "destroy_defence": record.get("destroy_defence"),
            "chance_time": record.get("chance_time"),
            "event_field_flag": record.get("event_field_flag"),
            "upgrade_item": record.get("upgrade_item"),
            "protect_shield": record.get("protect_shield"),
            "bonus_stat_upgrade": record.get("bonus_stat_upgrade"),
            "starforce_event_list": _json_or_none(record.get("starforce_event_list")),
            "event_success_rate": first_event.get("success_rate") if first_event else None,
            "event_destroy_decrease_rate": first_event.get("destroy_decrease_rate") if first_event else None,
            "event_cost_discount_rate": first_event.get("cost_discount_rate") if first_event else None,
            "event_plus_value": first_event.get("plus_value") if first_event else None,
            "event_range": first_event.get("starforce_event_range") if first_event else None,
            "channel": None,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return _empty_starforce_dataframe()
    return add_time_features(_merge_raw_columns(df, raw_df), "date_create")


def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    output = df.copy()
    if output.empty:
        return output

    if date_col not in output.columns:
        output["event_datetime"] = pd.NaT
    else:
        output["event_datetime"] = pd.to_datetime(output[date_col], errors="coerce")

    output["event_date"] = output["event_datetime"].dt.date
    output["year"] = output["event_datetime"].dt.year
    output["month"] = output["event_datetime"].dt.month
    output["day"] = output["event_datetime"].dt.day
    output["weekday"] = output["event_datetime"].dt.weekday
    output["weekday_kr"] = output["weekday"].map(
        lambda value: WEEKDAY_KR[int(value)] if pd.notna(value) and 0 <= int(value) <= 6 else None
    )
    output["hour"] = output["event_datetime"].dt.hour
    output["hour_label"] = output["hour"].map(lambda value: f"{int(value)}시" if pd.notna(value) else None)
    output["hour_band"] = output["hour"].map(_hour_band)
    output["is_weekend"] = output["weekday"].isin([5, 6])
    return output


def prepare_uploaded_dataframe(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    output = df.copy()
    output = _map_common_uploaded_columns(output)

    if "event_datetime" not in output.columns:
        date_col = "date_create" if "date_create" in output.columns else "event_date"
        output = add_time_features(output, date_col)

    if kind in {"cube", "potential"}:
        if "cube_type" not in output.columns and "potential_type" in output.columns:
            output["cube_type"] = output["potential_type"]
        if kind == "potential" and "cube_type" not in output.columns:
            output["cube_type"] = "잠재능력 재설정"
        if "is_grade_up" in output.columns:
            output["is_grade_up"] = output["is_grade_up"].map(_to_bool)
        else:
            output["is_grade_up"] = output.get("result_type", pd.Series(index=output.index)).map(_is_grade_up)

    if kind == "starforce":
        if "before_starforce" not in output.columns and "before_starforce_count" in output.columns:
            output["before_starforce"] = pd.to_numeric(output["before_starforce_count"], errors="coerce")
        if "after_starforce" not in output.columns and "after_starforce_count" in output.columns:
            output["after_starforce"] = pd.to_numeric(output["after_starforce_count"], errors="coerce")
        if "target_starforce" not in output.columns and "before_starforce" in output.columns:
            output["target_starforce"] = output["before_starforce"] + 1
        if "item_upgrade_result" not in output.columns and "result_type" in output.columns:
            output["item_upgrade_result"] = output["result_type"]

        for col in ("is_success", "is_destroyed", "is_drop"):
            if col in output.columns:
                output[col] = output[col].map(_to_bool)
        if "is_success" not in output.columns:
            output["is_success"] = [
                _starforce_is_success(result, before, after)
                for result, before, after in zip(
                    output.get("item_upgrade_result", pd.Series(index=output.index)),
                    output.get("before_starforce", pd.Series(index=output.index)),
                    output.get("after_starforce", pd.Series(index=output.index)),
                )
            ]
        if "is_destroyed" not in output.columns:
            output["is_destroyed"] = output.get("item_upgrade_result", pd.Series(index=output.index)).map(
                _starforce_is_destroyed
            )
        if "is_drop" not in output.columns:
            output["is_drop"] = [
                _starforce_is_drop(result, before, after)
                for result, before, after in zip(
                    output.get("item_upgrade_result", pd.Series(index=output.index)),
                    output.get("before_starforce", pd.Series(index=output.index)),
                    output.get("after_starforce", pd.Series(index=output.index)),
                )
            ]
        if "is_fail" not in output.columns:
            output["is_fail"] = ~output["is_success"].fillna(False).astype(bool)
        if "channel" not in output.columns:
            output["channel"] = None
    return output


def _map_common_uploaded_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    _copy_first_existing(output, "date_create", DATE_KEYS)
    _copy_first_existing(output, "character_name", CHARACTER_KEYS)
    _copy_first_existing(output, "world_name", WORLD_KEYS)
    _copy_first_existing(output, "item_name", ITEM_KEYS)
    _copy_first_existing(output, "result_type", RESULT_KEYS)
    _copy_first_existing(output, "channel", CHANNEL_KEYS)
    return output


def _raw_dataframe(raw_records: list[dict[str, Any]]) -> pd.DataFrame:
    if not raw_records:
        return pd.DataFrame()
    return pd.json_normalize(raw_records, sep="_")


def _merge_raw_columns(standard_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return standard_df
    output = standard_df.copy()
    for col in raw_df.columns:
        if col not in output.columns:
            output[col] = raw_df[col].values
    return output


def _copy_first_existing(df: pd.DataFrame, target_col: str, source_cols: tuple[str, ...]) -> None:
    if target_col in df.columns:
        return
    for col in source_cols:
        if col in df.columns:
            df[target_col] = df[col]
            return


def _hour_band(hour: float | int | None) -> str | None:
    if pd.isna(hour):
        return None
    hour_int = int(hour)
    if 0 <= hour_int <= 5:
        return "새벽"
    if 6 <= hour_int <= 11:
        return "오전"
    if 12 <= hour_int <= 17:
        return "오후"
    if 18 <= hour_int <= 23:
        return "저녁"
    return None


def _option_grade(options: Any) -> str | None:
    if isinstance(options, list) and options:
        first = options[0]
        if isinstance(first, dict):
            return first.get("grade")
    return None


def _json_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _contains_any(value: Any, keywords: tuple[str, ...]) -> bool:
    if value is None:
        return False
    if not isinstance(value, (list, dict, tuple, set)) and pd.isna(value):
        return False
    normalized = str(value).upper()
    return any(keyword.upper() in normalized for keyword in keywords)


def _is_grade_up(result_type: Any) -> bool:
    return _contains_any(result_type, GRADE_UP_KEYWORDS)


def _starforce_is_success(item_upgrade_result: Any, before: float, after: float) -> bool:
    if pd.notna(before) and pd.notna(after) and after > before:
        return True
    return _contains_any(item_upgrade_result, SUCCESS_KEYWORDS)


def _starforce_is_drop(item_upgrade_result: Any, before: float, after: float) -> bool:
    if pd.notna(before) and pd.notna(after) and after < before:
        return True
    return _contains_any(item_upgrade_result, DROP_KEYWORDS)


def _starforce_is_destroyed(item_upgrade_result: Any) -> bool:
    return _contains_any(item_upgrade_result, DESTROY_KEYWORDS)


def _first_starforce_event(value: Any) -> dict[str, Any] | None:
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            return first
    return None


def _to_number(value: Any) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) else np.nan


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "성공", "등급업"}


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record.get(key) is not None:
            return record.get(key)
    return None


def _channel_value(record: dict[str, Any]) -> Any:
    return _first_present(record, CHANNEL_KEYS)


def _empty_cube_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "id",
            "character_name",
            "world_name",
            "date_create",
            "event_datetime",
            "event_date",
            "weekday_kr",
            "hour",
            "hour_label",
            "hour_band",
            "item_name",
            "cube_type",
            "before_potential_grade",
            "after_potential_grade",
            "result_type",
            "is_grade_up",
            "channel",
        ]
    )


def _empty_starforce_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "id",
            "character_name",
            "world_name",
            "target_item",
            "item_upgrade_result",
            "date_create",
            "event_datetime",
            "event_date",
            "weekday_kr",
            "hour",
            "hour_label",
            "hour_band",
            "item_name",
            "before_starforce_count",
            "after_starforce_count",
            "before_starforce",
            "after_starforce",
            "target_starforce",
            "result_type",
            "is_success",
            "is_destroyed",
            "is_drop",
            "is_fail",
            "starcatch_result",
            "destroy_defence",
            "chance_time",
            "starforce_event_list",
            "event_success_rate",
            "event_destroy_decrease_rate",
            "event_cost_discount_rate",
            "event_plus_value",
            "event_range",
            "channel",
        ]
    )
