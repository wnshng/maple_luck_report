from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import CACHE_BYPASS_RECENT_DAYS, RAW_DATA_DIR
from .nexon_client import NexonMapleClient
from .preprocess import (
    normalize_cube_history,
    normalize_potential_history,
    normalize_starforce_history,
    prepare_uploaded_dataframe,
)
from .storage import (
    daily_raw_json_path,
    load_json_if_exists,
    raw_json_path,
    save_raw_json,
)


@dataclass(frozen=True)
class LoadedHistory:
    raw_records: list[dict]
    dataframe: pd.DataFrame
    raw_path: Path
    csv_path: Path | None = None
    messages: tuple[str, ...] = ()


def fetch_cube_history_dataframe(
    client: NexonMapleClient,
    start_date: str,
    end_date: str,
) -> LoadedHistory:
    return _fetch_history_dataframe_with_cache(
        client=client,
        kind="cube",
        start_date=start_date,
        end_date=end_date,
        normalize_fn=normalize_cube_history,
        fetch_daily_fn=client.get_all_cube_history_for_date,
    )


def fetch_starforce_history_dataframe(
    client: NexonMapleClient,
    start_date: str,
    end_date: str,
) -> LoadedHistory:
    return _fetch_history_dataframe_with_cache(
        client=client,
        kind="starforce",
        start_date=start_date,
        end_date=end_date,
        normalize_fn=normalize_starforce_history,
        fetch_daily_fn=client.get_all_starforce_history_for_date,
    )


def fetch_potential_history_dataframe(
    client: NexonMapleClient,
    start_date: str,
    end_date: str,
) -> LoadedHistory:
    return _fetch_history_dataframe_with_cache(
        client=client,
        kind="potential",
        start_date=start_date,
        end_date=end_date,
        normalize_fn=normalize_potential_history,
        fetch_daily_fn=client.get_all_potential_history_for_date,
    )


def read_uploaded_csv(uploaded_file, kind: str) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return prepare_uploaded_dataframe(df, kind)


def _fetch_history_dataframe_with_cache(
    client: NexonMapleClient,
    kind: str,
    start_date: str,
    end_date: str,
    normalize_fn: Callable[[list[dict]], pd.DataFrame],
    fetch_daily_fn: Callable[[str], list[dict]],
) -> LoadedHistory:
    cached = _load_exact_cached_history(kind, start_date, end_date)
    if cached is not None:
        client.last_debug_info.update(
            {
                "cache_enabled": True,
                "loaded_from_exact_cache": True,
                "requested_date_count": _date_range_length(start_date, end_date),
                "successful_date_count": _date_range_length(start_date, end_date),
                "cache_hit_count": _date_range_length(start_date, end_date),
                "api_call_count": 0,
                "zero_count_dates": [],
                "error_dates": [],
                "total_record_count": len(cached.raw_records),
            }
        )
        return cached

    records: list[dict] = []
    cache_hit_dates: list[str] = []
    api_call_dates: list[str] = []
    zero_count_dates: list[str] = []
    error_dates: list[dict[str, str]] = []
    messages: list[str] = []

    for target_date in _iterate_dates(start_date, end_date):
        target_str = target_date.isoformat()
        daily_cache_path = daily_raw_json_path(kind, target_str, RAW_DATA_DIR)
        daily_records: list[dict] | None = None

        if _can_use_daily_cache(target_date):
            cached_daily = load_json_if_exists(daily_cache_path)
            if isinstance(cached_daily, list):
                daily_records = cached_daily
                cache_hit_dates.append(target_str)

        if daily_records is None:
            try:
                daily_records = fetch_daily_fn(target_str)
                save_raw_json(daily_records, daily_cache_path)
                api_call_dates.append(target_str)
            except Exception as exc:
                if _is_fatal_history_error(exc):
                    raise
                error_dates.append({"date": target_str, "error": str(exc)})
                continue

        if not daily_records:
            zero_count_dates.append(target_str)
        records.extend(daily_records)

    df = normalize_fn(records)
    raw_path = raw_json_path(kind, start_date, end_date, RAW_DATA_DIR)
    save_raw_json(records, raw_path)

    if cache_hit_dates:
        messages.append(f"캐시 재사용 {len(cache_hit_dates)}일, 신규 API 호출 {len(api_call_dates)}일")
    if error_dates:
        messages.append(f"일부 날짜 조회 실패 {len(error_dates)}일, 불러온 데이터만으로 리포트를 생성했습니다.")

    client.last_debug_info.update(
        {
            "cache_enabled": True,
            "loaded_from_exact_cache": False,
            "cache_hit_dates": cache_hit_dates,
            "cache_hit_count": len(cache_hit_dates),
            "api_call_dates": api_call_dates,
            "api_call_count": len(api_call_dates),
            "requested_date_count": _date_range_length(start_date, end_date),
            "successful_date_count": _date_range_length(start_date, end_date) - len(error_dates),
            "zero_count_dates": zero_count_dates,
            "error_dates": error_dates,
            "total_record_count": len(records),
        }
    )
    return LoadedHistory(records, df, raw_path, None, tuple(messages))


def _load_exact_cached_history(kind: str, start_date: str, end_date: str) -> LoadedHistory | None:
    if not _can_use_exact_cache(end_date):
        return None

    raw_path = raw_json_path(kind, start_date, end_date, RAW_DATA_DIR)
    raw_records = load_json_if_exists(raw_path)
    if not isinstance(raw_records, list):
        return None

    prepared_df = prepare_uploaded_dataframe(normalize_for_cache(raw_records, kind), kind)
    return LoadedHistory(
        raw_records=raw_records,
        dataframe=prepared_df,
        raw_path=raw_path,
        csv_path=None,
        messages=("저장된 범위 캐시를 재사용했습니다.",),
    )


def normalize_for_cache(raw_records: list[dict], kind: str) -> pd.DataFrame:
    if kind == "cube":
        return normalize_cube_history(raw_records)
    if kind == "potential":
        return normalize_potential_history(raw_records)
    return normalize_starforce_history(raw_records)


def _iterate_dates(start_date: str, end_date: str):
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _date_range_length(start_date: str, end_date: str) -> int:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    return (end - start).days + 1


def _can_use_exact_cache(end_date: str) -> bool:
    return _parse_date(end_date) < date.today()


def _can_use_daily_cache(target_date: date) -> bool:
    return target_date <= (date.today() - timedelta(days=CACHE_BYPASS_RECENT_DAYS))


def _is_fatal_history_error(exc: Exception) -> bool:
    text = str(exc)
    fatal_markers = [
        "API Key가 올바르지 않거나 권한이 없습니다",
        "요청 파라미터 오류",
        "API 호출 실패: 400",
        "API 호출 실패: 401",
        "API 호출 실패: 403",
    ]
    return any(marker in text for marker in fatal_markers)
