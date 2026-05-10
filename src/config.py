from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

BASE_URL = "https://open.api.nexon.com"
MAPLE_BASE_PATH = "/maplestory/v1"
API_KEY_HEADER = "x-nxopen-api-key"
STARFORCE_MIN_DATE = "2023-12-27"
POTENTIAL_MIN_DATE = "2024-01-25"
MAX_LOOKBACK_YEARS = 2
DEFAULT_PAGE_SIZE = 1000
REQUEST_SLEEP_SECONDS = 0.05
MAX_REQUEST_RETRIES = 4
RETRY_BACKOFF_BASE_SECONDS = 0.75
MAX_RETRY_BACKOFF_SECONDS = 8.0
CACHE_BYPASS_RECENT_DAYS = 1
EXACT_CACHE_TTL_SECONDS = 600
PARALLEL_HISTORY_FETCH_WORKERS = 2

# Backward-compatible aliases for older modules/notebooks.
NEXON_OPEN_API_BASE_URL = BASE_URL
NEXON_API_KEY_HEADER = API_KEY_HEADER


@dataclass(frozen=True)
class AppSettings:
    """Runtime settings that should not include user secrets."""

    base_url: str = BASE_URL
    page_size: int = DEFAULT_PAGE_SIZE
    request_sleep_seconds: float = REQUEST_SLEEP_SECONDS


def ensure_data_dirs() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_env_api_key() -> str:
    return _get_config_value("NEXON_OPEN_API_KEY", "").strip()


def get_env_bool(name: str, default: bool = False) -> bool:
    value = _get_config_value(name, "true" if default else "false")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def is_streamlit_cloud_environment() -> bool:
    root_text = str(ROOT_DIR)
    if root_text.startswith("/mount/src"):
        return True
    runtime = _get_config_value("STREAMLIT_RUNTIME", "")
    return str(runtime).strip().lower() in {"streamlit_cloud", "community_cloud"}


def is_local_state_persistence_enabled() -> bool:
    if is_streamlit_cloud_environment():
        return False
    return get_env_bool("ENABLE_LOCAL_STATE_PERSISTENCE", default=False)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def clamp_date_range(start_date: date, end_date: date, data_type: str) -> tuple[date, date, list[str]]:
    """Clamp user-selected dates to the currently queryable range."""

    available_start, available_end = get_available_date_range(data_type)
    messages: list[str] = []
    clamped_start = start_date
    clamped_end = end_date

    if clamped_start < available_start:
        messages.append(
            f"{_data_type_label(data_type)}는 최대 최근 2년 범위만 조회 가능하여 시작일을 {available_start}로 자동 보정했습니다."
        )
        clamped_start = available_start

    if clamped_end > available_end:
        messages.append(f"종료일이 오늘 이후라 {available_end}로 자동 보정했습니다.")
        clamped_end = available_end

    if clamped_start > clamped_end:
        messages.append("조회 시작일이 종료일보다 늦습니다. 조회 가능한 날짜 범위를 다시 선택해주세요.")

    return clamped_start, clamped_end, messages


def get_available_date_range(data_type: str, today: date | None = None) -> tuple[date, date]:
    if today is None:
        today = get_today_kst()

    max_start_date = today - relativedelta(years=2) + timedelta(days=1)
    max_end_date = today
    normalized_type = data_type.lower().strip()

    if normalized_type == "starforce":
        api_min_date = datetime.strptime(STARFORCE_MIN_DATE, "%Y-%m-%d").date()
        start_date = max(max_start_date, api_min_date)
    elif normalized_type == "potential":
        api_min_date = datetime.strptime(POTENTIAL_MIN_DATE, "%Y-%m-%d").date()
        start_date = max(max_start_date, api_min_date)
    else:
        start_date = max_start_date

    return start_date, max_end_date


def get_today_kst() -> date:
    return datetime.now(ZoneInfo("Asia/Seoul")).date()


def get_default_date_range_years(years: int = 2) -> tuple[date, date]:
    end_date = get_today_kst()
    start_date = _subtract_years(end_date, years) + timedelta(days=1)
    if start_date > end_date:
        start_date = end_date
    return start_date, end_date


def get_default_two_year_range() -> tuple[date, date]:
    return get_default_date_range_years(2)


def _data_type_label(data_type: str) -> str:
    normalized_type = data_type.lower().strip()
    return {
        "starforce": "스타포스 강화 결과",
        "potential": "잠재능력 재설정 결과",
        "cube": "큐브 사용 결과",
    }.get(normalized_type, normalized_type)


def _minimum_date_for(data_type: str) -> date | None:
    if data_type == "starforce":
        return datetime.strptime(STARFORCE_MIN_DATE, "%Y-%m-%d").date()
    if data_type == "potential":
        return datetime.strptime(POTENTIAL_MIN_DATE, "%Y-%m-%d").date()
    if data_type == "cube":
        return None
    return None


def _subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year - years)


def _get_config_value(name: str, default: str | None = None) -> str:
    try:
        import streamlit as st

        if name in st.secrets:
            value = st.secrets.get(name)
            return str(value).strip() if value is not None else (default or "")
    except Exception:
        pass
    load_dotenv(ROOT_DIR / ".env")
    return os.getenv(name, default or "").strip()
