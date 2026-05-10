from __future__ import annotations

import copy
import logging
import time
from datetime import date, datetime, timedelta
from typing import Any

import requests

from .config import (
    API_KEY_HEADER,
    BASE_URL,
    DEFAULT_PAGE_SIZE,
    MAX_REQUEST_RETRIES,
    MAX_RETRY_BACKOFF_SECONDS,
    MAPLE_BASE_PATH,
    REQUEST_SLEEP_SECONDS,
    RETRY_BACKOFF_BASE_SECONDS,
)

logger = logging.getLogger(__name__)

USER_OUID_PATH = f"{MAPLE_BASE_PATH}/ouid"
CHARACTER_ID_PATH = f"{MAPLE_BASE_PATH}/id"
CHARACTER_LIST_PATH = f"{MAPLE_BASE_PATH}/character/list"
CHARACTER_BASIC_PATH = f"{MAPLE_BASE_PATH}/character/basic"
STARFORCE_HISTORY_PATH = f"{MAPLE_BASE_PATH}/history/starforce"
POTENTIAL_HISTORY_PATH = f"{MAPLE_BASE_PATH}/history/potential"
CUBE_HISTORY_PATH = f"{MAPLE_BASE_PATH}/history/cube"


class NexonAPIError(RuntimeError):
    """Raised when Nexon Open API returns an error response."""


class NexonMapleClient:
    def __init__(self, api_key: str):
        if not api_key or not api_key.strip():
            raise ValueError("Nexon Open API Key가 비어 있습니다. 사이드바에 API Key를 입력해 주세요.")

        self.api_key = api_key.strip()
        self.base_url = BASE_URL
        self.headers = {API_KEY_HEADER: self.api_key}
        self.last_debug_info: dict[str, Any] = {}
        self.session = requests.Session()
        self.session.headers.update(
            {
                **self.headers,
                "Accept": "application/json",
                "User-Agent": "maple-luck-report/0.4.0",
            }
        )

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        from .analytics.logger import log_api_call

        url = self.base_url + path
        safe_params = {key: value for key, value in (params or {}).items() if value is not None}
        attempt = 0
        while True:
            started_at = time.perf_counter()
            try:
                response = self.session.get(url, params=safe_params, timeout=30)
            except requests.RequestException as exc:
                response_time_ms = round((time.perf_counter() - started_at) * 1000, 2)
                self.last_debug_info = {
                    "path": path,
                    "params": safe_params,
                    "status_code": None,
                    "error_body": str(exc),
                    "retry_attempt": attempt,
                }
                log_api_call("nexon_maplestory", path, "failed", response_time_ms=response_time_ms, error_type=type(exc).__name__)
                if attempt >= MAX_REQUEST_RETRIES - 1:
                    raise NexonAPIError(f"API 요청 중 네트워크 오류가 발생했습니다: {exc}") from exc
                delay = _retry_delay(attempt)
                self.last_debug_info["retry_delay_seconds"] = delay
                time.sleep(delay)
                attempt += 1
                continue

            self.last_debug_info = {
                "path": path,
                "params": safe_params,
                "status_code": response.status_code,
                "retry_attempt": attempt,
            }
            response_time_ms = round((time.perf_counter() - started_at) * 1000, 2)

            if response.status_code == 200:
                data = response.json()
                if not isinstance(data, dict):
                    log_api_call("nexon_maplestory", path, "failed", response_time_ms=response_time_ms, error_type="invalid_response_format")
                    raise RuntimeError("API 응답 형식이 예상과 다릅니다.")

                self.last_debug_info.update(
                    {
                        "response_keys": list(data.keys()),
                        "raw_response_preview": _safe_preview(data),
                        "retry_count": attempt,
                        "response_time_ms": response_time_ms,
                    }
                )
                log_api_call("nexon_maplestory", path, "success", response_time_ms=response_time_ms)
                return data

            try:
                error_body: Any = response.json()
            except Exception:
                error_body = response.text

            self.last_debug_info["error_body"] = error_body
            self.last_debug_info["response_time_ms"] = response_time_ms
            error_text = str(error_body)
            log_api_call("nexon_maplestory", path, "failed", response_time_ms=response_time_ms, error_type=str(response.status_code))

            if response.status_code in {429, 500, 502, 503, 504} and attempt < MAX_REQUEST_RETRIES - 1:
                delay = _retry_delay(attempt, response.headers.get("Retry-After"))
                self.last_debug_info["retry_delay_seconds"] = delay
                time.sleep(delay)
                attempt += 1
                continue

            if response.status_code == 400:
                if "invalid api key" in error_text.lower() or "api key" in error_text.lower():
                    raise RuntimeError(f"API Key를 확인해 주세요. 응답: {error_body}")
                if "data being prepared" in error_text.lower() or "preparing" in error_text.lower():
                    raise RuntimeError(f"조회 대상 데이터가 아직 준비 중입니다. 잠시 후 다시 시도해 주세요. 응답: {error_body}")
                raise RuntimeError(f"요청 파라미터 오류입니다. date/cursor/count 값을 확인해주세요. 응답: {error_body}")
            if response.status_code in [401, 403]:
                raise RuntimeError(f"API Key가 올바르지 않거나 권한이 없습니다. 응답: {error_body}")
            if response.status_code == 429:
                raise RuntimeError(f"API 호출 한도를 초과했습니다. 잠시 후 다시 시도해주세요. 응답: {error_body}")
            if response.status_code >= 500:
                raise RuntimeError(f"Nexon API 서버 오류입니다. 응답: {error_body}")
            raise RuntimeError(f"API 호출 실패: {response.status_code}, 응답: {error_body}")

    def get_ouid(self, character_name: str) -> str:
        character_name = character_name.strip()
        if not character_name:
            raise ValueError("ouid 조회를 위해 캐릭터명을 입력해 주세요.")

        response = self._get(USER_OUID_PATH, params={"character_name": character_name})
        ouid = response.get("ouid")
        self.last_debug_info.update(
            {
                "response_keys": list(response.keys()),
                "ouid_found": bool(ouid),
            }
        )
        if not ouid:
            raise RuntimeError("ouid 조회 응답에 ouid가 없습니다.")
        return str(ouid)

    def get_character_id(self, character_name: str) -> str:
        character_name = character_name.strip()
        if not character_name:
            raise ValueError("캐릭터명으로 조회하려면 캐릭터명을 입력해 주세요.")

        response = self._get(CHARACTER_ID_PATH, params={"character_name": character_name})
        ocid = response.get("ocid")
        self.last_debug_info.update(
            {
                "response_keys": list(response.keys()),
                "ocid_found": bool(ocid),
            }
        )
        if not ocid:
            raise RuntimeError("캐릭터 식별자 조회 응답에 ocid가 없습니다.")
        return str(ocid)

    def get_character_list(self) -> dict[str, Any]:
        response = self._get(CHARACTER_LIST_PATH)
        self.last_debug_info.update(
            {
                "response_keys": list(response.keys()),
                "account_list_count": len(response.get("account_list", []))
                if isinstance(response.get("account_list"), list)
                else 0,
            }
        )
        return response

    def get_character_basic(self, ocid: str, query_date: str | None = None) -> dict[str, Any]:
        ocid = str(ocid).strip()
        if not ocid:
            raise ValueError("캐릭터 기본 정보를 조회하려면 ocid가 필요합니다.")

        params: dict[str, Any] = {"ocid": ocid}
        if query_date:
            params["date"] = query_date

        response = self._get(CHARACTER_BASIC_PATH, params=params)
        self.last_debug_info.update(
            {
                "response_keys": list(response.keys()),
                "character_basic_found": bool(response),
            }
        )
        return response

    def get_starforce_history_by_date(self, date: str, count: int = DEFAULT_PAGE_SIZE) -> dict[str, Any]:
        # Official docs: GET /maplestory/v1/history/starforce
        # First page must use `date` and `count`; starforce does not accept ouid/start_date/end_date.
        params = {
            "date": date,
            "count": str(count),
        }
        data = self._get(STARFORCE_HISTORY_PATH, params=params)
        self.last_debug_info.update(
            {
                "count": data.get("count"),
                "record_key_exists": "starforce_history" in data,
                "record_count": len(data.get("starforce_history", []))
                if isinstance(data.get("starforce_history", []), list)
                else None,
                "next_cursor_exists": bool(data.get("next_cursor")),
            }
        )
        return data

    def get_starforce_history_by_cursor(self, cursor: str, count: int = DEFAULT_PAGE_SIZE) -> dict[str, Any]:
        # Official docs: subsequent pages must use `cursor` and `count`.
        params = {
            "cursor": cursor,
            "count": str(count),
        }
        data = self._get(STARFORCE_HISTORY_PATH, params=params)
        self.last_debug_info.update(
            {
                "count": data.get("count"),
                "record_key_exists": "starforce_history" in data,
                "record_count": len(data.get("starforce_history", []))
                if isinstance(data.get("starforce_history", []), list)
                else None,
                "next_cursor_exists": bool(data.get("next_cursor")),
            }
        )
        return data

    def get_all_starforce_history_for_date(
        self,
        date: str,
        count: int = DEFAULT_PAGE_SIZE,
        max_pages: int = 300,
    ) -> list[dict[str, Any]]:
        # `starforce_history` is the official record list key and `next_cursor` is the only next-page key.
        return self._collect_history_for_date(
            target_date=date,
            record_key="starforce_history",
            request_by_date=self.get_starforce_history_by_date,
            request_by_cursor=self.get_starforce_history_by_cursor,
            count=count,
            max_pages=max_pages,
        )

    def get_all_starforce_history_by_date_range(
        self,
        start_date: str,
        end_date: str,
        count: int = DEFAULT_PAGE_SIZE,
    ) -> list[dict[str, Any]]:
        # Starforce does not accept a direct date range, so the app iterates KST dates one by one.
        start = _parse_yyyy_mm_dd(start_date)
        end = _parse_yyyy_mm_dd(end_date)
        if start > end:
            raise ValueError("시작일은 종료일보다 늦을 수 없습니다.")

        all_records: list[dict[str, Any]] = []
        zero_count_dates: list[str] = []
        error_dates: list[dict[str, str]] = []
        successful_dates = 0
        requested_dates = 0
        current = start

        while current <= end:
            current_date = current.isoformat()
            requested_dates += 1
            try:
                daily_records = self.get_all_starforce_history_for_date(current_date, count=count)
                all_records.extend(daily_records)
                successful_dates += 1
                if not daily_records:
                    zero_count_dates.append(current_date)
                self.last_debug_info.update(
                    {
                        "range_start_date": start_date,
                        "range_end_date": end_date,
                        "requested_date_count": requested_dates,
                        "successful_date_count": successful_dates,
                        "total_record_count": len(all_records),
                        "zero_count_dates": zero_count_dates,
                        "error_dates": error_dates,
                    }
                )
                time.sleep(REQUEST_SLEEP_SECONDS)
            except Exception as exc:
                error_dates.append({"date": current_date, "error": str(exc)})
                self.last_debug_info.update(
                    {
                        "range_start_date": start_date,
                        "range_end_date": end_date,
                        "requested_date_count": requested_dates,
                        "successful_date_count": successful_dates,
                        "total_record_count": len(all_records),
                        "zero_count_dates": zero_count_dates,
                        "error_dates": error_dates,
                    }
                )
                raise RuntimeError(f"{current_date} 스타포스 조회 중 실패했습니다: {exc}") from exc

            current += timedelta(days=1)

        return all_records

    def get_potential_history(
        self,
        start_date: str,
        end_date: str,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        del end_date
        response = self._get(POTENTIAL_HISTORY_PATH, params=self._history_params(start_date, cursor))
        self.last_debug_info.update(
            {
                "count": response.get("count"),
                "record_key_exists": "potential_history" in response,
                "record_count": len(response.get("potential_history", []))
                if isinstance(response.get("potential_history", []), list)
                else None,
                "next_cursor_exists": bool(response.get("next_cursor")),
            }
        )
        return response

    def get_all_potential_history(self, start_date: str, end_date: str) -> list[dict[str, Any]]:
        return self._collect_history_by_date_range(
            start_date=start_date,
            end_date=end_date,
            record_key="potential_history",
            request_by_date=self.get_potential_history_by_date,
            request_by_cursor=self.get_potential_history_by_cursor,
        )

    def get_cube_history(
        self,
        start_date: str,
        end_date: str,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        del end_date
        response = self._get(CUBE_HISTORY_PATH, params=self._history_params(start_date, cursor))
        self.last_debug_info.update(
            {
                "count": response.get("count"),
                "record_key_exists": "cube_history" in response,
                "record_count": len(response.get("cube_history", []))
                if isinstance(response.get("cube_history", []), list)
                else None,
                "next_cursor_exists": bool(response.get("next_cursor")),
            }
        )
        return response

    def get_all_cube_history(self, start_date: str, end_date: str) -> list[dict[str, Any]]:
        return self._collect_history_by_date_range(
            start_date=start_date,
            end_date=end_date,
            record_key="cube_history",
            request_by_date=self.get_cube_history_by_date,
            request_by_cursor=self.get_cube_history_by_cursor,
        )

    def get_potential_history_by_date(self, date: str, count: int = DEFAULT_PAGE_SIZE) -> dict[str, Any]:
        return self._get(POTENTIAL_HISTORY_PATH, params={"date": date, "count": count})

    def get_potential_history_by_cursor(self, cursor: str, count: int = DEFAULT_PAGE_SIZE) -> dict[str, Any]:
        return self._get(POTENTIAL_HISTORY_PATH, params={"cursor": cursor, "count": count})

    def get_cube_history_by_date(self, date: str, count: int = DEFAULT_PAGE_SIZE) -> dict[str, Any]:
        return self._get(CUBE_HISTORY_PATH, params={"date": date, "count": count})

    def get_cube_history_by_cursor(self, cursor: str, count: int = DEFAULT_PAGE_SIZE) -> dict[str, Any]:
        return self._get(CUBE_HISTORY_PATH, params={"cursor": cursor, "count": count})

    def get_all_potential_history_for_date(
        self,
        date: str,
        count: int = DEFAULT_PAGE_SIZE,
        max_pages: int = 300,
    ) -> list[dict[str, Any]]:
        return self._collect_history_for_date(
            target_date=date,
            record_key="potential_history",
            request_by_date=self.get_potential_history_by_date,
            request_by_cursor=self.get_potential_history_by_cursor,
            count=count,
            max_pages=max_pages,
        )

    def get_all_cube_history_for_date(
        self,
        date: str,
        count: int = DEFAULT_PAGE_SIZE,
        max_pages: int = 300,
    ) -> list[dict[str, Any]]:
        return self._collect_history_for_date(
            target_date=date,
            record_key="cube_history",
            request_by_date=self.get_cube_history_by_date,
            request_by_cursor=self.get_cube_history_by_cursor,
            count=count,
            max_pages=max_pages,
        )

    def _collect_history_by_date_range(
        self,
        start_date: str,
        end_date: str,
        record_key: str,
        request_by_date,
        request_by_cursor,
        count: int = DEFAULT_PAGE_SIZE,
        max_pages: int = 300,
    ) -> list[dict[str, Any]]:
        start = _parse_yyyy_mm_dd(start_date)
        end = _parse_yyyy_mm_dd(end_date)
        if start > end:
            raise ValueError("시작일은 종료일보다 늦을 수 없습니다.")

        all_records: list[dict[str, Any]] = []
        current = start
        while current <= end:
            current_day = current.isoformat()
            response = request_by_date(current_day, count=count)
            page = 0
            while True:
                page += 1
                records = response.get(record_key, [])
                if not isinstance(records, list):
                    raise RuntimeError(f"{record_key} 응답이 list가 아닙니다.")
                all_records.extend(records)

                next_cursor = response.get("next_cursor")
                self.last_debug_info.update(
                    {
                        "current_query_date": current_day,
                        "current_page": page,
                        "count": response.get("count"),
                        "record_key_exists": record_key in response,
                        "record_count": len(records),
                        "next_cursor_exists": bool(next_cursor),
                        "next_cursor": next_cursor,
                        "total_record_count": len(all_records),
                    }
                )
                if not next_cursor:
                    break
                if page >= max_pages:
                    raise RuntimeError("페이지 수가 너무 많아 중단했습니다. 날짜 범위를 줄여주세요.")

                time.sleep(REQUEST_SLEEP_SECONDS)
                response = request_by_cursor(str(next_cursor), count=count)

            current += timedelta(days=1)
            time.sleep(REQUEST_SLEEP_SECONDS)

        return all_records

    def _collect_history_for_date(
        self,
        target_date: str,
        record_key: str,
        request_by_date,
        request_by_cursor,
        count: int = DEFAULT_PAGE_SIZE,
        max_pages: int = 300,
    ) -> list[dict[str, Any]]:
        all_records: list[dict[str, Any]] = []
        page = 0
        response = request_by_date(target_date, count=count)

        while True:
            page += 1
            records = response.get(record_key, [])
            if not isinstance(records, list):
                raise RuntimeError(f"{record_key} 응답이 list가 아닙니다.")

            all_records.extend(records)
            next_cursor = response.get("next_cursor")
            self.last_debug_info.update(
                {
                    "current_query_date": target_date,
                    "current_page": page,
                    "count": response.get("count"),
                    "record_key_exists": record_key in response,
                    "record_count": len(records),
                    "next_cursor_exists": bool(next_cursor),
                    "next_cursor": next_cursor,
                    "total_record_count": len(all_records),
                }
            )

            if not next_cursor:
                break
            if page >= max_pages:
                raise RuntimeError("페이지 수가 너무 많아 중단했습니다. 날짜 범위를 줄여주세요.")

            time.sleep(REQUEST_SLEEP_SECONDS)
            response = request_by_cursor(str(next_cursor), count=count)

        return all_records

    def _history_params(self, target_date: str, cursor: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"count": DEFAULT_PAGE_SIZE}
        if cursor:
            params["cursor"] = cursor
        else:
            params["date"] = target_date
        return params


def _safe_preview(payload: dict[str, Any], max_items: int = 3) -> dict[str, Any]:
    preview = copy.deepcopy(payload)
    for key, value in list(preview.items()):
        if isinstance(value, list) and len(value) > max_items:
            preview[key] = value[:max_items]
            preview[f"{key}_truncated_count"] = len(value) - max_items
    return preview


def _parse_yyyy_mm_dd(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("날짜는 YYYY-MM-DD 형식이어야 합니다.") from exc


def _retry_delay(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), MAX_RETRY_BACKOFF_SECONDS)
        except ValueError:
            pass
    return min(RETRY_BACKOFF_BASE_SECONDS * (2**attempt), MAX_RETRY_BACKOFF_SECONDS)
