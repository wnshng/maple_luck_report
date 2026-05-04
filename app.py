from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from src.auth import get_nexon_login_guide
from src.config import (
    clamp_date_range,
    ensure_data_dirs,
    get_available_date_range,
    get_env_api_key,
    setup_logging,
)
from src.data_loader import (
    fetch_cube_history_dataframe,
    fetch_potential_history_dataframe,
    fetch_starforce_history_dataframe,
    read_uploaded_csv,
)
from src.effective_options import (
    STAT_OPTIONS,
    add_effective_option_features,
    summarize_effective_by_cube_type,
    summarize_effective_by_hour,
    summarize_effective_by_hour_band,
    summarize_effective_by_item,
    summarize_effective_by_weekday,
    summarize_effective_options,
)
from src.metrics import (
    CHANNEL_NOTICE,
    compute_cube_luck_metrics,
    get_best_hour,
    summarize_by_hour,
    summarize_starforce,
    summarize_starforce_by_hour,
    summarize_starforce_by_hour_band,
    summarize_starforce_by_star_count,
    summarize_starforce_by_weekday,
)
from src.nexon_client import NexonAPIError, NexonMapleClient
from src.visualizations import (
    plot_channel_rate,
    plot_cube_type_rate,
    plot_daily_attempts,
    plot_hour_band_rate,
    plot_hourly_rate,
    plot_item_rate,
    plot_starforce_stage_rate,
    plot_weekday_rate,
)


NOTICE_TEXT = """
- 본 서비스는 Nexon Open API로 조회 가능한 본인 히스토리를 기반으로 한 개인 통계 리포트입니다.
- 운 점수와 유효옵션률은 과거 데이터 기반 통계이며, 미래 결과를 보장하지 않습니다.
- API 응답에 채널 정보가 없을 경우 채널별 분석은 제공하지 않습니다.
- Nexon Open API로 수집한 데이터는 정책상 30일 이내 갱신 의무가 있을 수 있으므로, 배포 시 최신 정책을 확인해야 합니다.
- API Key는 저장하지 않으며, 사용자의 세션에서만 사용합니다.
"""

STARFORCE_NOTICE = """
- 스타포스 강화 결과는 2023-12-27 이후 데이터부터 조회 가능하며 최대 2년 범위에서 조회됩니다.
- 스타포스 확률 정보는 최대 5분 후 확인 가능하므로 방금 강화한 기록은 즉시 반영되지 않을 수 있습니다.
"""


def main() -> None:
    setup_logging()
    ensure_data_dirs()

    st.set_page_config(
        page_title="메이플 운빨 리포트",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_style()
    _init_state()

    st.title("메이플 운빨 리포트")
    st.caption("Nexon Open API 기반 큐브/잠재능력 재설정/스타포스 히스토리 분석")

    controls = _render_sidebar()
    _handle_csv_uploads()
    _handle_starforce_quick_tests(controls)
    _handle_api_load(controls)

    cube_df = _apply_filters(st.session_state["cube_df"], "cube")
    potential_df = _apply_filters(st.session_state["potential_df"], "potential")
    starforce_df = _apply_filters(st.session_state["starforce_df"], "starforce")
    cube_like_df = _combine_cube_like(cube_df, potential_df)
    effective_df = add_effective_option_features(
        cube_like_df,
        controls["job_name"],
        controls["selected_stats"],
    )

    cube_tab, starforce_tab, effective_tab, time_tab, raw_tab, debug_tab = st.tabs(
        ["큐브 운 리포트", "스타포스 운 리포트", "유효옵션 분석", "시간 분석", "원본 데이터", "API 디버그"]
    )

    with cube_tab:
        _render_cube_report(effective_df)

    with starforce_tab:
        _render_starforce_report(starforce_df)

    with effective_tab:
        _render_effective_report(effective_df, controls)

    with time_tab:
        _render_time_report(effective_df, starforce_df)

    with raw_tab:
        _render_raw_data(cube_df, potential_df, starforce_df, effective_df)

    with debug_tab:
        _render_api_debug(controls)

    st.divider()
    st.markdown(NOTICE_TEXT)


def _init_state() -> None:
    defaults = {
        "cube_df": pd.DataFrame(),
        "potential_df": pd.DataFrame(),
        "starforce_df": pd.DataFrame(),
        "api_debug": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _render_sidebar() -> dict[str, Any]:
    env_api_key = get_env_api_key()
    login_guide = get_nexon_login_guide()

    with st.sidebar:
        st.header("데이터 불러오기")
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
        else:
            st.info(login_guide.message)

        data_kind = st.radio("데이터 종류", ["큐브", "잠재능력 재설정", "스타포스", "전체"])
        range_type = "starforce" if data_kind == "스타포스" else ("potential" if data_kind == "잠재능력 재설정" else "cube")
        available_start, available_end = get_available_date_range(range_type)
        default_end = available_end
        default_start = max(available_end - timedelta(days=6), available_start)
        st.caption(f"현재 조회 가능 기간: {available_start} ~ {available_end}")
        start_date = st.date_input(
            "시작일",
            value=default_start,
            min_value=available_start,
            max_value=available_end,
        )
        end_date = st.date_input(
            "종료일",
            value=default_end,
            min_value=available_start,
            max_value=available_end,
        )
        auto_clamp = st.checkbox("기간 자동 보정 후 조회", value=True)

        st.divider()
        st.subheader("스타포스 빠른 테스트")
        recent_7_clicked = st.button("최근 7일 스타포스 테스트", width="stretch")
        recent_30_clicked = st.button("최근 30일 스타포스 테스트", width="stretch")

        st.divider()
        st.subheader("유효옵션 기준")
        character_name = st.text_input("캐릭터명", value="", placeholder="선택 입력")
        job_name = st.text_input("직업명", value="", placeholder="예: 메르세데스")
        selected_stats = st.multiselect(
            "주스탯/유효옵션",
            STAT_OPTIONS,
            default=["DEX", "공격력", "올스탯", "보스 데미지", "방어율 무시", "크리티컬 데미지"],
        )
        st.caption("유효옵션 기준은 절대적인 정답이 아니라 사용자가 선택한 기준입니다.")

        show_raw_response = st.checkbox("원본 응답 보기", value=False)
        load_clicked = st.button("데이터 불러오기", type="primary", width="stretch")

        st.divider()
        st.subheader("CSV 업로드 분석")
        st.caption("API 호출 없이 기존 CSV로 리포트를 만들 수 있습니다.")
        st.file_uploader("큐브 CSV", type=["csv"], key="cube_csv_upload")
        st.file_uploader("잠재능력 재설정 CSV", type=["csv"], key="potential_csv_upload")
        st.file_uploader("스타포스 CSV", type=["csv"], key="starforce_csv_upload")

    return {
        "api_key": api_key,
        "auth_method": auth_method,
        "data_kind": data_kind,
        "start_date": start_date,
        "end_date": end_date,
        "auto_clamp": auto_clamp,
        "recent_7_clicked": recent_7_clicked,
        "recent_30_clicked": recent_30_clicked,
        "character_name": character_name,
        "job_name": job_name,
        "selected_stats": selected_stats,
        "show_raw_response": show_raw_response,
        "load_clicked": load_clicked,
    }


def _handle_csv_uploads() -> None:
    uploads = {
        "cube": st.session_state.get("cube_csv_upload"),
        "potential": st.session_state.get("potential_csv_upload"),
        "starforce": st.session_state.get("starforce_csv_upload"),
    }
    labels = {"cube": "큐브", "potential": "잠재능력 재설정", "starforce": "스타포스"}
    for kind, uploaded_file in uploads.items():
        if uploaded_file is None:
            continue
        try:
            st.session_state[f"{kind}_df"] = read_uploaded_csv(uploaded_file, kind)
            st.sidebar.success(f"{labels[kind]} CSV를 불러왔습니다.")
        except Exception as exc:
            st.sidebar.error(f"{labels[kind]} CSV를 읽지 못했습니다: {exc}")


def _handle_starforce_quick_tests(controls: dict[str, Any]) -> None:
    days = None
    if controls["recent_7_clicked"]:
        days = 7
    elif controls["recent_30_clicked"]:
        days = 30

    if days is None:
        return
    if controls["auth_method"] != "API Key 직접 입력":
        st.sidebar.warning("현재는 API Key 직접 입력 방식에서만 스타포스 테스트 호출을 실행할 수 있습니다.")
        return
    if not controls["api_key"].strip():
        st.sidebar.warning("API Key를 입력한 뒤 스타포스 테스트 호출을 실행해 주세요.")
        return

    end_date = date.today()
    start_date = end_date - timedelta(days=days - 1)
    start_date, end_date, messages = clamp_date_range(start_date, end_date, "starforce")
    for message in messages:
        st.sidebar.info(message)

    try:
        client = NexonMapleClient(controls["api_key"])
        _load_history(client, "starforce", start_date.isoformat(), end_date.isoformat())
        st.sidebar.success(f"최근 {days}일 스타포스 테스트 호출을 완료했습니다.")
    except Exception as exc:
        st.sidebar.error(f"스타포스 테스트 호출 실패: {exc}")
        st.session_state["api_debug"]["starforce"] = {
            "error": str(exc),
            "query_start_date": start_date.isoformat(),
            "query_end_date": end_date.isoformat(),
        }


def _handle_api_load(controls: dict[str, Any]) -> None:
    if not controls["load_clicked"]:
        return
    if controls["auth_method"] != "API Key 직접 입력":
        st.sidebar.warning(
            "넥슨 게임 데이터 활용 로그인 방식은 추후 OAuth/동의 기반 연동으로 확장 예정입니다. 현재는 Open API Key 입력 방식으로 이용해주세요."
        )
        return
    if not controls["api_key"].strip():
        st.sidebar.warning("API Key를 입력해 주세요. CSV 업로드 분석은 API Key 없이 가능합니다.")
        return

    try:
        client = NexonMapleClient(controls["api_key"])
        for data_type in _selected_data_types(controls["data_kind"]):
            start_date, end_date = controls["start_date"], controls["end_date"]
            if controls["auto_clamp"]:
                start_date, end_date, messages = clamp_date_range(start_date, end_date, data_type)
                for message in messages:
                    st.sidebar.warning(message) if "조회 가능한 기간" in message else st.sidebar.info(message)
            elif start_date > end_date:
                st.sidebar.warning("시작일은 종료일보다 늦을 수 없습니다.")
                continue

            if start_date > end_date:
                continue

            _load_history(client, data_type, start_date.isoformat(), end_date.isoformat())
    except (NexonAPIError, RuntimeError, ValueError) as exc:
        st.sidebar.error(str(exc))
    except Exception as exc:
        st.sidebar.error(f"예상하지 못한 오류가 발생했습니다: {exc}")


def _load_history(
    client: NexonMapleClient,
    data_type: str,
    start_str: str,
    end_str: str,
) -> None:
    labels = {"cube": "큐브", "potential": "잠재능력 재설정", "starforce": "스타포스"}
    with st.spinner(f"{labels[data_type]} 히스토리를 수집하는 중입니다..."):
        if data_type == "cube":
            loaded = fetch_cube_history_dataframe(client, start_str, end_str)
        elif data_type == "potential":
            loaded = fetch_potential_history_dataframe(client, start_str, end_str)
        else:
            loaded = fetch_starforce_history_dataframe(client, start_str, end_str)

        st.session_state[f"{data_type}_df"] = loaded.dataframe
        st.session_state["api_debug"][data_type] = {
            **client.last_debug_info,
            "data_type": data_type,
            "query_start_date": start_str,
            "query_end_date": end_str,
            "record_count": len(loaded.raw_records),
            "raw_records_preview": loaded.raw_records[:3],
        }

        if data_type == "starforce":
            if loaded.raw_records:
                st.sidebar.success(f"스타포스 records 수: {len(loaded.raw_records):,}건")
            else:
                st.sidebar.info("조회된 스타포스 기록이 없습니다. 기간을 바꾸거나 최근 7일/30일 테스트를 실행해보세요.")

        st.sidebar.success(f"{labels[data_type]} CSV 저장: {loaded.csv_path}")


def _selected_data_types(data_kind: str) -> list[str]:
    return {
        "큐브": ["cube"],
        "잠재능력 재설정": ["potential"],
        "스타포스": ["starforce"],
        "전체": ["cube", "potential", "starforce"],
    }[data_kind]


def _apply_filters(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if df.empty:
        return df
    output = df.copy()
    label = {"cube": "큐브", "potential": "잠재능력 재설정", "starforce": "스타포스"}[kind]
    with st.sidebar.expander(f"{label} 필터", expanded=False):
        filter_columns = ["character_name", "world_name", "item_name"]
        if kind in {"cube", "potential"}:
            filter_columns.append("cube_type")
        if kind == "starforce":
            filter_columns.append("before_starforce")
        if kind in {"cube", "potential"} and "channel" in output.columns and output["channel"].notna().any():
            filter_columns.append("channel")
        for col in filter_columns:
            if col not in output.columns or not output[col].notna().any():
                continue
            options = sorted(output[col].dropna().unique().tolist())
            selected = st.multiselect(_filter_label(col), options, key=f"{kind}_{col}_filter")
            if selected:
                output = output[output[col].isin(selected)]
    return output


def _combine_cube_like(cube_df: pd.DataFrame, potential_df: pd.DataFrame) -> pd.DataFrame:
    frames = [df for df in [cube_df, potential_df] if not df.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _render_cube_report(df: pd.DataFrame) -> None:
    st.subheader("큐브 운 리포트")
    if df.empty:
        st.info("조회된 데이터가 없습니다. API로 불러오거나 CSV를 업로드해 주세요.")
        return

    report = compute_cube_luck_metrics(df)
    effective_summary = summarize_effective_options(df)
    best_hour = get_best_hour(df, success_col="is_grade_up", min_attempts=3)
    cols = st.columns(4)
    cols[0].metric("총 큐브 사용 횟수", f"{report.total_attempts:,}회")
    cols[1].metric("등급업 횟수", f"{report.success_count:,}회")
    cols[2].metric("등급업률", _format_rate(report.success_rate))
    cols[3].metric("운 점수", _format_score(report.luck_score))
    cols = st.columns(4)
    cols[0].metric("유효옵션 출현 횟수", f"{effective_summary['effective_count']:,}회")
    cols[1].metric("유효옵션 출현률", _format_rate(effective_summary["effective_rate"]))
    cols[2].metric("평균 유효 라인 수", f"{effective_summary['avg_effective_lines']:.2f}줄")
    cols[3].metric("최고 운 좋은 1시간", best_hour["hour_label"])
    cols = st.columns(3)
    cols[0].metric("최고 운 좋은 요일", report.best_weekday)
    cols[1].metric("최고 운 좋은 시간대", report.best_hour_band)
    cols[2].metric("최대 연속 실패", f"{report.max_consecutive_failure:,}회")
    st.caption("운 점수와 유효옵션률은 과거 데이터 기반 통계이며 미래 결과를 보장하지 않습니다.")

    col1, col2 = st.columns(2)
    with col1:
        if _has_rows(report.weekday_summary):
            st.plotly_chart(
                plot_weekday_rate(report.weekday_summary, "grade_up_rate"),
                width="stretch",
                key="cube_grade_up_weekday_chart",
            )
        else:
            st.info("큐브 요일별 등급업률 데이터가 부족합니다.")

        if _has_rows(report.hour_band_summary):
            st.plotly_chart(
                plot_hour_band_rate(report.hour_band_summary, "grade_up_rate"),
                width="stretch",
                key="cube_grade_up_hour_band_chart",
            )
        else:
            st.info("큐브 시간대별 등급업률 데이터가 부족합니다.")
    with col2:
        cube_grade_hour_df = summarize_by_hour(df, "is_grade_up")
        if _has_rows(cube_grade_hour_df):
            st.plotly_chart(
                plot_hourly_rate(cube_grade_hour_df, "success_rate", "큐브 등급업률 0~23시"),
                width="stretch",
                key="cube_grade_up_hourly_chart",
            )
        else:
            st.info("큐브 1시간 단위 등급업률 데이터가 부족합니다.")

        cube_daily_df = df.dropna(subset=["event_date"]) if "event_date" in df.columns else pd.DataFrame()
        if _has_rows(cube_daily_df):
            st.plotly_chart(
                plot_daily_attempts(df),
                width="stretch",
                key="cube_daily_attempts_chart",
            )
        else:
            st.info("큐브 일별 시도 데이터가 부족합니다.")
    _render_channel_section(
        report.channel_summary,
        report.best_channel,
        "grade_up_rate",
        chart_key="cube_channel_grade_up_chart",
    )


def _render_starforce_report(df: pd.DataFrame) -> None:
    st.subheader("스타포스 운 리포트")
    st.markdown(STARFORCE_NOTICE)

    if df is None or df.empty:
        st.warning("조회된 스타포스 기록이 없습니다. 기간을 바꾸거나 최근 7일/30일 테스트 호출을 실행해보세요.")
        st.info("API 디버그 탭에서 response keys, count, record_count를 확인할 수 있습니다.")
        return

    summary = summarize_starforce(df)

    weekday_summary = summarize_starforce_by_weekday(df)
    hourly_summary = summarize_starforce_by_hour(df)
    hour_band_summary = summarize_starforce_by_hour_band(df)
    star_count_summary = summarize_starforce_by_star_count(df)

    cols = st.columns(4)
    cols[0].metric("총 강화 시도", f"{summary['total_attempts']:,}회")
    cols[1].metric("성공 횟수", f"{summary['success_count']:,}회")
    cols[2].metric("성공률", _format_rate(summary["success_rate"]))
    cols[3].metric("운 점수", _format_score(summary["luck_score"]))
    cols = st.columns(4)
    cols[0].metric("파괴 횟수", f"{summary['destroy_count']:,}회")
    cols[1].metric("하락 횟수", f"{summary['drop_count']:,}회")
    cols[2].metric("스타캐치 성공 건수", f"{summary['starcatch_success_count']:,}회")
    cols[3].metric("찬스타임 건수", f"{summary['chance_time_count']:,}회")
    cols = st.columns(4)
    cols[0].metric("파괴방지 건수", f"{summary['destroy_defence_count']:,}회")
    cols[1].metric("운 좋은 요일", summary["best_weekday"])
    cols[2].metric("운 좋은 시간", summary["best_hour"])
    cols[3].metric("운 좋은 시간대", summary["best_hour_band"])
    cols = st.columns(2)
    cols[0].metric("최대 연속 실패", f"{summary['max_fail_streak'] or 0:,}회")
    cols[1].metric("원본 레코드 수", f"{len(df):,}건")
    st.caption("운 점수는 과거 데이터 기반 통계이며 미래 결과를 보장하지 않습니다.")

    col1, col2 = st.columns(2)
    with col1:
        if _has_rows(weekday_summary):
            st.plotly_chart(
                plot_weekday_rate(weekday_summary, "success_rate"),
                width="stretch",
                key="starforce_weekday_success_chart",
            )
        else:
            st.info("스타포스 요일별 성공률 데이터가 부족합니다.")

        if _has_rows(hour_band_summary):
            st.plotly_chart(
                plot_hour_band_rate(hour_band_summary, "success_rate"),
                width="stretch",
                key="starforce_hour_band_success_chart",
            )
        else:
            st.info("스타포스 시간대별 성공률 데이터가 부족합니다.")
    with col2:
        if _has_rows(hourly_summary):
            st.plotly_chart(
                plot_hourly_rate(hourly_summary, "success_rate", "스타포스 성공률 0~23시"),
                width="stretch",
                key="starforce_hourly_success_chart",
            )
        else:
            st.info("스타포스 1시간 단위 성공률 데이터가 부족합니다.")

        if _has_rows(star_count_summary):
            st.plotly_chart(
                plot_starforce_stage_rate(star_count_summary),
                width="stretch",
                key="starforce_star_count_success_chart",
            )
        else:
            st.info("스타포스 수치별 성공률 데이터가 부족합니다.")

    st.subheader("스타포스 수치별 성공률 표")
    if star_count_summary.empty:
        st.info("스타포스 수치별 요약 데이터가 없습니다.")
    else:
        st.dataframe(star_count_summary, width="stretch", hide_index=True)

    st.subheader("채널별 분석")
    st.info("스타포스 API 응답 스키마에 채널 필드가 없어 채널별 분석은 제공하지 않습니다.")

    with st.expander("스타포스 원본 데이터 보기", expanded=False):
        st.dataframe(df, width="stretch", hide_index=True)


def _render_effective_report(df: pd.DataFrame, controls: dict[str, Any]) -> None:
    st.subheader("유효옵션 분석")
    st.caption("유효옵션 기준은 절대적인 정답이 아니라 사용자가 선택한 기준입니다. 사냥용/보스용/드랍용 프리셋은 추후 확장 가능합니다.")
    if df.empty:
        st.info("조회된 큐브/잠재능력 재설정 데이터가 없습니다.")
        return

    summary = summarize_effective_options(df)
    st.write("선택한 기준:", ", ".join(controls["selected_stats"]) or "선택 없음")
    st.write("직업명:", controls["job_name"] or "직접 선택 기준 사용")
    examples = df["options_after"].head(5).tolist() if "options_after" in df.columns else []
    with st.expander("옵션 문자열 예시", expanded=False):
        st.write(examples)

    cols = st.columns(4)
    cols[0].metric("전체 사용 횟수", f"{summary['total_cube_uses']:,}회")
    cols[1].metric("유효옵션 출현 횟수", f"{summary['effective_count']:,}회")
    cols[2].metric("유효옵션 출현률", _format_rate(summary["effective_rate"]))
    cols[3].metric("평균 유효 라인 수", f"{summary['avg_effective_lines']:.2f}줄")
    col1, col2 = st.columns(2)
    with col1:
        effective_weekday_df = summarize_effective_by_weekday(df)
        if _has_rows(effective_weekday_df):
            st.plotly_chart(
                plot_weekday_rate(effective_weekday_df, "effective_rate"),
                width="stretch",
                key="cube_effective_weekday_chart",
            )
        else:
            st.info("유효옵션 요일별 데이터가 부족합니다.")

        effective_hour_band_df = summarize_effective_by_hour_band(df)
        if _has_rows(effective_hour_band_df):
            st.plotly_chart(
                plot_hour_band_rate(effective_hour_band_df, "effective_rate"),
                width="stretch",
                key="cube_effective_hour_band_chart",
            )
        else:
            st.info("유효옵션 시간대별 데이터가 부족합니다.")
    with col2:
        effective_hour_df = summarize_effective_by_hour(df)
        if _has_rows(effective_hour_df):
            st.plotly_chart(
                plot_hourly_rate(effective_hour_df, "success_rate", "1시간 단위 유효옵션률"),
                width="stretch",
                key="cube_effective_hourly_chart",
            )
        else:
            st.info("유효옵션 1시간 단위 데이터가 부족합니다.")

        effective_cube_type_df = summarize_effective_by_cube_type(df)
        if _has_rows(effective_cube_type_df):
            st.plotly_chart(
                plot_cube_type_rate(effective_cube_type_df),
                width="stretch",
                key="cube_effective_cube_type_chart",
            )
        else:
            st.info("큐브 타입별 유효옵션 데이터가 부족합니다.")

    st.subheader("아이템별 유효옵션률")
    item_summary = summarize_effective_by_item(df)
    if item_summary.empty:
        st.info("아이템별 유효옵션 데이터가 없습니다.")
    else:
        st.dataframe(item_summary, width="stretch", hide_index=True)


def _render_time_report(cube_df: pd.DataFrame, starforce_df: pd.DataFrame) -> None:
    st.subheader("시간 분석")
    st.caption("시도 횟수가 3회 미만인 시간은 표본 적음으로 표시됩니다.")
    col1, col2 = st.columns(2)
    with col1:
        if cube_df is not None and not cube_df.empty and "is_grade_up" in cube_df.columns:
            cube_hour_df = summarize_by_hour(cube_df, "is_grade_up")
            if cube_hour_df is not None and not cube_hour_df.empty:
                st.plotly_chart(
                    plot_hourly_rate(cube_hour_df, "success_rate", "큐브 등급업률 0~23시"),
                    width="stretch",
                    key="time_report_cube_grade_up_hourly",
                )
            else:
                st.info("큐브 등급업률 시간 분석에 사용할 데이터가 부족합니다.")
        else:
            st.info("큐브 등급업률 시간 분석에 사용할 데이터가 부족합니다.")

        if cube_df is not None and not cube_df.empty and "is_effective_option" in cube_df.columns:
            effective_hour_df = summarize_by_hour(cube_df, "is_effective_option")
            if effective_hour_df is not None and not effective_hour_df.empty:
                st.plotly_chart(
                    plot_hourly_rate(effective_hour_df, "success_rate", "큐브 유효옵션률 0~23시"),
                    width="stretch",
                    key="time_report_cube_effective_hourly",
                )
            else:
                st.info("큐브 유효옵션 시간 분석에 사용할 데이터가 부족합니다.")
        else:
            st.info("큐브 유효옵션 시간 분석에 사용할 데이터가 부족합니다.")
    with col2:
        if starforce_df is not None and not starforce_df.empty and "is_success" in starforce_df.columns:
            starforce_hour_df = summarize_by_hour(starforce_df, "is_success")
            if starforce_hour_df is not None and not starforce_hour_df.empty:
                st.plotly_chart(
                    plot_hourly_rate(starforce_hour_df, "success_rate", "스타포스 성공률 0~23시"),
                    width="stretch",
                    key="time_report_starforce_success_hourly",
                )
            else:
                st.info("스타포스 시간 분석에 사용할 데이터가 부족합니다.")
        else:
            st.info("조회된 스타포스 데이터가 없어 시간 분석을 표시하지 않습니다.")


def _render_raw_data(
    cube_df: pd.DataFrame,
    potential_df: pd.DataFrame,
    starforce_df: pd.DataFrame,
    effective_df: pd.DataFrame,
) -> None:
    st.subheader("원본/가공 데이터")
    st.caption("아래 표는 앱 내부 분석에 쓰는 가공 DataFrame입니다. API Key는 저장하지 않습니다.")
    st.markdown("**큐브 데이터**")
    st.dataframe(cube_df, width="stretch")
    st.markdown("**잠재능력 재설정 데이터**")
    st.dataframe(potential_df, width="stretch")
    st.markdown("**스타포스 데이터**")
    st.dataframe(starforce_df, width="stretch")
    st.markdown("**유효옵션 가공 데이터**")
    st.dataframe(effective_df, width="stretch")


def _render_api_debug(controls: dict[str, Any]) -> None:
    st.subheader("API 디버그")
    st.caption("API Key는 이 화면에 표시하지 않습니다.")
    debug = st.session_state.get("api_debug", {})
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
    st.write("next_cursor:", info.get("next_cursor"))
    st.write("현재 조회 기간:", f"{info.get('query_start_date')} ~ {info.get('query_end_date')}")
    st.write("요청한 날짜 수:", info.get("requested_date_count"))
    st.write("성공한 날짜 수:", info.get("successful_date_count"))
    st.write("총 records 수:", info.get("total_record_count"))
    st.write("count=0 날짜 목록:", info.get("zero_count_dates"))
    st.write("에러 발생 날짜 목록:", info.get("error_dates"))
    st.info(
        "스타포스가 계속 비어 있으면 params 안에 date/count 또는 cursor/count만 들어가는지, response_keys에 starforce_history가 있는지, count와 record_count가 0인지 먼저 확인해 주세요."
    )

    if controls.get("show_raw_response"):
        st.subheader("raw response preview")
        st.json(info.get("raw_response_preview") or info.get("raw_records_preview") or {})
    else:
        st.caption("원본 응답을 보려면 사이드바의 '원본 응답 보기'를 켜주세요.")


def _render_channel_section(
    channel_summary: pd.DataFrame | None,
    best_channel: str | None,
    rate_col: str,
    chart_key: str,
) -> None:
    st.subheader("채널별 운 분석")
    if channel_summary is None:
        st.info(CHANNEL_NOTICE)
        return
    if best_channel:
        st.success(f"최고 운 좋은 채널은 {best_channel}입니다.")
    if _has_rows(channel_summary):
        st.plotly_chart(
            plot_channel_rate(channel_summary, rate_col),
            width="stretch",
            key=chart_key,
        )
    else:
        st.info("채널별 분석에 사용할 데이터가 부족합니다.")


def _filter_label(column: str) -> str:
    return {
        "character_name": "캐릭터",
        "world_name": "월드",
        "item_name": "아이템",
        "cube_type": "큐브/재설정 타입",
        "before_starforce": "강화 시작 수치",
        "channel": "채널",
    }.get(column, column)


def _format_rate(value: float | None) -> str:
    if value is None:
        return "계산 불가"
    return f"{value * 100:.2f}%"


def _format_score(value: float | None) -> str:
    if value is None:
        return "계산 불가"
    return f"{value:.1f}점"


def _has_rows(df: pd.DataFrame | None) -> bool:
    return df is not None and not df.empty


def _inject_style() -> None:
    st.markdown(
        """
<style>
    .stApp {
        background:
            linear-gradient(180deg, rgba(237, 247, 255, 0.72), rgba(255, 255, 255, 1) 34%),
            radial-gradient(circle at top left, rgba(47, 128, 237, 0.10), transparent 28%);
    }
    [data-testid="stMetric"] {
        border: 1px solid rgba(49, 70, 101, 0.14);
        border-radius: 8px;
        padding: 14px 16px;
        background: rgba(255, 255, 255, 0.78);
    }
    [data-testid="stSidebar"] {
        background: #f6f9fd;
    }
</style>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
