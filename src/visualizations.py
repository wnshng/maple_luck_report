from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLOR_SEQUENCE = ["#2F80ED", "#00A676", "#F2994A", "#EB5757", "#9B51E0", "#56CCF2"]


def plot_weekday_rate(df_summary: pd.DataFrame, rate_col: str) -> go.Figure:
    order = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    return _bar_rate(df_summary, "weekday_kr", rate_col, "요일별 성공률", category_order=order)


def plot_hour_rate(df_summary: pd.DataFrame, rate_col: str) -> go.Figure:
    return _bar_rate(df_summary, "hour", rate_col, "시간별 성공률")


def plot_hour_band_rate(df_summary: pd.DataFrame, rate_col: str) -> go.Figure:
    return _bar_rate(
        df_summary,
        "hour_band",
        rate_col,
        "시간대별 성공률",
        category_order=["새벽", "오전", "오후", "저녁"],
    )


def plot_channel_rate(df_summary: pd.DataFrame, rate_col: str) -> go.Figure:
    return _bar_rate(df_summary, "channel", rate_col, "채널별 운 분석")


def plot_daily_attempts(df: pd.DataFrame) -> go.Figure:
    if df.empty or "event_date" not in df.columns:
        return _empty_figure("일별 시도 데이터가 없습니다.")

    daily = df.dropna(subset=["event_date"]).groupby("event_date").size().reset_index(name="attempt_count")
    if daily.empty:
        return _empty_figure("일별 시도 데이터가 없습니다.")

    fig = px.line(
        daily,
        x="event_date",
        y="attempt_count",
        markers=True,
        title="일별 시도 횟수",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    return _style_figure(fig)


def plot_item_rate(df_summary: pd.DataFrame, rate_col: str) -> go.Figure:
    if df_summary is not None and not df_summary.empty:
        df_summary = df_summary.sort_values("attempt_count", ascending=False).head(15)
    return _bar_rate(df_summary, "item_name", rate_col, "아이템별 성공률")


def plot_cube_type_rate(df_summary: pd.DataFrame, rate_col: str = "effective_rate") -> go.Figure:
    return _bar_rate(df_summary, "cube_type", rate_col, "큐브 타입별 유효옵션률")


def plot_starforce_stage_rate(df_summary: pd.DataFrame) -> go.Figure:
    if df_summary is None or df_summary.empty:
        return _empty_figure("스타포스 수치별 성공률 데이터가 없습니다.")
    plot_df = df_summary.copy()
    if "attempts" in plot_df.columns and "attempt_count" not in plot_df.columns:
        plot_df["attempt_count"] = plot_df["attempts"]
    x_col = "before_starforce" if "before_starforce" in plot_df.columns else "target_starforce"
    return _bar_rate(plot_df, x_col, "success_rate", "스타포스 수치별 성공률")


def plot_hourly_rate(
    hour_summary: pd.DataFrame,
    rate_col: str = "success_rate",
    title: str = "1시간 단위 성공률",
) -> go.Figure:
    if hour_summary is None or hour_summary.empty or rate_col not in hour_summary.columns:
        return _empty_figure(f"{title} 데이터가 없습니다.")

    plot_df = hour_summary.copy()
    if "hour_label" not in plot_df.columns and "hour" in plot_df.columns:
        plot_df["hour_label"] = plot_df["hour"].map(lambda hour: f"{int(hour)}시")
    if "attempts" not in plot_df.columns and "attempt_count" in plot_df.columns:
        plot_df["attempts"] = plot_df["attempt_count"]

    plot_df["rate_percent"] = plot_df[rate_col] * 100
    fig = px.bar(
        plot_df,
        x="hour_label",
        y="rate_percent",
        text="attempts",
        hover_data=[col for col in ["hour", "attempts", "success_count", "lift_vs_overall", "sample_note"] if col in plot_df.columns],
        title=title,
        color="rate_percent",
        color_continuous_scale=["#E8F3EE", "#00A676"],
        category_orders={"hour_label": [f"{hour}시" for hour in range(24)]},
    )
    fig.update_traces(texttemplate="n=%{text}", textposition="outside")
    fig.update_layout(yaxis_title="비율 (%)", xaxis_title=None, coloraxis_showscale=False)
    return _style_figure(fig)


def _bar_rate(
    df_summary: pd.DataFrame | None,
    x_col: str,
    rate_col: str,
    title: str,
    category_order: list[str] | None = None,
) -> go.Figure:
    if df_summary is None or df_summary.empty or x_col not in df_summary.columns or rate_col not in df_summary.columns:
        return _empty_figure(f"{title} 데이터가 없습니다.")

    plot_df = df_summary.copy()
    plot_df["rate_percent"] = plot_df[rate_col] * 100
    category_orders = {x_col: category_order} if category_order else None
    fig = px.bar(
        plot_df,
        x=x_col,
        y="rate_percent",
        text="attempt_count",
        hover_data=["attempt_count", "success_count", "luck_score"],
        title=title,
        color="rate_percent",
        color_continuous_scale=["#DCEBFF", "#2F80ED"],
        category_orders=category_orders,
    )
    fig.update_traces(texttemplate="n=%{text}", textposition="outside")
    fig.update_layout(yaxis_title="성공률 (%)", xaxis_title=None, coloraxis_showscale=False)
    return _style_figure(fig)


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def _style_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Pretendard, Apple SD Gothic Neo, sans-serif", size=13),
    )
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)")
    return fig
