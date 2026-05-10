from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLOR_SEQUENCE = ["#F97316", "#FB923C", "#F59E0B", "#F43F5E", "#C084FC", "#22C55E"]
CURRENT_THEME_MODE = "light"

LIGHT_PLOTLY_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(255,255,255,0.92)",
    "font_color": "#1F2937",
    "grid_color": "rgba(128,128,128,0.18)",
    "axis_line": "#CBD5E1",
    "legend_font": "#475569",
    "hover_bg": "#FFFFFF",
    "hover_font": "#111827",
    "hover_border": "#CBD5E1",
}

DARK_PLOTLY_THEME = {
    "paper_bgcolor": "#0F172A",
    "plot_bgcolor": "#1E293B",
    "font_color": "#F8FAFC",
    "grid_color": "#334155",
    "axis_line": "#475569",
    "legend_font": "#CBD5E1",
    "hover_bg": "#0F172A",
    "hover_font": "#F8FAFC",
    "hover_border": "#334155",
}


def set_plotly_theme_mode(theme_mode: str) -> None:
    global CURRENT_THEME_MODE
    CURRENT_THEME_MODE = "dark" if str(theme_mode).lower() == "dark" else "light"


def apply_plotly_theme(fig: go.Figure, theme_mode: str | None = None) -> go.Figure:
    mode = (theme_mode or CURRENT_THEME_MODE).lower()
    palette = DARK_PLOTLY_THEME if mode == "dark" else LIGHT_PLOTLY_THEME
    fig.update_layout(
        paper_bgcolor=palette["paper_bgcolor"],
        plot_bgcolor=palette["plot_bgcolor"],
        font=dict(color=palette["font_color"], family="Pretendard, Apple SD Gothic Neo, sans-serif", size=13),
        legend=dict(font=dict(color=palette["legend_font"])),
        hoverlabel=dict(
            bgcolor=palette["hover_bg"],
            font=dict(color=palette["hover_font"]),
            bordercolor=palette["hover_border"],
        ),
    )
    fig.update_xaxes(gridcolor=palette["grid_color"], linecolor=palette["axis_line"], zerolinecolor=palette["grid_color"])
    fig.update_yaxes(gridcolor=palette["grid_color"], linecolor=palette["axis_line"], zerolinecolor=palette["grid_color"])
    return fig


def shorten_label(label: object, max_len: int = 12) -> str:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return "-"
    text = str(label)
    replacements = {
        "잠재능력 재설정": "잠재 재설정",
        "화이트 에디셔널 큐브": "화이트 에디셔널",
    }
    text = replacements.get(text, text)
    return text if len(text) <= max_len else f"{text[: max_len - 3]}..."


def add_sample_size_opacity(plot_df: pd.DataFrame, attempt_col: str, min_attempts: int) -> list[float]:
    if attempt_col not in plot_df.columns:
        return [1.0] * len(plot_df)
    return plot_df[attempt_col].map(lambda value: 0.35 if int(value) < min_attempts else 1.0).tolist()


def format_rate_axis(fig: go.Figure, axis: str = "y") -> go.Figure:
    if axis == "x":
        fig.update_xaxes(title="확률 (%)")
    else:
        fig.update_yaxes(title="확률 (%)")
    return fig


def apply_chart_layout(fig: go.Figure, title: str, category_count: int, orientation: str = "auto") -> go.Figure:
    height = max(360, 240 + min(category_count, 30) * 18)
    fig.update_layout(title=title, height=height, margin=dict(l=20, r=20, t=60, b=80))
    if orientation != "h":
        fig.update_xaxes(tickangle=-45)
    return fig


def plot_weekday_rate(
    df_summary: pd.DataFrame,
    rate_col: str,
    title: str = "요일별 성공률",
    min_attempts: int = 10,
    show_low_sample: bool = True,
) -> go.Figure:
    order = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    return _bar_rate(df_summary, "weekday_kr", rate_col, title, category_order=order, min_attempts=min_attempts, show_low_sample=show_low_sample)


def plot_hour_rate(
    df_summary: pd.DataFrame,
    rate_col: str,
    title: str = "시간별 성공률",
    min_attempts: int = 10,
    show_low_sample: bool = True,
) -> go.Figure:
    return _bar_rate(df_summary, "hour", rate_col, title, min_attempts=min_attempts, show_low_sample=show_low_sample)


def plot_hour_band_rate(
    df_summary: pd.DataFrame,
    rate_col: str,
    title: str = "시간대별 성공률",
    min_attempts: int = 10,
    show_low_sample: bool = True,
) -> go.Figure:
    return _bar_rate(
        df_summary,
        "hour_band",
        rate_col,
        title,
        category_order=["새벽", "오전", "오후", "저녁"],
        min_attempts=min_attempts,
        show_low_sample=show_low_sample,
    )


def plot_channel_rate(df_summary: pd.DataFrame, rate_col: str, title: str = "채널별 운 분석") -> go.Figure:
    return _bar_rate(df_summary, "channel", rate_col, title)


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


def plot_item_rate(
    df_summary: pd.DataFrame,
    rate_col: str,
    title: str = "아이템별 성공률",
    min_attempts: int = 10,
    show_low_sample: bool = True,
) -> go.Figure:
    if df_summary is not None and not df_summary.empty:
        attempt_col = "attempt_count" if "attempt_count" in df_summary.columns else "attempts"
        df_summary = df_summary.sort_values(attempt_col, ascending=False).head(15)
    return _bar_rate(df_summary, "item_name", rate_col, title, min_attempts=min_attempts, show_low_sample=show_low_sample)


def plot_cube_type_rate(
    df_summary: pd.DataFrame,
    rate_col: str = "effective_rate",
    title: str = "큐브 타입별 유효옵션률",
    min_attempts: int = 10,
    show_low_sample: bool = True,
) -> go.Figure:
    if df_summary is None or df_summary.empty or "cube_type" not in df_summary.columns or rate_col not in df_summary.columns:
        return _empty_figure(f"{title} 데이터가 없습니다.")

    plot_df = df_summary.copy()
    attempt_col = "attempt_count" if "attempt_count" in plot_df.columns else "attempts"
    count_col = "success_count"
    if rate_col == "major_option_rate":
        count_col = "major_option_count"
    elif rate_col == "effective_option_rate":
        count_col = "effective_option_count"
    elif rate_col == "grade_up_rate":
        count_col = "grade_up_count"

    if attempt_col in plot_df.columns:
        plot_df["sample_note"] = plot_df[attempt_col].map(lambda value: "표본 부족" if int(value) < min_attempts else "기준 충족")
        if not show_low_sample:
            plot_df = plot_df[plot_df[attempt_col] >= min_attempts].copy()
            if plot_df.empty:
                return _empty_figure(f"{title} 데이터가 없습니다.")

    for gap_col, label_col in [("overall_gap_p", "overall_gap_label"), ("reference_gap_p", "reference_gap_label")]:
        if gap_col in plot_df.columns:
            plot_df[label_col] = plot_df[gap_col].map(lambda value: f"{value * 100:+.1f}%p" if pd.notna(value) else "기준 없음")

    plot_df["rate_percent"] = plot_df[rate_col] * 100
    plot_df["full_label"] = plot_df["cube_type"].astype(str)
    plot_df["short_label"] = plot_df["full_label"].map(shorten_cube_type_label)
    plot_df["adjusted_rate"] = plot_df.get("adjusted_rate", plot_df[rate_col])
    plot_df = plot_df.sort_values(["adjusted_rate", "rate_percent", attempt_col], ascending=[False, False, False]).copy()
    plot_df["rate_label"] = plot_df["rate_percent"].map(lambda value: f"{value:.1f}%")
    opacities = add_sample_size_opacity(plot_df, attempt_col, min_attempts)
    hover_cols = [attempt_col, count_col, "confidence", "overall_gap_label", "reference_gap_label", "sample_note", "full_label"]
    fig = go.Figure(
        data=[
            go.Bar(
                y=plot_df["short_label"],
                x=plot_df["rate_percent"],
                orientation="h",
                text=plot_df["rate_label"],
                textposition="outside",
                marker=dict(color="#F97316", opacity=opacities),
                customdata=np.column_stack(
                    [
                        plot_df[col].to_numpy() if col in plot_df.columns else np.array([None] * len(plot_df))
                        for col in hover_cols
                    ]
                ),
                hovertemplate=(
                    "큐브 타입: %{customdata[6]}<br>실제률: %{x:.1f}%<br>"
                    "시도 수: %{customdata[0]}<br>발생 수: %{customdata[1]}<br>"
                    "신뢰도: %{customdata[2]}<br>전체 평균 대비: %{customdata[3]}<br>"
                    "기준 확률 대비: %{customdata[4]}<br>%{customdata[5]}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(title=title, xaxis_title="확률 (%)", yaxis_title=None)
    fig.update_yaxes(categoryorder="array", categoryarray=plot_df["short_label"].tolist()[::-1])
    return apply_chart_layout(_style_figure(fig), title, len(plot_df), orientation="h")


def shorten_cube_type_label(cube_type: object) -> str:
    return shorten_label(cube_type, 14)


def plot_starforce_stage_rate(
    df_summary: pd.DataFrame,
    title: str = "스타포스 수치별 성공률",
    min_attempts: int = 10,
    show_low_sample: bool = True,
) -> go.Figure:
    if df_summary is None or df_summary.empty:
        return _empty_figure("스타포스 수치별 성공률 데이터가 없습니다.")
    plot_df = df_summary.copy()
    if "attempts" in plot_df.columns and "attempt_count" not in plot_df.columns:
        plot_df["attempt_count"] = plot_df["attempts"]
    x_col = "before_starforce" if "before_starforce" in plot_df.columns else "target_starforce"
    return _bar_rate(plot_df, x_col, "success_rate", title, min_attempts=min_attempts, show_low_sample=show_low_sample)


def plot_hourly_rate(
    hour_summary: pd.DataFrame,
    rate_col: str = "success_rate",
    title: str = "1시간 단위 성공률",
    min_attempts: int = 10,
    show_low_sample: bool = True,
) -> go.Figure:
    if hour_summary is None or hour_summary.empty or rate_col not in hour_summary.columns:
        return _empty_figure(f"{title} 데이터가 없습니다.")

    plot_df = hour_summary.copy()
    if "hour_label" not in plot_df.columns and "hour" in plot_df.columns:
        plot_df["hour_label"] = plot_df["hour"].map(lambda hour: f"{int(hour)}시")
    if "attempts" not in plot_df.columns and "attempt_count" in plot_df.columns:
        plot_df["attempts"] = plot_df["attempt_count"]
    if "attempts" in plot_df.columns:
        plot_df["sample_note"] = plot_df["attempts"].map(lambda value: "표본 부족" if int(value) < min_attempts else "기준 충족")
        if not show_low_sample:
            plot_df = plot_df[plot_df["attempts"] >= min_attempts].copy()
            if plot_df.empty:
                return _empty_figure(f"{title} 데이터가 없습니다.")
    for gap_col, label_col in [("overall_gap_p", "overall_gap_label"), ("reference_gap_p", "reference_gap_label")]:
        if gap_col in plot_df.columns:
            plot_df[label_col] = plot_df[gap_col].map(lambda value: f"{value * 100:+.1f}%p" if pd.notna(value) else "기준 없음")

    plot_df["rate_percent"] = plot_df[rate_col] * 100
    plot_df["rate_label"] = plot_df["rate_percent"].map(lambda value: f"{value:.1f}%" if pd.notna(value) else "")
    opacities = (
        plot_df["attempts"].map(lambda value: 0.35 if int(value) < min_attempts else 1.0).tolist()
        if "attempts" in plot_df.columns
        else 1.0
    )
    hover_cols = ["hour", "attempts", "success_count", "lift_vs_overall", "overall_gap_label", "reference_gap_label", "confidence", "sample_note"]
    customdata = np.column_stack(
        [
            plot_df[col].to_numpy() if col in plot_df.columns else np.array([None] * len(plot_df))
            for col in hover_cols
        ]
    )
    fig = go.Figure(
        data=[
            go.Bar(
                x=plot_df["hour_label"],
                y=plot_df["rate_percent"],
                text=plot_df["rate_label"],
                textposition="outside",
                marker=dict(color="#FB923C", opacity=opacities),
                customdata=customdata,
                hovertemplate=(
                    "시간: %{x}<br>실제률: %{y:.1f}%<br>"
                    "시도 수: %{customdata[1]}<br>성공/발생 수: %{customdata[2]}<br>"
                    "전체 평균 대비: %{customdata[4]}<br>기준 확률 대비: %{customdata[5]}<br>"
                    "신뢰도: %{customdata[6]}<br>%{customdata[7]}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(yaxis_title="확률 (%)", xaxis_title=None, title=title)
    fig.update_xaxes(categoryorder="array", categoryarray=[f"{hour}시" for hour in range(24)])
    return _style_figure(fig)


def plot_rate_heatmap(
    df_summary: pd.DataFrame,
    x_col: str,
    y_col: str,
    rate_col: str,
    title: str,
    x_order: list[str] | None = None,
    y_order: list[str] | None = None,
) -> go.Figure:
    if df_summary is None or df_summary.empty or any(col not in df_summary.columns for col in [x_col, y_col, rate_col]):
        return _empty_figure(f"{title} 데이터가 없습니다.")

    plot_df = df_summary.copy()
    attempt_col = "attempt_count" if "attempt_count" in plot_df.columns else "attempts"
    success_col = "success_count" if "success_count" in plot_df.columns else "success_count"
    confidence_col = "confidence" if "confidence" in plot_df.columns else None
    x_values = x_order or plot_df[x_col].dropna().astype(str).unique().tolist()
    y_values = y_order or plot_df[y_col].dropna().astype(str).unique().tolist()
    plot_df = plot_df.assign(_x_value=plot_df[x_col].astype(str), _y_value=plot_df[y_col].astype(str))
    rate_pivot = (
        plot_df.pivot(index="_y_value", columns="_x_value", values=rate_col)
        .reindex(index=y_values, columns=x_values)
        * 100
    )
    attempt_pivot = (
        plot_df.pivot(index="_y_value", columns="_x_value", values=attempt_col)
        .reindex(index=y_values, columns=x_values)
    )
    success_pivot = (
        plot_df.pivot(index="_y_value", columns="_x_value", values=success_col)
        .reindex(index=y_values, columns=x_values)
    )
    confidence_pivot = (
        plot_df.pivot(index="_y_value", columns="_x_value", values=confidence_col)
        .reindex(index=y_values, columns=x_values)
        if confidence_col
        else None
    )

    text = rate_pivot.map(lambda value: f"{value:.1f}%" if pd.notna(value) else "")
    if confidence_pivot is not None:
        customdata = np.dstack(
            [
                attempt_pivot.fillna(0).to_numpy(),
                success_pivot.fillna(0).to_numpy(),
                confidence_pivot.fillna("").to_numpy(),
            ]
        )
        hovertemplate = (
            f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>확률: %{{z:.2f}}%<br>"
            "시도: %{customdata[0]:.0f}<br>성공: %{customdata[1]:.0f}<br>"
            "신뢰도: %{customdata[2]}<extra></extra>"
        )
    else:
        customdata = np.dstack(
            [
                attempt_pivot.fillna(0).to_numpy(),
                success_pivot.fillna(0).to_numpy(),
            ]
        )
        hovertemplate = (
            f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>확률: %{{z:.2f}}%<br>"
            "시도: %{customdata[0]:.0f}<br>성공: %{customdata[1]:.0f}<extra></extra>"
        )
    fig = go.Figure(
        data=go.Heatmap(
            x=x_values,
            y=y_values,
            z=rate_pivot.values,
            text=text.values,
            texttemplate="%{text}",
            colorscale=[(0.0, "#FFF1E6"), (0.55, "#FDBA74"), (1.0, "#F97316")],
            colorbar_title="확률 (%)",
            customdata=customdata,
            hovertemplate=hovertemplate,
        )
    )
    fig.update_layout(title=title, xaxis_title=None, yaxis_title=None)
    return _style_figure(fig)


def plot_facet_rate_map(
    df_summary: pd.DataFrame,
    facet_col: str,
    x_col: str,
    y_col: str,
    rate_col: str,
    title: str,
    x_order: list[str] | None = None,
    y_order: list[str] | None = None,
    top_n_facets: int = 6,
) -> go.Figure:
    required_cols = [facet_col, x_col, y_col, rate_col]
    if df_summary is None or df_summary.empty or any(col not in df_summary.columns for col in required_cols):
        return _empty_figure(f"{title} 데이터가 없습니다.")

    plot_df = df_summary.copy()
    attempt_col = "attempt_count" if "attempt_count" in plot_df.columns else "attempts"
    facet_attempts = plot_df.groupby(facet_col)[attempt_col].sum().sort_values(ascending=False)
    facet_values = facet_attempts.head(top_n_facets).index.tolist()
    plot_df = plot_df[plot_df[facet_col].isin(facet_values)].copy()
    plot_df["rate_percent"] = plot_df[rate_col] * 100
    plot_df["rate_label"] = plot_df["rate_percent"].map(lambda value: f"{value:.1f}%" if pd.notna(value) else "")

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        facet_col=facet_col,
        color="rate_percent",
        text="rate_label",
        hover_data=[col for col in [attempt_col, "success_count", rate_col] if col in plot_df.columns],
        title=title,
        color_continuous_scale=["#EAF7F1", "#00A676"],
        category_orders={k: v for k, v in [(x_col, x_order), (y_col, y_order)] if v is not None},
    )
    fig.update_traces(marker=dict(size=18), textposition="top center")
    fig.update_layout(coloraxis_colorbar_title="확률 (%)")
    return _style_figure(fig)


def plot_multi_rate_bar(
    df_summary: pd.DataFrame,
    x_col: str,
    rate_map: dict[str, str],
    title: str,
) -> go.Figure:
    if df_summary is None or df_summary.empty or x_col not in df_summary.columns:
        return _empty_figure(f"{title} 데이터가 없습니다.")

    rows: list[dict[str, object]] = []
    attempt_col = "attempt_count" if "attempt_count" in df_summary.columns else "attempts"
    for metric_label, rate_col in rate_map.items():
        if rate_col not in df_summary.columns:
            continue
        for _, row in df_summary.iterrows():
            rows.append(
                {
                    x_col: row[x_col],
                    "metric_label": metric_label,
                    "rate_percent": (row[rate_col] * 100) if pd.notna(row[rate_col]) else np.nan,
                    "attempt_count": row.get(attempt_col),
                    "confidence": row.get("confidence"),
                }
            )
    if not rows:
        return _empty_figure(f"{title} 데이터가 없습니다.")

    plot_df = pd.DataFrame(rows)
    fig = px.bar(
        plot_df,
        x=x_col,
        y="rate_percent",
        color="metric_label",
        barmode="group",
        title=title,
        hover_data=[col for col in ["attempt_count", "confidence"] if col in plot_df.columns],
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_layout(yaxis_title="확률 (%)", xaxis_title=None, legend_title=None)
    return _style_figure(fig)


def plot_reference_gap_bar(
    df_summary: pd.DataFrame,
    label_col: str,
    gap_col: str,
    title: str,
) -> go.Figure:
    if df_summary is None or df_summary.empty or label_col not in df_summary.columns or gap_col not in df_summary.columns:
        return _empty_figure(f"{title} 데이터가 없습니다.")

    plot_df = df_summary.copy()
    plot_df["gap_percent"] = plot_df[gap_col] * 100
    plot_df["full_label"] = plot_df[label_col].astype(str)
    plot_df["short_label"] = plot_df["full_label"].map(lambda value: shorten_label(value, 16))
    hover_data = {
        "full_label": True,
        "confidence": "confidence" in plot_df.columns,
        "short_label": False,
    }
    if "attempt_count" in plot_df.columns:
        hover_data["attempt_count"] = True
    elif "attempts" in plot_df.columns:
        hover_data["attempts"] = True
    orientation = "h" if len(plot_df) > 8 else "v"
    if orientation == "h":
        fig = px.bar(
            plot_df,
            y="short_label",
            x="gap_percent",
            orientation="h",
            color="gap_percent",
            title=title,
            hover_data=hover_data,
            color_continuous_scale=["#EB5757", "#F2F2F2", "#00A676"],
        )
        fig.update_layout(xaxis_title="기준 대비 차이 (%p)", yaxis_title=None, coloraxis_showscale=False)
    else:
        fig = px.bar(
            plot_df,
            x="short_label",
            y="gap_percent",
            color="gap_percent",
            title=title,
            hover_data=hover_data,
            color_continuous_scale=["#EB5757", "#F2F2F2", "#00A676"],
        )
        fig.update_layout(yaxis_title="기준 대비 차이 (%p)", xaxis_title=None, coloraxis_showscale=False)
    return apply_chart_layout(_style_figure(fig), title, len(plot_df), orientation=orientation)


def plot_date_rate_trend(
    df_summary: pd.DataFrame,
    x_col: str,
    rate_map: dict[str, str],
    title: str,
) -> go.Figure:
    if df_summary is None or df_summary.empty or x_col not in df_summary.columns:
        return _empty_figure(f"{title} 데이터가 없습니다.")

    rows: list[dict[str, object]] = []
    attempt_col = "attempts" if "attempts" in df_summary.columns else "attempt_count"
    for label, rate_col in rate_map.items():
        if rate_col not in df_summary.columns:
            continue
        for _, row in df_summary.iterrows():
            rows.append(
                {
                    x_col: row[x_col],
                    "metric_label": label,
                    "rate_percent": row[rate_col] * 100 if pd.notna(row[rate_col]) else np.nan,
                    "attempts": row.get(attempt_col),
                    "confidence": row.get("confidence"),
                }
            )
    if not rows:
        return _empty_figure(f"{title} 데이터가 없습니다.")

    plot_df = pd.DataFrame(rows)
    fig = px.line(
        plot_df,
        x=x_col,
        y="rate_percent",
        color="metric_label",
        markers=True,
        title=title,
        hover_data=[col for col in ["attempts", "confidence"] if col in plot_df.columns],
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_layout(yaxis_title="확률 (%)", xaxis_title=None, legend_title=None)
    return _style_figure(fig)


def plot_day_of_month_rate(
    df_summary: pd.DataFrame,
    rate_col: str,
    title: str,
    min_attempts: int = 10,
    show_low_sample: bool = True,
) -> go.Figure:
    x_col = "day_of_month_label" if df_summary is not None and "day_of_month_label" in df_summary.columns else "day_of_month"
    category_order = [f"{day}일" for day in range(1, 32)] if x_col == "day_of_month_label" else list(range(1, 32))
    return _bar_rate(df_summary, x_col, rate_col, title, category_order=category_order, min_attempts=min_attempts, show_low_sample=show_low_sample)


def _bar_rate(
    df_summary: pd.DataFrame | None,
    x_col: str,
    rate_col: str,
    title: str,
    category_order: list[str] | None = None,
    min_attempts: int = 10,
    show_low_sample: bool = True,
) -> go.Figure:
    if df_summary is None or df_summary.empty or x_col not in df_summary.columns or rate_col not in df_summary.columns:
        return _empty_figure(f"{title} 데이터가 없습니다.")

    plot_df = df_summary.copy()
    attempt_col = "attempt_count" if "attempt_count" in plot_df.columns else "attempts"
    if attempt_col in plot_df.columns:
        plot_df["sample_note"] = plot_df[attempt_col].map(lambda value: "표본 부족" if int(value) < min_attempts else "기준 충족")
        if not show_low_sample:
            plot_df = plot_df[plot_df[attempt_col] >= min_attempts].copy()
            if plot_df.empty:
                return _empty_figure(f"{title} 데이터가 없습니다.")
    for gap_col, label_col in [("overall_gap_p", "overall_gap_label"), ("reference_gap_p", "reference_gap_label")]:
        if gap_col in plot_df.columns:
            plot_df[label_col] = plot_df[gap_col].map(lambda value: f"{value * 100:+.1f}%p" if pd.notna(value) else "기준 없음")
    plot_df["rate_percent"] = plot_df[rate_col] * 100
    plot_df["rate_label"] = plot_df["rate_percent"].map(lambda value: f"{value:.1f}%" if pd.notna(value) else "")
    plot_df["full_label"] = plot_df[x_col].astype(str)
    plot_df["short_label"] = plot_df["full_label"].map(lambda value: shorten_label(value, 14))
    category_orders = {"short_label": [shorten_label(value, 14) for value in category_order]} if category_order else None
    opacities = add_sample_size_opacity(plot_df, attempt_col, min_attempts)
    hover_cols = [attempt_col, "success_count", "overall_gap_label", "reference_gap_label", "confidence", "sample_note"]
    customdata = np.column_stack(
        [
            plot_df[col].to_numpy() if col in plot_df.columns else np.array([None] * len(plot_df))
            for col in hover_cols
        ]
    )
    orientation = "v"
    if x_col not in {"day_of_month", "day_of_month_label", "hour", "hour_label", "weekday_kr", "hour_band"} and len(plot_df) > 8:
        orientation = "h"
    fig = go.Figure(
        data=[
            go.Bar(
                x=plot_df["rate_percent"] if orientation == "h" else plot_df["short_label"],
                y=plot_df["short_label"] if orientation == "h" else plot_df["rate_percent"],
                text=plot_df["rate_label"],
                textposition="outside",
                marker=dict(color="#FB923C", opacity=opacities),
                customdata=customdata,
                hovertemplate=(
                    "조건: %{customdata[6]}<br>실제률: "
                    + ("%{x:.1f}%<br>" if orientation == "h" else "%{y:.1f}%<br>")
                    +
                    "시도 수: %{customdata[0]}<br>성공/발생 수: %{customdata[1]}<br>"
                    "전체 평균 대비: %{customdata[2]}<br>기준 확률 대비: %{customdata[3]}<br>"
                    "신뢰도: %{customdata[4]}<br>%{customdata[5]}<extra></extra>"
                ),
                orientation=orientation,
            )
        ]
    )
    fig.data[0].customdata = np.column_stack(
        [
            plot_df[col].to_numpy() if col in plot_df.columns else np.array([None] * len(plot_df))
            for col in hover_cols
        ]
        + [plot_df["full_label"].to_numpy()]
    )
    fig.update_layout(title=title, xaxis_title=None, yaxis_title=None)
    if category_orders:
        if orientation == "h":
            fig.update_yaxes(categoryorder="array", categoryarray=category_orders["short_label"])
        else:
            fig.update_xaxes(categoryorder="array", categoryarray=category_orders["short_label"])
    fig = format_rate_axis(fig, axis="x" if orientation == "h" else "y")
    return apply_chart_layout(_style_figure(fig), title, len(plot_df), orientation=orientation)


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
    )
    return apply_plotly_theme(fig)
