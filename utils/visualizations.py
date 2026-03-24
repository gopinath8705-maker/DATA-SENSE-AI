"""
utils/visualizations.py
All Plotly chart generation functions for DataSense AI.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from typing import Optional, List

# Consistent color palette
COLORS = px.colors.qualitative.Plotly
TEMPLATE = "plotly_white"


def plot_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: Optional[str] = None,
    top_n: int = 20
) -> go.Figure:
    """
    Bar chart: group x_col, aggregate y_col by mean.
    Limits to top_n categories for readability.
    """
    title = title or f"{y_col} by {x_col}"

    # Aggregate and sort
    agg = df.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(top_n).reset_index()

    fig = px.bar(
        agg,
        x=x_col,
        y=y_col,
        title=title,
        color=y_col,
        color_continuous_scale="Blues",
        template=TEMPLATE,
        text_auto=".2s",
    )
    fig.update_layout(
        xaxis_tickangle=-30,
        coloraxis_showscale=False,
        margin=dict(t=50, b=60),
    )
    return fig


def plot_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: Optional[str] = None
) -> go.Figure:
    """Line chart with markers."""
    title = title or f"{y_col} over {x_col}"

    # Sort by x for clean line
    plot_df = df[[x_col, y_col]].dropna().sort_values(x_col)

    fig = px.line(
        plot_df,
        x=x_col,
        y=y_col,
        title=title,
        markers=True,
        template=TEMPLATE,
        color_discrete_sequence=COLORS,
    )
    fig.update_traces(line_width=2.5, marker_size=6)
    fig.update_layout(margin=dict(t=50, b=40))
    return fig


def plot_pie_chart(
    df: pd.DataFrame,
    label_col: str,
    value_col: Optional[str] = None,
    top_n: int = 10,
    title: Optional[str] = None
) -> go.Figure:
    """
    Pie chart. If value_col provided, sums values per label.
    Otherwise counts occurrences.
    """
    if value_col:
        agg = df.groupby(label_col)[value_col].sum().sort_values(ascending=False).head(top_n)
        title = title or f"{value_col} Distribution by {label_col}"
    else:
        agg = df[label_col].value_counts().head(top_n)
        title = title or f"Distribution of {label_col}"

    fig = px.pie(
        values=agg.values,
        names=agg.index,
        title=title,
        template=TEMPLATE,
        color_discrete_sequence=COLORS,
        hole=0.35,  # Donut style
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(t=50, b=20))
    return fig


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: Optional[str] = None
) -> go.Figure:
    """Scatter plot with optional color dimension."""
    title = title or f"{x_col} vs {y_col}"

    plot_df = df[[x_col, y_col] + ([color_col] if color_col else [])].dropna()

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        template=TEMPLATE,
        trendline="ols",
        opacity=0.75,
        color_discrete_sequence=COLORS,
    )
    fig.update_traces(marker_size=7)
    fig.update_layout(margin=dict(t=50, b=40))
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: List[str],
    title: str = "Correlation Heatmap"
) -> go.Figure:
    """Annotated correlation heatmap for numeric columns."""
    corr = df[numeric_cols].corr().round(2)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr.values,
            texttemplate="%{text}",
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title=title,
        template=TEMPLATE,
        margin=dict(t=60, b=60, l=60, r=20),
        height=max(400, len(numeric_cols) * 50),
    )
    return fig


def plot_histogram(
    df: pd.DataFrame,
    col: str,
    title: Optional[str] = None
) -> go.Figure:
    """Histogram with KDE overlay."""
    title = title or f"Distribution of {col}"
    fig = px.histogram(
        df,
        x=col,
        nbins=30,
        title=title,
        marginal="box",
        template=TEMPLATE,
        color_discrete_sequence=[COLORS[0]],
    )
    fig.update_layout(margin=dict(t=50, b=40))
    return fig
