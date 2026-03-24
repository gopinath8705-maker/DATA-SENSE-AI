"""
components/dashboard.py
Auto-generated mini dashboard with KPI cards, charts, and filters.
"""

import streamlit as st
import pandas as pd
from typing import List

from utils.visualizations import (
    plot_bar_chart,
    plot_line_chart,
    plot_pie_chart,
    plot_histogram,
    plot_correlation_heatmap
)


def render_dashboard(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> None:
    """Render an auto-generated dashboard with KPIs, filters, and charts."""

    # ── Filters ───────────────────────────────────────────────────────────────
    st.markdown("#### 🔧 Filters")
    filter_col, filter_val_col = st.columns([1, 2])

    filtered_df = df.copy()

    if categorical_cols:
        selected_filter_col = filter_col.selectbox(
            "Filter by Column",
            ["(No filter)"] + categorical_cols,
            key="dash_filter_col"
        )
        if selected_filter_col != "(No filter)":
            unique_vals = df[selected_filter_col].dropna().unique().tolist()
            selected_vals = filter_val_col.multiselect(
                f"Select values for {selected_filter_col}",
                unique_vals,
                default=unique_vals[:min(5, len(unique_vals))],
                key="dash_filter_vals"
            )
            if selected_vals:
                filtered_df = df[df[selected_filter_col].isin(selected_vals)]

    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} rows after filters.")
    st.divider()

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    st.markdown("#### 📌 Key Metrics")
    kpi_cols = numeric_cols[:4]  # Show up to 4 KPIs

    if kpi_cols:
        cols = st.columns(len(kpi_cols))
        for i, col in enumerate(kpi_cols):
            series = filtered_df[col].dropna()
            total = series.sum()
            avg = series.mean()
            delta = avg - df[col].mean()

            with cols[i]:
                st.metric(
                    label=f"**{col}**",
                    value=f"{total:,.1f}",
                    delta=f"Avg: {avg:,.2f}",
                    help=f"Total sum of {col} (avg Δ vs full dataset)"
                )
    else:
        st.info("No numeric columns available for KPIs.")

    st.divider()

    # ── Charts Row 1 ─────────────────────────────────────────────────────────
    st.markdown("#### 📊 Charts")

    if numeric_cols and categorical_cols:
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            x = categorical_cols[0]
            y = numeric_cols[0]
            fig = plot_bar_chart(filtered_df, x, y, title=f"{y} by {x}")
            st.plotly_chart(fig, use_container_width=True, key="dash_bar_chart")

        with chart_col2:
            label = categorical_cols[0]
            val = numeric_cols[0] if numeric_cols else None
            fig = plot_pie_chart(filtered_df, label, val, title=f"Share of {val or label}")
            st.plotly_chart(fig, use_container_width=True, key="dash_pie_chart")

    # ── Charts Row 2 ─────────────────────────────────────────────────────────
    if len(numeric_cols) >= 1:
        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            fig = plot_histogram(filtered_df, numeric_cols[0])
            st.plotly_chart(fig, use_container_width=True, key="dash_histogram")

        with chart_col4:
            if len(numeric_cols) >= 2:
                tmp = filtered_df[numeric_cols[1]].reset_index()
                tmp.columns = ["index", numeric_cols[1]]
                fig = plot_line_chart(tmp, "index", numeric_cols[1], title=f"Trend: {numeric_cols[1]}")
                st.plotly_chart(fig, use_container_width=True, key="dash_line_chart")
            elif categorical_cols and numeric_cols:
                fig = plot_pie_chart(filtered_df, categorical_cols[0], numeric_cols[0])
                st.plotly_chart(fig, use_container_width=True, key="dash_pie_chart2")

    # ── Correlation Heatmap ───────────────────────────────────────────────────
    if len(numeric_cols) >= 3:
        st.markdown("#### 🔗 Correlation Matrix")
        fig = plot_correlation_heatmap(filtered_df, numeric_cols[:10])
        st.plotly_chart(fig, use_container_width=True, key="dash_corr_heatmap")
