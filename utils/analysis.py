"""
utils/analysis.py
Core data analysis functions: stats, column typing, insight detection.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any


def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Return metadata about the dataset: column types, missing values, memory usage."""

    # Build column info table
    col_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.values.astype(str),
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
        "Unique Values": df.nunique().values,
    })

    # Missing values summary
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    missing_df = pd.DataFrame({
        "Column": missing_cols.index,
        "Missing Count": missing_cols.values,
        "Missing %": (missing_cols.values / len(df) * 100).round(2)
    }) if len(missing_cols) > 0 else pd.DataFrame()

    # Memory usage in human-readable format
    mem_bytes = df.memory_usage(deep=True).sum()
    if mem_bytes < 1024:
        mem_str = f"{mem_bytes} bytes"
    elif mem_bytes < 1024 ** 2:
        mem_str = f"{mem_bytes / 1024:.1f} KB"
    else:
        mem_str = f"{mem_bytes / 1024 ** 2:.2f} MB"

    return {
        "column_info": col_info,
        "missing_values": {
            "total_missing": int(missing_counts.sum()),
            "columns_with_missing": int((missing_counts > 0).sum()),
            "missing_df": missing_df,
        },
        "memory_usage": mem_str,
    }


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for all columns (numeric + object)."""
    try:
        stats = df.describe(include="all").T
        stats.index.name = "Column"
        return stats.round(4)
    except Exception:
        return df.describe().T


def identify_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Classify DataFrame columns into numeric and categorical.

    Returns:
        (numeric_cols, categorical_cols)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Categorical: object/string OR low-cardinality numeric (e.g. codes)
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Also include datetime columns for context
    # (they won't be in numeric or categorical)

    return numeric_cols, categorical_cols


def detect_insights(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> List[Dict[str, str]]:
    """
    Rule-based automatic insight detection.
    Returns a list of insight dicts with 'title', 'value', 'type'.
    """
    insights = []

    # ── Numeric insights ──────────────────────────────────────────────────────
    for col in numeric_cols[:6]:  # Limit to first 6 numeric cols
        series = df[col].dropna()
        if series.empty:
            continue

        col_max = series.max()
        col_min = series.min()
        col_mean = series.mean()
        col_std = series.std()

        insights.append({
            "title": f"📈 Highest: {col}",
            "value": f"{col_max:,.2f}",
            "type": "success"
        })
        insights.append({
            "title": f"📉 Lowest: {col}",
            "value": f"{col_min:,.2f}",
            "type": "info"
        })
        insights.append({
            "title": f"📊 Avg: {col}",
            "value": f"{col_mean:,.2f}",
            "type": "info"
        })

        # Detect high variance / potential outliers
        if col_std > col_mean * 0.5 and col_mean != 0:
            insights.append({
                "title": f"⚠️ High Variance: {col}",
                "value": f"Std Dev = {col_std:,.2f}",
                "type": "warning"
            })

    # ── Categorical insights ──────────────────────────────────────────────────
    for col in categorical_cols[:4]:
        series = df[col].dropna()
        if series.empty:
            continue

        top_val = series.value_counts().idxmax()
        top_count = series.value_counts().max()
        pct = top_count / len(series) * 100

        insights.append({
            "title": f"🏆 Top in {col}",
            "value": f"{top_val} ({pct:.1f}%)",
            "type": "success"
        })

    # ── Correlation insight ───────────────────────────────────────────────────
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        max_corr = corr_matrix.unstack().idxmax()
        max_val = corr_matrix.unstack().max()

        if max_val > 0.7:
            insights.append({
                "title": f"🔗 Strong Correlation",
                "value": f"{max_corr[0]} ↔ {max_corr[1]}: {max_val:.2f}",
                "type": "warning"
            })

    # ── Missing data insight ──────────────────────────────────────────────────
    missing_pct = df.isnull().mean().max() * 100
    if missing_pct > 5:
        worst_col = df.isnull().mean().idxmax()
        insights.append({
            "title": "⚠️ Missing Data Detected",
            "value": f"{worst_col}: {missing_pct:.1f}% missing",
            "type": "warning"
        })

    return insights
