"""
utils/recommendations.py
AI recommendation engine + natural language chart generator.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go


# ─── Natural Language Chart Generator ────────────────────────────────────────

def parse_chart_request(
    prompt: str,
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    api_key: str = ""
) -> Optional[go.Figure]:
    """
    Convert a natural language chart request to a Plotly figure.
    Uses GPT to understand intent; falls back to rule matching.
    """
    if api_key:
        return _parse_chart_with_ai(prompt, df, numeric_cols, categorical_cols, api_key)
    return _parse_chart_rule_based(prompt, df, numeric_cols, categorical_cols)


def _parse_chart_with_ai(
    prompt: str,
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    api_key: str
) -> Optional[go.Figure]:
    """Use GPT to extract chart intent, then render with Plotly."""
    try:
        import openai, json
        client = openai.OpenAI(api_key=api_key)

        col_list = df.columns.tolist()
        system = (
            "You parse chart requests into JSON. Return ONLY valid JSON, no explanation. "
            "Schema: {\"chart_type\": \"bar|line|pie|scatter|histogram\", "
            "\"x\": \"column_name\", \"y\": \"column_name_or_null\", "
            "\"color\": \"column_name_or_null\", \"title\": \"string\"}"
        )
        user = (
            f"Columns available: {col_list}\n"
            f"Numeric columns: {numeric_cols}\n"
            f"Categorical columns: {categorical_cols}\n"
            f"Chart request: {prompt}"
        )

        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=200,
            temperature=0
        )
        raw = resp.choices[0].message.content.strip()
        # strip possible markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        spec = json.loads(raw)

        return _build_figure_from_spec(spec, df, numeric_cols, categorical_cols)

    except Exception:
        return _parse_chart_rule_based(prompt, df, numeric_cols, categorical_cols)


def _parse_chart_rule_based(
    prompt: str,
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> Optional[go.Figure]:
    """Regex/keyword rule-based chart intent parsing."""
    import re
    p = prompt.lower()

    # Detect chart type
    if any(w in p for w in ["bar", "column", "compare"]):
        chart_type = "bar"
    elif any(w in p for w in ["line", "trend", "over time", "time series"]):
        chart_type = "line"
    elif any(w in p for w in ["pie", "donut", "share", "portion", "distribution"]):
        chart_type = "pie"
    elif any(w in p for w in ["scatter", "correlation", "vs", "versus", "relationship"]):
        chart_type = "scatter"
    elif any(w in p for w in ["histogram", "frequency", "dist"]):
        chart_type = "histogram"
    else:
        chart_type = "bar"  # default

    # Detect mentioned columns
    mentioned_num = [c for c in numeric_cols if c.lower() in p]
    mentioned_cat = [c for c in categorical_cols if c.lower() in p]

    # Build spec
    spec = {"title": prompt.capitalize()}

    if chart_type == "bar":
        spec["chart_type"] = "bar"
        spec["x"] = mentioned_cat[0] if mentioned_cat else (categorical_cols[0] if categorical_cols else None)
        spec["y"] = mentioned_num[0] if mentioned_num else (numeric_cols[0] if numeric_cols else None)
    elif chart_type == "line":
        spec["chart_type"] = "line"
        spec["x"] = mentioned_cat[0] if mentioned_cat else df.columns[0]
        spec["y"] = mentioned_num[0] if mentioned_num else (numeric_cols[0] if numeric_cols else None)
    elif chart_type == "pie":
        spec["chart_type"] = "pie"
        spec["x"] = mentioned_cat[0] if mentioned_cat else (categorical_cols[0] if categorical_cols else None)
        spec["y"] = mentioned_num[0] if mentioned_num else None
    elif chart_type == "scatter":
        spec["chart_type"] = "scatter"
        spec["x"] = mentioned_num[0] if len(mentioned_num) > 0 else (numeric_cols[0] if len(numeric_cols) > 0 else None)
        spec["y"] = mentioned_num[1] if len(mentioned_num) > 1 else (numeric_cols[1] if len(numeric_cols) > 1 else None)
    elif chart_type == "histogram":
        spec["chart_type"] = "histogram"
        spec["x"] = mentioned_num[0] if mentioned_num else (numeric_cols[0] if numeric_cols else None)
        spec["y"] = None

    spec["color"] = None
    return _build_figure_from_spec(spec, df, numeric_cols, categorical_cols)


def _build_figure_from_spec(
    spec: Dict,
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> Optional[go.Figure]:
    """Build a Plotly figure from a parsed chart spec dict."""
    import plotly.express as px
    from utils.visualizations import plot_bar_chart, plot_line_chart, plot_pie_chart, plot_scatter, plot_histogram

    chart_type = spec.get("chart_type", "bar")
    x = spec.get("x")
    y = spec.get("y")
    color = spec.get("color")
    title = spec.get("title", "Generated Chart")

    try:
        if chart_type == "bar" and x and y and x in df.columns and y in df.columns:
            return plot_bar_chart(df, x, y, title=title)
        elif chart_type == "line" and x and y and x in df.columns and y in df.columns:
            return plot_line_chart(df, x, y, title=title)
        elif chart_type == "pie" and x and x in df.columns:
            return plot_pie_chart(df, x, y if y and y in df.columns else None, title=title)
        elif chart_type == "scatter" and x and y and x in df.columns and y in df.columns:
            color_col = color if color and color in df.columns else None
            return plot_scatter(df, x, y, color_col, title=title)
        elif chart_type == "histogram" and x and x in df.columns:
            return plot_histogram(df, x, title=title)
        else:
            # Fallback: auto-pick reasonable chart
            if numeric_cols and categorical_cols:
                return plot_bar_chart(df, categorical_cols[0], numeric_cols[0], title=title)
            return None
    except Exception:
        return None


# ─── AI Recommendation Engine ─────────────────────────────────────────────────

def generate_recommendations(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    api_key: str = ""
) -> List[Dict[str, str]]:
    """
    Generate actionable recommendations for the dataset.
    Returns list of {category, title, description, priority} dicts.
    """
    recommendations = []

    # 1. Missing data recommendations
    missing = df.isnull().sum()
    high_missing = missing[missing / len(df) > 0.2]
    if len(high_missing) > 0:
        for col in high_missing.index[:3]:
            pct = missing[col] / len(df) * 100
            recommendations.append({
                "category": "🧹 Data Quality",
                "title": f"High missingness in '{col}'",
                "description": f"{pct:.0f}% of values are missing. Consider imputation (median/mode) or dropping this column if >50% missing.",
                "priority": "high" if pct > 50 else "medium"
            })

    # 2. Correlation-based recommendations
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)
        pairs = corr.unstack()
        high_corr = pairs[pairs > 0.85].index.tolist()
        for (a, b) in list(set([(min(x, y), max(x, y)) for x, y in high_corr]))[:2]:
            recommendations.append({
                "category": "📊 Feature Engineering",
                "title": f"Redundant columns: '{a}' & '{b}'",
                "description": f"These columns have correlation > 0.85. Consider removing one to reduce multicollinearity in models.",
                "priority": "medium"
            })

    # 3. Low-variance columns
    for col in numeric_cols:
        cv = df[col].std() / (df[col].mean() + 1e-9)
        if cv < 0.01 and df[col].std() > 0:
            recommendations.append({
                "category": "🔍 Feature Selection",
                "title": f"Near-constant column: '{col}'",
                "description": f"This column has very low variance (CV={cv:.4f}). It may not add predictive value.",
                "priority": "low"
            })

    # 4. High-cardinality categoricals
    for col in categorical_cols:
        n_unique = df[col].nunique()
        if n_unique > len(df) * 0.5:
            recommendations.append({
                "category": "🏷️ Categorical Encoding",
                "title": f"High cardinality in '{col}'",
                "description": f"{n_unique} unique values — this column may need hashing or target encoding instead of one-hot encoding.",
                "priority": "medium"
            })

    # 5. Skewness
    for col in numeric_cols[:5]:
        try:
            skew = df[col].skew()
            if abs(skew) > 2:
                direction = "right (positive)" if skew > 0 else "left (negative)"
                recommendations.append({
                    "category": "📐 Distribution",
                    "title": f"Skewed distribution: '{col}'",
                    "description": f"Skewness = {skew:.2f} ({direction} skew). Consider log or Box-Cox transformation for modeling.",
                    "priority": "medium"
                })
        except Exception:
            pass

    # 6. Small dataset
    if len(df) < 100:
        recommendations.append({
            "category": "📦 Data Collection",
            "title": "Small dataset",
            "description": f"Only {len(df)} rows. ML models and statistical tests may be unreliable. Collect more data if possible.",
            "priority": "high"
        })

    # 7. Duplicate rows
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        recommendations.append({
            "category": "🧹 Data Quality",
            "title": f"{n_dup} duplicate rows detected",
            "description": "Duplicate rows can bias analysis. Remove them with df.drop_duplicates().",
            "priority": "high" if n_dup / len(df) > 0.05 else "low"
        })

    # 8. AI-powered recommendations (if API key available)
    if api_key and len(recommendations) < 5:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            summary = (
                f"Dataset: {df.shape[0]} rows, {df.shape[1]} cols. "
                f"Numeric: {numeric_cols[:5]}. Categorical: {categorical_cols[:5]}."
            )
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": f"Give 2 concise data analysis recommendations for this dataset: {summary}"
                }]
            )
            ai_text = resp.choices[0].message.content.strip()
            recommendations.append({
                "category": "🤖 AI Recommendations",
                "title": "GPT-Powered Suggestions",
                "description": ai_text,
                "priority": "info"
            })
        except Exception:
            pass

    return recommendations[:12]  # Cap at 12 recommendations


def explain_chart_with_ai(
    chart_type: str,
    x_col: str,
    y_col: Optional[str],
    df: pd.DataFrame,
    api_key: str = ""
) -> str:
    """
    Generate an AI explanation for a specific chart.
    Falls back to rule-based explanation if no API key.
    """
    if not api_key:
        return _explain_chart_rule_based(chart_type, x_col, y_col, df)

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        # Build chart stats summary
        stats = []
        if y_col and y_col in df.columns and x_col in df.columns:
            agg = df.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(5)
            stats = [f"{k}: {v:.2f}" for k, v in agg.items()]

        prompt = (
            f"I have a {chart_type} chart showing {y_col or x_col} by {x_col}. "
            f"Top values: {', '.join(stats) if stats else 'N/A'}. "
            "Give a 2-3 sentence plain-English insight about what this chart reveals, "
            "including trends, outliers, or actionable observations."
        )

        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content.strip()

    except Exception:
        return _explain_chart_rule_based(chart_type, x_col, y_col, df)


def _explain_chart_rule_based(
    chart_type: str,
    x_col: str,
    y_col: Optional[str],
    df: pd.DataFrame
) -> str:
    """Rule-based chart explanation."""
    lines = []
    if y_col and y_col in df.columns and x_col in df.columns:
        try:
            agg = df.groupby(x_col)[y_col].mean()
            top = agg.idxmax()
            bottom = agg.idxmin()
            lines.append(f"**{top}** has the highest average {y_col} ({agg.max():.2f}).")
            lines.append(f"**{bottom}** has the lowest ({agg.min():.2f}).")
            spread = agg.max() - agg.min()
            lines.append(f"The range across {x_col} is **{spread:.2f}**, indicating {'high' if spread > agg.mean() else 'moderate'} variation.")
        except Exception:
            lines.append(f"This {chart_type} chart compares {y_col} across different {x_col} values.")
    elif x_col in df.columns and pd.api.types.is_numeric_dtype(df[x_col]):
        s = df[x_col].dropna()
        lines.append(f"**{x_col}** ranges from {s.min():.2f} to {s.max():.2f} with a mean of {s.mean():.2f}.")
        skew = s.skew()
        if abs(skew) > 1:
            lines.append(f"The distribution is {'right' if skew > 0 else 'left'}-skewed (skewness: {skew:.2f}).")
    else:
        lines.append(f"This {chart_type} chart visualizes patterns in {x_col}.")
        lines.append("Add an OpenAI API key for AI-powered chart explanations.")

    return " ".join(lines)
