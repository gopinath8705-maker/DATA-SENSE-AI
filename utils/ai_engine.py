"""
utils/ai_engine.py
AI-powered features: dataset explanation, chat with data, insight generation.
Uses OpenAI when an API key is provided; falls back to rule-based responses.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_dataset_summary(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> str:
    """Build a compact text summary of the dataset to inject into prompts."""
    lines = [
        f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns.",
        f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}.",
        f"Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:10])}.",
    ]

    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols):
        lines.append(f"Columns with missing values: {', '.join(missing_cols.index.tolist())}.")

    if numeric_cols:
        stats = df[numeric_cols].describe().T[["mean", "min", "max"]]
        for col, row in stats.iterrows():
            lines.append(f"  {col}: mean={row['mean']:.2f}, min={row['min']:.2f}, max={row['max']:.2f}")

    if categorical_cols:
        for col in categorical_cols[:5]:
            top = df[col].value_counts().head(3)
            lines.append(f"  {col} top values: {', '.join([f'{v} ({c})' for v, c in top.items()])}")

    return "\n".join(lines)


def _call_openai(prompt: str, api_key: str, system: str = "") -> str:
    """
    Call the OpenAI Chat Completions API.
    Returns the response text or an error message.
    """
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=800,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except ImportError:
        return "⚠️ OpenAI package not installed. Run: `pip install openai`"
    except Exception as e:
        return f"⚠️ OpenAI API error: {e}"


# ─── Rule-based fallback responses ────────────────────────────────────────────

def _explain_dataset_fallback(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> str:
    """Generate a rule-based dataset explanation when no API key is provided."""
    lines = [
        "## 📊 Dataset Overview\n",
        f"This dataset contains **{df.shape[0]:,} rows** and **{df.shape[1]} columns**.",
        f"It has **{len(numeric_cols)} numeric** and **{len(categorical_cols)} categorical** columns.\n",
        "### 🔢 Numeric Summary",
    ]

    for col in numeric_cols[:5]:
        s = df[col].dropna()
        lines.append(
            f"- **{col}**: ranges from `{s.min():.2f}` to `{s.max():.2f}`, "
            f"average `{s.mean():.2f}`, std dev `{s.std():.2f}`."
        )

    if categorical_cols:
        lines.append("\n### 🏷️ Categorical Summary")
        for col in categorical_cols[:5]:
            top = df[col].value_counts().idxmax()
            pct = df[col].value_counts().max() / len(df) * 100
            lines.append(f"- **{col}**: most common value is `{top}` ({pct:.1f}% of rows).")

    missing = df.isnull().sum().sum()
    if missing > 0:
        lines.append(f"\n⚠️ **Missing values**: {missing:,} total missing values detected.")
    else:
        lines.append("\n✅ **No missing values** detected in the dataset.")

    lines.append(
        "\n> 💡 *Add your OpenAI API key in the sidebar for AI-generated insights and trend analysis.*"
    )
    return "\n".join(lines)


def _chat_fallback(df: pd.DataFrame, question: str) -> str:
    """Rule-based answers to common dataset questions."""
    q = question.lower()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Which column has highest values?
    if "highest" in q and ("column" in q or "values" in q) and numeric_cols:
        col = df[numeric_cols].mean().idxmax()
        val = df[numeric_cols].mean().max()
        return f"The column with the highest average value is **{col}** with a mean of **{val:,.2f}**."

    # Which column has lowest values?
    if "lowest" in q and ("column" in q or "values" in q) and numeric_cols:
        col = df[numeric_cols].mean().idxmin()
        val = df[numeric_cols].mean().min()
        return f"The column with the lowest average value is **{col}** with a mean of **{val:,.2f}**."

    # Most common category / top category
    if ("most" in q or "top" in q or "common" in q) and categorical_cols:
        col = categorical_cols[0]
        # Try to find a mentioned column
        for c in categorical_cols:
            if c.lower() in q:
                col = c
                break
        top = df[col].value_counts().idxmax()
        count = df[col].value_counts().max()
        return f"The most common value in **{col}** is **{top}** with **{count:,}** occurrences."

    # Missing values
    if "missing" in q or "null" in q or "nan" in q:
        total = df.isnull().sum().sum()
        if total == 0:
            return "✅ There are **no missing values** in this dataset."
        worst = df.isnull().sum().idxmax()
        worst_count = df.isnull().sum().max()
        return (
            f"There are **{total:,}** total missing values. "
            f"The column with the most missing data is **{worst}** ({worst_count:,} nulls)."
        )

    # Row count / shape
    if "how many rows" in q or "row count" in q:
        return f"The dataset has **{df.shape[0]:,} rows** and **{df.shape[1]} columns**."

    # Sum
    if "sum" in q or "total" in q:
        if numeric_cols:
            totals = df[numeric_cols].sum()
            summary = ", ".join([f"**{c}**: {v:,.2f}" for c, v in totals.items()][:5])
            return f"Column totals: {summary}"

    # Average / mean
    if "average" in q or "mean" in q:
        if numeric_cols:
            for c in numeric_cols:
                if c.lower() in q:
                    return f"The average of **{c}** is **{df[c].mean():,.2f}**."
            means = df[numeric_cols].mean()
            summary = ", ".join([f"**{c}**: {v:,.2f}" for c, v in means.items()][:5])
            return f"Column averages: {summary}"

    # Columns list
    if "columns" in q or "features" in q:
        return f"The dataset has {df.shape[1]} columns: {', '.join(df.columns.tolist())}."

    return (
        f"I found **{df.shape[0]:,} rows** and **{df.shape[1]} columns** in your dataset. "
        "For more advanced questions, please add your OpenAI API key in the sidebar to enable "
        "full natural language querying with LangChain."
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def explain_dataset(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    api_key: str = ""
) -> str:
    """
    Generate an AI explanation of the dataset.
    Uses OpenAI if api_key provided, otherwise fallback.
    """
    if not api_key:
        return _explain_dataset_fallback(df, numeric_cols, categorical_cols)

    summary = _build_dataset_summary(df, numeric_cols, categorical_cols)
    prompt = (
        f"You are a data analyst. Analyze this dataset summary and provide a clear, "
        f"structured explanation covering: key trends, anomalies, notable highest/lowest values, "
        f"and potential insights. Use markdown formatting.\n\n{summary}"
    )
    return _call_openai(prompt, api_key, system="You are an expert data analyst providing clear insights.")


def generate_ai_insights(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    api_key: str = ""
) -> str:
    """
    Generate detailed AI insights about the dataset.
    """
    if not api_key:
        return (
            "🔒 **AI Insights require an OpenAI API key.**\n\n"
            "Add your key in the sidebar to unlock:\n"
            "- Key trend identification\n"
            "- Anomaly detection\n"
            "- Correlation analysis\n"
            "- Business recommendations"
        )

    summary = _build_dataset_summary(df, numeric_cols, categorical_cols)
    prompt = (
        f"Analyze this dataset and provide 5-7 specific, actionable insights. "
        f"For each insight include: what it is, why it matters, and a recommendation. "
        f"Format as markdown with clear headers.\n\n{summary}"
    )
    return _call_openai(prompt, api_key, system="You are a senior business intelligence analyst.")


def chat_with_data(df: pd.DataFrame, question: str, api_key: str = "") -> str:
    """
    Answer a natural language question about the dataset.
    Uses LangChain Pandas Agent if api_key provided, otherwise rule-based fallback.
    """
    if not api_key:
        return _chat_fallback(df, question)

    # Try LangChain Pandas Agent
    try:
        from langchain_experimental.agents import create_pandas_dataframe_agent
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=api_key,
            temperature=0
        )
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=False,
            allow_dangerous_code=True,
            agent_type="openai-tools"
        )
        result = agent.invoke({"input": question})
        return result.get("output", str(result))

    except ImportError:
        # LangChain not installed — try direct OpenAI
        summary = _build_dataset_summary(
            df,
            df.select_dtypes(include=np.number).columns.tolist(),
            df.select_dtypes(include=["object", "category"]).columns.tolist()
        )
        prompt = (
            f"Based on this dataset summary, answer the following question concisely:\n\n"
            f"Dataset:\n{summary}\n\nQuestion: {question}"
        )
        return _call_openai(prompt, api_key)

    except Exception as e:
        # Fallback to rule-based if LangChain fails
        fallback = _chat_fallback(df, question)
        return f"{fallback}\n\n*(LangChain agent error: {e})*"
