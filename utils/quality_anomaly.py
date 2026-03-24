"""
utils/quality_anomaly.py
Dataset quality scoring and anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple


# ─── Dataset Quality Scoring ──────────────────────────────────────────────────

def compute_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute a composite data quality score (0-100) across 5 dimensions.
    Returns score breakdown and actionable recommendations.
    """
    scores = {}
    issues = []
    recommendations = []

    # 1. Completeness (0-25 pts): fraction of non-null values
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    completeness_pct = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 100
    completeness_score = completeness_pct * 0.25  # scale to 25
    scores["Completeness"] = round(completeness_score, 1)

    if completeness_pct < 95:
        issues.append(f"🔴 {missing_cells:,} missing values ({100 - completeness_pct:.1f}% of cells)")
        recommendations.append("Fill or drop rows/columns with excessive missing values.")
    else:
        recommendations.append("✅ Completeness looks great — minimal missing values.")

    # 2. Uniqueness (0-20 pts): penalize duplicate rows
    n_duplicates = df.duplicated().sum()
    dup_pct = n_duplicates / len(df) * 100 if len(df) > 0 else 0
    uniqueness_score = max(0, 20 - dup_pct * 2)
    scores["Uniqueness"] = round(uniqueness_score, 1)

    if n_duplicates > 0:
        issues.append(f"🟡 {n_duplicates} duplicate rows ({dup_pct:.1f}%)")
        recommendations.append("Remove duplicate rows to avoid skewed analysis.")

    # 3. Consistency (0-20 pts): mixed types in object columns
    inconsistency_count = 0
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(100)
        numeric_looking = sample.str.match(r'^-?\d+\.?\d*$', na=False).sum()
        if 0 < numeric_looking < len(sample) * 0.8:
            inconsistency_count += 1

    consistency_score = max(0, 20 - inconsistency_count * 4)
    scores["Consistency"] = round(consistency_score, 1)

    if inconsistency_count > 0:
        issues.append(f"🟡 {inconsistency_count} columns with mixed numeric/text values")
        recommendations.append("Standardize column types — some text columns may need conversion.")

    # 4. Validity (0-20 pts): outlier load in numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    outlier_flags = 0
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 3:
            z_scores = np.abs((series - series.mean()) / (series.std() + 1e-9))
            outlier_pct = (z_scores > 3).mean() * 100
            if outlier_pct > 5:
                outlier_flags += 1

    validity_score = max(0, 20 - outlier_flags * 4)
    scores["Validity"] = round(validity_score, 1)

    if outlier_flags > 0:
        issues.append(f"🟠 {outlier_flags} numeric columns have >5% extreme outliers")
        recommendations.append("Investigate and cap/remove extreme outliers before modeling.")

    # 5. Size adequacy (0-15 pts): enough rows for analysis
    n_rows = len(df)
    if n_rows >= 1000:
        size_score = 15
    elif n_rows >= 100:
        size_score = 10
    elif n_rows >= 20:
        size_score = 5
    else:
        size_score = 2
        issues.append(f"🔴 Very small dataset ({n_rows} rows) — results may be unreliable.")
        recommendations.append("Collect more data for statistically significant insights.")

    scores["Size Adequacy"] = size_score

    total = sum(scores.values())

    if total >= 85:
        grade = "A"
        grade_label = "Excellent"
        grade_color = "#22c55e"
    elif total >= 70:
        grade = "B"
        grade_label = "Good"
        grade_color = "#84cc16"
    elif total >= 55:
        grade = "C"
        grade_label = "Fair"
        grade_color = "#f59e0b"
    elif total >= 40:
        grade = "D"
        grade_label = "Poor"
        grade_color = "#ef4444"
    else:
        grade = "F"
        grade_label = "Critical"
        grade_color = "#7f1d1d"

    return {
        "total_score": round(total, 1),
        "max_score": 100,
        "grade": grade,
        "grade_label": grade_label,
        "grade_color": grade_color,
        "dimension_scores": scores,
        "issues": issues,
        "recommendations": recommendations,
    }


# ─── Anomaly Detection ────────────────────────────────────────────────────────

def detect_anomalies_zscore(df: pd.DataFrame, numeric_cols: List[str], threshold: float = 3.0) -> pd.DataFrame:
    """
    Z-score based anomaly detection.
    Returns a DataFrame with anomaly flags and scores per row.
    """
    if not numeric_cols:
        return pd.DataFrame()

    work = df[numeric_cols].copy()
    z_scores = work.apply(lambda col: np.abs((col - col.mean()) / (col.std() + 1e-9)))

    anomaly_mask = z_scores > threshold
    anomaly_score = z_scores.max(axis=1)
    n_anomalous_features = anomaly_mask.sum(axis=1)

    result = df.copy()
    result["_anomaly_score"] = anomaly_score.round(3)
    result["_n_anomalous_cols"] = n_anomalous_features
    result["_is_anomaly"] = anomaly_mask.any(axis=1)

    return result


def detect_anomalies_iqr(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """
    IQR-based per-column outlier detection.
    Returns bounds and outlier counts per column.
    """
    results = {}
    for col in numeric_cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]

        results[col] = {
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "outlier_count": len(outliers),
            "outlier_pct": round(len(outliers) / len(series) * 100, 2),
            "example_outliers": outliers.head(5).tolist(),
        }
    return results


def detect_anomalies_isolation_forest(
    df: pd.DataFrame,
    numeric_cols: List[str],
    contamination: float = 0.05
) -> Tuple[pd.DataFrame, str]:
    """
    Isolation Forest anomaly detection (sklearn).
    Returns annotated DataFrame and status message.
    """
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        work = df[numeric_cols].dropna()
        if len(work) < 10:
            return df, "Not enough data for Isolation Forest (need ≥ 10 rows)."

        scaler = StandardScaler()
        X = scaler.fit_transform(work)

        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(X)
        scores = iso.score_samples(X)

        result = df.copy()
        result["_if_anomaly"] = False
        result["_if_score"] = np.nan
        result.loc[work.index, "_if_anomaly"] = preds == -1
        result.loc[work.index, "_if_score"] = -scores  # higher = more anomalous

        n_anomalies = (preds == -1).sum()
        return result, f"✅ Isolation Forest found {n_anomalies} anomalies ({n_anomalies/len(work)*100:.1f}%)."

    except ImportError:
        return df, "scikit-learn not installed."
    except Exception as e:
        return df, f"Error: {e}"
