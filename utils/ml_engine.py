"""
utils/ml_engine.py
Predictive analytics: auto-select models, train, evaluate, forecast.
Supports regression and classification based on target column type.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """Detect whether the ML task is regression or classification."""
    series = df[target_col].dropna()
    n_unique = series.nunique()

    if pd.api.types.is_numeric_dtype(series):
        # If low cardinality numeric → treat as classification
        if n_unique <= 10:
            return "classification"
        return "regression"
    return "classification"


def prepare_features(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> Tuple:
    """Encode and prepare X, y for ML."""
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    work = df[feature_cols + [target_col]].copy().dropna()

    X = work[feature_cols].copy()
    y = work[target_col].copy()

    # Encode categorical features
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Encode target if classification
    le_target = None
    if not pd.api.types.is_numeric_dtype(y):
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))

    return X.values, np.array(y), le_target


def run_regression(X_train, X_test, y_train, y_test) -> Dict[str, Any]:
    """Train and evaluate multiple regression models."""
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42),
    }

    results = {}
    best_score = -np.inf
    best_name = None
    best_model = None

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            results[name] = {"R² Score": round(r2, 4), "MAE": round(mae, 4), "RMSE": round(rmse, 4)}

            if r2 > best_score:
                best_score = r2
                best_name = name
                best_model = model
        except Exception as e:
            results[name] = {"error": str(e)}

    return {
        "results": results,
        "best_model_name": best_name,
        "best_model": best_model,
        "best_score": best_score,
        "task": "regression"
    }


def run_classification(X_train, X_test, y_train, y_test) -> Dict[str, Any]:
    """Train and evaluate multiple classification models."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, f1_score

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
    }

    results = {}
    best_score = -np.inf
    best_name = None
    best_model = None

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

            results[name] = {"Accuracy": round(acc, 4), "F1 Score (weighted)": round(f1, 4)}

            if acc > best_score:
                best_score = acc
                best_name = name
                best_model = model
        except Exception as e:
            results[name] = {"error": str(e)}

    return {
        "results": results,
        "best_model_name": best_name,
        "best_model": best_model,
        "best_score": best_score,
        "task": "classification"
    }


def get_feature_importance(model, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """Extract feature importances if model supports it."""
    try:
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(15)
        return fi_df
    except AttributeError:
        try:
            coefs = np.abs(model.coef_).flatten()[:len(feature_names)]
            fi_df = pd.DataFrame({
                "Feature": feature_names[:len(coefs)],
                "Importance": coefs
            }).sort_values("Importance", ascending=False).head(15)
            return fi_df
        except Exception:
            return None


def run_forecast(df: pd.DataFrame, target_col: str, periods: int = 10) -> Optional[pd.DataFrame]:
    """
    Simple time-series-style forecast using linear extrapolation.
    Works on any numeric sequence (index as time proxy).
    """
    try:
        from sklearn.linear_model import LinearRegression

        series = df[target_col].dropna().reset_index(drop=True)
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values

        model = LinearRegression()
        model.fit(X, y)

        future_X = np.arange(len(series), len(series) + periods).reshape(-1, 1)
        preds = model.predict(future_X)

        historical = pd.DataFrame({
            "Index": range(len(series)),
            target_col: y,
            "Type": "Historical"
        })
        forecast = pd.DataFrame({
            "Index": range(len(series), len(series) + periods),
            target_col: preds,
            "Type": "Forecast"
        })
        return pd.concat([historical, forecast], ignore_index=True)
    except Exception:
        return None


def run_predictive_analysis(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Full pipeline: detect task, prepare data, train models, evaluate.
    Returns comprehensive results dict.
    """
    try:
        from sklearn.model_selection import train_test_split

        task = detect_task_type(df, target_col)
        X, y, le_target = prepare_features(df, target_col, feature_cols)

        if len(X) < 10:
            return {"error": "Not enough data rows (need at least 10)."}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if task == "regression":
            result = run_regression(X_train, X_test, y_train, y_test)
        else:
            result = run_classification(X_train, X_test, y_train, y_test)

        # Feature importance
        if result.get("best_model"):
            fi = get_feature_importance(result["best_model"], feature_cols)
            result["feature_importance"] = fi

        result["task"] = task
        result["n_samples"] = len(X)
        result["le_target"] = le_target

        return result

    except ImportError:
        return {"error": "scikit-learn not installed. Run: pip install scikit-learn"}
    except Exception as e:
        return {"error": str(e)}
