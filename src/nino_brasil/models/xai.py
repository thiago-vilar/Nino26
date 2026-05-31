from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance


def permutation_importance_frame(
    model,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    scoring: str | None = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return permutation importance as a tidy DataFrame."""
    result = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def shap_importance_frame(
    model,
    X: pd.DataFrame,
    *,
    max_rows: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return mean absolute SHAP values by feature."""
    import shap

    sample = X
    if len(X) > max_rows:
        sample = X.sample(max_rows, random_state=random_state)
    explain_model = model
    explain_sample = sample
    if isinstance(model, Pipeline):
        explain_model = model.steps[-1][1]
        if len(model.steps) > 1:
            transformed = model[:-1].transform(sample)
            explain_sample = pd.DataFrame(transformed, index=sample.index, columns=sample.columns)
    explainer = shap.Explainer(explain_model, explain_sample)
    values = explainer(explain_sample)
    arr = values.values
    if arr.ndim == 3:
        arr = arr[..., -1]
    mean_abs = np.abs(arr).mean(axis=0)
    return pd.DataFrame({"feature": sample.columns, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap",
        ascending=False,
    )


def feature_group_weights(
    importance: pd.DataFrame,
    feature_groups: dict[str, str],
    *,
    value_column: str,
) -> pd.DataFrame:
    """Aggregate feature importances into ocean, atmosphere and other groups."""
    frame = importance.copy()
    frame["group"] = frame["feature"].map(feature_groups).fillna("other")
    grouped = frame.groupby("group", as_index=False)[value_column].sum()
    total = grouped[value_column].abs().sum()
    grouped["weight"] = grouped[value_column].abs() / total if total else 0.0
    return grouped.sort_values("weight", ascending=False)
