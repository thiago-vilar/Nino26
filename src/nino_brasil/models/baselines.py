from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Ridge:
    """Fit a ridge regression baseline."""
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model


def fit_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Fit a random forest baseline."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate baseline regression metrics."""
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    corr = float(np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1])
    bias = float(np.mean(y_pred - y_true))
    return {"rmse": float(rmse), "mae": float(mae), "correlation": corr, "bias": bias}
