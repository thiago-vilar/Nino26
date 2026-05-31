from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # pragma: no cover - compatibility for older scikit-learn
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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


def fit_random_forest_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Fit a random forest classification baseline."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X, y)
    return model


def fit_xgboost_regressor(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
):
    """Fit an XGBoost regressor when xgboost is installed."""
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def fit_xgboost_classifier(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
):
    """Fit an XGBoost classifier when xgboost is installed."""
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate baseline regression metrics."""
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    if np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel())[0, 1])
    bias = float(np.mean(y_pred - y_true))
    return {"rmse": float(rmse), "mae": float(mae), "correlation": corr, "bias": bias}


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate binary event metrics from probabilities or scores."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()
    y_pred = (y_score >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    if np.unique(y_true).size < 2:
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, y_score))

    hit_rate = tp / (tp + fn) if (tp + fn) else float("nan")
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else float("nan")

    return {
        "brier": float(brier_score_loss(y_true, y_score)),
        "roc_auc": roc_auc,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "hit_rate": float(hit_rate),
        "false_alarm_rate": float(false_alarm_rate),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
