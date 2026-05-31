from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nino_brasil.data.anomalies import daily_anomaly, fit_daily_standardization, standardized_anomaly
from nino_brasil.features.precipitation_events import event_mask, local_percentile_threshold
from nino_brasil.models.baselines import classification_metrics, regression_metrics
from nino_brasil.models.feature_matrix import FeatureMatrix, SpatialStrategy, build_feature_matrix
from nino_brasil.models.xai import permutation_importance_frame, shap_importance_frame


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class WalkForwardOutput:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    importances: pd.DataFrame


def make_walk_forward_folds(
    times: pd.DatetimeIndex,
    *,
    initial_train_years: int = 20,
    validation_years: int = 2,
    test_years: int = 1,
    step_years: int = 1,
) -> list[WalkForwardFold]:
    """Create chronological walk-forward blocks."""
    if times.empty:
        raise ValueError("times must not be empty.")
    start = pd.Timestamp(times.min()).normalize()
    last = pd.Timestamp(times.max()).normalize()
    folds: list[WalkForwardFold] = []
    fold_idx = 0
    train_end = start + pd.DateOffset(years=initial_train_years) - pd.Timedelta(days=1)
    while True:
        validation_start = train_end + pd.Timedelta(days=1)
        validation_end = validation_start + pd.DateOffset(years=validation_years) - pd.Timedelta(days=1)
        test_start = validation_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)
        if test_start > last:
            break
        folds.append(
            WalkForwardFold(
                name=f"fold_{fold_idx:02d}",
                train_start=start,
                train_end=min(train_end, last),
                validation_start=validation_start,
                validation_end=min(validation_end, last),
                test_start=test_start,
                test_end=min(test_end, last),
            )
        )
        fold_idx += 1
        train_end = train_end + pd.DateOffset(years=step_years)
    if not folds:
        raise ValueError("No walk-forward fold could be built for the supplied time span.")
    return folds


def _season(month: int) -> str:
    if month in {12, 1, 2}:
        return "DJF"
    if month in {3, 4, 5}:
        return "MAM"
    if month in {6, 7, 8}:
        return "JJA"
    return "SON"


def _make_regressor(name: str, random_state: int) -> Pipeline:
    if name == "ridge":
        estimator = Ridge(alpha=1.0)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    if name == "rf":
        estimator = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    if name == "xgboost":
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        estimator = LGBMRegressor(random_state=random_state, n_jobs=-1)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    raise ValueError(f"Unknown regression model: {name}")


def _make_classifier(name: str, random_state: int) -> Pipeline:
    if name in {"logistic", "ridge"}:
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced")
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    if name == "rf":
        estimator = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    if name == "xgboost":
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        estimator = LGBMClassifier(random_state=random_state, n_jobs=-1, class_weight="balanced")
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    raise ValueError(f"Unknown classification model: {name}")


def _split_mask(target_time: pd.Series, fold: WalkForwardFold, split: str) -> pd.Series:
    if split == "train":
        return (target_time >= fold.train_start) & (target_time <= fold.train_end)
    if split == "validation":
        return (target_time >= fold.validation_start) & (target_time <= fold.validation_end)
    if split == "test":
        return (target_time >= fold.test_start) & (target_time <= fold.test_end)
    raise ValueError("split must be train, validation or test.")


def _predict_class_score(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    score = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-score))


def _metric_rows(
    y_true: np.ndarray,
    y_pred_or_score: np.ndarray,
    *,
    target_time: pd.Series,
    base: dict[str, object],
    task: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metric_fn = regression_metrics if task == "regression" else classification_metrics

    all_metrics = metric_fn(y_true, y_pred_or_score)
    rows.append({**base, "season": "all", **all_metrics})
    seasons = target_time.dt.month.map(_season)
    for season in ["DJF", "MAM", "JJA", "SON"]:
        mask = seasons == season
        if mask.sum() == 0:
            continue
        values = metric_fn(y_true[mask.to_numpy()], y_pred_or_score[mask.to_numpy()])
        rows.append({**base, "season": season, **values})
    return rows


def _prediction_frame(
    y_true: np.ndarray,
    y_pred_or_score: np.ndarray,
    *,
    index: pd.Index,
    target_time: pd.Series,
    base: dict[str, object],
    column: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "origin_time": pd.DatetimeIndex(index),
            "target_time": target_time.to_numpy(),
            "target": column,
            "y_true": y_true,
            "y_pred": y_pred_or_score,
        }
    )
    for key, value in base.items():
        frame[key] = value
    frame["season"] = pd.Series(pd.DatetimeIndex(frame["target_time"]).month).map(_season).to_numpy()
    return frame


def _importance_scoring(task: str) -> str:
    if task == "regression":
        return "neg_root_mean_squared_error"
    return "neg_brier_score"


def _tree_model_name(model_name: str) -> bool:
    return model_name in {"rf", "xgboost", "lightgbm"}


def _importance_frames(
    model: Pipeline,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    *,
    base: dict[str, object],
    matrix: FeatureMatrix,
    task: str,
    model_name: str,
    random_state: int,
    n_repeats: int,
    compute_shap: bool,
    shap_max_rows: int,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    y_for_importance = y_eval.astype(int) if task == "classification" else y_eval
    permutation = permutation_importance_frame(
        model,
        X_eval,
        y_for_importance,
        scoring=_importance_scoring(task),
        n_repeats=n_repeats,
        random_state=random_state,
    )
    permutation["method"] = "permutation"
    permutation["season"] = "all"
    permutation["importance_value"] = permutation["importance_mean"]
    permutation["group"] = permutation["feature"].map(matrix.feature_groups).fillna("other")
    for key, value in base.items():
        permutation[key] = value
    frames.append(permutation)

    if compute_shap and _tree_model_name(model_name):
        try:
            shap_frame = shap_importance_frame(
                model,
                X_eval,
                max_rows=shap_max_rows,
                random_state=random_state,
            )
        except Exception as exc:  # pragma: no cover - SHAP can vary by optional backend
            shap_frame = pd.DataFrame(
                {
                    "feature": ["__shap_error__"],
                    "mean_abs_shap": [np.nan],
                    "error": [f"{type(exc).__name__}: {exc}"],
                }
            )
        shap_frame = shap_frame.rename(columns={"mean_abs_shap": "importance_mean"})
        shap_frame["importance_std"] = np.nan
        shap_frame["method"] = "shap"
        shap_frame["season"] = "all"
        shap_frame["importance_value"] = shap_frame["importance_mean"]
        shap_frame["group"] = shap_frame["feature"].map(matrix.feature_groups).fillna("other")
        for key, value in base.items():
            shap_frame[key] = value
        frames.append(shap_frame)

    return frames


def _fit_evaluate_matrix(
    matrix: FeatureMatrix,
    *,
    fold: WalkForwardFold,
    model_names: Sequence[str],
    task: str,
    random_state: int,
    compute_importance: bool,
    compute_shap: bool,
    importance_n_repeats: int,
    shap_max_rows: int,
) -> WalkForwardOutput:
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    train_mask = _split_mask(matrix.target_time, fold, "train")

    for split in ["validation", "test"]:
        split_mask = _split_mask(matrix.target_time, fold, split)
        if split_mask.sum() == 0:
            continue
        for target_column in matrix.y.columns:
            y_train = matrix.y.loc[train_mask, target_column].dropna()
            if y_train.empty:
                continue
            train_index = y_train.index
            X_train = matrix.X.loc[train_index]
            y_eval = matrix.y.loc[split_mask, target_column].dropna()
            if y_eval.empty:
                continue
            eval_index = y_eval.index
            X_eval = matrix.X.loc[eval_index]

            if task == "classification" and np.unique(y_train.astype(int)).size < 2:
                continue

            for model_name in model_names:
                model = (
                    _make_regressor(model_name, random_state)
                    if task == "regression"
                    else _make_classifier(model_name, random_state)
                )
                model.fit(X_train, y_train.astype(int) if task == "classification" else y_train)
                y_pred = (
                    model.predict(X_eval)
                    if task == "regression"
                    else _predict_class_score(model, X_eval)
                )
                base = {
                    "fold": fold.name,
                    "split": split,
                    "lag_days": matrix.lag_days,
                    "model": model_name,
                    "task": task,
                    "region": target_column,
                }
                metric_rows.extend(
                    _metric_rows(
                        y_eval.to_numpy(),
                        y_pred,
                        target_time=matrix.target_time.loc[eval_index],
                        base=base,
                        task=task,
                    )
                )
                prediction_frames.append(
                    _prediction_frame(
                        y_eval.to_numpy(),
                        y_pred,
                        index=eval_index,
                        target_time=matrix.target_time.loc[eval_index],
                        base=base,
                        column=target_column,
                    )
                )
                if compute_importance and split == "test":
                    importance_frames.extend(
                        _importance_frames(
                            model,
                            X_eval,
                            y_eval,
                            base=base,
                            matrix=matrix,
                            task=task,
                            model_name=model_name,
                            random_state=random_state,
                            n_repeats=importance_n_repeats,
                            compute_shap=compute_shap,
                            shap_max_rows=shap_max_rows,
                        )
                    )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    importances = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    return WalkForwardOutput(metrics=metrics, predictions=predictions, importances=importances)


def _standardize_predictors_for_fold(
    predictors: xr.Dataset,
    fold: WalkForwardFold,
    *,
    time_name: str,
    window_days: int,
) -> xr.Dataset:
    transformed = {}
    train_times = predictors[time_name].where(predictors[time_name] <= np.datetime64(fold.train_end), drop=True)
    for variable in predictors.data_vars:
        da = predictors[variable]
        if time_name not in da.dims:
            transformed[variable] = da
            continue
        clim, std = fit_daily_standardization(
            da,
            train_times=train_times,
            window_days=window_days,
            time_name=time_name,
        )
        transformed[variable] = standardized_anomaly(
            da,
            climatology=clim,
            std=std.where(std != 0),
            window_days=window_days,
            time_name=time_name,
        )
    return xr.Dataset(transformed, coords=predictors.coords, attrs=predictors.attrs)


def run_walk_forward(
    predictors: xr.Dataset,
    target: xr.DataArray,
    *,
    lags_days: Sequence[int],
    folds: Sequence[WalkForwardFold] | None = None,
    regression_models: Sequence[str] = ("ridge", "rf", "xgboost"),
    classification_models: Sequence[str] = ("logistic", "rf", "xgboost"),
    dry_percentile: float = 10.0,
    wet_percentile: float = 90.0,
    time_name: str = "time",
    climatology_window_days: int = 15,
    predictor_strategy: SpatialStrategy = "mean",
    target_strategy: SpatialStrategy = "mean",
    standardize_predictors: bool = True,
    compute_importance: bool = True,
    compute_shap: bool = False,
    importance_n_repeats: int = 5,
    shap_max_rows: int = 1000,
    random_state: int = 42,
) -> WalkForwardOutput:
    """Run leakage-safe walk-forward regression and event classification."""
    time_index = pd.DatetimeIndex(target[time_name].values)
    fold_list = list(folds) if folds is not None else make_walk_forward_folds(time_index)

    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_importances: list[pd.DataFrame] = []
    for fold in fold_list:
        target_train_times = target[time_name].where(
            (target[time_name] >= np.datetime64(fold.train_start))
            & (target[time_name] <= np.datetime64(fold.train_end)),
            drop=True,
        )
        target_clim, _ = fit_daily_standardization(
            target,
            train_times=target_train_times,
            window_days=climatology_window_days,
            time_name=time_name,
        )
        target_anomaly = daily_anomaly(
            target,
            climatology=target_clim,
            window_days=climatology_window_days,
            time_name=time_name,
        )
        dry_threshold = local_percentile_threshold(
            target,
            dry_percentile,
            train_times=target_train_times,
            time_name=time_name,
        )
        wet_threshold = local_percentile_threshold(
            target,
            wet_percentile,
            train_times=target_train_times,
            time_name=time_name,
        )
        dry_target = event_mask(
            target,
            dry_percentile,
            "dry",
            threshold=dry_threshold,
            time_name=time_name,
        )
        wet_target = event_mask(
            target,
            wet_percentile,
            "wet",
            threshold=wet_threshold,
            time_name=time_name,
        )

        fold_predictors = (
            _standardize_predictors_for_fold(
                predictors,
                fold,
                time_name=time_name,
                window_days=climatology_window_days,
            )
            if standardize_predictors
            else predictors
        )

        for lag in lags_days:
            regression_matrix = build_feature_matrix(
                fold_predictors,
                target_anomaly,
                lag_days=int(lag),
                time_name=time_name,
                predictor_strategy=predictor_strategy,
                target_strategy=target_strategy,
                climatology_window_days=climatology_window_days,
            )
            regression_output = _fit_evaluate_matrix(
                regression_matrix,
                fold=fold,
                model_names=regression_models,
                task="regression",
                random_state=random_state,
                compute_importance=compute_importance,
                compute_shap=compute_shap,
                importance_n_repeats=importance_n_repeats,
                shap_max_rows=shap_max_rows,
            )
            all_metrics.append(regression_output.metrics)
            all_predictions.append(regression_output.predictions)
            all_importances.append(regression_output.importances)

            for event_name, event_target in [("dry", dry_target), ("wet", wet_target)]:
                class_matrix = build_feature_matrix(
                    fold_predictors,
                    event_target.astype(int).rename(event_name),
                    lag_days=int(lag),
                    time_name=time_name,
                    predictor_strategy=predictor_strategy,
                    target_strategy=target_strategy,
                    climatology_window_days=climatology_window_days,
                )
                class_output = _fit_evaluate_matrix(
                    class_matrix,
                    fold=fold,
                    model_names=classification_models,
                    task="classification",
                    random_state=random_state,
                    compute_importance=compute_importance,
                    compute_shap=compute_shap,
                    importance_n_repeats=importance_n_repeats,
                    shap_max_rows=shap_max_rows,
                )
                if not class_output.metrics.empty:
                    class_output.metrics["event"] = event_name
                if not class_output.predictions.empty:
                    class_output.predictions["event"] = event_name
                if not class_output.importances.empty:
                    class_output.importances["event"] = event_name
                all_metrics.append(class_output.metrics)
                all_predictions.append(class_output.predictions)
                all_importances.append(class_output.importances)

    metric_frames = [frame for frame in all_metrics if not frame.empty]
    prediction_frames = [frame for frame in all_predictions if not frame.empty]
    importance_frames = [frame for frame in all_importances if not frame.empty]
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    importances = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    return WalkForwardOutput(metrics=metrics, predictions=predictions, importances=importances)
