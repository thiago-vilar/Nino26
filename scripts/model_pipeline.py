from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.config import load_config, project_path
from nino_brasil.data.zarr_store import dataframe_to_zarr
from nino_brasil.models.walk_forward import make_walk_forward_folds, run_walk_forward


COMMON_RESOLUTION_DEG = 0.25


def _assert_common_resolution(ds: xr.Dataset, path: str, expected: float = COMMON_RESOLUTION_DEG) -> None:
    for axis in ("lat", "lon"):
        if axis in ds.coords and ds.sizes.get(axis, 0) > 1:
            step = float(np.abs(np.diff(ds[axis].values)).mean())
            if not np.isclose(step, expected, atol=1e-3):
                raise ValueError(
                    f"{path}: {axis} step {step:.4f} deg differs from the common "
                    f"{expected} deg grid; run regrid-zarr on it first."
                )


def _open_predictors(paths: list[str]) -> xr.Dataset:
    datasets = [xr.open_zarr(path) for path in paths]
    try:
        for ds, path in zip(datasets, paths):
            _assert_common_resolution(ds, path)
        # no_conflicts: fail loudly on overlapping variables/coords instead of
        # silently keeping the first dataset's values.
        return xr.merge(datasets, compat="no_conflicts")
    except Exception:
        for ds in datasets:
            ds.close()
        raise


def _group_weights(importances: pd.DataFrame) -> pd.DataFrame:
    if importances.empty:
        return pd.DataFrame()
    frame = importances.copy()
    frame["abs_importance"] = frame["importance_value"].abs()
    keys = [
        column
        for column in ["fold", "split", "lag_days", "model", "task", "region", "event", "method", "season"]
        if column in frame.columns
    ]
    grouped = frame.groupby([*keys, "group"], dropna=False, as_index=False)["abs_importance"].sum()
    totals = grouped.groupby(keys, dropna=False)["abs_importance"].transform("sum")
    grouped["weight"] = grouped["abs_importance"] / totals.where(totals != 0)
    return grouped.sort_values([*keys, "weight"], ascending=[*[True] * len(keys), False])


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def metrics_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate walk-forward metrics into a compact CSV-friendly table."""
    keys = [column for column in ["model", "task", "lag_days", "region", "event", "season"] if column in metrics.columns]
    metric_columns = [
        column
        for column in ["rmse", "mae", "correlation", "bias", "brier", "roc_auc", "precision", "recall", "hit_rate", "false_alarm_rate", "f1"]
        if column in metrics.columns
    ]
    if metrics.empty or not keys or not metric_columns:
        return pd.DataFrame(columns=[*keys, "metric", "mean", "std", "n"])
    long = metrics.melt(
        id_vars=keys,
        value_vars=metric_columns,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    if long.empty:
        return pd.DataFrame(columns=[*keys, "metric", "mean", "std", "n"])
    return (
        long.groupby([*keys, "metric"], dropna=False)["value"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .round({"mean": 4, "std": 4})
    )


def baseline_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare fitted models against baseline rows when those rows are present."""
    columns = [
        "model",
        "lag_days",
        "region",
        "season",
        "event",
        "task",
        "metric",
        "model_value",
        "persistence_baseline",
        "climatology_baseline",
        "delta_vs_persistence",
        "delta_vs_climatology",
    ]
    if metrics.empty or "model" not in metrics.columns:
        return pd.DataFrame(columns=columns)

    baseline_models = {"persistence", "climatology"}
    if not baseline_models.intersection(set(metrics["model"].dropna())):
        return pd.DataFrame(columns=columns)

    summary = metrics_summary(metrics)
    if summary.empty:
        return pd.DataFrame(columns=columns)
    keys = [column for column in ["lag_days", "region", "season", "event", "task", "metric"] if column in summary.columns]
    values = summary.rename(columns={"mean": "model_value"})
    base = values[values["model"].isin(baseline_models)][[*keys, "model", "model_value"]]
    fitted = values[~values["model"].isin(baseline_models)][[*keys, "model", "model_value"]]
    if fitted.empty:
        return pd.DataFrame(columns=columns)

    pivot = base.pivot_table(index=keys, columns="model", values="model_value", aggfunc="mean").reset_index()
    out = fitted.merge(pivot, on=keys, how="left")
    out = out.rename(columns={"persistence": "persistence_baseline", "climatology": "climatology_baseline"})
    if "persistence_baseline" not in out.columns:
        out["persistence_baseline"] = np.nan
    if "climatology_baseline" not in out.columns:
        out["climatology_baseline"] = np.nan
    out["delta_vs_persistence"] = out["model_value"] - out.get("persistence_baseline")
    out["delta_vs_climatology"] = out["model_value"] - out.get("climatology_baseline")
    return out.reindex(columns=columns)


def importance_top20_by_lag_model(importances: pd.DataFrame) -> pd.DataFrame:
    """Return top features by lag/model/task from numeric importance outputs."""
    keys = [column for column in ["lag_days", "model", "task", "method", "event"] if column in importances.columns]
    columns = [*keys, "feature", "group", "importance_mean", "importance_std", "n_rows"]
    if importances.empty or "importance_mean" not in importances.columns or "feature" not in importances.columns:
        return pd.DataFrame(columns=columns)
    frame = importances.copy()
    frame = frame[~frame["feature"].astype(str).str.startswith("__")]
    frame = frame.dropna(subset=["importance_mean"])
    if frame.empty:
        return pd.DataFrame(columns=columns)
    aggregate_keys = [*keys, "feature"]
    if "group" in frame.columns:
        aggregate_keys.append("group")
    grouped = (
        frame.groupby(aggregate_keys, dropna=False)["importance_mean"]
        .agg(importance_mean="mean", importance_std="std", n_rows="count")
        .reset_index()
    )
    if not keys:
        return grouped.sort_values("importance_mean", ascending=False).head(20).reset_index(drop=True)
    return (
        grouped.sort_values("importance_mean", ascending=False)
        .groupby(keys, dropna=False, group_keys=False)
        .head(20)
        .reset_index(drop=True)
        .round({"importance_mean": 6, "importance_std": 6})
    )


def shap_group_summary(importances: pd.DataFrame) -> pd.DataFrame:
    """Aggregate SHAP importances by feature group and across folds."""
    columns = [
        "group",
        "lag_days",
        "model",
        "task",
        "region",
        "event",
        "mean_abs_shap_sum_mean",
        "mean_abs_shap_sum_std",
        "n_folds",
    ]
    if importances.empty or "method" not in importances.columns:
        return pd.DataFrame(columns=columns)
    shap = importances[importances["method"] == "shap"].copy()
    if shap.empty:
        return pd.DataFrame(columns=columns)
    keys = [column for column in ["group", "fold", "lag_days", "model", "task", "region", "event"] if column in shap.columns]
    fold_sums = shap.groupby(keys, dropna=False)["importance_value"].sum().reset_index(name="mean_abs_shap_sum")
    aggregate_keys = [column for column in keys if column != "fold"]
    if not aggregate_keys:
        return pd.DataFrame(columns=columns)
    return (
        fold_sums.groupby(aggregate_keys, dropna=False)["mean_abs_shap_sum"]
        .agg(mean_abs_shap_sum_mean="mean", mean_abs_shap_sum_std="std", n_folds="count")
        .reset_index()
        .round({"mean_abs_shap_sum_mean": 6, "mean_abs_shap_sum_std": 6})
    )


def xai_method_agreement(importances: pd.DataFrame) -> pd.DataFrame:
    """Calculate Spearman rank agreement between permutation and SHAP importances."""
    columns = ["fold", "lag_days", "model", "task", "region", "event", "spearman_rank_corr", "n_features"]
    if importances.empty or "method" not in importances.columns:
        return pd.DataFrame(columns=columns)
    frame = importances[
        importances["method"].isin({"permutation", "shap"})
        & ~importances["feature"].astype(str).str.startswith("__")
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    keys = [column for column in ["fold", "lag_days", "model", "task", "region", "event"] if column in frame.columns]
    if not keys:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for key_values, group in frame.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        pivot = group.pivot_table(index="feature", columns="method", values="importance_value", aggfunc="mean").dropna()
        if {"permutation", "shap"}.issubset(pivot.columns) and len(pivot) >= 2:
            corr = pivot["permutation"].rank().corr(pivot["shap"].rank(), method="pearson")
            rows.append(
                {
                    **dict(zip(keys, key_values)),
                    "spearman_rank_corr": round(float(corr), 6) if pd.notna(corr) else np.nan,
                    "n_features": int(len(pivot)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NINO-BRASIL phase 5 leakage-safe classical ML/XAI evaluation.")
    parser.add_argument("--predictor-zarr", action="append", required=True, help="Regridded predictor Zarr path.")
    parser.add_argument("--target-zarr", required=True, help="Regridded CHIRPS target Zarr path.")
    parser.add_argument("--target-var", required=True, help="Target precipitation variable name.")
    parser.add_argument("--lag-days", type=int, action="append", help="Lag in days; defaults to project weekly horizons.")
    parser.add_argument("--initial-train-years", type=int, default=20)
    parser.add_argument("--validation-years", type=int, default=2)
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--step-years", type=int, default=1)
    parser.add_argument("--predictor-strategy", choices=["mean", "flatten"], default="mean")
    parser.add_argument("--target-strategy", choices=["mean", "flatten"], default="mean")
    parser.add_argument("--regression-model", action="append", choices=["ridge", "rf", "xgboost", "lightgbm"])
    parser.add_argument("--classification-model", action="append", choices=["logistic", "rf", "xgboost", "lightgbm"])
    parser.add_argument("--no-importance", action="store_true", help="Skip permutation importance.")
    parser.add_argument("--importance-repeats", type=int, default=5)
    parser.add_argument(
        "--permutation-feature-limit",
        type=int,
        default=500,
        help="Skip permutation importance above this feature count to protect flatten runs.",
    )
    parser.add_argument("--shap", action="store_true", help="Calculate SHAP importances for tree models.")
    parser.add_argument("--shap-max-rows", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42, help="Seed for models, importances and SHAP sampling.")
    parser.add_argument("--output-dir", default="data/processed/zarr/modeling")
    parser.add_argument("--table-output-dir", default="data/processed/parquet/modeling")
    parser.add_argument("--dry-run", action="store_true", help="Open data and print planned folds without fitting models.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config()
    lags = args.lag_days or cfg["time"]["lag_days"]

    predictors = _open_predictors(args.predictor_zarr)
    target_ds = xr.open_zarr(args.target_zarr)
    try:
        target = target_ds[args.target_var]
        folds = make_walk_forward_folds(
            pd.DatetimeIndex(target["time"].values),
            initial_train_years=args.initial_train_years,
            validation_years=args.validation_years,
            test_years=args.test_years,
            step_years=args.step_years,
        )
        if args.dry_run:
            print(f"predictor variables: {list(predictors.data_vars)}")
            print(f"target: {args.target_var}")
            print(f"lags: {lags}")
            print(f"folds: {len(folds)}")
            for fold in folds[:5]:
                print(f"- {fold}")
            return 0

        output = run_walk_forward(
            predictors,
            target,
            lags_days=lags,
            folds=folds,
            regression_models=args.regression_model or ("ridge", "rf", "lightgbm"),
            classification_models=args.classification_model or ("logistic", "rf", "lightgbm"),
            predictor_strategy=args.predictor_strategy,
            target_strategy=args.target_strategy,
            climatology_window_days=cfg["time"].get("climatology_window_days", 15),
            dry_percentile=cfg["targets"].get("dry_percentile", 10),
            lower_percentile=cfg["targets"].get("lower_percentile", 25),
            upper_percentile=cfg["targets"].get("upper_percentile", 75),
            wet_percentile=cfg["targets"].get("wet_percentile", 90),
            compute_importance=not args.no_importance,
            compute_shap=args.shap,
            importance_n_repeats=args.importance_repeats,
            permutation_feature_limit=args.permutation_feature_limit,
            shap_max_rows=args.shap_max_rows,
            random_state=args.random_state,
        )

        output_dir = project_path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "walk_forward_metrics.zarr"
        predictions_path = output_dir / "walk_forward_predictions.zarr"
        importances_path = output_dir / "walk_forward_importances.zarr"
        group_weights_path = output_dir / "walk_forward_group_weights.zarr"
        dataframe_to_zarr(output.metrics, metrics_path, overwrite=True, attrs={"artifact": "walk_forward_metrics"})
        dataframe_to_zarr(output.predictions, predictions_path, overwrite=True, attrs={"artifact": "walk_forward_predictions"})
        dataframe_to_zarr(output.importances, importances_path, overwrite=True, attrs={"artifact": "walk_forward_importances"})
        group_weights = _group_weights(output.importances)
        dataframe_to_zarr(group_weights, group_weights_path, overwrite=True, attrs={"artifact": "walk_forward_group_weights"})
        table_dir = project_path(args.table_output_dir)
        metrics_summary_path = _write_csv(metrics_summary(output.metrics), table_dir / "walk_forward_metrics_summary.csv")
        baseline_path = _write_csv(baseline_comparison(output.metrics), table_dir / "walk_forward_baseline_comparison.csv")
        top20_path = _write_csv(importance_top20_by_lag_model(output.importances), table_dir / "importance_top20_by_lag_model.csv")
        shap_summary_path = _write_csv(shap_group_summary(output.importances), table_dir / "shap_group_summary.csv")
        agreement_path = _write_csv(xai_method_agreement(output.importances), table_dir / "xai_method_agreement.csv")
        print(f"metrics: {metrics_path}")
        print(f"predictions: {predictions_path}")
        print(f"importances: {importances_path}")
        print(f"group weights: {group_weights_path}")
        print(f"metrics summary: {metrics_summary_path}")
        print(f"baseline comparison: {baseline_path}")
        print(f"importance top20: {top20_path}")
        print(f"shap group summary: {shap_summary_path}")
        print(f"xai method agreement: {agreement_path}")
        return 0
    finally:
        predictors.close()
        target_ds.close()


if __name__ == "__main__":
    raise SystemExit(main())
