from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.config import load_config, project_path
from nino_brasil.models.walk_forward import make_walk_forward_folds, run_walk_forward


def _open_predictors(paths: list[str]) -> xr.Dataset:
    datasets = [xr.open_zarr(path) for path in paths]
    try:
        return xr.merge(datasets, compat="override")
    except BaseException:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NINO-BRASIL leakage-safe model evaluation.")
    parser.add_argument("--predictor-zarr", action="append", required=True, help="Regridded predictor Zarr path.")
    parser.add_argument("--target-zarr", required=True, help="Regridded CHIRPS target Zarr path.")
    parser.add_argument("--target-var", required=True, help="Target precipitation variable name.")
    parser.add_argument("--lag-days", type=int, action="append", help="Lag in days; defaults to project monthly lags.")
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
    parser.add_argument("--shap", action="store_true", help="Calculate SHAP importances for tree models.")
    parser.add_argument("--shap-max-rows", type=int, default=1000)
    parser.add_argument("--output-dir", default="data/processed/parquet/modeling")
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
            regression_models=args.regression_model or ("ridge", "rf", "xgboost"),
            classification_models=args.classification_model or ("logistic", "rf", "xgboost"),
            predictor_strategy=args.predictor_strategy,
            target_strategy=args.target_strategy,
            climatology_window_days=cfg["time"].get("climatology_window_days", 15),
            dry_percentile=cfg["targets"].get("dry_percentile", 10),
            wet_percentile=cfg["targets"].get("wet_percentile", 90),
            compute_importance=not args.no_importance,
            compute_shap=args.shap,
            importance_n_repeats=args.importance_repeats,
            shap_max_rows=args.shap_max_rows,
        )

        output_dir = project_path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output.metrics.to_parquet(output_dir / "walk_forward_metrics.parquet", index=False)
        output.predictions.to_parquet(output_dir / "walk_forward_predictions.parquet", index=False)
        output.importances.to_parquet(output_dir / "walk_forward_importances.parquet", index=False)
        group_weights = _group_weights(output.importances)
        group_weights.to_parquet(output_dir / "walk_forward_group_weights.parquet", index=False)
        print(f"metrics: {output_dir / 'walk_forward_metrics.parquet'}")
        print(f"predictions: {output_dir / 'walk_forward_predictions.parquet'}")
        print(f"importances: {output_dir / 'walk_forward_importances.parquet'}")
        print(f"group weights: {output_dir / 'walk_forward_group_weights.parquet'}")
        return 0
    finally:
        predictors.close()
        target_ds.close()


if __name__ == "__main__":
    raise SystemExit(main())
