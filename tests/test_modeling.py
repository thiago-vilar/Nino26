from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.models.feature_matrix import build_feature_matrix
from nino_brasil.models.walk_forward import (
    _nino34_for_fold,
    _train_times,
    make_walk_forward_folds,
    run_walk_forward,
)


def synthetic_daily_fields() -> tuple[xr.Dataset, xr.DataArray]:
    rng = np.random.default_rng(7)
    time = pd.date_range("2001-01-01", "2005-12-31", freq="D")
    lat = np.array([-5.0, 5.0])
    lon = np.array([190.0, 220.0, 240.0])
    seasonal = np.sin(2 * np.pi * np.arange(time.size) / 365.0)
    sst = 27.0 + seasonal[:, None, None] + 0.05 * rng.normal(size=(time.size, lat.size, lon.size))
    slp = 1010.0 - seasonal[:, None, None] + 0.05 * rng.normal(size=(time.size, lat.size, lon.size))
    lagged = np.roll(seasonal, 30)
    precip = 5.0 + lagged[:, None, None] + rng.gamma(2.0, 0.4, size=(time.size, lat.size, lon.size))
    predictors = xr.Dataset(
        {
            "sst": (("time", "lat", "lon"), sst),
            "slp": (("time", "lat", "lon"), slp),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    target = xr.DataArray(
        precip,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="precip",
    )
    return predictors, target


class ModelingTests(unittest.TestCase):
    def test_build_feature_matrix_tracks_target_time(self) -> None:
        predictors, target = synthetic_daily_fields()
        matrix = build_feature_matrix(predictors, target, lag_days=30)

        self.assertIn("sst", matrix.X.columns)
        self.assertIn("slp", matrix.X.columns)
        self.assertIn("nino34_ssta", matrix.X.columns)
        self.assertEqual(matrix.lag_days, 30)
        self.assertEqual(matrix.target_time.iloc[0], matrix.X.index[0] + pd.Timedelta(days=30))

    def test_nino34_fold_feature_ignores_post_train_values(self) -> None:
        predictors, _ = synthetic_daily_fields()
        train_end = pd.Timestamp("2003-12-31")
        base = _nino34_for_fold(
            predictors,
            _train_times(predictors["time"], train_end),
            sst_variable="sst",
            time_name="time",
            window_days=15,
        )

        perturbed = predictors.copy(deep=True)
        perturbed["sst"].loc[{"time": slice("2004-01-01", None)}] += 5.0
        shifted = _nino34_for_fold(
            perturbed,
            _train_times(perturbed["time"], train_end),
            sst_variable="sst",
            time_name="time",
            window_days=15,
        )

        # Train-only climatology: perturbing validation/test SST must not move
        # the feature inside the training block (leakage regression guard).
        np.testing.assert_allclose(
            base.sel(time=slice(None, train_end)).values,
            shifted.sel(time=slice(None, train_end)).values,
        )
        # Post-train values must reflect the +5 perturbation. Day-of-year 366 is
        # absent from the 2001-2003 train climatology and yields NaN, so drop it.
        diff = (shifted - base).sel(time=slice("2004-01-01", None)).dropna("time")
        self.assertGreater(diff.sizes["time"], 0)
        self.assertTrue(bool((diff > 4.0).all().item()))

    def test_walk_forward_outputs_regression_and_classification_metrics(self) -> None:
        predictors, target = synthetic_daily_fields()
        folds = make_walk_forward_folds(
            pd.DatetimeIndex(target.time.values),
            initial_train_years=2,
            validation_years=1,
            test_years=1,
            step_years=1,
        )
        output = run_walk_forward(
            predictors,
            target,
            lags_days=[30],
            folds=folds,
            regression_models=("ridge",),
            classification_models=("logistic",),
        )

        self.assertFalse(output.metrics.empty)
        self.assertFalse(output.predictions.empty)
        self.assertFalse(output.importances.empty)
        self.assertIn("regression", set(output.metrics["task"]))
        self.assertIn("classification", set(output.metrics["task"]))
        self.assertTrue({"below_p25", "above_p75"}.issubset(set(output.metrics["event"].dropna())))
        self.assertIn("rmse", output.metrics.columns)
        self.assertIn("brier", output.metrics.columns)
        self.assertIn("importance_value", output.importances.columns)
        self.assertEqual(set(output.metrics["lag_days"]), {30})


if __name__ == "__main__":
    unittest.main()
