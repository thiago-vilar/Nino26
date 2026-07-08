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

from nino_brasil.models.feature_matrix import (
    add_causal_monthly_ocean_features,
    align_target_to_predictor_matrix,
    build_feature_matrix,
    build_predictor_matrix,
    infer_feature_group,
)
from nino_brasil.models.ablation import PHYSICS_ABLATIONS
from nino_brasil.models.progression import (
    build_daily_enso_peak_progression_table,
    build_enso_peak_progression_table,
    build_nino34_cluster_progression_table,
    classify_nino_peak,
    cluster_pixel_events,
)
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
    def test_monthly_oras_features_are_attached_only_after_release(self) -> None:
        daily = xr.Dataset(
            {"sst": ("time", np.arange(50, dtype=float))},
            coords={"time": pd.date_range("2001-02-01", periods=50, freq="D")},
        )
        monthly = xr.Dataset(
            {"ohc": ("time", [10.0, 20.0])},
            coords={"time": pd.to_datetime(["2001-01-01", "2001-02-01"])},
        )
        merged, groups = add_causal_monthly_ocean_features(daily, monthly, publication_lag_days=15)

        name = "oras5_monthly__ohc"
        self.assertTrue(np.isnan(float(merged[name].sel(time="2001-02-14"))))
        self.assertEqual(float(merged[name].sel(time="2001-02-15")), 10.0)
        self.assertEqual(float(merged[name].sel(time="2001-03-15")), 20.0)
        self.assertEqual(groups[name], "ocean_monthly")

    def test_build_feature_matrix_tracks_target_time(self) -> None:
        predictors, target = synthetic_daily_fields()
        matrix = build_feature_matrix(predictors, target, lag_days=30)

        self.assertIn("sst", matrix.X.columns)
        self.assertIn("slp", matrix.X.columns)
        self.assertIn("nino34_ssta", matrix.X.columns)
        self.assertEqual(matrix.lag_days, 30)
        self.assertEqual(matrix.target_time.iloc[0], matrix.X.index[0] + pd.Timedelta(days=30))

    def test_reusable_predictor_matrix_matches_feature_matrix(self) -> None:
        predictors, target = synthetic_daily_fields()
        direct = build_feature_matrix(predictors, target, lag_days=30)
        reusable = build_predictor_matrix(predictors, lag_days=30)
        aligned = align_target_to_predictor_matrix(reusable, target)

        pd.testing.assert_frame_equal(aligned.X, direct.X)
        pd.testing.assert_frame_equal(aligned.y, direct.y)
        pd.testing.assert_series_equal(aligned.target_time, direct.target_time)

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
            embargo_days=30,
        )
        self.assertGreaterEqual((folds[0].validation_start - folds[0].train_end).days, 31)
        self.assertGreaterEqual((folds[0].test_start - folds[0].validation_end).days, 31)
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

    def test_physics_precalc_features_have_ablation_group(self) -> None:
        self.assertEqual(infer_feature_group("wwv"), "physics_precalc")
        self.assertEqual(infer_feature_group("thermocline_tilt"), "physics_precalc")
        self.assertEqual(infer_feature_group("ssta_tendency"), "physics_precalc")
        self.assertEqual(infer_feature_group("wind_stress_curl"), "physics_precalc")
        self.assertEqual(PHYSICS_ABLATIONS["G_with_physics"], {"ocean", "atmosphere", "physics_precalc"})
        self.assertEqual(PHYSICS_ABLATIONS["H_without_physics"], {"ocean", "atmosphere"})

    def test_enso_peak_progression_targets_muito_forte_el_nino_peak(self) -> None:
        time = pd.date_range("2001-01-01", "2002-12-01", freq="MS")
        ssta = np.linspace(-0.2, 2.2, len(time))
        monthly = pd.DataFrame(
            {
                "time": time,
                "nino34_ssta_mean": ssta,
                "d20_anomaly_m": np.linspace(0.0, 25.0, len(time)),
                "ohc_300_anomaly": np.linspace(0.0, 3.0, len(time)),
            }
        )

        table = build_enso_peak_progression_table(
            monthly,
            lead_months=[6],
            feature_columns=["nino34_ssta_mean", "d20_anomaly_m", "ohc_300_anomaly"],
        )
        row = table[table["origin_time"] == pd.Timestamp("2002-06-01")].iloc[0]

        self.assertEqual(classify_nino_peak(2.0), "muito_forte")
        self.assertEqual(row["future_peak_class"], "muito_forte")
        self.assertEqual(int(row["months_to_peak"]), 6)
        self.assertTrue(bool(row["will_muito_forte_el_nino"]))
        self.assertIn("d20_anomaly_m_delta_3m", table.columns)

    def test_daily_enso_progression_uses_daily_features_and_reference_peaks(self) -> None:
        time = pd.date_range("2001-01-01", periods=10, freq="D")
        daily = pd.DataFrame(
            {
                "time": time,
                "nino34_ssta": np.linspace(0.1, 2.1, len(time)),
                "d20_anomaly_m": np.linspace(0.0, 9.0, len(time)),
            }
        )
        peaks = pd.DataFrame(
            {
                "peak_time": [pd.Timestamp("2001-01-06")],
                "peak_ssta_c": [2.05],
                "peak_class": ["muito_forte"],
            }
        )

        table = build_daily_enso_peak_progression_table(
            daily,
            peaks,
            lead_days=[7],
            feature_columns=["nino34_ssta", "d20_anomaly_m"],
            history_windows_days=[3],
        )
        row = table[table["origin_time"] == pd.Timestamp("2001-01-01")].iloc[0]

        self.assertEqual(row["future_peak_class"], "muito_forte")
        self.assertEqual(int(row["days_to_peak"]), 5)
        self.assertTrue(bool(row["will_muito_forte_el_nino"]))
        self.assertIn("nino34_ssta_mean_3d", table.columns)
        self.assertIn("future_daily_max_ssta", table.columns)

    def test_nino34_cluster_progression_uses_clustered_pixel_targets(self) -> None:
        time = pd.date_range("2001-01-01", periods=6, freq="D")
        lat = np.array([-10.0, -5.0])
        lon = np.array([-45.0, -40.0])
        events = xr.DataArray(
            np.array(
                [
                    [[0, 0], [1, 1]],
                    [[0, 1], [1, 1]],
                    [[0, 0], [1, 1]],
                    [[1, 0], [0, 0]],
                    [[1, 1], [0, 0]],
                    [[1, 0], [0, 0]],
                ],
                dtype=float,
            ),
            coords={"time": time, "lat": lat, "lon": lon},
            dims=("time", "lat", "lon"),
            name="dry_event",
        )
        nino34 = pd.DataFrame(
            {
                "time": time,
                "nino34_ssta_mean": np.linspace(0.0, 1.0, len(time)),
                "nino34_ssta_slope_lon": np.linspace(-0.1, 0.2, len(time)),
            }
        )

        clusters = cluster_pixel_events(events, n_clusters=2, random_state=0)
        table = build_nino34_cluster_progression_table(
            nino34,
            events,
            clusters,
            lags_days=[1],
            feature_columns=["nino34_ssta_mean", "nino34_ssta_slope_lon"],
        )

        self.assertEqual(set(clusters["cluster"]), {0, 1})
        self.assertFalse(table.empty)
        self.assertEqual(set(table["lag_days"]), {1})
        self.assertTrue(table["target_event_rate"].between(0.0, 1.0).all())
        self.assertIn("nino34_ssta_mean_delta_3m", table.columns)


if __name__ == "__main__":
    unittest.main()
