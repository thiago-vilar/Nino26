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

from nino_brasil.data.anomalies import daily_anomaly, fit_daily_climatology, standardized_anomaly
from nino_brasil.features.distributions import fit_power_law_tail
from nino_brasil.features.ocean_heat import CP0, RHO0, ocean_heat_content
from nino_brasil.features.precipitation_events import event_mask, local_percentile_threshold
from nino_brasil.features.thermocline import d20_depth


class FeatureTests(unittest.TestCase):
    def test_ocean_heat_content_constant_profile(self) -> None:
        depth = np.array([0.0, 100.0, 200.0, 300.0])
        temperature = xr.DataArray(
            np.full((depth.size,), 10.0),
            coords={"depth": depth},
            dims=("depth",),
        )
        ohc = ocean_heat_content(temperature, 300.0)
        self.assertAlmostEqual(float(ohc), RHO0 * CP0 * 10.0 * 300.0)

    def test_d20_depth_uses_nearest_isotherm_layer(self) -> None:
        temperature = xr.DataArray(
            [26.0, 22.0, 19.0, 16.0],
            coords={"depth": [0.0, 50.0, 100.0, 150.0]},
            dims=("depth",),
        )
        self.assertEqual(float(d20_depth(temperature)), 100.0)

    def test_daily_anomaly_can_be_fit_on_train_block_only(self) -> None:
        time = pd.date_range("2001-01-01", "2003-12-31", freq="D")
        seasonal = np.sin(2 * np.pi * (time.dayofyear.to_numpy() - 1) / 365.0)
        values = seasonal.copy()
        values[time.year == 2003] += 10.0
        da = xr.DataArray(values, coords={"time": time}, dims=("time",))

        train_times = da.time.where(da.time.dt.year <= 2002, drop=True)
        climatology = fit_daily_climatology(da, train_times=train_times, window_days=1)
        anomaly = daily_anomaly(da, climatology=climatology, window_days=1)

        self.assertAlmostEqual(
            float(anomaly.sel(time=slice("2003-01-01", "2003-12-31")).mean()),
            10.0,
            places=6,
        )

    def test_standardized_anomaly_uses_supplied_smoothed_basis(self) -> None:
        time = pd.date_range("2001-01-01", "2002-12-31", freq="D")
        da = xr.DataArray(np.ones(time.size) * 3.0, coords={"time": time}, dims=("time",))
        clim = xr.DataArray(np.ones(365) * 1.0, coords={"dayofyear": np.arange(1, 366)}, dims=("dayofyear",))
        std = xr.DataArray(np.ones(365) * 2.0, coords={"dayofyear": np.arange(1, 366)}, dims=("dayofyear",))

        z = standardized_anomaly(da, climatology=clim, std=std, window_days=1)
        self.assertAlmostEqual(float(z.mean()), 1.0)

    def test_percentile_threshold_can_be_fit_on_train_block_only(self) -> None:
        time = pd.date_range("2001-01-01", periods=6, freq="D")
        precip = xr.DataArray([0.0, 0.0, 0.0, 100.0, 100.0, 100.0], coords={"time": time}, dims=("time",))
        train_times = precip.time.isel(time=slice(0, 3))
        threshold = local_percentile_threshold(precip, 90.0, train_times=train_times)
        wet = event_mask(precip, 90.0, "wet", threshold=threshold)

        self.assertEqual(float(threshold), 0.0)
        self.assertTrue(bool(wet.isel(time=3)))

    def test_power_law_tail_diagnostic_returns_comparisons(self) -> None:
        rng = np.random.default_rng(11)
        values = (rng.pareto(2.5, size=2000) + 1.0) * 3.0
        fit = fit_power_law_tail(values, variable="synthetic", min_tail=100)

        self.assertEqual(fit.variable, "synthetic")
        self.assertGreater(fit.alpha, 1.0)
        self.assertTrue(np.isfinite(fit.llr_power_law_vs_lognormal))
        self.assertIn(fit.preferred_distribution, {"power_law", "lognormal", "exponential"})


if __name__ == "__main__":
    unittest.main()
