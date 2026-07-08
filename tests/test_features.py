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
from nino_brasil.features.dhw import degree_heating_weeks, equatorial_dhw_indices, thermal_window_mean
from nino_brasil.features.nino import (
    atl3_sst_index,
    atl4_sst_index,
    iod_sst_index,
    nino34_sst_index,
    tropical_atlantic_sst_indices,
)
from nino_brasil.features.ocean_heat import (
    CP0,
    RHO0,
    ocean_heat_content,
    layer_ocean_heat_content,
    ohc_tendency,
    warm_water_volume,
    warm_water_volume_m3,
)
from nino_brasil.features.precipitation_events import event_mask, local_percentile_threshold, local_percentile_thresholds
from nino_brasil.features.tendency_features import equatorial_zonal_stress, ssta_tendency, wind_stress_curl
from nino_brasil.features.thermocline import d20_depth, d20_tendency, thermocline_tilt, thermocline_tilt_slope


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

    def test_layer_ohc_clips_reconstructed_cells_at_exact_boundary(self) -> None:
        temperature = xr.DataArray(
            np.full(5, 10.0),
            coords={"depth": [0.5, 100.0, 300.0, 650.0, 760.0]},
            dims=("depth",),
        )
        ohc = layer_ocean_heat_content(temperature, 0.0, 700.0)
        self.assertAlmostEqual(float(ohc), RHO0 * CP0 * 10.0 * 700.0, places=1)

    def test_layer_ohc_rejects_unbracketed_700m_limit(self) -> None:
        temperature = xr.DataArray(
            np.full(4, 10.0),
            coords={"depth": [0.5, 100.0, 300.0, 500.0]},
            dims=("depth",),
        )
        with self.assertRaisesRegex(ValueError, "cannot integrate exactly"):
            layer_ocean_heat_content(temperature, 0.0, 700.0)

    def test_d20_depth_linearly_interpolates_isotherm(self) -> None:
        temperature = xr.DataArray(
            [26.0, 22.0, 19.0, 16.0],
            coords={"depth": [0.0, 50.0, 100.0, 150.0]},
            dims=("depth",),
        )
        self.assertAlmostEqual(float(d20_depth(temperature)), 83.333333, places=4)

    def test_d20_depth_is_missing_without_a_physical_crossing(self) -> None:
        temperature = xr.DataArray(
            [18.0, 17.0, 15.0],
            coords={"depth": [0.0, 50.0, 100.0]},
            dims=("depth",),
        )
        self.assertTrue(np.isnan(float(d20_depth(temperature))))

    def test_warm_water_volume_averages_equatorial_pacific_d20(self) -> None:
        lat = np.array([-10.0, 0.0, 10.0])
        lon = np.array([100.0, 140.0, 180.0, 220.0, 260.0, 300.0])
        d20 = xr.DataArray(
            np.tile(lon / 10.0, (lat.size, 1)),
            coords={"lat": lat, "lon": lon},
            dims=("lat", "lon"),
            name="d20",
        )

        wwv = warm_water_volume(d20)

        self.assertEqual(wwv.name, "wwv")
        self.assertEqual(wwv.attrs["physics"], "recharge_oscillator_wwv")
        self.assertAlmostEqual(float(wwv), 20.0)

    def test_thermocline_tilt_returns_east_minus_west_d20(self) -> None:
        lat = np.array([-10.0, 0.0, 10.0])
        lon = np.array([100.0, 140.0, 180.0, 220.0, 260.0, 300.0])
        d20 = xr.DataArray(
            np.tile(lon / 10.0, (lat.size, 1)),
            coords={"lat": lat, "lon": lon},
            dims=("lat", "lon"),
        )

        tilt = thermocline_tilt(d20)

        self.assertEqual(tilt.name, "thermocline_tilt")
        self.assertEqual(tilt.attrs["physics"], "bjerknes_tilt")
        self.assertAlmostEqual(float(tilt), 8.0)
        self.assertAlmostEqual(float(thermocline_tilt_slope(d20)), 0.1)

    def test_warm_water_volume_m3_integrates_positive_cell_volume(self) -> None:
        d20 = xr.DataArray(
            np.full((3, 3), 100.0),
            coords={"lat": [-1.0, 0.0, 1.0], "lon": [120.0, 121.0, 122.0]},
            dims=("lat", "lon"),
        )
        volume = warm_water_volume_m3(d20, lat_bounds=(-1.0, 1.0), lon_bounds=(120.0, 122.0))
        self.assertGreater(float(volume), 1.0e12)
        self.assertEqual(volume.attrs["units"], "m3")

    def test_pacific_boxes_accept_negative_longitude_convention(self) -> None:
        lat = np.array([0.0])
        lon = np.array([140.0, 180.0, -140.0, -90.0])
        d20 = xr.DataArray(
            [[14.0, 18.0, 22.0, 27.0]],
            coords={"lat": lat, "lon": lon},
            dims=("lat", "lon"),
        )

        wwv = warm_water_volume(d20)
        tilt = thermocline_tilt(d20)

        self.assertAlmostEqual(float(wwv), 20.25)
        self.assertAlmostEqual(float(tilt), 8.5)

    def test_physical_tendencies_use_actual_time_delta(self) -> None:
        time = pd.to_datetime(["2001-01-01", "2001-01-02", "2001-01-04"])
        d20 = xr.DataArray([10.0, 13.0, 19.0], coords={"time": time}, dims=("time",))
        ohc = xr.DataArray([0.0, 86400.0, 259200.0], coords={"time": time}, dims=("time",), name="ohc")
        ssta = xr.DataArray([0.0, 1.0, 5.0], coords={"time": time}, dims=("time",))

        np.testing.assert_allclose(d20_tendency(d20).values, [3.0, 3.0])
        np.testing.assert_allclose(ohc_tendency(ohc).values, [1.0, 1.0])
        np.testing.assert_allclose(ssta_tendency(ssta).values, [1.0, 2.0])

    def test_wind_stress_curl_and_equatorial_zonal_stress(self) -> None:
        lat = np.array([-1.0, 0.0, 1.0])
        lon = np.array([0.0, 1.0, 2.0])
        meters_per_degree = np.deg2rad(1.0) * 6.371e6
        tau_y_values = meters_per_degree * np.cos(np.deg2rad(lat))[:, None] * lon[None, :]
        tau_y = xr.DataArray(tau_y_values, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
        tau_x = xr.zeros_like(tau_y)

        curl = wind_stress_curl(tau_x, tau_y)
        stress = equatorial_zonal_stress(tau_x, lon_bounds=(0.0, 2.0))

        self.assertEqual(curl.attrs["physics"], "ekman_pumping")
        np.testing.assert_allclose(curl.values, np.ones_like(curl.values), rtol=1e-6)
        self.assertAlmostEqual(float(stress), 0.0)

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

    def test_percentile_thresholds_fit_multiple_quantiles_in_one_result(self) -> None:
        time = pd.date_range("2001-01-01", periods=5, freq="D")
        precip = xr.DataArray([0.0, 1.0, 2.0, 3.0, 100.0], coords={"time": time}, dims=("time",))
        train_times = precip.time.isel(time=slice(0, 4))

        thresholds = local_percentile_thresholds(precip, [25.0, 75.0], train_times=train_times)

        self.assertIn("quantile", thresholds.dims)
        self.assertAlmostEqual(float(thresholds.sel(quantile=0.25)), 0.75)
        self.assertAlmostEqual(float(thresholds.sel(quantile=0.75)), 2.25)

    def test_power_law_tail_diagnostic_returns_comparisons(self) -> None:
        rng = np.random.default_rng(11)
        values = (rng.pareto(2.5, size=2000) + 1.0) * 3.0
        fit = fit_power_law_tail(values, variable="synthetic", min_tail=100)

        self.assertEqual(fit.variable, "synthetic")
        self.assertGreater(fit.alpha, 1.0)
        self.assertTrue(np.isfinite(fit.llr_power_law_vs_lognormal))
        self.assertIn(fit.preferred_distribution, {"power_law", "lognormal", "exponential"})

    def test_sst_indices_cover_pacific_atlantic_and_iod_boxes(self) -> None:
        lat = np.arange(-30.0, 31.0, 1.0)
        lon = np.arange(0.0, 360.0, 1.0)
        values = lon[None, :] + lat[:, None] * 0.0
        sst = xr.DataArray(values, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))

        self.assertAlmostEqual(float(nino34_sst_index(sst)), 215.0, places=6)
        self.assertAlmostEqual(float(atl3_sst_index(sst)), 6990.0 / 21.0, places=6)
        self.assertAlmostEqual(float(atl4_sst_index(sst)), 322.5, places=6)
        atl = tropical_atlantic_sst_indices(sst)
        self.assertTrue({"atl3_sst", "atl4_sst", "tna_sst", "tsa_sst"}.issubset(set(atl.data_vars)))
        self.assertAlmostEqual(float(iod_sst_index(sst)), -40.0, places=6)

    def test_degree_heating_weeks_accumulates_daily_hotspots_as_c_weeks(self) -> None:
        time = pd.date_range("2001-01-01", periods=14, freq="D")
        ssta = xr.DataArray(np.full(time.size, 2.0), coords={"time": time}, dims=("time",))

        dhw = degree_heating_weeks(ssta, threshold_c=1.0, window_weeks=2)

        self.assertTrue(np.isnan(float(dhw.isel(time=12))))
        self.assertAlmostEqual(float(dhw.isel(time=13)), 4.0)
        self.assertEqual(dhw.attrs["units"], "degree_C_weeks")
        self.assertEqual(dhw.attrs["hotspot_rule"], "SSTA if SSTA >= threshold_c else 0")

    def test_degree_heating_weeks_can_require_consecutive_hotspot_gate(self) -> None:
        time = pd.date_range("2001-01-01", periods=21, freq="D")
        values = np.r_[np.full(13, 2.0), 0.0, np.full(7, 2.0)]
        ssta = xr.DataArray(values, coords={"time": time}, dims=("time",))

        dhw = degree_heating_weeks(
            ssta,
            threshold_c=1.0,
            window_weeks=2,
            require_consecutive_weeks=2,
        )

        self.assertTrue(np.isnan(float(dhw.isel(time=12))))
        self.assertEqual(float(dhw.isel(time=13)), 0.0)
        self.assertEqual(float(dhw.isel(time=20)), 0.0)
        self.assertEqual(dhw.attrs["require_consecutive_weeks"], 2)

        continuous = xr.DataArray(np.full(14, 2.0), coords={"time": time[:14]}, dims=("time",))
        gated = degree_heating_weeks(
            continuous,
            threshold_c=1.0,
            window_weeks=2,
            require_consecutive_weeks=2,
        )
        self.assertAlmostEqual(float(gated.isel(time=13)), 4.0)

    def test_thermal_window_mean_uses_same_12_week_basis(self) -> None:
        time = pd.date_range("2001-01-01", periods=14, freq="D")
        ssta = xr.DataArray(np.arange(time.size, dtype=float), coords={"time": time}, dims=("time",))

        mean = thermal_window_mean(ssta, window_weeks=2)

        self.assertTrue(np.isnan(float(mean.isel(time=12))))
        self.assertAlmostEqual(float(mean.isel(time=13)), 6.5)

    def test_equatorial_dhw_indices_emit_expected_weekly_boxes(self) -> None:
        time = pd.date_range("2001-01-01", periods=21, freq="D")
        lat = np.array([-5.0, 0.0, 5.0])
        lon = np.array([120.0, 180.0, 205.0, 240.0, 280.0])
        values = np.full((time.size, lat.size, lon.size), 2.0)
        ssta = xr.DataArray(values, coords={"time": time, "lat": lat, "lon": lon}, dims=("time", "lat", "lon"))

        ds = equatorial_dhw_indices(ssta, threshold_c=1.0, window_weeks=2)

        self.assertIn("equatorial_guide_dhw_2w", ds)
        self.assertIn("west_equatorial_guide_dhw_2w", ds)
        self.assertIn("east_equatorial_guide_dhw_2w", ds)
        self.assertGreaterEqual(ds.sizes["time"], 3)

    def test_equatorial_dhw_accumulates_daily_before_weekly_reduction(self) -> None:
        time = pd.date_range("2001-01-01", periods=14, freq="D")
        lat = np.array([-5.0, 0.0, 5.0])
        lon = np.array([190.0, 215.0, 240.0])
        values = np.zeros((time.size, lat.size, lon.size), dtype=float)
        values[6, :, :] = 8.0
        ssta = xr.DataArray(values, coords={"time": time, "lat": lat, "lon": lon}, dims=("time", "lat", "lon"))

        ds = equatorial_dhw_indices(ssta, threshold_c=1.0, window_weeks=1, weekly_reduction="max")

        weekly_peak = float(ds["nino34_dhw_1w"].max(skipna=True))
        self.assertAlmostEqual(weekly_peak, 8.0 / 7.0)
        self.assertEqual(
            ds["nino34_dhw_1w"].attrs["accumulation_resolution"],
            "native_daily_before_weekly_resample",
        )




class LongitudeBoundaryRegressionTests(unittest.TestCase):
    def test_atl3_on_offset_grid_without_lon_zero_cell(self) -> None:
        """Regressao: ATL3 (20W-0E) em grade deslocada (sem celula em 0.0)
        nao pode disparar wrap espurio com segmento vazio (quebrava no OISST
        bruto global 0.125-359.875)."""
        lat = np.arange(-29.875, 30.0, 0.25)
        lon = np.arange(0.125, 360.0, 0.25)
        time = pd.date_range("1981-09-01", periods=3, freq="D")
        values = np.ones((time.size, lat.size, lon.size)) * 25.0
        sst = xr.DataArray(values, coords={"time": time, "lat": lat, "lon": lon}, dims=("time", "lat", "lon"))

        atl3 = atl3_sst_index(sst)
        self.assertEqual(atl3.shape, (3,))
        self.assertTrue(np.allclose(atl3.values, 25.0))

        atl = tropical_atlantic_sst_indices(sst)
        for name in ("atl3_sst", "atl4_sst", "tna_sst", "tsa_sst"):
            self.assertEqual(atl[name].shape, (3,))
            self.assertTrue(np.allclose(atl[name].values, 25.0))


if __name__ == "__main__":
    unittest.main()
