from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

from nino_brasil.features.marine_heatwave import detect_mhw, export_mhw_catalog
from nino_brasil.features.nino34_event_table import build_nino34_event_table, export_nino34_event_tables
from nino_brasil.features.nino34_reference import (
    build_monthly_nino34_sst_reference,
    build_nino34_sst_p90_peaks,
    build_nino34_sst_peak_reference,
    save_nino34_sst_p90_plot,
)
from nino_brasil.maps.plot_choropleths import export_choropleth_table
from nino_brasil.maps.plot_pixel_maps import export_pixel_table, save_pixel_map


class NumericOutputTests(unittest.TestCase):
    def test_pixel_map_exports_csv_before_png(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            da = xr.DataArray(
                np.array([[1.23456, np.nan], [3.0, 4.0]]),
                coords={"lat": [-5.0, 0.0], "lon": [190.0, 191.0]},
                dims=("lat", "lon"),
                name="signal",
            )
            png_path = tmp_dir / "signal.png"

            save_pixel_map(da, png_path, "Signal", value_name="signal")

            csv_path = png_path.with_suffix(".csv")
            self.assertTrue(png_path.exists())
            self.assertTrue(csv_path.exists())
            table = pd.read_csv(csv_path)
            self.assertEqual(list(table.columns), ["lat", "lon", "signal"])
            self.assertEqual(len(table), 3)
        finally:
            shutil.rmtree(tmp_dir)

    def test_choropleth_export_removes_geometry_and_adds_rank(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            gdf = gpd.GeoDataFrame(
                {"NM_UF": ["A", "B"], "signal": [0.2, 0.8]},
                geometry=[Point(0, 0), Point(1, 1)],
                crs="EPSG:4326",
            )
            path = export_choropleth_table(gdf, "signal", tmp_dir / "choropleth.csv")
            table = pd.read_csv(path)

            self.assertEqual(list(table.columns), ["NM_UF", "signal", "rank"])
            self.assertEqual(table.loc[table["NM_UF"] == "B", "rank"].iloc[0], 1)
        finally:
            shutil.rmtree(tmp_dir)

    def test_nino34_event_table_exports_monthly_and_peak_tables(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            time = pd.date_range("2001-01-01", "2002-12-31", freq="D")
            lat = [-5.0, 0.0]
            lon = [190.0, 240.0]
            warm = np.where(time.year == 2002, 1.0, 0.1)
            shape = (time.size, len(lat), len(lon))
            ssta_values = warm[:, None, None] + np.zeros(shape)
            ssta_values = ssta_values + np.linspace(0.0, 0.2, len(lon))[None, None, :]
            ssta = xr.DataArray(
                ssta_values,
                coords={"time": time, "lat": lat, "lon": lon},
                dims=("time", "lat", "lon"),
            )
            d20 = xr.full_like(ssta, 120.0)
            ohc300 = xr.full_like(ssta, 1.2e9)
            ohc700 = xr.full_like(ssta, 2.4e9)
            thermocline = xr.full_like(ssta, 90.0)

            table = build_nino34_event_table(
                ssta,
                d20,
                ohc300,
                ohc700,
                thermocline,
                peak_years=[2002],
            )
            monthly_path, peak_path = export_nino34_event_tables(table, tmp_dir)

            self.assertEqual(len(table), 24)
            self.assertIn("nino34_ssta_slope_lon", table.columns)
            self.assertTrue((table[table["year"] == 2002]["signal_duration_months"] >= 1).all())
            self.assertTrue(monthly_path.exists())
            self.assertTrue(peak_path.exists())
            self.assertEqual(len(pd.read_csv(peak_path)), 12)
        finally:
            shutil.rmtree(tmp_dir)

    def test_mhw_catalog_detects_numeric_events(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            time = pd.date_range("2001-01-01", periods=20, freq="D")
            values = np.ones((20, 2, 2)) * 25.0
            values[10:16, :, :] = 30.0
            sst = xr.DataArray(
                values,
                coords={"time": time, "lat": [-1.0, 1.0], "lon": [190.0, 191.0]},
                dims=("time", "lat", "lon"),
            )
            train_times = sst.time.isel(time=slice(0, 10))

            catalog = detect_mhw(sst, train_times=train_times)
            path = export_mhw_catalog(catalog, tmp_dir / "mhw_catalog.csv")

            self.assertEqual(len(catalog), 1)
            self.assertEqual(int(catalog["duration_days"].iloc[0]), 6)
            self.assertTrue(path.exists())
        finally:
            shutil.rmtree(tmp_dir)

    def test_local_oisst_nino34_reference_and_p90_peaks(self) -> None:
        time = pd.date_range("2000-01-01", "2001-12-31", freq="D")
        monthly_signal = {
            1: -0.2,
            2: 0.1,
            3: 0.3,
            4: 0.7,
            5: 0.9,
            6: 1.1,
            7: 1.3,
            8: 1.7,
            9: 2.1,
            10: 2.2,
            11: 1.9,
            12: 1.5,
        }
        ssta = np.array([monthly_signal[stamp.month] if stamp.year == 2000 else 0.1 for stamp in time])
        daily = pd.DataFrame(
            {
                "time": time,
                "nino34_sst": 27.0 + ssta,
                "nino34_ssta": ssta,
            }
        )

        monthly = build_monthly_nino34_sst_reference(daily)
        peaks = build_nino34_sst_peak_reference(monthly, min_duration_months=5)
        p90_peaks = build_nino34_sst_p90_peaks(monthly)

        self.assertEqual(len(monthly), 24)
        self.assertIn("nino34_ssta_c", monthly.columns)
        self.assertTrue(monthly["source"].str.contains("OISST").all())
        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks.iloc[0]["peak_class"], "super_el_nino")
        self.assertEqual(len(p90_peaks), 1)
        self.assertGreaterEqual(
            p90_peaks.iloc[0]["peak_nino34_anom_c"],
            p90_peaks.iloc[0]["percentile_threshold_c"],
        )

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            plot_path = save_nino34_sst_p90_plot(monthly, p90_peaks, tmp_dir / "p90.png")
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
