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
from nino_brasil.maps.plot_choropleths import export_choropleth_table
from nino_brasil.maps.plot_pixel_maps import export_pixel_table, save_pixel_map
from nino_brasil.data.download_noaa_psl import (
    build_noaa_psl_nino34_peak_reference,
    parse_noaa_psl_nino34_anom,
)

from scripts.model_pipeline import (
    importance_top20_by_lag_model,
    metrics_summary,
    xai_method_agreement,
)


class NumericOutputTests(unittest.TestCase):
    def test_pixel_map_exports_csv_before_png(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            da = xr.DataArray(
                np.array([[1.23456, np.nan], [3.0, 4.0]]),
                coords={"lat": [-5.0, 0.0], "lon": [190.0, 191.0]},
                dims=("lat", "lon"),
                name="skill",
            )
            png_path = tmp_dir / "skill.png"

            save_pixel_map(da, png_path, "Skill", value_name="skill")

            csv_path = png_path.with_suffix(".csv")
            self.assertTrue(png_path.exists())
            self.assertTrue(csv_path.exists())
            table = pd.read_csv(csv_path)
            self.assertEqual(list(table.columns), ["lat", "lon", "skill"])
            self.assertEqual(len(table), 3)
        finally:
            shutil.rmtree(tmp_dir)

    def test_choropleth_export_removes_geometry_and_adds_rank(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            gdf = gpd.GeoDataFrame(
                {"NM_UF": ["A", "B"], "skill": [0.2, 0.8]},
                geometry=[Point(0, 0), Point(1, 1)],
                crs="EPSG:4326",
            )
            path = export_choropleth_table(gdf, "skill", tmp_dir / "choropleth.csv")
            table = pd.read_csv(path)

            self.assertEqual(list(table.columns), ["NM_UF", "skill", "rank"])
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

    def test_model_pipeline_numeric_summaries(self) -> None:
        metrics = pd.DataFrame(
            {
                "model": ["ridge", "ridge", "lightgbm"],
                "task": ["regression", "regression", "classification"],
                "lag_days": [7, 7, 14],
                "season": ["all", "DJF", "all"],
                "region": ["precip", "precip", "precip"],
                "rmse": [1.0, 1.2, np.nan],
                "brier": [np.nan, np.nan, 0.2],
                "roc_auc": [np.nan, np.nan, 0.7],
            }
        )
        summary = metrics_summary(metrics)
        self.assertIn("metric", summary.columns)
        self.assertIn("mean", summary.columns)

        importances = pd.DataFrame(
            {
                "fold": ["f0", "f0", "f0", "f0"],
                "lag_days": [7, 7, 7, 7],
                "model": ["lightgbm"] * 4,
                "task": ["classification"] * 4,
                "region": ["precip"] * 4,
                "method": ["permutation", "permutation", "shap", "shap"],
                "feature": ["sst", "slp", "sst", "slp"],
                "group": ["ocean", "atmosphere", "ocean", "atmosphere"],
                "importance_mean": [0.4, 0.1, 0.3, 0.2],
                "importance_std": [0.01, 0.01, np.nan, np.nan],
                "importance_value": [0.4, 0.1, 0.3, 0.2],
            }
        )
        top20 = importance_top20_by_lag_model(importances)
        agreement = xai_method_agreement(importances)

        self.assertEqual(top20.iloc[0]["feature"], "sst")
        self.assertEqual(int(agreement["n_features"].iloc[0]), 2)

    def test_noaa_psl_nino34_parser_and_reference_peaks(self) -> None:
        text = "\n".join(
            [
                "        2000        2001",
                " 2000  -0.10   0.10   0.20   0.60   0.80   1.10   1.30   1.70   2.10   2.20   1.90   1.60",
                " 2001   1.20   0.80   0.40   0.10  -0.10 -99.99 -99.99 -99.99 -99.99 -99.99 -99.99 -99.99",
                "   -99.99",
                "  Nino Anom 3.4 Index  using NOAA ERSST v6 from NCEI",
            ]
        )

        monthly = parse_noaa_psl_nino34_anom(text)
        peaks = build_noaa_psl_nino34_peak_reference(monthly, min_duration_months=5)

        self.assertEqual(len(monthly), 24)
        self.assertTrue(monthly["nino34_anom_c"].isna().any())
        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks.iloc[0]["peak_class"], "super_el_nino")


if __name__ == "__main__":
    unittest.main()
