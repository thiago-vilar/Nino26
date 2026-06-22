from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.data.download_ocean_monthly import (
    ORAS5_SINGLE_LEVEL_VARIABLES,
    align_monthly_features_causally,
    latest_complete_oras5_month,
    oras5_request,
    oras5_variable_zarr_path,
    process_oras5_archive,
)


class MonthlyOceanPipelineTests(unittest.TestCase):
    def test_request_groups_each_vertical_kind_without_daily_fields(self) -> None:
        request = oras5_request(2001, "single", list(range(1, 13)))
        self.assertEqual(request["vertical_resolution"], "single_level")
        self.assertEqual(tuple(request["variable"]), ORAS5_SINGLE_LEVEL_VARIABLES)
        self.assertEqual(len(request["month"]), 12)
        self.assertNotIn("day", request)
        self.assertNotIn("time", request)

    def test_latest_complete_month_respects_publication_lag(self) -> None:
        self.assertEqual(latest_complete_oras5_month(pd.Timestamp("2026-06-20")), (2026, 5))
        self.assertEqual(latest_complete_oras5_month(pd.Timestamp("2026-06-10")), (2026, 4))

    def test_causal_alignment_does_not_expose_month_before_release(self) -> None:
        monthly = xr.Dataset(
            {"ohc": ("time", [10.0, 20.0])},
            coords={"time": pd.to_datetime(["2001-01-01", "2001-02-01"])},
            attrs={"source_frequency": "monthly_mean"},
        )
        daily = pd.date_range("2001-02-10", "2001-03-20", freq="D")
        aligned = align_monthly_features_causally(monthly, daily, publication_lag_days=15)
        self.assertTrue(np.isnan(float(aligned["ohc"].sel(time="2001-02-14"))))
        self.assertEqual(float(aligned["ohc"].sel(time="2001-02-15")), 10.0)
        self.assertEqual(float(aligned["ohc"].sel(time="2001-03-15")), 20.0)
        self.assertEqual(aligned.attrs["daily_observation_claim"], "false")

    def test_archive_processing_preserves_twelve_months(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nc_path = root / "single.nc"
            zip_path = root / "single.zip"
            output_root = root / "processed"
            time = pd.date_range("2001-01-01", periods=12, freq="MS")
            y = np.arange(3)
            x = np.arange(5)
            nav_lat = xr.DataArray(np.tile(np.array([-5.0, 0.0, 5.0])[:, None], (1, 5)), dims=("y", "x"))
            nav_lon = xr.DataArray(np.tile(np.array([120.0, 160.0, 200.0, 240.0, 280.0])[None, :], (3, 1)), dims=("y", "x"))
            values = np.ones((12, 3, 5), dtype=np.float32)
            xr.Dataset(
                {variable: (("time", "y", "x"), values * (index + 1)) for index, variable in enumerate(ORAS5_SINGLE_LEVEL_VARIABLES)},
                coords={"time": time, "y": y, "x": x, "nav_lat": nav_lat, "nav_lon": nav_lon},
            ).to_netcdf(nc_path)
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.write(nc_path, arcname=nc_path.name)

            outputs = process_oras5_archive(
                zip_path,
                year=2001,
                months=list(range(1, 13)),
                kind="single",
                output_root=output_root,
            )

            self.assertEqual(len(outputs), len(ORAS5_SINGLE_LEVEL_VARIABLES))
            d20_path = oras5_variable_zarr_path(output_root, 2001, "depth_of_20_c_isotherm")
            with xr.open_zarr(d20_path, consolidated=True) as ds:
                self.assertEqual(ds.sizes["time"], 12)
                self.assertEqual(ds.sizes["lat"], 41)
                self.assertEqual(ds.sizes["lon"], 641)
                self.assertAlmostEqual(float(np.median(np.diff(ds["lat"]))), 0.25)
                self.assertEqual(float(ds["lat"].values[0]), -5.0)
                self.assertEqual(float(ds["lon"].values[0]), 120.0)
                self.assertEqual(ds.attrs["source_frequency"], "monthly_mean")
                self.assertEqual(ds.attrs["temporal_transform"], "none")
                self.assertEqual(ds.attrs["spatial_information_gain"], "none; interpolation only aligns comparison nodes")


if __name__ == "__main__":
    unittest.main()
