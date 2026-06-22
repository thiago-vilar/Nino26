from __future__ import annotations

import shutil
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

from nino_brasil.data.zarr_store import (
    dataframe_to_zarr,
    netcdf_to_daily_zarr,
    standardize_dataset_to_daily,
    zip_netcdf_to_zarr,
    zarr_to_dataframe,
)


class ZarrStoreTests(unittest.TestCase):
    def test_dataframe_roundtrip_preserves_table_shape(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            path = tmp_dir / "table.zarr"
            frame = pd.DataFrame(
                {
                    "fold": ["fold_00", "fold_01"],
                    "lag_days": [7, 14],
                    "rmse": [1.25, 1.5],
                    "target_time": pd.date_range("2001-01-01", periods=2),
                }
            )

            dataframe_to_zarr(frame, path)
            restored = zarr_to_dataframe(path)

            self.assertEqual(list(restored.columns), list(frame.columns))
            self.assertEqual(len(restored), len(frame))
            self.assertEqual(restored["fold"].tolist(), frame["fold"].tolist())
            self.assertEqual(restored["lag_days"].tolist(), frame["lag_days"].tolist())
            self.assertEqual(pd.Timestamp(restored["target_time"].iloc[0]), frame["target_time"].iloc[0])
        finally:
            shutil.rmtree(tmp_dir)

    def test_netcdf_to_daily_zarr_aggregates_subdaily_variable_alias(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            raw_path = tmp_dir / "era5.nc"
            zarr_path = tmp_dir / "era5_daily.zarr"
            ds = xr.Dataset(
                {"u10": (("time",), np.arange(8, dtype=float))},
                coords={"time": pd.date_range("2001-01-01", periods=8, freq="6h")},
            )
            ds.to_netcdf(raw_path)

            netcdf_to_daily_zarr(
                raw_path,
                zarr_path,
                variables=["10m_u_component_of_wind"],
                variable_aliases={"10m_u_component_of_wind": ["u10"]},
                source_frequency="subdaily",
            )

            restored = xr.open_zarr(zarr_path)
            try:
                self.assertEqual(list(restored.data_vars), ["10m_u_component_of_wind"])
                self.assertEqual(restored.sizes["time"], 2)
                self.assertEqual(restored["10m_u_component_of_wind"].values.tolist(), [1.5, 5.5])
            finally:
                restored.close()
        finally:
            shutil.rmtree(tmp_dir)

    def test_netcdf_to_daily_zarr_rejects_monthly_cache(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            raw_path = tmp_dir / "monthly_source.nc"
            zarr_path = tmp_dir / "monthly_source_daily.zarr"
            ds = xr.Dataset(
                {"so": (("time",), np.array([35.0]))},
                coords={"time": pd.to_datetime(["2001-02-15"])},
            )
            ds.to_netcdf(raw_path)

            with self.assertRaisesRegex(ValueError, "forbids monthly-to-daily"):
                netcdf_to_daily_zarr(
                    raw_path,
                    zarr_path,
                    variables=["salinity"],
                    variable_aliases={"salinity": ["so"]},
                    source_frequency="monthly",
                    daily_start="2001-02-01",
                    daily_end="2001-02-28",
                )
            self.assertFalse(zarr_path.exists())
        finally:
            shutil.rmtree(tmp_dir)

    def test_standardize_rejects_monthly_steps(self) -> None:
        ds = xr.Dataset(
            {"d20": (("time",), np.array([100.0, 120.0, 140.0]))},
            coords={"time": pd.to_datetime(["2001-01-30", "2001-02-27", "2001-03-30"])},
        )

        with self.assertRaisesRegex(ValueError, "forbids monthly-to-daily"):
            standardize_dataset_to_daily(
                ds,
                source_frequency="monthly",
                daily_start="2001-01-01",
                daily_end="2001-03-31",
            )

    def test_standardize_daily_regular_calendar_returns_identity(self) -> None:
        time = pd.date_range("2001-01-01", periods=3, freq="D")
        ds = xr.Dataset({"sst": (("time",), np.arange(3, dtype=float))}, coords={"time": time})

        daily = standardize_dataset_to_daily(ds, source_frequency="daily")

        self.assertEqual(daily.attrs["nino_brasil_daily_transform"], "identity_daily")
        self.assertEqual(daily["time"].values.tolist(), ds["time"].values.tolist())

    def test_zip_netcdf_to_zarr_rejects_path_traversal_member(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            zip_path = tmp_dir / "unsafe.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("../escape.nc", "not a netcdf")

            with self.assertRaises(ValueError):
                zip_netcdf_to_zarr(zip_path, tmp_dir / "extract", tmp_dir / "out.zarr")
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
