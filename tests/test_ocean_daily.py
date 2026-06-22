from __future__ import annotations

import calendar
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.data.download_ocean_daily import (
    GLORYS_DATASET_ID,
    _regrid_ufs_to_canonical_grid,
    build_ocean_daily_features,
    glorys_operational_commands,
    glorys_subset_command,
    process_glorys_year,
    ufs_ocean_members,
)


class DailyOceanPipelineTests(unittest.TestCase):
    def test_glorys_groups_variables_domain_and_year_in_one_request(self) -> None:
        command = glorys_subset_command(year=2001, output_root=Path("raw"))

        self.assertIn("copernicusmarine", Path(command[0]).name)
        self.assertEqual(command[1], "subset")
        self.assertEqual(command.count("--dataset-id"), 1)
        self.assertIn(GLORYS_DATASET_ID, command)
        self.assertEqual(command.count("--variable"), 3)
        self.assertIn("thetao", command)
        self.assertIn("so", command)
        self.assertIn("zos", command)
        self.assertEqual(command[command.index("--minimum-longitude") + 1], "119.9")
        self.assertEqual(command[command.index("--maximum-longitude") + 1], "280.1")
        self.assertEqual(command[command.index("--maximum-depth") + 1], "800.0")
        self.assertEqual(command[command.index("--start-datetime") + 1], "2001-01-01T00:00:00")
        self.assertEqual(command[command.index("--end-datetime") + 1], "2001-12-31T23:59:59")

    def test_operational_tail_uses_three_source_datasets(self) -> None:
        commands = glorys_operational_commands(
            start_date="2026-05-27",
            end_date="2026-06-19",
            output_root=Path("raw"),
        )
        self.assertEqual(set(commands), {"thetao", "so", "zos"})
        for variable, command in commands.items():
            self.assertEqual(command.count("--variable"), 1)
            self.assertEqual(command[command.index("--variable") + 1], variable)
            self.assertEqual(command[command.index("--maximum-depth") + 1], "800.0")

    def test_ufs_member_selection_requires_every_real_day(self) -> None:
        year = 1984
        names = [
            f"{year}/{timestamp:%Y%m%d}12/ctrl/ocn.ana.{timestamp:%Y%m%d}12.nc"
            for timestamp in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        ]
        selected = ufs_ocean_members(names, year)
        self.assertEqual(len(selected), 366)
        self.assertEqual(len(selected), 366 if calendar.isleap(year) else 365)

        with self.assertRaisesRegex(ValueError, "expected 366"):
            ufs_ocean_members(names[:-1], year)

    def test_glorys_processing_and_daily_features_remain_daily(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.zarr"
            processed = root / "processed.zarr"
            features = root / "features.zarr"
            time = pd.date_range("2001-01-01", periods=3, freq="D")
            depth = np.array([0.0, 50.0, 100.0, 200.0, 300.0, 500.0, 700.0])
            lat = np.arange(-61, 62, dtype=float) / 12.0
            lon = np.arange(1439, 3362, dtype=float) / 12.0
            profile = 27.0 - 0.05 * depth
            thetao = np.broadcast_to(
                profile[None, :, None, None],
                (time.size, depth.size, lat.size, lon.size),
            ).astype(np.float32)
            so = np.full_like(thetao, 35.0)
            zos = np.zeros((time.size, lat.size, lon.size), dtype=np.float32)
            xr.Dataset(
                {
                    "thetao": (("time", "depth", "latitude", "longitude"), thetao),
                    "so": (("time", "depth", "latitude", "longitude"), so),
                    "zos": (("time", "latitude", "longitude"), zos),
                },
                coords={"time": time, "depth": depth, "latitude": lat, "longitude": lon},
            ).to_zarr(source, mode="w", consolidated=True, zarr_format=2)

            process_glorys_year(source, processed)
            build_ocean_daily_features(processed, features)

            with xr.open_zarr(processed, consolidated=True) as ds:
                self.assertEqual(ds.sizes["time"], 3)
                self.assertAlmostEqual(float(np.median(np.diff(ds["lat"]))), 0.25, places=5)
                self.assertEqual(float(ds["lat"].values[0]), -5.0)
                self.assertEqual(float(ds["lat"].values[-1]), 5.0)
                self.assertEqual(float(ds["lon"].values[0]), 120.0)
                self.assertEqual(float(ds["lon"].values[-1]), 280.0)
                self.assertEqual(ds.attrs["temporal_transform"], "none")
            with xr.open_zarr(features, consolidated=True) as ds:
                self.assertEqual(ds.sizes["time"], 3)
                self.assertIn("d20_nino34_mean_m", ds)
                self.assertIn("ohc_300_700_nino34_j_m2", ds)
                self.assertIn("wwv_equatorial_pacific_m3", ds)
                self.assertIn("ocean_source_code", ds)
                np.testing.assert_allclose(ds["ocean_source_code"].values, 2.0)
                self.assertTrue(np.all(np.isfinite(ds["d20_nino34_mean_m"])))
                self.assertEqual(ds.attrs["source_frequency"], "daily")

    def test_ufs_is_interpolated_to_the_same_canonical_0p25_grid(self) -> None:
        source = xr.Dataset(
            {"sea_surface_height": (("time", "lat", "lon"), np.ones((1, 11, 161), dtype=np.float32))},
            coords={
                "time": pd.DatetimeIndex(["1981-01-01"]),
                "lat": np.arange(-5.0, 6.0, 1.0),
                "lon": np.arange(120.0, 281.0, 1.0),
            },
        )
        regridded = _regrid_ufs_to_canonical_grid(source)
        self.assertEqual(regridded.sizes["lat"], 41)
        self.assertEqual(regridded.sizes["lon"], 641)
        self.assertAlmostEqual(float(np.median(np.diff(regridded["lat"]))), 0.25)
        self.assertEqual(regridded.attrs["spatial_information_gain"], "none; interpolation only aligns comparison nodes")


if __name__ == "__main__":
    unittest.main()
