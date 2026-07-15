from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import scripts.build_phase7_pacific_cube as builder


def _write_store(root: Path, year: int, dates: pd.DatetimeIndex) -> None:
    depth = np.asarray([0.5, 50.0, 100.0, 150.0, 200.0, 300.0])
    lat = np.arange(4, dtype="float32")
    lon = np.arange(8, dtype="float32")
    shape = (len(dates), len(depth), len(lat), len(lon))
    dataset = xr.Dataset(
        {
            "potential_temperature": (
                ("time", "depth", "lat", "lon"),
                np.full(shape, 20.0, dtype="float32"),
            ),
            "salinity": (
                ("time", "depth", "lat", "lon"),
                np.full(shape, 35.0, dtype="float32"),
            ),
            "sea_surface_height": (
                ("time", "lat", "lon"),
                np.ones((len(dates), len(lat), len(lon)), dtype="float32"),
            ),
        },
        coords={"time": dates, "depth": depth, "lat": lat, "lon": lon},
    )
    path = (
        root
        / str(year)
        / f"glorys12_equatorial_pacific_{year}_daily_0p25.zarr"
    )
    path.parent.mkdir(parents=True)
    dataset.to_zarr(path, mode="w", consolidated=True, zarr_format=2)


def test_phase7_builder_concatenates_daily_before_weekly_and_emits_time_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "glorys12"
    _write_store(source, 2000, pd.date_range("2000-12-25", periods=7, freq="D"))
    _write_store(source, 2001, pd.date_range("2001-01-01", periods=7, freq="D"))
    monkeypatch.setattr(builder, "SOURCE_ROOT", source)
    weekly = builder.build(2000, 2001, spatial_step=2).compute()
    assert weekly.sizes["time"] == 2
    assert weekly.sizes["lat"] == 2
    assert weekly.sizes["lon"] == 4
    assert weekly["complete_week"].dims == ("time",)
    assert bool(weekly["complete_week"].all())
    assert int(weekly["expected_day_count"].min()) == 7
    assert weekly["ssh_m"].notnull().all()


def test_phase7_builder_rejects_daily_calendar_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "glorys12"
    _write_store(source, 2000, pd.date_range("2000-12-25", periods=7, freq="D"))
    _write_store(source, 2001, pd.date_range("2001-01-02", periods=7, freq="D"))
    monkeypatch.setattr(builder, "SOURCE_ROOT", source)
    with pytest.raises(ValueError, match="lacunas"):
        builder.build(2000, 2001, spatial_step=2)
