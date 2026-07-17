"""Regressões das correções de atualização da Fase 1 (jul/2026).

Cobre três defeitos observados em produção:
1. ``download_cds`` usava ``pd`` sem importar pandas (NameError na cauda ERA5);
2. o fallback de regrid sem xesmf interpolava fonte 0..360 numa grade alvo
   -180..180 e gravava o ano inteiro como NaN;
3. o check de cobertura exigia o ano civil completo do OISST 1981, cuja série
   real começa em 1981-09-01, forçando re-download infinito.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT))

from nino_brasil.data.regrid import regrid_dataset


def test_era5_expected_month_bounds_uses_pandas() -> None:
    from nino_brasil.data.download_cds import _era5_expected_month_bounds

    start, end = _era5_expected_month_bounds(2025, 1)
    assert start == pd.Timestamp("2025-01-01")
    assert end == pd.Timestamp("2025-01-31")


def test_era5_expected_month_bounds_caps_partial_current_month() -> None:
    from nino_brasil.data.download_cds import _era5_expected_month_bounds

    today = pd.Timestamp.today().normalize()
    start, end = _era5_expected_month_bounds(today.year, today.month)
    assert start == today.replace(day=1)
    assert end <= today - pd.Timedelta(days=7)


def test_regrid_fallback_converts_0_360_source_to_negative_target() -> None:
    lon = np.arange(0.0, 360.0, 1.0)
    lat = np.arange(-40.0, 41.0, 1.0)
    values = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (lat.size, lon.size)).copy()
    source = xr.Dataset(
        {"sst": (("lat", "lon"), values)},
        coords={"lat": lat, "lon": lon},
    )
    target = xr.Dataset(coords={"lat": np.arange(-35.0, 8.0, 0.25), "lon": np.arange(-170.0, -29.75, 0.25)})
    out = regrid_dataset(source, target)
    assert int(out["sst"].notnull().sum()) > 0
    assert bool(out["sst"].notnull().all())


def test_expected_daily_bounds_respects_oisst_real_start() -> None:
    from scripts.curate_and_resume_downloads import expected_daily_bounds

    start_1981, end_1981 = expected_daily_bounds(1981, "noaa_oisst")
    assert start_1981 == pd.Timestamp("1981-09-01")
    assert end_1981 == pd.Timestamp("1981-12-31")

    start_1982, _ = expected_daily_bounds(1982, "noaa_oisst")
    assert start_1982 == pd.Timestamp("1982-01-01")

    chirps_start, _ = expected_daily_bounds(1981, "chirps")
    assert chirps_start == pd.Timestamp("1981-01-01")


def test_era5_accumulated_zarr_with_sum_contract_is_reused(tmp_path: Path) -> None:
    from nino_brasil.data.download_cds import _zarr_daily_values_complete, _zarr_valid

    path = tmp_path / "surface_latent_heat_flux.zarr"
    time = pd.date_range("2026-01-01", "2026-01-31", freq="D")
    dataset = xr.Dataset(
        {
            "surface_latent_heat_flux": (
                ("time", "latitude", "longitude"),
                np.ones((len(time), 1, 1), dtype="float32"),
            )
        },
        coords={"time": time, "latitude": [0.0], "longitude": [200.0]},
        attrs={"nino_brasil_daily_aggregation": "sum"},
    )
    dataset.to_zarr(path, mode="w", consolidated=True, zarr_format=2)

    assert _zarr_valid(path, expected_variable="surface_latent_heat_flux")
    assert _zarr_daily_values_complete(
        path,
        variable="surface_latent_heat_flux",
        expected_start=time[0],
        expected_end=time[-1],
    )


def test_era5_fast_inventory_rejects_calendar_gap_without_scanning_values(tmp_path: Path) -> None:
    from nino_brasil.data.download_cds import _zarr_daily_values_complete

    path = tmp_path / "calendar_gap.zarr"
    time = pd.DatetimeIndex(["2026-01-01", "2026-01-03"])
    xr.Dataset(
        {"temperature": (("time", "latitude", "longitude"), np.full((2, 1, 1), np.nan))},
        coords={"time": time, "latitude": [0.0], "longitude": [200.0]},
    ).to_zarr(path, mode="w", consolidated=True, zarr_format=2)

    assert not _zarr_daily_values_complete(
        path,
        variable="temperature",
        expected_start=pd.Timestamp("2026-01-01"),
        expected_end=pd.Timestamp("2026-01-03"),
    )
