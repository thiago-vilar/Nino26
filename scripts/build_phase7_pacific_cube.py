#!/usr/bin/env python3
"""Build the real weekly Pacific spatial input cube for F7/F8.

The builder reads source-separated GLORYS daily stores, selects physically
interpretable levels, concatenates daily data across years *before* one W-SUN
resample, and writes a compact spatial cube.  The 31 scalar F2 predictors are
fused later by the model; they are not broadcast into fake spatial fields.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SOURCE_ROOT = ROOT / "data" / "processed" / "zarr" / "ocean_daily" / "glorys12"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "zarr" / "modeling" / "phase7_pacific_weekly.zarr"


def _annual_store(year: int) -> Path:
    candidates = sorted((SOURCE_ROOT / str(year)).glob("glorys12_equatorial_pacific_*_daily_0p25.zarr"))
    if not candidates:
        raise FileNotFoundError(f"GLORYS diario ausente para {year}: {SOURCE_ROOT / str(year)}")
    return candidates[0]


def _select_channels(dataset: xr.Dataset) -> xr.Dataset:
    required = {"potential_temperature", "salinity", "sea_surface_height"}
    if missing := required.difference(dataset.data_vars):
        raise KeyError(f"GLORYS sem variaveis: {sorted(missing)}")
    temperature = dataset["potential_temperature"]
    salinity = dataset["salinity"]
    channels: dict[str, xr.DataArray] = {
        "thetao_surface_c": temperature.isel(depth=0, drop=True),
        "salinity_surface_psu": salinity.isel(depth=0, drop=True),
        "ssh_m": dataset["sea_surface_height"],
    }
    for depth in (50, 100, 150, 200, 300):
        channels[f"thetao_{depth}m_c"] = temperature.sel(depth=depth, method="nearest", drop=True)
    return xr.Dataset(channels)


def build(start_year: int, end_year: int, *, spatial_step: int) -> xr.Dataset:
    annual: list[xr.Dataset] = []
    paths: list[str] = []
    reference_coordinates: tuple[np.ndarray, np.ndarray] | None = None
    for year in range(start_year, end_year + 1):
        path = _annual_store(year)
        paths.append(str(path))
        selected = _select_channels(xr.open_zarr(path, consolidated=False))
        coordinates = (
            np.asarray(selected.lat.values),
            np.asarray(selected.lon.values),
        )
        if reference_coordinates is None:
            reference_coordinates = coordinates
        elif not all(
            np.array_equal(left, right)
            for left, right in zip(reference_coordinates, coordinates)
        ):
            raise ValueError(f"Grid GLORYS mudou em {year}; regrid implicito recusado.")
        annual.append(selected)
    daily = xr.concat(annual, dim="time", data_vars="minimal", coords="minimal", compat="override")
    daily = daily.sortby("time")
    daily_index = pd.DatetimeIndex(daily.time.values)
    if daily_index.has_duplicates:
        examples = daily_index[daily_index.duplicated(keep=False)][:5]
        raise ValueError(f"Datas GLORYS duplicadas: {list(map(str, examples))}")
    if len(daily_index) > 1 and not np.all(
        np.diff(daily_index.values).astype("timedelta64[D]")
        == np.timedelta64(1, "D")
    ):
        raise ValueError("Serie GLORYS diaria possui lacunas; imputacao silenciosa recusada.")
    if spatial_step > 1:
        daily = daily.coarsen(lat=spatial_step, lon=spatial_step, boundary="trim").mean()
    weekly_raw = daily.resample(time="W-SUN").mean(skipna=True)
    count = daily["ssh_m"].resample(time="W-SUN").count()
    calendar_count = xr.DataArray(
        np.ones(len(daily_index), dtype="uint8"),
        coords={"time": daily.time},
        dims=("time",),
    ).resample(time="W-SUN").sum()
    static_ocean = daily["ssh_m"].notnull().any("time")
    complete_week = ((count >= 7) | ~static_ocean).all(("lat", "lon")) & (
        calendar_count == 7
    )
    # Incomplete temporal weeks remain as audited coordinates but never become
    # model inputs. Static land/mask NaNs do not invalidate an otherwise full
    # ocean week.
    weekly = weekly_raw.where(complete_week)
    weekly["valid_day_count"] = count.astype("int8")
    weekly["expected_day_count"] = calendar_count.astype("int8")
    weekly["complete_week"] = complete_week.astype("int8")
    weekly.attrs.update(
        {
            "phase": 7,
            "role": "real_spatial_pacific_predictors",
            "source": "GLORYS12 daily source-separated stores",
            "daily_concat_before_weekly_resample": True,
            "weekly_anchor": "W-SUN",
            "incomplete_week_policy": (
                "coordinate retained; every spatial predictor set NaN; complete_week=0"
            ),
            "spatial_operation": (
                "native_0p25" if spatial_step == 1 else f"block_mean_{spatial_step}x{spatial_step}"
            ),
            "scalar_fusion": "31 physical F2 variables are a separate named sequence branch",
            "input_stores_json": json.dumps(paths),
        }
    )
    return weekly


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1993)
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="ultimo ano civil completo oficial; 2026 parcial deve ser estudo separado",
    )
    parser.add_argument("--spatial-step", type=int, default=4, help="4 => block mean 1 degree")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.start_year > args.end_year or args.spatial_step < 1:
        parser.error("periodo/spatial-step invalido")
    output = args.output
    if output is None:
        output = (
            DEFAULT_OUTPUT
            if (args.start_year, args.end_year) == (1993, 2025)
            else DEFAULT_OUTPUT.with_name(
                f"phase7_pacific_weekly_{args.start_year}_{args.end_year}_smoke.zarr"
            )
        )
    if args.validate_only:
        if not output.exists():
            raise FileNotFoundError(output)
        existing = xr.open_zarr(output, consolidated=None)
        required = {
            "ssh_m",
            "thetao_surface_c",
            "valid_day_count",
            "expected_day_count",
            "complete_week",
        }
        if missing := required.difference(existing.data_vars):
            raise ValueError(f"Cubo F7 incompleto: {sorted(missing)}")
        if pd.DatetimeIndex(existing.time.values).has_duplicates:
            raise ValueError("Cubo F7 tem timestamps duplicados.")
        weekly_index = pd.DatetimeIndex(existing.time.values)
        if len(weekly_index) > 1 and not np.all(np.diff(weekly_index.values) == np.timedelta64(7, "D")):
            raise ValueError("Cubo F7 nao e uma serie W-SUN continua.")
        if existing["complete_week"].dims != ("time",):
            raise ValueError("complete_week F7 deve ser um gate temporal unidimensional.")
        incomplete = existing["complete_week"] == 0
        if bool(existing["ssh_m"].where(incomplete).notnull().any().compute()):
            raise ValueError("Semana F7 incompleta contem preditores utilizaveis.")
        print(f"[ok] {output} {dict(existing.sizes)}")
        return 0
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} ja existe; use --overwrite conscientemente.")
    dataset = build(args.start_year, args.end_year, spatial_step=args.spatial_step)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        quarantine = ROOT / "data/quarantine/phase7_cube" / stamp / temporary.name
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temporary), str(quarantine))
        print(f"[archive] staging F7 incompleto -> {quarantine.relative_to(ROOT)}")
    for variable in dataset.variables:
        # Compressor/serializer belongs to the source store, not to scientific
        # lineage, and cannot be copied across Zarr generations.
        dataset[variable].encoding = {}
    chunk_plan = {
        "time": min(52, dataset.sizes["time"]),
        "lat": min(16, dataset.sizes["lat"]),
        "lon": min(64, dataset.sizes["lon"]),
    }
    dataset = dataset.chunk(chunk_plan)
    encoding = {
        name: {
            "chunks": (
                chunk_plan["time"],
                chunk_plan["lat"],
                chunk_plan["lon"],
            )
        }
        for name in dataset.data_vars
        if dataset[name].ndim == 3
    }
    try:
        dataset.to_zarr(
            temporary,
            mode="w",
            consolidated=True,
            encoding=encoding,
            zarr_format=2,
        )
    except Exception:
        if temporary.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            quarantine = ROOT / "data/quarantine/phase7_cube" / stamp / temporary.name
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temporary), str(quarantine))
        raise
    if output.exists():
        backup = output.with_name(f"{output.name}.previous")
        if backup.exists():
            raise FileExistsError(f"backup preexistente: {backup}")
        os.replace(output, backup)
    os.replace(temporary, output)
    print(
        f"[F7-input] {output} | weeks={dataset.sizes['time']} | "
        f"grid={dataset.sizes['lat']}x{dataset.sizes['lon']} | "
        f"channels={len(dataset.data_vars)-3}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
