#!/usr/bin/env python3
"""Atualiza o cache atmosferico ERA5 diario usado nas Fases 3/4.

O cache historico pode vir de Zarr anual por variavel, enquanto atualizacoes
operacionais recentes podem existir apenas como Zarr mensal por tipo
single/pressure. Este script aceita os dois formatos e materializa as quatro
colunas consumidas pela Fase 3.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import fase4_features as F  # noqa: E402

COLUMNS = [
    "atm_10m_u_component_of_wind",
    "atm_mean_sea_level_pressure",
    "atm_total_column_water_vapour",
    "atm_u_component_of_wind_850hpa",
]


def _monthly_kind_year(year: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for month in range(1, 13):
        single = ROOT / f"data/processed/zarr/era5/single_levels/{year}/era5_single_nino34_{year}{month:02d}_daily.zarr"
        pressure = ROOT / f"data/processed/zarr/era5/pressure_levels/{year}/era5_pressure_nino34_{year}{month:02d}_daily.zarr"
        if not single.exists() or not pressure.exists():
            continue

        ds = xr.open_zarr(single)
        try:
            frame = pd.DataFrame(
                {
                    "atm_10m_u_component_of_wind": ds["10m_u_component_of_wind"]
                    .mean(dim=["latitude", "longitude"])
                    .compute()
                    .to_series(),
                    "atm_mean_sea_level_pressure": ds["mean_sea_level_pressure"]
                    .mean(dim=["latitude", "longitude"])
                    .compute()
                    .to_series(),
                    "atm_total_column_water_vapour": ds["total_column_water_vapour"]
                    .mean(dim=["latitude", "longitude"])
                    .compute()
                    .to_series(),
                }
            )
        finally:
            ds.close()

        dp = xr.open_zarr(pressure)
        try:
            frame["atm_u_component_of_wind_850hpa"] = (
                dp["u_component_of_wind"]
                .sel(pressure_level=850)
                .mean(dim=["latitude", "longitude"])
                .compute()
                .to_series()
            )
        finally:
            dp.close()
        rows.append(frame)

    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(rows).sort_index()


def _annual_variable_year(year: int) -> pd.DataFrame:
    cols = {
        "atm_10m_u_component_of_wind": F.era5_box_index(
            "10m_u_component_of_wind", kind="single", region="nino34", years=[year]
        ),
        "atm_mean_sea_level_pressure": F.era5_box_index(
            "mean_sea_level_pressure", kind="single", region="nino34", years=[year]
        ),
        "atm_total_column_water_vapour": F.era5_box_index(
            "total_column_water_vapour", kind="single", region="nino34", years=[year]
        ),
        "atm_u_component_of_wind_850hpa": F.era5_box_index(
            "u_component_of_wind", kind="pressure", region="nino34", level=850, years=[year]
        ),
    }
    cols = {k: v for k, v in cols.items() if len(v)}
    if len(cols) != len(COLUMNS):
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(cols, axis=1).sort_index()


def build_cache(start_year: int, end_year: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        frame = _annual_variable_year(year)
        source = "annual-variable"
        if frame.empty:
            frame = _monthly_kind_year(year)
            source = "monthly-kind"
        if frame.empty:
            print(f"[skip] ERA5 atmo cache {year}: sem Zarr compativel")
            continue
        frame = frame[COLUMNS]
        frame.index = pd.to_datetime(frame.index).normalize()
        frame = frame[~frame.index.duplicated(keep="first")]
        pieces.append(frame)
        print(f"[ok] ERA5 atmo cache {year}: {len(frame)} dias ({source})")

    if not pieces:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.concat(pieces).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "time"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1981)
    parser.add_argument("--end-year", type=int, default=pd.Timestamp.today().year)
    parser.add_argument(
        "--output",
        default="data/processed/parquet/features/era5_nino34_atmo_cache.csv",
    )
    args = parser.parse_args(argv)

    out = build_cache(args.start_year, args.end_year)
    if out.empty:
        raise RuntimeError("nenhum dado ERA5 atmosferico encontrado para o cache")
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output)
    print(f"[ok] {output.relative_to(ROOT)} {out.index.min().date()}..{out.index.max().date()} rows={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
