#!/usr/bin/env python3
"""Organize local Phase 1 daily products into the independent Phase 2 input."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.features.phase3_diagnostics import load_daily_ocean_features


OISST_DAILY = ROOT / "data/processed/parquet/features/nino34_daily_oisst.csv"
OCEAN_ROOT = ROOT / "data/processed/zarr/features/ocean_daily"
OUTPUT_CSV = ROOT / "data/processed/numeric-tables/fase2/nino34_physical_daily.csv"
OUTPUT_ZARR = ROOT / "data/processed/zarr/features/phase2_nino34_physical_daily.zarr"
COVERAGE_CSV = ROOT / "data/processed/numeric-tables/fase2/phase2_daily_input_coverage.csv"

OCEAN_COLUMNS = (
    "d20_nino34_mean_m",
    "thermocline_tilt_m",
    "thermocline_tilt_slope_m_per_degree",
    "ohc_0_100_nino34_j_m2",
    "ohc_0_300_nino34_j_m2",
    "ohc_0_700_nino34_j_m2",
    "ohc_300_700_nino34_j_m2",
    "ssh_nino34_mean_m",
    "wwv_equatorial_pacific_m3",
    "temperature_50m_nino34_c",
    "temperature_100m_nino34_c",
    "temperature_150m_nino34_c",
    "temperature_200m_nino34_c",
    "temperature_300m_nino34_c",
    "temperature_500m_nino34_c",
    "temperature_700m_nino34_c",
    "ocean_source_code",
)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(staging, index=False)
    os.replace(staging, path)


def _coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in ("nino34_ssta", *OCEAN_COLUMNS):
        values = pd.to_numeric(frame[column], errors="coerce")
        valid_dates = pd.DatetimeIndex(frame.loc[values.notna(), "time"])
        missing_internal = 0
        if len(valid_dates):
            missing_internal = len(pd.date_range(valid_dates.min(), valid_dates.max(), freq="D").difference(valid_dates))
        rows.append(
            {
                "variavel": column,
                "inicio_valido": valid_dates.min().date() if len(valid_dates) else None,
                "fim_valido": valid_dates.max().date() if len(valid_dates) else None,
                "dias_validos": int(values.notna().sum()),
                "lacunas_diarias_internas": int(missing_internal),
                "cobertura_percentual_no_eixo": round(100.0 * float(values.notna().mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    if not OISST_DAILY.exists():
        raise FileNotFoundError(f"Execute build-nino34-daily-index primeiro: {OISST_DAILY}")
    oisst = pd.read_csv(OISST_DAILY, parse_dates=["time"], usecols=["time", "nino34_ssta"])
    ocean = load_daily_ocean_features(OCEAN_ROOT)
    missing = sorted(set(OCEAN_COLUMNS) - set(ocean.columns))
    if missing:
        raise ValueError(f"Features oceanicas diarias ausentes: {missing}")
    daily = oisst.merge(ocean[["time", *OCEAN_COLUMNS]], on="time", how="inner", validate="one_to_one")
    daily = daily.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    if daily.empty or daily["time"].duplicated().any():
        raise ValueError("Entrada diaria da Fase 2 vazia ou duplicada")
    daily = daily[["time", "nino34_ssta", *OCEAN_COLUMNS]]
    coverage = _coverage(daily)
    _atomic_csv(daily, OUTPUT_CSV)
    _atomic_csv(coverage, COVERAGE_CSV)

    dataset = xr.Dataset.from_dataframe(daily.set_index("time"))
    dataset.attrs.update(
        artifact="phase2_daily_input",
        frequency="daily",
        source="Phase 1 canonical Zarrs: NOAA OISST + UFS+GLORYS",
        weekly_destination="complete W-SUN weeks only",
    )
    OUTPUT_ZARR.parent.mkdir(parents=True, exist_ok=True)
    staging = OUTPUT_ZARR.with_name(OUTPUT_ZARR.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    dataset.to_zarr(staging, mode="w", consolidated=True, zarr_format=2)
    with xr.open_zarr(staging, consolidated=True) as check:
        if check.sizes.get("time", 0) != len(daily):
            raise ValueError("Zarr diario da Fase 2 perdeu datas")
    if OUTPUT_ZARR.exists():
        shutil.rmtree(OUTPUT_ZARR)
    staging.replace(OUTPUT_ZARR)
    print(f"F2 entrada diaria: {daily.time.min().date()}..{daily.time.max().date()} ({len(daily)} dias)")
    print(f"F2 cobertura por variavel: {COVERAGE_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
