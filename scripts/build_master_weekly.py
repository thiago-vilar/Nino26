#!/usr/bin/env python3
"""Build and audit the Phase 2 weekly Pacific master.

Outputs are atomic and include the compatibility/raw master (all contracted physical
variables), an explicitly versioned source-adjusted companion, the variable
contract, coverage/validation/seam tables, CTD validation, and a SHA256 run
manifest.  ``ocean_source_code`` is provenance metadata and never one of the
contracted physical predictors.

Examples
--------
python scripts/build_master_weekly.py --era5-years 1981:2026 --strict
python scripts/build_master_weekly.py --validate-only --strict
python scripts/build_master_weekly.py --dry-run --era5-years 1981:2026
"""
from __future__ import annotations

import argparse
import glob
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nino_brasil.data.audit import AuditLog, sha256_file  # noqa: E402
from nino_brasil.data.phase2_master import (  # noqa: E402
    ATMOSPHERIC_COLUMNS,
    OCEAN_COLUMNS,
    PHYSICAL_COLUMNS,
    coverage_audit,
    day_of_year_anomaly,
    normalize_era5_daily_units,
    seam_audit,
    source_aware_ocean_adjustment,
    validate_master,
    variable_contract_frame,
    vector_wind_stress_x,
    weekly_source_mode,
)


FEAT = ROOT / "data/processed/parquet/features"
STATS = ROOT / "data/processed/parquet/statistics"
ERA5 = ROOT / "data/processed/zarr/era5"
CTD = ROOT / "data/processed/zarr/ctd_noaa/wod"
DEFAULT_PHYSICAL = FEAT / "nino34_physical_signal.csv"
DEFAULT_ERA5_CACHE = FEAT / "era5_nino34_daily_cache.parquet"
DEFAULT_MASTER = FEAT / "nino34_master_weekly.csv"
DEFAULT_MASTER_ZARR = ROOT / "data/processed/zarr/features/nino34_master_weekly.zarr"
DEFAULT_ADJUSTED = FEAT / "nino34_master_weekly_source_adjusted_v1.csv"
DEFAULT_AUDIT = STATS / "phase2_master_audit.csv"
DEFAULT_ADJUSTED_AUDIT = STATS / "phase2_master_source_adjusted_v1_audit.csv"
DEFAULT_VALIDATION = STATS / "phase2_master_validation.csv"
DEFAULT_SEAM = STATS / "phase2_ocean_source_seam_audit.csv"
DEFAULT_CONTRACT = STATS / "phase2_variable_contract.csv"
DEFAULT_CTD = STATS / "phase2_ctd_validation.csv"
DEFAULT_MANIFEST = STATS / "phase2_master_run_manifest.json"
DEFAULT_LEDGER = ROOT / "data/audit/ledger.jsonl"

OCEAN_INPUT_TO_OUTPUT = {
    "nino34_ssta": "nino34_ssta",
    "d20_nino34_mean_m": "d20_m",
    "thermocline_tilt_m": "tilt_m",
    "thermocline_tilt_slope_m_per_degree": "tilt_slope",
    "ohc_0_100_nino34_j_m2": "ohc_0_100",
    "ohc_0_300_nino34_j_m2": "ohc_0_300",
    "ohc_0_700_nino34_j_m2": "ohc_0_700",
    "ohc_300_700_nino34_j_m2": "ohc_300_700",
    "ssh_nino34_mean_m": "ssh_m",
    "wwv_equatorial_pacific_m3": "wwv",
    "temperature_50m_nino34_c": "t50m",
    "temperature_100m_nino34_c": "t100m",
    "temperature_150m_nino34_c": "t150m",
    "temperature_200m_nino34_c": "t200m",
    "temperature_300m_nino34_c": "t300m",
    "temperature_500m_nino34_c": "t500m",
    "temperature_700m_nino34_c": "t700m",
}
ERA5_SINGLE = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "mslp",
    "total_column_water_vapour": "tcwv",
    "surface_latent_heat_flux": "slhf",
    "surface_sensible_heat_flux": "sshf",
    "surface_net_solar_radiation": "ssr",
    "surface_net_thermal_radiation": "str",
}
ERA5_PRESSURE = {
    "u_component_of_wind": [("u850", 850.0), ("u200", 200.0)],
    "vertical_velocity": [("omega850", 850.0), ("omega500", 500.0)],
    "divergence": [("div850", 850.0)],
}
ATMO_RAW_COLUMNS = (
    "u10",
    "v10",
    "mslp",
    "tcwv",
    "slhf",
    "sshf",
    "ssr",
    "str",
    "u850",
    "u200",
    "omega850",
    "omega500",
    "div850",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_year_range(value: str) -> range:
    try:
        start_text, end_text = value.split(":", maxsplit=1)
        start, end = int(start_text), int(end_text)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError("use START:END, for example 1981:2026") from exc
    if start > end or start < 1900 or end > 2100:
        raise argparse.ArgumentTypeError(f"invalid year range: {value}")
    return range(start, end + 1)


def _open_zarr(path: Path | str):
    import xarray as xr

    try:
        return xr.open_zarr(path)
    except Exception:
        return xr.open_zarr(path, consolidated=False)


def _box_mean(array):
    return array.mean([dimension for dimension in array.dims if dimension != "time"])


def _atomic_to_csv(frame: pd.DataFrame, path: Path, *, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=index)
    os.replace(temporary, path)


def _atomic_to_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_weekly_zarr(frame: pd.DataFrame, path: Path) -> None:
    import shutil
    dataset = xr.Dataset.from_dataframe(frame)
    dataset.attrs.update(frequency="weekly", weekly_anchor="W-SUN", canonical=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    dataset.to_zarr(staging, mode="w", consolidated=True, zarr_format=2)
    check = xr.open_zarr(staging, consolidated=True)
    try:
        if check.sizes.get("week_ending_sunday", 0) != len(frame):
            raise ValueError("Zarr semanal não preservou toda a grade W-SUN")
    finally:
        check.close()
    if path.exists():
        shutil.rmtree(path)
    staging.replace(path)


def _monthly_kind_frame(year: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for month in range(1, 13):
        single_path = ERA5 / f"single_levels/{year}/era5_single_nino34_{year}{month:02d}_daily.zarr"
        pressure_path = ERA5 / f"pressure_levels/{year}/era5_pressure_nino34_{year}{month:02d}_daily.zarr"
        if not single_path.exists() or not pressure_path.exists():
            continue
        single = _open_zarr(single_path)
        pressure = _open_zarr(pressure_path)
        try:
            frame = pd.DataFrame(
                {
                    output: _box_mean(single[variable]).to_pandas()
                    for variable, output in ERA5_SINGLE.items()
                    if variable in single
                }
            )
            level_name = "pressure_level" if "pressure_level" in pressure.coords else "level"
            for variable, levels in ERA5_PRESSURE.items():
                if variable not in pressure:
                    continue
                for output, level in levels:
                    frame[output] = _box_mean(pressure[variable].sel({level_name: level})).to_pandas()
            rows.append(frame)
        finally:
            single.close()
            pressure.close()
    if not rows:
        return pd.DataFrame(columns=ATMO_RAW_COLUMNS)
    frame = pd.concat(rows).sort_index()
    frame.index = pd.to_datetime(frame.index)
    return frame[~frame.index.duplicated(keep="last")]


def _annual_variable_series(years: range) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for variable, output in ERA5_SINGLE.items():
        parts: list[pd.Series] = []
        for year in years:
            paths = glob.glob(str(ERA5 / f"single_levels/{year}/{variable}/*nino34*.zarr"))
            if not paths:
                continue
            for path in sorted(paths):
                dataset = _open_zarr(path)
                try:
                    name = str(next(iter(dataset.data_vars)))
                    parts.append(_box_mean(dataset[name]).to_pandas())
                finally:
                    dataset.close()
        if parts:
            columns[output] = pd.concat(parts).sort_index()
            print(f"  [era5] {output}: {len(columns[output])} dias")
    for variable, levels in ERA5_PRESSURE.items():
        pieces: dict[str, list[pd.Series]] = {output: [] for output, _ in levels}
        for year in years:
            paths = glob.glob(str(ERA5 / f"pressure_levels/{year}/{variable}/*nino34*.zarr"))
            if not paths:
                continue
            for path in sorted(paths):
                dataset = _open_zarr(path)
                try:
                    name = str(next(iter(dataset.data_vars)))
                    level_name = "pressure_level" if "pressure_level" in dataset.coords else "level"
                    for output, level in levels:
                        pieces[output].append(_box_mean(dataset[name].sel({level_name: level})).to_pandas())
                finally:
                    dataset.close()
        for output, _ in levels:
            if pieces[output]:
                columns[output] = pd.concat(pieces[output]).sort_index()
                print(f"  [era5] {output}: {len(columns[output])} dias")
    if not columns:
        return pd.DataFrame(columns=ATMO_RAW_COLUMNS)
    frame = pd.DataFrame(columns)
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def extract_era5(
    years: range,
    *,
    cache_path: Path = DEFAULT_ERA5_CACHE,
    force_reextract: bool = False,
) -> pd.DataFrame:
    """Extract raw daily box means; unit/sign conversion occurs after caching."""
    # A F2 lê diretamente os Zarrs canônicos da F1. O antigo Parquet diário
    # duplicava a série e podia ficar dessincronizado com a cauda operacional.
    frames: list[pd.DataFrame] = []
    for year in years:
        annual = _annual_variable_series(range(year, year + 1))
        monthly = _monthly_kind_frame(year)
        if annual.empty:
            replacement = monthly
        elif monthly.empty:
            replacement = annual
        else:
            replacement = annual.combine_first(monthly)
        if not replacement.empty:
            frames.append(replacement)
    if not frames:
        return pd.DataFrame(columns=ATMO_RAW_COLUMNS)
    direct = pd.concat(frames).sort_index()
    direct = direct[~direct.index.duplicated(keep="last")]
    print(f"  [era5] leitura direta dos Zarrs F1: {direct.shape}")
    return direct.loc[f"{min(years)}-01-01":f"{max(years)}-12-31"].sort_index()


def _atmospheric_weekly(era5_raw: pd.DataFrame) -> pd.DataFrame:
    if era5_raw.empty:
        return pd.DataFrame(columns=ATMOSPHERIC_COLUMNS, dtype=float)
    missing = sorted(set(ATMO_RAW_COLUMNS) - set(era5_raw.columns))
    if missing:
        raise ValueError(f"ERA5 cache lacks required columns: {missing}")
    daily = normalize_era5_daily_units(era5_raw[list(ATMO_RAW_COLUMNS)])
    tau = vector_wind_stress_x(daily["u10"], daily["v10"])
    daily["tau_x_anom"] = day_of_year_anomaly(pd.Series(tau, index=daily.index, name="tau_x"))
    for column in ATMO_RAW_COLUMNS:
        daily[f"{column}_anom"] = day_of_year_anomaly(daily[column])
    keep = ["tau_x_anom", *[f"{column}_anom" for column in ATMO_RAW_COLUMNS]]
    weekly = daily[keep].resample("W-SUN").mean()
    return weekly.reindex(columns=ATMOSPHERIC_COLUMNS)


def build_products(
    physical_path: Path,
    era5_years: range,
    *,
    era5_cache: Path = DEFAULT_ERA5_CACHE,
    ocean_only: bool = False,
    force_reextract_era5: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    use_columns = ["time", *OCEAN_INPUT_TO_OUTPUT, "ocean_source_code"]
    physical = pd.read_csv(physical_path, parse_dates=["time"], usecols=use_columns).set_index("time").sort_index()
    if physical.index.duplicated().any():
        raise ValueError(f"physical input contains {int(physical.index.duplicated().sum())} duplicate dates")
    ocean_daily = physical[list(OCEAN_INPUT_TO_OUTPUT)].rename(columns=OCEAN_INPUT_TO_OUTPUT)
    source_daily = pd.to_numeric(physical["ocean_source_code"], errors="coerce")
    raw_ocean_weekly = ocean_daily.resample("W-SUN").mean()
    adjusted_ocean_daily = source_aware_ocean_adjustment(ocean_daily, source_daily)
    adjusted_ocean_weekly = adjusted_ocean_daily.resample("W-SUN").mean()
    source_weekly = weekly_source_mode(source_daily)

    if ocean_only:
        atmospheric_weekly = pd.DataFrame(index=raw_ocean_weekly.index, columns=ATMOSPHERIC_COLUMNS, dtype=float)
    else:
        era5_raw = extract_era5(era5_years, cache_path=era5_cache, force_reextract=force_reextract_era5)
        atmospheric_weekly = _atmospheric_weekly(era5_raw)

    raw = raw_ocean_weekly.join(atmospheric_weekly, how="outer")
    adjusted = adjusted_ocean_weekly.join(atmospheric_weekly, how="outer")
    if raw.empty:
        raise ValueError("weekly master is empty")
    full_index = pd.date_range(raw.index.min(), raw.index.max(), freq="W-SUN", name="week_ending_sunday")
    raw = raw.reindex(full_index)
    adjusted = adjusted.reindex(full_index)
    source_weekly = source_weekly.reindex(full_index)
    for frame in (raw, adjusted):
        frame["ocean_source_code"] = source_weekly
        frame.index.name = "week_ending_sunday"
    raw = raw.reindex(columns=[*PHYSICAL_COLUMNS, "ocean_source_code"])
    adjusted = adjusted.reindex(columns=[*PHYSICAL_COLUMNS, "ocean_source_code"])
    seam = seam_audit(raw[list(OCEAN_COLUMNS)], adjusted[list(OCEAN_COLUMNS)], source_weekly)
    return raw, adjusted, seam


def build(era5_years: range, ocean_only: bool = False) -> pd.DataFrame:
    """Backwards-compatible API returning the raw/compatibility master."""
    raw, _, _ = build_products(DEFAULT_PHYSICAL, era5_years, ocean_only=ocean_only)
    return raw


def validate(master: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible validation API."""
    return validate_master(master)


def audit(master: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible coverage audit API."""
    return coverage_audit(master, representation="raw_compatibility")


def ctd_validation(master: pd.DataFrame, years: range, *, ctd_root: Path = CTD) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    import xarray as xr

    for year in years:
        path = ctd_root / str(year) / f"wod_ctd_{year}.zarr"
        if not path.exists():
            continue
        dataset = xr.open_zarr(path, consolidated=False)
        try:
            if "thermocline_depth" not in dataset:
                continue
            depths = np.asarray(dataset["thermocline_depth"].values, dtype=float)
            depths = depths[np.isfinite(depths)]
        finally:
            dataset.close()
        reanalysis = pd.to_numeric(master.loc[master.index.year == year, "d20_m"], errors="coerce").dropna()
        ctd_mean = float(np.mean(depths)) if len(depths) else np.nan
        reanalysis_mean = float(reanalysis.mean()) if len(reanalysis) else np.nan
        source_values = pd.to_numeric(master.loc[master.index.year == year, "ocean_source_code"], errors="coerce").dropna()
        source_code = int(source_values.mode().iloc[0]) if len(source_values) else pd.NA
        rows.append(
            {
                "ano": year,
                "ocean_source_code": source_code,
                "n_perfis_ctd_nino34": int(len(depths)),
                "termoclina_ctd_media_m": round(ctd_mean, 3) if np.isfinite(ctd_mean) else np.nan,
                "d20_reanalise_media_m": round(reanalysis_mean, 3) if np.isfinite(reanalysis_mean) else np.nan,
                "diferenca_ctd_menos_reanalise_m": round(ctd_mean - reanalysis_mean, 3)
                if np.isfinite(ctd_mean) and np.isfinite(reanalysis_mean)
                else np.nan,
                "diferenca_m": round(ctd_mean - reanalysis_mean, 3)
                if np.isfinite(ctd_mean) and np.isfinite(reanalysis_mean)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _git_state() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"revision": None, "dirty": None, "error": str(exc)}


def _environment() -> dict[str, Any]:
    dependencies: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "xarray", "zarr", "pyarrow"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "dependencies": dependencies,
    }


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _manifest(
    *,
    run_id: str,
    args: argparse.Namespace,
    started_at: str,
    raw: pd.DataFrame,
    adjusted: pd.DataFrame,
    outputs: list[Path],
) -> dict[str, Any]:
    input_paths = [args.physical_input]
    if not args.ocean_only and args.era5_cache.exists():
        input_paths.append(args.era5_cache)
    return {
        "schema_version": "phase2-master-manifest/1.0",
        "run_id": run_id,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "command": [sys.executable, *sys.argv],
        "git": _git_state(),
        "environment": _environment(),
        "code": [
            _artifact(Path(__file__)),
            _artifact(ROOT / "src/nino_brasil/data/phase2_master.py"),
            _artifact(ROOT / "src/nino_brasil/data/audit.py"),
        ],
        "source_versions": {
            "surface_sst": "NOAA OISST v2.1",
            "atmosphere": "ECMWF ERA5 hourly records aggregated to daily box means",
            "ocean_historical_bridge": "NOAA UFS Marine Reanalysis",
            "ocean_primary": "Copernicus GLORYS12V1",
            "ocean_operational_tail": "Copernicus GLO12 analysis only",
            "in_situ": "NOAA World Ocean Database CTD",
        },
        "contract": {
            "physical_variable_count": len(PHYSICAL_COLUMNS),
            "physical_columns": list(PHYSICAL_COLUMNS),
            "metadata_columns": ["ocean_source_code"],
            "source_adjusted_version": "source_adjusted_v1",
            "predictive_cv_warning": "fit climatology/detrending on each training fold; do not leak this full-record transform",
        },
        "inputs": [_artifact(path) for path in input_paths],
        "outputs": [_artifact(path) for path in outputs if path.exists()],
        "raw_shape": list(raw.shape),
        "source_adjusted_shape": list(adjusted.shape),
        "period": [raw.index.min().isoformat(), raw.index.max().isoformat()],
        "options": {
            "era5_years": [min(args.era5_years), max(args.era5_years)],
            "ocean_only": args.ocean_only,
            "skip_ctd": args.skip_ctd,
            "strict": args.strict,
        },
    }


def _load_master(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["week_ending_sunday"]).set_index("week_ending_sunday")
    return frame.reindex(columns=[*PHYSICAL_COLUMNS, "ocean_source_code"])


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--era5-years", type=parse_year_range, default=parse_year_range(f"1981:{pd.Timestamp.today().year}"))
    parser.add_argument("--physical-input", type=Path, default=DEFAULT_PHYSICAL)
    parser.add_argument("--era5-cache", type=Path, default=DEFAULT_ERA5_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--zarr-output", type=Path, default=DEFAULT_MASTER_ZARR)
    parser.add_argument("--adjusted-output", type=Path, default=DEFAULT_ADJUSTED)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--adjusted-audit-output", type=Path, default=DEFAULT_ADJUSTED_AUDIT)
    parser.add_argument("--validation-output", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--seam-output", type=Path, default=DEFAULT_SEAM)
    parser.add_argument("--contract-output", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--ctd-output", type=Path, default=DEFAULT_CTD)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--ctd-years", type=parse_year_range)
    parser.add_argument("--ocean-only", action="store_true", help="intermediate product; atmosphere is left missing")
    parser.add_argument("--force-reextract-era5", action="store_true")
    parser.add_argument("--skip-ctd", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="validate the existing --output without rebuilding")
    parser.add_argument("--dry-run", action="store_true", help="print resolved inputs/outputs without reading arrays or writing")
    parser.add_argument("--run-id")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _resolve_paths(args: argparse.Namespace) -> None:
    for name in (
        "physical_input",
        "era5_cache",
        "output",
        "zarr_output",
        "adjusted_output",
        "audit_output",
        "adjusted_audit_output",
        "validation_output",
        "seam_output",
        "contract_output",
        "ctd_output",
        "manifest_output",
        "ledger",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    _resolve_paths(args)
    run_id = args.run_id or f"phase2_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    if args.dry_run:
        print(f"run_id={run_id}")
        print(f"physical_input={args.physical_input} exists={args.physical_input.exists()}")
        print(f"era5_years={min(args.era5_years)}:{max(args.era5_years)} ocean_only={args.ocean_only}")
        print("era5_input=Zarrs canônicos da F1 (leitura direta; sem cache Parquet diário)")
        print(f"master={args.output}")
        print(f"master_zarr={args.zarr_output}")
        print(f"source_adjusted_v1={args.adjusted_output}")
        print(f"contract={len(PHYSICAL_COLUMNS)} physical variables + ocean_source_code metadata")
        return 0 if args.physical_input.exists() or args.validate_only else 2

    if args.validate_only:
        if not args.output.exists():
            parser.error(f"master not found: {args.output}")
        raw = _load_master(args.output)
        validation = validate_master(raw)
        _atomic_to_csv(validation, args.validation_output, index=False)
        _atomic_to_csv(coverage_audit(raw, representation="raw_compatibility"), args.audit_output, index=False)
        print(validation.to_string(index=False))
        failed = validation.loc[(validation["severidade"] == "error") & ~validation["passou"]]
        return 1 if args.strict and len(failed) else 0

    if not args.physical_input.exists():
        parser.error(f"physical input not found: {args.physical_input}")
    started_at = _utc_now()
    ledger = AuditLog(args.ledger)
    ledger.record(
        run_id=run_id,
        task_id="phase2_build_master_weekly",
        dataset="nino34_master_weekly",
        source_version="phase2_master_contract/1.0",
        status="started",
        physical_input=str(args.physical_input),
        era5_years=[min(args.era5_years), max(args.era5_years)],
    )
    try:
        raw, adjusted, seam = build_products(
            args.physical_input,
            args.era5_years,
            era5_cache=args.era5_cache,
            ocean_only=args.ocean_only,
            force_reextract_era5=args.force_reextract_era5,
        )
        validation = validate_master(raw)
        if args.ocean_only:
            atmosphere_empty = validation["checagem"].eq("nenhuma_variavel_totalmente_vazia")
            validation.loc[atmosphere_empty, "severidade"] = "warning"
            validation.loc[atmosphere_empty, "detalhe"] += "; esperado em --ocean-only"
        failed = validation.loc[(validation["severidade"] == "error") & ~validation["passou"]]
        if args.strict and len(failed):
            # Scientific gate before any canonical artefact replacement.  The
            # previous valid master remains byte-for-byte intact on bad input.
            ledger.record(
                run_id=run_id,
                task_id="phase2_build_master_weekly",
                dataset="nino34_master_weekly",
                source_version="phase2_master_contract/1.0",
                status="failed",
                validation_failure_count=int(len(failed)),
                validation_failures=failed[["checagem", "detalhe"]].to_dict(orient="records"),
                canonical_master_preserved=True,
            )
            print("[falha] gate cientifico; master canonico anterior preservado:")
            print(failed.to_string(index=False))
            return 1
        raw_audit = coverage_audit(raw, representation="raw_compatibility")
        adjusted_audit = coverage_audit(adjusted, representation="source_adjusted_v1")
        contract = variable_contract_frame()

        _atomic_to_csv(adjusted, args.adjusted_output)
        _atomic_to_csv(raw_audit, args.audit_output, index=False)
        _atomic_to_csv(adjusted_audit, args.adjusted_audit_output, index=False)
        _atomic_to_csv(validation, args.validation_output, index=False)
        _atomic_to_csv(seam, args.seam_output, index=False)
        _atomic_to_csv(contract, args.contract_output, index=False)
        outputs = [
            args.output,
            args.zarr_output,
            args.adjusted_output,
            args.audit_output,
            args.adjusted_audit_output,
            args.validation_output,
            args.seam_output,
            args.contract_output,
        ]
        ctd_years = args.ctd_years or range(min(args.era5_years), max(args.era5_years) + 1)
        if not args.skip_ctd:
            ctd = ctd_validation(raw, ctd_years)
            _atomic_to_csv(ctd, args.ctd_output, index=False)
            outputs.append(args.ctd_output)
        # The canonical compatibility master is committed last among the data
        # tables, so an earlier partial failure cannot replace the prior one.
        _atomic_to_csv(raw, args.output)
        _atomic_weekly_zarr(raw, args.zarr_output)
        manifest = _manifest(
            run_id=run_id,
            args=args,
            started_at=started_at,
            raw=raw,
            adjusted=adjusted,
            outputs=outputs,
        )
        _atomic_to_json(manifest, args.manifest_output)
        outputs.append(args.manifest_output)
        ledger.record(
            run_id=run_id,
            task_id="phase2_build_master_weekly",
            dataset="nino34_master_weekly",
            source_version="phase2_master_contract/1.0",
            status="ok",
            outputs=[_artifact(path) for path in outputs],
            rows=len(raw),
            physical_variable_count=len(PHYSICAL_COLUMNS),
            validation_failure_count=int(len(failed)),
            validation_failures=failed["checagem"].tolist(),
        )
    except BaseException as exc:
        ledger.record(
            run_id=run_id,
            task_id="phase2_build_master_weekly",
            dataset="nino34_master_weekly",
            source_version="phase2_master_contract/1.0",
            status="error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise

    print(f"[ok] raw master: {_display_path(args.output)} {raw.shape}")
    print(f"[ok] source-adjusted v1: {_display_path(args.adjusted_output)} {adjusted.shape}")
    print(f"[ok] manifest: {_display_path(args.manifest_output)} run_id={run_id}")
    if len(failed):
        print("[falha] validacoes estritas:")
        print(failed.to_string(index=False))
    return 1 if args.strict and len(failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
