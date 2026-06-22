from __future__ import annotations

import argparse
import calendar
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import pandas as pd
from tqdm import tqdm
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.config import load_config, project_path
from nino_brasil.data.audit import AuditLog, dataset_summary
from nino_brasil.data.availability import available_end_date, iter_available_years, record_source_latency, requested_end_date
from nino_brasil.data.credentials import cds_credentials_status
from nino_brasil.data.download_cds import (
    ATMOSPHERE_AREAS,
    ERA5_PRESSURE_VARIABLES,
    ERA5_SINGLE_VARIABLES,
    download_era5_pressure_month,
    download_era5_single_month,
    ingest_era5_pressure_month,
    ingest_era5_pressure_year,
    ingest_era5_pressure_year_kind,
    ingest_era5_pressure_year_variable,
    ingest_era5_single_month,
    ingest_era5_single_year,
    ingest_era5_single_year_kind,
    ingest_era5_single_year_variable,
)
from nino_brasil.data.download_chirps import download_chirps_year
from nino_brasil.data.download_ctd_noaa import (
    THERMOCLINE_MAX_DEPTH_M,
    THERMOCLINE_MIN_LEVELS,
    download_wod_ctd_year,
    etl_wod_ctd_year,
)
from nino_brasil.data.download_ibge import download_ibge
from nino_brasil.data.download_oisst import download_oisst_year
from nino_brasil.data.download_noaa_psl import download_noaa_psl_nino34_anom
from nino_brasil.data.download_validation_insitu import download_argo_year, download_tao_triton_year
from nino_brasil.data.regrid import normalize_for_common_grid, regrid_dataset, target_grid_from_config
from nino_brasil.data.zarr_store import ZARR_FORMAT, chunk_plan, dataframe_to_zarr
from nino_brasil.data.anomalies import daily_anomaly, dayofyear_climatology
from nino_brasil.features.nino import nino34_sst_index
from nino_brasil.models.progression import build_daily_enso_peak_progression_table


T = TypeVar("T")
from nino_brasil.features.distributions import diagnose_dataset_distributions


DATA_DIRS = [
    "data/raw/ibge",
    "data/raw/chirps",
    "data/raw/cpc_noaa",
    "data/raw/cpc_noaa/nino34",
    "data/raw/ocean_daily/glorys12",
    "data/raw/ocean_daily/glorys12_operational",
    "data/raw/ocean_monthly/oras5",
    "data/raw/era5",
    "data/raw/ctd_noaa/wod",
    "data/raw/tao_triton",
    "data/raw/argo",
    "data/interim/ibge",
    "data/interim/ctd_noaa",
    "data/interim/brazil_precipitation",
    "data/interim/nino34",
    "data/processed/zarr",
    "data/processed/zarr/ctd_noaa",
    "data/processed/zarr/ctd_noaa/wod",
    "data/processed/zarr/distributions",
    "data/processed/zarr/features",
    "data/processed/zarr/statistics",
    "data/processed/zarr/modeling",
    "data/processed/zarr/ocean_daily/glorys12",
    "data/processed/zarr/ocean_daily/glorys12_operational",
    "data/processed/zarr/ocean_daily/noaa_ufs",
    "data/processed/zarr/ocean_monthly/oras5",
    "data/processed/zarr/features/ocean_daily",
    "data/processed/zarr/features/ocean_monthly/oras5",
    "data/processed/parquet/features",
    "data/processed/parquet/modeling",
    "data/processed/zarr/validation/tao_triton",
    "data/processed/zarr/validation/argo",
    "data/processed/geotiff",
    "data/audit",
    "data/state",
]

PROJECT_START_YEAR = 1981


def cmd_init(_: argparse.Namespace) -> int:
    for item in DATA_DIRS:
        path = project_path(item)
        path.mkdir(parents=True, exist_ok=True)
        keep = path / ".gitkeep"
        keep.touch(exist_ok=True)
        print(f"ok: {path}")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    cfg = load_config()
    usage = shutil.disk_usage(ROOT)
    print(f"project: {ROOT}")
    print(f"free disk: {usage.free / 1024**3:.1f} GB")
    print()
    print("domains:")
    for name, domain in cfg["domains"].items():
        print(f"- {name}: {domain}")
    print()
    print("data folders:")
    for item in DATA_DIRS:
        path = project_path(item)
        exists = "ok" if path.exists() else "missing"
        file_count = sum(1 for p in path.rglob("*") if p.is_file()) if path.exists() else 0
        print(f"- {exists}: {item} ({file_count} files)")
    return 0


def cmd_plan(_: argparse.Namespace) -> int:
    cfg = load_config()
    chirps_resolution = cfg.get("modeling", {}).get("chirps_resolution", "p25")
    print("data pipeline order:")
    print(f"project period: {PROJECT_START_YEAR}-latest available by source, daily master calendar")
    print("1. init")
    print("2. download-ibge --product uf")
    print("3. download-ibge --product municipios")
    print(f"4. download CHIRPS daily precipitation at {chirps_resolution}")
    print("5. download-oisst: daily SST/SSTA primary source for Nino 3.4")
    print("6. ingest-era5: request annual-kind by region/type, split to annual daily Zarr by variable")
    print("7. ocean_daily_pipeline.py: ingest originally daily NOAA UFS/GLORYS ocean fields")
    print("8. ocean_monthly_pipeline.py: ingest ORAS5 monthly means without daily promotion")
    print("9. audit_ocean_phase2.py: verify daily continuity, monthly integrity and source transitions")
    print("10. download-ctd: WOD CTD annual raw files, thermocline QC and Zarr")
    print("11. download-validation: TAO/TRITON and Argo in-situ validation for Nino 3.4")
    print("12. regrid-zarr: reconcile only sources that require a common spatial grid")
    print("13. diagnose-distributions: optional QC/EDA tail diagnostics")
    print("14. download-nino34-reference: NOAA PSL monthly peak labels/reference")
    print("15. build-nino34-daily-index: daily OISST Nino 3.4 SSTA trajectory")
    print("16. build-enso-peak-progression: daily OISST trajectory labeled by NOAA PSL peaks")
    print("17. phase 3: Nino 3.4 anomaly alignment, thermocline, slope and signal duration diagnostics")
    print("18. phase 4: exploratory statistics with regression, PCA, KNN and variable screening")
    print("19. phase 5B: build Nino3.4 -> Brazil pixel-cluster event targets")
    print("20. model_pipeline.py: event-centered walk-forward metrics, predictions and importances")
    print("21. phase 6A: train native spatial CNN baseline for ENSO peak progression")
    print("22. phase 6B: train spatiotemporal memory model for nonlinear ENSO progression")
    print("23. phase 6C: train Brazil cluster decoder for P10/P25/P75/P90 teleconnections")
    print("24. optional fallback: CMIP6 pretraining/fine-tuning only if native phase 6 skill gates fail")
    print("25. phase 8: isolated Ham2019 exploratory benchmark for saved weights/data")
    print("26. audit: verify status and failed tasks before continuing")
    return 0


def iter_years(
    start_year: int,
    end_year: int | None,
    source: str | None = None,
    cfg: dict | None = None,
) -> Iterable[int]:
    if end_year is not None and end_year < start_year:
        raise ValueError("end-year must be greater than or equal to start-year.")
    if start_year < PROJECT_START_YEAR:
        raise ValueError(f"project period starts at {PROJECT_START_YEAR}; received {start_year}.")
    if source is not None:
        return iter_available_years(start_year, end_year, source, cfg or load_config())
    if end_year is None:
        raise ValueError("end-year is required when no source is supplied.")
    return range(start_year, end_year + 1)


def iter_months(months: list[int] | None) -> list[int]:
    return months or list(range(1, 13))


def iter_complete_annual_years(
    start_year: int,
    end_year: int | None,
    source: str,
    cfg: dict,
) -> range:
    resolved_end = requested_end_date(cfg, source, end_year).date()
    final_year = resolved_end.year if resolved_end >= date(resolved_end.year, 12, 31) else resolved_end.year - 1
    if final_year < start_year:
        raise ValueError(f"No complete annual {source} years available for {start_year} after source latency.")
    return range(start_year, final_year + 1)


def iter_complete_year_months(
    start_year: int,
    end_year: int | None,
    source: str,
    cfg: dict,
    months: list[int] | None,
) -> list[tuple[int, int]]:
    resolved_end = requested_end_date(cfg, source, end_year).date()
    requested_months = iter_months(months)
    year_months: list[tuple[int, int]] = []
    for year in range(start_year, resolved_end.year + 1):
        for month in requested_months:
            month_end = date(year, month, calendar.monthrange(year, month)[1])
            if month_end <= resolved_end:
                year_months.append((year, month))
    if not year_months:
        raise ValueError(
            f"No complete monthly {source} periods available for {start_year} "
            f"through {resolved_end.isoformat()}."
        )
    return year_months


def run_or_continue(label: str, action: Callable[[], T], *, continue_on_error: bool) -> T | None:
    try:
        return action()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        if not continue_on_error:
            raise
        print(f"{label} - erro: {type(exc).__name__}: {exc}")
        return None


def _looks_like_cds_request_size_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = [
        "500",
        "cost",
        "costs",
        "internal server",
        "internal server error",
        "limit",
        "limits",
        "payload",
        "too large",
        "request too large",
        "403",
    ]
    return any(marker in text for marker in markers)


def _split_variable_group(variables: list[str]) -> tuple[list[str], list[str]]:
    midpoint = max(1, len(variables) // 2)
    return variables[:midpoint], variables[midpoint:]


def _ingest_era5_annual_auto_group(
    *,
    year: int,
    region: str,
    kind: str,
    variables: list[str],
    raw_dir: Path,
    zarr_root: Path,
    dry_run: bool,
    overwrite: bool,
    include_hash: bool,
    delete_raw_after_zarr: bool,
    audit: AuditLog,
) -> list[Path]:
    """Try the largest annual CDS request and split only when CDS rejects it."""
    if not variables:
        return []

    group = ", ".join(variables)
    if len(variables) > 1:
        try:
            if kind == "single":
                return ingest_era5_single_year_kind(
                    year=year,
                    region=region,
                    raw_dir=raw_dir,
                    zarr_root=zarr_root,
                    variables=variables,
                    dry_run=dry_run,
                    overwrite=overwrite,
                    include_hash=include_hash,
                    delete_raw_after_zarr=delete_raw_after_zarr,
                    audit=audit,
                )
            return ingest_era5_pressure_year_kind(
                year=year,
                region=region,
                raw_dir=raw_dir,
                zarr_root=zarr_root,
                variables=variables,
                dry_run=dry_run,
                overwrite=overwrite,
                include_hash=include_hash,
                delete_raw_after_zarr=delete_raw_after_zarr,
                audit=audit,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            if not _looks_like_cds_request_size_error(exc):
                raise
            left, right = _split_variable_group(variables)
            print(
                f"ERA5 {year} {region} {kind} [{group}] - CDS rejeitou o grupo; "
                f"dividindo em {len(left)} + {len(right)} variaveis"
            )
            paths: list[Path] = []
            paths.extend(
                _ingest_era5_annual_auto_group(
                    year=year,
                    region=region,
                    kind=kind,
                    variables=left,
                    raw_dir=raw_dir,
                    zarr_root=zarr_root,
                    dry_run=dry_run,
                    overwrite=overwrite,
                    include_hash=include_hash,
                    delete_raw_after_zarr=delete_raw_after_zarr,
                    audit=audit,
                )
            )
            paths.extend(
                _ingest_era5_annual_auto_group(
                    year=year,
                    region=region,
                    kind=kind,
                    variables=right,
                    raw_dir=raw_dir,
                    zarr_root=zarr_root,
                    dry_run=dry_run,
                    overwrite=overwrite,
                    include_hash=include_hash,
                    delete_raw_after_zarr=delete_raw_after_zarr,
                    audit=audit,
                )
            )
            return paths

    variable = variables[0]
    try:
        if kind == "single":
            return [
                ingest_era5_single_year_variable(
                    year=year,
                    region=region,
                    variable=variable,
                    raw_dir=raw_dir,
                    zarr_root=zarr_root,
                    dry_run=dry_run,
                    overwrite=overwrite,
                    include_hash=include_hash,
                    delete_raw_after_zarr=delete_raw_after_zarr,
                    audit=audit,
                )
            ]
        return [
            ingest_era5_pressure_year_variable(
                year=year,
                region=region,
                variable=variable,
                raw_dir=raw_dir,
                zarr_root=zarr_root,
                dry_run=dry_run,
                overwrite=overwrite,
                include_hash=include_hash,
                delete_raw_after_zarr=delete_raw_after_zarr,
                audit=audit,
            )
        ]
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        if not _looks_like_cds_request_size_error(exc):
            raise
        print(f"ERA5 {year} {region} {kind} {variable} - requisicao anual rejeitada; fallback mensal")
        if kind == "single":
            return [
                ingest_era5_single_year(
                    year=year,
                    region=region,
                    raw_dir=raw_dir,
                    zarr_root=zarr_root,
                    variables=[variable],
                    dry_run=dry_run,
                    overwrite=overwrite,
                    include_hash=include_hash,
                    audit=audit,
                )
            ]
        return [
            ingest_era5_pressure_year(
                year=year,
                region=region,
                raw_dir=raw_dir,
                zarr_root=zarr_root,
                variables=[variable],
                dry_run=dry_run,
                overwrite=overwrite,
                include_hash=include_hash,
                audit=audit,
            )
        ]


def _selected_variables(requested: list[str] | None, allowed: list[str]) -> list[str] | None:
    if not requested:
        return None
    return [variable for variable in requested if variable in allowed]


def cmd_check_cds(_: argparse.Namespace) -> int:
    status = cds_credentials_status()
    print("Copernicus CDS credentials:")
    print(f"- CDS_API_URL: {status['CDS_API_URL']}")
    print(f"- CDS_API_KEY: {status['CDS_API_KEY']}")
    print(f"- ready: {status['ready']}")
    if status["ready"] != "yes":
        print()
        print("Set CDS_API_KEY in your shell or in a local .env file.")
        return 1
    return 0


def cmd_audit(_: argparse.Namespace) -> int:
    AuditLog().print_summary()
    return 0


def _default_oisst_zarr_paths() -> list[Path]:
    processed = sorted(project_path("data/processed/zarr/cpc_noaa/oisst").glob("*.zarr"))
    if processed:
        return processed
    return sorted(project_path("data/processed/zarr/regridded").glob("noaa_oisst_*.zarr"))


def cmd_build_nino34_daily_index(args: argparse.Namespace) -> int:
    paths = [_path_arg(path) for path in (args.input_zarr or [])] or _default_oisst_zarr_paths()
    output_zarr = _path_arg(args.output_zarr)
    output_csv = _path_arg(args.output_csv)
    audit = AuditLog()
    task_id = "build_nino34_daily_oisst"

    if args.dry_run:
        print("DRY RUN build Nino 3.4 daily index")
        print(f"inputs={len(paths)}")
        for path in paths[:10]:
            print(f"- {path}")
        if len(paths) > 10:
            print("...")
        print(f"sst_var={args.sst_var}")
        print(f"climatology={args.climatology_start}..{args.climatology_end}; window_days={args.window_days}")
        print(f"output_zarr={output_zarr}")
        print(f"output_csv={output_csv}")
        return 0

    if not paths:
        raise FileNotFoundError("No OISST Zarr inputs found. Run download-oisst/regrid-zarr first or pass --input-zarr.")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing OISST Zarr inputs: {missing[:5]}")

    audit.record(
        task_id=task_id,
        dataset="nino34_daily_oisst",
        status="started",
        input_paths=[str(path) for path in paths],
        output_zarr=str(output_zarr),
        output_csv=str(output_csv),
        climatology_start=args.climatology_start,
        climatology_end=args.climatology_end,
    )
    datasets = [xr.open_zarr(path) for path in paths]
    try:
        ds = xr.concat(datasets, dim="time").sortby("time") if len(datasets) > 1 else datasets[0].sortby("time")
        if args.sst_var not in ds:
            raise KeyError(f"{args.sst_var!r} not found in OISST dataset variables: {list(ds.data_vars)}")
        index = nino34_sst_index(ds[args.sst_var])
        climatology_source = index.sel(time=slice(args.climatology_start, args.climatology_end))
        climatology = dayofyear_climatology(climatology_source, args.window_days)
        ssta = daily_anomaly(index, climatology=climatology, window_days=args.window_days).rename("nino34_ssta")

        table_data = {
            "time": pd.DatetimeIndex(index["time"].values),
            "nino34_sst": index.to_pandas().to_numpy(),
            "nino34_ssta": ssta.to_pandas().to_numpy(),
        }
        frame = pd.DataFrame(table_data).sort_values("time").reset_index(drop=True)
        frame["year"] = frame["time"].dt.year
        frame["month"] = frame["time"].dt.month
        frame["day"] = frame["time"].dt.day
        rolling_days = args.rolling_days or [7, 30, 90, 180]
        for window in rolling_days:
            frame[f"nino34_ssta_mean_{window}d"] = frame["nino34_ssta"].rolling(window, min_periods=1).mean()
            frame[f"nino34_ssta_delta_{window}d"] = frame["nino34_ssta"] - frame["nino34_ssta"].shift(window)
        frame["source"] = "NOAA OISST v2.1 daily SST"
        frame["climatology"] = f"{args.climatology_start}:{args.climatology_end}"

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_csv, index=False)
        dataframe_to_zarr(
            frame,
            output_zarr,
            overwrite=True,
            attrs={
                "artifact": "nino34_daily_oisst",
                "climatology_start": args.climatology_start,
                "climatology_end": args.climatology_end,
                "window_days": int(args.window_days),
            },
        )
        audit.record(
            task_id=task_id,
            dataset="nino34_daily_oisst",
            status="ok",
            output_zarr=str(output_zarr),
            output_csv=str(output_csv),
            rows=int(len(frame)),
            first_date=str(frame["time"].min().date()),
            last_date=str(frame["time"].max().date()),
            zarr_summary=dataset_summary(output_zarr, zarr=True),
        )
        print(f"nino34 daily csv: {output_csv}")
        print(f"nino34 daily zarr: {output_zarr}")
        return 0
    except Exception as exc:
        audit.record(
            task_id=task_id,
            dataset="nino34_daily_oisst",
            status="error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    finally:
        for dataset in datasets:
            dataset.close()


def cmd_download_nino34_reference(args: argparse.Namespace) -> int:
    raw_path = _path_arg(args.raw_path)
    csv_path = _path_arg(args.csv_path)
    zarr_path = _path_arg(args.zarr_path)
    peaks_csv_path = _path_arg(args.peaks_csv_path)
    peaks_zarr_path = _path_arg(args.peaks_zarr_path)
    audit = AuditLog()
    task_id = "download_noaa_psl_nino34_reference"

    if args.dry_run:
        print("DRY RUN download NOAA PSL Nino 3.4 reference")
        print(f"raw={raw_path}")
        print(f"csv={csv_path}")
        print(f"zarr={zarr_path}")
        print(f"peaks_csv={peaks_csv_path}")
        print(f"peaks_zarr={peaks_zarr_path}")
        return 0

    audit.record(
        task_id=task_id,
        dataset="noaa_psl_nino34_reference",
        status="started",
        raw_path=str(raw_path),
        csv_path=str(csv_path),
        zarr_path=str(zarr_path),
        peaks_csv_path=str(peaks_csv_path),
        peaks_zarr_path=str(peaks_zarr_path),
    )
    try:
        output = download_noaa_psl_nino34_anom(
            raw_path=raw_path,
            csv_path=csv_path,
            zarr_path=zarr_path,
            peaks_csv_path=peaks_csv_path,
            peaks_zarr_path=peaks_zarr_path,
            overwrite=args.overwrite,
        )
        audit.record(
            task_id=task_id,
            dataset="noaa_psl_nino34_reference",
            status="ok",
            raw_path=str(output.raw_path),
            csv_path=str(output.csv_path),
            zarr_path=str(output.zarr_path),
            peaks_csv_path=str(output.peaks_csv_path),
            peaks_zarr_path=str(output.peaks_zarr_path),
            rows=output.rows,
            peaks=output.peaks,
        )
        print(f"nino34 reference csv: {output.csv_path}")
        print(f"nino34 reference peaks: {output.peaks_csv_path}")
        return 0
    except Exception as exc:
        audit.record(
            task_id=task_id,
            dataset="noaa_psl_nino34_reference",
            status="error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


def cmd_build_enso_peak_progression(args: argparse.Namespace) -> int:
    daily_csv = _path_arg(args.daily_csv)
    peaks_csv = _path_arg(args.peaks_csv)
    output_zarr = _path_arg(args.output_zarr)
    output_csv = _path_arg(args.output_csv)
    lead_days = args.lead_days or [30, 60, 90, 120, 180, 270, 365]
    history_days = args.history_days or [7, 30, 90, 180]
    audit = AuditLog()
    task_id = "build_enso_peak_progression"

    if args.dry_run:
        print("DRY RUN build ENSO peak progression")
        print(f"daily_csv={daily_csv}")
        print(f"peaks_csv={peaks_csv}")
        print(f"lead_days={lead_days}")
        print(f"history_days={history_days}")
        print(f"output_zarr={output_zarr}")
        print(f"output_csv={output_csv}")
        return 0

    if not daily_csv.exists():
        raise FileNotFoundError(f"Daily Nino 3.4 CSV not found: {daily_csv}")
    if not peaks_csv.exists():
        raise FileNotFoundError(f"NOAA PSL reference peaks CSV not found: {peaks_csv}")

    audit.record(
        task_id=task_id,
        dataset="enso_peak_progression",
        status="started",
        daily_csv=str(daily_csv),
        peaks_csv=str(peaks_csv),
        output_zarr=str(output_zarr),
        output_csv=str(output_csv),
        lead_days=lead_days,
        history_days=history_days,
    )
    try:
        daily = pd.read_csv(daily_csv, parse_dates=["time"])
        peaks = pd.read_csv(peaks_csv, parse_dates=["peak_time"])
        table = build_daily_enso_peak_progression_table(
            daily,
            peaks,
            lead_days=lead_days,
            history_windows_days=history_days,
        )
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_csv, index=False)
        dataframe_to_zarr(
            table,
            output_zarr,
            overwrite=True,
            attrs={
                "artifact": "enso_peak_progression",
                "dynamic_signal": "daily_oisst_nino34_ssta",
                "peak_reference": "noaa_psl_monthly_nino34_ersst_v6",
            },
        )
        audit.record(
            task_id=task_id,
            dataset="enso_peak_progression",
            status="ok",
            output_zarr=str(output_zarr),
            output_csv=str(output_csv),
            rows=int(len(table)),
            zarr_summary=dataset_summary(output_zarr, zarr=True),
        )
        print(f"enso peak progression csv: {output_csv}")
        print(f"enso peak progression zarr: {output_zarr}")
        return 0
    except Exception as exc:
        audit.record(
            task_id=task_id,
            dataset="enso_peak_progression",
            status="error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_path(value)


def cmd_regrid_zarr(args: argparse.Namespace) -> int:
    cfg = load_config()
    input_path = _path_arg(args.input)
    output_path = _path_arg(args.output)
    method = args.method or cfg.get("modeling", {}).get("grid", {}).get("regrid_method", "bilinear")
    audit = AuditLog()
    task_id = f"regrid_{input_path.stem}"

    if args.dry_run:
        print(f"DRY RUN regrid: {input_path} -> {output_path}")
        print(f"method={method}; lon_convention={args.lon_convention}")
        print(f"target_grid={cfg.get('modeling', {}).get('grid', {})}")
        return 0

    audit.record(
        task_id=task_id,
        dataset=args.dataset,
        status="started",
        input_path=str(input_path),
        output_path=str(output_path),
        method=method,
    )
    try:
        ds = xr.open_zarr(input_path)
        try:
            normalized = normalize_for_common_grid(ds, convention=args.lon_convention)
            regridded = regrid_dataset(
                normalized,
                target_grid_from_config(cfg),
                method=method,
            )
            if output_path.exists() and not args.overwrite:
                dataset_summary(output_path, zarr=True)
                print(f"zarr exists: {output_path}")
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                chunks = chunk_plan(regridded)
                if chunks:
                    regridded = regridded.chunk(chunks)

                # Zarr v3 inputs expose backend-specific encodings such as
                # ``serializer`` and ``compressors``.  Reusing those encodings
                # while writing the project's Zarr v2 outputs raises
                # ``Zarr format 2 arrays do not support serializer``.  The
                # regridded values/chunks are independent of those source
                # encodings, so discard them before serializing.
                regridded = regridded.drop_encoding()

                # Build and validate beside the destination before replacing
                # it.  A failed write must not destroy the last usable store.
                staging_path = output_path.with_name(f".{output_path.name}.building")
                backup_path = output_path.with_name(f".{output_path.name}.backup")
                if staging_path.exists():
                    shutil.rmtree(staging_path)
                regridded.to_zarr(staging_path, mode="w", consolidated=True, zarr_format=ZARR_FORMAT)
                dataset_summary(staging_path, zarr=True)

                if backup_path.exists():
                    shutil.rmtree(backup_path)
                if output_path.exists():
                    output_path.replace(backup_path)
                try:
                    staging_path.replace(output_path)
                except BaseException:
                    if backup_path.exists() and not output_path.exists():
                        backup_path.replace(output_path)
                    raise
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                print(f"regridded zarr written: {output_path}")
            audit.record(
                task_id=task_id,
                dataset=args.dataset,
                status="ok",
                input_path=str(input_path),
                output_path=str(output_path),
                output_summary=dataset_summary(output_path, zarr=True),
                method=method,
            )
        finally:
            ds.close()
        return 0
    except Exception as exc:
        audit.record(
            task_id=task_id,
            dataset=args.dataset,
            status="error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


def cmd_diagnose_distributions(args: argparse.Namespace) -> int:
    input_path = _path_arg(args.input)
    output_path = _path_arg(args.output)
    csv_output_path = _path_arg(args.csv_output)
    audit = AuditLog()
    task_id = f"diagnose_distributions_{input_path.stem}"

    if args.dry_run:
        print(f"DRY RUN diagnose distributions: {input_path} -> {output_path}")
        print(f"DRY RUN diagnose distributions CSV: {csv_output_path}")
        print(f"variables={args.variable or 'all'}; tail={args.tail}; sample_size={args.sample_size}")
        return 0

    audit.record(
        task_id=task_id,
        dataset=args.dataset,
        status="started",
        input_path=str(input_path),
        output_path=str(output_path),
        variables=args.variable or "all",
        tail=args.tail,
    )
    try:
        ds = xr.open_zarr(input_path)
        try:
            diagnostics = diagnose_dataset_distributions(
                ds,
                variables=args.variable,
                tail=args.tail,
                sample_size=args.sample_size,
                min_tail=args.min_tail,
                random_state=args.random_state,
            )
        finally:
            ds.close()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe_to_zarr(
            diagnostics,
            output_path,
            overwrite=True,
            attrs={
                "dataset": args.dataset,
                "source_zarr": str(input_path),
                "tail": args.tail,
            },
        )
        csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(csv_output_path, index=False)
        audit.record(
            task_id=task_id,
            dataset=args.dataset,
            status="ok",
            input_path=str(input_path),
            output_path=str(output_path),
            csv_output_path=str(csv_output_path),
            rows=int(len(diagnostics)),
        )
        print(f"distribution diagnostics: {output_path}")
        print(f"distribution diagnostics csv: {csv_output_path}")
        return 0
    except Exception as exc:
        audit.record(
            task_id=task_id,
            dataset=args.dataset,
            status="error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


def cmd_download_chirps(args: argparse.Namespace) -> int:
    cfg = load_config()
    audit = AuditLog()
    record_source_latency(audit, "chirps", cfg)
    resolution = args.resolution or cfg.get("modeling", {}).get("chirps_resolution", "p25")
    raw_dir = project_path("data/raw/chirps")
    for year in iter_years(args.start_year, args.end_year, "chirps", cfg):
        run_or_continue(
            f"CHIRPS {year}",
            lambda year=year: download_chirps_year(
                year=year,
                raw_dir=raw_dir,
                resolution=resolution,
                overwrite=args.overwrite,
                dry_run=not args.execute,
            ),
            continue_on_error=args.continue_on_error,
        )
    return 0


def cmd_download_oisst(args: argparse.Namespace) -> int:
    cfg = load_config()
    audit = AuditLog()
    record_source_latency(audit, "noaa_oisst", cfg)
    raw_dir = project_path("data/raw/cpc_noaa/oisst")
    for year in iter_years(args.start_year, args.end_year, "noaa_oisst", cfg):
        run_or_continue(
            f"OISST {year}",
            lambda year=year: download_oisst_year(
                year=year,
                raw_dir=raw_dir,
                overwrite=args.overwrite,
                dry_run=not args.execute,
            ),
            continue_on_error=args.continue_on_error,
        )
    return 0


def cmd_download_ctd(args: argparse.Namespace) -> int:
    cfg = load_config()
    raw_dir = project_path("data/raw/ctd_noaa")
    zarr_root = project_path("data/processed/zarr")
    audit = AuditLog()
    record_source_latency(audit, "noaa_wod_ctd", cfg)
    for year in tqdm(iter_years(args.start_year, args.end_year, "noaa_wod_ctd", cfg), desc="WOD CTD years", unit="year"):
        raw_path = run_or_continue(
            f"CTD/WOD download {year}",
            lambda year=year: download_wod_ctd_year(
                year=year,
                raw_dir=raw_dir,
                dry_run=not args.execute,
                overwrite=args.overwrite,
                allow_missing_source=not args.stop_on_missing_source,
                include_hash=args.hash,
                audit=audit,
            ),
            continue_on_error=args.continue_on_error,
        )
        if raw_path is None:
            continue
        if args.raw_only:
            continue
        if args.execute and not raw_path.exists():
            continue
        run_or_continue(
            f"CTD/WOD ETL {year}",
            lambda year=year: etl_wod_ctd_year(
                year=year,
                raw_dir=raw_dir,
                zarr_root=zarr_root,
                dry_run=not args.execute,
                overwrite=args.overwrite,
                max_depth_m=args.max_depth,
                depth_step_m=args.depth_step,
                min_levels=args.min_levels,
                good_flags=args.good_flag or [0],
                require_salinity=args.require_salinity,
                include_hash=args.hash,
                audit=audit,
            ),
            continue_on_error=args.continue_on_error,
        )
    return 0


def cmd_etl_ctd(args: argparse.Namespace) -> int:
    cfg = load_config()
    raw_dir = project_path("data/raw/ctd_noaa")
    zarr_root = project_path("data/processed/zarr")
    audit = AuditLog()
    record_source_latency(audit, "noaa_wod_ctd", cfg)
    for year in tqdm(iter_years(args.start_year, args.end_year, "noaa_wod_ctd", cfg), desc="WOD CTD ETL years", unit="year"):
        run_or_continue(
            f"CTD/WOD ETL {year}",
            lambda year=year: etl_wod_ctd_year(
                year=year,
                raw_dir=raw_dir,
                zarr_root=zarr_root,
                dry_run=not args.execute,
                overwrite=args.overwrite,
                max_depth_m=args.max_depth,
                depth_step_m=args.depth_step,
                min_levels=args.min_levels,
                good_flags=args.good_flag or [0],
                require_salinity=args.require_salinity,
                include_hash=args.hash,
                audit=audit,
            ),
            continue_on_error=args.continue_on_error,
        )
    return 0


def cmd_download_era5(args: argparse.Namespace) -> int:
    cfg = load_config()
    raw_dir = project_path("data/raw/era5")
    zarr_root = project_path("data/processed/zarr")
    regions = args.region or list(ATMOSPHERE_AREAS)
    months = iter_months(args.month)
    if args.annual_zarr and args.raw_only:
        raise ValueError("--annual-zarr cannot be combined with --raw-only.")
    if args.annual_zarr and args.month and sorted(set(args.month)) != list(range(1, 13)):
        raise ValueError("--annual-zarr needs the full year; omit --month or pass all 12 months.")
    single_variables = _selected_variables(args.variable, ERA5_SINGLE_VARIABLES)
    pressure_variables = _selected_variables(args.variable, ERA5_PRESSURE_VARIABLES)
    if args.variable and args.kind == "single" and not single_variables:
        raise ValueError(f"No requested variable belongs to ERA5 single levels: {args.variable}")
    if args.variable and args.kind == "pressure" and not pressure_variables:
        raise ValueError(f"No requested variable belongs to ERA5 pressure levels: {args.variable}")
    # None means "all variables of the family"; an empty list means "nothing requested".
    request_single = args.kind in {"single", "both"} and (single_variables is None or bool(single_variables))
    request_pressure = args.kind in {"pressure", "both"} and (pressure_variables is None or bool(pressure_variables))
    audit = AuditLog()
    record_source_latency(audit, "era5", cfg)
    if args.annual_zarr:
        years = iter_complete_annual_years(args.start_year, args.end_year, "era5", cfg)
        if args.request_mode == "monthly-kind":
            tasks = [
                (year, region)
                for year in years
                for region in regions
            ]
            for year, region in tqdm(tasks, desc="ERA5 annual monthly-kind tasks", unit="task"):
                if request_single:
                    run_or_continue(
                        f"ERA5 single annual {year} {region}",
                        lambda year=year, region=region: ingest_era5_single_year(
                            year=year,
                            region=region,
                            raw_dir=raw_dir,
                            zarr_root=zarr_root,
                            months=months,
                            variables=single_variables,
                            dry_run=not args.execute,
                            overwrite=args.overwrite,
                            include_hash=args.hash,
                            audit=audit,
                        ),
                        continue_on_error=args.continue_on_error,
                    )
                if request_pressure:
                    run_or_continue(
                        f"ERA5 pressure annual {year} {region}",
                        lambda year=year, region=region: ingest_era5_pressure_year(
                            year=year,
                            region=region,
                            raw_dir=raw_dir,
                            zarr_root=zarr_root,
                            months=months,
                            variables=pressure_variables,
                            dry_run=not args.execute,
                            overwrite=args.overwrite,
                            include_hash=args.hash,
                            audit=audit,
                        ),
                        continue_on_error=args.continue_on_error,
                    )
            return 0

        if args.request_mode == "annual-kind":
            tasks = [
                (year, region)
                for year in years
                for region in regions
            ]
            total_tasks = len(tasks) * int(request_single) + len(tasks) * int(request_pressure)
            print(f"ERA5 annual-kind CDS requests: {total_tasks}")
            for year, region in tqdm(tasks, desc="ERA5 annual-kind years/regions", unit="task"):
                if request_single:
                    run_or_continue(
                        f"ERA5 single annual-kind {year} {region}",
                        lambda year=year, region=region: ingest_era5_single_year_kind(
                            year=year,
                            region=region,
                            raw_dir=raw_dir,
                            zarr_root=zarr_root,
                            variables=single_variables,
                            dry_run=not args.execute,
                            overwrite=args.overwrite,
                            include_hash=args.hash,
                            delete_raw_after_zarr=args.delete_raw_after_zarr,
                            audit=audit,
                        ),
                        continue_on_error=args.continue_on_error,
                    )
                if request_pressure:
                    run_or_continue(
                        f"ERA5 pressure annual-kind {year} {region}",
                        lambda year=year, region=region: ingest_era5_pressure_year_kind(
                            year=year,
                            region=region,
                            raw_dir=raw_dir,
                            zarr_root=zarr_root,
                            variables=pressure_variables,
                            dry_run=not args.execute,
                            overwrite=args.overwrite,
                            include_hash=args.hash,
                            delete_raw_after_zarr=args.delete_raw_after_zarr,
                            audit=audit,
                        ),
                        continue_on_error=args.continue_on_error,
                    )
            return 0

        if args.request_mode == "annual-auto":
            tasks = [
                (year, region)
                for year in years
                for region in regions
            ]
            total_families = len(tasks) * int(request_single) + len(tasks) * int(request_pressure)
            print(
                "ERA5 annual-auto groups: "
                f"{total_families}; tenta annual-kind agregado e divide apenas se o CDS rejeitar"
            )
            for year, region in tqdm(tasks, desc="ERA5 annual-auto years/regions", unit="task"):
                if request_single:
                    run_or_continue(
                        f"ERA5 single annual-auto {year} {region}",
                        lambda year=year, region=region: _ingest_era5_annual_auto_group(
                            year=year,
                            region=region,
                            kind="single",
                            variables=single_variables or ERA5_SINGLE_VARIABLES,
                            raw_dir=raw_dir,
                            zarr_root=zarr_root,
                            dry_run=not args.execute,
                            overwrite=args.overwrite,
                            include_hash=args.hash,
                            delete_raw_after_zarr=args.delete_raw_after_zarr,
                            audit=audit,
                        ),
                        continue_on_error=args.continue_on_error,
                    )
                if request_pressure:
                    run_or_continue(
                        f"ERA5 pressure annual-auto {year} {region}",
                        lambda year=year, region=region: _ingest_era5_annual_auto_group(
                            year=year,
                            region=region,
                            kind="pressure",
                            variables=pressure_variables or ERA5_PRESSURE_VARIABLES,
                            raw_dir=raw_dir,
                            zarr_root=zarr_root,
                            dry_run=not args.execute,
                            overwrite=args.overwrite,
                            include_hash=args.hash,
                            delete_raw_after_zarr=args.delete_raw_after_zarr,
                            audit=audit,
                        ),
                        continue_on_error=args.continue_on_error,
                    )
            return 0

        single_task_variables = single_variables or ERA5_SINGLE_VARIABLES
        pressure_task_variables = pressure_variables or ERA5_PRESSURE_VARIABLES
        tasks: list[tuple[int, str, str, str]] = []
        for year in years:
            for region in regions:
                if request_single:
                    tasks.extend((year, region, "single", variable) for variable in single_task_variables)
                if request_pressure:
                    tasks.extend((year, region, "pressure", variable) for variable in pressure_task_variables)

        print(f"ERA5 annual variable tasks: {len(tasks)}")
        for year, region, kind, variable in tasks:
            if kind == "single":
                run_or_continue(
                    f"ERA5 single annual {year} {region} {variable}",
                    lambda year=year, region=region, variable=variable: ingest_era5_single_year_variable(
                        year=year,
                        region=region,
                        variable=variable,
                        raw_dir=raw_dir,
                        zarr_root=zarr_root,
                        dry_run=not args.execute,
                        overwrite=args.overwrite,
                        include_hash=args.hash,
                        delete_raw_after_zarr=args.delete_raw_after_zarr,
                        audit=audit,
                    ),
                    continue_on_error=args.continue_on_error,
                )
            else:
                run_or_continue(
                    f"ERA5 pressure annual {year} {region} {variable}",
                    lambda year=year, region=region, variable=variable: ingest_era5_pressure_year_variable(
                        year=year,
                        region=region,
                        variable=variable,
                        raw_dir=raw_dir,
                        zarr_root=zarr_root,
                        dry_run=not args.execute,
                        overwrite=args.overwrite,
                        include_hash=args.hash,
                        delete_raw_after_zarr=args.delete_raw_after_zarr,
                        audit=audit,
                    ),
                    continue_on_error=args.continue_on_error,
                )
        return 0

    year_months = iter_complete_year_months(args.start_year, args.end_year, "era5", cfg, args.month)
    tasks = [
        (year, month, region)
        for year, month in year_months
        for region in regions
    ]
    for year, month, region in tqdm(tasks, desc="ERA5 tasks", unit="task"):
        if args.raw_only:
            if request_single:
                run_or_continue(
                    f"ERA5 single raw {year}-{month:02d} {region}",
                    lambda year=year, month=month, region=region: download_era5_single_month(
                        year=year,
                        month=month,
                        region=region,
                        raw_dir=raw_dir,
                        variables=single_variables,
                        dry_run=not args.execute,
                        overwrite=args.overwrite,
                    ),
                    continue_on_error=args.continue_on_error,
                )
            if request_pressure:
                run_or_continue(
                    f"ERA5 pressure raw {year}-{month:02d} {region}",
                    lambda year=year, month=month, region=region: download_era5_pressure_month(
                        year=year,
                        month=month,
                        region=region,
                        raw_dir=raw_dir,
                        variables=pressure_variables,
                        dry_run=not args.execute,
                        overwrite=args.overwrite,
                    ),
                    continue_on_error=args.continue_on_error,
                )
        else:
            if request_single:
                run_or_continue(
                    f"ERA5 single {year}-{month:02d} {region}",
                    lambda year=year, month=month, region=region: ingest_era5_single_month(
                        year=year,
                        month=month,
                        region=region,
                        raw_dir=raw_dir,
                        zarr_root=zarr_root,
                        variables=single_variables,
                        dry_run=not args.execute,
                        overwrite=args.overwrite,
                        include_hash=args.hash,
                        audit=audit,
                    ),
                    continue_on_error=args.continue_on_error,
                )
            if request_pressure:
                run_or_continue(
                    f"ERA5 pressure {year}-{month:02d} {region}",
                    lambda year=year, month=month, region=region: ingest_era5_pressure_month(
                        year=year,
                        month=month,
                        region=region,
                        raw_dir=raw_dir,
                        zarr_root=zarr_root,
                        variables=pressure_variables,
                        dry_run=not args.execute,
                        overwrite=args.overwrite,
                        include_hash=args.hash,
                        audit=audit,
                    ),
                    continue_on_error=args.continue_on_error,
                )
    return 0


def cmd_download_validation(args: argparse.Namespace) -> int:
    raw_tao = project_path("data/raw/tao_triton")
    raw_argo = project_path("data/raw/argo")
    audit = AuditLog()
    end_date = date.fromisoformat(args.end_date) if args.end_date else date.today()
    end_year = args.end_year or end_date.year
    years = range(args.start_year, end_year + 1)

    if args.source in {"tao_triton", "all"}:
        tao_products = args.tao_product or ["temperature", "salinity"]
        tasks = [(year, product) for year in years for product in tao_products]
        for year, product in tqdm(tasks, desc="TAO/TRITON validation", unit="task"):
            run_or_continue(
                f"TAO/TRITON {product} {year}",
                lambda year=year, product=product: download_tao_triton_year(
                    year=year,
                    raw_dir=raw_tao,
                    product=product,
                    max_depth_m=args.max_depth,
                    end_date=end_date,
                    dry_run=not args.execute,
                    overwrite=args.overwrite,
                    include_hash=args.hash,
                    audit=audit,
                ),
                continue_on_error=args.continue_on_error,
            )

    if args.source in {"argo", "all"}:
        argo_start = max(args.start_year, args.argo_start_year)
        for year in tqdm(range(argo_start, end_year + 1), desc="Argo validation", unit="year"):
            run_or_continue(
                f"Argo Nino34 {year}",
                lambda year=year: download_argo_year(
                    year=year,
                    raw_dir=raw_argo,
                    max_depth_m=args.max_depth,
                    end_date=end_date,
                    dry_run=not args.execute,
                    overwrite=args.overwrite,
                    include_hash=args.hash,
                    audit=audit,
                ),
                continue_on_error=args.continue_on_error,
            )
    return 0


def cmd_download_ibge(args: argparse.Namespace) -> int:
    raw_dir = project_path("data/raw/ibge")
    interim_dir = project_path("data/interim/ibge")
    download_ibge(
        product_id=args.product,
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        extract=not args.no_extract,
        overwrite=args.overwrite,
    )
    return 0


def cmd_download_all(args: argparse.Namespace) -> int:
    cfg = load_config()
    dry_run = not args.execute
    chirps_resolution = args.chirps_resolution or cfg.get("modeling", {}).get("chirps_resolution", "p25")
    print(f"download-all dry_run={dry_run}")
    print(f"years={args.start_year}-latest available by source")

    raw_ibge = project_path("data/raw/ibge")
    interim_ibge = project_path("data/interim/ibge")
    for product in ["uf", "municipios"]:
        run_or_continue(
            f"IBGE {product}",
            lambda product=product: download_ibge(
                product_id=product,
                raw_dir=raw_ibge,
                interim_dir=interim_ibge,
                extract=True,
                overwrite=args.overwrite,
                dry_run=dry_run,
            ),
            continue_on_error=args.continue_on_error,
        )

    audit = AuditLog()
    record_source_latency(audit, "chirps", cfg)
    for year in iter_years(args.start_year, args.end_year, "chirps", cfg):
        run_or_continue(
            f"CHIRPS {year}",
            lambda year=year: download_chirps_year(
                year=year,
                raw_dir=project_path("data/raw/chirps"),
                resolution=chirps_resolution,
                overwrite=args.overwrite,
                dry_run=dry_run,
            ),
            continue_on_error=args.continue_on_error,
        )
    record_source_latency(audit, "noaa_oisst", cfg)
    for year in iter_years(args.start_year, args.end_year, "noaa_oisst", cfg):
        run_or_continue(
            f"OISST {year}",
            lambda year=year: download_oisst_year(
                year=year,
                raw_dir=project_path("data/raw/cpc_noaa/oisst"),
                overwrite=args.overwrite,
                dry_run=dry_run,
            ),
            continue_on_error=args.continue_on_error,
        )

    if args.include_cds:
        months = iter_months(args.month)
        record_source_latency(audit, "era5", cfg)
        if args.month:
            tasks = [
                (year, month, region)
                for year in iter_years(args.start_year, args.end_year, "era5", cfg)
                for month in months
                for region in ATMOSPHERE_AREAS
            ]
            for year, month, region in tqdm(tasks, desc="ERA5 ingest months", unit="task"):
                run_or_continue(
                    f"ERA5 single {year}-{month:02d} {region}",
                    lambda year=year, month=month, region=region: ingest_era5_single_month(
                        year=year,
                        month=month,
                        region=region,
                        raw_dir=project_path("data/raw/era5"),
                        zarr_root=project_path("data/processed/zarr"),
                        dry_run=dry_run,
                        overwrite=args.overwrite,
                        include_hash=args.hash,
                        audit=audit,
                    ),
                    continue_on_error=args.continue_on_error,
                )
                run_or_continue(
                    f"ERA5 pressure {year}-{month:02d} {region}",
                    lambda year=year, month=month, region=region: ingest_era5_pressure_month(
                        year=year,
                        month=month,
                        region=region,
                        raw_dir=project_path("data/raw/era5"),
                        zarr_root=project_path("data/processed/zarr"),
                        dry_run=dry_run,
                        overwrite=args.overwrite,
                        include_hash=args.hash,
                        audit=audit,
                    ),
                    continue_on_error=args.continue_on_error,
                )
        else:
            tasks = [
                (year, region)
                for year in iter_years(args.start_year, args.end_year, "era5", cfg)
                for region in ATMOSPHERE_AREAS
            ]
            for year, region in tqdm(tasks, desc="ERA5 annual ingest", unit="task"):
                run_or_continue(
                    f"ERA5 single annual {year} {region}",
                    lambda year=year, region=region: ingest_era5_single_year(
                        year=year,
                        region=region,
                        raw_dir=project_path("data/raw/era5"),
                        zarr_root=project_path("data/processed/zarr"),
                        dry_run=dry_run,
                        overwrite=args.overwrite,
                        include_hash=args.hash,
                        audit=audit,
                    ),
                    continue_on_error=args.continue_on_error,
                )
                run_or_continue(
                    f"ERA5 pressure annual {year} {region}",
                    lambda year=year, region=region: ingest_era5_pressure_year(
                        year=year,
                        region=region,
                        raw_dir=project_path("data/raw/era5"),
                        zarr_root=project_path("data/processed/zarr"),
                        dry_run=dry_run,
                        overwrite=args.overwrite,
                        include_hash=args.hash,
                        audit=audit,
                    ),
                    continue_on_error=args.continue_on_error,
                )
    else:
        print("CDS downloads skipped. Add --include-cds to include ERA5.")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Main entrypoint for NINO-BRASIL data ingestion and ETL."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="Create the data folder structure.")
    init_p.set_defaults(func=cmd_init)

    status_p = sub.add_parser("status", help="Show domains, folders and free disk.")
    status_p.set_defaults(func=cmd_status)

    plan_p = sub.add_parser("plan", help="Show the recommended data pipeline order.")
    plan_p.set_defaults(func=cmd_plan)

    cds_p = sub.add_parser("check-cds", help="Check Copernicus CDS credentials.")
    cds_p.set_defaults(func=cmd_check_cds)

    audit_p = sub.add_parser("audit", help="Summarize the local audit ledger.")
    audit_p.set_defaults(func=cmd_audit)

    nino34_daily_p = sub.add_parser(
        "build-nino34-daily-index",
        help="Build daily Nino 3.4 SST/SSTA trajectory from OISST Zarr inputs.",
    )
    nino34_daily_p.add_argument("--input-zarr", action="append", help="OISST Zarr input; defaults to local annual OISST stores.")
    nino34_daily_p.add_argument("--sst-var", default="sst")
    nino34_daily_p.add_argument("--climatology-start", default="1991-01-01")
    nino34_daily_p.add_argument("--climatology-end", default="2020-12-31")
    nino34_daily_p.add_argument("--window-days", type=int, default=15)
    nino34_daily_p.add_argument("--rolling-days", type=int, action="append")
    nino34_daily_p.add_argument("--output-zarr", default="data/processed/zarr/features/nino34_daily_oisst.zarr")
    nino34_daily_p.add_argument("--output-csv", default="data/processed/parquet/features/nino34_daily_oisst.csv")
    nino34_daily_p.add_argument("--dry-run", action="store_true")
    nino34_daily_p.set_defaults(func=cmd_build_nino34_daily_index)

    nino34_ref_p = sub.add_parser(
        "download-nino34-reference",
        help="Download NOAA PSL monthly Nino 3.4 reference index and detected peak labels.",
    )
    nino34_ref_p.add_argument("--raw-path", default="data/raw/cpc_noaa/nino34/nina34.anom.data")
    nino34_ref_p.add_argument("--csv-path", default="data/processed/parquet/features/noaa_psl_nino34_monthly.csv")
    nino34_ref_p.add_argument("--zarr-path", default="data/processed/zarr/features/noaa_psl_nino34_monthly.zarr")
    nino34_ref_p.add_argument("--peaks-csv-path", default="data/processed/parquet/features/noaa_psl_nino34_reference_peaks.csv")
    nino34_ref_p.add_argument("--peaks-zarr-path", default="data/processed/zarr/features/noaa_psl_nino34_reference_peaks.zarr")
    nino34_ref_p.add_argument("--overwrite", action="store_true")
    nino34_ref_p.add_argument("--dry-run", action="store_true")
    nino34_ref_p.set_defaults(func=cmd_download_nino34_reference)

    enso_peak_p = sub.add_parser(
        "build-enso-peak-progression",
        help="Build daily ENSO peak-progression matrix from OISST daily features and NOAA PSL peak labels.",
    )
    enso_peak_p.add_argument("--daily-csv", default="data/processed/parquet/features/nino34_daily_oisst.csv")
    enso_peak_p.add_argument("--peaks-csv", default="data/processed/parquet/features/noaa_psl_nino34_reference_peaks.csv")
    enso_peak_p.add_argument("--lead-days", type=int, action="append")
    enso_peak_p.add_argument("--history-days", type=int, action="append")
    enso_peak_p.add_argument("--output-zarr", default="data/processed/zarr/modeling/enso_peak_progression.zarr")
    enso_peak_p.add_argument("--output-csv", default="data/processed/parquet/modeling/enso_peak_progression.csv")
    enso_peak_p.add_argument("--dry-run", action="store_true")
    enso_peak_p.set_defaults(func=cmd_build_enso_peak_progression)

    regrid_p = sub.add_parser("regrid-zarr", help="Regrid a Zarr cube to the common modeling grid.")
    regrid_p.add_argument("--input", required=True, help="Input Zarr path.")
    regrid_p.add_argument("--output", required=True, help="Output Zarr path.")
    regrid_p.add_argument("--dataset", default="unknown", help="Dataset name for audit ledger.")
    regrid_p.add_argument("--method", choices=["bilinear", "nearest_s2d", "conservative"])
    regrid_p.add_argument("--lon-convention", choices=["0_360", "-180_180"], default="0_360")
    regrid_p.add_argument("--overwrite", action="store_true")
    regrid_p.add_argument("--dry-run", action="store_true")
    regrid_p.set_defaults(func=cmd_regrid_zarr)

    dist_p = sub.add_parser(
        "diagnose-distributions",
        help="Fit power-law tail diagnostics and compare against lognormal/exponential.",
    )
    dist_p.add_argument("--input", required=True, help="Input Zarr path.")
    dist_p.add_argument("--output", default="data/processed/zarr/distributions/distribution_diagnostics.zarr")
    dist_p.add_argument("--csv-output", default="data/processed/parquet/distribution_diagnostics.csv")
    dist_p.add_argument("--dataset", default="unknown", help="Dataset name for audit ledger.")
    dist_p.add_argument("--variable", action="append", help="Variable to diagnose; repeat for many.")
    dist_p.add_argument("--tail", choices=["upper", "absolute"], default="upper")
    dist_p.add_argument("--sample-size", type=int, default=200_000)
    dist_p.add_argument("--min-tail", type=int, default=50)
    dist_p.add_argument("--random-state", type=int, default=42)
    dist_p.add_argument("--dry-run", action="store_true")
    dist_p.set_defaults(func=cmd_diagnose_distributions)

    ibge_p = sub.add_parser("download-ibge", help="Download official IBGE shapefiles.")
    ibge_p.add_argument(
        "--product",
        choices=["uf", "municipios"],
        default="uf",
        help="IBGE product to download.",
    )
    ibge_p.add_argument("--overwrite", action="store_true", help="Replace existing file.")
    ibge_p.add_argument("--no-extract", action="store_true", help="Do not unzip the file.")
    ibge_p.set_defaults(func=cmd_download_ibge)

    chirps_p = sub.add_parser("download-chirps", help="Download CHIRPS annual NetCDF files.")
    chirps_p.add_argument("--start-year", type=int, required=True)
    chirps_p.add_argument("--end-year", type=int)
    chirps_p.add_argument("--resolution", choices=["p25", "p05"])
    chirps_p.add_argument("--execute", action="store_true", help="Actually download files.")
    chirps_p.add_argument("--overwrite", action="store_true")
    chirps_p.add_argument("--continue-on-error", action="store_true", help="Log item failures and keep processing later years.")
    chirps_p.set_defaults(func=cmd_download_chirps)

    oisst_p = sub.add_parser("download-oisst", help="Download NOAA OISST annual NetCDF files.")
    oisst_p.add_argument("--start-year", type=int, required=True)
    oisst_p.add_argument("--end-year", type=int)
    oisst_p.add_argument("--execute", action="store_true", help="Actually download files.")
    oisst_p.add_argument("--overwrite", action="store_true")
    oisst_p.add_argument("--continue-on-error", action="store_true", help="Log item failures and keep processing later years.")
    oisst_p.set_defaults(func=cmd_download_oisst)

    ctd_p = sub.add_parser("download-ctd", help="Download NOAA WOD CTD annual NetCDF and convert to thermocline Zarr.")
    ctd_p.add_argument("--start-year", type=int, required=True)
    ctd_p.add_argument("--end-year", type=int)
    ctd_p.add_argument("--execute", action="store_true", help="Actually download and process files.")
    ctd_p.add_argument("--overwrite", action="store_true")
    ctd_p.add_argument("--raw-only", action="store_true", help="Download raw NetCDF only; skip TEOS-10 Zarr ETL.")
    ctd_p.add_argument("--stop-on-missing-source", action="store_true", help="Fail if NOAA has not published a year.")
    ctd_p.add_argument("--max-depth", type=float, default=THERMOCLINE_MAX_DEPTH_M)
    ctd_p.add_argument("--depth-step", type=float, default=5.0)
    ctd_p.add_argument("--min-levels", type=int, default=THERMOCLINE_MIN_LEVELS)
    ctd_p.add_argument("--good-flag", type=int, action="append")
    ctd_p.add_argument("--require-salinity", action="store_true", help="Require salinity profiles; default keeps temperature-only profiles for thermocline diagnostics.")
    ctd_p.add_argument("--hash", action="store_true", help="Compute SHA256 for raw files.")
    ctd_p.add_argument("--continue-on-error", action="store_true", help="Log item failures and keep processing later years.")
    ctd_p.set_defaults(func=cmd_download_ctd)

    etl_ctd_p = sub.add_parser("etl-ctd", help="Convert existing NOAA WOD CTD raw files to thermocline Zarr.")
    etl_ctd_p.add_argument("--start-year", type=int, required=True)
    etl_ctd_p.add_argument("--end-year", type=int)
    etl_ctd_p.add_argument("--execute", action="store_true", help="Actually process files.")
    etl_ctd_p.add_argument("--overwrite", action="store_true")
    etl_ctd_p.add_argument("--max-depth", type=float, default=THERMOCLINE_MAX_DEPTH_M)
    etl_ctd_p.add_argument("--depth-step", type=float, default=5.0)
    etl_ctd_p.add_argument("--min-levels", type=int, default=THERMOCLINE_MIN_LEVELS)
    etl_ctd_p.add_argument("--good-flag", type=int, action="append")
    etl_ctd_p.add_argument("--require-salinity", action="store_true", help="Require salinity profiles; default keeps temperature-only profiles for thermocline diagnostics.")
    etl_ctd_p.add_argument("--hash", action="store_true", help="Compute SHA256 for raw files.")
    etl_ctd_p.add_argument("--continue-on-error", action="store_true", help="Log item failures and keep processing later years.")
    etl_ctd_p.set_defaults(func=cmd_etl_ctd)

    era5_p = sub.add_parser("download-era5", help="Download ERA5 files through CDS.")
    era5_p.add_argument("--start-year", type=int, required=True)
    era5_p.add_argument("--end-year", type=int)
    era5_p.add_argument("--month", type=int, action="append", choices=range(1, 13))
    era5_p.add_argument("--kind", choices=["single", "pressure", "both"], default="both")
    era5_p.add_argument("--region", action="append", choices=sorted(ATMOSPHERE_AREAS))
    era5_p.add_argument(
        "--variable",
        action="append",
        choices=sorted(set(ERA5_SINGLE_VARIABLES + ERA5_PRESSURE_VARIABLES)),
        help="Download/ingest one ERA5 variable; repeat for many. Omit to request all variables for the selected kind.",
    )
    era5_p.add_argument("--execute", action="store_true", help="Actually submit CDS requests.")
    era5_p.add_argument("--overwrite", action="store_true")
    era5_p.add_argument("--raw-only", action="store_true", help="Skip Zarr conversion.")
    era5_p.add_argument("--annual-zarr", action="store_true", help="Cache yearly ERA5 files and write annual daily Zarr stores.")
    era5_p.add_argument(
        "--delete-raw-after-zarr",
        action="store_true",
        help="Delete each raw annual NetCDF cache after its yearly variable Zarr is validated.",
    )
    era5_p.add_argument(
        "--request-mode",
        choices=["annual-kind", "annual-auto", "annual-variable", "monthly-kind"],
        default="annual-kind",
        help=(
            "With --annual-zarr, request one full year per kind/region by default. "
            "Use annual-auto to try the largest accepted yearly request and split on CDS cost/payload rejection."
        ),
    )
    era5_p.add_argument("--hash", action="store_true", help="Compute SHA256 for raw files.")
    era5_p.add_argument("--continue-on-error", action="store_true", help="Log item failures and keep processing later tasks.")
    era5_p.set_defaults(func=cmd_download_era5)

    validation_p = sub.add_parser("download-validation", help="Download in-situ validation data for Nino 3.4 from TAO/TRITON and Argo.")
    validation_p.add_argument("--source", choices=["tao_triton", "argo", "all"], default="all")
    validation_p.add_argument("--start-year", type=int, required=True)
    validation_p.add_argument("--end-year", type=int)
    validation_p.add_argument("--end-date", help="Final date for the last year, YYYY-MM-DD. Defaults to today.")
    validation_p.add_argument("--max-depth", type=float, default=300.0)
    validation_p.add_argument(
        "--tao-product",
        action="append",
        choices=["temperature", "salinity"],
        help="TAO/TRITON product to fetch; repeat for many. Defaults to temperature and salinity.",
    )
    validation_p.add_argument("--argo-start-year", type=int, default=1999, help="First Argo year to query when --source is argo/all.")
    validation_p.add_argument("--execute", action="store_true", help="Actually download files.")
    validation_p.add_argument("--overwrite", action="store_true")
    validation_p.add_argument("--hash", action="store_true", help="Compute SHA256 for raw files.")
    validation_p.add_argument("--continue-on-error", action="store_true", help="Log item failures and keep processing later tasks.")
    validation_p.set_defaults(func=cmd_download_validation)

    all_p = sub.add_parser("download-all", help="Run the full download plan.")
    all_p.add_argument("--start-year", type=int, required=True)
    all_p.add_argument("--end-year", type=int)
    all_p.add_argument("--month", type=int, action="append", choices=range(1, 13))
    all_p.add_argument("--chirps-resolution", choices=["p25", "p05"])
    all_p.add_argument("--include-cds", action="store_true", help="Include ERA5.")
    all_p.add_argument("--execute", action="store_true", help="Actually download/submit requests.")
    all_p.add_argument("--overwrite", action="store_true")
    all_p.add_argument("--hash", action="store_true", help="Compute SHA256 for raw CDS files.")
    all_p.add_argument("--continue-on-error", action="store_true", help="Log item failures and keep processing later tasks.")
    all_p.set_defaults(func=cmd_download_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
