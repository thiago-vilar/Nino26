from __future__ import annotations

import argparse
import calendar
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, TypeVar

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
    ORAS5_VARIABLES,
    download_era5_pressure_month,
    download_era5_single_month,
    download_oras_month,
    ingest_era5_pressure_month,
    ingest_era5_pressure_year,
    ingest_era5_pressure_year_kind,
    ingest_era5_pressure_year_variable,
    ingest_era5_single_month,
    ingest_era5_single_year,
    ingest_era5_single_year_kind,
    ingest_era5_single_year_variable,
    ingest_oras_month,
    ingest_oras_year_kind,
    ingest_oras_year_variable,
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
from nino_brasil.data.download_validation_insitu import download_argo_year, download_tao_triton_year
from nino_brasil.data.regrid import normalize_for_common_grid, regrid_dataset, target_grid_from_config
from nino_brasil.data.zarr_store import chunk_plan, dataframe_to_zarr


T = TypeVar("T")
from nino_brasil.features.distributions import diagnose_dataset_distributions


DATA_DIRS = [
    "data/raw/ibge",
    "data/raw/chirps",
    "data/raw/cpc_noaa",
    "data/raw/oras",
    "data/raw/era5",
    "data/raw/ctd_noaa/wod",
    "data/raw/tao_triton",
    "data/raw/argo",
    "data/interim/oras",
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
    print("7. ingest-oras: request annual-kind by year, split to annual daily Zarr by variable")
    print("8. download-ctd: WOD CTD annual raw files, thermocline QC and Zarr")
    print("9. download-validation: TAO/TRITON and Argo in-situ validation for Nino 3.4")
    print("10. regrid-zarr: reconcile each modeling cube to the common modeling grid")
    print("11. diagnose-distributions: optional QC/EDA tail diagnostics")
    print("12. phase 3: Nino 3.4 anomaly alignment, thermocline, slope and signal duration diagnostics")
    print("13. phase 4: exploratory statistics with regression, PCA, KNN and variable screening")
    print("14. model_pipeline.py: phase 5 walk-forward metrics, predictions and importances")
    print("15. audit: verify status and failed tasks before continuing")
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
            if output_path.exists() and args.overwrite:
                shutil.rmtree(output_path)
            if output_path.exists() and not args.overwrite:
                dataset_summary(output_path, zarr=True)
                print(f"zarr exists: {output_path}")
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                chunks = chunk_plan(regridded)
                if chunks:
                    regridded = regridded.chunk(chunks)
                regridded.to_zarr(output_path, mode="w", consolidated=True, zarr_format=2)
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
    audit = AuditLog()
    task_id = f"diagnose_distributions_{input_path.stem}"

    if args.dry_run:
        print(f"DRY RUN diagnose distributions: {input_path} -> {output_path}")
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
        audit.record(
            task_id=task_id,
            dataset=args.dataset,
            status="ok",
            input_path=str(input_path),
            output_path=str(output_path),
            rows=int(len(diagnostics)),
        )
        print(f"distribution diagnostics: {output_path}")
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


def cmd_download_oras(args: argparse.Namespace) -> int:
    cfg = load_config()
    raw_dir = project_path("data/raw/oras")
    interim_dir = project_path("data/interim/oras")
    zarr_root = project_path("data/processed/zarr")
    months = iter_months(args.month)
    if args.annual_zarr and args.raw_only:
        raise ValueError("--annual-zarr cannot be combined with --raw-only.")
    if args.annual_zarr and args.month and sorted(set(args.month)) != list(range(1, 13)):
        raise ValueError("--annual-zarr needs the full year; omit --month or pass all 12 months.")
    variables = _selected_variables(args.variable, ORAS5_VARIABLES)
    if args.variable and not variables:
        raise ValueError(f"No requested variable belongs to ORAS5: {args.variable}")
    audit = AuditLog()
    record_source_latency(audit, "oras5", cfg)

    if args.annual_zarr:
        years = iter_complete_annual_years(args.start_year, args.end_year, "oras5", cfg)
        if args.request_mode == "annual-kind":
            years_list = list(years)
            print(f"ORAS annual-kind CDS requests: {len(years_list)}")
            for year in tqdm(years_list, desc="ORAS annual-kind years", unit="year"):
                run_or_continue(
                    f"ORAS5 annual-kind {year}",
                    lambda year=year: ingest_oras_year_kind(
                        year=year,
                        raw_dir=raw_dir,
                        interim_dir=interim_dir,
                        zarr_root=zarr_root,
                        months=months,
                        variables=variables,
                        dry_run=not args.execute,
                        overwrite=args.overwrite,
                        include_hash=args.hash,
                        delete_raw_after_zarr=args.delete_raw_after_zarr,
                        audit=audit,
                    ),
                    continue_on_error=args.continue_on_error,
                )
            return 0

        task_variables = variables or ORAS5_VARIABLES
        tasks = [
            (year, variable)
            for year in years
            for variable in task_variables
        ]
        print(f"ORAS annual variable tasks: {len(tasks)}")
        for year, variable in tasks:
            run_or_continue(
                f"ORAS5 annual {year} {variable}",
                lambda year=year, variable=variable: ingest_oras_year_variable(
                    year=year,
                    variable=variable,
                    raw_dir=raw_dir,
                    interim_dir=interim_dir,
                    zarr_root=zarr_root,
                    months=months,
                    dry_run=not args.execute,
                    overwrite=args.overwrite,
                    include_hash=args.hash,
                    delete_raw_after_zarr=args.delete_raw_after_zarr,
                    audit=audit,
                ),
                continue_on_error=args.continue_on_error,
            )
        return 0

    year_months = iter_complete_year_months(args.start_year, args.end_year, "oras5", cfg, args.month)
    tasks = [
        (year, month)
        for year, month in year_months
    ]
    for year, month in tqdm(tasks, desc="ORAS tasks", unit="task"):
        if args.raw_only:
            run_or_continue(
                f"ORAS5 raw {year}-{month:02d}",
                lambda year=year, month=month: download_oras_month(
                    year=year,
                    month=month,
                    raw_dir=raw_dir,
                    variables=variables,
                    dry_run=not args.execute,
                    overwrite=args.overwrite,
                ),
                continue_on_error=args.continue_on_error,
            )
        else:
            run_or_continue(
                f"ORAS5 {year}-{month:02d}",
                lambda year=year, month=month: ingest_oras_month(
                    year=year,
                    month=month,
                    raw_dir=raw_dir,
                    interim_dir=interim_dir,
                    zarr_root=zarr_root,
                    variables=variables,
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
        record_source_latency(audit, "oras5", cfg)
        oras_tasks = [
            (year, month)
            for year in iter_years(args.start_year, args.end_year, "oras5", cfg)
            for month in months
        ]
        for year, month in tqdm(oras_tasks, desc="ORAS ingest months", unit="month"):
            for variable in ORAS5_VARIABLES:
                run_or_continue(
                    f"ORAS5 {year}-{month:02d} {variable}",
                    lambda year=year, month=month, variable=variable: ingest_oras_month(
                        year=year,
                        month=month,
                        raw_dir=project_path("data/raw/oras"),
                        interim_dir=project_path("data/interim/oras"),
                        zarr_root=project_path("data/processed/zarr"),
                        variables=[variable],
                        dry_run=dry_run,
                        overwrite=args.overwrite,
                        include_hash=args.hash,
                        audit=audit,
                    ),
                    continue_on_error=args.continue_on_error,
                )
    else:
        print("CDS downloads skipped. Add --include-cds to include ERA5 and ORAS.")

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
        choices=["annual-kind", "annual-variable", "monthly-kind"],
        default="annual-kind",
        help="With --annual-zarr, request one full year per kind/region by default; use annual-variable or monthly-kind as fallbacks.",
    )
    era5_p.add_argument("--hash", action="store_true", help="Compute SHA256 for raw files.")
    era5_p.add_argument("--continue-on-error", action="store_true", help="Log item failures and keep processing later tasks.")
    era5_p.set_defaults(func=cmd_download_era5)

    oras_p = sub.add_parser("download-oras", help="Download ORAS5 files through CDS.")
    oras_p.add_argument("--start-year", type=int, required=True)
    oras_p.add_argument("--end-year", type=int)
    oras_p.add_argument("--month", type=int, action="append", choices=range(1, 13))
    oras_p.add_argument(
        "--variable",
        action="append",
        choices=ORAS5_VARIABLES,
        help="Download/ingest one ORAS5 variable; repeat for many. Omit to request all variables.",
    )
    oras_p.add_argument("--execute", action="store_true", help="Actually submit CDS requests.")
    oras_p.add_argument("--overwrite", action="store_true")
    oras_p.add_argument("--raw-only", action="store_true", help="Skip Zarr conversion.")
    oras_p.add_argument("--annual-zarr", action="store_true", help="Cache yearly ORAS5 files and write annual daily Zarr stores by variable.")
    oras_p.add_argument(
        "--delete-raw-after-zarr",
        action="store_true",
        help="Delete each raw annual ZIP cache after its yearly variable Zarr is validated.",
    )
    oras_p.add_argument(
        "--request-mode",
        choices=["annual-kind", "annual-variable"],
        default="annual-kind",
        help="With --annual-zarr, request one full year with all selected variables by default; use annual-variable as fallback.",
    )
    oras_p.add_argument("--hash", action="store_true", help="Compute SHA256 for raw files.")
    oras_p.add_argument("--continue-on-error", action="store_true", help="Log item failures and keep processing later tasks.")
    oras_p.set_defaults(func=cmd_download_oras)

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
    all_p.add_argument("--include-cds", action="store_true", help="Include ERA5 and ORAS.")
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
