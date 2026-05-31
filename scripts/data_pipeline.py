from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from tqdm import tqdm
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.config import load_config, project_path
from nino_brasil.data.audit import AuditLog, dataset_summary
from nino_brasil.data.availability import iter_available_years, record_source_latency
from nino_brasil.data.credentials import cds_credentials_status
from nino_brasil.data.download_cds import (
    ATMOSPHERE_AREAS,
    download_era5_pressure_month,
    download_era5_single_month,
    download_oras_month,
    ingest_era5_pressure_month,
    ingest_era5_single_month,
    ingest_oras_month,
)
from nino_brasil.data.download_chirps import download_chirps_year
from nino_brasil.data.download_ctd_noaa import (
    download_wod_ctd_year,
    etl_wod_ctd_year,
)
from nino_brasil.data.download_ibge import download_ibge
from nino_brasil.data.download_oisst import download_oisst_year
from nino_brasil.data.regrid import normalize_for_common_grid, regrid_dataset, target_grid_from_config
from nino_brasil.features.distributions import diagnose_dataset_distributions


DATA_DIRS = [
    "data/raw/ibge",
    "data/raw/chirps",
    "data/raw/cpc_noaa",
    "data/raw/oras",
    "data/raw/era5",
    "data/raw/ctd_noaa/wod",
    "data/interim/oras",
    "data/interim/ibge",
    "data/interim/ctd_noaa",
    "data/interim/brazil_precipitation",
    "data/interim/pacific_warming",
    "data/interim/atmosphere_bridge",
    "data/processed/zarr",
    "data/processed/zarr/ctd_noaa",
    "data/processed/zarr/ctd_noaa/wod",
    "data/processed/parquet",
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
    print("5. download-oisst: Pacific daily SST/SSTA primary source")
    print("6. ingest-era5: download raw monthly chunks, validate, convert to Zarr")
    print("7. ingest-oras: download raw monthly chunks, validate, convert to Zarr")
    print("8. download-ctd: WOD CTD annual raw files, QC, TEOS-10 and Zarr")
    print("9. regrid-zarr: reconcile each modeling cube to the common grid")
    print("10. diagnose-distributions: optional QC/EDA tail diagnostics")
    print("11. model_pipeline.py: walk-forward metrics, predictions and importances")
    print("12. audit: verify status and failed tasks before continuing")
    return 0


def iter_years(
    start_year: int,
    end_year: int | None,
    source: str | None = None,
    cfg: dict | None = None,
) -> range:
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
                regridded.to_zarr(output_path, mode="w", consolidated=True)
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
    except BaseException as exc:
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
        diagnostics.to_parquet(output_path, index=False)
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
    except BaseException as exc:
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
        download_chirps_year(
            year=year,
            raw_dir=raw_dir,
            resolution=resolution,
            overwrite=args.overwrite,
            dry_run=not args.execute,
        )
    return 0


def cmd_download_oisst(args: argparse.Namespace) -> int:
    cfg = load_config()
    audit = AuditLog()
    record_source_latency(audit, "noaa_oisst", cfg)
    raw_dir = project_path("data/raw/cpc_noaa/oisst")
    for year in iter_years(args.start_year, args.end_year, "noaa_oisst", cfg):
        download_oisst_year(
            year=year,
            raw_dir=raw_dir,
            overwrite=args.overwrite,
            dry_run=not args.execute,
        )
    return 0


def cmd_download_ctd(args: argparse.Namespace) -> int:
    cfg = load_config()
    raw_dir = project_path("data/raw/ctd_noaa")
    zarr_root = project_path("data/processed/zarr")
    audit = AuditLog()
    record_source_latency(audit, "noaa_wod_ctd", cfg)
    for year in tqdm(iter_years(args.start_year, args.end_year, "noaa_wod_ctd", cfg), desc="WOD CTD years", unit="year"):
        raw_path = download_wod_ctd_year(
            year=year,
            raw_dir=raw_dir,
            dry_run=not args.execute,
            overwrite=args.overwrite,
            allow_missing_source=not args.stop_on_missing_source,
            include_hash=args.hash,
            audit=audit,
        )
        if args.raw_only:
            continue
        if args.execute and not raw_path.exists():
            continue
        etl_wod_ctd_year(
            year=year,
            raw_dir=raw_dir,
            zarr_root=zarr_root,
            dry_run=not args.execute,
            overwrite=args.overwrite,
            max_depth_m=args.max_depth,
            depth_step_m=args.depth_step,
            min_levels=args.min_levels,
            good_flags=args.good_flag or [0],
            include_hash=args.hash,
            audit=audit,
        )
    return 0


def cmd_etl_ctd(args: argparse.Namespace) -> int:
    cfg = load_config()
    raw_dir = project_path("data/raw/ctd_noaa")
    zarr_root = project_path("data/processed/zarr")
    audit = AuditLog()
    record_source_latency(audit, "noaa_wod_ctd", cfg)
    for year in tqdm(iter_years(args.start_year, args.end_year, "noaa_wod_ctd", cfg), desc="WOD CTD ETL years", unit="year"):
        etl_wod_ctd_year(
            year=year,
            raw_dir=raw_dir,
            zarr_root=zarr_root,
            dry_run=not args.execute,
            overwrite=args.overwrite,
            max_depth_m=args.max_depth,
            depth_step_m=args.depth_step,
            min_levels=args.min_levels,
            good_flags=args.good_flag or [0],
            include_hash=args.hash,
            audit=audit,
        )
    return 0


def cmd_download_era5(args: argparse.Namespace) -> int:
    cfg = load_config()
    raw_dir = project_path("data/raw/era5")
    zarr_root = project_path("data/processed/zarr")
    regions = args.region or list(ATMOSPHERE_AREAS)
    months = iter_months(args.month)
    audit = AuditLog()
    record_source_latency(audit, "era5", cfg)
    tasks = [
        (year, month, region)
        for year in iter_years(args.start_year, args.end_year, "era5", cfg)
        for month in months
        for region in regions
    ]
    for year, month, region in tqdm(tasks, desc="ERA5 tasks", unit="task"):
        if args.raw_only:
            if args.kind in {"single", "both"}:
                download_era5_single_month(
                    year=year,
                    month=month,
                    region=region,
                    raw_dir=raw_dir,
                    dry_run=not args.execute,
                    overwrite=args.overwrite,
                )
            if args.kind in {"pressure", "both"}:
                download_era5_pressure_month(
                    year=year,
                    month=month,
                    region=region,
                    raw_dir=raw_dir,
                    dry_run=not args.execute,
                    overwrite=args.overwrite,
                )
        else:
            if args.kind in {"single", "both"}:
                ingest_era5_single_month(
                    year=year,
                    month=month,
                    region=region,
                    raw_dir=raw_dir,
                    zarr_root=zarr_root,
                    dry_run=not args.execute,
                    overwrite=args.overwrite,
                    include_hash=args.hash,
                    audit=audit,
                )
            if args.kind in {"pressure", "both"}:
                ingest_era5_pressure_month(
                    year=year,
                    month=month,
                    region=region,
                    raw_dir=raw_dir,
                    zarr_root=zarr_root,
                    dry_run=not args.execute,
                    overwrite=args.overwrite,
                    include_hash=args.hash,
                    audit=audit,
                )
    return 0


def cmd_download_oras(args: argparse.Namespace) -> int:
    cfg = load_config()
    raw_dir = project_path("data/raw/oras")
    interim_dir = project_path("data/interim/oras")
    zarr_root = project_path("data/processed/zarr")
    months = iter_months(args.month)
    audit = AuditLog()
    record_source_latency(audit, "oras5", cfg)
    tasks = [
        (year, month)
        for year in iter_years(args.start_year, args.end_year, "oras5", cfg)
        for month in months
    ]
    for year, month in tqdm(tasks, desc="ORAS tasks", unit="task"):
        if args.raw_only:
            download_oras_month(
                year=year,
                month=month,
                raw_dir=raw_dir,
                dry_run=not args.execute,
                overwrite=args.overwrite,
            )
        else:
            ingest_oras_month(
                year=year,
                month=month,
                raw_dir=raw_dir,
                interim_dir=interim_dir,
                zarr_root=zarr_root,
                dry_run=not args.execute,
                overwrite=args.overwrite,
                include_hash=args.hash,
                audit=audit,
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
        download_ibge(
            product_id=product,
            raw_dir=raw_ibge,
            interim_dir=interim_ibge,
            extract=True,
            overwrite=args.overwrite,
            dry_run=dry_run,
        )

    audit = AuditLog()
    record_source_latency(audit, "chirps", cfg)
    for year in iter_years(args.start_year, args.end_year, "chirps", cfg):
        download_chirps_year(
            year=year,
            raw_dir=project_path("data/raw/chirps"),
            resolution=chirps_resolution,
            overwrite=args.overwrite,
            dry_run=dry_run,
        )
    record_source_latency(audit, "noaa_oisst", cfg)
    for year in iter_years(args.start_year, args.end_year, "noaa_oisst", cfg):
        download_oisst_year(
            year=year,
            raw_dir=project_path("data/raw/cpc_noaa/oisst"),
            overwrite=args.overwrite,
            dry_run=dry_run,
        )

    if args.include_cds:
        months = iter_months(args.month)
        record_source_latency(audit, "era5", cfg)
        tasks = [
            (year, month)
            for year in iter_years(args.start_year, args.end_year, "era5", cfg)
            for month in months
        ]
        for year, month in tqdm(tasks, desc="CDS ingest months", unit="month"):
            for region in ATMOSPHERE_AREAS:
                ingest_era5_single_month(
                    year=year,
                    month=month,
                    region=region,
                    raw_dir=project_path("data/raw/era5"),
                    zarr_root=project_path("data/processed/zarr"),
                    dry_run=dry_run,
                    overwrite=args.overwrite,
                    include_hash=args.hash,
                    audit=audit,
                )
                ingest_era5_pressure_month(
                    year=year,
                    month=month,
                    region=region,
                    raw_dir=project_path("data/raw/era5"),
                    zarr_root=project_path("data/processed/zarr"),
                    dry_run=dry_run,
                    overwrite=args.overwrite,
                    include_hash=args.hash,
                    audit=audit,
                )
        record_source_latency(audit, "oras5", cfg)
        oras_tasks = [
            (year, month)
            for year in iter_years(args.start_year, args.end_year, "oras5", cfg)
            for month in months
        ]
        for year, month in tqdm(oras_tasks, desc="ORAS ingest months", unit="month"):
            ingest_oras_month(
                year=year,
                month=month,
                raw_dir=project_path("data/raw/oras"),
                interim_dir=project_path("data/interim/oras"),
                zarr_root=project_path("data/processed/zarr"),
                dry_run=dry_run,
                overwrite=args.overwrite,
                include_hash=args.hash,
                audit=audit,
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
    dist_p.add_argument("--output", default="data/processed/parquet/distributions/distribution_diagnostics.parquet")
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
    chirps_p.set_defaults(func=cmd_download_chirps)

    oisst_p = sub.add_parser("download-oisst", help="Download NOAA OISST annual NetCDF files.")
    oisst_p.add_argument("--start-year", type=int, required=True)
    oisst_p.add_argument("--end-year", type=int)
    oisst_p.add_argument("--execute", action="store_true", help="Actually download files.")
    oisst_p.add_argument("--overwrite", action="store_true")
    oisst_p.set_defaults(func=cmd_download_oisst)

    ctd_p = sub.add_parser("download-ctd", help="Download NOAA WOD CTD annual NetCDF and convert to Zarr.")
    ctd_p.add_argument("--start-year", type=int, required=True)
    ctd_p.add_argument("--end-year", type=int)
    ctd_p.add_argument("--execute", action="store_true", help="Actually download and process files.")
    ctd_p.add_argument("--overwrite", action="store_true")
    ctd_p.add_argument("--raw-only", action="store_true", help="Download raw NetCDF only; skip TEOS-10 Zarr ETL.")
    ctd_p.add_argument("--stop-on-missing-source", action="store_true", help="Fail if NOAA has not published a year.")
    ctd_p.add_argument("--max-depth", type=float, default=700.0)
    ctd_p.add_argument("--depth-step", type=float, default=5.0)
    ctd_p.add_argument("--min-levels", type=int, default=5)
    ctd_p.add_argument("--good-flag", type=int, action="append")
    ctd_p.add_argument("--hash", action="store_true", help="Compute SHA256 for raw files.")
    ctd_p.set_defaults(func=cmd_download_ctd)

    etl_ctd_p = sub.add_parser("etl-ctd", help="Convert existing NOAA WOD CTD raw files to TEOS-10 Zarr.")
    etl_ctd_p.add_argument("--start-year", type=int, required=True)
    etl_ctd_p.add_argument("--end-year", type=int)
    etl_ctd_p.add_argument("--execute", action="store_true", help="Actually process files.")
    etl_ctd_p.add_argument("--overwrite", action="store_true")
    etl_ctd_p.add_argument("--max-depth", type=float, default=700.0)
    etl_ctd_p.add_argument("--depth-step", type=float, default=5.0)
    etl_ctd_p.add_argument("--min-levels", type=int, default=5)
    etl_ctd_p.add_argument("--good-flag", type=int, action="append")
    etl_ctd_p.add_argument("--hash", action="store_true", help="Compute SHA256 for raw files.")
    etl_ctd_p.set_defaults(func=cmd_etl_ctd)

    era5_p = sub.add_parser("download-era5", help="Download ERA5 monthly files through CDS.")
    era5_p.add_argument("--start-year", type=int, required=True)
    era5_p.add_argument("--end-year", type=int)
    era5_p.add_argument("--month", type=int, action="append", choices=range(1, 13))
    era5_p.add_argument("--kind", choices=["single", "pressure", "both"], default="both")
    era5_p.add_argument("--region", action="append", choices=sorted(ATMOSPHERE_AREAS))
    era5_p.add_argument("--execute", action="store_true", help="Actually submit CDS requests.")
    era5_p.add_argument("--overwrite", action="store_true")
    era5_p.add_argument("--raw-only", action="store_true", help="Skip Zarr conversion.")
    era5_p.add_argument("--hash", action="store_true", help="Compute SHA256 for raw files.")
    era5_p.set_defaults(func=cmd_download_era5)

    oras_p = sub.add_parser("download-oras", help="Download ORAS5 monthly files through CDS.")
    oras_p.add_argument("--start-year", type=int, required=True)
    oras_p.add_argument("--end-year", type=int)
    oras_p.add_argument("--month", type=int, action="append", choices=range(1, 13))
    oras_p.add_argument("--execute", action="store_true", help="Actually submit CDS requests.")
    oras_p.add_argument("--overwrite", action="store_true")
    oras_p.add_argument("--raw-only", action="store_true", help="Skip Zarr conversion.")
    oras_p.add_argument("--hash", action="store_true", help="Compute SHA256 for raw files.")
    oras_p.set_defaults(func=cmd_download_oras)

    all_p = sub.add_parser("download-all", help="Run the full download plan.")
    all_p.add_argument("--start-year", type=int, required=True)
    all_p.add_argument("--end-year", type=int)
    all_p.add_argument("--month", type=int, action="append", choices=range(1, 13))
    all_p.add_argument("--chirps-resolution", choices=["p25", "p05"])
    all_p.add_argument("--include-cds", action="store_true", help="Include ERA5 and ORAS.")
    all_p.add_argument("--execute", action="store_true", help="Actually download/submit requests.")
    all_p.add_argument("--overwrite", action="store_true")
    all_p.add_argument("--hash", action="store_true", help="Compute SHA256 for raw CDS files.")
    all_p.set_defaults(func=cmd_download_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
