from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.data.download_ocean_daily import (
    GLORYS_DEFAULT_VARIABLES,
    GLORYS_START_YEAR,
    UFS_END_YEAR,
    UFS_START_YEAR,
    build_ocean_daily_features,
    download_glorys_operational,
    download_glorys_years,
    glorys_zarr_path,
    ingest_ufs_years,
    process_glorys_year,
    process_glorys_operational,
    ufs_zarr_path,
)
from nino_brasil.data.credentials import load_local_env


RAW_GLORYS_ROOT = ROOT / "data/raw/ocean_daily/glorys12"
RAW_OPERATIONAL_ROOT = ROOT / "data/raw/ocean_daily/glorys12_operational"
PROCESSED_GLORYS_ROOT = ROOT / "data/processed/zarr/ocean_daily/glorys12"
PROCESSED_OPERATIONAL_ROOT = ROOT / "data/processed/zarr/ocean_daily/glorys12_operational"
PROCESSED_UFS_ROOT = ROOT / "data/processed/zarr/ocean_daily/noaa_ufs"
FEATURE_ROOT = ROOT / "data/processed/zarr/features/ocean_daily"


def _years(start: int, end: int) -> list[int]:
    if end < start:
        raise ValueError("end-year must be greater than or equal to start-year.")
    return list(range(start, end + 1))


def _feature_path(source: str, year: int) -> Path:
    return FEATURE_ROOT / source / str(year) / f"{source}_ocean_features_{year}_daily.zarr"


def _delete_validated_raw(path: Path, root: Path) -> None:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved_root not in resolved.parents:
        raise ValueError(f"Refusing to delete outside {resolved_root}: {resolved}")
    if path.exists():
        shutil.rmtree(path)
        print(f"validated raw cache deleted: {path}")


def cmd_plan(args: argparse.Namespace) -> int:
    ufs_start = args.start_year
    ufs_end = min(args.transition_year - 1, args.end_year)
    glorys_start = max(args.transition_year, args.start_year)
    glorys_end = args.end_year
    ufs_years = _years(ufs_start, ufs_end) if ufs_start <= ufs_end else []
    glorys_years = _years(glorys_start, glorys_end) if glorys_start <= glorys_end else []
    print("Daily ocean contract: no monthly/weekly expansion to daily.")
    if ufs_years:
        print(
            f"NOAA UFS: {len(ufs_years)} annual Zarr tasks ({ufs_years[0]}-{ufs_years[-1]}), "
            "bilinearly aligned from the native 1-degree grid to canonical 0.25-degree nodes."
        )
    if glorys_years:
        print(
            f"GLORYS12: {len(glorys_years)} annual Copernicus Marine requests "
            f"({glorys_years[0]}-{glorys_years[-1]}), with thetao+so+zos grouped in each request."
        )
    print("Domain: 5S-5N, 120E-80W; raw depth: 0-800 m; OHC analysis layers: 0-300/0-700 m.")
    print("Canonical comparison grid: 0.25 degree for UFS, GLORYS12 and GLO12; interpolation adds no UFS detail.")
    print("The source transition remains explicit in separate stores and source metadata.")
    return 0


def cmd_download_glorys(args: argparse.Namespace) -> int:
    download_glorys_years(
        years=_years(args.start_year, args.end_year),
        output_root=RAW_GLORYS_ROOT,
        variables=args.variable or list(GLORYS_DEFAULT_VARIABLES),
        end_date=args.end_date,
        execute=args.execute,
        overwrite=args.overwrite,
    )
    return 0


def cmd_process_glorys(args: argparse.Namespace) -> int:
    for year in _years(args.start_year, args.end_year):
        source = glorys_zarr_path(RAW_GLORYS_ROOT, year)
        output = PROCESSED_GLORYS_ROOT / str(year) / f"glorys12_equatorial_pacific_{year}_daily_0p25.zarr"
        process_glorys_year(source, output, overwrite=args.overwrite)
        if args.build_features:
            build_ocean_daily_features(output, _feature_path("glorys12", year), overwrite=args.overwrite)
        if args.delete_source_after_zarr:
            _delete_validated_raw(source, RAW_GLORYS_ROOT)
    return 0


def cmd_ingest_glorys(args: argparse.Namespace) -> int:
    """Download, process, validate, and optionally release one annual cache at a time."""
    for year in _years(args.start_year, args.end_year):
        download_glorys_years(
            years=[year],
            output_root=RAW_GLORYS_ROOT,
            variables=args.variable or list(GLORYS_DEFAULT_VARIABLES),
            end_date=args.end_date,
            execute=args.execute,
            overwrite=args.overwrite,
        )
        if not args.execute:
            continue
        source = glorys_zarr_path(RAW_GLORYS_ROOT, year)
        output = PROCESSED_GLORYS_ROOT / str(year) / f"glorys12_equatorial_pacific_{year}_daily_0p25.zarr"
        process_glorys_year(source, output, overwrite=args.overwrite)
        build_ocean_daily_features(output, _feature_path("glorys12", year), overwrite=args.overwrite)
        if args.delete_source_after_zarr:
            _delete_validated_raw(source, RAW_GLORYS_ROOT)
    return 0


def cmd_ingest_ufs(args: argparse.Namespace) -> int:
    years = _years(args.start_year, args.end_year)
    outputs = ingest_ufs_years(
        years=years,
        output_root=PROCESSED_UFS_ROOT,
        execute=args.execute,
        overwrite=args.overwrite,
        block_size_mb=args.block_size_mb,
    )
    if args.execute and args.build_features:
        for year, output in zip(years, outputs, strict=True):
            build_ocean_daily_features(output, _feature_path("noaa_ufs", year), overwrite=args.overwrite)
    return 0


def cmd_ingest_glorys_operational(args: argparse.Namespace) -> int:
    import pandas as pd

    start = pd.Timestamp(args.start_date).normalize()
    end = pd.Timestamp(args.end_date).normalize() if args.end_date else pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    sources = download_glorys_operational(
        start_date=start,
        end_date=end,
        output_root=RAW_OPERATIONAL_ROOT,
        execute=args.execute,
        overwrite=args.overwrite,
    )
    if not args.execute:
        return 0
    slug = f"{start:%Y%m%d}_{end:%Y%m%d}"
    output = PROCESSED_OPERATIONAL_ROOT / str(start.year) / f"glorys12_operational_{slug}_daily_0p25.zarr"
    process_glorys_operational(sources, output, overwrite=args.overwrite)
    feature = FEATURE_ROOT / "glorys12_operational" / str(start.year) / f"glorys12_operational_ocean_features_{slug}_daily.zarr"
    build_ocean_daily_features(output, feature, overwrite=args.overwrite)
    if args.delete_source_after_zarr:
        for source in sources.values():
            _delete_validated_raw(source, RAW_OPERATIONAL_ROOT)
    return 0


def cmd_build_features(args: argparse.Namespace) -> int:
    for year in _years(args.start_year, args.end_year):
        if args.source == "noaa_ufs":
            source = ufs_zarr_path(PROCESSED_UFS_ROOT, year)
        else:
            source = PROCESSED_GLORYS_ROOT / str(year) / f"glorys12_equatorial_pacific_{year}_daily_0p25.zarr"
        build_ocean_daily_features(source, _feature_path(args.source, year), overwrite=args.overwrite)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Daily ocean reanalysis pipeline. Coarse-frequency promotion to daily is forbidden."
    )
    sub = root.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Show the minimum viable annual request plan.")
    plan.add_argument("--start-year", type=int, default=1981)
    plan.add_argument("--end-year", type=int, default=2025)
    plan.add_argument("--transition-year", type=int, default=1993)
    plan.set_defaults(func=cmd_plan)

    download = sub.add_parser("download-glorys", help="Download grouped annual GLORYS daily Zarr stores.")
    download.add_argument("--start-year", type=int, required=True, choices=range(GLORYS_START_YEAR, 2101))
    download.add_argument("--end-year", type=int, required=True)
    download.add_argument("--end-date")
    download.add_argument("--variable", action="append", choices=list(GLORYS_DEFAULT_VARIABLES))
    download.add_argument("--execute", action="store_true")
    download.add_argument("--overwrite", action="store_true")
    download.set_defaults(func=cmd_download_glorys)

    process = sub.add_parser("process-glorys", help="Coarsen downloaded GLORYS daily stores to 0.25 degree Zarr.")
    process.add_argument("--start-year", type=int, required=True)
    process.add_argument("--end-year", type=int, required=True)
    process.add_argument("--build-features", action="store_true")
    process.add_argument("--delete-source-after-zarr", action="store_true")
    process.add_argument("--overwrite", action="store_true")
    process.set_defaults(func=cmd_process_glorys)

    ingest = sub.add_parser("ingest-glorys", help="Run one annual GLORYS request at a time and build validated Zarr/features.")
    ingest.add_argument("--start-year", type=int, required=True, choices=range(GLORYS_START_YEAR, 2101))
    ingest.add_argument("--end-year", type=int, required=True)
    ingest.add_argument("--end-date")
    ingest.add_argument("--variable", action="append", choices=list(GLORYS_DEFAULT_VARIABLES))
    ingest.add_argument("--execute", action="store_true")
    ingest.add_argument("--overwrite", action="store_true")
    ingest.add_argument("--delete-source-after-zarr", action="store_true")
    ingest.set_defaults(func=cmd_ingest_glorys)

    ufs = sub.add_parser("ingest-ufs", help="Stream selected NOAA UFS daily members directly into annual Zarr.")
    ufs.add_argument("--start-year", type=int, required=True, choices=range(UFS_START_YEAR, UFS_END_YEAR + 1))
    ufs.add_argument("--end-year", type=int, required=True)
    ufs.add_argument("--block-size-mb", type=int, default=64)
    ufs.add_argument("--build-features", action="store_true")
    ufs.add_argument("--execute", action="store_true")
    ufs.add_argument("--overwrite", action="store_true")
    ufs.set_defaults(func=cmd_ingest_ufs)

    operational = sub.add_parser(
        "ingest-glorys-operational",
        help="Ingest the post-multiyear GLO12 daily analysis tail; today/future dates are rejected.",
    )
    operational.add_argument("--start-date", required=True)
    operational.add_argument("--end-date")
    operational.add_argument("--execute", action="store_true")
    operational.add_argument("--overwrite", action="store_true")
    operational.add_argument("--delete-source-after-zarr", action="store_true")
    operational.set_defaults(func=cmd_ingest_glorys_operational)

    features = sub.add_parser("build-features", help="Build source-neutral ENSO/deep-ocean features from daily cubes.")
    features.add_argument("--source", choices=["noaa_ufs", "glorys12"], required=True)
    features.add_argument("--start-year", type=int, required=True)
    features.add_argument("--end-year", type=int, required=True)
    features.add_argument("--overwrite", action="store_true")
    features.set_defaults(func=cmd_build_features)
    return root


def main() -> int:
    load_local_env()
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
