from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.config import load_config, project_path
from nino_brasil.data.credentials import cds_credentials_status
from nino_brasil.data.download_cds import (
    ATMOSPHERE_AREAS,
    download_era5_pressure_month,
    download_era5_single_month,
    download_oras_month,
)
from nino_brasil.data.download_chirps import download_chirps_year
from nino_brasil.data.download_ibge import download_ibge
from nino_brasil.data.download_oisst import download_oisst_year


DATA_DIRS = [
    "data/raw/ibge",
    "data/raw/chirps",
    "data/raw/cpc_noaa",
    "data/raw/oras",
    "data/raw/era5",
    "data/interim/ibge",
    "data/interim/brazil_precipitation",
    "data/interim/pacific_warming",
    "data/interim/atmosphere_bridge",
    "data/processed/zarr",
    "data/processed/parquet",
    "data/processed/geotiff",
]


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
    print("data pipeline order:")
    print("1. init")
    print("2. download-ibge --product uf")
    print("3. download-ibge --product municipios")
    print("4. implement/download precipitation grid")
    print("5. implement/download Pacific SST/SSTA")
    print("6. build first lagged correlation maps")
    print("7. add ERA5 atmosphere bridge")
    print("8. add ORAS/ORAS5 subsurface ocean")
    return 0


def iter_years(start_year: int, end_year: int) -> range:
    if end_year < start_year:
        raise ValueError("end-year must be greater than or equal to start-year.")
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


def cmd_download_chirps(args: argparse.Namespace) -> int:
    raw_dir = project_path("data/raw/chirps")
    for year in iter_years(args.start_year, args.end_year):
        download_chirps_year(
            year=year,
            raw_dir=raw_dir,
            resolution=args.resolution,
            overwrite=args.overwrite,
            dry_run=not args.execute,
        )
    return 0


def cmd_download_oisst(args: argparse.Namespace) -> int:
    raw_dir = project_path("data/raw/cpc_noaa/oisst")
    for year in iter_years(args.start_year, args.end_year):
        download_oisst_year(
            year=year,
            raw_dir=raw_dir,
            overwrite=args.overwrite,
            dry_run=not args.execute,
        )
    return 0


def cmd_download_era5(args: argparse.Namespace) -> int:
    raw_dir = project_path("data/raw/era5")
    regions = args.region or list(ATMOSPHERE_AREAS)
    months = iter_months(args.month)
    for year in iter_years(args.start_year, args.end_year):
        for month in months:
            for region in regions:
                if args.kind in {"single", "both"}:
                    download_era5_single_month(
                        year=year,
                        month=month,
                        region=region,
                        raw_dir=raw_dir,
                        dry_run=not args.execute,
                    )
                if args.kind in {"pressure", "both"}:
                    download_era5_pressure_month(
                        year=year,
                        month=month,
                        region=region,
                        raw_dir=raw_dir,
                        dry_run=not args.execute,
                    )
    return 0


def cmd_download_oras(args: argparse.Namespace) -> int:
    raw_dir = project_path("data/raw/oras")
    months = iter_months(args.month)
    for year in iter_years(args.start_year, args.end_year):
        for month in months:
            download_oras_month(
                year=year,
                month=month,
                raw_dir=raw_dir,
                dry_run=not args.execute,
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
    dry_run = not args.execute
    print(f"download-all dry_run={dry_run}")
    print(f"years={args.start_year}-{args.end_year}")

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

    for year in iter_years(args.start_year, args.end_year):
        download_chirps_year(
            year=year,
            raw_dir=project_path("data/raw/chirps"),
            resolution=args.chirps_resolution,
            overwrite=args.overwrite,
            dry_run=dry_run,
        )
        download_oisst_year(
            year=year,
            raw_dir=project_path("data/raw/cpc_noaa/oisst"),
            overwrite=args.overwrite,
            dry_run=dry_run,
        )

    if args.include_cds:
        months = iter_months(args.month)
        for year in iter_years(args.start_year, args.end_year):
            for month in months:
                for region in ATMOSPHERE_AREAS:
                    download_era5_single_month(
                        year=year,
                        month=month,
                        region=region,
                        raw_dir=project_path("data/raw/era5"),
                        dry_run=dry_run,
                    )
                    download_era5_pressure_month(
                        year=year,
                        month=month,
                        region=region,
                        raw_dir=project_path("data/raw/era5"),
                        dry_run=dry_run,
                    )
                download_oras_month(
                    year=year,
                    month=month,
                    raw_dir=project_path("data/raw/oras"),
                    dry_run=dry_run,
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
    chirps_p.add_argument("--end-year", type=int, required=True)
    chirps_p.add_argument("--resolution", choices=["p25", "p05"], default="p25")
    chirps_p.add_argument("--execute", action="store_true", help="Actually download files.")
    chirps_p.add_argument("--overwrite", action="store_true")
    chirps_p.set_defaults(func=cmd_download_chirps)

    oisst_p = sub.add_parser("download-oisst", help="Download NOAA OISST annual NetCDF files.")
    oisst_p.add_argument("--start-year", type=int, required=True)
    oisst_p.add_argument("--end-year", type=int, required=True)
    oisst_p.add_argument("--execute", action="store_true", help="Actually download files.")
    oisst_p.add_argument("--overwrite", action="store_true")
    oisst_p.set_defaults(func=cmd_download_oisst)

    era5_p = sub.add_parser("download-era5", help="Download ERA5 monthly files through CDS.")
    era5_p.add_argument("--start-year", type=int, required=True)
    era5_p.add_argument("--end-year", type=int, required=True)
    era5_p.add_argument("--month", type=int, action="append", choices=range(1, 13))
    era5_p.add_argument("--kind", choices=["single", "pressure", "both"], default="both")
    era5_p.add_argument("--region", action="append", choices=sorted(ATMOSPHERE_AREAS))
    era5_p.add_argument("--execute", action="store_true", help="Actually submit CDS requests.")
    era5_p.set_defaults(func=cmd_download_era5)

    oras_p = sub.add_parser("download-oras", help="Download ORAS5 monthly files through CDS.")
    oras_p.add_argument("--start-year", type=int, required=True)
    oras_p.add_argument("--end-year", type=int, required=True)
    oras_p.add_argument("--month", type=int, action="append", choices=range(1, 13))
    oras_p.add_argument("--execute", action="store_true", help="Actually submit CDS requests.")
    oras_p.set_defaults(func=cmd_download_oras)

    all_p = sub.add_parser("download-all", help="Run the full download plan.")
    all_p.add_argument("--start-year", type=int, required=True)
    all_p.add_argument("--end-year", type=int, required=True)
    all_p.add_argument("--month", type=int, action="append", choices=range(1, 13))
    all_p.add_argument("--chirps-resolution", choices=["p25", "p05"], default="p25")
    all_p.add_argument("--include-cds", action="store_true", help="Include ERA5 and ORAS.")
    all_p.add_argument("--execute", action="store_true", help="Actually download/submit requests.")
    all_p.add_argument("--overwrite", action="store_true")
    all_p.set_defaults(func=cmd_download_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
