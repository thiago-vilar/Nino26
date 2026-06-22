from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.data.download_ocean_monthly import (
    ORAS5_ALL_LEVEL_VARIABLES,
    ORAS5_SINGLE_LEVEL_VARIABLES,
    ORAS5_VARIABLES,
    build_oras5_monthly_features,
    ingest_oras5_years,
    latest_complete_oras5_month,
    months_for_year,
    oras5_feature_path,
    oras5_variable_zarr_path,
)


RAW_ROOT = ROOT / "data/raw/ocean_monthly/oras5"
PROCESSED_ROOT = ROOT / "data/processed/zarr/ocean_monthly/oras5"
FEATURE_ROOT = ROOT / "data/processed/zarr/features/ocean_monthly/oras5"


def _years(start: int, end: int) -> list[int]:
    if end < start:
        raise ValueError("end-year must be greater than or equal to start-year.")
    return list(range(start, end + 1))


def _resolved_end(args: argparse.Namespace) -> tuple[int, int]:
    if args.end_year is not None and args.end_month is not None:
        return int(args.end_year), int(args.end_month)
    latest_year, latest_month = latest_complete_oras5_month()
    return int(args.end_year or latest_year), int(args.end_month or (latest_month if (args.end_year or latest_year) == latest_year else 12))


def cmd_plan(args: argparse.Namespace) -> int:
    end_year, end_month = _resolved_end(args)
    years = _years(args.start_year, end_year)
    tasks = sum(2 for year in years if months_for_year(year, end_year=end_year, end_month=end_month))
    print("ORAS5 contract: monthly means remain monthly; no daily expansion.")
    print(f"Period: {args.start_year}-01 through {end_year}-{end_month:02d}.")
    print(f"CDS requests: {tasks} maximum (one single-level + one all-level request per year).")
    print(f"Single-level grouped variables: {', '.join(ORAS5_SINGLE_LEVEL_VARIABLES)}")
    print(f"All-level grouped variables: {', '.join(ORAS5_ALL_LEVEL_VARIABLES)}")
    print("Derived monthly features: WWV and thermocline Tilt from the original monthly D20 field.")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    end_year, end_month = _resolved_end(args)
    ingest_oras5_years(
        years=_years(args.start_year, end_year),
        raw_root=RAW_ROOT,
        output_root=PROCESSED_ROOT,
        end_year=end_year,
        end_month=end_month,
        execute=args.execute,
        overwrite=args.overwrite,
        delete_raw_after_zarr=args.delete_raw_after_zarr,
        continue_on_error=args.continue_on_error,
    )
    if args.execute and args.build_features:
        for year in _years(args.start_year, end_year):
            if months_for_year(year, end_year=end_year, end_month=end_month):
                build_oras5_monthly_features(
                    year=year,
                    source_root=PROCESSED_ROOT,
                    output_root=FEATURE_ROOT,
                    overwrite=args.overwrite,
                )
    return 0


def cmd_build_features(args: argparse.Namespace) -> int:
    for year in _years(args.start_year, args.end_year):
        build_oras5_monthly_features(
            year=year,
            source_root=PROCESSED_ROOT,
            output_root=FEATURE_ROOT,
            overwrite=args.overwrite,
        )
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    end_year, end_month = _resolved_end(args)
    missing: list[str] = []
    invalid: list[str] = []
    import pandas as pd
    import xarray as xr

    for year in _years(args.start_year, end_year):
        months = months_for_year(year, end_year=end_year, end_month=end_month)
        if not months:
            continue
        expected = pd.DatetimeIndex([pd.Timestamp(year=year, month=month, day=1) for month in months])
        for variable in ORAS5_VARIABLES:
            path = oras5_variable_zarr_path(PROCESSED_ROOT, year, variable)
            if not path.exists():
                missing.append(str(path))
                continue
            with xr.open_zarr(path, consolidated=None) as ds:
                actual = pd.DatetimeIndex(ds["time"].values)
                if not actual.equals(expected) or str(ds.attrs.get("source_frequency")) != "monthly_mean":
                    invalid.append(str(path))
        feature = oras5_feature_path(FEATURE_ROOT, year)
        if not feature.exists():
            missing.append(str(feature))
    print(f"ORAS5 monthly audit: missing={len(missing)} invalid={len(invalid)}")
    for path in missing[:20]:
        print(f"MISSING {path}")
    for path in invalid[:20]:
        print(f"INVALID {path}")
    return 1 if missing or invalid else 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="ORAS5 monthly pipeline. Monthly values are never promoted to daily observations.")
    sub = root.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Show grouped annual CDS request count.")
    plan.add_argument("--start-year", type=int, default=1981)
    plan.add_argument("--end-year", type=int)
    plan.add_argument("--end-month", type=int, choices=range(1, 13))
    plan.set_defaults(func=cmd_plan)

    ingest = sub.add_parser("ingest", help="Download grouped ORAS5 annual monthly means and write monthly Zarr stores.")
    ingest.add_argument("--start-year", type=int, default=1981)
    ingest.add_argument("--end-year", type=int)
    ingest.add_argument("--end-month", type=int, choices=range(1, 13))
    ingest.add_argument("--build-features", action="store_true")
    ingest.add_argument("--delete-raw-after-zarr", action="store_true")
    ingest.add_argument("--execute", action="store_true")
    ingest.add_argument("--overwrite", action="store_true")
    ingest.add_argument("--continue-on-error", action="store_true")
    ingest.set_defaults(func=cmd_ingest)

    features = sub.add_parser("build-features", help="Build native-frequency monthly D20/OHC/WWV/Tilt features.")
    features.add_argument("--start-year", type=int, required=True)
    features.add_argument("--end-year", type=int, required=True)
    features.add_argument("--overwrite", action="store_true")
    features.set_defaults(func=cmd_build_features)

    audit = sub.add_parser("audit", help="Validate monthly time axes, variables and feature stores.")
    audit.add_argument("--start-year", type=int, default=1981)
    audit.add_argument("--end-year", type=int)
    audit.add_argument("--end-month", type=int, choices=range(1, 13))
    audit.set_defaults(func=cmd_audit)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
