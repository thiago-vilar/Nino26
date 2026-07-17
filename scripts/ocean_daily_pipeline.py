from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.data.download_ocean_daily import (
    GLORYS_DEFAULT_VARIABLES,
    GLORYS_PROCESSED_VARIABLES,
    GLORYS_START_YEAR,
    OCEAN_FEATURE_VARIABLES,
    UFS_END_YEAR,
    UFS_START_YEAR,
    build_ocean_daily_features,
    daily_store_valid,
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
        expected_start = pd.Timestamp(f"{year}-01-01")
        expected_end = min(
            pd.Timestamp(f"{year}-12-31"),
            pd.Timestamp(args.end_date).normalize() if args.end_date else pd.Timestamp(f"{year}-12-31"),
        )
        output = PROCESSED_GLORYS_ROOT / str(year) / f"glorys12_equatorial_pacific_{year}_daily_0p25.zarr"
        feature = _feature_path("glorys12", year)
        processed_valid = daily_store_valid(
            output,
            required_variables=GLORYS_PROCESSED_VARIABLES,
            expected_start=expected_start,
            expected_end=expected_end,
            require_canonical_grid=True,
        )
        if processed_valid and not args.overwrite:
            if daily_store_valid(
                feature,
                required_variables=OCEAN_FEATURE_VARIABLES,
                expected_start=expected_start,
                expected_end=expected_end,
            ):
                print(f"valid final GLORYS Zarr and features exist; download skipped: {year}")
            else:
                print(f"valid final GLORYS Zarr exists; rebuilding features locally: {year}")
                build_ocean_daily_features(output, feature, overwrite=feature.exists())
            continue
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
        process_glorys_year(source, output, overwrite=args.overwrite)
        build_ocean_daily_features(output, feature, overwrite=args.overwrite)
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


def _valid_operational_stores() -> list[Path]:
    return [
        path
        for path in sorted(PROCESSED_OPERATIONAL_ROOT.rglob("*.zarr"))
        if daily_store_valid(
            path,
            required_variables=GLORYS_PROCESSED_VARIABLES,
            require_canonical_grid=True,
        )
    ]


def _quarantine_store(path: Path, quarantine_root: Path) -> None:
    if not path.exists():
        return
    quarantine_root.mkdir(parents=True, exist_ok=True)
    target = quarantine_root / path.name
    if target.exists():
        stamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        target = quarantine_root / f"{path.stem}_{stamp}{path.suffix}"
    shutil.move(str(path), str(target))
    print(f"fragmento operacional em quarentena: {path} -> {target}")


def _consolidate_operational_window(window_start: pd.Timestamp, window_end: pd.Timestamp) -> Path | None:
    """Une os fragmentos GLO12 num único store canônico da janela operacional.

    A atualização incremental baixa apenas a cauda que falta, mas a auditoria
    (audit_ocean_phase2) exige exatamente um cubo e um store de features
    cobrindo my_end+1..operational_end. Aqui os fragmentos são concatenados,
    as features são recalculadas para a janela completa e os stores parciais
    antigos vão para data/quarantine (nada é apagado).
    """
    from nino_brasil.data.zarr_store import ZARR_FORMAT

    stores = _valid_operational_stores()
    if not stores:
        print("consolidação operacional: nenhum store válido encontrado")
        return None
    opened = [(path, xr.open_zarr(path, consolidated=None)) for path in stores]
    try:
        combined = xr.concat(
            [dataset for _, dataset in opened],
            dim="time",
            data_vars="all",
            coords="minimal",
            compat="override",
        ).sortby("time")
        index = pd.DatetimeIndex(combined["time"].values).normalize()
        combined = combined.isel(time=~index.duplicated(keep="last"))
        combined = combined.sel(time=slice(window_start, window_end))
        index = pd.DatetimeIndex(combined["time"].values).normalize()
        if not len(index):
            print("consolidação operacional: nenhum dia disponível dentro da janela")
            return None
        actual_end = index.max()
        gaps = pd.date_range(window_start, actual_end, freq="D").difference(index)
        if len(gaps):
            raise ValueError(
                f"lacunas internas na janela operacional GLO12: {[str(day.date()) for day in gaps[:5]]}"
            )
        if actual_end < window_end:
            print(
                f"consolidação operacional: GLO12 disponível até {actual_end.date()}; "
                f"janela solicitada terminava em {window_end.date()}"
            )
        slug = f"{window_start:%Y%m%d}_{actual_end:%Y%m%d}"
        canonical = PROCESSED_OPERATIONAL_ROOT / str(window_start.year) / f"glorys12_operational_{slug}_daily_0p25.zarr"
        canonical_feature = FEATURE_ROOT / "glorys12_operational" / str(window_start.year) / (
            f"glorys12_operational_ocean_features_{slug}_daily.zarr"
        )
        needs_merge = not (len(stores) == 1 and stores[0] == canonical)
        if needs_merge:
            combined = combined.load()
            combined.attrs = dict(opened[-1][1].attrs)
            for name in list(combined.variables):
                combined[name].encoding = {}
    finally:
        for _, dataset in opened:
            dataset.close()
    if needs_merge:
        if canonical.exists():
            shutil.rmtree(canonical)
        canonical.parent.mkdir(parents=True, exist_ok=True)
        combined.chunk({"time": min(31, int(combined.sizes["time"]))}).to_zarr(
            canonical, mode="w", consolidated=True, zarr_format=ZARR_FORMAT
        )
        print(f"store operacional consolidado: {canonical}")
    if not daily_store_valid(
        canonical,
        required_variables=GLORYS_PROCESSED_VARIABLES,
        expected_start=window_start,
        expected_end=actual_end,
        require_canonical_grid=True,
    ):
        raise ValueError(f"store operacional consolidado falhou na validação: {canonical}")
    if not daily_store_valid(
        canonical_feature,
        required_variables=OCEAN_FEATURE_VARIABLES,
        expected_start=window_start,
        expected_end=actual_end,
    ):
        build_ocean_daily_features(canonical, canonical_feature, overwrite=canonical_feature.exists())
    for path in sorted(PROCESSED_OPERATIONAL_ROOT.rglob("*.zarr")):
        if path != canonical:
            _quarantine_store(path, ROOT / "data/quarantine/zarr/ocean_daily/glorys12_operational")
    for path in sorted((FEATURE_ROOT / "glorys12_operational").rglob("*.zarr")):
        if path != canonical_feature:
            _quarantine_store(path, ROOT / "data/quarantine/zarr/features/ocean_daily/glorys12_operational")
    return canonical


def cmd_ingest_glorys_operational(args: argparse.Namespace) -> int:
    window_start = pd.Timestamp(args.start_date).normalize()
    end = pd.Timestamp(args.end_date).normalize() if args.end_date else pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    start = window_start
    if not args.overwrite:
        covered = pd.DatetimeIndex([])
        for processed in _valid_operational_stores():
            with xr.open_zarr(processed, consolidated=None) as dataset:
                index = pd.DatetimeIndex(dataset.time.values).normalize()
            covered = covered.union(index)
        requested = pd.date_range(window_start, end, freq="D")
        missing = requested.difference(covered)
        if missing.empty:
            print(f"operational GLORYS already complete through {end.date()}; download skipped")
            if args.execute:
                _consolidate_operational_window(window_start, end)
            return 0
        start = missing[0]
        print(f"operational GLORYS incremental tail: {start.date()}..{end.date()}")
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
    if args.delete_source_after_zarr:
        for source in sources.values():
            _delete_validated_raw(source, RAW_OPERATIONAL_ROOT)
    # A cauda recém-processada vira parte do store canônico da janela; as
    # features são recalculadas lá dentro para o período completo.
    _consolidate_operational_window(window_start, end)
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
