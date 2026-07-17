from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.config import load_config, project_path
from nino_brasil.data.audit import AuditLog
from nino_brasil.data.availability import available_end_date, iter_available_years, record_source_latency
from nino_brasil.data.download_chirps import download_chirps_year
from nino_brasil.data.download_ibge import download_ibge
from nino_brasil.data.download_oisst import download_oisst_year
from nino_brasil.data.regrid import normalize_for_common_grid, regrid_dataset, target_grid_from_config
from nino_brasil.data.zarr_store import ZARR_FORMAT, netcdf_to_daily_zarr, validate_netcdf, validate_zarr


Source = str
Stage = str


@dataclass(frozen=True)
class CheckResult:
    source: Source
    stage: Stage
    year: int | None
    label: str
    path: Path
    ok: bool
    detail: str


@dataclass(frozen=True)
class PendingAction:
    source: Source
    stage: Stage
    year: int | None
    label: str
    path: Path
    detail: str
    run: Callable[[], Path | None]


def chirps_raw_path(year: int, resolution: str) -> Path:
    return project_path(f"data/raw/chirps/{resolution}/chirps-v2.0.{year}.days_{resolution}.nc")


def chirps_zarr_path(year: int, resolution: str) -> Path:
    return project_path(f"data/processed/zarr/brazil_precipitation/chirps_{resolution}_{year}.zarr")


def chirps_regrid_path(year: int, resolution: str) -> Path:
    return project_path(f"data/processed/zarr/regridded/chirps_{resolution}_{year}.zarr")


def oisst_raw_path(year: int) -> Path:
    return project_path(f"data/raw/cpc_noaa/oisst/sst.day.mean.{year}.nc")


def oisst_zarr_path(year: int) -> Path:
    return project_path(f"data/processed/zarr/cpc_noaa/oisst/sst.day.mean.{year}.zarr")


def oisst_regrid_path(year: int) -> Path:
    return project_path(f"data/processed/zarr/regridded/noaa_oisst_{year}.zarr")


def safe_chunks(ds: xr.Dataset) -> dict[str, int]:
    chunks: dict[str, int] = {}
    for dim, size in ds.sizes.items():
        name = dim.lower()
        if "time" in name:
            chunks[dim] = min(int(size), 31)
        elif "depth" in name or "level" in name:
            chunks[dim] = min(int(size), 10)
        elif "lat" in name or name in {"y", "j"}:
            chunks[dim] = min(int(size), 200)
        elif "lon" in name or name in {"x", "i"}:
            chunks[dim] = min(int(size), 200)
    return chunks


def check_netcdf(path: Path, *, fast: bool = False) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.suffix == ".part":
        return False, "partial"
    if fast:
        return True, "exists"
    try:
        summary = validate_netcdf(path)
        return True, f"valid netcdf; vars={','.join(summary['variables'])}"
    except BaseException as exc:
        return False, f"invalid netcdf: {type(exc).__name__}: {exc}"


def check_zarr(path: Path, *, fast: bool = False) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if fast:
        return True, "exists"
    try:
        summary = validate_zarr(path)
        return True, f"valid zarr; vars={','.join(summary['variables'])}"
    except BaseException as exc:
        return False, f"invalid zarr: {type(exc).__name__}: {exc}"


# Primeiro dia real de dados de cada fonte; antes disso não existe arquivo a
# baixar. OISST v2.1 começa em 1981-09-01: exigir o ano civil completo forçava
# um re-download infinito de 1981 a cada execução.
SOURCE_DATA_START = {
    "noaa_oisst": pd.Timestamp("1981-09-01"),
}


def expected_daily_bounds(year: int, source: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = max(pd.Timestamp(f"{year}-01-01"), SOURCE_DATA_START.get(source, pd.Timestamp(f"{year}-01-01")))
    end = min(pd.Timestamp(f"{year}-12-31"), available_end_date(source, load_config()))
    return start, end


def check_daily_zarr_coverage(
    path: Path,
    *,
    variable: str,
    year: int,
    source: str,
    fast: bool = False,
) -> tuple[bool, str]:
    """Validate the final daily time axis before considering any raw cache."""
    basic_ok, detail = check_zarr(path, fast=fast)
    if not basic_ok or fast:
        return basic_ok, detail
    expected_start, expected_end = expected_daily_bounds(year, source)
    try:
        with xr.open_zarr(path, consolidated=None) as dataset:
            if variable not in dataset or "time" not in dataset.coords:
                return False, f"missing {variable} or time"
            index = pd.DatetimeIndex(dataset.time.values).normalize()
            empty_days = _sampled_empty_days(dataset[variable], index)
        expected = pd.date_range(expected_start, expected_end, freq="D")
        missing = expected.difference(index)
        extra_duplicates = int(index.duplicated().sum())
        ok = not missing.size and not extra_duplicates and index.is_monotonic_increasing and not empty_days
        return ok, (
            f"coverage={index.min().date()}..{index.max().date()}; "
            f"expected_through={expected_end.date()}; missing_days={len(missing)}; duplicates={extra_duplicates}; "
            f"sampled_empty_days={len(empty_days)}"
        )
    except (OSError, ValueError, KeyError, IndexError) as exc:
        return False, f"invalid daily coverage: {type(exc).__name__}: {exc}"


def _sampled_empty_days(array: xr.DataArray, index: pd.DatetimeIndex, samples: int = 6) -> list[str]:
    """Sample days along the axis and flag those without any finite value.

    Um eixo de tempo completo não garante dados: um regrid quebrado grava o
    ano inteiro como NaN. Amostrar o primeiro/último dia e o interior pega
    tanto o store totalmente vazio quanto a cauda vazia sem custar uma
    varredura completa do ano.
    """
    if "time" not in array.dims or not len(index):
        return []
    positions = sorted({0, len(index) - 1, *(int(round(k * (len(index) - 1) / (samples - 1))) for k in range(1, samples - 1))})
    spatial = [dim for dim in array.dims if dim != "time"]
    empty: list[str] = []
    for position in positions:
        day = array.isel(time=position)
        has_value = day.notnull()
        if spatial:
            has_value = has_value.any()
        if not bool(has_value.compute().values):
            empty.append(str(index[position].date()))
    return empty


def check_daily_netcdf_coverage(
    path: Path,
    *,
    variable: str,
    year: int,
    source: str,
    fast: bool = False,
) -> tuple[bool, str]:
    basic_ok, detail = check_netcdf(path, fast=fast)
    if not basic_ok or fast:
        return basic_ok, detail
    expected_start, expected_end = expected_daily_bounds(year, source)
    try:
        with xr.open_dataset(path, chunks={}) as dataset:
            if variable not in dataset or "time" not in dataset.coords:
                return False, f"missing {variable} or time"
            index = pd.DatetimeIndex(dataset.time.values).normalize()
        missing = pd.date_range(expected_start, expected_end, freq="D").difference(index)
        ok = not missing.size and not index.duplicated().any() and index.is_monotonic_increasing
        return ok, f"coverage={index.min().date()}..{index.max().date()}; expected_through={expected_end.date()}; missing_days={len(missing)}"
    except (OSError, ValueError, KeyError, IndexError) as exc:
        return False, f"invalid daily coverage: {type(exc).__name__}: {exc}"


def download_with_retry(action: Callable[[], Path], *, retries: int, wait_seconds: int) -> Path:
    attempt = 0
    while True:
        attempt += 1
        try:
            return action()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            retryable = status is None or int(status) >= 500
            if attempt > retries or not retryable:
                raise
            print(f"retryable HTTP error on attempt {attempt}/{retries}: {exc}")
            time.sleep(wait_seconds)


def materialize_daily_source(
    raw_path: Path,
    zarr_path: Path,
    regrid_path: Path,
    *,
    variable: str,
    dataset: str,
    build_zarr: bool,
    build_regrid: bool,
    overwrite: bool,
) -> Path:
    """Complete all requested local stages in the same resumable action."""
    result = raw_path
    if build_zarr:
        result = netcdf_to_daily_zarr(
            raw_path,
            zarr_path,
            variables=[variable],
            source_frequency="daily",
            overwrite=overwrite or zarr_path.exists(),
        )
    if build_regrid:
        result = regrid_zarr(
            zarr_path,
            regrid_path,
            dataset=dataset,
            overwrite=overwrite or regrid_path.exists(),
        )
    return result


def download_and_materialize_daily_source(
    download: Callable[[], Path],
    zarr_path: Path,
    regrid_path: Path,
    **kwargs: object,
) -> Path:
    raw_path = download()
    return materialize_daily_source(raw_path, zarr_path, regrid_path, **kwargs)


def regrid_zarr(input_path: Path, output_path: Path, *, dataset: str, overwrite: bool = False) -> Path:
    if output_path.exists() and not overwrite:
        validate_zarr(output_path)
        print(f"regrid exists: {output_path}")
        return output_path

    cfg = load_config()
    audit = AuditLog()
    task_id = f"curate_regrid_{dataset}"
    audit.record(
        task_id=task_id,
        dataset=dataset,
        status="started",
        input_path=str(input_path),
        output_path=str(output_path),
    )
    try:
        ds = xr.open_zarr(input_path)
        try:
            normalized = normalize_for_common_grid(ds, convention="0_360")
            regridded = regrid_dataset(
                normalized,
                target_grid_from_config(cfg),
                method=cfg.get("modeling", {}).get("grid", {}).get("regrid_method", "bilinear"),
            )
            chunks = safe_chunks(regridded)
            if chunks:
                regridded = regridded.chunk(chunks)
            if output_path.exists() and overwrite:
                shutil.rmtree(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            regridded.to_zarr(output_path, mode="w", consolidated=True, zarr_format=ZARR_FORMAT)
        finally:
            ds.close()
        validate_zarr(output_path)
        audit.record(task_id=task_id, dataset=dataset, status="ok", output_path=str(output_path))
        print(f"regrid written: {output_path}")
        return output_path
    except BaseException as exc:
        audit.record(
            task_id=task_id,
            dataset=dataset,
            status="error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


def iter_years(source: str, start_year: int, end_year: int | None) -> list[int]:
    cfg = load_config()
    return list(iter_available_years(start_year, end_year, source, cfg))


def add_ibge_actions(
    checks: list[CheckResult],
    pending: list[PendingAction],
    *,
    selected_stages: set[Stage],
    fast: bool,
    overwrite: bool,
) -> None:
    if "raw" not in selected_stages and "all" not in selected_stages:
        return
    raw_dir = project_path("data/raw/ibge")
    interim_dir = project_path("data/interim/ibge")
    products = {
        "uf": (raw_dir / "BR_UF_2024.zip", interim_dir / "BR_UF_2024"),
        "municipios": (raw_dir / "BR_Municipios_2024.zip", interim_dir / "BR_Municipios_2024"),
        "regioes": (raw_dir / "BR_Regioes_2024.zip", interim_dir / "BR_Regioes_2024"),
        "biomas": (
            raw_dir / "2025_Biomas-e-Sistema-Costeiro-Marinho-do-Brasil-1-250000_shp.zip",
            interim_dir / "Biomas_2025",
        ),
    }
    for product, (raw_path, extracted_path) in products.items():
        raw_ok = raw_path.exists()
        extracted_ok = extracted_path.exists() and any(extracted_path.iterdir())
        ok = raw_ok and extracted_ok
        detail = "ok" if ok else f"raw={'ok' if raw_ok else 'missing'}; extract={'ok' if extracted_ok else 'missing'}"
        checks.append(CheckResult("ibge", "raw", None, product, raw_path, ok, detail))
        if not ok:
            pending.append(
                PendingAction(
                    "ibge",
                    "raw",
                    None,
                    product,
                    raw_path,
                    detail,
                    lambda product=product: download_ibge(
                        product_id=product,
                        raw_dir=raw_dir,
                        interim_dir=interim_dir,
                        extract=True,
                        overwrite=overwrite,
                    ),
                )
            )


def add_chirps_actions(
    checks: list[CheckResult],
    pending: list[PendingAction],
    *,
    years: list[int],
    resolution: str,
    selected_stages: set[Stage],
    fast: bool,
    overwrite: bool,
    retries: int,
    retry_wait: int,
) -> None:
    for year in years:
        raw_path = chirps_raw_path(year, resolution)
        zarr_path = chirps_zarr_path(year, resolution)
        regrid_path = chirps_regrid_path(year, resolution)
        final_zarr_ok, final_zarr_detail = check_daily_zarr_coverage(
            zarr_path, variable="precip", year=year, source="chirps", fast=fast
        )
        final_regrid_ok, final_regrid_detail = check_daily_zarr_coverage(
            regrid_path, variable="precip", year=year, source="chirps", fast=fast
        )
        if final_regrid_ok or final_zarr_ok:
            checks.append(CheckResult("chirps", "raw", year, f"CHIRPS {resolution} {year}", raw_path, True, "cache not required; final Zarr validated"))
            checks.append(CheckResult("chirps", "zarr", year, f"CHIRPS {resolution} {year}", zarr_path, final_zarr_ok, final_zarr_detail))
            checks.append(CheckResult("chirps", "regrid", year, f"CHIRPS {resolution} {year}", regrid_path, final_regrid_ok, final_regrid_detail))
            if final_zarr_ok and not final_regrid_ok and ("regrid" in selected_stages or "all" in selected_stages):
                pending.append(PendingAction(
                    "chirps", "regrid", year, f"CHIRPS {resolution} {year}", regrid_path, final_regrid_detail,
                    lambda zarr_path=zarr_path, regrid_path=regrid_path, year=year: regrid_zarr(
                        zarr_path, regrid_path, dataset=f"chirps_{resolution}_{year}", overwrite=overwrite or regrid_path.exists()
                    ),
                ))
            continue
        raw_ok, raw_detail = check_daily_netcdf_coverage(
            raw_path, variable="precip", year=year, source="chirps", fast=fast
        )
        checks.append(CheckResult("chirps", "raw", year, f"CHIRPS {resolution} {year}", raw_path, raw_ok, raw_detail))
        if ("raw" in selected_stages or "all" in selected_stages) and not raw_ok:
            pending.append(
                PendingAction(
                    "chirps",
                    "raw",
                    year,
                    f"CHIRPS {resolution} {year}",
                    raw_path,
                    raw_detail,
                    lambda year=year, raw_path=raw_path, zarr_path=zarr_path, regrid_path=regrid_path: download_and_materialize_daily_source(
                        lambda: download_with_retry(
                            lambda: download_chirps_year(
                                year=year,
                                raw_dir=project_path("data/raw/chirps"),
                                resolution=resolution,
                                overwrite=overwrite or raw_path.exists(),
                                dry_run=False,
                            ),
                            retries=retries,
                            wait_seconds=retry_wait,
                        ),
                        zarr_path,
                        regrid_path,
                        variable="precip",
                        dataset=f"chirps_{resolution}_{year}",
                        build_zarr="all" in selected_stages or "zarr" in selected_stages or "regrid" in selected_stages,
                        build_regrid="all" in selected_stages or "regrid" in selected_stages,
                        overwrite=overwrite,
                    ),
                )
            )
            continue

        zarr_path = chirps_zarr_path(year, resolution)
        zarr_ok, zarr_detail = check_daily_zarr_coverage(
            zarr_path, variable="precip", year=year, source="chirps", fast=fast
        )
        checks.append(CheckResult("chirps", "zarr", year, f"CHIRPS {resolution} {year}", zarr_path, zarr_ok, zarr_detail))
        if raw_ok and ("zarr" in selected_stages or "all" in selected_stages) and not zarr_ok:
            pending.append(
                PendingAction(
                    "chirps",
                    "zarr",
                    year,
                    f"CHIRPS {resolution} {year}",
                    zarr_path,
                    zarr_detail,
                    lambda raw_path=raw_path, zarr_path=zarr_path, regrid_path=regrid_path: materialize_daily_source(
                        raw_path,
                        zarr_path,
                        regrid_path,
                        variable="precip",
                        dataset=f"chirps_{resolution}_{year}",
                        build_zarr=True,
                        build_regrid="all" in selected_stages or "regrid" in selected_stages,
                        overwrite=overwrite,
                    ),
                )
            )
            continue

        regrid_path = chirps_regrid_path(year, resolution)
        regrid_ok, regrid_detail = check_daily_zarr_coverage(
            regrid_path, variable="precip", year=year, source="chirps", fast=fast
        )
        checks.append(CheckResult("chirps", "regrid", year, f"CHIRPS {resolution} {year}", regrid_path, regrid_ok, regrid_detail))
        if zarr_ok and ("regrid" in selected_stages or "all" in selected_stages) and not regrid_ok:
            pending.append(
                PendingAction(
                    "chirps",
                    "regrid",
                    year,
                    f"CHIRPS {resolution} {year}",
                    regrid_path,
                    regrid_detail,
                    lambda zarr_path=zarr_path, regrid_path=regrid_path, year=year: regrid_zarr(
                        zarr_path,
                        regrid_path,
                        dataset=f"chirps_{resolution}_{year}",
                        overwrite=overwrite or regrid_path.exists(),
                    ),
                )
            )


def add_oisst_actions(
    checks: list[CheckResult],
    pending: list[PendingAction],
    *,
    years: list[int],
    selected_stages: set[Stage],
    fast: bool,
    overwrite: bool,
    retries: int,
    retry_wait: int,
) -> None:
    for year in years:
        raw_path = oisst_raw_path(year)
        zarr_path = oisst_zarr_path(year)
        regrid_path = oisst_regrid_path(year)
        final_zarr_ok, final_zarr_detail = check_daily_zarr_coverage(
            zarr_path, variable="sst", year=year, source="noaa_oisst", fast=fast
        )
        final_regrid_ok, final_regrid_detail = check_daily_zarr_coverage(
            regrid_path, variable="sst", year=year, source="noaa_oisst", fast=fast
        )
        if final_regrid_ok or final_zarr_ok:
            checks.append(CheckResult("oisst", "raw", year, f"OISST {year}", raw_path, True, "cache not required; final Zarr validated"))
            checks.append(CheckResult("oisst", "zarr", year, f"OISST {year}", zarr_path, final_zarr_ok, final_zarr_detail))
            checks.append(CheckResult("oisst", "regrid", year, f"OISST {year}", regrid_path, final_regrid_ok, final_regrid_detail))
            if final_zarr_ok and not final_regrid_ok and ("regrid" in selected_stages or "all" in selected_stages):
                pending.append(PendingAction(
                    "oisst", "regrid", year, f"OISST {year}", regrid_path, final_regrid_detail,
                    lambda zarr_path=zarr_path, regrid_path=regrid_path, year=year: regrid_zarr(
                        zarr_path, regrid_path, dataset=f"noaa_oisst_{year}", overwrite=overwrite or regrid_path.exists()
                    ),
                ))
            continue
        raw_ok, raw_detail = check_daily_netcdf_coverage(
            raw_path, variable="sst", year=year, source="noaa_oisst", fast=fast
        )
        checks.append(CheckResult("oisst", "raw", year, f"OISST {year}", raw_path, raw_ok, raw_detail))
        if ("raw" in selected_stages or "all" in selected_stages) and not raw_ok:
            pending.append(
                PendingAction(
                    "oisst",
                    "raw",
                    year,
                    f"OISST {year}",
                    raw_path,
                    raw_detail,
                    lambda year=year, raw_path=raw_path, zarr_path=zarr_path, regrid_path=regrid_path: download_and_materialize_daily_source(
                        lambda: download_with_retry(
                            lambda: download_oisst_year(
                                year=year,
                                raw_dir=project_path("data/raw/cpc_noaa/oisst"),
                                overwrite=overwrite or raw_path.exists(),
                                dry_run=False,
                            ),
                            retries=retries,
                            wait_seconds=retry_wait,
                        ),
                        zarr_path,
                        regrid_path,
                        variable="sst",
                        dataset=f"noaa_oisst_{year}",
                        build_zarr="all" in selected_stages or "zarr" in selected_stages or "regrid" in selected_stages,
                        build_regrid="all" in selected_stages or "regrid" in selected_stages,
                        overwrite=overwrite,
                    ),
                )
            )
            continue

        zarr_path = oisst_zarr_path(year)
        zarr_ok, zarr_detail = check_daily_zarr_coverage(
            zarr_path, variable="sst", year=year, source="noaa_oisst", fast=fast
        )
        checks.append(CheckResult("oisst", "zarr", year, f"OISST {year}", zarr_path, zarr_ok, zarr_detail))
        if raw_ok and ("zarr" in selected_stages or "all" in selected_stages) and not zarr_ok:
            pending.append(
                PendingAction(
                    "oisst",
                    "zarr",
                    year,
                    f"OISST {year}",
                    zarr_path,
                    zarr_detail,
                    lambda raw_path=raw_path, zarr_path=zarr_path, regrid_path=regrid_path: materialize_daily_source(
                        raw_path,
                        zarr_path,
                        regrid_path,
                        variable="sst",
                        dataset=f"noaa_oisst_{year}",
                        build_zarr=True,
                        build_regrid="all" in selected_stages or "regrid" in selected_stages,
                        overwrite=overwrite,
                    ),
                )
            )
            continue

        regrid_path = oisst_regrid_path(year)
        regrid_ok, regrid_detail = check_daily_zarr_coverage(
            regrid_path, variable="sst", year=year, source="noaa_oisst", fast=fast
        )
        checks.append(CheckResult("oisst", "regrid", year, f"OISST {year}", regrid_path, regrid_ok, regrid_detail))
        if zarr_ok and ("regrid" in selected_stages or "all" in selected_stages) and not regrid_ok:
            pending.append(
                PendingAction(
                    "oisst",
                    "regrid",
                    year,
                    f"OISST {year}",
                    regrid_path,
                    regrid_detail,
                    lambda zarr_path=zarr_path, regrid_path=regrid_path, year=year: regrid_zarr(
                        zarr_path,
                        regrid_path,
                        dataset=f"noaa_oisst_{year}",
                        overwrite=overwrite or regrid_path.exists(),
                    ),
                )
            )


def add_cds_ctd_placeholders(
    checks: list[CheckResult],
    pending: list[PendingAction],
    *,
    selected_sources: set[Source],
    selected_stages: set[Stage],
    start_year: int,
    end_year: int | None,
) -> None:
    commands = {
        "era5": f"python scripts/data_pipeline.py download-era5 --start-year {start_year}"
        + (f" --end-year {end_year}" if end_year else "")
        + " --kind both --region nino34 --region brazil --annual-zarr --request-mode annual-kind --delete-raw-after-zarr --execute --continue-on-error",
        "ocean": "python scripts/run_ocean_phase2.py --execute --continue-on-error",
        "ctd": f"python scripts/data_pipeline.py download-ctd --start-year {start_year}"
        + (f" --end-year {end_year}" if end_year else "")
        + " --max-depth 300 --min-levels 3 --execute --continue-on-error",
    }
    for source, command in commands.items():
        if source not in selected_sources and "all" not in selected_sources:
            continue
        path = project_path(
            {
                "era5": "data/raw/era5",
                "ocean": "data/processed/zarr/ocean_daily",
                "ctd": "data/raw/ctd_noaa/wod",
            }[source]
        )
        has_files = path.exists() and any(p.is_file() and p.name != ".gitkeep" for p in path.rglob("*"))
        checks.append(
            CheckResult(
                source,
                "raw",
                None,
                source.upper(),
                path,
                has_files,
                "has local files" if has_files else f"pending; use: {command}",
            )
        )


def selected_sources(value: str) -> set[Source]:
    if value == "core":
        return {"ibge", "chirps", "oisst"}
    if value == "all":
        return {"all"}
    return {item.strip() for item in value.split(",") if item.strip()}


def selected_stages(value: str) -> set[Stage]:
    if value == "all":
        return {"all"}
    return {item.strip() for item in value.split(",") if item.strip()}


def print_summary(checks: list[CheckResult], pending: list[PendingAction]) -> None:
    print()
    print("CURADORIA LOCAL")
    print("=" * 80)
    for source in ["ibge", "chirps", "oisst", "era5", "ocean", "ctd"]:
        source_checks = [item for item in checks if item.source == source]
        if not source_checks:
            continue
        ok = sum(1 for item in source_checks if item.ok)
        total = len(source_checks)
        print(f"- {source}: {ok}/{total} checks ok")
        for stage in ["raw", "zarr", "regrid"]:
            stage_checks = [item for item in source_checks if item.stage == stage]
            if not stage_checks:
                continue
            ok_years = [item.year for item in stage_checks if item.ok and item.year is not None]
            missing = [item for item in stage_checks if not item.ok]
            if ok_years:
                print(f"  - {stage}: ok {min(ok_years)}-{max(ok_years)} ({len(ok_years)} anos)")
            elif any(item.ok for item in stage_checks):
                print(f"  - {stage}: ok")
            else:
                print(f"  - {stage}: nenhum ok")
            if missing:
                first = missing[0]
                year_label = f" {first.year}" if first.year is not None else ""
                print(f"    primeiro pendente:{year_label} ({first.detail}) -> {first.path}")
    print()
    print("PENDENCIAS ORDENADAS")
    print("=" * 80)
    if not pending:
        print("Nenhuma pendencia acionavel para os filtros escolhidos.")
        return
    for idx, action in enumerate(pending[:20], start=1):
        year_label = f" {action.year}" if action.year is not None else ""
        print(f"{idx:02d}. {action.source}/{action.stage}{year_label}: {action.detail}")
        print(f"    {action.path}")
    if len(pending) > 20:
        print(f"... mais {len(pending) - 20} pendencias.")


def execute_pending(pending: list[PendingAction], *, limit: int, continue_on_error: bool) -> int:
    if not pending:
        print("Nada para executar.")
        return 0
    actions = pending if limit == 0 else pending[:limit]
    failures = 0
    print()
    print(f"EXECUTANDO {len(actions)} ACAO(OES)")
    print("=" * 80)
    for idx, action in enumerate(actions, start=1):
        year_label = f" {action.year}" if action.year is not None else ""
        print(f"[{idx}/{len(actions)}] {action.source}/{action.stage}{year_label}: {action.label}")
        try:
            # Revalida imediatamente antes da escrita. Isso evita refazer um
            # produto que tenha sido concluído por outra etapa/processo desde
            # a montagem da lista de pendências.
            if action.stage == "raw" and action.path.suffix.lower() == ".nc":
                if action.source == "chirps" and action.year is not None:
                    current_ok = check_daily_netcdf_coverage(
                        action.path, variable="precip", year=action.year, source="chirps", fast=False
                    )[0]
                elif action.source == "oisst" and action.year is not None:
                    current_ok = check_daily_netcdf_coverage(
                        action.path, variable="sst", year=action.year, source="noaa_oisst", fast=False
                    )[0]
                else:
                    current_ok = check_netcdf(action.path, fast=False)[0]
            elif action.stage in {"zarr", "regrid"}:
                if action.source == "chirps" and action.year is not None:
                    current_ok = check_daily_zarr_coverage(
                        action.path, variable="precip", year=action.year, source="chirps", fast=False
                    )[0]
                elif action.source == "oisst" and action.year is not None:
                    current_ok = check_daily_zarr_coverage(
                        action.path, variable="sst", year=action.year, source="noaa_oisst", fast=False
                    )[0]
                else:
                    current_ok = check_zarr(action.path, fast=False)[0]
            else:
                current_ok = False
            if current_ok:
                print(f"[skip] já existe e foi validado: {action.path}")
                continue
            action.run()
        except BaseException as exc:
            failures += 1
            print(f"ERRO: {type(exc).__name__}: {exc}")
            if not continue_on_error:
                raise
    if failures:
        print(f"Concluido com {failures} falha(s) em acoes individuais.")
        return 1
    return 0


def purge_validated_raw(
    checks: list[CheckResult],
    *,
    resolution: str,
    execute: bool,
) -> list[Path]:
    """Apaga o cache bruto anual de CHIRPS/OISST quando zarr e regrid validam.

    Implementa a política ``raw_cache_policy`` do project.yaml: o bruto é um
    cache temporário; os produtos finais são os Zarr diários. O expurgo usa
    somente checagens da execução corrente (nunca com --fast) e exige zarr e
    regrid válidos para o mesmo ano.
    """
    status: dict[tuple[Source, Stage, int | None], bool] = {}
    for check in checks:
        key = (check.source, check.stage, check.year)
        status[key] = check.ok and status.get(key, True)
    purged: list[Path] = []
    for (source, stage, year), ok in sorted(status.items(), key=lambda item: (item[0][0], str(item[0][2]))):
        if stage != "zarr" or not ok or year is None:
            continue
        if not status.get((source, "regrid", year), False):
            continue
        if source == "oisst":
            raw = oisst_raw_path(year)
        elif source == "chirps":
            raw = chirps_raw_path(year, resolution)
        else:
            continue
        if raw.exists():
            purged.append(raw)
            if execute:
                raw.unlink()
                print(f"purge raw {source} {year}: {raw}")
            else:
                print(f"purge raw pendente {source} {year}: {raw}")
    return purged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Curate local NINO-BRASIL data status and resume the next missing download/transform."
    )
    parser.add_argument("--source", default="core", help="core, all, or comma list: ibge,chirps,oisst,era5,ocean,ctd")
    parser.add_argument("--stage", default="all", help="all, raw, zarr, regrid, or comma list.")
    parser.add_argument("--start-year", type=int, default=1981)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--chirps-resolution", choices=["p25", "p05"], default="p25")
    parser.add_argument("--execute", action="store_true", help="Actually run pending actions. Without this, only reports.")
    parser.add_argument("--limit", type=int, default=1, help="Actions to run with --execute; use 0 for all pending actions.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Only check existence; do not open NetCDF/Zarr metadata.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=int, default=30)
    parser.add_argument("--continue-on-error", action="store_true", help="Keep running later pending actions after an item fails.")
    parser.add_argument(
        "--purge-raw",
        action="store_true",
        help="Apaga o NetCDF bruto anual de CHIRPS/OISST quando o Zarr diário e o regrid do ano validam (política raw_cache_policy).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sources = selected_sources(args.source)
    stages = selected_stages(args.stage)
    cfg = load_config()
    audit = AuditLog()

    checks: list[CheckResult] = []
    pending: list[PendingAction] = []

    if "ibge" in sources or "all" in sources:
        add_ibge_actions(checks, pending, selected_stages=stages, fast=args.fast, overwrite=args.overwrite)

    if "chirps" in sources or "all" in sources:
        record_source_latency(audit, "chirps", cfg)
        add_chirps_actions(
            checks,
            pending,
            years=iter_years("chirps", args.start_year, args.end_year),
            resolution=args.chirps_resolution,
            selected_stages=stages,
            fast=args.fast,
            overwrite=args.overwrite,
            retries=args.retries,
            retry_wait=args.retry_wait,
        )

    if "oisst" in sources or "all" in sources:
        record_source_latency(audit, "noaa_oisst", cfg)
        add_oisst_actions(
            checks,
            pending,
            years=iter_years("noaa_oisst", args.start_year, args.end_year),
            selected_stages=stages,
            fast=args.fast,
            overwrite=args.overwrite,
            retries=args.retries,
            retry_wait=args.retry_wait,
        )

    add_cds_ctd_placeholders(
        checks,
        pending,
        selected_sources=sources,
        selected_stages=stages,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    print_summary(checks, pending)
    returncode = 0
    if args.purge_raw:
        if args.fast:
            print("purge-raw ignorado: exige checagem completa (sem --fast).")
        else:
            # A flag pede o expurgo explicitamente; não depende de --execute,
            # que controla apenas as ações de download/transformação.
            purge_validated_raw(checks, resolution=args.chirps_resolution, execute=True)
    if args.execute:
        returncode = execute_pending(pending, limit=args.limit, continue_on_error=args.continue_on_error)
    else:
        print()
        print("Modo relatorio. Para executar a proxima pendencia:")
        print(
            "python scripts/curate_and_resume_downloads.py "
            f"--source {args.source} --stage {args.stage} --execute"
        )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
