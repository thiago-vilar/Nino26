from __future__ import annotations

import calendar
import json
import logging
import os
import shutil
import tempfile
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from nino_brasil.data.audit import AuditLog, dataset_summary, file_info
from nino_brasil.data.credentials import DEFAULT_CDS_API_URL, load_local_env
from nino_brasil.data.zarr_store import (
    ZARR_FORMAT,
    chunk_plan,
    netcdf_collection_to_daily_zarr,
    netcdf_to_daily_zarr,
    zip_netcdf_to_daily_zarr,
)


for _logger_name in ("cdsapi", "cads_api_client", "ecmwfapi"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)


ERA5_SINGLE_VARIABLES = [
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_column_water_vapour",
    "surface_latent_heat_flux",
    "surface_sensible_heat_flux",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
]

ERA5_PRESSURE_VARIABLES = [
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
    "vertical_velocity",
    "divergence",
]

ERA5_PRESSURE_LEVELS = ["200", "500", "850"]

# O CDS entrega estes fluxos/radiações como acumulações horárias em J m-2.
# A soma diária preserva a grandeza acumulada. Campos instantâneos usam média.
ERA5_SINGLE_ACCUMULATED_VARIABLES = frozenset(
    {
        "surface_latent_heat_flux",
        "surface_sensible_heat_flux",
        "surface_net_solar_radiation",
        "surface_net_thermal_radiation",
    }
)


def era5_daily_aggregation(variable: str) -> str:
    return "sum" if variable in ERA5_SINGLE_ACCUMULATED_VARIABLES else "mean"

ERA5_SINGLE_VARIABLE_ALIASES = {
    "mean_sea_level_pressure": ["msl"],
    "10m_u_component_of_wind": ["u10"],
    "10m_v_component_of_wind": ["v10"],
    "total_column_water_vapour": ["tcwv"],
    "surface_latent_heat_flux": ["slhf"],
    "surface_sensible_heat_flux": ["sshf"],
    "surface_net_solar_radiation": ["ssr"],
    "surface_net_thermal_radiation": ["str"],
}

ERA5_PRESSURE_VARIABLE_ALIASES = {
    "u_component_of_wind": ["u"],
    "v_component_of_wind": ["v"],
    "specific_humidity": ["q"],
    "geopotential": ["z"],
    "vertical_velocity": ["w"],
    "divergence": ["d"],
}

ATMOSPHERE_AREAS = {
    "nino34": [5, -170, -5, -120],
    "brazil": [7, -75, -35, -30],
    "iod_west": [10, 50, -10, 70],
    "iod_east": [0, 90, -10, 110],
}

def _days(year: int, month: int) -> list[str]:
    last_day = calendar.monthrange(year, month)[1]
    available = date.today() - timedelta(days=7)
    if (year, month) == (available.year, available.month):
        last_day = min(last_day, available.day)
    elif (year, month) > (available.year, available.month):
        return []
    return [f"{day:02d}" for day in range(1, last_day + 1)]


def _all_days() -> list[str]:
    return [f"{day:02d}" for day in range(1, 32)]


def _months(months: list[int] | None = None) -> list[str]:
    return [f"{month:02d}" for month in (months or list(range(1, 13)))]


def _month(month: int) -> str:
    return f"{month:02d}"


def _month_bounds(year: int, month: int) -> tuple[str, str]:
    days = _days(year, month)
    if not days:
        raise ValueError(f"ERA5 ainda indisponível para {year}-{month:02d}")
    last_day = int(days[-1])
    return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}"


def _months_bounds(year: int, months: list[int] | None = None) -> tuple[str, str]:
    selected_months = sorted(set(months or list(range(1, 13))))
    if not selected_months:
        raise ValueError("At least one month is required.")
    invalid = [month for month in selected_months if month < 1 or month > 12]
    if invalid:
        raise ValueError(f"Invalid month(s): {invalid}")
    start_month = selected_months[0]
    end_month = selected_months[-1]
    last_day = calendar.monthrange(year, end_month)[1]
    return f"{year}-{start_month:02d}-01", f"{year}-{end_month:02d}-{last_day:02d}"


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


def _resolve_variables(variables: list[str] | None, default: list[str]) -> list[str]:
    return list(variables) if variables else list(default)


def _variable_slug(variables: list[str] | None) -> str | None:
    if not variables:
        return None
    return "__".join(_safe_name(variable) for variable in variables)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    parsed = int(value)
    return max(minimum, parsed)


def _env_float(name: str, default: float, *, minimum: float = 1.0) -> float:
    value = os.environ.get(name)
    if not value:
        return default
    parsed = float(value)
    return max(minimum, parsed)


def _group_slug(variables: list[str], default: list[str]) -> str:
    return "all_variables" if variables == default else _variable_slug(variables) or "selected_variables"


def _era5_annual_zarr_path(
    *,
    zarr_root: Path,
    kind: str,
    year: int,
    region: str,
    variables: list[str] | None = None,
) -> Path:
    slug = _variable_slug(variables)
    type_dir = "single_levels" if kind == "single" else "pressure_levels"
    prefix = "era5_single" if kind == "single" else "era5_pressure"
    if slug:
        return zarr_root / "era5" / type_dir / str(year) / slug / f"{prefix}_{region}_{slug}_{year}_daily.zarr"
    return zarr_root / "era5" / type_dir / str(year) / f"{prefix}_{region}_{year}_daily.zarr"


def _era5_variable_zarr_path(
    *,
    zarr_root: Path,
    kind: str,
    year: int,
    region: str,
    variable: str,
) -> Path:
    return _era5_annual_zarr_path(
        zarr_root=zarr_root,
        kind=kind,
        year=year,
        region=region,
        variables=[variable],
    )

def _zarr_valid(path: Path, *, expected_variable: str | None = None) -> bool:
    if not path.exists():
        return False
    # O resumo de auditoria não inclui atributos globais. Ler diretamente os
    # metadados impede que acumulados válidos sejam baixados em toda execução.
    with xr.open_zarr(path, consolidated=None) as dataset:
        variables = set(dataset.data_vars)
        attrs = dict(dataset.attrs)
    if not variables:
        return False
    if expected_variable and expected_variable not in variables:
        return False
    if expected_variable:
        expected_aggregation = era5_daily_aggregation(expected_variable)
        recorded = str(attrs.get("nino_brasil_daily_aggregation", ""))
        # Produtos legados instantâneos já eram calculados por média e podem
        # ser reutilizados. Acumulações legadas sem contrato explícito devem
        # ser refeitas, pois antes também recebiam média indevidamente.
        if recorded != expected_aggregation and not (not recorded and expected_aggregation == "mean"):
            return False
    return True


def _zarr_daily_values_complete(
    path: Path,
    *,
    variable: str,
    expected_start: pd.Timestamp,
    expected_end: pd.Timestamp,
) -> bool:
    """Validate daily calendar coverage using Zarr metadata only.

    Phase 1 uses this inexpensive inventory check to decide whether a remote
    request is needed. Reading every spatial chunk of every existing store made
    a no-op update slower than the original download. Value-level/finite-data
    sanity remains the responsibility of the Phase 2 audit.
    """
    if not path.exists():
        return False
    try:
        with xr.open_zarr(path, consolidated=None) as dataset:
            if variable not in dataset or "time" not in dataset.coords:
                return False
            array = dataset[variable]
            if "time" not in array.dims:
                return False
            valid_days = pd.DatetimeIndex(array["time"].values).normalize()
        expected = pd.date_range(expected_start, expected_end, freq="D")
        return (
            valid_days.is_unique
            and valid_days.is_monotonic_increasing
            and expected.difference(valid_days).empty
        )
    except (OSError, ValueError, KeyError, IndexError):
        return False


def _era5_expected_month_bounds(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = _month_bounds(year, month)
    availability = pd.Timestamp.today().normalize() - pd.Timedelta(days=7)
    return pd.Timestamp(start), min(pd.Timestamp(end), availability)


def era5_month_zarr_path(
    zarr_root: Path,
    *,
    kind: str,
    year: int,
    month: int,
    region: str,
    variable: str,
) -> Path:
    """Return the canonical monthly ERA5 daily store path."""
    task_id = f"era5_{kind}_{region}_{variable}_{year}{month:02d}"
    return zarr_root / "era5" / f"{kind}_levels" / str(year) / variable / f"{task_id}_daily.zarr"


def era5_month_zarr_complete(
    zarr_root: Path,
    *,
    kind: str,
    year: int,
    month: int,
    region: str,
    variable: str,
) -> bool:
    """Fast inventory predicate for one canonical ERA5 monthly store."""
    path = era5_month_zarr_path(
        zarr_root, kind=kind, year=year, month=month, region=region, variable=variable
    )
    expected_start, expected_end = _era5_expected_month_bounds(year, month)
    return _zarr_valid(path, expected_variable=variable) and _zarr_daily_values_complete(
        path,
        variable=variable,
        expected_start=expected_start,
        expected_end=expected_end,
    )


def _all_zarrs_valid(paths: list[Path], *, overwrite: bool, expected_variables: list[str] | None = None) -> bool:
    if overwrite:
        return False
    for index, path in enumerate(paths):
        expected_variable = expected_variables[index] if expected_variables and index < len(expected_variables) else None
        if not _zarr_valid(path, expected_variable=expected_variable):
            return False
    return True


def _combine_monthly_zarrs_to_annual(
    monthly_paths: list[Path],
    annual_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    if annual_path.exists() and not overwrite:
        dataset_summary(annual_path, zarr=True)
        print(f"skip valid annual zarr: {annual_path}")
        return annual_path

    missing = [path for path in monthly_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing monthly ERA5 Zarr stores before annual consolidation: {missing[:3]}")

    if annual_path.exists() and overwrite:
        shutil.rmtree(annual_path)

    annual_path.parent.mkdir(parents=True, exist_ok=True)
    opened = [xr.open_zarr(path, chunks={}) for path in monthly_paths]
    try:
        annual = xr.concat(opened, dim="time", data_vars="all", coords="minimal", compat="override").sortby("time")
        annual.attrs.update(
            {
                "nino_brasil_temporal_standard": "daily",
                "nino_brasil_era5_storage": "annual_zarr_from_monthly_cache",
            }
        )
        chunks = chunk_plan(annual)
        if chunks:
            annual = annual.chunk(chunks)
        annual.to_zarr(annual_path, mode="w", consolidated=True, zarr_format=ZARR_FORMAT)
        annual.close()
    finally:
        for ds in opened:
            ds.close()

    dataset_summary(annual_path, zarr=True)
    print(f"annual zarr written: {annual_path}")
    return annual_path


def _client() -> cdsapi.Client:
    load_local_env()
    key = os.environ.get("CDS_API_KEY")
    url = os.environ.get("CDS_API_URL", DEFAULT_CDS_API_URL)
    if not key:
        raise RuntimeError("CDS_API_KEY is not set.")
    if os.environ.get("NINO_CDS_VERBOSE", "").lower() not in {"1", "true", "yes"}:
        logging.disable(logging.INFO)
    retry_max = _env_int("NINO_CDS_RETRY_MAX", 5)
    sleep_max = _env_float("NINO_CDS_SLEEP_MAX", 60.0)
    timeout = _env_float("NINO_CDS_TIMEOUT", 120.0)
    return cdsapi.Client(
        url=url,
        key=key,
        retry_max=retry_max,
        sleep_max=sleep_max,
        timeout=timeout,
    )


def _download_cds_result(result: object, target: Path, *, label: str | None = None) -> Path:
    size = int(getattr(result, "content_length", 0) or 0)
    url = getattr(result, "location")
    session = getattr(result, "session")
    verify = getattr(result, "verify", True)
    timeout = getattr(result, "timeout", 60)
    retry_max = int(getattr(result, "retry_max", 5) or 5)
    sleep_max = float(getattr(result, "sleep_max", 120) or 120)

    mode = "w"
    downloaded = 0
    headers = None
    wait_seconds = 10.0
    desc = f"{label} download" if label else "download"

    for _ in range(retry_max):
        response = session.get(url, stream=True, verify=verify, headers=headers, timeout=timeout)
        try:
            response.raise_for_status()
            with tqdm(
                total=size or None,
                initial=downloaded,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                ascii=True,
                leave=True,
                desc=desc,
            ) as pbar:
                with target.open(mode + "b") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))
        finally:
            response.close()

        if not size or downloaded >= size:
            return target

        time.sleep(wait_seconds)
        mode = "a"
        downloaded = target.stat().st_size if target.exists() else 0
        headers = {"Range": f"bytes={downloaded}-"}
        wait_seconds = min(sleep_max, wait_seconds * 1.5)

    raise RuntimeError(f"Download incomplete for {target}: {downloaded}/{size} bytes.")


def retrieve_cds(
    *,
    dataset: str,
    request: dict[str, object],
    output_path: Path,
    dry_run: bool = True,
    overwrite: bool = False,
    quiet: bool = False,
    label: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"DRY RUN cds {dataset} -> {output_path}")
        print(json.dumps(request, indent=2))
        return output_path

    if output_path.exists() and not overwrite:
        if not quiet:
            print(f"exists: {output_path}")
        return output_path

    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    client = _client()
    try:
        if label:
            print(f"{label} - aguardando CDS")
        result = client.retrieve(dataset, request)
        if label:
            print(f"{label} - transferindo")
        _download_cds_result(result, temp_path, label=label)
        temp_path.replace(output_path)
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
        raise
    if not quiet:
        print(f"downloaded: {output_path}")
    return output_path


def _cache_to_daily_zarr(
    raw_path: Path,
    zarr_path: Path,
    *,
    variables: list[str],
    variable_aliases: dict[str, list[str]],
    source_frequency: str,
    aggregation: str = "mean",
    daily_start: str,
    daily_end: str,
    overwrite: bool = False,
    quiet: bool = False,
) -> Path:
    if zipfile.is_zipfile(raw_path):
        with tempfile.TemporaryDirectory(prefix=f"{raw_path.stem}_", dir=raw_path.parent) as tmp_dir:
            return zip_netcdf_to_daily_zarr(
                raw_path,
                Path(tmp_dir),
                zarr_path,
                variables=variables,
                variable_aliases=variable_aliases,
                source_frequency=source_frequency,
                aggregation=aggregation,
                daily_start=daily_start,
                daily_end=daily_end,
                overwrite=overwrite,
                quiet=quiet,
            )
    return netcdf_to_daily_zarr(
        raw_path,
        zarr_path,
        variables=variables,
        variable_aliases=variable_aliases,
        source_frequency=source_frequency,
        aggregation=aggregation,
        daily_start=daily_start,
        daily_end=daily_end,
        overwrite=overwrite,
        quiet=quiet,
    )


def _netcdf_files_with_variable(
    nc_files: list[Path],
    variable: str,
    aliases: dict[str, list[str]],
) -> list[Path]:
    candidates = {variable, *aliases.get(variable, [])}
    matches: list[Path] = []
    for path in nc_files:
        ds = xr.open_dataset(path)
        try:
            if candidates.intersection(ds.data_vars):
                matches.append(path)
        finally:
            ds.close()
    return matches


def download_era5_single_month(
    *,
    year: int,
    month: int,
    region: str,
    raw_dir: Path,
    variables: list[str] | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
) -> Path:
    request = _era5_single_request(year, month, region, variables=variables)
    slug = _variable_slug(variables)
    if slug:
        output_path = raw_dir / "single_levels" / str(year) / slug / f"era5_single_{region}_{slug}_{year}{month:02d}.nc"
    else:
        output_path = raw_dir / "single_levels" / str(year) / f"era5_single_{region}_{year}{month:02d}.nc"
    return retrieve_cds(
        dataset="reanalysis-era5-single-levels",
        request=request,
        output_path=output_path,
        dry_run=dry_run,
        overwrite=overwrite,
    )


def download_era5_pressure_month(
    *,
    year: int,
    month: int,
    region: str,
    raw_dir: Path,
    variables: list[str] | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
) -> Path:
    request = _era5_pressure_request(year, month, region, variables=variables)
    slug = _variable_slug(variables)
    if slug:
        output_path = raw_dir / "pressure_levels" / str(year) / slug / f"era5_pressure_{region}_{slug}_{year}{month:02d}.nc"
    else:
        output_path = raw_dir / "pressure_levels" / str(year) / f"era5_pressure_{region}_{year}{month:02d}.nc"
    return retrieve_cds(
        dataset="reanalysis-era5-pressure-levels",
        request=request,
        output_path=output_path,
        dry_run=dry_run,
        overwrite=overwrite,
    )


def _task_record_start(
    audit: AuditLog,
    *,
    task_id: str,
    dataset: str,
    raw_path: Path,
    zarr_path: Path,
    request: dict[str, object],
) -> None:
    audit.record(
        task_id=task_id,
        dataset=dataset,
        status="started",
        raw_path=str(raw_path),
        zarr_path=str(zarr_path),
        request=request,
    )


def _record_ok(
    audit: AuditLog,
    *,
    task_id: str,
    dataset: str,
    raw_path: Path,
    zarr_path: Path,
    include_hash: bool,
) -> None:
    audit.record(
        task_id=task_id,
        dataset=dataset,
        status="ok",
        raw=file_info(raw_path, include_hash=include_hash),
        zarr={"path": str(zarr_path), "summary": dataset_summary(zarr_path, zarr=True)},
    )


def _record_error(
    audit: AuditLog,
    *,
    task_id: str,
    dataset: str,
    error: BaseException,
) -> None:
    audit.record(
        task_id=task_id,
        dataset=dataset,
        status="error",
        error_type=type(error).__name__,
        error=str(error),
    )


def _era5_task_label(*, year: int, region: str, kind: str, variable: str) -> str:
    return f"ERA5 {year} {region} {kind} {variable}"


def _delete_raw_cache(
    raw_path: Path,
    audit: AuditLog,
    *,
    task_id: str,
    dataset: str,
    label: str | None = None,
) -> None:
    if not raw_path.exists():
        return
    size_bytes = raw_path.stat().st_size
    raw_path.unlink()
    audit.record(
        task_id=task_id,
        dataset=dataset,
        status="raw_cache_deleted",
        raw_path=str(raw_path),
        size_bytes=size_bytes,
    )
    print(f"{label or task_id} - raw cache apagado")


def _delete_dir_cache(
    path: Path,
    audit: AuditLog,
    *,
    task_id: str,
    dataset: str,
    label: str | None = None,
) -> None:
    if not path.exists():
        return
    shutil.rmtree(path)
    audit.record(
        task_id=task_id,
        dataset=dataset,
        status="interim_cache_deleted",
        path=str(path),
    )
    print(f"{label or task_id} - interim cache apagado")


def _era5_single_request(year: int, month: int, region: str, *, variables: list[str] | None = None) -> dict[str, object]:
    return {
        "product_type": "reanalysis",
        "variable": _resolve_variables(variables, ERA5_SINGLE_VARIABLES),
        "year": str(year),
        "month": _month(month),
        "day": _days(year, month),
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": ATMOSPHERE_AREAS[region],
    }


def _era5_pressure_request(year: int, month: int, region: str, *, variables: list[str] | None = None) -> dict[str, object]:
    return {
        "product_type": "reanalysis",
        "variable": _resolve_variables(variables, ERA5_PRESSURE_VARIABLES),
        "pressure_level": ERA5_PRESSURE_LEVELS,
        "year": str(year),
        "month": _month(month),
        "day": _days(year, month),
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": ATMOSPHERE_AREAS[region],
    }


def _era5_single_year_request(year: int, region: str, *, variable: str) -> dict[str, object]:
    return {
        "product_type": "reanalysis",
        "variable": [variable],
        "year": str(year),
        "month": _months(),
        "day": _all_days(),
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": ATMOSPHERE_AREAS[region],
    }


def _era5_single_year_kind_request(year: int, region: str, *, variables: list[str] | None = None) -> dict[str, object]:
    return {
        "product_type": "reanalysis",
        "variable": _resolve_variables(variables, ERA5_SINGLE_VARIABLES),
        "year": str(year),
        "month": _months(),
        "day": _all_days(),
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": ATMOSPHERE_AREAS[region],
    }


def _era5_pressure_year_request(year: int, region: str, *, variable: str) -> dict[str, object]:
    return {
        "product_type": "reanalysis",
        "variable": [variable],
        "pressure_level": ERA5_PRESSURE_LEVELS,
        "year": str(year),
        "month": _months(),
        "day": _all_days(),
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": ATMOSPHERE_AREAS[region],
    }


def _era5_pressure_year_kind_request(year: int, region: str, *, variables: list[str] | None = None) -> dict[str, object]:
    return {
        "product_type": "reanalysis",
        "variable": _resolve_variables(variables, ERA5_PRESSURE_VARIABLES),
        "pressure_level": ERA5_PRESSURE_LEVELS,
        "year": str(year),
        "month": _months(),
        "day": _all_days(),
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": ATMOSPHERE_AREAS[region],
    }


def ingest_era5_single_month(
    *,
    year: int,
    month: int,
    region: str,
    raw_dir: Path,
    zarr_root: Path,
    variables: list[str] | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    include_hash: bool = False,
    audit: AuditLog | None = None,
) -> Path:
    audit = audit or AuditLog()
    requested_variables = _resolve_variables(variables, ERA5_SINGLE_VARIABLES)
    if len(requested_variables) != 1:
        raise ValueError("ERA5 mensal deve ser solicitado por variável para preservar a agregação diária correta.")
    slug = _variable_slug(variables)
    task_id = f"era5_single_{region}_{slug}_{year}{month:02d}" if slug else f"era5_single_{region}_{year}{month:02d}"
    request = _era5_single_request(year, month, region, variables=variables)
    if slug:
        raw_path = raw_dir / "single_levels" / str(year) / slug / f"{task_id}.nc"
        zarr_path = zarr_root / "era5" / "single_levels" / str(year) / slug / f"{task_id}_daily.zarr"
    else:
        raw_path = raw_dir / "single_levels" / str(year) / f"{task_id}.nc"
        zarr_path = zarr_root / "era5" / "single_levels" / str(year) / f"{task_id}_daily.zarr"

    if dry_run:
        print(f"DRY RUN ingest {task_id}: raw={raw_path} zarr={zarr_path}")
        return zarr_path

    _task_record_start(audit, task_id=task_id, dataset="era5_single", raw_path=raw_path, zarr_path=zarr_path, request=request)
    try:
        expected_variable = requested_variables[0] if len(requested_variables) == 1 else None
        expected_start, expected_end = _era5_expected_month_bounds(year, month)
        valid_existing = _zarr_valid(zarr_path, expected_variable=expected_variable) and _zarr_daily_values_complete(
            zarr_path,
            variable=requested_variables[0],
            expected_start=expected_start,
            expected_end=expected_end,
        )
        replace_invalid = zarr_path.exists() and not valid_existing
        if valid_existing and not overwrite:
            print(f"skip valid zarr: {zarr_path}")
        else:
            retrieve_cds(
                dataset="reanalysis-era5-single-levels",
                request=request,
                output_path=raw_path,
                dry_run=False,
                overwrite=overwrite or replace_invalid,
            )
            start, end = _month_bounds(year, month)
            _cache_to_daily_zarr(
                raw_path,
                zarr_path,
                variables=requested_variables,
                variable_aliases=ERA5_SINGLE_VARIABLE_ALIASES,
                source_frequency="subdaily",
                aggregation=era5_daily_aggregation(requested_variables[0]),
                daily_start=start,
                daily_end=end,
                overwrite=overwrite or replace_invalid,
            )
        _record_ok(audit, task_id=task_id, dataset="era5_single", raw_path=raw_path, zarr_path=zarr_path, include_hash=include_hash)
        if raw_path.exists():
            raw_path.unlink()
            print(f"{task_id} - raw cache apagado")
        return zarr_path
    except BaseException as exc:
        _record_error(audit, task_id=task_id, dataset="era5_single", error=exc)
        raise


def ingest_era5_pressure_month(
    *,
    year: int,
    month: int,
    region: str,
    raw_dir: Path,
    zarr_root: Path,
    variables: list[str] | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    include_hash: bool = False,
    audit: AuditLog | None = None,
) -> Path:
    audit = audit or AuditLog()
    requested_variables = _resolve_variables(variables, ERA5_PRESSURE_VARIABLES)
    if len(requested_variables) != 1:
        raise ValueError("ERA5 mensal deve ser solicitado por variável para preservar a agregação diária correta.")
    slug = _variable_slug(variables)
    task_id = f"era5_pressure_{region}_{slug}_{year}{month:02d}" if slug else f"era5_pressure_{region}_{year}{month:02d}"
    request = _era5_pressure_request(year, month, region, variables=variables)
    if slug:
        raw_path = raw_dir / "pressure_levels" / str(year) / slug / f"{task_id}.nc"
        zarr_path = zarr_root / "era5" / "pressure_levels" / str(year) / slug / f"{task_id}_daily.zarr"
    else:
        raw_path = raw_dir / "pressure_levels" / str(year) / f"{task_id}.nc"
        zarr_path = zarr_root / "era5" / "pressure_levels" / str(year) / f"{task_id}_daily.zarr"

    if dry_run:
        print(f"DRY RUN ingest {task_id}: raw={raw_path} zarr={zarr_path}")
        return zarr_path

    _task_record_start(audit, task_id=task_id, dataset="era5_pressure", raw_path=raw_path, zarr_path=zarr_path, request=request)
    try:
        expected_variable = requested_variables[0] if len(requested_variables) == 1 else None
        expected_start, expected_end = _era5_expected_month_bounds(year, month)
        valid_existing = _zarr_valid(zarr_path, expected_variable=expected_variable) and _zarr_daily_values_complete(
            zarr_path,
            variable=requested_variables[0],
            expected_start=expected_start,
            expected_end=expected_end,
        )
        replace_invalid = zarr_path.exists() and not valid_existing
        if valid_existing and not overwrite:
            print(f"skip valid zarr: {zarr_path}")
        else:
            retrieve_cds(
                dataset="reanalysis-era5-pressure-levels",
                request=request,
                output_path=raw_path,
                dry_run=False,
                overwrite=overwrite or replace_invalid,
            )
            start, end = _month_bounds(year, month)
            _cache_to_daily_zarr(
                raw_path,
                zarr_path,
                variables=requested_variables,
                variable_aliases=ERA5_PRESSURE_VARIABLE_ALIASES,
                source_frequency="subdaily",
                aggregation=era5_daily_aggregation(requested_variables[0]),
                daily_start=start,
                daily_end=end,
                overwrite=overwrite or replace_invalid,
            )
        _record_ok(audit, task_id=task_id, dataset="era5_pressure", raw_path=raw_path, zarr_path=zarr_path, include_hash=include_hash)
        if raw_path.exists():
            raw_path.unlink()
            print(f"{task_id} - raw cache apagado")
        return zarr_path
    except BaseException as exc:
        _record_error(audit, task_id=task_id, dataset="era5_pressure", error=exc)
        raise


def ingest_era5_single_year(
    *,
    year: int,
    region: str,
    raw_dir: Path,
    zarr_root: Path,
    months: list[int] | None = None,
    variables: list[str] | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    include_hash: bool = False,
    audit: AuditLog | None = None,
) -> Path:
    audit = audit or AuditLog()
    selected_months = months or list(range(1, 13))
    annual_path = _era5_annual_zarr_path(
        zarr_root=zarr_root,
        kind="single",
        year=year,
        region=region,
        variables=variables,
    )
    slug = _variable_slug(variables)
    task_id = f"era5_single_{region}_{slug}_{year}" if slug else f"era5_single_{region}_{year}"

    if dry_run:
        print(f"DRY RUN annual ingest {task_id}: zarr={annual_path}")
        return annual_path

    if annual_path.exists() and not overwrite:
        dataset_summary(annual_path, zarr=True)
        print(f"skip valid annual zarr: {annual_path}")
        return annual_path

    audit.record(
        task_id=task_id,
        dataset="era5_single_annual",
        status="started",
        year=year,
        region=region,
        months=selected_months,
        zarr_path=str(annual_path),
    )
    try:
        monthly_paths = [
            ingest_era5_single_month(
                year=year,
                month=month,
                region=region,
                raw_dir=raw_dir,
                zarr_root=zarr_root,
                variables=variables,
                dry_run=False,
                overwrite=overwrite,
                include_hash=include_hash,
                audit=audit,
            )
            for month in selected_months
        ]
        result = _combine_monthly_zarrs_to_annual(monthly_paths, annual_path, overwrite=overwrite)
        audit.record(
            task_id=task_id,
            dataset="era5_single_annual",
            status="ok",
            year=year,
            region=region,
            zarr={"path": str(result), "summary": dataset_summary(result, zarr=True)},
        )
        return result
    except BaseException as exc:
        _record_error(audit, task_id=task_id, dataset="era5_single_annual", error=exc)
        raise


def ingest_era5_pressure_year(
    *,
    year: int,
    region: str,
    raw_dir: Path,
    zarr_root: Path,
    months: list[int] | None = None,
    variables: list[str] | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    include_hash: bool = False,
    audit: AuditLog | None = None,
) -> Path:
    audit = audit or AuditLog()
    selected_months = months or list(range(1, 13))
    annual_path = _era5_annual_zarr_path(
        zarr_root=zarr_root,
        kind="pressure",
        year=year,
        region=region,
        variables=variables,
    )
    slug = _variable_slug(variables)
    task_id = f"era5_pressure_{region}_{slug}_{year}" if slug else f"era5_pressure_{region}_{year}"

    if dry_run:
        print(f"DRY RUN annual ingest {task_id}: zarr={annual_path}")
        return annual_path

    if annual_path.exists() and not overwrite:
        dataset_summary(annual_path, zarr=True)
        print(f"skip valid annual zarr: {annual_path}")
        return annual_path

    audit.record(
        task_id=task_id,
        dataset="era5_pressure_annual",
        status="started",
        year=year,
        region=region,
        months=selected_months,
        zarr_path=str(annual_path),
    )
    try:
        monthly_paths = [
            ingest_era5_pressure_month(
                year=year,
                month=month,
                region=region,
                raw_dir=raw_dir,
                zarr_root=zarr_root,
                variables=variables,
                dry_run=False,
                overwrite=overwrite,
                include_hash=include_hash,
                audit=audit,
            )
            for month in selected_months
        ]
        result = _combine_monthly_zarrs_to_annual(monthly_paths, annual_path, overwrite=overwrite)
        audit.record(
            task_id=task_id,
            dataset="era5_pressure_annual",
            status="ok",
            year=year,
            region=region,
            zarr={"path": str(result), "summary": dataset_summary(result, zarr=True)},
        )
        return result
    except BaseException as exc:
        _record_error(audit, task_id=task_id, dataset="era5_pressure_annual", error=exc)
        raise


def ingest_era5_single_year_variable(
    *,
    year: int,
    region: str,
    variable: str,
    raw_dir: Path,
    zarr_root: Path,
    dry_run: bool = True,
    overwrite: bool = False,
    include_hash: bool = False,
    delete_raw_after_zarr: bool = False,
    audit: AuditLog | None = None,
) -> Path:
    if variable not in ERA5_SINGLE_VARIABLES:
        raise ValueError(f"No requested variable belongs to ERA5 single levels: {variable}")
    audit = audit or AuditLog()
    request = _era5_single_year_request(year, region, variable=variable)
    slug = _safe_name(variable)
    task_id = f"era5_single_{region}_{slug}_{year}"
    label = _era5_task_label(year=year, region=region, kind="single", variable=variable)
    raw_path = raw_dir / "single_levels" / str(year) / slug / f"{task_id}.nc"
    zarr_path = _era5_annual_zarr_path(
        zarr_root=zarr_root,
        kind="single",
        year=year,
        region=region,
        variables=[variable],
    )

    if dry_run:
        print(f"DRY RUN annual variable ingest {task_id}: raw={raw_path} zarr={zarr_path}")
        print(json.dumps(request, indent=2))
        return zarr_path

    _task_record_start(audit, task_id=task_id, dataset="era5_single_annual_variable", raw_path=raw_path, zarr_path=zarr_path, request=request)
    try:
        if _zarr_valid(zarr_path, expected_variable=variable) and not overwrite:
            print(f"{label} - ja existe")
        else:
            print(f"{label} - baixando")
            retrieve_cds(
                dataset="reanalysis-era5-single-levels",
                request=request,
                output_path=raw_path,
                dry_run=False,
                overwrite=overwrite,
                quiet=True,
                label=label,
            )
            print(f"{label} - convertendo zarr")
            _cache_to_daily_zarr(
                raw_path,
                zarr_path,
                variables=[variable],
                variable_aliases=ERA5_SINGLE_VARIABLE_ALIASES,
                source_frequency="subdaily",
                aggregation=era5_daily_aggregation(variable),
                daily_start=f"{year}-01-01",
                daily_end=f"{year}-12-31",
                overwrite=overwrite,
                quiet=True,
            )
            print(f"{label} - ok")
        _record_ok(audit, task_id=task_id, dataset="era5_single_annual_variable", raw_path=raw_path, zarr_path=zarr_path, include_hash=include_hash)
        if delete_raw_after_zarr:
            _delete_raw_cache(raw_path, audit, task_id=task_id, dataset="era5_single_annual_variable", label=label)
        return zarr_path
    except BaseException as exc:
        _record_error(audit, task_id=task_id, dataset="era5_single_annual_variable", error=exc)
        raise


def ingest_era5_pressure_year_variable(
    *,
    year: int,
    region: str,
    variable: str,
    raw_dir: Path,
    zarr_root: Path,
    dry_run: bool = True,
    overwrite: bool = False,
    include_hash: bool = False,
    delete_raw_after_zarr: bool = False,
    audit: AuditLog | None = None,
) -> Path:
    if variable not in ERA5_PRESSURE_VARIABLES:
        raise ValueError(f"No requested variable belongs to ERA5 pressure levels: {variable}")
    audit = audit or AuditLog()
    request = _era5_pressure_year_request(year, region, variable=variable)
    slug = _safe_name(variable)
    task_id = f"era5_pressure_{region}_{slug}_{year}"
    label = _era5_task_label(year=year, region=region, kind="pressure", variable=variable)
    raw_path = raw_dir / "pressure_levels" / str(year) / slug / f"{task_id}.nc"
    zarr_path = _era5_annual_zarr_path(
        zarr_root=zarr_root,
        kind="pressure",
        year=year,
        region=region,
        variables=[variable],
    )

    if dry_run:
        print(f"DRY RUN annual variable ingest {task_id}: raw={raw_path} zarr={zarr_path}")
        print(json.dumps(request, indent=2))
        return zarr_path

    _task_record_start(audit, task_id=task_id, dataset="era5_pressure_annual_variable", raw_path=raw_path, zarr_path=zarr_path, request=request)
    try:
        if _zarr_valid(zarr_path, expected_variable=variable) and not overwrite:
            print(f"{label} - ja existe")
        else:
            print(f"{label} - baixando")
            retrieve_cds(
                dataset="reanalysis-era5-pressure-levels",
                request=request,
                output_path=raw_path,
                dry_run=False,
                overwrite=overwrite,
                quiet=True,
                label=label,
            )
            print(f"{label} - convertendo zarr")
            _cache_to_daily_zarr(
                raw_path,
                zarr_path,
                variables=[variable],
                variable_aliases=ERA5_PRESSURE_VARIABLE_ALIASES,
                source_frequency="subdaily",
                aggregation=era5_daily_aggregation(variable),
                daily_start=f"{year}-01-01",
                daily_end=f"{year}-12-31",
                overwrite=overwrite,
                quiet=True,
            )
            print(f"{label} - ok")
        _record_ok(audit, task_id=task_id, dataset="era5_pressure_annual_variable", raw_path=raw_path, zarr_path=zarr_path, include_hash=include_hash)
        if delete_raw_after_zarr:
            _delete_raw_cache(raw_path, audit, task_id=task_id, dataset="era5_pressure_annual_variable", label=label)
        return zarr_path
    except BaseException as exc:
        _record_error(audit, task_id=task_id, dataset="era5_pressure_annual_variable", error=exc)
        raise


def _ingest_era5_year_kind(
    *,
    year: int,
    region: str,
    kind: str,
    raw_dir: Path,
    zarr_root: Path,
    variables: list[str] | None,
    default_variables: list[str],
    request: dict[str, object],
    variable_aliases: dict[str, list[str]],
    dataset_name: str,
    dry_run: bool,
    overwrite: bool,
    include_hash: bool,
    delete_raw_after_zarr: bool,
    audit: AuditLog | None,
) -> list[Path]:
    audit = audit or AuditLog()
    selected_variables = _resolve_variables(variables, default_variables)
    slug = _group_slug(selected_variables, default_variables)
    type_dir = "single_levels" if kind == "single" else "pressure_levels"
    prefix = "era5_single" if kind == "single" else "era5_pressure"
    task_id = f"{prefix}_{region}_{slug}_{year}"
    label = f"ERA5 {year} {region} {kind} {slug}"
    raw_path = raw_dir / type_dir / str(year) / "_annual_kind" / f"{task_id}.nc"
    zarr_paths = [
        _era5_variable_zarr_path(
            zarr_root=zarr_root,
            kind=kind,
            year=year,
            region=region,
            variable=variable,
        )
        for variable in selected_variables
    ]

    if dry_run:
        print(f"DRY RUN annual-kind ingest {task_id}: raw={raw_path}")
        for variable, zarr_path in zip(selected_variables, zarr_paths):
            print(f"  zarr {variable}: {zarr_path}")
        print(json.dumps(request, indent=2))
        return zarr_paths

    all_valid = _all_zarrs_valid(zarr_paths, overwrite=overwrite, expected_variables=selected_variables)
    audit.record(
        task_id=task_id,
        dataset=dataset_name,
        status="started",
        year=year,
        region=region,
        variables=selected_variables,
        request_mode="annual_kind",
        raw_path=str(raw_path),
        zarr_paths=[str(path) for path in zarr_paths],
        request=request,
    )
    try:
        if all_valid:
            print(f"{label} - ja existe ({len(selected_variables)} variaveis)")
        else:
            print(f"{label} - baixando 1 requisicao CDS para {len(selected_variables)} variaveis")
            retrieve_cds(
                dataset="reanalysis-era5-single-levels" if kind == "single" else "reanalysis-era5-pressure-levels",
                request=request,
                output_path=raw_path,
                dry_run=False,
                overwrite=overwrite,
                quiet=True,
                label=label,
            )

        for variable, zarr_path in tqdm(
            list(zip(selected_variables, zarr_paths)),
            desc=f"{label} -> zarr",
            unit="var",
            ascii=True,
        ):
            variable_label = _era5_task_label(year=year, region=region, kind=kind, variable=variable)
            if _zarr_valid(zarr_path, expected_variable=variable) and not overwrite:
                print(f"{variable_label} - ja existe")
            else:
                print(f"{variable_label} - convertendo zarr")
                _cache_to_daily_zarr(
                    raw_path,
                    zarr_path,
                    variables=[variable],
                    variable_aliases=variable_aliases,
                    source_frequency="subdaily",
                    aggregation=era5_daily_aggregation(variable),
                    daily_start=f"{year}-01-01",
                    daily_end=f"{year}-12-31",
                    overwrite=overwrite,
                    quiet=True,
                )
                print(f"{variable_label} - ok")
            _record_ok(
                audit,
                task_id=f"{prefix}_{region}_{_safe_name(variable)}_{year}",
                dataset=f"{dataset_name}_variable",
                raw_path=raw_path,
                zarr_path=zarr_path,
                include_hash=include_hash,
            )

        audit.record(
            task_id=task_id,
            dataset=dataset_name,
            status="ok",
            year=year,
            region=region,
            variables=selected_variables,
            zarr_paths=[str(path) for path in zarr_paths],
        )
        if delete_raw_after_zarr:
            _delete_raw_cache(raw_path, audit, task_id=task_id, dataset=dataset_name, label=label)
        return zarr_paths
    except BaseException as exc:
        _record_error(audit, task_id=task_id, dataset=dataset_name, error=exc)
        raise


def ingest_era5_single_year_kind(
    *,
    year: int,
    region: str,
    raw_dir: Path,
    zarr_root: Path,
    variables: list[str] | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    include_hash: bool = False,
    delete_raw_after_zarr: bool = False,
    audit: AuditLog | None = None,
) -> list[Path]:
    selected_variables = _resolve_variables(variables, ERA5_SINGLE_VARIABLES)
    invalid = [variable for variable in selected_variables if variable not in ERA5_SINGLE_VARIABLES]
    if invalid:
        raise ValueError(f"No requested variable belongs to ERA5 single levels: {invalid}")
    return _ingest_era5_year_kind(
        year=year,
        region=region,
        kind="single",
        raw_dir=raw_dir,
        zarr_root=zarr_root,
        variables=selected_variables,
        default_variables=ERA5_SINGLE_VARIABLES,
        request=_era5_single_year_kind_request(year, region, variables=selected_variables),
        variable_aliases=ERA5_SINGLE_VARIABLE_ALIASES,
        dataset_name="era5_single_annual_kind",
        dry_run=dry_run,
        overwrite=overwrite,
        include_hash=include_hash,
        delete_raw_after_zarr=delete_raw_after_zarr,
        audit=audit,
    )


def ingest_era5_pressure_year_kind(
    *,
    year: int,
    region: str,
    raw_dir: Path,
    zarr_root: Path,
    variables: list[str] | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    include_hash: bool = False,
    delete_raw_after_zarr: bool = False,
    audit: AuditLog | None = None,
) -> list[Path]:
    selected_variables = _resolve_variables(variables, ERA5_PRESSURE_VARIABLES)
    invalid = [variable for variable in selected_variables if variable not in ERA5_PRESSURE_VARIABLES]
    if invalid:
        raise ValueError(f"No requested variable belongs to ERA5 pressure levels: {invalid}")
    return _ingest_era5_year_kind(
        year=year,
        region=region,
        kind="pressure",
        raw_dir=raw_dir,
        zarr_root=zarr_root,
        variables=selected_variables,
        default_variables=ERA5_PRESSURE_VARIABLES,
        request=_era5_pressure_year_kind_request(year, region, variables=selected_variables),
        variable_aliases=ERA5_PRESSURE_VARIABLE_ALIASES,
        dataset_name="era5_pressure_annual_kind",
        dry_run=dry_run,
        overwrite=overwrite,
        include_hash=include_hash,
        delete_raw_after_zarr=delete_raw_after_zarr,
        audit=audit,
    )
