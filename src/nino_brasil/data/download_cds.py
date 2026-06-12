from __future__ import annotations

import calendar
import json
import logging
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import cdsapi
import xarray as xr
from tqdm import tqdm

from nino_brasil.data.audit import AuditLog, dataset_summary, file_info
from nino_brasil.data.credentials import DEFAULT_CDS_API_URL, load_local_env
from nino_brasil.data.zarr_store import chunk_plan, netcdf_to_daily_zarr, zip_netcdf_to_daily_zarr


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

ORAS5_VARIABLES = [
    "potential_temperature",
    "salinity",
    "ocean_heat_content_for_the_upper_300m",
    "ocean_heat_content_for_the_upper_700m",
]

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

ORAS5_VARIABLE_ALIASES = {
    "potential_temperature": ["thetao", "temperature"],
    "salinity": ["so"],
    "ocean_heat_content_for_the_upper_300m": ["ohc_0_300m", "ohc300"],
    "ocean_heat_content_for_the_upper_700m": ["ohc_0_700m", "ohc700"],
}

ATMOSPHERE_AREAS = {
    "nino34": [5, -170, -5, -120],
    "brazil": [7, -75, -35, -30],
}


def _days(year: int, month: int) -> list[str]:
    last_day = calendar.monthrange(year, month)[1]
    return [f"{day:02d}" for day in range(1, last_day + 1)]


def _all_days() -> list[str]:
    return [f"{day:02d}" for day in range(1, 32)]


def _months(months: list[int] | None = None) -> list[str]:
    return [f"{month:02d}" for month in (months or list(range(1, 13)))]


def _month(month: int) -> str:
    return f"{month:02d}"


def _month_bounds(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]
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
        annual.to_zarr(annual_path, mode="w", consolidated=True, zarr_format=2)
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
    return cdsapi.Client(url=url, key=key)


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


def download_oras_month(
    *,
    year: int,
    month: int,
    raw_dir: Path,
    variables: list[str] | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
) -> Path:
    request = _oras_request(year, month, variables=variables)
    slug = _variable_slug(variables)
    if slug:
        output_path = raw_dir / str(year) / slug / f"oras5_{slug}_{year}{month:02d}.zip"
    else:
        output_path = raw_dir / str(year) / f"oras5_{year}{month:02d}.zip"
    return retrieve_cds(
        dataset="reanalysis-oras5",
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


def _oras_task_label(*, year: int, variable: str) -> str:
    return f"ORAS5 {year} {variable}"


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


def _oras_request(year: int, month: int, *, variables: list[str] | None = None) -> dict[str, object]:
    product_type = "consolidated" if year <= 2014 else "operational"
    return {
        "product_type": product_type,
        "vertical_resolution": "all_levels",
        "variable": _resolve_variables(variables, ORAS5_VARIABLES),
        "year": str(year),
        "month": _month(month),
        "data_format": "netcdf",
    }


def _oras_year_request(year: int, *, variable: str, months: list[int] | None = None) -> dict[str, object]:
    product_type = "consolidated" if year <= 2014 else "operational"
    return {
        "product_type": product_type,
        "vertical_resolution": "all_levels",
        "variable": [variable],
        "year": str(year),
        "month": _months(months),
        "data_format": "netcdf",
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
        if zarr_path.exists() and not overwrite:
            dataset_summary(zarr_path, zarr=True)
            print(f"skip valid zarr: {zarr_path}")
        else:
            retrieve_cds(
                dataset="reanalysis-era5-single-levels",
                request=request,
                output_path=raw_path,
                dry_run=False,
                overwrite=overwrite,
            )
            start, end = _month_bounds(year, month)
            _cache_to_daily_zarr(
                raw_path,
                zarr_path,
                variables=requested_variables,
                variable_aliases=ERA5_SINGLE_VARIABLE_ALIASES,
                source_frequency="subdaily",
                aggregation="mean",
                daily_start=start,
                daily_end=end,
                overwrite=overwrite,
            )
        _record_ok(audit, task_id=task_id, dataset="era5_single", raw_path=raw_path, zarr_path=zarr_path, include_hash=include_hash)
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
        if zarr_path.exists() and not overwrite:
            dataset_summary(zarr_path, zarr=True)
            print(f"skip valid zarr: {zarr_path}")
        else:
            retrieve_cds(
                dataset="reanalysis-era5-pressure-levels",
                request=request,
                output_path=raw_path,
                dry_run=False,
                overwrite=overwrite,
            )
            start, end = _month_bounds(year, month)
            _cache_to_daily_zarr(
                raw_path,
                zarr_path,
                variables=requested_variables,
                variable_aliases=ERA5_PRESSURE_VARIABLE_ALIASES,
                source_frequency="subdaily",
                aggregation="mean",
                daily_start=start,
                daily_end=end,
                overwrite=overwrite,
            )
        _record_ok(audit, task_id=task_id, dataset="era5_pressure", raw_path=raw_path, zarr_path=zarr_path, include_hash=include_hash)
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
        if zarr_path.exists() and not overwrite:
            dataset_summary(zarr_path, zarr=True)
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
                aggregation="mean",
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
        if zarr_path.exists() and not overwrite:
            dataset_summary(zarr_path, zarr=True)
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
                aggregation="mean",
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


def ingest_oras_month(
    *,
    year: int,
    month: int,
    raw_dir: Path,
    interim_dir: Path,
    zarr_root: Path,
    variables: list[str] | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    include_hash: bool = False,
    audit: AuditLog | None = None,
) -> Path:
    audit = audit or AuditLog()
    requested_variables = _resolve_variables(variables, ORAS5_VARIABLES)
    slug = _variable_slug(variables)
    task_id = f"oras5_{slug}_{year}{month:02d}" if slug else f"oras5_{year}{month:02d}"
    request = _oras_request(year, month, variables=variables)
    if slug:
        raw_path = raw_dir / str(year) / slug / f"{task_id}.zip"
        extract_dir = interim_dir / str(year) / slug / task_id
        zarr_path = zarr_root / "oras" / str(year) / slug / f"{task_id}_daily.zarr"
    else:
        raw_path = raw_dir / str(year) / f"{task_id}.zip"
        extract_dir = interim_dir / str(year) / task_id
        zarr_path = zarr_root / "oras" / str(year) / f"{task_id}_daily.zarr"

    if dry_run:
        print(f"DRY RUN ingest {task_id}: raw={raw_path} zarr={zarr_path}")
        return zarr_path

    _task_record_start(audit, task_id=task_id, dataset="oras5", raw_path=raw_path, zarr_path=zarr_path, request=request)
    try:
        if zarr_path.exists() and not overwrite:
            dataset_summary(zarr_path, zarr=True)
            print(f"skip valid zarr: {zarr_path}")
        else:
            retrieve_cds(
                dataset="reanalysis-oras5",
                request=request,
                output_path=raw_path,
                dry_run=False,
                overwrite=overwrite,
            )
            if extract_dir.exists() and overwrite:
                shutil.rmtree(extract_dir)
            start, end = _month_bounds(year, month)
            zip_netcdf_to_daily_zarr(
                raw_path,
                extract_dir,
                zarr_path,
                variables=requested_variables,
                variable_aliases=ORAS5_VARIABLE_ALIASES,
                source_frequency="monthly",
                daily_start=start,
                daily_end=end,
                overwrite=overwrite,
            )
        _record_ok(audit, task_id=task_id, dataset="oras5", raw_path=raw_path, zarr_path=zarr_path, include_hash=include_hash)
        return zarr_path
    except BaseException as exc:
        _record_error(audit, task_id=task_id, dataset="oras5", error=exc)
        raise


def ingest_oras_year_variable(
    *,
    year: int,
    variable: str,
    raw_dir: Path,
    interim_dir: Path,
    zarr_root: Path,
    months: list[int] | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    include_hash: bool = False,
    delete_raw_after_zarr: bool = False,
    audit: AuditLog | None = None,
) -> Path:
    if variable not in ORAS5_VARIABLES:
        raise ValueError(f"No requested variable belongs to ORAS5: {variable}")

    audit = audit or AuditLog()
    selected_months = sorted(set(months or list(range(1, 13))))
    request = _oras_year_request(year, variable=variable, months=selected_months)
    slug = _safe_name(variable)
    task_id = f"oras5_{slug}_{year}"
    label = _oras_task_label(year=year, variable=variable)
    raw_path = raw_dir / str(year) / slug / f"{task_id}.zip"
    extract_dir = interim_dir / str(year) / slug / task_id
    zarr_path = zarr_root / "oras" / str(year) / slug / f"{task_id}_daily.zarr"

    if dry_run:
        print(f"DRY RUN annual variable ingest {task_id}: raw={raw_path} zarr={zarr_path}")
        print(json.dumps(request, indent=2))
        return zarr_path

    _task_record_start(audit, task_id=task_id, dataset="oras5_annual_variable", raw_path=raw_path, zarr_path=zarr_path, request=request)
    try:
        if zarr_path.exists() and not overwrite:
            dataset_summary(zarr_path, zarr=True)
            print(f"{label} - ja existe")
        else:
            print(f"{label} - baixando")
            retrieve_cds(
                dataset="reanalysis-oras5",
                request=request,
                output_path=raw_path,
                dry_run=False,
                overwrite=overwrite,
                quiet=True,
                label=label,
            )
            if extract_dir.exists() and overwrite:
                shutil.rmtree(extract_dir)
            start, end = _months_bounds(year, selected_months)
            print(f"{label} - convertendo zarr")
            zip_netcdf_to_daily_zarr(
                raw_path,
                extract_dir,
                zarr_path,
                variables=[variable],
                variable_aliases=ORAS5_VARIABLE_ALIASES,
                source_frequency="monthly",
                daily_start=start,
                daily_end=end,
                overwrite=overwrite,
                quiet=True,
            )
            print(f"{label} - ok")
        _record_ok(audit, task_id=task_id, dataset="oras5_annual_variable", raw_path=raw_path, zarr_path=zarr_path, include_hash=include_hash)
        if delete_raw_after_zarr:
            _delete_raw_cache(raw_path, audit, task_id=task_id, dataset="oras5_annual_variable", label=label)
        return zarr_path
    except BaseException as exc:
        _record_error(audit, task_id=task_id, dataset="oras5_annual_variable", error=exc)
        raise
