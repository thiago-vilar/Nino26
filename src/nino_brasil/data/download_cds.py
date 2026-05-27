from __future__ import annotations

import calendar
import json
import shutil
from pathlib import Path

import cdsapi

from nino_brasil.data.audit import AuditLog, dataset_summary, file_info
from nino_brasil.data.credentials import DEFAULT_CDS_API_URL, load_local_env
from nino_brasil.data.zarr_store import netcdf_to_zarr, zip_netcdf_to_zarr
import os


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

ATMOSPHERE_AREAS = {
    "west_pacific": [30, 120, -35, 180],
    "east_pacific_brazil": [30, -180, -35, -30],
}


def _days(year: int, month: int) -> list[str]:
    last_day = calendar.monthrange(year, month)[1]
    return [f"{day:02d}" for day in range(1, last_day + 1)]


def _month(month: int) -> str:
    return f"{month:02d}"


def _client() -> cdsapi.Client:
    load_local_env()
    key = os.environ.get("CDS_API_KEY")
    url = os.environ.get("CDS_API_URL", DEFAULT_CDS_API_URL)
    if not key:
        raise RuntimeError("CDS_API_KEY is not set.")
    return cdsapi.Client(url=url, key=key)


def retrieve_cds(
    *,
    dataset: str,
    request: dict[str, object],
    output_path: Path,
    dry_run: bool = True,
    overwrite: bool = False,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"DRY RUN cds {dataset} -> {output_path}")
        print(json.dumps(request, indent=2))
        return output_path

    if output_path.exists() and not overwrite:
        print(f"exists: {output_path}")
        return output_path

    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    client = _client()
    try:
        client.retrieve(dataset, request, str(temp_path))
        temp_path.replace(output_path)
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
        raise
    print(f"downloaded: {output_path}")
    return output_path


def download_era5_single_month(
    *,
    year: int,
    month: int,
    region: str,
    raw_dir: Path,
    dry_run: bool = True,
    overwrite: bool = False,
) -> Path:
    area = ATMOSPHERE_AREAS[region]
    request = {
        "product_type": "reanalysis",
        "variable": ERA5_SINGLE_VARIABLES,
        "year": str(year),
        "month": _month(month),
        "day": _days(year, month),
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }
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
    dry_run: bool = True,
    overwrite: bool = False,
) -> Path:
    area = ATMOSPHERE_AREAS[region]
    request = {
        "product_type": "reanalysis",
        "variable": ERA5_PRESSURE_VARIABLES,
        "pressure_level": ERA5_PRESSURE_LEVELS,
        "year": str(year),
        "month": _month(month),
        "day": _days(year, month),
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }
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
    dry_run: bool = True,
    overwrite: bool = False,
) -> Path:
    product_type = "consolidated" if year <= 2014 else "operational"
    request = {
        "product_type": product_type,
        "vertical_resolution": "all_levels",
        "variable": [
            "potential_temperature",
            "salinity",
            "sea_surface_temperature",
            "ocean_heat_content_for_the_upper_300m",
            "ocean_heat_content_for_the_upper_700m",
        ],
        "year": str(year),
        "month": _month(month),
        "data_format": "netcdf",
    }
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


def _era5_single_request(year: int, month: int, region: str) -> dict[str, object]:
    return {
        "product_type": "reanalysis",
        "variable": ERA5_SINGLE_VARIABLES,
        "year": str(year),
        "month": _month(month),
        "day": _days(year, month),
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": ATMOSPHERE_AREAS[region],
    }


def _era5_pressure_request(year: int, month: int, region: str) -> dict[str, object]:
    return {
        "product_type": "reanalysis",
        "variable": ERA5_PRESSURE_VARIABLES,
        "pressure_level": ERA5_PRESSURE_LEVELS,
        "year": str(year),
        "month": _month(month),
        "day": _days(year, month),
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": ATMOSPHERE_AREAS[region],
    }


def _oras_request(year: int, month: int) -> dict[str, object]:
    product_type = "consolidated" if year <= 2014 else "operational"
    return {
        "product_type": product_type,
        "vertical_resolution": "all_levels",
        "variable": [
            "potential_temperature",
            "salinity",
            "sea_surface_temperature",
            "ocean_heat_content_for_the_upper_300m",
            "ocean_heat_content_for_the_upper_700m",
        ],
        "year": str(year),
        "month": _month(month),
        "data_format": "netcdf",
    }


def ingest_era5_single_month(
    *,
    year: int,
    month: int,
    region: str,
    raw_dir: Path,
    zarr_root: Path,
    dry_run: bool = True,
    overwrite: bool = False,
    include_hash: bool = False,
    audit: AuditLog | None = None,
) -> Path:
    audit = audit or AuditLog()
    task_id = f"era5_single_{region}_{year}{month:02d}"
    request = _era5_single_request(year, month, region)
    raw_path = raw_dir / "single_levels" / str(year) / f"{task_id}.nc"
    zarr_path = zarr_root / "era5" / "single_levels" / str(year) / f"{task_id}.zarr"

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
            netcdf_to_zarr(raw_path, zarr_path, overwrite=overwrite)
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
    dry_run: bool = True,
    overwrite: bool = False,
    include_hash: bool = False,
    audit: AuditLog | None = None,
) -> Path:
    audit = audit or AuditLog()
    task_id = f"era5_pressure_{region}_{year}{month:02d}"
    request = _era5_pressure_request(year, month, region)
    raw_path = raw_dir / "pressure_levels" / str(year) / f"{task_id}.nc"
    zarr_path = zarr_root / "era5" / "pressure_levels" / str(year) / f"{task_id}.zarr"

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
            netcdf_to_zarr(raw_path, zarr_path, overwrite=overwrite)
        _record_ok(audit, task_id=task_id, dataset="era5_pressure", raw_path=raw_path, zarr_path=zarr_path, include_hash=include_hash)
        return zarr_path
    except BaseException as exc:
        _record_error(audit, task_id=task_id, dataset="era5_pressure", error=exc)
        raise


def ingest_oras_month(
    *,
    year: int,
    month: int,
    raw_dir: Path,
    interim_dir: Path,
    zarr_root: Path,
    dry_run: bool = True,
    overwrite: bool = False,
    include_hash: bool = False,
    audit: AuditLog | None = None,
) -> Path:
    audit = audit or AuditLog()
    task_id = f"oras5_{year}{month:02d}"
    request = _oras_request(year, month)
    raw_path = raw_dir / str(year) / f"{task_id}.zip"
    extract_dir = interim_dir / str(year) / task_id
    zarr_path = zarr_root / "oras" / str(year) / f"{task_id}.zarr"

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
            zip_netcdf_to_zarr(raw_path, extract_dir, zarr_path, overwrite=overwrite)
        _record_ok(audit, task_id=task_id, dataset="oras5", raw_path=raw_path, zarr_path=zarr_path, include_hash=include_hash)
        return zarr_path
    except BaseException as exc:
        _record_error(audit, task_id=task_id, dataset="oras5", error=exc)
        raise
