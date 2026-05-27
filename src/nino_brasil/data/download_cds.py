from __future__ import annotations

import calendar
import json
from pathlib import Path

import cdsapi

from nino_brasil.data.credentials import DEFAULT_CDS_API_URL, load_local_env
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
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"DRY RUN cds {dataset} -> {output_path}")
        print(json.dumps(request, indent=2))
        return output_path

    client = _client()
    client.retrieve(dataset, request, str(output_path))
    print(f"downloaded: {output_path}")
    return output_path


def download_era5_single_month(
    *,
    year: int,
    month: int,
    region: str,
    raw_dir: Path,
    dry_run: bool = True,
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
    )


def download_era5_pressure_month(
    *,
    year: int,
    month: int,
    region: str,
    raw_dir: Path,
    dry_run: bool = True,
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
    )


def download_oras_month(
    *,
    year: int,
    month: int,
    raw_dir: Path,
    dry_run: bool = True,
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
    )
