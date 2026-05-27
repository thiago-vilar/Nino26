from __future__ import annotations

from pathlib import Path

from nino_brasil.data.download_http import download_url


CHIRPS_BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf"


def chirps_url(year: int, resolution: str) -> str:
    if resolution not in {"p05", "p25"}:
        raise ValueError("resolution must be 'p05' or 'p25'.")
    return f"{CHIRPS_BASE_URL}/{resolution}/chirps-v2.0.{year}.days_{resolution}.nc"


def download_chirps_year(
    *,
    year: int,
    raw_dir: Path,
    resolution: str = "p25",
    overwrite: bool = False,
    dry_run: bool = False,
) -> Path:
    url = chirps_url(year, resolution)
    output_path = raw_dir / resolution / f"chirps-v2.0.{year}.days_{resolution}.nc"

    if dry_run:
        print(f"DRY RUN chirps {year}: {url} -> {output_path}")
        return output_path

    return download_url(url, output_path, overwrite=overwrite)
