from __future__ import annotations

from pathlib import Path

from nino_brasil.data.download_http import download_url


OISST_BASE_URL = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres"


def oisst_url(year: int) -> str:
    return f"{OISST_BASE_URL}/sst.day.mean.{year}.nc"


def download_oisst_year(
    *,
    year: int,
    raw_dir: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Path:
    url = oisst_url(year)
    output_path = raw_dir / f"sst.day.mean.{year}.nc"

    if dry_run:
        print(f"DRY RUN oisst {year}: {url} -> {output_path}")
        return output_path

    return download_url(url, output_path, overwrite=overwrite)
