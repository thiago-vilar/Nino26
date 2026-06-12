from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm import tqdm

from nino_brasil.data.audit import AuditLog, file_info


PMEL_ERDDAP = "https://data.pmel.noaa.gov/pmel/erddap/tabledap"
IFREMER_ERDDAP = "https://erddap.ifremer.fr/erddap/tabledap"

NINO34_LAT_MIN = -5.0
NINO34_LAT_MAX = 5.0
NINO34_LON_MIN_360 = 190.0
NINO34_LON_MAX_360 = 240.0
NINO34_LON_MIN_180 = -170.0
NINO34_LON_MAX_180 = -120.0

TAO_PRODUCTS = {
    "temperature": {
        "dataset": "pmelTaoDyT",
        "variables": ["array", "station", "wmo_platform_code", "longitude", "latitude", "time", "depth", "T_20", "QT_5020", "ST_6020"],
        "quality": ("QT_5020", 1, 3),
    },
    "salinity": {
        "dataset": "pmelTaoDyS",
        "variables": ["array", "station", "wmo_platform_code", "longitude", "latitude", "time", "depth", "S_41", "QS_5041", "SS_6041"],
        "quality": ("QS_5041", 1, 3),
    },
}

ARGO_VARIABLES = [
    "time",
    "latitude",
    "longitude",
    "pres",
    "temp",
    "psal",
    "pres_qc",
    "temp_qc",
    "psal_qc",
    "data_mode",
    "platform_number",
]


def download_tao_triton_year(
    *,
    year: int,
    raw_dir: Path,
    product: str,
    max_depth_m: float = 300.0,
    end_date: date | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    include_hash: bool = False,
    audit: AuditLog | None = None,
) -> Path:
    if product not in TAO_PRODUCTS:
        raise ValueError(f"Unknown TAO/TRITON product: {product}")
    spec = TAO_PRODUCTS[product]
    output_path = raw_dir / product / f"tao_triton_{product}_{year}.csv"
    start, end = _year_time_bounds(year, end_date=end_date)
    quality_name, quality_min, quality_max = spec["quality"]
    constraints = [
        ("time>=", start),
        ("time<=", end),
        ("latitude>=", NINO34_LAT_MIN),
        ("latitude<=", NINO34_LAT_MAX),
        ("longitude>=", NINO34_LON_MIN_360),
        ("longitude<=", NINO34_LON_MAX_360),
        ("depth<=", max_depth_m),
        (f"{quality_name}>=", quality_min),
        (f"{quality_name}<=", quality_max),
    ]
    url = _erddap_url(PMEL_ERDDAP, str(spec["dataset"]), spec["variables"], constraints)
    return _download_erddap_csv(
        url=url,
        output_path=output_path,
        task_id=f"tao_triton_{product}_{year}",
        dataset=f"tao_triton_{product}",
        dry_run=dry_run,
        overwrite=overwrite,
        include_hash=include_hash,
        audit=audit,
    )


def download_argo_year(
    *,
    year: int,
    raw_dir: Path,
    max_depth_m: float = 300.0,
    end_date: date | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    include_hash: bool = False,
    audit: AuditLog | None = None,
) -> Path:
    output_path = raw_dir / f"argo_nino34_{year}.csv"
    start, end = _year_time_bounds(year, end_date=end_date)
    constraints = [
        ("time>=", start),
        ("time<=", end),
        ("latitude>=", NINO34_LAT_MIN),
        ("latitude<=", NINO34_LAT_MAX),
        ("longitude>=", NINO34_LON_MIN_180),
        ("longitude<=", NINO34_LON_MAX_180),
        ("pres<=", max_depth_m),
    ]
    url = _erddap_url(IFREMER_ERDDAP, "ArgoFloats", ARGO_VARIABLES, constraints)
    return _download_erddap_csv(
        url=url,
        output_path=output_path,
        task_id=f"argo_nino34_{year}",
        dataset="argo_gdac_nino34",
        dry_run=dry_run,
        overwrite=overwrite,
        include_hash=include_hash,
        audit=audit,
    )


def _download_erddap_csv(
    *,
    url: str,
    output_path: Path,
    task_id: str,
    dataset: str,
    dry_run: bool,
    overwrite: bool,
    include_hash: bool,
    audit: AuditLog | None,
) -> Path:
    audit = audit or AuditLog()
    if dry_run:
        print(f"DRY RUN {dataset}: {url} -> {output_path}")
        return output_path
    if output_path.exists() and not overwrite:
        print(f"exists: {output_path}")
        return output_path

    audit.record(task_id=task_id, dataset=dataset, status="started", step="download", url=url, raw_path=str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    try:
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            if response.status_code == 404:
                audit.record(task_id=task_id, dataset=dataset, status="missing_source", step="download", url=url, raw_path=str(output_path))
                print(f"{task_id} - sem dados")
                return output_path
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0)) or None
            with temp_path.open("wb") as fh:
                with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=output_path.name) as bar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        bar.update(len(chunk))
        temp_path.replace(output_path)
        audit.record(task_id=task_id, dataset=dataset, status="ok", step="download", url=url, raw=file_info(output_path, include_hash=include_hash))
        print(f"downloaded: {output_path}")
        return output_path
    except BaseException as exc:
        if temp_path.exists():
            temp_path.unlink()
        audit.record(task_id=task_id, dataset=dataset, status="error", step="download", url=url, error_type=type(exc).__name__, error=str(exc))
        raise


def _erddap_url(base_url: str, dataset: str, variables: list[str], constraints: list[tuple[str, object]]) -> str:
    query = ",".join(variables)
    for name, value in constraints:
        query += "&" + name + _format_constraint_value(value)
    return f"{base_url}/{dataset}.csvp?{quote(query, safe=',&=<>:.-_')}"


def _format_constraint_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def _year_time_bounds(year: int, *, end_date: date | None = None) -> tuple[str, str]:
    limit = end_date or date.today()
    start_date = date(year, 1, 1)
    final_date = min(date(year, 12, 31), limit)
    if final_date < start_date:
        final_date = start_date
    return (
        datetime.combine(start_date, datetime.min.time()).strftime("%Y-%m-%dT00:00:00Z"),
        datetime.combine(final_date, datetime.max.time()).strftime("%Y-%m-%dT23:59:59Z"),
    )
