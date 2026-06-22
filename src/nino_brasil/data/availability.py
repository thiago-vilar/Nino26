from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd


DEFAULT_SOURCE_LATENCY_DAYS = {
    "chirps": 45,
    "noaa_oisst": 7,
    "era5": 7,
    "glorys12": 30,
    "glorys12_operational": 1,
    "noaa_ufs": 0,
    "oras5": 15,
    "noaa_wod_ctd": 365,
    "tao_triton": 3,
    "argo_gdac": 3,
}


def source_latency_days(source: str, config: dict[str, Any] | None = None) -> int:
    """Return source latency in days from config, falling back to conservative defaults."""
    configured = (config or {}).get("source_latency_days", {})
    return int(configured.get(source, DEFAULT_SOURCE_LATENCY_DAYS.get(source, 0)))


def available_end_date(
    source: str,
    config: dict[str, Any] | None = None,
    as_of: date | pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Estimate the latest usable date for a source after latency."""
    today = pd.Timestamp(as_of if as_of is not None else date.today()).normalize()
    return today - pd.Timedelta(days=source_latency_days(source, config))


def project_start_date(config: dict[str, Any]) -> pd.Timestamp:
    """Return the fixed historical start date."""
    return pd.Timestamp(config["period"]["start"])


def requested_end_date(
    config: dict[str, Any],
    source: str,
    requested_year: int | None = None,
) -> pd.Timestamp:
    """Resolve dynamic period end without extending beyond source availability."""
    source_end = available_end_date(source, config)
    configured_end = config.get("period", {}).get("end")
    if configured_end and str(configured_end).lower() not in {"latest_available", "dynamic"}:
        source_end = min(source_end, pd.Timestamp(configured_end))
    if requested_year is not None:
        source_end = min(source_end, pd.Timestamp(f"{requested_year}-12-31"))
    return source_end


def iter_available_years(
    start_year: int,
    end_year: int | None,
    source: str,
    config: dict[str, Any],
) -> range:
    """Iterate project years capped at the latest available date for a source."""
    project_start = project_start_date(config).year
    if start_year < project_start:
        raise ValueError(f"project period starts at {project_start}; received {start_year}.")
    resolved_end = requested_end_date(config, source, end_year)
    final_year = int(resolved_end.year)
    if final_year < start_year:
        raise ValueError(f"No available {source} data for {start_year} after source latency.")
    return range(start_year, final_year + 1)


def record_source_latency(audit: Any, source: str, config: dict[str, Any]) -> None:
    """Record source latency and resolved end date in the audit ledger."""
    audit.record(
        task_id=f"source_latency_{source}",
        dataset=source,
        status="metadata",
        latency_days=source_latency_days(source, config),
        available_through=str(available_end_date(source, config).date()),
        policy="period_end_resolved_from_latest_available_source_data",
    )
