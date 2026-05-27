from __future__ import annotations

import xarray as xr


def rolling_accumulation(
    precipitation: xr.DataArray,
    window_days: int,
    time_name: str = "time",
) -> xr.DataArray:
    """Calculate rolling precipitation accumulation."""
    return precipitation.rolling({time_name: window_days}, min_periods=window_days).sum()


def local_percentile_threshold(
    precipitation: xr.DataArray,
    percentile: float,
    time_name: str = "time",
) -> xr.DataArray:
    """Calculate a local percentile threshold for each pixel."""
    return precipitation.quantile(percentile / 100.0, dim=time_name)


def event_mask(
    precipitation: xr.DataArray,
    percentile: float,
    kind: str,
    time_name: str = "time",
) -> xr.DataArray:
    """Create dry or wet event masks using local percentiles."""
    threshold = local_percentile_threshold(precipitation, percentile, time_name)
    if kind == "dry":
        return precipitation <= threshold
    if kind == "wet":
        return precipitation >= threshold
    raise ValueError("kind must be either 'dry' or 'wet'.")
