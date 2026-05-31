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
    train_times: xr.DataArray | list | None = None,
    time_name: str = "time",
) -> xr.DataArray:
    """Calculate a local percentile threshold, optionally on a training block."""
    sample = precipitation.sel({time_name: train_times}) if train_times is not None else precipitation
    return sample.quantile(percentile / 100.0, dim=time_name)


def event_mask(
    precipitation: xr.DataArray,
    percentile: float,
    kind: str,
    threshold: xr.DataArray | None = None,
    train_times: xr.DataArray | list | None = None,
    time_name: str = "time",
) -> xr.DataArray:
    """Create dry or wet event masks using local percentiles."""
    limit = threshold
    if limit is None:
        limit = local_percentile_threshold(precipitation, percentile, train_times, time_name)
    if kind == "dry":
        return precipitation <= limit
    if kind == "wet":
        return precipitation >= limit
    raise ValueError("kind must be either 'dry' or 'wet'.")
