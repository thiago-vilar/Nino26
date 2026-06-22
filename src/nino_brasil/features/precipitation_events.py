from __future__ import annotations

import xarray as xr


def _training_sample(
    precipitation: xr.DataArray,
    train_times: xr.DataArray | list | None,
    time_name: str,
) -> xr.DataArray:
    return precipitation.sel({time_name: train_times}) if train_times is not None else precipitation


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
    sample = _training_sample(precipitation, train_times, time_name)
    if sample.chunks is not None:
        # xarray quantile requires a single chunk along the reduced dimension.
        sample = sample.chunk({time_name: -1})
    return sample.quantile(percentile / 100.0, dim=time_name)


def local_percentile_thresholds(
    precipitation: xr.DataArray,
    percentiles: list[float] | tuple[float, ...] | set[float],
    train_times: xr.DataArray | list | None = None,
    time_name: str = "time",
) -> xr.DataArray:
    """Calculate multiple local percentile thresholds in one training pass."""
    if not percentiles:
        raise ValueError("percentiles must contain at least one value.")
    sample = _training_sample(precipitation, train_times, time_name)
    if sample.chunks is not None:
        # One rechunk and one quantile pass for all requested thresholds.
        sample = sample.chunk({time_name: -1})
    quantiles = sorted({float(percentile) / 100.0 for percentile in percentiles})
    return sample.quantile(quantiles, dim=time_name)


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
