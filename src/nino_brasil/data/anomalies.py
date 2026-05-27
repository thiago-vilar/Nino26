from __future__ import annotations

import xarray as xr


def dayofyear_climatology(
    da: xr.DataArray,
    window_days: int = 15,
    time_name: str = "time",
) -> xr.DataArray:
    """Calculate smoothed day-of-year climatology."""
    if window_days < 1 or window_days % 2 == 0:
        raise ValueError("window_days must be a positive odd integer.")

    clim = da.groupby(f"{time_name}.dayofyear").mean(time_name)
    half = window_days // 2
    return clim.rolling(dayofyear=window_days, center=True, min_periods=1).mean().pad(
        dayofyear=(half, half),
        mode="wrap",
    ).rolling(dayofyear=window_days, center=True, min_periods=1).mean().isel(
        dayofyear=slice(half, -half)
    )


def daily_anomaly(
    da: xr.DataArray,
    climatology: xr.DataArray | None = None,
    window_days: int = 15,
    time_name: str = "time",
) -> xr.DataArray:
    """Calculate daily anomaly against day-of-year climatology."""
    clim = climatology if climatology is not None else dayofyear_climatology(da, window_days, time_name)
    return da.groupby(f"{time_name}.dayofyear") - clim


def standardized_anomaly(
    da: xr.DataArray,
    time_name: str = "time",
) -> xr.DataArray:
    """Calculate standardized anomaly by day of year."""
    grouped = da.groupby(f"{time_name}.dayofyear")
    mean = grouped.mean(time_name)
    std = grouped.std(time_name)
    return (grouped - mean) / std
