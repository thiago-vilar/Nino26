from __future__ import annotations

import xarray as xr


def _smooth_dayofyear_stat(stat: xr.DataArray, window_days: int) -> xr.DataArray:
    if window_days < 1 or window_days % 2 == 0:
        raise ValueError("window_days must be a positive odd integer.")
    if window_days == 1:
        return stat

    half = window_days // 2
    return (
        stat.rolling(dayofyear=window_days, center=True, min_periods=1)
        .mean()
        .pad(dayofyear=(half, half), mode="wrap")
        .rolling(dayofyear=window_days, center=True, min_periods=1)
        .mean()
        .isel(dayofyear=slice(half, -half))
    )


def dayofyear_climatology(
    da: xr.DataArray,
    window_days: int = 15,
    time_name: str = "time",
) -> xr.DataArray:
    """Calculate smoothed day-of-year climatology from the supplied sample."""
    clim = da.groupby(f"{time_name}.dayofyear").mean(time_name)
    return _smooth_dayofyear_stat(clim, window_days)


def dayofyear_std(
    da: xr.DataArray,
    window_days: int = 15,
    time_name: str = "time",
) -> xr.DataArray:
    """Calculate smoothed day-of-year standard deviation from the supplied sample."""
    std = da.groupby(f"{time_name}.dayofyear").std(time_name)
    return _smooth_dayofyear_stat(std, window_days)


def fit_daily_climatology(
    da: xr.DataArray,
    train_times: xr.DataArray | list | None = None,
    window_days: int = 15,
    time_name: str = "time",
) -> xr.DataArray:
    """Fit a daily climatology on a training block only."""
    train = da.sel({time_name: train_times}) if train_times is not None else da
    return dayofyear_climatology(train, window_days=window_days, time_name=time_name)


def fit_daily_standardization(
    da: xr.DataArray,
    train_times: xr.DataArray | list | None = None,
    window_days: int = 15,
    time_name: str = "time",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Fit smoothed daily mean and standard deviation on a training block only."""
    train = da.sel({time_name: train_times}) if train_times is not None else da
    mean = dayofyear_climatology(train, window_days=window_days, time_name=time_name)
    std = dayofyear_std(train, window_days=window_days, time_name=time_name)
    return mean, std


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
    climatology: xr.DataArray | None = None,
    std: xr.DataArray | None = None,
    window_days: int = 15,
    time_name: str = "time",
) -> xr.DataArray:
    """Calculate standardized anomaly using the same smoothed climatology basis."""
    clim = climatology if climatology is not None else dayofyear_climatology(da, window_days, time_name)
    scale = std if std is not None else dayofyear_std(da, window_days, time_name)
    return daily_anomaly(da, clim, window_days, time_name).groupby(f"{time_name}.dayofyear") / scale
