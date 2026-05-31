from __future__ import annotations

import pandas as pd
import xarray as xr


def align_predictor_target(
    predictors: xr.Dataset,
    target: xr.DataArray,
    lag_days: int,
    time_name: str = "time",
) -> tuple[xr.Dataset, xr.DataArray]:
    """Align X(t) with Y(t + lag)."""
    if lag_days < 0:
        raise ValueError("lag_days must be non-negative.")

    shifted_target = target.shift({time_name: -lag_days})

    # Pacific predictors and Brazil targets live on different spatial grids, so
    # alignment must happen on time only.
    common_time = predictors.indexes[time_name].intersection(shifted_target.indexes[time_name])
    aligned_x = predictors.sel({time_name: common_time})
    aligned_y = shifted_target.sel({time_name: common_time})
    target_time = pd.DatetimeIndex(common_time) + pd.to_timedelta(lag_days, unit="D")
    aligned_x = aligned_x.assign_coords(target_time=(time_name, target_time))
    aligned_y = aligned_y.assign_coords(target_time=(time_name, target_time))

    spatial_dims = [dim for dim in aligned_y.dims if dim != time_name]
    valid_time = aligned_y.notnull().any(dim=spatial_dims) if spatial_dims else aligned_y.notnull()
    valid_time = valid_time.fillna(False)

    return aligned_x.sel({time_name: aligned_y[time_name][valid_time]}), aligned_y.sel(
        {time_name: aligned_y[time_name][valid_time]}
    )


def build_lagged_targets(
    predictors: xr.Dataset,
    target: xr.DataArray,
    lag_days: list[int],
    time_name: str = "time",
) -> dict[int, tuple[xr.Dataset, xr.DataArray]]:
    """Build aligned predictor-target pairs for many lags."""
    return {
        lag: align_predictor_target(predictors, target, lag, time_name)
        for lag in lag_days
    }
