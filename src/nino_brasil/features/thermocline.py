from __future__ import annotations

import numpy as np
import xarray as xr

from nino_brasil.features.spatial import coordinate_range_mask


def d20_depth(temperature: xr.DataArray, depth_name: str = "depth") -> xr.DataArray:
    """Estimate D20 by linear interpolation across the first 20 C crossing.

    A nearest-level estimate is grid-resolution dependent and can introduce
    artificial steps.  Profiles that never cross 20 C return NaN rather than
    silently selecting an unrelated depth.
    """

    def _profile_d20(profile: np.ndarray, depth: np.ndarray) -> np.float32:
        valid = np.isfinite(profile) & np.isfinite(depth)
        values = np.asarray(profile[valid], dtype=float)
        levels = np.asarray(depth[valid], dtype=float)
        if values.size < 2:
            return np.float32(np.nan)
        order = np.argsort(levels)
        values = values[order]
        levels = levels[order]
        crossing = np.where((values[:-1] >= 20.0) & (values[1:] <= 20.0))[0]
        if crossing.size == 0:
            return np.float32(np.nan)
        index = int(crossing[0])
        t0, t1 = values[index], values[index + 1]
        z0, z1 = levels[index], levels[index + 1]
        if t1 == t0:
            return np.float32((z0 + z1) / 2.0)
        return np.float32(z0 + (20.0 - t0) * (z1 - z0) / (t1 - t0))

    d20 = xr.apply_ufunc(
        _profile_d20,
        temperature,
        temperature[depth_name],
        input_core_dims=[[depth_name], [depth_name]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
    )
    d20.name = "d20"
    d20.attrs.update({"long_name": "depth of the 20 degree Celsius isotherm", "units": "m"})
    return d20


def max_gradient_thermocline_depth(
    temperature: xr.DataArray,
    depth_name: str = "depth",
) -> xr.DataArray:
    """Estimate thermocline depth as maximum vertical temperature gradient."""
    grad = temperature.differentiate(depth_name)
    idx = abs(grad).argmax(depth_name)
    return temperature[depth_name].isel({depth_name: idx})


def mixed_layer_depth_threshold(
    temperature: xr.DataArray,
    threshold_c: float = 0.5,
    depth_name: str = "depth",
) -> xr.DataArray:
    """Estimate mixed layer depth using a surface temperature threshold."""
    surface = temperature.isel({depth_name: 0})
    diff = abs(temperature - surface)
    masked = diff.where(diff >= threshold_c)
    idx = masked.fillna(np.inf).argmin(depth_name)
    return temperature[depth_name].isel({depth_name: idx})


def thermocline_tilt(
    d20: xr.DataArray,
    lat_bounds: tuple[float, float] = (-5.0, 5.0),
    lon_east: tuple[float, float] = (220.0, 270.0),
    lon_west: tuple[float, float] = (140.0, 180.0),
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.DataArray:
    """Return east-minus-west D20 thermocline tilt for the equatorial Pacific."""
    if lat_name not in d20.coords or lon_name not in d20.coords:
        raise KeyError(f"d20 must contain {lat_name!r} and {lon_name!r} coordinates.")
    equatorial = d20.where(coordinate_range_mask(d20[lat_name], lat_bounds), drop=True)
    east = equatorial.where(coordinate_range_mask(equatorial[lon_name], lon_east, circular=True), drop=True).mean(
        [lat_name, lon_name],
        skipna=True,
    )
    west = equatorial.where(coordinate_range_mask(equatorial[lon_name], lon_west, circular=True), drop=True).mean(
        [lat_name, lon_name],
        skipna=True,
    )
    tilt = east - west
    tilt.name = "thermocline_tilt"
    tilt.attrs.update({"units": "m", "physics": "bjerknes_tilt"})
    return tilt


def thermocline_tilt_slope(
    d20: xr.DataArray,
    lat_bounds: tuple[float, float] = (-5.0, 5.0),
    lon_bounds: tuple[float, float] = (140.0, 270.0),
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.DataArray:
    """Return the mean longitudinal D20 gradient in m per degree east."""
    if lat_name not in d20.coords or lon_name not in d20.coords:
        raise KeyError(f"d20 must contain {lat_name!r} and {lon_name!r} coordinates.")
    equatorial = d20.where(coordinate_range_mask(d20[lat_name], lat_bounds), drop=True)
    equatorial = equatorial.where(
        coordinate_range_mask(equatorial[lon_name], lon_bounds, circular=True),
        drop=True,
    )
    zonal = equatorial.mean(lat_name, skipna=True).sortby(lon_name)
    slope = zonal.differentiate(lon_name).mean(lon_name, skipna=True)
    slope.name = "thermocline_tilt_slope"
    slope.attrs.update({"units": "m degree_east-1", "physics": "bjerknes_tilt"})
    return slope


def d20_tendency(d20: xr.DataArray, time_name: str = "time") -> xr.DataArray:
    """Return dD20/dt in m day-1."""
    if time_name not in d20.coords and time_name not in d20.dims:
        raise KeyError(f"d20 must contain {time_name!r}.")
    dt_days = d20[time_name].diff(time_name) / np.timedelta64(1, "D")
    tendency = d20.diff(time_name) / dt_days
    tendency = tendency.assign_coords({time_name: d20[time_name].isel({time_name: slice(1, None)})})
    tendency.name = "d20_tendency"
    tendency.attrs.update({"units": "m day-1", "physics": "recharge_lhs"})
    return tendency
