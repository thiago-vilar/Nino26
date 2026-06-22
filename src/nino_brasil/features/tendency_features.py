from __future__ import annotations

import numpy as np
import xarray as xr

from nino_brasil.features.spatial import coordinate_range_mask


EARTH_RADIUS_M = 6.371e6


def ssta_tendency(ssta: xr.DataArray, time_name: str = "time") -> xr.DataArray:
    """Return dSSTA/dt in K day-1."""
    if time_name not in ssta.coords and time_name not in ssta.dims:
        raise KeyError(f"ssta must contain {time_name!r}.")
    dt_days = ssta[time_name].diff(time_name) / np.timedelta64(1, "D")
    tendency = ssta.diff(time_name) / dt_days
    tendency = tendency.assign_coords({time_name: ssta[time_name].isel({time_name: slice(1, None)})})
    tendency.name = "ssta_tendency"
    tendency.attrs.update({"units": "K day-1", "physics": "sst_transport"})
    return tendency


def wind_stress_curl(
    tau_x: xr.DataArray,
    tau_y: xr.DataArray,
    lon_name: str = "lon",
    lat_name: str = "lat",
) -> xr.DataArray:
    """Return wind stress curl, d(tau_y)/dx - d(tau_x)/dy, in Pa m-1."""
    if lon_name not in tau_x.coords or lat_name not in tau_x.coords:
        raise KeyError(f"tau_x must contain {lon_name!r} and {lat_name!r} coordinates.")
    if lon_name not in tau_y.coords or lat_name not in tau_y.coords:
        raise KeyError(f"tau_y must contain {lon_name!r} and {lat_name!r} coordinates.")

    tau_x, tau_y = xr.align(tau_x, tau_y, join="exact")
    meters_per_degree_lat = np.deg2rad(1.0) * EARTH_RADIUS_M
    cos_lat = np.cos(np.deg2rad(tau_x[lat_name])).where(lambda value: abs(value) > 1.0e-6)
    meters_per_degree_lon = meters_per_degree_lat * cos_lat

    dtau_y_dx = tau_y.differentiate(lon_name) / meters_per_degree_lon
    dtau_x_dy = tau_x.differentiate(lat_name) / meters_per_degree_lat
    curl = dtau_y_dx - dtau_x_dy
    curl.name = "wind_stress_curl"
    curl.attrs.update({"units": "Pa m-1", "physics": "ekman_pumping"})
    return curl


def equatorial_zonal_stress(
    tau_x: xr.DataArray,
    lat_bounds: tuple[float, float] = (-5.0, 5.0),
    lon_bounds: tuple[float, float] = (120.0, 280.0),
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.DataArray:
    """Return equatorial mean zonal wind stress."""
    if lat_name not in tau_x.coords or lon_name not in tau_x.coords:
        raise KeyError(f"tau_x must contain {lat_name!r} and {lon_name!r} coordinates.")
    subset = tau_x.where(coordinate_range_mask(tau_x[lat_name], lat_bounds), drop=True)
    subset = subset.where(coordinate_range_mask(subset[lon_name], lon_bounds, circular=True), drop=True)
    stress = subset.mean([lat_name, lon_name], skipna=True)
    stress.name = "equatorial_zonal_stress"
    stress.attrs.update({"units": tau_x.attrs.get("units", "Pa"), "physics": "bjerknes_wind_forcing"})
    return stress
