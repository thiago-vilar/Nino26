from __future__ import annotations

import numpy as np
import xarray as xr

from nino_brasil.data.anomalies import daily_anomaly, dayofyear_climatology


def _normalize_lon_da(da: xr.DataArray, lon_name: str) -> xr.DataArray:
    lon = da[lon_name]
    return da.assign_coords({lon_name: lon % 360}).sortby(lon_name)


def area_weighted_mean(
    da: xr.DataArray,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.DataArray:
    """Calculate a latitude-weighted spatial mean."""
    weights = np.cos(np.deg2rad(da[lat_name]))
    weights = weights / weights.mean()
    return da.weighted(weights).mean((lat_name, lon_name))


def nino34_sst_index(
    sst: xr.DataArray,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.DataArray:
    """Calculate the Niño 3.4 SST index over 5S-5N, 170W-120W."""
    field = _normalize_lon_da(sst, lon_name)
    lat = field[lat_name]
    lat_slice = slice(-5.0, 5.0) if lat[0] < lat[-1] else slice(5.0, -5.0)
    nino = field.sel({lat_name: lat_slice, lon_name: slice(190.0, 240.0)})
    index = area_weighted_mean(nino, lat_name=lat_name, lon_name=lon_name)
    return index.rename("nino34_sst")


def nino34_ssta_index(
    sst: xr.DataArray,
    climatology: xr.DataArray | None = None,
    window_days: int = 15,
    time_name: str = "time",
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.DataArray:
    """Calculate daily Niño 3.4 SST anomaly from OISST or another SST field."""
    index = nino34_sst_index(sst, lat_name=lat_name, lon_name=lon_name)
    clim = climatology if climatology is not None else dayofyear_climatology(index, window_days, time_name)
    return daily_anomaly(index, climatology=clim, window_days=window_days, time_name=time_name).rename("nino34_ssta")
