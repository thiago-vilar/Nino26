from __future__ import annotations

import numpy as np
import xarray as xr

from nino_brasil.data.anomalies import daily_anomaly, dayofyear_climatology


def _normalize_lon_da(da: xr.DataArray, lon_name: str) -> xr.DataArray:
    lon = da[lon_name]
    return da.assign_coords({lon_name: lon % 360}).sortby(lon_name)


def _lat_slice(da: xr.DataArray, bounds: tuple[float, float], lat_name: str) -> slice:
    south, north = bounds
    lat = da[lat_name]
    return slice(south, north) if lat[0] < lat[-1] else slice(north, south)


def _select_lon_bounds(da: xr.DataArray, bounds: tuple[float, float], lon_name: str) -> xr.DataArray:
    west, east = (bounds[0] % 360.0, bounds[1] % 360.0)
    if west <= east:
        return da.sel({lon_name: slice(west, east)})
    left = da.sel({lon_name: slice(west, float(da[lon_name].max()))})
    right = da.sel({lon_name: slice(float(da[lon_name].min()), east)})
    # Em grades deslocadas (ex.: OISST 0.125-359.875) uma borda leste em 0E
    # gera segmento vazio; concatenar segmento vazio quebra o xr.concat.
    parts = [part for part in (left, right) if part.sizes.get(lon_name, 0) > 0]
    if not parts:
        raise ValueError(f"Longitude selection {bounds} produced no cells for {lon_name}.")
    if len(parts) == 1:
        return parts[0]
    return xr.concat(parts, dim=lon_name).sortby(lon_name)


def area_weighted_mean(
    da: xr.DataArray,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.DataArray:
    """Calculate a latitude-weighted spatial mean."""
    weights = np.cos(np.deg2rad(da[lat_name]))
    weights = weights / weights.mean()
    return da.weighted(weights).mean((lat_name, lon_name))


def sst_box_index(
    sst: xr.DataArray,
    *,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    name: str,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.DataArray:
    """Calculate an area-weighted SST index for any latitude/longitude box."""
    field = _normalize_lon_da(sst, lon_name)
    box = field.sel({lat_name: _lat_slice(field, lat_bounds, lat_name)})
    box = _select_lon_bounds(box, lon_bounds, lon_name)
    index = area_weighted_mean(box, lat_name=lat_name, lon_name=lon_name)
    return index.rename(name)


def nino34_sst_index(
    sst: xr.DataArray,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.DataArray:
    """Calculate the Niño 3.4 SST index over 5S-5N, 170W-120W."""
    return sst_box_index(
        sst,
        lat_bounds=(-5.0, 5.0),
        lon_bounds=(-170.0, -120.0),
        name="nino34_sst",
        lat_name=lat_name,
        lon_name=lon_name,
    )


def iod_sst_index(
    sst: xr.DataArray,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.DataArray:
    """Calculate the dipole mode index as western minus eastern Indian Ocean SST."""
    west = sst_box_index(
        sst,
        lat_bounds=(-10.0, 10.0),
        lon_bounds=(50.0, 70.0),
        name="iod_west_sst",
        lat_name=lat_name,
        lon_name=lon_name,
    )
    east = sst_box_index(
        sst,
        lat_bounds=(-10.0, 0.0),
        lon_bounds=(90.0, 110.0),
        name="iod_east_sst",
        lat_name=lat_name,
        lon_name=lon_name,
    )
    return (west - east).rename("iod_dmi_sst")


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
