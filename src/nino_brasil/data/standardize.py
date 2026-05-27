from __future__ import annotations

import xarray as xr


def normalize_longitudes(ds: xr.Dataset, lon_name: str = "lon", convention: str = "0_360") -> xr.Dataset:
    """Normalize longitude coordinate to either 0..360 or -180..180."""
    if lon_name not in ds.coords:
        return ds

    lon = ds[lon_name]
    if convention == "0_360":
        new_lon = lon % 360
    elif convention == "-180_180":
        new_lon = ((lon + 180) % 360) - 180
    else:
        raise ValueError(f"Unsupported longitude convention: {convention}")

    return ds.assign_coords({lon_name: new_lon}).sortby(lon_name)


def subset_latlon(
    ds: xr.Dataset,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.Dataset:
    """Subset a dataset by latitude and longitude bounds."""
    if lat_name not in ds.coords or lon_name not in ds.coords:
        raise KeyError(f"Dataset must contain {lat_name!r} and {lon_name!r} coordinates.")

    lat = ds[lat_name]
    lat_slice = slice(lat_min, lat_max) if lat[0] < lat[-1] else slice(lat_max, lat_min)
    return ds.sel({lat_name: lat_slice, lon_name: slice(lon_min, lon_max)})


def daily_mean(ds: xr.Dataset, time_name: str = "time") -> xr.Dataset:
    """Aggregate sub-daily data to daily means."""
    if time_name not in ds.coords:
        raise KeyError(f"Dataset must contain {time_name!r}.")
    return ds.resample({time_name: "1D"}).mean()


def daily_sum(ds: xr.Dataset, time_name: str = "time") -> xr.Dataset:
    """Aggregate sub-daily data to daily sums."""
    if time_name not in ds.coords:
        raise KeyError(f"Dataset must contain {time_name!r}.")
    return ds.resample({time_name: "1D"}).sum()
