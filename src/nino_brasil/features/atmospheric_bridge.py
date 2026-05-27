from __future__ import annotations

import xarray as xr


def wind_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Calculate wind speed from zonal and meridional components."""
    return (u**2 + v**2) ** 0.5


def vertically_named(ds: xr.Dataset, variable: str, level_hpa: int) -> xr.DataArray:
    """Select a pressure level and rename variable as variable+level."""
    if "level" in ds.coords:
        da = ds[variable].sel(level=level_hpa)
    elif "pressure_level" in ds.coords:
        da = ds[variable].sel(pressure_level=level_hpa)
    else:
        raise KeyError("Dataset must contain 'level' or 'pressure_level'.")
    return da.rename(f"{variable}{level_hpa}")
