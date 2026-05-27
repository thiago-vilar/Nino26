from __future__ import annotations

import xarray as xr


RHO0 = 1025.0
CP0 = 3990.0


def layer_mean_temperature(
    temperature: xr.DataArray,
    depth_max_m: float,
    depth_name: str = "depth",
) -> xr.DataArray:
    """Calculate mean temperature from surface to depth_max_m."""
    layer = temperature.sel({depth_name: slice(0, depth_max_m)})
    return layer.mean(depth_name)


def ocean_heat_content(
    temperature: xr.DataArray,
    depth_max_m: float,
    depth_name: str = "depth",
    rho: float = RHO0,
    cp: float = CP0,
) -> xr.DataArray:
    """Approximate ocean heat content from surface to depth_max_m."""
    layer = temperature.sel({depth_name: slice(0, depth_max_m)})
    return (rho * cp * layer).integrate(depth_name)
