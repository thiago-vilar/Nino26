from __future__ import annotations

import numpy as np
import xarray as xr


def d20_depth(temperature: xr.DataArray, depth_name: str = "depth") -> xr.DataArray:
    """Estimate D20, the depth of the 20 C isotherm, by nearest layer."""
    diff = abs(temperature - 20.0)
    idx = diff.argmin(depth_name)
    return temperature[depth_name].isel({depth_name: idx})


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
