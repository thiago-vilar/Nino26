from __future__ import annotations

import numpy as np
import xarray as xr

from nino_brasil.features.spatial import coordinate_range_mask


RHO0 = 1025.0
CP0 = 3990.0
EARTH_RADIUS_M = 6_371_000.0


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


def layer_ocean_heat_content(
    temperature: xr.DataArray,
    depth_min_m: float,
    depth_max_m: float,
    depth_name: str = "depth",
    rho: float = RHO0,
    cp: float = CP0,
) -> xr.DataArray:
    """Heat content for a bounded layer using clipped cell thicknesses.

    Depth coordinates are treated as layer centres.  Cell interfaces are
    reconstructed halfway between centres, clipped exactly at the requested
    bounds, and the shallowest interface is anchored at the surface.  A level
    below the requested lower boundary is therefore required for a defensible
    0-700 m integral.
    """
    if depth_max_m <= depth_min_m:
        raise ValueError("depth_max_m must be greater than depth_min_m.")
    depth = np.asarray(temperature[depth_name].values, dtype=float)
    if depth.ndim != 1 or depth.size < 2:
        raise ValueError(f"Insufficient depth levels in {depth_min_m}-{depth_max_m} m layer.")
    order = np.argsort(depth)
    levels = depth[order]
    if np.any(np.diff(levels) <= 0):
        raise ValueError("Depth coordinate must be strictly increasing.")
    interfaces = np.empty(levels.size + 1, dtype=float)
    interfaces[1:-1] = 0.5 * (levels[:-1] + levels[1:])
    interfaces[0] = max(0.0, levels[0] - 0.5 * (levels[1] - levels[0]))
    interfaces[-1] = levels[-1] + 0.5 * (levels[-1] - levels[-2])
    if depth_min_m < interfaces[0] - 1e-6 or depth_max_m > interfaces[-1] + 1e-6:
        raise ValueError(
            f"Depth levels cover cell interfaces {interfaces[0]:.3f}-{interfaces[-1]:.3f} m; "
            f"cannot integrate exactly over {depth_min_m}-{depth_max_m} m."
        )
    overlap = np.maximum(
        0.0,
        np.minimum(interfaces[1:], depth_max_m) - np.maximum(interfaces[:-1], depth_min_m),
    )
    weights = xr.DataArray(
        overlap.astype(np.float32),
        coords={depth_name: temperature[depth_name].isel({depth_name: order})},
        dims=(depth_name,),
        name="effective_layer_thickness",
    )
    ordered = temperature.isel({depth_name: order})
    result = (rho * cp * ordered * weights).sum(depth_name, skipna=True, min_count=1)
    result.attrs.update(
        {
            "units": "J m-2",
            "depth_min_m": float(depth_min_m),
            "depth_max_m": float(depth_max_m),
            "method": "rho0_cp0_cell_center_integral_with_clipped_interfaces",
            "rho_kg_m3": float(rho),
            "cp_j_kg_k": float(cp),
        }
    )
    return result


def warm_water_volume(
    d20: xr.DataArray,
    lat_bounds: tuple[float, float] = (-5.0, 5.0),
    lon_bounds: tuple[float, float] = (120.0, 280.0),
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.DataArray:
    """Return the equatorial warm water volume proxy from D20."""
    if lat_name not in d20.coords or lon_name not in d20.coords:
        raise KeyError(f"d20 must contain {lat_name!r} and {lon_name!r} coordinates.")
    subset = d20.where(coordinate_range_mask(d20[lat_name], lat_bounds), drop=True)
    subset = subset.where(coordinate_range_mask(subset[lon_name], lon_bounds, circular=True), drop=True)
    wwv = subset.mean([lat_name, lon_name], skipna=True)
    wwv.name = "wwv"
    wwv.attrs.update({"units": "m", "physics": "recharge_oscillator_wwv"})
    return wwv


def warm_water_volume_m3(
    d20: xr.DataArray,
    lat_bounds: tuple[float, float] = (-5.0, 5.0),
    lon_bounds: tuple[float, float] = (120.0, 280.0),
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.DataArray:
    """Integrate D20 over spherical grid-cell area to obtain WWV in m3."""
    if lat_name not in d20.coords or lon_name not in d20.coords:
        raise KeyError(f"d20 must contain {lat_name!r} and {lon_name!r} coordinates.")
    subset = d20.where(coordinate_range_mask(d20[lat_name], lat_bounds), drop=True)
    subset = subset.where(coordinate_range_mask(subset[lon_name], lon_bounds, circular=True), drop=True)
    lat = np.asarray(subset[lat_name].values, dtype=float)
    lon = np.asarray(subset[lon_name].values, dtype=float)
    if lat.size < 2 or lon.size < 2:
        raise ValueError("WWV integration needs at least two latitude and longitude cells.")
    dlat = float(np.median(np.abs(np.diff(lat))))
    lon_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lon)))
    dlon = float(np.median(np.abs(np.diff(lon_unwrapped))))
    area = (
        EARTH_RADIUS_M**2
        * np.deg2rad(dlat)
        * np.deg2rad(dlon)
        * np.cos(np.deg2rad(subset[lat_name]))
    )
    volume = (subset * area).sum([lat_name, lon_name], skipna=True, min_count=1)
    volume.name = "wwv_m3"
    volume.attrs.update({"units": "m3", "physics": "recharge_oscillator_wwv", "integration": "D20_cell_area"})
    return volume


def ohc_tendency(ohc: xr.DataArray, time_name: str = "time") -> xr.DataArray:
    """Return dOHC/dt in W m-2 from an OHC series in J m-2."""
    if time_name not in ohc.coords and time_name not in ohc.dims:
        raise KeyError(f"ohc must contain {time_name!r}.")
    dt_days = ohc[time_name].diff(time_name) / np.timedelta64(1, "D")
    tendency = (ohc.diff(time_name) / dt_days) / 86400.0
    tendency = tendency.assign_coords({time_name: ohc[time_name].isel({time_name: slice(1, None)})})
    tendency.name = f"{ohc.name or 'ohc'}_tendency"
    tendency.attrs.update({"units": "W m-2", "physics": "heat_budget_lhs"})
    return tendency
