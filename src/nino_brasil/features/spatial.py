from __future__ import annotations

import xarray as xr


def coordinate_range_mask(
    coord: xr.DataArray,
    bounds: tuple[float, float],
    *,
    circular: bool = False,
) -> xr.DataArray:
    """Build a mask for linear or circular coordinate ranges."""
    lower, upper = sorted(bounds)
    if not circular:
        return (coord >= lower) & (coord <= upper)

    coord_min = float(coord.min())
    coord_max = float(coord.max())
    raw_lower, raw_upper = bounds
    if coord_min < 0.0 and (raw_lower > 180.0 or raw_upper > 180.0):
        lower = ((raw_lower + 180.0) % 360.0) - 180.0
        upper = ((raw_upper + 180.0) % 360.0) - 180.0
    elif coord_max > 180.0 and (raw_lower < 0.0 or raw_upper < 0.0):
        lower = raw_lower % 360.0
        upper = raw_upper % 360.0
    else:
        lower, upper = raw_lower, raw_upper
    return ((coord >= lower) & (coord <= upper)) if lower <= upper else ((coord >= lower) | (coord <= upper))
