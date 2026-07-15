"""Target construction contracts shared by Phases 4, 6 and 8."""

from .chirps_native import (
    BRAZIL_BBOX,
    build_native_weekly_targets,
    concat_daily_native,
    native_grid_hash,
    native_pixel_table,
    target_to_frame,
    validate_native_target,
    weekly_native_precipitation,
)

__all__ = [
    "BRAZIL_BBOX",
    "build_native_weekly_targets",
    "concat_daily_native",
    "native_grid_hash",
    "native_pixel_table",
    "target_to_frame",
    "validate_native_target",
    "weekly_native_precipitation",
]
