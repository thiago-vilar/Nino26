from __future__ import annotations

import xarray as xr


def summarize_dataset(ds: xr.Dataset) -> dict[str, object]:
    """Return a small quality-control summary for an xarray dataset."""
    summary: dict[str, object] = {
        "dims": dict(ds.sizes),
        "variables": list(ds.data_vars),
        "coords": list(ds.coords),
    }
    if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
        summary["time_start"] = str(ds.time.min().values)
        summary["time_end"] = str(ds.time.max().values)
        summary["time_count"] = int(ds.sizes["time"])
    return summary


def count_missing(da: xr.DataArray) -> int:
    """Count missing values in a DataArray."""
    return int(da.isnull().sum().values)
