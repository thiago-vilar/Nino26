from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import xarray as xr

from nino_brasil.config import load_config
from nino_brasil.data.standardize import normalize_longitudes


def _coord_name(ds: xr.Dataset | xr.DataArray, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
    raise KeyError(f"Dataset must contain one of these coordinates: {candidates}")


def target_grid_from_config(
    config: Mapping[str, object] | str | Path | None = None,
    section: str = "modeling",
) -> xr.Dataset:
    """Build the common modeling grid declared in project.yaml."""
    cfg = load_config(config) if config is None or isinstance(config, (str, Path)) else dict(config)
    grid = cfg[section]["grid"]
    resolution = float(grid["resolution_degrees"])
    lat = np.arange(
        float(grid["latitude_min"]),
        float(grid["latitude_max"]) + resolution * 0.5,
        resolution,
    )
    lon = np.arange(
        float(grid["longitude_min"]),
        float(grid["longitude_max"]) + resolution * 0.5,
        resolution,
    )
    return xr.Dataset(coords={"lat": lat, "lon": lon})


def normalize_for_common_grid(
    ds: xr.Dataset,
    lon_name: str | None = None,
    convention: str = "0_360",
) -> xr.Dataset:
    """Normalize longitude convention before regridding to the common grid."""
    lon = lon_name or _coord_name(ds, ("lon", "longitude", "x"))
    if lon != "lon":
        ds = ds.rename({lon: "lon"})
    lat = _coord_name(ds, ("lat", "latitude", "y"))
    if lat != "lat":
        ds = ds.rename({lat: "lat"})
    return normalize_longitudes(ds, lon_name="lon", convention=convention)


def regrid_dataset(
    ds: xr.Dataset,
    target_grid: xr.Dataset | None = None,
    *,
    method: str = "bilinear",
    periodic: bool = False,
    reuse_weights: bool = False,
    weights: str | Path | None = None,
) -> xr.Dataset:
    """Regrid a dataset with xesmf to the configured common modeling grid."""
    target = target_grid if target_grid is not None else target_grid_from_config()
    try:
        import xesmf as xe
    except ImportError as exc:  # pragma: no cover - depends on optional native stack
        if method not in {"bilinear", "nearest_s2d"}:
            raise ImportError("Install xesmf and ESMF/ESMPy to use conservative regridding.") from exc
        interp_method = "nearest" if method == "nearest_s2d" else "linear"
        out = ds.interp(lat=target["lat"], lon=target["lon"], method=interp_method)
        out.attrs.update(ds.attrs)
        out.attrs["regrid_method"] = f"xarray_{interp_method}"
        out.attrs["regrid_target"] = "project_common_grid"
        out.attrs["regrid_fallback"] = "xesmf_unavailable"
        return out

    regridder = xe.Regridder(
        ds,
        target,
        method,
        periodic=periodic,
        reuse_weights=reuse_weights,
        filename=str(weights) if weights is not None else None,
    )
    out = regridder(ds)
    out.attrs.update(ds.attrs)
    out.attrs["regrid_method"] = method
    out.attrs["regrid_target"] = "project_common_grid"
    return out


def regrid_to_project_grid(
    ds: xr.Dataset,
    *,
    lon_convention: str = "0_360",
    method: str = "bilinear",
) -> xr.Dataset:
    """Normalize coordinates and regrid to the project common grid."""
    normalized = normalize_for_common_grid(ds, convention=lon_convention)
    return regrid_dataset(normalized, target_grid_from_config(), method=method)
