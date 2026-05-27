from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import xarray as xr


def _chunk_plan(ds: xr.Dataset) -> dict[str, int]:
    chunks: dict[str, int] = {}
    for dim, size in ds.sizes.items():
        name = dim.lower()
        if "time" in name:
            chunks[dim] = min(int(size), 31)
        elif "depth" in name or "level" in name:
            chunks[dim] = min(int(size), 10)
        elif "lat" in name or name in {"y", "j"}:
            chunks[dim] = min(int(size), 200)
        elif "lon" in name or name in {"x", "i"}:
            chunks[dim] = min(int(size), 200)
    return chunks


def validate_netcdf(path: Path) -> dict[str, object]:
    ds = xr.open_dataset(path)
    try:
        return {
            "dims": dict(ds.sizes),
            "variables": list(ds.data_vars),
            "coords": list(ds.coords),
        }
    finally:
        ds.close()


def validate_zarr(path: Path) -> dict[str, object]:
    ds = xr.open_zarr(path)
    try:
        return {
            "dims": dict(ds.sizes),
            "variables": list(ds.data_vars),
            "coords": list(ds.coords),
        }
    finally:
        ds.close()


def netcdf_to_zarr(
    raw_path: Path,
    zarr_path: Path,
    *,
    overwrite: bool = False,
    consolidated: bool = True,
) -> Path:
    if zarr_path.exists() and not overwrite:
        validate_zarr(zarr_path)
        print(f"zarr exists: {zarr_path}")
        return zarr_path

    if zarr_path.exists() and overwrite:
        shutil.rmtree(zarr_path)

    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(raw_path, chunks={})
    try:
        chunks = _chunk_plan(ds)
        if chunks:
            ds = ds.chunk(chunks)
        ds.to_zarr(zarr_path, mode="w", consolidated=consolidated)
    finally:
        ds.close()

    validate_zarr(zarr_path)
    print(f"zarr written: {zarr_path}")
    return zarr_path


def zip_netcdf_to_zarr(
    zip_path: Path,
    extract_dir: Path,
    zarr_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    nc_files = sorted(extract_dir.rglob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found after extracting {zip_path}")

    if zarr_path.exists() and not overwrite:
        validate_zarr(zarr_path)
        print(f"zarr exists: {zarr_path}")
        return zarr_path

    if zarr_path.exists() and overwrite:
        shutil.rmtree(zarr_path)

    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.open_mfdataset(nc_files, combine="by_coords", chunks={})
    try:
        chunks = _chunk_plan(ds)
        if chunks:
            ds = ds.chunk(chunks)
        ds.to_zarr(zarr_path, mode="w", consolidated=True)
    finally:
        ds.close()

    validate_zarr(zarr_path)
    print(f"zarr written: {zarr_path}")
    return zarr_path
