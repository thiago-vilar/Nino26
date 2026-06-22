from __future__ import annotations

import calendar
import shutil
import tempfile
import zipfile
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from nino_brasil.data.audit import AuditLog, dataset_summary, file_info
from nino_brasil.data.download_cds import retrieve_cds
from nino_brasil.data.download_ocean_daily import (
    TARGET_RESOLUTION_DEGREES,
    _atomic_write_zarr,
    _canonical_ocean_coordinates,
    _validate_canonical_ocean_grid,
)


ORAS5_DATASET_ID = "reanalysis-oras5"
ORAS5_START_YEAR = 1958
ORAS5_CONSOLIDATED_END_YEAR = 2014
ORAS5_LATENCY_DAYS = 15
ORAS5_SOURCE_MAX_DEPTH_M = 800.0
ORAS5_LAT_BOUNDS = (-5.0, 5.0)
ORAS5_LON_BOUNDS_360 = (120.0, 280.0)
ORAS5_REGRID_LAT_BOUNDS = (-6.0, 6.0)
ORAS5_REGRID_LON_BOUNDS_360 = (119.0, 281.0)

ORAS5_SINGLE_LEVEL_VARIABLES = (
    "depth_of_20_c_isotherm",
    "ocean_heat_content_for_the_upper_300m",
    "ocean_heat_content_for_the_upper_700m",
    "sea_surface_height",
    "sea_surface_salinity",
)
ORAS5_ALL_LEVEL_VARIABLES = ("potential_temperature", "salinity")
ORAS5_VARIABLES = ORAS5_SINGLE_LEVEL_VARIABLES + ORAS5_ALL_LEVEL_VARIABLES

ORAS5_ALIASES = {
    "depth_of_20_c_isotherm": ("so20chgt", "d20"),
    "ocean_heat_content_for_the_upper_300m": ("sohtc300", "ohc_0_300m", "ohc300"),
    "ocean_heat_content_for_the_upper_700m": ("sohtc700", "ohc_0_700m", "ohc700"),
    "sea_surface_height": ("sossheig", "ssh", "zos"),
    "sea_surface_salinity": ("sosaline", "sss"),
    "potential_temperature": ("votemper", "thetao", "temperature"),
    "salinity": ("vosaline", "so"),
}


def latest_complete_oras5_month(today: date | pd.Timestamp | None = None) -> tuple[int, int]:
    """Return the latest complete month expected to be published with latency."""
    current = pd.Timestamp(today or date.today()).normalize()
    cutoff = current - pd.Timedelta(days=ORAS5_LATENCY_DAYS)
    complete_month_end = cutoff.to_period("M").start_time - pd.Timedelta(days=1)
    return int(complete_month_end.year), int(complete_month_end.month)


def months_for_year(
    year: int,
    *,
    end_year: int | None = None,
    end_month: int | None = None,
) -> list[int]:
    if end_year is None or end_month is None:
        end_year, end_month = latest_complete_oras5_month()
    if year < end_year:
        return list(range(1, 13))
    if year == end_year:
        return list(range(1, int(end_month) + 1))
    return []


def oras5_request(
    year: int,
    kind: str,
    months: Sequence[int],
    *,
    variables: Sequence[str] | None = None,
) -> dict[str, object]:
    if kind not in {"single", "all"}:
        raise ValueError("kind must be 'single' or 'all'.")
    if not months:
        raise ValueError(f"No ORAS5 months requested for {year}.")
    selected = tuple(variables or (ORAS5_SINGLE_LEVEL_VARIABLES if kind == "single" else ORAS5_ALL_LEVEL_VARIABLES))
    allowed = set(ORAS5_SINGLE_LEVEL_VARIABLES if kind == "single" else ORAS5_ALL_LEVEL_VARIABLES)
    invalid = set(selected).difference(allowed)
    if invalid:
        raise ValueError(f"Variables do not belong to ORAS5 {kind}: {sorted(invalid)}")
    return {
        "product_type": "consolidated" if year <= ORAS5_CONSOLIDATED_END_YEAR else "operational",
        "vertical_resolution": "single_level" if kind == "single" else "all_levels",
        "variable": list(selected),
        "year": str(year),
        "month": [f"{month:02d}" for month in months],
        "data_format": "netcdf",
    }


def oras5_raw_path(root: Path, year: int, kind: str, variable: str | None = None) -> Path:
    suffix = f"_{variable}" if variable else ""
    return root / str(year) / "_annual_kind" / f"oras5_{kind}{suffix}_{year}.zip"


def oras5_variable_zarr_path(root: Path, year: int, variable: str) -> Path:
    return root / str(year) / variable / f"oras5_{variable}_{year}_monthly.zarr"


def oras5_feature_path(root: Path, year: int) -> Path:
    return root / str(year) / f"oras5_ocean_features_{year}_monthly.zarr"


def oras5_monthly_store_valid(path: Path, variable: str, year: int, months: Sequence[int]) -> bool:
    if not path.exists():
        return False
    try:
        with xr.open_zarr(path, consolidated=None) as ds:
            if variable not in ds.data_vars or str(ds.attrs.get("source_frequency")) != "monthly_mean":
                return False
            actual = pd.DatetimeIndex(ds["time"].values)
            expected = pd.DatetimeIndex([pd.Timestamp(year=year, month=month, day=1) for month in months])
            if not actual.equals(expected):
                return False
            _validate_canonical_ocean_grid(ds)
            return True
    except (OSError, ValueError, KeyError):
        return False


def _find_time_name(ds: xr.Dataset) -> str:
    for name in ("time", "time_counter", "valid_time", "date"):
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError("ORAS5 source has no recognizable time coordinate.")


def _find_variable(ds: xr.Dataset, variable: str) -> str | None:
    for candidate in (variable, *ORAS5_ALIASES.get(variable, ())):
        if candidate in ds.data_vars:
            return candidate
    return None


def _find_depth_name(ds: xr.Dataset) -> str | None:
    for name in ("depth", "deptht", "lev", "level"):
        if name in ds.coords or name in ds.dims:
            return name
    return None


def _subset_oras5(ds: xr.Dataset) -> xr.Dataset:
    subset = ds
    coordinate_candidates = [
        name
        for name in ("nav_lat", "latitude", "lat", "nav_lon", "longitude", "lon")
        if name in subset.variables and name not in subset.coords
    ]
    if coordinate_candidates:
        subset = subset.set_coords(coordinate_candidates)
    depth_name = _find_depth_name(subset)
    if depth_name is not None and subset[depth_name].ndim == 1:
        depth_values = np.asarray(subset[depth_name].values, dtype=float)
        keep = np.flatnonzero(depth_values <= ORAS5_SOURCE_MAX_DEPTH_M)
        if not keep.size:
            raise ValueError("ORAS5 source has no levels above 800 m depth.")
        subset = subset.isel({depth_name: keep})

    lat_name = next((name for name in ("nav_lat", "latitude", "lat") if name in subset.coords), None)
    lon_name = next((name for name in ("nav_lon", "longitude", "lon") if name in subset.coords), None)
    if lat_name is None or lon_name is None:
        raise KeyError("ORAS5 source is missing latitude/longitude coordinates.")
    lat = subset[lat_name]
    lon = subset[lon_name] % 360.0
    mask = (
        (lat >= ORAS5_REGRID_LAT_BOUNDS[0])
        & (lat <= ORAS5_REGRID_LAT_BOUNDS[1])
        & (lon >= ORAS5_REGRID_LON_BOUNDS_360[0])
        & (lon <= ORAS5_REGRID_LON_BOUNDS_360[1])
    ).load()
    if not bool(mask.any()):
        raise ValueError("ORAS5 equatorial Pacific subset is empty.")
    if mask.ndim == 1:
        dim = mask.dims[0]
        indices = np.flatnonzero(mask.values)
        subset = subset.isel({dim: slice(int(indices.min()), int(indices.max()) + 1)})
    elif mask.ndim == 2:
        y_dim, x_dim = mask.dims
        rows, cols = np.where(np.asarray(mask.values, dtype=bool))
        subset = subset.isel(
            {
                y_dim: slice(int(rows.min()), int(rows.max()) + 1),
                x_dim: slice(int(cols.min()), int(cols.max()) + 1),
            }
        )
        local_mask = mask.isel(
            {
                y_dim: slice(int(rows.min()), int(rows.max()) + 1),
                x_dim: slice(int(cols.min()), int(cols.max()) + 1),
            }
        )
        subset = subset.where(local_mask)
    else:
        raise ValueError(f"Unsupported ORAS5 spatial coordinate shape: {mask.dims}")
    subset.attrs.update(
        {
            "source": "ECMWF ORAS5 monthly ocean reanalysis",
            "source_dataset_id": ORAS5_DATASET_ID,
            "source_frequency": "monthly_mean",
            "temporal_transform": "none",
            "domain": "5S-5N, 120E-80W",
            "source_maximum_depth_m": ORAS5_SOURCE_MAX_DEPTH_M,
        }
    )
    return subset


def _regrid_oras5_to_canonical(ds: xr.Dataset, variable: str) -> xr.Dataset:
    """Bilinearly align one ORAS5 curvilinear field to canonical 0.25-degree nodes."""
    from scipy.spatial import Delaunay

    array = ds[variable]
    lat, lon, spatial_dims = _spatial_coordinates(array)
    if lat.ndim == 1 and lon.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(np.asarray(lon.values), np.asarray(lat.values))
    elif lat.ndim == 2 and lon.ndim == 2 and lat.dims == lon.dims:
        lat_grid = np.asarray(lat.values)
        lon_grid = np.asarray(lon.values) % 360.0
    else:
        raise ValueError(f"Unsupported ORAS5 regrid coordinates: lat={lat.dims}, lon={lon.dims}")

    source_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    finite_source = np.all(np.isfinite(source_points), axis=1)
    source_indices = np.flatnonzero(finite_source)
    triangulation = Delaunay(source_points[finite_source])

    target_lat, target_lon = _canonical_ocean_coordinates(TARGET_RESOLUTION_DEGREES)
    target_lon_grid, target_lat_grid = np.meshgrid(target_lon, target_lat)
    target_points = np.column_stack([target_lon_grid.ravel(), target_lat_grid.ravel()])
    simplex = triangulation.find_simplex(target_points)
    outside = simplex < 0
    safe_simplex = simplex.copy()
    safe_simplex[outside] = 0
    transforms = triangulation.transform[safe_simplex]
    delta = target_points - transforms[:, 2]
    first_weights = np.einsum("nij,nj->ni", transforms[:, :2], delta)
    weights = np.column_stack([first_weights, 1.0 - first_weights.sum(axis=1)])
    vertices = source_indices[triangulation.simplices[safe_simplex]]

    leading_dims = [dim for dim in array.dims if dim not in spatial_dims]
    ordered = array.transpose(*leading_dims, *spatial_dims)
    leading_shape = tuple(int(ordered.sizes[dim]) for dim in leading_dims)
    flat = np.asarray(ordered.values).reshape((-1, lon_grid.size))
    vertex_values = flat[:, vertices]
    interpolated = np.sum(vertex_values * weights[None, :, :], axis=-1)
    interpolated[:, outside] = np.nan
    interpolated[np.any(~np.isfinite(vertex_values), axis=-1)] = np.nan
    values = interpolated.reshape((*leading_shape, target_lat.size, target_lon.size)).astype(np.float32)
    coords = {dim: ordered.coords[dim] for dim in leading_dims if dim in ordered.coords}
    coords.update({"lat": target_lat, "lon": target_lon})
    result = xr.Dataset(
        {variable: ((*leading_dims, "lat", "lon"), values)},
        coords=coords,
        attrs=dict(ds.attrs),
    )
    result[variable].attrs.update(array.attrs)
    result.attrs.update(
        {
            "stored_horizontal_resolution_degrees": TARGET_RESOLUTION_DEGREES,
            "spatial_processing": "scipy_delaunay_linear_interpolation_from_ORAS5_curvilinear_grid",
            "spatial_information_gain": "none; interpolation only aligns comparison nodes",
            "canonical_grid": "5S-5N,120E-280E inclusive,0.25_degree_nodes",
        }
    )
    _validate_canonical_ocean_grid(result)
    return result


def _canonical_months(ds: xr.Dataset, year: int, months: Sequence[int]) -> xr.Dataset:
    time_name = _find_time_name(ds)
    if time_name != "time":
        ds = ds.rename({time_name: "time"})
    index = pd.DatetimeIndex(ds["time"].values).to_period("M").to_timestamp()
    ds = ds.assign_coords(time=index).sortby("time")
    if pd.DatetimeIndex(ds["time"].values).has_duplicates:
        ds = ds.groupby("time").mean()
    expected = pd.DatetimeIndex([pd.Timestamp(year=year, month=month, day=1) for month in months])
    actual = pd.DatetimeIndex(ds["time"].values)
    if not actual.equals(expected):
        raise ValueError(
            f"ORAS5 {year} monthly calendar mismatch: expected={expected.strftime('%Y-%m').tolist()} "
            f"actual={actual.strftime('%Y-%m').tolist()}"
        )
    ds.attrs["monthly_time_label"] = "first day of source mean month; no daily expansion"
    return ds


def _netcdf_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".nc", ".nc4"})


def _files_for_variable(files: Sequence[Path], variable: str) -> list[Path]:
    matches: list[Path] = []
    for path in files:
        with xr.open_dataset(path) as ds:
            if _find_variable(ds, variable) is not None:
                matches.append(path)
    return matches


def process_oras5_archive(
    raw_path: Path,
    *,
    year: int,
    months: Sequence[int],
    kind: str,
    output_root: Path,
    variables: Sequence[str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    selected_variables = tuple(variables or (ORAS5_SINGLE_LEVEL_VARIABLES if kind == "single" else ORAS5_ALL_LEVEL_VARIABLES))
    outputs = [oras5_variable_zarr_path(output_root, year, variable) for variable in selected_variables]
    if all(
        oras5_monthly_store_valid(path, variable, year, months)
        for variable, path in zip(selected_variables, outputs, strict=True)
    ) and not overwrite:
        for path in outputs:
            dataset_summary(path, zarr=True)
        return outputs
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    with tempfile.TemporaryDirectory(prefix=f"oras5_{kind}_{year}_", dir=raw_path.parent) as tmp:
        extracted = Path(tmp)
        if zipfile.is_zipfile(raw_path):
            with zipfile.ZipFile(raw_path) as archive:
                archive.extractall(extracted)
        else:
            target = extracted / raw_path.name
            shutil.copy2(raw_path, target)
        files = _netcdf_files(extracted)
        if not files:
            raise FileNotFoundError(f"No NetCDF files extracted from {raw_path}")
        for variable, output in zip(selected_variables, outputs, strict=True):
            if oras5_monthly_store_valid(output, variable, year, months) and not overwrite:
                dataset_summary(output, zarr=True)
                continue
            replace_invalid = output.exists() and not oras5_monthly_store_valid(output, variable, year, months)
            variable_files = _files_for_variable(files, variable)
            if not variable_files:
                raise KeyError(f"ORAS5 archive {raw_path} is missing {variable}.")
            opened = [xr.open_dataset(path, chunks={}) for path in variable_files]
            try:
                combined = xr.combine_by_coords(opened, combine_attrs="override")
                actual = _find_variable(combined, variable)
                if actual is None:
                    raise KeyError(variable)
                promote = [
                    name
                    for name in ("nav_lat", "latitude", "lat", "nav_lon", "longitude", "lon")
                    if name in combined.variables and name not in combined.coords
                ]
                if promote:
                    combined = combined.set_coords(promote)
                selected = combined[[actual]].rename({actual: variable})
                selected = _regrid_oras5_to_canonical(_subset_oras5(selected), variable)
                selected = _canonical_months(selected, year, months)
                selected.attrs.update(
                    {
                        "variable_contract": variable,
                        "vertical_kind": "single_level" if kind == "single" else "all_levels",
                    }
                )
                _atomic_write_zarr(selected, output, overwrite=overwrite or replace_invalid)
            finally:
                for ds in opened:
                    ds.close()
    return outputs


def ingest_oras5_year(
    *,
    year: int,
    months: Sequence[int],
    raw_root: Path,
    output_root: Path,
    execute: bool = False,
    overwrite: bool = False,
    delete_raw_after_zarr: bool = False,
    include_hash: bool = False,
    fallback_split_variables: bool = True,
) -> list[Path]:
    if year < ORAS5_START_YEAR:
        raise ValueError(f"ORAS5 starts in {ORAS5_START_YEAR}.")
    audit = AuditLog()
    all_outputs: list[Path] = []
    for kind in ("single", "all"):
        request = oras5_request(year, kind, months)
        raw_path = oras5_raw_path(raw_root, year, kind)
        variables = ORAS5_SINGLE_LEVEL_VARIABLES if kind == "single" else ORAS5_ALL_LEVEL_VARIABLES
        outputs = [oras5_variable_zarr_path(output_root, year, variable) for variable in variables]
        all_outputs.extend(outputs)
        task_id = f"oras5_monthly_{kind}_{year}"
        if all(
            oras5_monthly_store_valid(path, variable, year, months)
            for variable, path in zip(variables, outputs, strict=True)
        ) and not overwrite:
            print(f"ORAS5 {year} {kind}: all monthly Zarr stores already exist")
            continue
        print(f"ORAS5 {year} {kind}: one CDS request for {len(variables)} variables -> {raw_path}")
        if not execute:
            print(request)
            continue
        audit.record(
            task_id=task_id,
            dataset="oras5_monthly",
            status="started",
            year=year,
            kind=kind,
            months=list(months),
            request=request,
            raw_path=str(raw_path),
        )
        try:
            retrieve_cds(
                dataset=ORAS5_DATASET_ID,
                request=request,
                output_path=raw_path,
                dry_run=False,
                overwrite=overwrite,
                label=f"ORAS5 monthly {year} {kind}",
            )
            process_oras5_archive(
                raw_path,
                year=year,
                months=months,
                kind=kind,
                output_root=output_root,
                variables=variables,
                overwrite=overwrite,
            )
            audit.record(
                task_id=task_id,
                dataset="oras5_monthly",
                status="ok",
                year=year,
                raw=file_info(raw_path, include_hash=include_hash),
                outputs=[{"path": str(path), "summary": dataset_summary(path, zarr=True)} for path in outputs],
            )
            if delete_raw_after_zarr and raw_path.exists():
                size = raw_path.stat().st_size
                raw_path.unlink()
                audit.record(
                    task_id=task_id,
                    dataset="oras5_monthly",
                    status="raw_cache_deleted",
                    raw_path=str(raw_path),
                    size_bytes=size,
                )
        except BaseException as exc:
            audit.record(
                task_id=task_id,
                dataset="oras5_monthly",
                status="error",
                year=year,
                kind=kind,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            if not fallback_split_variables:
                raise
            print(f"ORAS5 {year} {kind}: grouped request failed; splitting only this group by variable")
            for variable in variables:
                output = oras5_variable_zarr_path(output_root, year, variable)
                if oras5_monthly_store_valid(output, variable, year, months) and not overwrite:
                    dataset_summary(output, zarr=True)
                    continue
                variable_raw = oras5_raw_path(raw_root, year, kind, variable)
                variable_request = oras5_request(year, kind, months, variables=[variable])
                variable_task = f"oras5_monthly_{variable}_{year}"
                audit.record(
                    task_id=variable_task,
                    dataset="oras5_monthly_variable_fallback",
                    status="started",
                    year=year,
                    variable=variable,
                    request=variable_request,
                    raw_path=str(variable_raw),
                )
                try:
                    retrieve_cds(
                        dataset=ORAS5_DATASET_ID,
                        request=variable_request,
                        output_path=variable_raw,
                        dry_run=False,
                        overwrite=overwrite,
                        label=f"ORAS5 monthly {year} {variable}",
                    )
                    process_oras5_archive(
                        variable_raw,
                        year=year,
                        months=months,
                        kind=kind,
                        output_root=output_root,
                        variables=[variable],
                        overwrite=overwrite,
                    )
                    audit.record(
                        task_id=variable_task,
                        dataset="oras5_monthly_variable_fallback",
                        status="ok",
                        year=year,
                        variable=variable,
                        raw=file_info(variable_raw, include_hash=include_hash),
                        output={"path": str(output), "summary": dataset_summary(output, zarr=True)},
                    )
                    if delete_raw_after_zarr and variable_raw.exists():
                        variable_raw.unlink()
                except BaseException as variable_exc:
                    audit.record(
                        task_id=variable_task,
                        dataset="oras5_monthly_variable_fallback",
                        status="error",
                        year=year,
                        variable=variable,
                        error_type=type(variable_exc).__name__,
                        error=str(variable_exc),
                    )
                    raise
            if delete_raw_after_zarr and raw_path.exists():
                raw_path.unlink()
    return all_outputs


def ingest_oras5_years(
    *,
    years: Iterable[int],
    raw_root: Path,
    output_root: Path,
    end_year: int,
    end_month: int,
    execute: bool = False,
    overwrite: bool = False,
    delete_raw_after_zarr: bool = False,
    continue_on_error: bool = False,
) -> list[Path]:
    outputs: list[Path] = []
    for year in years:
        months = months_for_year(year, end_year=end_year, end_month=end_month)
        if not months:
            continue
        try:
            outputs.extend(
                ingest_oras5_year(
                    year=year,
                    months=months,
                    raw_root=raw_root,
                    output_root=output_root,
                    execute=execute,
                    overwrite=overwrite,
                    delete_raw_after_zarr=delete_raw_after_zarr,
                )
            )
        except BaseException as exc:
            print(f"ORAS5 {year}: error: {exc}")
            if not continue_on_error:
                raise
    return outputs


def _spatial_coordinates(array: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, list[str]]:
    lat_name = next((name for name in ("nav_lat", "latitude", "lat") if name in array.coords), None)
    lon_name = next((name for name in ("nav_lon", "longitude", "lon") if name in array.coords), None)
    if lat_name is None or lon_name is None:
        raise KeyError("Monthly ocean field has no latitude/longitude coordinates.")
    lat = array[lat_name]
    lon = array[lon_name] % 360.0
    spatial_dims = list(dict.fromkeys([*lat.dims, *lon.dims]))
    return lat, lon, spatial_dims


def _box_mean(
    array: xr.DataArray,
    *,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
) -> xr.DataArray:
    lat, lon, spatial_dims = _spatial_coordinates(array)
    mask = (
        (lat >= min(lat_bounds))
        & (lat <= max(lat_bounds))
        & (lon >= min(lon_bounds))
        & (lon <= max(lon_bounds))
    )
    return array.where(mask).mean(spatial_dims, skipna=True)


def _curvilinear_cell_area(array: xr.DataArray) -> xr.DataArray:
    lat, lon, spatial_dims = _spatial_coordinates(array)
    earth_radius = 6_371_000.0
    if lat.ndim == 1 and lon.ndim == 1:
        dlat = float(np.median(np.abs(np.diff(np.asarray(lat.values, dtype=float)))))
        lon_values = np.rad2deg(np.unwrap(np.deg2rad(np.asarray(lon.values, dtype=float))))
        dlon = float(np.median(np.abs(np.diff(lon_values))))
        return earth_radius**2 * np.deg2rad(dlat) * np.deg2rad(dlon) * np.cos(np.deg2rad(lat))
    if lat.ndim != 2 or lon.ndim != 2 or lat.dims != lon.dims:
        raise ValueError(f"Unsupported ORAS5 grid for cell area: lat={lat.dims}, lon={lon.dims}")
    lat_values = np.asarray(lat.values, dtype=float)
    lon_values = np.rad2deg(np.unwrap(np.deg2rad(np.asarray(lon.values, dtype=float)), axis=1))
    dlat = np.abs(np.gradient(lat_values, axis=0))
    dlon = np.abs(np.gradient(lon_values, axis=1))
    area_values = earth_radius**2 * np.deg2rad(dlat) * np.deg2rad(dlon) * np.cos(np.deg2rad(lat_values))
    return xr.DataArray(area_values, coords=lat.coords, dims=spatial_dims, name="cell_area_approx")


def _wwv_from_d20(d20: xr.DataArray) -> xr.DataArray:
    lat, lon, spatial_dims = _spatial_coordinates(d20)
    mask = (
        (lat >= ORAS5_LAT_BOUNDS[0])
        & (lat <= ORAS5_LAT_BOUNDS[1])
        & (lon >= ORAS5_LON_BOUNDS_360[0])
        & (lon <= ORAS5_LON_BOUNDS_360[1])
    )
    area = _curvilinear_cell_area(d20)
    result = (d20.where(mask) * area.where(mask)).sum(spatial_dims, skipna=True, min_count=1)
    result.name = "wwv_equatorial_pacific_m3"
    result.attrs.update(
        {
            "units": "m3",
            "method": "D20_times_spherical_cell_area",
            "cell_area_note": "curvilinear area estimated from local latitude/longitude gradients",
        }
    )
    return result


def _tilt_from_d20(d20: xr.DataArray) -> xr.DataArray:
    east = _box_mean(d20, lat_bounds=(-5.0, 5.0), lon_bounds=(220.0, 270.0))
    west = _box_mean(d20, lat_bounds=(-5.0, 5.0), lon_bounds=(140.0, 180.0))
    tilt = east - west
    tilt.name = "thermocline_tilt_east_minus_west_m"
    tilt.attrs.update({"units": "m", "method": "east_box_mean_D20_minus_west_box_mean_D20"})
    return tilt


def build_oras5_monthly_features(
    *,
    year: int,
    source_root: Path,
    output_root: Path,
    overwrite: bool = False,
) -> Path:
    paths = {variable: oras5_variable_zarr_path(source_root, year, variable) for variable in ORAS5_VARIABLES}
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing ORAS5 monthly stores for {year}: {missing[:3]}")
    opened = {variable: xr.open_zarr(path, consolidated=None) for variable, path in paths.items()}
    try:
        arrays = {variable: ds[variable] for variable, ds in opened.items()}
        d20 = arrays["depth_of_20_c_isotherm"]
        expected_time = pd.DatetimeIndex(d20["time"].values)
        output = oras5_feature_path(output_root, year)
        required_features = {
            "d20_nino34_mean_m",
            "ohc_0_300_nino34_j_m2",
            "ohc_0_700_nino34_j_m2",
            "wwv_equatorial_pacific_m3",
            "thermocline_tilt_east_minus_west_m",
        }
        valid_existing = False
        if output.exists():
            try:
                with xr.open_zarr(output, consolidated=None) as existing:
                    valid_existing = required_features.issubset(existing.data_vars) and pd.DatetimeIndex(
                        existing["time"].values
                    ).equals(expected_time)
            except (OSError, ValueError, KeyError):
                valid_existing = False
        if valid_existing and not overwrite:
            print(f"valid monthly feature store exists: {output}")
            return output
        features = xr.Dataset(
            {
                "d20_nino34_mean_m": _box_mean(d20, lat_bounds=(-5.0, 5.0), lon_bounds=(190.0, 240.0)),
                "ohc_0_300_nino34_j_m2": _box_mean(
                    arrays["ocean_heat_content_for_the_upper_300m"],
                    lat_bounds=(-5.0, 5.0),
                    lon_bounds=(190.0, 240.0),
                ),
                "ohc_0_700_nino34_j_m2": _box_mean(
                    arrays["ocean_heat_content_for_the_upper_700m"],
                    lat_bounds=(-5.0, 5.0),
                    lon_bounds=(190.0, 240.0),
                ),
                "ssh_nino34_mean_m": _box_mean(
                    arrays["sea_surface_height"], lat_bounds=(-5.0, 5.0), lon_bounds=(190.0, 240.0)
                ),
                "sss_nino34_mean": _box_mean(
                    arrays["sea_surface_salinity"], lat_bounds=(-5.0, 5.0), lon_bounds=(190.0, 240.0)
                ),
                "wwv_equatorial_pacific_m3": _wwv_from_d20(d20),
                "thermocline_tilt_east_minus_west_m": _tilt_from_d20(d20),
            }
        )
        temperature = arrays["potential_temperature"]
        depth_name = _find_depth_name(opened["potential_temperature"])
        if depth_name is not None:
            for level in (50.0, 100.0, 150.0, 200.0, 300.0, 500.0, 700.0):
                features[f"temperature_{int(level)}m_nino34_c"] = _box_mean(
                    temperature.interp({depth_name: level}),
                    lat_bounds=(-5.0, 5.0),
                    lon_bounds=(190.0, 240.0),
                )
        features = features.astype({name: np.float32 for name in features.data_vars})
        features.attrs.update(
            {
                "source": "ECMWF ORAS5 monthly ocean reanalysis",
                "source_frequency": "monthly_mean",
                "temporal_interpolation": "forbidden_and_not_used",
                "feature_contract": "nino_brasil_monthly_ocean_v1",
                "availability_policy": "use only after source month is complete and published",
            }
        )
        return _atomic_write_zarr(
            features,
            output,
            overwrite=overwrite or (output.exists() and not valid_existing),
        )
    finally:
        for ds in opened.values():
            ds.close()


def align_monthly_features_causally(
    monthly: xr.Dataset,
    daily_time: pd.DatetimeIndex,
    *,
    publication_lag_days: int = ORAS5_LATENCY_DAYS,
) -> xr.Dataset:
    """Expose monthly covariates on daily rows only after their publication date.

    Values remain explicitly monthly covariates.  This function is for model
    matrix alignment and must not be presented as a daily observation product.
    """
    if "time" not in monthly.coords:
        raise KeyError("Monthly feature store has no time coordinate.")
    month_start = pd.DatetimeIndex(monthly["time"].values).to_period("M").to_timestamp()
    release_dates = month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(days=publication_lag_days)
    released = monthly.assign_coords(time=release_dates)
    aligned = released.reindex(time=pd.DatetimeIndex(daily_time), method="ffill")
    aligned.attrs.update(monthly.attrs)
    aligned.attrs.update(
        {
            "temporal_role": "monthly_covariate_aligned_to_daily_model_rows",
            "source_frequency": "monthly_mean",
            "publication_lag_days": int(publication_lag_days),
            "daily_observation_claim": "false",
            "alignment_method": "last_published_month_forward_carry_for_model_matrix_only",
        }
    )
    return aligned
