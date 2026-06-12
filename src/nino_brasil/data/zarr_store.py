from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


def chunk_plan(ds: xr.Dataset) -> dict[str, int]:
    """Project-wide chunking policy for Zarr stores (single source of truth)."""
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


def _first_present(items: Iterable[str], candidates: Iterable[str]) -> str | None:
    item_set = set(items)
    for candidate in candidates:
        if candidate in item_set:
            return candidate
    return None


def _time_coord_name(ds: xr.Dataset) -> str | None:
    for candidate in ["time", "valid_time", "date"]:
        if candidate in ds.coords or candidate in ds.dims:
            return candidate
    for name in ds.coords:
        if np.issubdtype(ds[name].dtype, np.datetime64):
            return str(name)
    return None


def _select_daily_variables(
    ds: xr.Dataset,
    variables: list[str] | None,
    aliases: dict[str, list[str]] | None,
) -> xr.Dataset:
    if not variables:
        return ds

    selected: dict[str, xr.DataArray] = {}
    missing: list[str] = []
    available = list(ds.data_vars)
    for variable in variables:
        candidates = [variable, *(aliases or {}).get(variable, [])]
        actual = _first_present(available, candidates)
        if actual is None:
            missing.append(variable)
            continue
        array = ds[actual]
        selected[variable] = array.rename(variable) if actual != variable else array

    if missing:
        raise KeyError(f"Variables not found in NetCDF/Zarr source: {missing}; available={available}")
    return xr.Dataset(selected, coords=ds.coords, attrs=dict(ds.attrs))


def _calendar_bounds(
    time: xr.DataArray,
    daily_start: str | pd.Timestamp | None,
    daily_end: str | pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if daily_start is not None:
        start = pd.Timestamp(daily_start).normalize()
    else:
        start = pd.Timestamp(time.values[0]).normalize()
    if daily_end is not None:
        end = pd.Timestamp(daily_end).normalize()
    else:
        end = pd.Timestamp(time.values[-1]).normalize()
    if end < start:
        raise ValueError(f"daily_end {end.date()} is before daily_start {start.date()}.")
    return start, end


def _daily_calendar(
    ds: xr.Dataset,
    time_name: str,
    daily_start: str | pd.Timestamp | None,
    daily_end: str | pd.Timestamp | None,
) -> pd.DatetimeIndex:
    start, end = _calendar_bounds(ds[time_name], daily_start, daily_end)
    return pd.date_range(start, end, freq="D", name=time_name)


def _expand_single_step_to_daily(ds: xr.Dataset, time_name: str, calendar: pd.DatetimeIndex) -> xr.Dataset:
    base = ds.isel({time_name: 0}, drop=True)
    return base.expand_dims({time_name: calendar})


def standardize_dataset_to_daily(
    ds: xr.Dataset,
    *,
    source_frequency: str = "daily",
    aggregation: str = "mean",
    time_name: str | None = None,
    daily_start: str | pd.Timestamp | None = None,
    daily_end: str | pd.Timestamp | None = None,
) -> xr.Dataset:
    """Return a daily dataset from daily, subdaily, weekly, or monthly source data."""
    resolved_time = time_name or _time_coord_name(ds)
    if resolved_time is None:
        return ds

    if resolved_time != "time":
        ds = ds.rename({resolved_time: "time"})
        resolved_time = "time"

    ds = ds.sortby(resolved_time)
    frequency = source_frequency.lower()
    if frequency in {"subdaily", "hourly", "6hourly"}:
        if aggregation == "sum":
            daily = ds.resample({resolved_time: "1D"}).sum()
        elif aggregation == "max":
            daily = ds.resample({resolved_time: "1D"}).max()
        elif aggregation == "min":
            daily = ds.resample({resolved_time: "1D"}).min()
        else:
            daily = ds.resample({resolved_time: "1D"}).mean()
    elif frequency == "daily":
        daily = ds.resample({resolved_time: "1D"}).mean()
    elif frequency in {"weekly", "monthly"}:
        calendar = _daily_calendar(ds, resolved_time, daily_start, daily_end)
        if ds.sizes.get(resolved_time, 0) == 1:
            daily = _expand_single_step_to_daily(ds, resolved_time, calendar)
        else:
            daily = ds.reindex({resolved_time: calendar}, method="ffill")
    else:
        raise ValueError(f"Unsupported source_frequency: {source_frequency}")

    daily.attrs.update(ds.attrs)
    daily.attrs.update(
        {
            "nino_brasil_temporal_standard": "daily",
            "nino_brasil_source_frequency": source_frequency,
            "nino_brasil_daily_transform": "resample_1d" if frequency in {"daily", "subdaily", "hourly", "6hourly"} else "forward_fill_to_daily_calendar",
        }
    )
    return daily


def _series_to_zarr_values(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.to_numpy(dtype="datetime64[ns]")
    if pd.api.types.is_numeric_dtype(series):
        if pd.api.types.is_integer_dtype(series) and series.isna().any():
            return series.astype(float).to_numpy()
        return series.to_numpy()
    if pd.api.types.is_bool_dtype(series) and not series.isna().any():
        return series.to_numpy(dtype=bool)
    return series.fillna("").astype(str).to_numpy(dtype=str)


def dataframe_to_zarr(
    frame: pd.DataFrame,
    zarr_path: Path,
    *,
    overwrite: bool = True,
    row_chunk_size: int = 100_000,
    attrs: dict[str, object] | None = None,
) -> Path:
    """Persist a tabular DataFrame as a row-oriented Zarr store."""
    if zarr_path.exists() and not overwrite:
        validate_zarr(zarr_path)
        print(f"zarr exists: {zarr_path}")
        return zarr_path

    if zarr_path.exists() and overwrite:
        shutil.rmtree(zarr_path)

    table = frame.reset_index(drop=True).copy()
    table.columns = [str(column) for column in table.columns]
    if table.columns.duplicated().any():
        duplicated = sorted(set(table.columns[table.columns.duplicated()]))
        raise ValueError(f"Duplicate columns cannot be written to Zarr: {duplicated}")

    data_vars = {
        column: (("row",), _series_to_zarr_values(table[column]))
        for column in table.columns
    }
    ds = xr.Dataset(data_vars, coords={"row": np.arange(len(table), dtype=np.int64)})
    ds.attrs.update(
        {
            "table_format": "nino_brasil_dataframe_zarr_v1",
            "row_count": int(len(table)),
            "pandas_columns": json.dumps(list(table.columns)),
            "pandas_dtypes": json.dumps({column: str(dtype) for column, dtype in table.dtypes.items()}),
        }
    )
    if attrs:
        ds.attrs.update(attrs)

    if len(table) > 0:
        ds = ds.chunk({"row": min(len(table), row_chunk_size)})

    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(zarr_path, mode="w", consolidated=True, zarr_format=2)
    validate_zarr(zarr_path)
    print(f"zarr table written: {zarr_path}")
    return zarr_path


def zarr_to_dataframe(zarr_path: Path) -> pd.DataFrame:
    """Read a DataFrame written by dataframe_to_zarr."""
    ds = xr.open_zarr(zarr_path)
    try:
        columns = json.loads(ds.attrs.get("pandas_columns", "[]")) or list(ds.data_vars)
        dtypes = json.loads(ds.attrs.get("pandas_dtypes", "{}"))
        data: dict[str, object] = {}
        for column in columns:
            values = ds[column].values
            if str(dtypes.get(column, "")).startswith("datetime64"):
                data[column] = pd.to_datetime(values)
            else:
                data[column] = values
        return pd.DataFrame(data)
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
        chunks = chunk_plan(ds)
        if chunks:
            ds = ds.chunk(chunks)
        ds.to_zarr(zarr_path, mode="w", consolidated=consolidated)
    finally:
        ds.close()

    validate_zarr(zarr_path)
    print(f"zarr written: {zarr_path}")
    return zarr_path


def netcdf_to_daily_zarr(
    raw_path: Path,
    zarr_path: Path,
    *,
    variables: list[str] | None = None,
    variable_aliases: dict[str, list[str]] | None = None,
    source_frequency: str = "daily",
    aggregation: str = "mean",
    daily_start: str | pd.Timestamp | None = None,
    daily_end: str | pd.Timestamp | None = None,
    overwrite: bool = False,
    consolidated: bool = True,
    quiet: bool = False,
) -> Path:
    """Convert NetCDF cache to a daily Zarr store, optionally one variable at a time."""
    if zarr_path.exists() and not overwrite:
        validate_zarr(zarr_path)
        if not quiet:
            print(f"daily zarr exists: {zarr_path}")
        return zarr_path

    if zarr_path.exists() and overwrite:
        shutil.rmtree(zarr_path)

    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(raw_path, chunks={})
    try:
        selected = _select_daily_variables(ds, variables, variable_aliases)
        daily = standardize_dataset_to_daily(
            selected,
            source_frequency=source_frequency,
            aggregation=aggregation,
            daily_start=daily_start,
            daily_end=daily_end,
        )
        chunks = chunk_plan(daily)
        if chunks:
            daily = daily.chunk(chunks)
        daily.to_zarr(zarr_path, mode="w", consolidated=consolidated, zarr_format=2)
    finally:
        ds.close()

    validate_zarr(zarr_path)
    if not quiet:
        print(f"daily zarr written: {zarr_path}")
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
        chunks = chunk_plan(ds)
        if chunks:
            ds = ds.chunk(chunks)
        ds.to_zarr(zarr_path, mode="w", consolidated=True)
    finally:
        ds.close()

    validate_zarr(zarr_path)
    print(f"zarr written: {zarr_path}")
    return zarr_path


def zip_netcdf_to_daily_zarr(
    zip_path: Path,
    extract_dir: Path,
    zarr_path: Path,
    *,
    variables: list[str] | None = None,
    variable_aliases: dict[str, list[str]] | None = None,
    source_frequency: str = "monthly",
    aggregation: str = "mean",
    daily_start: str | pd.Timestamp | None = None,
    daily_end: str | pd.Timestamp | None = None,
    overwrite: bool = False,
    quiet: bool = False,
) -> Path:
    """Extract a zipped NetCDF cache and write a daily Zarr store."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    nc_files = sorted(extract_dir.rglob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found after extracting {zip_path}")

    if zarr_path.exists() and not overwrite:
        validate_zarr(zarr_path)
        if not quiet:
            print(f"daily zarr exists: {zarr_path}")
        return zarr_path

    if zarr_path.exists() and overwrite:
        shutil.rmtree(zarr_path)

    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.open_mfdataset(nc_files, combine="by_coords", chunks={})
    try:
        selected = _select_daily_variables(ds, variables, variable_aliases)
        daily = standardize_dataset_to_daily(
            selected,
            source_frequency=source_frequency,
            aggregation=aggregation,
            daily_start=daily_start,
            daily_end=daily_end,
        )
        chunks = chunk_plan(daily)
        if chunks:
            daily = daily.chunk(chunks)
        daily.to_zarr(zarr_path, mode="w", consolidated=True, zarr_format=2)
    finally:
        ds.close()

    validate_zarr(zarr_path)
    if not quiet:
        print(f"daily zarr written: {zarr_path}")
    return zarr_path
