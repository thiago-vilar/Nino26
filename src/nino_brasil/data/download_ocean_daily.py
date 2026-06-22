from __future__ import annotations

import calendar
import io
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from numcodecs import Blosc

from nino_brasil.data.audit import AuditLog, dataset_summary
from nino_brasil.data.zarr_store import ZARR_FORMAT
from nino_brasil.features.ocean_heat import layer_ocean_heat_content, warm_water_volume_m3
from nino_brasil.features.spatial import coordinate_range_mask
from nino_brasil.features.thermocline import d20_depth, thermocline_tilt, thermocline_tilt_slope


GLORYS_DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
GLORYS_OPERATIONAL_DATASETS = {
    "thetao": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
    "so": "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
    "zos": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
}
GLORYS_START_YEAR = 1993
GLORYS_DEFAULT_VARIABLES = ("thetao", "so", "zos")
UFS_ARCHIVE_URL = (
    "https://noaa-ufs-rnrmarine-pds.s3.amazonaws.com/"
    "ng-godas-1deg/3dvar/ana/1979_2019.zip"
)
UFS_START_YEAR = 1979
UFS_END_YEAR = 2019

PACIFIC_LAT_BOUNDS = (-5.0, 5.0)
PACIFIC_LON_BOUNDS_360 = (120.0, 280.0)
# One native GLORYS cell of halo is requested so 3x3 block means can be
# centred exactly on the canonical 0.25-degree nodes at the domain edges.
GLORYS_REQUEST_LAT_BOUNDS = (-5.1, 5.1)
GLORYS_REQUEST_LON_BOUNDS_360 = (119.9, 280.1)
SOURCE_MAX_DEPTH_M = 800.0
ANALYSIS_MAX_DEPTH_M = 700.0
TARGET_RESOLUTION_DEGREES = 0.25

CANONICAL_VARIABLES = {
    "thetao": "potential_temperature",
    "so": "salinity",
    "zos": "sea_surface_height",
}


def _year_bounds(year: int, end_date: str | pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    if end_date is not None:
        end = min(end, pd.Timestamp(end_date).normalize())
    if end < start:
        raise ValueError(f"No requested dates remain for {year}.")
    return start, end


def glorys_zarr_path(root: Path, year: int) -> Path:
    return root / str(year) / f"glorys12_equatorial_pacific_{year}_daily.zarr"


def glorys_operational_zarr_path(root: Path, variable: str, start_date: str, end_date: str) -> Path:
    start = pd.Timestamp(start_date).strftime("%Y%m%d")
    end = pd.Timestamp(end_date).strftime("%Y%m%d")
    return root / f"{start}_{end}" / f"glorys12_operational_{variable}_{start}_{end}_daily.zarr"


def glorys_subset_command(
    *,
    year: int,
    output_root: Path,
    variables: Sequence[str] = GLORYS_DEFAULT_VARIABLES,
    end_date: str | pd.Timestamp | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Build one annual Copernicus Marine request with all required variables."""
    start, end = _year_bounds(year, end_date)
    output = glorys_zarr_path(output_root, year)
    executable_name = "copernicusmarine.exe" if sys.platform == "win32" else "copernicusmarine"
    executable = Path(sys.executable).with_name(executable_name)
    command = [
        str(executable) if executable.exists() else executable_name,
        "subset",
        "--dataset-id",
        GLORYS_DATASET_ID,
    ]
    for variable in variables:
        command.extend(["--variable", variable])
    command.extend(
        [
            "--minimum-longitude",
            str(GLORYS_REQUEST_LON_BOUNDS_360[0]),
            "--maximum-longitude",
            str(GLORYS_REQUEST_LON_BOUNDS_360[1]),
            "--minimum-latitude",
            str(GLORYS_REQUEST_LAT_BOUNDS[0]),
            "--maximum-latitude",
            str(GLORYS_REQUEST_LAT_BOUNDS[1]),
            "--minimum-depth",
            "0",
            "--maximum-depth",
            str(SOURCE_MAX_DEPTH_M),
            "--start-datetime",
            start.strftime("%Y-%m-%dT00:00:00"),
            "--end-datetime",
            end.strftime("%Y-%m-%dT23:59:59"),
            "--coordinates-selection-method",
            "inside",
            "--service",
            "timeseries",
            "--output-directory",
            str(output.parent),
            "--output-filename",
            output.name,
            "--file-format",
            "zarr",
            "--overwrite" if overwrite else "--skip-existing",
        ]
    )
    return command


def format_command(command: Sequence[str]) -> str:
    """Return a CMD-safe command string for display and runbooks."""
    rendered: list[str] = []
    for item in command:
        value = str(item)
        rendered.append(f'"{value}"' if any(char.isspace() for char in value) else value)
    return " ".join(rendered)


def download_glorys_years(
    *,
    years: Iterable[int],
    output_root: Path,
    variables: Sequence[str] = GLORYS_DEFAULT_VARIABLES,
    end_date: str | pd.Timestamp | None = None,
    execute: bool = False,
    overwrite: bool = False,
) -> list[Path]:
    """Download one compressed annual Zarr per year and group all variables."""
    outputs: list[Path] = []
    for year in years:
        if year < GLORYS_START_YEAR:
            raise ValueError(f"GLORYS12 daily data starts in {GLORYS_START_YEAR}; received {year}.")
        output = glorys_zarr_path(output_root, year)
        start, end = _year_bounds(year, end_date)
        raw_required = set(variables)
        valid_existing = _daily_store_valid(
            output,
            required_variables=raw_required,
            expected_start=start,
            expected_end=end,
        )
        effective_overwrite = overwrite or (output.exists() and not valid_existing)
        command = glorys_subset_command(
            year=year,
            output_root=output_root,
            variables=variables,
            end_date=end_date,
            overwrite=effective_overwrite,
        )
        print(format_command(command))
        outputs.append(output)
        if not execute:
            continue
        if valid_existing and not overwrite:
            print(f"valid daily source exists: {output}")
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)
        if not _daily_store_valid(output, required_variables=raw_required, expected_start=start, expected_end=end):
            raise ValueError(f"Downloaded GLORYS store failed validation: {output}")
    return outputs


def glorys_operational_commands(
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    output_root: Path,
    overwrite: bool = False,
) -> dict[str, list[str]]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("Operational GLORYS end date precedes start date.")
    executable_name = "copernicusmarine.exe" if sys.platform == "win32" else "copernicusmarine"
    executable = Path(sys.executable).with_name(executable_name)
    commands: dict[str, list[str]] = {}
    for variable, dataset_id in GLORYS_OPERATIONAL_DATASETS.items():
        output = glorys_operational_zarr_path(output_root, variable, str(start.date()), str(end.date()))
        commands[variable] = [
            str(executable) if executable.exists() else executable_name,
            "subset",
            "--dataset-id",
            dataset_id,
            "--variable",
            variable,
            "--minimum-longitude",
            str(GLORYS_REQUEST_LON_BOUNDS_360[0]),
            "--maximum-longitude",
            str(GLORYS_REQUEST_LON_BOUNDS_360[1]),
            "--minimum-latitude",
            str(GLORYS_REQUEST_LAT_BOUNDS[0]),
            "--maximum-latitude",
            str(GLORYS_REQUEST_LAT_BOUNDS[1]),
            "--minimum-depth",
            "0",
            "--maximum-depth",
            str(SOURCE_MAX_DEPTH_M),
            "--start-datetime",
            start.strftime("%Y-%m-%dT00:00:00"),
            "--end-datetime",
            end.strftime("%Y-%m-%dT23:59:59"),
            "--coordinates-selection-method",
            "inside",
            "--service",
            "timeseries",
            "--output-directory",
            str(output.parent),
            "--output-filename",
            output.name,
            "--file-format",
            "zarr",
            "--overwrite" if overwrite else "--skip-existing",
        ]
    return commands


def download_glorys_operational(
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    output_root: Path,
    execute: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end >= pd.Timestamp.now().normalize():
        raise ValueError("Historical analysis ingest refuses today/future dates; use at most yesterday.")
    commands = glorys_operational_commands(
        start_date=start,
        end_date=end,
        output_root=output_root,
        overwrite=overwrite,
    )
    outputs: dict[str, Path] = {}
    for variable, command in commands.items():
        output = glorys_operational_zarr_path(output_root, variable, str(start.date()), str(end.date()))
        outputs[variable] = output
        valid_existing = _daily_store_valid(
            output,
            required_variables={variable},
            expected_start=start,
            expected_end=end,
        )
        if output.exists() and not valid_existing and not overwrite:
            command = glorys_operational_commands(
                start_date=start,
                end_date=end,
                output_root=output_root,
                overwrite=True,
            )[variable]
        print(format_command(command))
        if execute:
            if valid_existing and not overwrite:
                print(f"valid operational source exists: {output}")
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True)
            if not _daily_store_valid(output, required_variables={variable}, expected_start=start, expected_end=end):
                raise ValueError(f"Downloaded operational GLORYS store failed validation: {output}")
    return outputs


def _rename_glorys(ds: xr.Dataset) -> xr.Dataset:
    coordinate_mapping: dict[str, str] = {}
    for candidates, canonical in [
        (("latitude", "nav_lat"), "lat"),
        (("longitude", "nav_lon"), "lon"),
        (("depth", "deptht", "lev"), "depth"),
        (("time",), "time"),
    ]:
        for candidate in candidates:
            if candidate in ds.coords or candidate in ds.dims:
                if candidate != canonical:
                    coordinate_mapping[candidate] = canonical
                break
    renamed = ds.rename(coordinate_mapping)
    variable_mapping = {name: target for name, target in CANONICAL_VARIABLES.items() if name in renamed.data_vars}
    renamed = renamed.rename(variable_mapping)
    required = {"potential_temperature", "sea_surface_height"}
    missing = required.difference(renamed.data_vars)
    if missing:
        raise KeyError(f"GLORYS store is missing required variables: {sorted(missing)}")
    return renamed


def _normalize_longitude_360(ds: xr.Dataset) -> xr.Dataset:
    if "lon" not in ds.coords or ds["lon"].ndim != 1:
        raise ValueError("Daily ocean processing requires a one-dimensional longitude coordinate.")
    normalized = ds.assign_coords(lon=(ds["lon"] % 360.0)).sortby("lon")
    _, unique_indices = np.unique(np.asarray(normalized["lon"].values), return_index=True)
    if unique_indices.size != normalized.sizes["lon"]:
        normalized = normalized.isel(lon=np.sort(unique_indices))
    return normalized


def _regular_grid_factor(coord: xr.DataArray, target_resolution: float) -> int:
    values = np.asarray(coord.values, dtype=float)
    if values.size < 2:
        return 1
    native = float(np.median(np.abs(np.diff(values))))
    if native >= target_resolution * 0.99:
        return 1
    factor = int(round(target_resolution / native))
    if factor < 1 or not np.isclose(native * factor, target_resolution, atol=0.01):
        raise ValueError(f"Cannot coarsen native resolution {native} to {target_resolution} degrees exactly.")
    return factor


def _canonical_ocean_coordinates(
    target_resolution: float = TARGET_RESOLUTION_DEGREES,
) -> tuple[np.ndarray, np.ndarray]:
    lat = np.arange(
        PACIFIC_LAT_BOUNDS[0],
        PACIFIC_LAT_BOUNDS[1] + target_resolution * 0.5,
        target_resolution,
        dtype=np.float64,
    )
    lon = np.arange(
        PACIFIC_LON_BOUNDS_360[0],
        PACIFIC_LON_BOUNDS_360[1] + target_resolution * 0.5,
        target_resolution,
        dtype=np.float64,
    )
    return lat, lon


def _aligned_block_indices(coord: xr.DataArray, targets: np.ndarray, factor: int) -> np.ndarray:
    values = np.asarray(coord.values, dtype=float)
    native = float(np.median(np.abs(np.diff(values))))
    half_block = factor // 2
    expected_start = float(targets[0] - half_block * native)
    start = int(np.argmin(np.abs(values - expected_start)))
    stop = start + targets.size * factor
    if start < 0 or stop > values.size:
        raise ValueError("GLORYS source does not include the halo required for canonical 0.25-degree blocks.")
    indices = np.arange(start, stop)
    centers = values[indices].reshape(targets.size, factor).mean(axis=1)
    tolerance = max(native * 0.05, 1e-5)
    if not np.allclose(centers, targets, atol=tolerance, rtol=0.0):
        raise ValueError(
            "GLORYS native grid cannot be aligned exactly with the canonical 0.25-degree grid; "
            "refusing a phase-shifted comparison grid."
        )
    return indices


def _coarsen_glorys_to_canonical_grid(ds: xr.Dataset, target_resolution: float) -> xr.Dataset:
    target_lat, target_lon = _canonical_ocean_coordinates(target_resolution)
    lat_factor = _regular_grid_factor(ds["lat"], target_resolution)
    lon_factor = _regular_grid_factor(ds["lon"], target_resolution)
    if lat_factor == 1 and lon_factor == 1:
        return ds.interp(lat=target_lat, lon=target_lon, method="linear")
    lat_indices = _aligned_block_indices(ds["lat"], target_lat, lat_factor)
    lon_indices = _aligned_block_indices(ds["lon"], target_lon, lon_factor)
    selected = ds.isel(lat=lat_indices, lon=lon_indices)
    coarsened = selected.coarsen(
        lat=lat_factor,
        lon=lon_factor,
        boundary="exact",
    ).mean()
    return coarsened.assign_coords(lat=target_lat, lon=target_lon)


def _regrid_ufs_to_canonical_grid(
    ds: xr.Dataset,
    target_resolution: float = TARGET_RESOLUTION_DEGREES,
) -> xr.Dataset:
    """Interpolate coarse UFS fields to common nodes without claiming added information."""
    target_lat, target_lon = _canonical_ocean_coordinates(target_resolution)
    attrs = dict(ds.attrs)
    out = ds.interp(lat=target_lat, lon=target_lon, method="linear")
    out.attrs.update(attrs)
    out.attrs.update(
        {
            "native_horizontal_resolution_degrees": 1.0,
            "stored_horizontal_resolution_degrees": target_resolution,
            "spatial_processing": f"bilinear_interpolation_from_1_degree_to_{target_resolution}_degree",
            "spatial_information_gain": "none; interpolation only aligns comparison nodes",
            "canonical_grid": "5S-5N,120E-280E inclusive,0.25_degree_nodes",
        }
    )
    return out


def _validate_canonical_ocean_grid(
    ds: xr.Dataset,
    target_resolution: float = TARGET_RESOLUTION_DEGREES,
) -> None:
    target_lat, target_lon = _canonical_ocean_coordinates(target_resolution)
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise KeyError("Ocean store is missing canonical lat/lon coordinates.")
    if not np.allclose(np.asarray(ds["lat"].values), target_lat, atol=1e-6, rtol=0.0):
        raise ValueError("Ocean store latitude is not the canonical 0.25-degree grid.")
    if not np.allclose(np.asarray(ds["lon"].values), target_lon, atol=1e-6, rtol=0.0):
        raise ValueError("Ocean store longitude is not the canonical 0.25-degree grid.")


def _validate_daily_time(ds: xr.Dataset, *, expected_year: int | None = None) -> None:
    if "time" not in ds.coords:
        raise KeyError("Daily ocean store has no time coordinate.")
    index = pd.DatetimeIndex(ds["time"].values).normalize()
    if index.empty or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Daily ocean time coordinate is empty, duplicated, or unordered.")
    if len(index) > 1 and not np.all(np.diff(index.values) == np.timedelta64(1, "D")):
        raise ValueError("Daily ocean time coordinate contains gaps.")
    if expected_year is not None and set(index.year) != {expected_year}:
        raise ValueError(f"Expected only {expected_year} but found years {sorted(set(index.year))}.")


def _daily_store_valid(
    path: Path,
    *,
    required_variables: set[str],
    expected_start: pd.Timestamp | None = None,
    expected_end: pd.Timestamp | None = None,
    require_canonical_grid: bool = False,
) -> bool:
    if not path.exists():
        return False
    try:
        with xr.open_zarr(path, consolidated=None) as ds:
            if not required_variables.issubset(ds.data_vars):
                return False
            _validate_daily_time(ds)
            index = pd.DatetimeIndex(ds["time"].values).normalize()
            if expected_start is not None and index[0] != pd.Timestamp(expected_start).normalize():
                return False
            if expected_end is not None and index[-1] != pd.Timestamp(expected_end).normalize():
                return False
            if require_canonical_grid:
                _validate_canonical_ocean_grid(ds)
            return True
    except (OSError, ValueError, KeyError, IndexError):
        return False


def _zarr_encoding(ds: xr.Dataset) -> dict[str, dict[str, object]]:
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    encoding: dict[str, dict[str, object]] = {}
    for name, variable in ds.data_vars.items():
        chunks = []
        for dim in variable.dims:
            size = int(variable.sizes[dim])
            if dim == "time":
                chunks.append(min(size, 31))
            elif dim == "depth":
                chunks.append(min(size, 10))
            else:
                chunks.append(min(size, 180))
        encoding[name] = {"compressor": compressor, "dtype": "float32", "chunks": tuple(chunks)}
    return encoding


def _rechunk_for_storage(ds: xr.Dataset) -> xr.Dataset:
    chunks: dict[str, int] = {}
    for dim, size in ds.sizes.items():
        if dim == "time":
            chunks[dim] = min(int(size), 31)
        elif dim == "depth":
            chunks[dim] = min(int(size), 10)
        else:
            chunks[dim] = min(int(size), 180)
    return ds.chunk(chunks)


def _atomic_write_zarr(ds: xr.Dataset, output: Path, *, overwrite: bool = False) -> Path:
    if output.exists() and not overwrite:
        dataset_summary(output, zarr=True)
        print(f"zarr exists: {output}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.building")
    backup = output.with_name(f".{output.name}.backup")
    if staging.exists():
        shutil.rmtree(staging)
    clean = _rechunk_for_storage(ds.drop_encoding())
    clean.to_zarr(
        staging,
        mode="w",
        consolidated=True,
        zarr_format=ZARR_FORMAT,
        encoding=_zarr_encoding(clean),
    )
    dataset_summary(staging, zarr=True)
    if backup.exists():
        shutil.rmtree(backup)
    if output.exists():
        output.replace(backup)
    try:
        staging.replace(output)
    except BaseException:
        if backup.exists() and not output.exists():
            backup.replace(output)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    return output


def _process_glorys_dataset(
    source_ds: xr.Dataset,
    output: Path,
    *,
    source_label: str,
    source_dataset_id: str,
    target_resolution: float,
    overwrite: bool,
) -> Path:
    ds = _normalize_longitude_360(_rename_glorys(source_ds))
    ds = ds.assign_coords(time=pd.DatetimeIndex(ds["time"].values).normalize())
    _validate_daily_time(ds)
    ds = _coarsen_glorys_to_canonical_grid(ds, target_resolution)
    _validate_canonical_ocean_grid(ds, target_resolution)
    ds = ds.astype({name: np.float32 for name in ds.data_vars})
    ds.attrs.update(
        {
            "source": source_label,
            "source_dataset_id": source_dataset_id,
            "source_frequency": "daily_mean",
            "temporal_transform": "none",
            "daily_time_label": "calendar_date_at_00Z; values remain original daily means",
            "native_horizontal_resolution_degrees": 1.0 / 12.0,
            "stored_horizontal_resolution_degrees": target_resolution,
            "spatial_processing": f"aligned_3x3_block_mean_to_{target_resolution}_degree",
            "canonical_grid": "5S-5N,120E-280E inclusive,0.25_degree_nodes",
            "domain": "5S-5N, 120E-80W",
            "source_maximum_depth_m": SOURCE_MAX_DEPTH_M,
            "analysis_maximum_depth_m": ANALYSIS_MAX_DEPTH_M,
        }
    )
    index = pd.DatetimeIndex(ds["time"].values).normalize()
    required = {"potential_temperature", "salinity", "sea_surface_height"}
    valid_existing = _daily_store_valid(
        output,
        required_variables=required,
        expected_start=index[0],
        expected_end=index[-1],
        require_canonical_grid=True,
    )
    return _atomic_write_zarr(
        ds,
        output,
        overwrite=overwrite or (output.exists() and not valid_existing),
    )


def process_glorys_year(
    source: Path,
    output: Path,
    *,
    target_resolution: float = TARGET_RESOLUTION_DEGREES,
    overwrite: bool = False,
) -> Path:
    """Normalize and coarsen one native GLORYS daily Zarr to the project grid."""
    source_ds = xr.open_zarr(source, consolidated=None)
    try:
        return _process_glorys_dataset(
            source_ds,
            output,
            source_label="Copernicus Marine GLORYS12V1 daily reanalysis",
            source_dataset_id=GLORYS_DATASET_ID,
            target_resolution=target_resolution,
            overwrite=overwrite,
        )
    finally:
        source_ds.close()


def process_glorys_operational(
    sources: dict[str, Path],
    output: Path,
    *,
    target_resolution: float = TARGET_RESOLUTION_DEGREES,
    overwrite: bool = False,
) -> Path:
    opened = {variable: xr.open_zarr(path, consolidated=None) for variable, path in sources.items()}
    try:
        selected: list[xr.Dataset] = []
        for variable, ds in opened.items():
            if variable not in ds.data_vars:
                raise KeyError(f"Operational GLORYS {sources[variable]} is missing {variable}.")
            selected.append(ds[[variable]])
        merged = xr.merge(selected, compat="override", join="exact")
        return _process_glorys_dataset(
            merged,
            output,
            source_label="Copernicus Marine GLO12 operational daily analysis",
            source_dataset_id=";".join(GLORYS_OPERATIONAL_DATASETS.values()),
            target_resolution=target_resolution,
            overwrite=overwrite,
        )
    finally:
        for ds in opened.values():
            ds.close()


def ufs_zarr_path(root: Path, year: int) -> Path:
    return root / str(year) / f"noaa_ufs_equatorial_pacific_{year}_daily.zarr"


def ufs_ocean_members(names: Sequence[str], year: int) -> list[str]:
    prefix = f"{year}/"
    members = sorted(name for name in names if name.startswith(prefix) and "/ocn.ana." in name and name.endswith(".nc"))
    expected = 366 if calendar.isleap(year) else 365
    if len(members) != expected:
        raise ValueError(f"NOAA UFS {year} has {len(members)} ocean members; expected {expected}.")
    return members


def _ufs_member_time(name: str) -> pd.Timestamp:
    match = re.search(r"ocn\.ana\.(\d{10})\.nc$", name)
    if not match:
        raise ValueError(f"Cannot parse NOAA UFS timestamp from {name!r}.")
    return pd.to_datetime(match.group(1), format="%Y%m%d%H")


def _masked_float32(values: object) -> np.ndarray:
    return np.asarray(np.ma.filled(values, np.nan), dtype=np.float32)


def _ufs_member_dataset(payload: bytes, name: str) -> xr.Dataset:
    import netCDF4

    nc = netCDF4.Dataset("memory.nc", memory=payload)
    try:
        depth = np.asarray(nc.variables["Layer"][:], dtype=float)
        lat = np.asarray(nc.variables["lath"][:], dtype=float)
        lon = np.asarray(nc.variables["lonh"][:], dtype=float) % 360.0
        depth_idx = np.flatnonzero(depth <= SOURCE_MAX_DEPTH_M)
        lat_idx = np.flatnonzero((lat >= PACIFIC_LAT_BOUNDS[0]) & (lat <= PACIFIC_LAT_BOUNDS[1]))
        lon_idx = np.flatnonzero((lon >= PACIFIC_LON_BOUNDS_360[0]) & (lon <= PACIFIC_LON_BOUNDS_360[1]))
        if not depth_idx.size or not lat_idx.size or not lon_idx.size:
            raise ValueError("NOAA UFS equatorial Pacific subset is empty.")
        zslice = slice(int(depth_idx[0]), int(depth_idx[-1]) + 1)
        yslice = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)
        xslice = slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)
        time = pd.DatetimeIndex([_ufs_member_time(name).normalize()])
        variables: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
                "potential_temperature": (
                    ("time", "depth", "lat", "lon"),
                    _masked_float32(nc.variables["Temp"][0:1, zslice, yslice, xslice]),
                ),
                "salinity": (
                    ("time", "depth", "lat", "lon"),
                    _masked_float32(nc.variables["Salt"][0:1, zslice, yslice, xslice]),
                ),
                "sea_surface_height": (
                    ("time", "lat", "lon"),
                    _masked_float32(nc.variables["ave_ssh"][0:1, yslice, xslice]),
                ),
            }
        if "h" in nc.variables and nc.variables["h"].ndim == 4:
            variables["layer_thickness"] = (
                ("time", "depth", "lat", "lon"),
                _masked_float32(nc.variables["h"][0:1, zslice, yslice, xslice]),
            )
        return xr.Dataset(
            variables,
            coords={
                "time": time,
                "depth": depth[zslice].astype(np.float32),
                "lat": lat[yslice].astype(np.float32),
                "lon": lon[xslice].astype(np.float32),
            },
            attrs={
                "source": "NOAA UFS Marine Reanalysis daily analysis",
                "source_archive": UFS_ARCHIVE_URL,
                "source_frequency": "daily_analysis_24h_window_centered_12Z",
                "temporal_transform": "none",
                "daily_time_label": "analysis date at 00Z; source window remains centered at 12Z",
                "domain": "5S-5N, 120E-80W",
                "source_maximum_depth_m": SOURCE_MAX_DEPTH_M,
                "analysis_maximum_depth_m": ANALYSIS_MAX_DEPTH_M,
            },
        )
    finally:
        nc.close()


def _append_ufs_batch(batch: list[xr.Dataset], staging: Path, *, first: bool) -> None:
    combined = xr.concat(batch, dim="time").sortby("time")
    _validate_daily_time(combined)
    regridded = _regrid_ufs_to_canonical_grid(combined)
    _validate_canonical_ocean_grid(regridded)
    clean = _rechunk_for_storage(regridded.drop_encoding())
    if first:
        clean.to_zarr(
            staging,
            mode="w",
            consolidated=False,
            zarr_format=ZARR_FORMAT,
            encoding=_zarr_encoding(clean),
        )
    else:
        clean.to_zarr(staging, mode="a", append_dim="time", consolidated=False, zarr_format=ZARR_FORMAT)


def ingest_ufs_years(
    *,
    years: Sequence[int],
    output_root: Path,
    execute: bool = False,
    overwrite: bool = False,
    block_size_mb: int = 64,
) -> list[Path]:
    """Stream selected daily members from the remote 320 GB ZIP into annual Zarr.

    The full archive is never downloaded.  One browser-independent HTTP file
    handle and a readahead cache are reused across all requested years.
    """
    outputs = [ufs_zarr_path(output_root, year) for year in years]
    for year, output in zip(years, outputs, strict=True):
        if year < UFS_START_YEAR or year > UFS_END_YEAR:
            raise ValueError(f"NOAA UFS Marine Reanalysis covers {UFS_START_YEAR}-{UFS_END_YEAR}; received {year}.")
        print(f"NOAA UFS {year}: remote ZIP -> {output}")
    if not execute:
        return outputs

    import fsspec

    remote = fsspec.open(
        UFS_ARCHIVE_URL,
        "rb",
        block_size=block_size_mb * 1024 * 1024,
        cache_type="readahead",
    ).open()
    audit = AuditLog()
    try:
        archive = zipfile.ZipFile(remote)
        try:
            names = archive.namelist()
            for year, output in zip(years, outputs, strict=True):
                expected_start = pd.Timestamp(f"{year}-01-01")
                expected_end = pd.Timestamp(f"{year}-12-31")
                valid_existing = _daily_store_valid(
                    output,
                    required_variables={"potential_temperature", "salinity", "sea_surface_height"},
                    expected_start=expected_start,
                    expected_end=expected_end,
                    require_canonical_grid=True,
                )
                if valid_existing and not overwrite:
                    dataset_summary(output, zarr=True)
                    print(f"zarr exists: {output}")
                    continue
                members = ufs_ocean_members(names, year)
                output.parent.mkdir(parents=True, exist_ok=True)
                staging = output.with_name(f".{output.name}.building")
                backup = output.with_name(f".{output.name}.backup")
                if staging.exists():
                    shutil.rmtree(staging)
                audit.record(
                    task_id=f"noaa_ufs_ocean_daily_{year}",
                    dataset="noaa_ufs_marine_reanalysis",
                    status="started",
                    year=year,
                    source_archive=UFS_ARCHIVE_URL,
                    member_count=len(members),
                    output=str(output),
                )
                batch: list[xr.Dataset] = []
                current_month: int | None = None
                first = True
                for member in members:
                    timestamp = _ufs_member_time(member)
                    if current_month is not None and timestamp.month != current_month:
                        _append_ufs_batch(batch, staging, first=first)
                        first = False
                        batch = []
                    batch.append(_ufs_member_dataset(archive.read(member), member))
                    current_month = timestamp.month
                if batch:
                    _append_ufs_batch(batch, staging, first=first)
                import zarr

                zarr.consolidate_metadata(staging)
                staged_ds = xr.open_zarr(staging, consolidated=True)
                try:
                    _validate_daily_time(staged_ds, expected_year=year)
                    _validate_canonical_ocean_grid(staged_ds)
                finally:
                    staged_ds.close()
                if backup.exists():
                    shutil.rmtree(backup)
                if output.exists():
                    output.replace(backup)
                try:
                    staging.replace(output)
                except BaseException:
                    if backup.exists() and not output.exists():
                        backup.replace(output)
                    raise
                if backup.exists():
                    shutil.rmtree(backup)
                audit.record(
                    task_id=f"noaa_ufs_ocean_daily_{year}",
                    dataset="noaa_ufs_marine_reanalysis",
                    status="ok",
                    year=year,
                    output=str(output),
                    summary=dataset_summary(output, zarr=True),
                )
        finally:
            archive.close()
    finally:
        remote.close()
    return outputs


def _nino34_mean(array: xr.DataArray) -> xr.DataArray:
    subset = array.where(coordinate_range_mask(array["lat"], (-5.0, 5.0)), drop=True)
    subset = subset.where(coordinate_range_mask(subset["lon"], (190.0, 240.0), circular=True), drop=True)
    return subset.mean(["lat", "lon"], skipna=True)


def build_ocean_daily_features(source: Path, output: Path, *, overwrite: bool = False) -> Path:
    """Build source-neutral daily ENSO and deep-ocean features from a daily cube."""
    source_ds = xr.open_zarr(source, consolidated=None)
    try:
        ds = _normalize_longitude_360(source_ds)
        _validate_daily_time(ds)
        source_index = pd.DatetimeIndex(ds["time"].values).normalize()
        required_features = {
            "d20_nino34_mean_m",
            "ohc_0_300_nino34_j_m2",
            "ohc_0_700_nino34_j_m2",
            "wwv_equatorial_pacific_m3",
            "thermocline_tilt_m",
            "thermocline_tilt_slope_m_per_degree",
            "ocean_source_code",
        }
        valid_existing = _daily_store_valid(
            output,
            required_variables=required_features,
            expected_start=source_index[0],
            expected_end=source_index[-1],
        )
        if valid_existing and not overwrite:
            print(f"valid daily feature store exists: {output}")
            return output
        if "potential_temperature" not in ds:
            raise KeyError("Daily ocean cube needs potential_temperature.")
        temperature = ds["potential_temperature"]
        if "depth" in temperature.dims:
            temperature = temperature.chunk({"depth": -1})
        d20 = d20_depth(temperature)
        ohc_0_100 = layer_ocean_heat_content(temperature, 0.0, 100.0)
        ohc_0_300 = layer_ocean_heat_content(temperature, 0.0, 300.0)
        ohc_0_700 = layer_ocean_heat_content(temperature, 0.0, 700.0)
        ohc_300_700 = layer_ocean_heat_content(temperature, 300.0, 700.0)
        features = xr.Dataset(
            {
                "d20_nino34_mean_m": _nino34_mean(d20),
                "ohc_0_100_nino34_j_m2": _nino34_mean(ohc_0_100),
                "ohc_0_300_nino34_j_m2": _nino34_mean(ohc_0_300),
                "ohc_0_700_nino34_j_m2": _nino34_mean(ohc_0_700),
                "ohc_300_700_nino34_j_m2": _nino34_mean(ohc_300_700),
                "wwv_equatorial_pacific_m3": warm_water_volume_m3(d20),
                "thermocline_tilt_m": thermocline_tilt(d20),
                "thermocline_tilt_slope_m_per_degree": thermocline_tilt_slope(d20),
            }
        )
        source_name = str(ds.attrs.get("source", "daily ocean reanalysis"))
        source_lower = source_name.lower()
        source_code = (
            1.0
            if "ufs" in source_lower
            else 3.0
            if "operational" in source_lower or "glo12" in source_lower
            else 2.0
            if "glorys" in source_lower
            else 0.0
        )
        features["ocean_source_code"] = xr.DataArray(
            np.full(features.sizes["time"], source_code, dtype=np.float32),
            coords={"time": features["time"]},
            dims=("time",),
        )
        for level in (50.0, 100.0, 150.0, 200.0, 300.0, 500.0, 700.0):
            features[f"temperature_{int(level)}m_nino34_c"] = _nino34_mean(temperature.interp(depth=level))
        if "sea_surface_height" in ds:
            features["ssh_nino34_mean_m"] = _nino34_mean(ds["sea_surface_height"])
        if "salinity" in ds:
            features["sss_nino34_mean"] = _nino34_mean(ds["salinity"].isel(depth=0))
        features = features.astype({name: np.float32 for name in features.data_vars})
        features.attrs.update(
            {
                "source_cube": str(source),
                "source": source_name,
                "source_frequency": "daily",
                "temporal_interpolation": "forbidden_and_not_used",
                "feature_contract": "nino_brasil_daily_ocean_v1",
                "ohc_method": "rho0_cp0_cell_center_integral_with_clipped_interfaces",
                "d20_method": "linear_interpolation_first_20C_crossing",
                "ocean_source_code_mapping": "0=unknown,1=NOAA_UFS,2=GLORYS12_MY,3=GLO12_ANFC",
            }
        )
        return _atomic_write_zarr(
            features,
            output,
            overwrite=overwrite or (output.exists() and not valid_existing),
        )
    finally:
        source_ds.close()
