from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import requests
import xarray as xr

from nino_brasil.data.audit import AuditLog, dataset_summary, file_info
from nino_brasil.data.download_http import download_url

try:
    import gsw
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    gsw = None


WOD_FILESERVER = "https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/wod"
PACIFIC_LAT_MIN = -35.0
PACIFIC_LAT_MAX = 30.0
PACIFIC_LON_MIN_360 = 120.0
PACIFIC_LON_MAX_360 = 290.0


def wod_ctd_http_url(year: int) -> str:
    return f"{WOD_FILESERVER}/{year}/wod_ctd_{year}.nc"


def wod_ctd_raw_path(raw_dir: Path, year: int) -> Path:
    return raw_dir / "wod" / str(year) / f"wod_ctd_{year}.nc"


def wod_ctd_zarr_path(zarr_root: Path, year: int) -> Path:
    return zarr_root / "ctd_noaa" / "wod" / str(year) / f"wod_ctd_{year}.zarr"


def download_wod_ctd_year(
    *,
    year: int,
    raw_dir: Path,
    dry_run: bool = True,
    overwrite: bool = False,
    allow_missing_source: bool = True,
    include_hash: bool = False,
    audit: AuditLog | None = None,
) -> Path:
    audit = audit or AuditLog()
    task_id = f"wod_ctd_download_{year}"
    url = wod_ctd_http_url(year)
    output_path = wod_ctd_raw_path(raw_dir, year)

    if dry_run:
        print(f"DRY RUN WOD CTD {year}: {url} -> {output_path}")
        return output_path

    audit.record(
        task_id=task_id,
        dataset="noaa_wod_ctd",
        status="started",
        step="download",
        url=url,
        raw_path=str(output_path),
    )
    try:
        path = download_url(url, output_path, overwrite=overwrite, resume=True)
        audit.record(
            task_id=task_id,
            dataset="noaa_wod_ctd",
            status="ok",
            step="download",
            url=url,
            raw=file_info(path, include_hash=include_hash),
        )
        return path
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if allow_missing_source and status_code == 404:
            audit.record(
                task_id=task_id,
                dataset="noaa_wod_ctd",
                status="missing_source",
                step="download",
                url=url,
                http_status=status_code,
                raw_path=str(output_path),
            )
            print(f"missing source for WOD CTD {year}: {url}")
            return output_path
        audit.record(
            task_id=task_id,
            dataset="noaa_wod_ctd",
            status="error",
            step="download",
            url=url,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    except BaseException as exc:
        audit.record(
            task_id=task_id,
            dataset="noaa_wod_ctd",
            status="error",
            step="download",
            url=url,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


def etl_wod_ctd_year(
    *,
    year: int,
    raw_dir: Path,
    zarr_root: Path,
    dry_run: bool = True,
    overwrite: bool = False,
    max_depth_m: float = 700.0,
    depth_step_m: float = 5.0,
    min_levels: int = 5,
    good_flags: Iterable[int] = (0,),
    include_hash: bool = False,
    audit: AuditLog | None = None,
) -> Path:
    audit = audit or AuditLog()
    task_id = f"wod_ctd_etl_{year}"
    raw_path = wod_ctd_raw_path(raw_dir, year)
    zarr_path = wod_ctd_zarr_path(zarr_root, year)

    if dry_run:
        print(f"DRY RUN WOD CTD ETL {year}: {raw_path} -> {zarr_path}")
        return zarr_path

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw WOD CTD file not found: {raw_path}")

    if zarr_path.exists() and not overwrite:
        dataset_summary(zarr_path, zarr=True)
        print(f"skip valid CTD Zarr: {zarr_path}")
        return zarr_path

    audit.record(
        task_id=task_id,
        dataset="noaa_wod_ctd",
        status="started",
        step="etl",
        raw_path=str(raw_path),
        zarr_path=str(zarr_path),
    )
    try:
        if gsw is None:
            raise RuntimeError("Missing dependency 'gsw'. Run: python -m pip install gsw")

        if zarr_path.exists() and overwrite:
            shutil.rmtree(zarr_path)

        ds = xr.open_dataset(raw_path, decode_times=False, mask_and_scale=True)
        try:
            out = wod_ctd_to_teos10_dataset(
                ds,
                year=year,
                source_url=wod_ctd_http_url(year),
                max_depth_m=max_depth_m,
                depth_step_m=depth_step_m,
                min_levels=min_levels,
                good_flags=tuple(int(flag) for flag in good_flags),
            )
        finally:
            ds.close()

        zarr_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            out.to_zarr(zarr_path, mode="w", consolidated=True)
        finally:
            out.close()

        audit.record(
            task_id=task_id,
            dataset="noaa_wod_ctd",
            status="ok",
            step="etl",
            raw=file_info(raw_path, include_hash=include_hash),
            zarr={"path": str(zarr_path), "summary": dataset_summary(zarr_path, zarr=True)},
        )
        print(f"CTD Zarr written: {zarr_path}")
        return zarr_path
    except BaseException as exc:
        audit.record(
            task_id=task_id,
            dataset="noaa_wod_ctd",
            status="error",
            step="etl",
            raw_path=str(raw_path),
            zarr_path=str(zarr_path),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


def wod_ctd_to_teos10_dataset(
    ds: xr.Dataset,
    *,
    year: int,
    source_url: str,
    max_depth_m: float,
    depth_step_m: float,
    min_levels: int,
    good_flags: tuple[int, ...],
) -> xr.Dataset:
    _require_wod_variables(ds)

    lat = _as_float_array(ds["lat"])
    lon_raw = _as_float_array(ds["lon"])
    lon_360 = _to_360(lon_raw)
    domain_mask = (
        (lat >= PACIFIC_LAT_MIN)
        & (lat <= PACIFIC_LAT_MAX)
        & (lon_360 >= PACIFIC_LON_MIN_360)
        & (lon_360 <= PACIFIC_LON_MAX_360)
    )
    selected_casts = np.flatnonzero(domain_mask)

    z_row = _row_sizes(ds["z_row_size"])
    t_row = _row_sizes(ds["Temperature_row_size"])
    s_row = _row_sizes(ds["Salinity_row_size"])
    z_start = _starts(z_row)
    t_start = _starts(t_row)
    s_start = _starts(s_row)

    z_values = _as_float_array(ds["z"])
    temp_values = _as_float_array(ds["Temperature"])
    sal_values = _as_float_array(ds["Salinity"])
    z_flags = _as_int_array(ds["z_WODflag"])
    temp_flags = _as_int_array(ds["Temperature_WODflag"])
    sal_flags = _as_int_array(ds["Salinity_WODflag"])
    temp_profile_flags = _as_int_array(ds["Temperature_WODprofileflag"])
    sal_profile_flags = _as_int_array(ds["Salinity_WODprofileflag"])

    dates = _optional_int_array(ds, "date", len(lat))
    time_days = _optional_float_array(ds, "time", len(lat))
    cast_ids = _optional_int_array(ds, "wod_unique_cast", len(lat))

    depth_grid = np.arange(0.0, max_depth_m + depth_step_m * 0.5, depth_step_m, dtype=np.float32)
    temp_profiles: list[np.ndarray] = []
    sal_profiles: list[np.ndarray] = []
    out_lat: list[float] = []
    out_lon: list[float] = []
    out_lon_360: list[float] = []
    out_date: list[int] = []
    out_time_days: list[float] = []
    out_cast_ids: list[int] = []
    out_valid_levels: list[int] = []

    for cast in selected_casts:
        if int(temp_profile_flags[cast]) not in good_flags:
            continue
        if int(sal_profile_flags[cast]) not in good_flags:
            continue

        n = min(int(z_row[cast]), int(t_row[cast]), int(s_row[cast]))
        if n < min_levels:
            continue

        z0 = int(z_start[cast])
        t0 = int(t_start[cast])
        s0 = int(s_start[cast])
        z = z_values[z0 : z0 + n]
        temp = temp_values[t0 : t0 + n]
        sal = sal_values[s0 : s0 + n]
        ok = (
            np.isfinite(z)
            & np.isfinite(temp)
            & np.isfinite(sal)
            & (z >= 0.0)
            & (z <= max_depth_m)
            & (sal >= 0.0)
            & (sal <= 45.0)
            & (temp >= -5.0)
            & (temp <= 45.0)
            & np.isin(z_flags[z0 : z0 + n], good_flags)
            & np.isin(temp_flags[t0 : t0 + n], good_flags)
            & np.isin(sal_flags[s0 : s0 + n], good_flags)
        )
        if int(ok.sum()) < min_levels:
            continue

        prepared = _prepare_profile(z[ok], temp[ok], sal[ok], min_levels=min_levels)
        if prepared is None:
            continue
        z_clean, temp_clean, sal_clean = prepared

        temp_grid = _interp_to_depth_grid(z_clean, temp_clean, depth_grid)
        sal_grid = _interp_to_depth_grid(z_clean, sal_clean, depth_grid)
        valid_levels = np.isfinite(temp_grid) & np.isfinite(sal_grid)
        if int(valid_levels.sum()) < min_levels:
            continue

        temp_profiles.append(temp_grid.astype(np.float32))
        sal_profiles.append(sal_grid.astype(np.float32))
        out_lat.append(float(lat[cast]))
        out_lon.append(float(lon_raw[cast]))
        out_lon_360.append(float(lon_360[cast]))
        out_date.append(int(dates[cast]))
        out_time_days.append(float(time_days[cast]))
        out_cast_ids.append(int(cast_ids[cast]))
        out_valid_levels.append(int(valid_levels.sum()))

    if not temp_profiles:
        raise ValueError(
            "No WOD CTD profiles passed the Pacific domain and QC filters "
            f"for year {year}."
        )

    temp_array = np.vstack(temp_profiles).astype(np.float32)
    sal_array = np.vstack(sal_profiles).astype(np.float32)
    lat_array = np.asarray(out_lat, dtype=np.float32)
    lon_array = np.asarray(out_lon, dtype=np.float32)
    lon_360_array = np.asarray(out_lon_360, dtype=np.float32)
    pressure = gsw.p_from_z(-depth_grid[None, :], lat_array[:, None]).astype(np.float32)
    absolute_salinity = gsw.SA_from_SP(
        sal_array,
        pressure,
        lon_360_array[:, None],
        lat_array[:, None],
    ).astype(np.float32)
    conservative_temperature = gsw.CT_from_t(
        absolute_salinity,
        temp_array,
        pressure,
    ).astype(np.float32)
    sigma0 = gsw.sigma0(absolute_salinity, conservative_temperature).astype(np.float32)

    thermocline_depth = _gradient_depths(conservative_temperature, depth_grid)
    halocline_depth = _gradient_depths(absolute_salinity, depth_grid)
    pycnocline_depth = _gradient_depths(sigma0, depth_grid)
    profile = np.arange(temp_array.shape[0], dtype=np.int32)
    date_yyyymmdd = np.asarray(out_date, dtype=np.int32)

    out = xr.Dataset(
        data_vars={
            "in_situ_temperature": (
                ("profile", "depth"),
                temp_array,
                {"units": "degree_C", "source_variable": "Temperature"},
            ),
            "practical_salinity": (
                ("profile", "depth"),
                sal_array,
                {"units": "1e-3", "source_variable": "Salinity"},
            ),
            "pressure": (
                ("profile", "depth"),
                pressure,
                {"units": "dbar", "method": "TEOS-10 gsw.p_from_z"},
            ),
            "absolute_salinity": (
                ("profile", "depth"),
                absolute_salinity,
                {"units": "g kg-1", "method": "TEOS-10 gsw.SA_from_SP"},
            ),
            "conservative_temperature": (
                ("profile", "depth"),
                conservative_temperature,
                {"units": "degree_C", "method": "TEOS-10 gsw.CT_from_t"},
            ),
            "sigma0": (
                ("profile", "depth"),
                sigma0,
                {"units": "kg m-3", "method": "TEOS-10 gsw.sigma0"},
            ),
            "thermocline_depth": (
                "profile",
                thermocline_depth,
                {"units": "m", "method": "max abs dCT/dz"},
            ),
            "halocline_depth": (
                "profile",
                halocline_depth,
                {"units": "m", "method": "max abs dSA/dz"},
            ),
            "pycnocline_depth": (
                "profile",
                pycnocline_depth,
                {"units": "m", "method": "max abs dsigma0/dz"},
            ),
            "date_yyyymmdd": ("profile", date_yyyymmdd),
            "time_days_since_1770": ("profile", np.asarray(out_time_days, dtype=np.float64)),
            "wod_unique_cast": ("profile", np.asarray(out_cast_ids, dtype=np.int64)),
            "valid_depth_levels": ("profile", np.asarray(out_valid_levels, dtype=np.int16)),
        },
        coords={
            "profile": profile,
            "depth": ("depth", depth_grid, {"units": "m", "positive": "down"}),
            "time": ("profile", _parse_dates(date_yyyymmdd)),
            "lat": ("profile", lat_array, {"units": "degrees_north"}),
            "lon": ("profile", lon_array, {"units": "degrees_east"}),
            "lon_360": ("profile", lon_360_array, {"units": "degrees_east"}),
        },
        attrs={
            "title": "NOAA WOD CTD profiles processed with TEOS-10",
            "source": "NOAA NCEI World Ocean Database CTD annual NetCDF",
            "source_url": source_url,
            "year": year,
            "domain": "Pacific 35S-30N, 120E-70W",
            "longitude_convention_for_filter": "0-360, 70W = 290E",
            "qc_policy": f"WOD observation and profile flags in {good_flags}",
            "depth_grid_m": f"0-{max_depth_m:g} step {depth_step_m:g}",
            "processing": "QC, Pacific crop, vertical interpolation, TEOS-10, stratification depths",
        },
    )
    return out


def _require_wod_variables(ds: xr.Dataset) -> None:
    required = [
        "lat",
        "lon",
        "z",
        "z_row_size",
        "z_WODflag",
        "Temperature",
        "Temperature_row_size",
        "Temperature_WODflag",
        "Temperature_WODprofileflag",
        "Salinity",
        "Salinity_row_size",
        "Salinity_WODflag",
        "Salinity_WODprofileflag",
    ]
    missing = [name for name in required if name not in ds]
    if missing:
        available = ", ".join(sorted(ds.variables))
        raise ValueError(f"WOD CTD file is missing variables {missing}. Available: {available}")


def _as_float_array(da: xr.DataArray) -> np.ndarray:
    return np.asarray(da.values, dtype=np.float64)


def _as_int_array(da: xr.DataArray) -> np.ndarray:
    return np.asarray(np.nan_to_num(da.values, nan=-99999), dtype=np.int64)


def _row_sizes(da: xr.DataArray) -> np.ndarray:
    values = _as_int_array(da)
    values[values < 0] = 0
    return values


def _optional_int_array(ds: xr.Dataset, name: str, size: int) -> np.ndarray:
    if name in ds:
        return _as_int_array(ds[name])
    return np.full(size, -99999, dtype=np.int64)


def _optional_float_array(ds: xr.Dataset, name: str, size: int) -> np.ndarray:
    if name in ds:
        return _as_float_array(ds[name])
    return np.full(size, np.nan, dtype=np.float64)


def _starts(row_sizes: np.ndarray) -> np.ndarray:
    if row_sizes.size == 0:
        return row_sizes
    starts = np.empty_like(row_sizes)
    starts[0] = 0
    starts[1:] = np.cumsum(row_sizes[:-1])
    return starts


def _to_360(lon: np.ndarray) -> np.ndarray:
    return np.mod(lon, 360.0)


def _prepare_profile(
    z: np.ndarray,
    temp: np.ndarray,
    sal: np.ndarray,
    *,
    min_levels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    order = np.argsort(z)
    z = z[order]
    temp = temp[order]
    sal = sal[order]
    _, unique_idx = np.unique(z, return_index=True)
    unique_idx.sort()
    if unique_idx.size < min_levels:
        return None
    return z[unique_idx], temp[unique_idx], sal[unique_idx]


def _interp_to_depth_grid(values_z: np.ndarray, values: np.ndarray, depth_grid: np.ndarray) -> np.ndarray:
    out = np.full(depth_grid.shape, np.nan, dtype=np.float32)
    inside = (depth_grid >= values_z.min()) & (depth_grid <= values_z.max())
    if inside.any():
        out[inside] = np.interp(depth_grid[inside], values_z, values).astype(np.float32)
    return out


def _gradient_depths(values: np.ndarray, depth: np.ndarray) -> np.ndarray:
    return np.asarray([_max_abs_gradient_depth(row, depth) for row in values], dtype=np.float32)


def _max_abs_gradient_depth(values: np.ndarray, depth: np.ndarray) -> float:
    ok = np.isfinite(values) & np.isfinite(depth)
    if int(ok.sum()) < 3:
        return np.nan
    x = depth[ok].astype(np.float64)
    y = values[ok].astype(np.float64)
    if np.unique(x).size < 3:
        return np.nan
    gradient = np.gradient(y, x)
    finite = np.isfinite(gradient)
    if not finite.any():
        return np.nan
    return float(x[np.nanargmax(np.abs(gradient))])


def _parse_dates(date_yyyymmdd: np.ndarray) -> np.ndarray:
    out = np.full(date_yyyymmdd.shape, np.datetime64("NaT", "D"), dtype="datetime64[D]")
    for index, raw in enumerate(date_yyyymmdd):
        value = int(raw)
        if value <= 0:
            continue
        try:
            parsed = datetime.strptime(f"{value:08d}", "%Y%m%d")
        except ValueError:
            continue
        out[index] = np.datetime64(parsed.date())
    return out
