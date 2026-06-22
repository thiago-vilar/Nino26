from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


def _coord_name(da: xr.DataArray, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in da.coords or candidate in da.dims:
            return candidate
    return None


def _require_time(da: xr.DataArray, time_name: str) -> None:
    if time_name not in da.coords and time_name not in da.dims:
        raise KeyError(f"DataArray must contain {time_name!r}.")


def _monthly_field(da: xr.DataArray, time_name: str) -> xr.DataArray:
    _require_time(da, time_name)
    return da.sortby(time_name).resample({time_name: "MS"}).mean()


def _spatial_dims(da: xr.DataArray, time_name: str) -> list[str]:
    return [dim for dim in da.dims if dim != time_name]


def _monthly_scalar(da: xr.DataArray, time_name: str, *, reducer: str = "mean") -> xr.DataArray:
    monthly = _monthly_field(da, time_name)
    dims = _spatial_dims(monthly, time_name)
    if not dims:
        return monthly
    if reducer == "max":
        return monthly.max(dim=dims, skipna=True)
    return monthly.mean(dim=dims, skipna=True)


def _monthly_anomaly(monthly: xr.DataArray, time_name: str) -> xr.DataArray:
    climatology = monthly.groupby(f"{time_name}.month").mean(time_name, skipna=True)
    return monthly.groupby(f"{time_name}.month") - climatology


def _monthly_lon_slope(ssta: xr.DataArray, time_name: str) -> xr.DataArray:
    monthly = _monthly_field(ssta, time_name)
    lon_name = _coord_name(monthly, ("lon", "longitude", "x"))
    if lon_name is None or lon_name not in monthly.dims:
        base = _monthly_scalar(ssta, time_name)
        return xr.full_like(base, np.nan).rename("nino34_ssta_slope_lon")
    monthly = monthly.sortby(lon_name)
    slope = monthly.differentiate(lon_name)
    dims = _spatial_dims(slope, time_name)
    if dims:
        slope = slope.mean(dim=dims, skipna=True)
    return slope.rename("nino34_ssta_slope_lon")


def _series(da: xr.DataArray, name: str, time_name: str) -> pd.Series:
    values = da.to_series()
    values.index = pd.DatetimeIndex(values.index, name=time_name)
    return values.rename(name)


def _event_phase_columns(table: pd.DataFrame, threshold_c: float) -> pd.DataFrame:
    out = table.copy()
    warm = out["nino34_ssta_mean"] > threshold_c
    out["signal_duration_months"] = 0
    out["event_phase"] = "neutral"
    if out.empty:
        return out

    run_id = (warm != warm.shift(fill_value=False)).cumsum()
    for _, idx in out[warm].groupby(run_id[warm]).groups.items():
        run = out.loc[idx]
        peak_idx = run["nino34_ssta_mean"].idxmax()
        out.loc[idx, "signal_duration_months"] = len(run)
        out.loc[run.index[run.index < peak_idx], "event_phase"] = "onset"
        out.loc[peak_idx, "event_phase"] = "peak"
        out.loc[run.index[run.index > peak_idx], "event_phase"] = "decay"
    return out


def build_nino34_event_table(
    ssta: xr.DataArray,
    d20: xr.DataArray,
    ohc_300: xr.DataArray,
    ohc_700: xr.DataArray,
    thermocline_depth: xr.DataArray,
    *,
    peak_years: Iterable[int],
    time_name: str = "time",
    event_threshold_c: float = 0.5,
) -> pd.DataFrame:
    """Build a monthly numeric diagnostic table for historical Nino 3.4 events."""
    ssta_mean = _monthly_scalar(ssta, time_name).rename("nino34_ssta_mean")
    ssta_max = _monthly_scalar(ssta, time_name, reducer="max").rename("nino34_ssta_max")
    slope_lon = _monthly_lon_slope(ssta, time_name)

    d20_mean = _monthly_scalar(d20, time_name).rename("d20_mean_m")
    d20_anomaly = _monthly_anomaly(d20_mean, time_name).rename("d20_anomaly_m")
    thermocline_mean = _monthly_scalar(thermocline_depth, time_name).rename("thermocline_depth_mean_m")
    ohc_300_mean = _monthly_scalar(ohc_300, time_name).rename("ohc_300_mean")
    ohc_300_anomaly = _monthly_anomaly(ohc_300_mean, time_name).rename("ohc_300_anomaly")
    ohc_700_mean = _monthly_scalar(ohc_700, time_name).rename("ohc_700_mean")
    ohc_700_anomaly = _monthly_anomaly(ohc_700_mean, time_name).rename("ohc_700_anomaly")

    frame = pd.concat(
        [
            _series(ssta_mean, "nino34_ssta_mean", time_name),
            _series(ssta_max, "nino34_ssta_max", time_name),
            _series(slope_lon, "nino34_ssta_slope_lon", time_name),
            _series(d20_mean, "d20_mean_m", time_name),
            _series(d20_anomaly, "d20_anomaly_m", time_name),
            _series(thermocline_mean, "thermocline_depth_mean_m", time_name),
            _series(ohc_300_mean, "ohc_300_mean", time_name),
            _series(ohc_300_anomaly, "ohc_300_anomaly", time_name),
            _series(ohc_700_mean, "ohc_700_mean", time_name),
            _series(ohc_700_anomaly, "ohc_700_anomaly", time_name),
        ],
        axis=1,
    ).reset_index()
    frame["year"] = frame[time_name].dt.year
    frame["month"] = frame[time_name].dt.month
    frame["is_peak_year"] = frame["year"].isin({int(year) for year in peak_years})
    frame = _event_phase_columns(frame, event_threshold_c)

    columns = [
        "year",
        "month",
        time_name,
        "nino34_ssta_mean",
        "nino34_ssta_max",
        "nino34_ssta_slope_lon",
        "d20_mean_m",
        "d20_anomaly_m",
        "thermocline_depth_mean_m",
        "ohc_300_mean",
        "ohc_300_anomaly",
        "ohc_700_mean",
        "ohc_700_anomaly",
        "signal_duration_months",
        "event_phase",
        "is_peak_year",
    ]
    return frame[columns].round(
        {
            "nino34_ssta_mean": 4,
            "nino34_ssta_max": 4,
            "nino34_ssta_slope_lon": 6,
            "d20_mean_m": 4,
            "d20_anomaly_m": 4,
            "thermocline_depth_mean_m": 4,
            "ohc_300_mean": 4,
            "ohc_300_anomaly": 4,
            "ohc_700_mean": 4,
            "ohc_700_anomaly": 4,
        }
    )


def export_nino34_event_tables(
    table: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write the monthly table and peak-year comparison CSVs."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    monthly_path = output / "nino34_event_table_monthly.csv"
    peak_path = output / "nino34_peak_comparison.csv"
    table.to_csv(monthly_path, index=False)
    table[table["is_peak_year"]].to_csv(peak_path, index=False)
    return monthly_path, peak_path
