from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nino_brasil.features.precipitation_events import local_percentile_threshold


def _coord_name(da: xr.DataArray, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in da.coords or candidate in da.dims:
            return candidate
    return None


def _spatial_dims(da: xr.DataArray, time_name: str) -> list[str]:
    return [dim for dim in da.dims if dim != time_name]


def _weighted_centroid(field: xr.DataArray) -> tuple[float, float]:
    lat_name = _coord_name(field, ("lat", "latitude", "y"))
    lon_name = _coord_name(field, ("lon", "longitude", "x"))
    if lat_name is None or lon_name is None:
        return float("nan"), float("nan")
    positive = field.where(field > 0)
    total = positive.sum(skipna=True)
    if not np.isfinite(float(total)) or float(total) <= 0:
        return float("nan"), float("nan")
    lat = (positive * field[lat_name]).sum(skipna=True) / total
    lon = (positive * field[lon_name]).sum(skipna=True) / total
    return float(lat), float(lon)


def detect_mhw(
    sst: xr.DataArray,
    *,
    percentile: float = 90.0,
    min_duration_days: int = 5,
    train_times: xr.DataArray | list | None = None,
    time_name: str = "time",
    oni: xr.DataArray | None = None,
) -> pd.DataFrame:
    """Detect regional marine heatwave events and return a numeric catalog."""
    if time_name not in sst.coords and time_name not in sst.dims:
        raise KeyError(f"sst must contain {time_name!r}.")
    threshold = local_percentile_threshold(sst, percentile, train_times=train_times, time_name=time_name)
    intensity = (sst - threshold).where(sst > threshold)
    dims = _spatial_dims(intensity, time_name)
    regional_intensity = intensity.mean(dim=dims, skipna=True) if dims else intensity
    series = regional_intensity.to_series().fillna(0.0)
    series.index = pd.DatetimeIndex(series.index, name=time_name)
    active = series > 0

    rows: list[dict[str, object]] = []
    if active.empty:
        return _mhw_frame(rows)

    run_id = (active != active.shift(fill_value=False)).cumsum()
    for _, idx in series[active].groupby(run_id[active]).groups.items():
        event = series.loc[idx]
        if len(event) < min_duration_days:
            continue
        peak_date = event.idxmax()
        peak_field = intensity.sel({time_name: np.datetime64(peak_date)})
        lat_centroid, lon_centroid = _weighted_centroid(peak_field)
        overlap_with_oni = pd.NA
        if oni is not None:
            oni_event = oni.sel({time_name: slice(event.index.min(), event.index.max())})
            overlap_with_oni = bool((oni_event >= 0.5).any().item()) if oni_event.sizes.get(time_name, 0) else False
        rows.append(
            {
                "onset_date": event.index.min().date().isoformat(),
                "peak_date": peak_date.date().isoformat(),
                "end_date": event.index.max().date().isoformat(),
                "duration_days": int(len(event)),
                "max_intensity_c": round(float(event.max()), 4),
                "mean_intensity_c": round(float(event.mean()), 4),
                "cumulative_heat": round(float(event.sum()), 4),
                "lat_centroid": round(lat_centroid, 4) if np.isfinite(lat_centroid) else np.nan,
                "lon_centroid": round(lon_centroid, 4) if np.isfinite(lon_centroid) else np.nan,
                "overlap_with_oni": overlap_with_oni,
                "neb_precip_anomaly_next_90d": np.nan,
                "sul_precip_anomaly_next_90d": np.nan,
            }
        )
    return _mhw_frame(rows)


def _mhw_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "onset_date",
        "peak_date",
        "end_date",
        "duration_days",
        "max_intensity_c",
        "mean_intensity_c",
        "cumulative_heat",
        "lat_centroid",
        "lon_centroid",
        "overlap_with_oni",
        "neb_precip_anomaly_next_90d",
        "sul_precip_anomaly_next_90d",
    ]
    return pd.DataFrame(rows, columns=columns)


def export_mhw_catalog(catalog: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a marine heatwave catalog CSV."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output, index=False)
    return output
