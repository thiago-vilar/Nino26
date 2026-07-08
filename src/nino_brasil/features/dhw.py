from __future__ import annotations

import numpy as np
import xarray as xr

from nino_brasil.features.nino import sst_box_index


def infer_samples_per_week(time: xr.DataArray) -> float:
    """Infer the number of samples per week from a regular time coordinate."""
    values = time.values
    if values.size < 2:
        return 1.0
    deltas = np.diff(values).astype("timedelta64[s]").astype("float64")
    median_days = np.nanmedian(deltas) / 86400.0
    if not np.isfinite(median_days) or median_days <= 0:
        return 1.0
    return float(7.0 / median_days)


def degree_heating_weeks(
    ssta: xr.DataArray,
    *,
    threshold_c: float = 0.5,
    window_weeks: int = 12,
    time_name: str = "time",
    samples_per_week: float | None = None,
) -> xr.DataArray:
    """Compute Nino 3.4 Degree Heating Weeks from daily SST anomalies.

    The NINO26 DHW protocol is anchored to the El Nino thermal threshold:
    daily HotSpot = SSTA when SSTA >= ``threshold_c``; values below the
    threshold are discarded as background variability. The rolling 12-week
    accumulation is then expressed in degree C-weeks.
    """
    if window_weeks < 1:
        raise ValueError("window_weeks must be positive.")
    spw = samples_per_week or infer_samples_per_week(ssta[time_name])
    window_samples = max(1, int(round(window_weeks * spw)))
    hotspot = xr.where(ssta >= threshold_c, ssta, 0.0)
    dhw = hotspot.rolling({time_name: window_samples}, min_periods=window_samples).sum() / spw
    dhw.attrs.update(
        {
            "units": "degree_C_weeks",
            "threshold_c": float(threshold_c),
            "hotspot_rule": "SSTA if SSTA >= threshold_c else 0",
            "window_weeks": int(window_weeks),
            "samples_per_week": float(spw),
            "valid_after_weeks": int(window_weeks),
        }
    )
    return dhw.rename(f"dhw_{window_weeks}w")


def thermal_window_mean(
    ssta: xr.DataArray,
    *,
    window_weeks: int = 12,
    time_name: str = "time",
    samples_per_week: float | None = None,
) -> xr.DataArray:
    """Rolling mean SSTA over the same thermal window used to validate DHW."""
    if window_weeks < 1:
        raise ValueError("window_weeks must be positive.")
    spw = samples_per_week or infer_samples_per_week(ssta[time_name])
    window_samples = max(1, int(round(window_weeks * spw)))
    mean = ssta.rolling({time_name: window_samples}, min_periods=window_samples).mean()
    mean.attrs.update(
        {
            "units": "degree_C",
            "window_weeks": int(window_weeks),
            "samples_per_week": float(spw),
            "valid_after_weeks": int(window_weeks),
        }
    )
    return mean.rename(f"ssta_mean_{window_weeks}w")


def weekly_mean(da: xr.DataArray, *, time_name: str = "time", label: str = "right") -> xr.DataArray:
    """Aggregate a daily or subweekly series to weekly means."""
    return da.resample({time_name: "W-SUN"}, label=label).mean()


def weekly_reduce(
    da: xr.DataArray,
    *,
    how: str = "max",
    time_name: str = "time",
    label: str = "right",
) -> xr.DataArray:
    """Reduce a native-resolution series to the canonical 7-day weekly axis."""
    resampled = da.resample({time_name: "W-SUN"}, label=label)
    if how == "max":
        out = resampled.max()
    elif how == "mean":
        out = resampled.mean()
    else:
        raise ValueError("how must be 'max' or 'mean'.")
    out.attrs.update(da.attrs)
    out.attrs["weekly_reduction"] = how
    return out


def equatorial_dhw_indices(
    ssta_field: xr.DataArray,
    *,
    threshold_c: float = 0.5,
    window_weeks: int = 12,
    time_name: str = "time",
    lat_name: str = "lat",
    lon_name: str = "lon",
    weekly_reduction: str = "max",
) -> xr.Dataset:
    """Build weekly DHW indices for the Fase 3F equatorial pathway.

    The order is intentionally daily first, weekly second:
    area-average daily SSTA -> daily DHW rolling accumulation -> weekly reduce.
    """
    boxes = {
        "nino34": {"lat_bounds": (-5.0, 5.0), "lon_bounds": (-170.0, -120.0)},
        "nino4": {"lat_bounds": (-5.0, 5.0), "lon_bounds": (160.0, 210.0)},
        "equatorial_guide": {"lat_bounds": (-2.0, 2.0), "lon_bounds": (120.0, 280.0)},
        "west_equatorial_guide": {"lat_bounds": (-2.0, 2.0), "lon_bounds": (120.0, 205.0)},
        "east_equatorial_guide": {"lat_bounds": (-2.0, 2.0), "lon_bounds": (205.0, 280.0)},
    }
    data_vars: dict[str, xr.DataArray] = {}
    for name, bounds in boxes.items():
        index = sst_box_index(
            ssta_field,
            lat_bounds=bounds["lat_bounds"],
            lon_bounds=bounds["lon_bounds"],
            name=f"{name}_ssta",
            lat_name=lat_name,
            lon_name=lon_name,
        )
        daily_dhw = degree_heating_weeks(
            index,
            threshold_c=threshold_c,
            window_weeks=window_weeks,
            time_name=time_name,
        )
        weekly_dhw = weekly_reduce(daily_dhw, how=weekly_reduction, time_name=time_name)
        weekly_dhw.attrs["accumulation_resolution"] = "native_daily_before_weekly_resample"
        data_vars[f"{name}_dhw_{window_weeks}w"] = weekly_dhw.rename(f"{name}_dhw_{window_weeks}w")
    return xr.Dataset(data_vars)
