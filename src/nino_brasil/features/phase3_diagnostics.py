from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from nino_brasil.data.zarr_store import dataframe_to_zarr
from nino_brasil.features.nino34_event_table import build_nino34_event_table, export_nino34_event_tables


DAILY_OCEAN_SOURCE_PRIORITY = {
    "noaa_ufs": 1,
    "glorys12": 2,
    "glorys12_operational": 3,
}

PHASE3_DAILY_TENDENCY_COLUMNS = [
    "nino34_ssta",
    "d20_nino34_mean_m",
    "ohc_0_100_nino34_j_m2",
    "ohc_0_300_nino34_j_m2",
    "ohc_0_700_nino34_j_m2",
    "ohc_300_700_nino34_j_m2",
    "wwv_equatorial_pacific_m3",
    "thermocline_tilt_m",
    "thermocline_tilt_slope_m_per_degree",
    "ssh_nino34_mean_m",
    "sss_nino34_mean",
]


@dataclass(frozen=True)
class Phase3DiagnosticsOutput:
    physical_signal_csv_path: Path
    physical_signal_zarr_path: Path
    thermocline_zarr_path: Path
    peak_signal_zarr_path: Path
    signal_slope_duration_zarr_path: Path
    event_table_csv_path: Path
    peak_comparison_csv_path: Path
    physics_precalc_csv_path: Path
    rows_daily: int
    rows_monthly: int
    first_date: str
    last_date: str


def _discover_source_paths(ocean_daily_root: Path, source: str) -> list[Path]:
    root = ocean_daily_root / source
    if not root.exists():
        return []
    return sorted(path for path in root.glob("*/*.zarr") if path.is_dir())


def discover_daily_ocean_feature_paths(
    ocean_daily_root: Path,
    *,
    sources: Iterable[str] = DAILY_OCEAN_SOURCE_PRIORITY.keys(),
) -> dict[str, list[Path]]:
    """Return annual/tail daily-ocean feature stores grouped by source."""
    return {source: _discover_source_paths(ocean_daily_root, source) for source in sources}


def _frame_from_ocean_feature_store(path: Path, *, source: str) -> pd.DataFrame:
    ds = xr.open_zarr(path)
    try:
        if "time" not in ds.coords and "time" not in ds.dims:
            raise KeyError(f"{path} does not contain a time coordinate.")
        frame = pd.DataFrame({"time": pd.DatetimeIndex(ds["time"].values)})
        for variable in ds.data_vars:
            array = ds[variable]
            if tuple(array.dims) == ("time",):
                frame[variable] = array.values
        frame["ocean_feature_source"] = source
        frame["ocean_feature_store"] = str(path)
        frame["ocean_source_priority"] = DAILY_OCEAN_SOURCE_PRIORITY.get(source, 0)
        return frame
    finally:
        ds.close()


def load_daily_ocean_features(
    ocean_daily_root: Path,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load the source-prioritized daily ocean scalar feature series for Phase 3."""
    frames: list[pd.DataFrame] = []
    for source, paths in discover_daily_ocean_feature_paths(ocean_daily_root).items():
        for path in paths:
            frames.append(_frame_from_ocean_feature_store(path, source=source))
    if not frames:
        raise FileNotFoundError(f"No daily ocean feature Zarr stores were found under {ocean_daily_root}.")

    ocean = pd.concat(frames, ignore_index=True)
    ocean["time"] = pd.to_datetime(ocean["time"])
    if start_date is not None:
        ocean = ocean[ocean["time"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        ocean = ocean[ocean["time"] <= pd.Timestamp(end_date)]

    ocean = (
        ocean.sort_values(["time", "ocean_source_priority"])
        .drop_duplicates(subset=["time"], keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )
    if ocean.empty:
        raise ValueError("Daily ocean feature selection is empty after date filtering.")
    return ocean


def _read_daily_nino34_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Nino 3.4 daily OISST CSV not found: {path}")
    frame = pd.read_csv(path, parse_dates=["time"])
    if "time" not in frame or "nino34_ssta" not in frame:
        raise KeyError(f"{path} must contain 'time' and 'nino34_ssta'.")
    return frame.sort_values("time").reset_index(drop=True)


def _read_monthly_nino34_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["month_start", "nino34_anom_c", "nino34_anom_3mo_mean_c"])
    frame = pd.read_csv(path, parse_dates=["time"])
    if "time" not in frame:
        raise KeyError(f"{path} must contain 'time'.")
    frame["month_start"] = frame["time"].dt.to_period("M").dt.to_timestamp()
    columns = [
        column
        for column in ["month_start", "nino34_anom_c", "nino34_anom_3mo_mean_c", "source", "source_url"]
        if column in frame.columns
    ]
    return frame[columns].drop_duplicates("month_start").reset_index(drop=True)


def _run_duration(mask: pd.Series) -> pd.Series:
    mask = mask.fillna(False).astype(bool)
    run_id = (mask != mask.shift(fill_value=False)).cumsum()
    duration = mask.groupby(run_id).cumcount() + 1
    return duration.where(mask, 0).astype(int)


def _add_daily_tendencies_and_durations(
    frame: pd.DataFrame,
    *,
    tendency_windows_days: Sequence[int],
    event_threshold_c: float,
) -> pd.DataFrame:
    out = frame.sort_values("time").reset_index(drop=True).copy()
    if "nino34_ssta" in out:
        out[f"nino34_ssta_duration_ge_{str(event_threshold_c).replace('.', 'p')}c_days"] = _run_duration(
            pd.to_numeric(out["nino34_ssta"], errors="coerce") >= event_threshold_c
        )
        p90_threshold = float(np.nanpercentile(pd.to_numeric(out["nino34_ssta"], errors="coerce").dropna(), 90.0))
        out["nino34_ssta_daily_p90_threshold_c"] = p90_threshold
        out["nino34_ssta_above_daily_p90"] = pd.to_numeric(out["nino34_ssta"], errors="coerce") >= p90_threshold
        out["nino34_ssta_duration_ge_daily_p90_days"] = _run_duration(out["nino34_ssta_above_daily_p90"])

    for column in PHASE3_DAILY_TENDENCY_COLUMNS:
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        for window in tendency_windows_days:
            out[f"{column}_delta_{window}d"] = values - values.shift(int(window))
            out[f"{column}_mean_{window}d"] = values.rolling(int(window), min_periods=1).mean()
    return out


def _merge_daily_oisst_ocean_monthly(
    *,
    daily_nino34_csv: Path,
    ocean_daily_root: Path,
    monthly_nino34_csv: Path | None,
    start_date: str | pd.Timestamp | None,
    end_date: str | pd.Timestamp | None,
    tendency_windows_days: Sequence[int],
    event_threshold_c: float,
) -> pd.DataFrame:
    daily = _read_daily_nino34_csv(daily_nino34_csv)
    ocean = load_daily_ocean_features(ocean_daily_root, start_date=start_date, end_date=end_date)

    merged = pd.merge(daily, ocean, on="time", how="inner", suffixes=("", "_ocean"))
    if start_date is not None:
        merged = merged[merged["time"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        merged = merged[merged["time"] <= pd.Timestamp(end_date)]
    merged = merged.sort_values("time").reset_index(drop=True)
    if merged.empty:
        raise ValueError("No overlapping daily records between OISST Nino 3.4 and daily ocean features.")

    merged["month_start"] = merged["time"].dt.to_period("M").dt.to_timestamp()
    monthly = _read_monthly_nino34_csv(monthly_nino34_csv)
    if not monthly.empty:
        merged = pd.merge(merged, monthly, on="month_start", how="left", suffixes=("", "_monthly"))

    merged["year"] = merged["time"].dt.year
    merged["month"] = merged["time"].dt.month
    merged["day"] = merged["time"].dt.day
    return _add_daily_tendencies_and_durations(
        merged,
        tendency_windows_days=tendency_windows_days,
        event_threshold_c=event_threshold_c,
    )


def _dataarray_from_frame(frame: pd.DataFrame, column: str) -> xr.DataArray:
    if column not in frame.columns:
        raise KeyError(f"Phase 3 physical signal is missing required column {column!r}.")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return xr.DataArray(
        values,
        coords={"time": pd.DatetimeIndex(frame["time"])},
        dims=("time",),
        name=column,
    )


def _build_event_and_peak_tables(
    physical: pd.DataFrame,
    *,
    monthly_nino34_csv: Path | None,
    peak_years: Iterable[int],
    event_threshold_c: float,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    required = [
        "nino34_ssta",
        "d20_nino34_mean_m",
        "ohc_0_300_nino34_j_m2",
        "ohc_0_700_nino34_j_m2",
    ]
    clean = physical.dropna(subset=[column for column in required if column in physical.columns]).copy()
    if clean.empty:
        raise ValueError("No complete daily rows are available for monthly Nino 3.4 event diagnostics.")

    d20 = _dataarray_from_frame(clean, "d20_nino34_mean_m")
    table = build_nino34_event_table(
        _dataarray_from_frame(clean, "nino34_ssta"),
        d20,
        _dataarray_from_frame(clean, "ohc_0_300_nino34_j_m2"),
        _dataarray_from_frame(clean, "ohc_0_700_nino34_j_m2"),
        d20,
        peak_years=peak_years,
        event_threshold_c=event_threshold_c,
    )
    table["month_start"] = pd.to_datetime(table["time"]).dt.to_period("M").dt.to_timestamp()
    monthly_nino34 = _read_monthly_nino34_csv(monthly_nino34_csv)
    if not monthly_nino34.empty:
        table = pd.merge(table, monthly_nino34, on="month_start", how="left", suffixes=("", "_monthly"))

    monthly_path, peak_path = export_nino34_event_tables(table, output_dir)
    peaks = table[table["is_peak_year"]].copy().reset_index(drop=True)
    return table, peaks, monthly_path, peak_path


def _thermocline_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "time",
        "year",
        "month",
        "day",
        "ocean_feature_source",
        "ocean_source_code",
        "d20_nino34_mean_m",
        "thermocline_tilt_m",
        "thermocline_tilt_slope_m_per_degree",
        "wwv_equatorial_pacific_m3",
        "ohc_0_100_nino34_j_m2",
        "ohc_0_300_nino34_j_m2",
        "ohc_0_700_nino34_j_m2",
        "ohc_300_700_nino34_j_m2",
        "ssh_nino34_mean_m",
        "sss_nino34_mean",
    ]
    columns = [column for column in preferred if column in frame.columns]
    columns.extend(
        column
        for column in frame.columns
        if column.startswith("temperature_") and column.endswith("_nino34_c") and column not in columns
    )
    columns.extend(
        column
        for column in frame.columns
        if (
            column.startswith("d20_nino34_mean_m_")
            or column.startswith("ohc_")
            or column.startswith("wwv_equatorial_pacific_m3_")
            or column.startswith("thermocline_tilt")
        )
        and column not in columns
    )
    return columns


def build_phase3_diagnostics(
    *,
    daily_nino34_csv: Path,
    ocean_daily_root: Path,
    monthly_nino34_csv: Path | None,
    physical_signal_csv_path: Path,
    physical_signal_zarr_path: Path,
    thermocline_zarr_path: Path,
    peak_signal_zarr_path: Path,
    signal_slope_duration_zarr_path: Path,
    output_table_dir: Path,
    physics_precalc_csv_path: Path,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    peak_years: Iterable[int] = (1982, 1983, 1997, 1998, 2015, 2016, 2023, 2024),
    event_threshold_c: float = 0.5,
    tendency_windows_days: Sequence[int] = (7, 30, 90),
) -> Phase3DiagnosticsOutput:
    """Materialize Phase 3 Nino 3.4 physical diagnostics from local Phase 1/2 products."""
    physical = _merge_daily_oisst_ocean_monthly(
        daily_nino34_csv=daily_nino34_csv,
        ocean_daily_root=ocean_daily_root,
        monthly_nino34_csv=monthly_nino34_csv,
        start_date=start_date,
        end_date=end_date,
        tendency_windows_days=tendency_windows_days,
        event_threshold_c=event_threshold_c,
    )

    physical_signal_csv_path.parent.mkdir(parents=True, exist_ok=True)
    physics_precalc_csv_path.parent.mkdir(parents=True, exist_ok=True)
    physical.to_csv(physical_signal_csv_path, index=False)
    physical.to_csv(physics_precalc_csv_path, index=False)
    dataframe_to_zarr(
        physical,
        physical_signal_zarr_path,
        overwrite=True,
        attrs={
            "artifact": "nino34_physical_signal",
            "phase": "3",
            "source_contract": "local OISST daily Nino3.4 + daily ocean scalar diagnostics + OISST-derived monthly labels",
            "event_threshold_c": float(event_threshold_c),
        },
    )

    thermocline = physical[_thermocline_columns(physical)].copy()
    dataframe_to_zarr(
        thermocline,
        thermocline_zarr_path,
        overwrite=True,
        attrs={
            "artifact": "nino34_thermocline_diagnostics",
            "phase": "3",
            "temporal_resolution": "daily",
        },
    )

    monthly, peaks, monthly_path, peak_path = _build_event_and_peak_tables(
        physical,
        monthly_nino34_csv=monthly_nino34_csv,
        peak_years=peak_years,
        event_threshold_c=event_threshold_c,
        output_dir=output_table_dir,
    )
    dataframe_to_zarr(
        peaks,
        peak_signal_zarr_path,
        overwrite=True,
        attrs={
            "artifact": "nino34_peak_signal_comparison",
            "phase": "3",
            "peak_years": ",".join(str(year) for year in peak_years),
        },
    )

    slope_duration_columns = [
        column
        for column in [
            "time",
            "year",
            "month",
            "nino34_ssta_mean",
            "nino34_ssta_max",
            "nino34_ssta_slope_lon",
            "signal_duration_months",
            "event_phase",
            "is_peak_year",
            "d20_mean_m",
            "d20_anomaly_m",
            "ohc_300_anomaly",
            "ohc_700_anomaly",
            "nino34_anom_c",
            "nino34_anom_3mo_mean_c",
        ]
        if column in monthly.columns
    ]
    dataframe_to_zarr(
        monthly[slope_duration_columns].copy(),
        signal_slope_duration_zarr_path,
        overwrite=True,
        attrs={
            "artifact": "nino34_signal_slope_duration",
            "phase": "3",
            "event_threshold_c": float(event_threshold_c),
        },
    )

    return Phase3DiagnosticsOutput(
        physical_signal_csv_path=physical_signal_csv_path,
        physical_signal_zarr_path=physical_signal_zarr_path,
        thermocline_zarr_path=thermocline_zarr_path,
        peak_signal_zarr_path=peak_signal_zarr_path,
        signal_slope_duration_zarr_path=signal_slope_duration_zarr_path,
        event_table_csv_path=monthly_path,
        peak_comparison_csv_path=peak_path,
        physics_precalc_csv_path=physics_precalc_csv_path,
        rows_daily=int(len(physical)),
        rows_monthly=int(len(monthly)),
        first_date=str(pd.Timestamp(physical["time"].min()).date()),
        last_date=str(pd.Timestamp(physical["time"].max()).date()),
    )
