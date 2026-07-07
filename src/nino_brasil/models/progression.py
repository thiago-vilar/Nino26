from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class NinoPeakThresholds:
    weak: float = 0.5
    moderate: float = 1.0
    strong: float = 1.5
    super: float = 2.0


DEFAULT_NINO_THRESHOLDS = NinoPeakThresholds()


def classify_nino_peak(
    peak_ssta_c: float,
    thresholds: NinoPeakThresholds = DEFAULT_NINO_THRESHOLDS,
) -> str:
    """Classify a future Nino 3.4 peak by fixed SSTA intensity thresholds."""
    if pd.isna(peak_ssta_c) or peak_ssta_c < thresholds.weak:
        return "neutral"
    if peak_ssta_c < thresholds.moderate:
        return "weak_el_nino"
    if peak_ssta_c < thresholds.strong:
        return "moderate_el_nino"
    if peak_ssta_c < thresholds.super:
        return "strong_el_nino"
    return "super_el_nino"


def _month_delta(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def _numeric_feature_columns(
    frame: pd.DataFrame,
    *,
    time_column: str,
    excluded: set[str],
) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column == time_column or column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(str(column))
    return columns


def _history_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    windows: Sequence[int],
    suffix: str = "m",
) -> pd.DataFrame:
    """Create causal rolling and tendency features from past/current values."""
    out = pd.DataFrame(index=frame.index)
    for column in feature_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        out[column] = values
        for window in windows:
            out[f"{column}_mean_{window}{suffix}"] = values.rolling(window, min_periods=1).mean()
            out[f"{column}_delta_{window}{suffix}"] = values - values.shift(window)
    return out


def build_enso_peak_progression_table(
    monthly_features: pd.DataFrame,
    *,
    time_column: str = "time",
    ssta_column: str = "nino34_ssta_mean",
    lead_months: Sequence[int] = (1, 3, 6, 9, 12),
    feature_columns: Sequence[str] | None = None,
    history_windows: Sequence[int] = (3, 6, 12),
    thresholds: NinoPeakThresholds = DEFAULT_NINO_THRESHOLDS,
) -> pd.DataFrame:
    """Build an event-centered matrix for learning progression to Nino peaks.

    Each row is an origin month and lead horizon. Targets describe the strongest
    Nino 3.4 peak observed from the origin through that horizon.
    """
    required = {time_column, ssta_column}
    missing = required.difference(monthly_features.columns)
    if missing:
        raise KeyError(f"monthly_features is missing required columns: {sorted(missing)}")
    if not lead_months:
        raise ValueError("lead_months must contain at least one horizon.")

    frame = monthly_features.copy()
    frame[time_column] = pd.to_datetime(frame[time_column])
    frame = frame.sort_values(time_column).reset_index(drop=True)
    frame[ssta_column] = pd.to_numeric(frame[ssta_column], errors="coerce")
    selected_features = list(
        feature_columns
        or _numeric_feature_columns(
            frame,
            time_column=time_column,
            excluded={"year", "month", "is_peak_year"},
        )
    )
    history = _history_feature_frame(frame, feature_columns=selected_features, windows=history_windows)

    rows: list[dict[str, object]] = []
    for origin_idx, origin_row in frame.iterrows():
        origin_time = pd.Timestamp(origin_row[time_column])
        feature_values = history.loc[origin_idx].to_dict()
        for horizon in lead_months:
            if horizon < 0:
                raise ValueError("lead_months must be non-negative.")
            horizon_end = origin_time + pd.DateOffset(months=int(horizon))
            window = frame[(frame[time_column] >= origin_time) & (frame[time_column] <= horizon_end)]
            valid_window = window.dropna(subset=[ssta_column])
            if valid_window.empty:
                continue
            peak_idx = valid_window[ssta_column].idxmax()
            peak_row = frame.loc[peak_idx]
            peak_time = pd.Timestamp(peak_row[time_column])
            peak_ssta = float(peak_row[ssta_column])
            rows.append(
                {
                    "origin_time": origin_time,
                    "horizon_months": int(horizon),
                    "target_peak_time": peak_time,
                    "months_to_peak": _month_delta(origin_time, peak_time),
                    "future_peak_ssta": peak_ssta,
                    "future_peak_class": classify_nino_peak(peak_ssta, thresholds),
                    "will_el_nino": bool(peak_ssta >= thresholds.weak),
                    "will_strong_el_nino": bool(peak_ssta >= thresholds.strong),
                    "will_super_el_nino": bool(peak_ssta >= thresholds.super),
                    **feature_values,
                }
            )
    return pd.DataFrame(rows)


def build_daily_enso_peak_progression_table(
    daily_features: pd.DataFrame,
    reference_peaks: pd.DataFrame,
    *,
    time_column: str = "time",
    daily_ssta_column: str = "nino34_ssta",
    lead_days: Sequence[int] = (30, 60, 90, 120, 180, 270, 365),
    feature_columns: Sequence[str] | None = None,
    history_windows_days: Sequence[int] = (7, 30, 90, 180),
    peak_time_column: str = "peak_time",
    peak_ssta_column: str = "peak_ssta_c",
    peak_class_column: str = "peak_class",
    thresholds: NinoPeakThresholds = DEFAULT_NINO_THRESHOLDS,
) -> pd.DataFrame:
    """Build daily ENSO progression rows using daily OISST/SSTA features.

    NOAA monthly peak references should supply labels; daily features carry the
    fast trajectory that the model learns.
    """
    if time_column not in daily_features.columns:
        raise KeyError(f"daily_features is missing {time_column!r}.")
    if daily_ssta_column not in daily_features.columns:
        raise KeyError(f"daily_features is missing {daily_ssta_column!r}.")
    if not lead_days:
        raise ValueError("lead_days must contain at least one horizon.")

    frame = daily_features.copy()
    frame[time_column] = pd.to_datetime(frame[time_column])
    frame = frame.sort_values(time_column).reset_index(drop=True)
    frame[daily_ssta_column] = pd.to_numeric(frame[daily_ssta_column], errors="coerce")
    selected_features = list(
        feature_columns
        or _numeric_feature_columns(frame, time_column=time_column, excluded={"year", "month", "day"})
    )
    history = _history_feature_frame(
        frame,
        feature_columns=selected_features,
        windows=history_windows_days,
        suffix="d",
    )

    peaks = reference_peaks.copy()
    if peaks.empty:
        peaks = pd.DataFrame(columns=[peak_time_column, peak_ssta_column, peak_class_column])
    for column in (peak_time_column, peak_ssta_column, peak_class_column):
        if column not in peaks.columns:
            peaks[column] = np.nan
    peaks[peak_time_column] = pd.to_datetime(peaks[peak_time_column])
    peaks[peak_ssta_column] = pd.to_numeric(peaks[peak_ssta_column], errors="coerce")
    peaks = peaks.dropna(subset=[peak_time_column]).sort_values(peak_time_column)

    time_index = pd.DatetimeIndex(frame[time_column])
    time_ns = time_index.to_numpy(dtype="datetime64[ns]").astype("int64")
    ssta_values = frame[daily_ssta_column].to_numpy(dtype=float)
    history_records = history.to_dict(orient="records")
    peak_time_index = pd.DatetimeIndex(peaks[peak_time_column]) if not peaks.empty else pd.DatetimeIndex([])
    peak_time_ns = peak_time_index.to_numpy(dtype="datetime64[ns]").astype("int64")
    peak_ssta_values = peaks[peak_ssta_column].to_numpy(dtype=float) if not peaks.empty else np.array([], dtype=float)
    peak_classes = peaks[peak_class_column].astype(str).to_numpy() if not peaks.empty else np.array([], dtype=str)

    rows: list[dict[str, object]] = []
    for origin_idx, origin_time in enumerate(time_index):
        feature_values = history_records[origin_idx]
        for horizon in lead_days:
            if horizon < 0:
                raise ValueError("lead_days must be non-negative.")
            horizon_end = origin_time + pd.Timedelta(days=int(horizon))
            horizon_end_ns = horizon_end.value
            daily_end_idx = int(np.searchsorted(time_ns, horizon_end_ns, side="right"))
            if daily_end_idx <= origin_idx:
                continue
            daily_segment = ssta_values[origin_idx:daily_end_idx]
            if daily_segment.size == 0 or np.isnan(daily_segment).all():
                continue
            daily_peak_idx = origin_idx + int(np.nanargmax(daily_segment))

            peak_start_idx = int(np.searchsorted(peak_time_ns, origin_time.value, side="left"))
            peak_end_idx = int(np.searchsorted(peak_time_ns, horizon_end_ns, side="right"))
            has_reference_peak = peak_end_idx > peak_start_idx
            if has_reference_peak:
                candidate_ssta = peak_ssta_values[peak_start_idx:peak_end_idx]
                valid = ~np.isnan(candidate_ssta)
                if valid.any():
                    local_candidates = np.where(valid)[0]
                    best_local = local_candidates[int(np.nanargmax(candidate_ssta[valid]))]
                    ref_idx = peak_start_idx + int(best_local)
                    target_peak_time = pd.Timestamp(peak_time_index[ref_idx])
                    future_peak_ssta = float(peak_ssta_values[ref_idx])
                    future_peak_class = str(peak_classes[ref_idx])
                    days_to_peak = int((target_peak_time - origin_time).days)
                else:
                    has_reference_peak = False
            if not has_reference_peak:
                target_peak_time = pd.NaT
                future_peak_ssta = np.nan
                future_peak_class = "neutral"
                days_to_peak = np.nan

            rows.append(
                {
                    "origin_time": origin_time,
                    "horizon_days": int(horizon),
                    "target_peak_time": target_peak_time,
                    "days_to_peak": days_to_peak,
                    "future_peak_ssta": future_peak_ssta,
                    "future_peak_class": future_peak_class,
                    "has_reference_peak": bool(has_reference_peak),
                    "will_el_nino": bool(has_reference_peak and future_peak_ssta >= thresholds.weak),
                    "will_strong_el_nino": bool(has_reference_peak and future_peak_ssta >= thresholds.strong),
                    "will_super_el_nino": bool(has_reference_peak and future_peak_ssta >= thresholds.super),
                    "future_daily_max_ssta": float(ssta_values[daily_peak_idx]),
                    "future_daily_max_time": pd.Timestamp(time_index[daily_peak_idx]),
                    **feature_values,
                }
            )
    return pd.DataFrame(rows)


def _spatial_dims(da: xr.DataArray, time_name: str) -> list[str]:
    return [dim for dim in da.dims if dim != time_name]


def _pixel_event_frame(
    pixel_events: xr.DataArray,
    *,
    time_name: str,
) -> tuple[pd.DataFrame, list[str]]:
    if time_name not in pixel_events.dims:
        raise KeyError(f"pixel_events must contain dimension {time_name!r}.")
    spatial_dims = _spatial_dims(pixel_events, time_name)
    if not spatial_dims:
        raise ValueError("pixel_events must contain at least one spatial dimension.")
    stacked = pixel_events.stack(pixel=spatial_dims).transpose(time_name, "pixel")
    frame = stacked.to_pandas()
    frame.index = pd.DatetimeIndex(frame.index, name=time_name)
    return frame, spatial_dims


def cluster_pixel_events(
    pixel_events: xr.DataArray,
    *,
    n_clusters: int,
    time_name: str = "time",
    random_state: int = 42,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Cluster Brazil pixels by their event time series."""
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1.")
    event_frame, spatial_dims = _pixel_event_frame(pixel_events, time_name=time_name)
    pixel_matrix = event_frame.T.astype(float).replace([np.inf, -np.inf], np.nan).fillna(fill_value)
    if pixel_matrix.empty:
        raise ValueError("pixel_events has no pixels to cluster.")
    if n_clusters > len(pixel_matrix):
        raise ValueError("n_clusters cannot exceed the number of pixels.")

    scaled = StandardScaler().fit_transform(pixel_matrix)
    labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(scaled)

    pixel_index = pd.MultiIndex.from_tuples(pixel_matrix.index, names=spatial_dims)
    clusters = pixel_index.to_frame(index=False)
    clusters["cluster"] = labels.astype(int)
    clusters["event_frequency"] = pixel_matrix.mean(axis=1).to_numpy()
    clusters["event_mean"] = pixel_matrix.mean(axis=1).to_numpy()
    clusters["event_std"] = pixel_matrix.std(axis=1).to_numpy()
    return clusters.sort_values(["cluster", *spatial_dims]).reset_index(drop=True)


def build_nino34_cluster_progression_table(
    nino34_features: pd.DataFrame,
    pixel_events: xr.DataArray,
    pixel_clusters: pd.DataFrame,
    *,
    time_column: str = "time",
    time_name: str = "time",
    lags_days: Sequence[int] = (30, 60, 90, 120, 150, 180),
    feature_columns: Sequence[str] | None = None,
    history_windows: Sequence[int] = (3, 6, 12),
    cluster_column: str = "cluster",
    target_name: str = "target_event_rate",
) -> pd.DataFrame:
    """Build Nino3.4 -> Brazil pixel-cluster event progression targets."""
    if time_column not in nino34_features.columns:
        raise KeyError(f"nino34_features is missing {time_column!r}.")
    if cluster_column not in pixel_clusters.columns:
        raise KeyError(f"pixel_clusters is missing {cluster_column!r}.")
    if not lags_days:
        raise ValueError("lags_days must contain at least one horizon.")

    event_frame, spatial_dims = _pixel_event_frame(pixel_events, time_name=time_name)
    required_cluster_columns = set(spatial_dims)
    missing = required_cluster_columns.difference(pixel_clusters.columns)
    if missing:
        raise KeyError(f"pixel_clusters is missing spatial columns: {sorted(missing)}")

    cluster_lookup = pixel_clusters.set_index(spatial_dims)[cluster_column].to_dict()
    pixel_clusters_for_frame = []
    for pixel in event_frame.columns:
        key = pixel if isinstance(pixel, tuple) else (pixel,)
        pixel_clusters_for_frame.append(cluster_lookup[key])
    cluster_labels = pd.Series(pixel_clusters_for_frame, index=event_frame.columns)
    cluster_event_rates = event_frame.T.groupby(cluster_labels).mean().T.sort_index(axis=1)

    frame = nino34_features.copy()
    frame[time_column] = pd.to_datetime(frame[time_column])
    frame = frame.sort_values(time_column).reset_index(drop=True)
    selected_features = list(
        feature_columns
        or _numeric_feature_columns(
            frame,
            time_column=time_column,
            excluded={"year", "month", "is_peak_year"},
        )
    )
    history = _history_feature_frame(frame, feature_columns=selected_features, windows=history_windows)

    rows: list[dict[str, object]] = []
    for origin_idx, origin_row in frame.iterrows():
        origin_time = pd.Timestamp(origin_row[time_column])
        feature_values = history.loc[origin_idx].to_dict()
        for lag in lags_days:
            if lag < 0:
                raise ValueError("lags_days must be non-negative.")
            target_time = origin_time + pd.Timedelta(days=int(lag))
            if target_time not in cluster_event_rates.index:
                continue
            for cluster_id, value in cluster_event_rates.loc[target_time].items():
                rows.append(
                    {
                        "origin_time": origin_time,
                        "target_time": target_time,
                        "lag_days": int(lag),
                        "cluster": int(cluster_id),
                        target_name: float(value),
                        **feature_values,
                    }
                )
    return pd.DataFrame(rows)
