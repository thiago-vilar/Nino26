from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from nino_brasil.data.download_http import download_url
from nino_brasil.data.zarr_store import dataframe_to_zarr


NOAA_PSL_NINO34_ANOM_URL = "https://psl.noaa.gov/data/correlation/nina34.anom.data"
NOAA_PSL_NINO34_SOURCE = "NOAA PSL Nino 3.4 anomaly index, NOAA ERSST v6 from NCEI"


@dataclass(frozen=True)
class Nino34PeakThresholds:
    weak: float = 0.5
    moderate: float = 1.0
    strong: float = 1.5
    super: float = 2.0


@dataclass(frozen=True)
class NoaaPslNino34Output:
    raw_path: Path
    csv_path: Path
    zarr_path: Path
    peaks_csv_path: Path
    peaks_zarr_path: Path
    rows: int
    peaks: int


def classify_nino34_peak(
    value_c: float,
    thresholds: Nino34PeakThresholds = Nino34PeakThresholds(),
) -> str:
    """Classify Nino 3.4 anomaly intensity with ONI-style thresholds."""
    if pd.isna(value_c) or value_c < thresholds.weak:
        return "neutral"
    if value_c < thresholds.moderate:
        return "weak_el_nino"
    if value_c < thresholds.strong:
        return "moderate_el_nino"
    if value_c < thresholds.super:
        return "strong_el_nino"
    return "super_el_nino"


def parse_noaa_psl_nino34_anom(
    text: str,
    *,
    missing_value: float = -99.99,
) -> pd.DataFrame:
    """Parse NOAA PSL `nina34.anom.data` into a monthly tidy table."""
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 13 or not parts[0].isdigit():
            continue
        year = int(parts[0])
        for month, raw_value in enumerate(parts[1:], start=1):
            value = float(raw_value)
            if np.isclose(value, missing_value):
                value = np.nan
            rows.append(
                {
                    "time": pd.Timestamp(year=year, month=month, day=1),
                    "year": year,
                    "month": month,
                    "nino34_anom_c": value,
                    "source": NOAA_PSL_NINO34_SOURCE,
                    "source_url": NOAA_PSL_NINO34_ANOM_URL,
                }
            )
    if not rows:
        raise ValueError("No NOAA PSL Nino 3.4 monthly rows were parsed.")
    frame = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    frame["nino34_anom_3mo_mean_c"] = (
        frame["nino34_anom_c"].rolling(window=3, center=True, min_periods=3).mean()
    )
    return frame


def build_noaa_psl_nino34_peak_reference(
    index: pd.DataFrame,
    *,
    time_column: str = "time",
    monthly_column: str = "nino34_anom_c",
    intensity_column: str = "nino34_anom_3mo_mean_c",
    threshold_c: float = 0.5,
    min_duration_months: int = 5,
    thresholds: Nino34PeakThresholds = Nino34PeakThresholds(),
) -> pd.DataFrame:
    """Detect reference El Nino peaks from the NOAA PSL monthly index."""
    required = {time_column, monthly_column, intensity_column}
    missing = required.difference(index.columns)
    if missing:
        raise KeyError(f"index is missing required columns: {sorted(missing)}")
    if min_duration_months < 1:
        raise ValueError("min_duration_months must be >= 1.")

    frame = index.copy()
    frame[time_column] = pd.to_datetime(frame[time_column])
    frame = frame.sort_values(time_column).reset_index(drop=True)
    valid = frame.dropna(subset=[intensity_column]).copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "event_start",
                "event_end",
                "duration_months",
                "peak_time",
                "peak_ssta_c",
                "peak_monthly_ssta_c",
                "peak_class",
                "threshold_basis",
                "source",
                "source_url",
            ]
        )

    warm = valid[intensity_column] >= threshold_c
    run_id = (warm != warm.shift(fill_value=False)).cumsum()
    rows: list[dict[str, object]] = []
    for _, group_index in valid[warm].groupby(run_id[warm]).groups.items():
        event = valid.loc[group_index].copy()
        if len(event) < min_duration_months:
            continue
        peak_idx = event[intensity_column].idxmax()
        peak = frame.loc[peak_idx]
        start = pd.Timestamp(event[time_column].iloc[0])
        end = pd.Timestamp(event[time_column].iloc[-1])
        peak_time = pd.Timestamp(peak[time_column])
        peak_ssta = float(peak[intensity_column])
        suffix = f"{start.year}_{end.year}" if start.year != end.year else str(start.year)
        rows.append(
            {
                "event_id": f"el_nino_{suffix}",
                "event_start": start,
                "event_end": end,
                "duration_months": int(len(event)),
                "peak_time": peak_time,
                "peak_ssta_c": peak_ssta,
                "peak_monthly_ssta_c": float(peak[monthly_column]),
                "peak_class": classify_nino34_peak(peak_ssta, thresholds),
                "threshold_basis": f"{intensity_column}>={threshold_c} for {min_duration_months}+ months",
                "source": NOAA_PSL_NINO34_SOURCE,
                "source_url": NOAA_PSL_NINO34_ANOM_URL,
            }
        )
    return pd.DataFrame(rows)


def download_noaa_psl_nino34_anom(
    *,
    raw_path: Path,
    csv_path: Path,
    zarr_path: Path,
    peaks_csv_path: Path,
    peaks_zarr_path: Path,
    overwrite: bool = False,
    url: str = NOAA_PSL_NINO34_ANOM_URL,
) -> NoaaPslNino34Output:
    """Download NOAA PSL Nino 3.4 index and write monthly/peak products."""
    raw = download_url(url, raw_path, overwrite=overwrite, resume=False)
    text = raw.read_text(encoding="ascii", errors="replace")
    monthly = parse_noaa_psl_nino34_anom(text)
    peaks = build_noaa_psl_nino34_peak_reference(monthly)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    peaks_csv_path.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(csv_path, index=False)
    peaks.to_csv(peaks_csv_path, index=False)

    dataframe_to_zarr(monthly, zarr_path, overwrite=True, attrs={"artifact": "noaa_psl_nino34_anom"})
    dataframe_to_zarr(
        peaks,
        peaks_zarr_path,
        overwrite=True,
        attrs={"artifact": "noaa_psl_nino34_reference_peaks"},
    )
    return NoaaPslNino34Output(
        raw_path=raw,
        csv_path=csv_path,
        zarr_path=zarr_path,
        peaks_csv_path=peaks_csv_path,
        peaks_zarr_path=peaks_zarr_path,
        rows=int(len(monthly)),
        peaks=int(len(peaks)),
    )
