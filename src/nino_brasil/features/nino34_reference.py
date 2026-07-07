from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nino_brasil.data.zarr_store import dataframe_to_zarr


OISST_NINO34_SOURCE = "NOAA OISST v2.1 daily SST, local NINO-BRASIL processing"


@dataclass(frozen=True)
class Nino34PeakThresholds:
    weak: float = 0.5
    moderate: float = 1.0
    strong: float = 1.5
    super: float = 2.0


@dataclass(frozen=True)
class Nino34SstReferenceOutput:
    csv_path: Path
    zarr_path: Path
    peaks_csv_path: Path
    peaks_zarr_path: Path
    rows: int
    peaks: int
    p90_peaks_csv_path: Path | None = None
    p90_peaks_zarr_path: Path | None = None
    p90_plot_path: Path | None = None
    p90_threshold_c: float | None = None
    p90_peaks: int | None = None


@dataclass(frozen=True)
class Nino34SstP90Output:
    peaks_csv_path: Path
    peaks_zarr_path: Path
    plot_path: Path
    percentile: float
    threshold_c: float
    peaks: int


def classify_nino34_peak(
    value_c: float,
    thresholds: Nino34PeakThresholds = Nino34PeakThresholds(),
) -> str:
    """Classify Nino 3.4 anomaly intensity using fixed SST-anomaly thresholds."""
    if pd.isna(value_c) or value_c < thresholds.weak:
        return "neutral"
    if value_c < thresholds.moderate:
        return "weak_el_nino"
    if value_c < thresholds.strong:
        return "moderate_el_nino"
    if value_c < thresholds.super:
        return "strong_el_nino"
    return "super_el_nino"


def build_monthly_nino34_sst_reference(
    daily: pd.DataFrame,
    *,
    time_column: str = "time",
    sst_column: str = "nino34_sst",
    ssta_column: str = "nino34_ssta",
    source: str = OISST_NINO34_SOURCE,
) -> pd.DataFrame:
    """Build a monthly Nino 3.4 reference table from local daily OISST output."""
    required = {time_column, sst_column, ssta_column}
    missing = required.difference(daily.columns)
    if missing:
        raise KeyError(f"daily Nino 3.4 table is missing columns: {sorted(missing)}")

    frame = daily.copy()
    frame[time_column] = pd.to_datetime(frame[time_column])
    frame[sst_column] = pd.to_numeric(frame[sst_column], errors="coerce")
    frame[ssta_column] = pd.to_numeric(frame[ssta_column], errors="coerce")
    frame = frame.dropna(subset=[time_column]).sort_values(time_column)
    if frame.empty:
        raise ValueError("daily Nino 3.4 table is empty.")

    resampled = frame.set_index(time_column)[[sst_column, ssta_column]].resample("MS")
    monthly = (
        resampled.mean()
        .reset_index()
        .rename(columns={time_column: "time", sst_column: "nino34_sst_c", ssta_column: "nino34_ssta_c"})
    )
    monthly["valid_days"] = resampled[ssta_column].count().to_numpy(dtype=int)
    monthly["days_in_month"] = monthly["time"].dt.days_in_month
    monthly["month_complete"] = monthly["valid_days"] >= monthly["days_in_month"]
    monthly["year"] = monthly["time"].dt.year
    monthly["month"] = monthly["time"].dt.month
    monthly["nino34_ssta_3mo_mean_c"] = monthly["nino34_ssta_c"].rolling(
        window=3,
        center=True,
        min_periods=3,
    ).mean()
    # Backward-compatible aliases inside Phase 3 only. They are OISST-derived, not an external index.
    monthly["nino34_anom_c"] = monthly["nino34_ssta_c"]
    monthly["nino34_anom_3mo_mean_c"] = monthly["nino34_ssta_3mo_mean_c"]
    monthly["source"] = source
    monthly["source_url"] = ""
    return monthly[
        [
            "time",
            "year",
            "month",
            "valid_days",
            "days_in_month",
            "month_complete",
            "nino34_sst_c",
            "nino34_ssta_c",
            "nino34_ssta_3mo_mean_c",
            "nino34_anom_c",
            "nino34_anom_3mo_mean_c",
            "source",
            "source_url",
        ]
    ]


def _filter_complete_months(index: pd.DataFrame) -> pd.DataFrame:
    if "month_complete" not in index.columns:
        return index.copy()
    complete = index["month_complete"]
    if complete.dtype == object:
        mask = complete.astype(str).str.lower().isin({"true", "1", "yes", "y"})
    else:
        mask = complete.fillna(False).astype(bool)
    return index[mask].copy()


def build_nino34_sst_peak_reference(
    index: pd.DataFrame,
    *,
    time_column: str = "time",
    monthly_column: str = "nino34_ssta_c",
    intensity_column: str = "nino34_ssta_3mo_mean_c",
    threshold_c: float = 0.5,
    min_duration_months: int = 5,
    thresholds: Nino34PeakThresholds = Nino34PeakThresholds(),
) -> pd.DataFrame:
    """Detect warm Nino 3.4 events from the local OISST-derived monthly index."""
    required = {time_column, monthly_column, intensity_column}
    missing = required.difference(index.columns)
    if missing:
        raise KeyError(f"index is missing required columns: {sorted(missing)}")
    if min_duration_months < 1:
        raise ValueError("min_duration_months must be >= 1.")

    frame = index.copy()
    frame[time_column] = pd.to_datetime(frame[time_column])
    frame = frame.sort_values(time_column).reset_index(drop=True)
    frame = _filter_complete_months(frame).reset_index(drop=True)
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
                "event_id": f"el_nino_oisst_{suffix}",
                "event_start": start,
                "event_end": end,
                "duration_months": int(len(event)),
                "peak_time": peak_time,
                "peak_ssta_c": peak_ssta,
                "peak_monthly_ssta_c": float(peak[monthly_column]),
                "peak_class": classify_nino34_peak(peak_ssta, thresholds),
                "threshold_basis": f"{intensity_column}>={threshold_c} for {min_duration_months}+ months on local OISST",
                "source": str(peak.get("source", OISST_NINO34_SOURCE)),
                "source_url": str(peak.get("source_url", "")),
            }
        )
    return pd.DataFrame(rows)


def _percentile_label(percentile: float) -> str:
    return f"p{int(percentile)}" if float(percentile).is_integer() else f"p{percentile:g}".replace(".", "p")


def build_nino34_sst_p90_peaks(
    index: pd.DataFrame,
    *,
    time_column: str = "time",
    monthly_column: str = "nino34_ssta_c",
    percentile: float = 90.0,
) -> pd.DataFrame:
    """Detect local OISST-derived monthly Nino 3.4 anomaly peaks above a percentile."""
    required = {time_column, monthly_column}
    missing = required.difference(index.columns)
    if missing:
        raise KeyError(f"index is missing required columns: {sorted(missing)}")
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be between 0 and 100.")

    frame = index.copy()
    frame[time_column] = pd.to_datetime(frame[time_column])
    frame[monthly_column] = pd.to_numeric(frame[monthly_column], errors="coerce")
    frame = frame.sort_values(time_column).reset_index(drop=True)
    frame = _filter_complete_months(frame).reset_index(drop=True)
    valid = frame.dropna(subset=[monthly_column]).copy()
    if valid.empty:
        raise ValueError("No valid local monthly Nino 3.4 anomalies are available for percentile analysis.")

    threshold_c = float(np.nanpercentile(valid[monthly_column].to_numpy(dtype=float), percentile))
    above = valid[monthly_column] >= threshold_c
    run_id = (above != above.shift(fill_value=False)).cumsum()
    label = _percentile_label(percentile)

    rows: list[dict[str, object]] = []
    for _, group_index in valid[above].groupby(run_id[above]).groups.items():
        run = valid.loc[group_index].copy()
        peak_idx = run[monthly_column].idxmax()
        peak = frame.loc[peak_idx]
        start = pd.Timestamp(run[time_column].iloc[0])
        end = pd.Timestamp(run[time_column].iloc[-1])
        peak_time = pd.Timestamp(peak[time_column])
        rows.append(
            {
                "event_id": f"nino34_oisst_{label}_peak_{peak_time.year}_{peak_time.month:02d}",
                "event_start": start,
                "event_end": end,
                "duration_months": int(len(run)),
                "peak_time": peak_time,
                "peak_nino34_anom_c": float(peak[monthly_column]),
                "peak_rank": np.nan,
                "percentile": float(percentile),
                "percentile_threshold_c": threshold_c,
                "threshold_basis": f"{monthly_column}>={label} ({threshold_c:.3f} C) on local OISST monthly valid values",
                "source": str(peak.get("source", OISST_NINO34_SOURCE)),
                "source_url": str(peak.get("source_url", "")),
            }
        )

    peaks = pd.DataFrame(rows)
    if peaks.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "event_start",
                "event_end",
                "duration_months",
                "peak_time",
                "peak_nino34_anom_c",
                "peak_rank",
                "percentile",
                "percentile_threshold_c",
                "threshold_basis",
                "source",
                "source_url",
            ]
        )
    peaks["peak_rank"] = peaks["peak_nino34_anom_c"].rank(method="dense", ascending=False).astype(int)
    return peaks.sort_values("peak_time").reset_index(drop=True)


def save_nino34_sst_p90_plot(
    index: pd.DataFrame,
    peaks: pd.DataFrame,
    output_path: Path,
    *,
    time_column: str = "time",
    monthly_column: str = "nino34_ssta_c",
    percentile: float = 90.0,
    threshold_c: float | None = None,
) -> Path:
    """Save a simple local OISST Nino 3.4 anomaly chart highlighting P90 peaks."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    frame = index.copy()
    frame[time_column] = pd.to_datetime(frame[time_column])
    frame[monthly_column] = pd.to_numeric(frame[monthly_column], errors="coerce")
    frame = frame.dropna(subset=[monthly_column]).sort_values(time_column).reset_index(drop=True)
    frame = _filter_complete_months(frame).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid local monthly Nino 3.4 anomalies are available for plotting.")
    if threshold_c is None:
        threshold_c = float(np.nanpercentile(frame[monthly_column].to_numpy(dtype=float), percentile))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(frame[time_column], frame[monthly_column], color="#1f77b4", linewidth=1.0, label="Nino 3.4 OISST mensal")
    ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.8)
    ax.axhline(
        threshold_c,
        color="#d62728",
        linewidth=1.2,
        linestyle="--",
        label=f"P{percentile:g} = {threshold_c:.2f} C",
    )
    ax.fill_between(
        frame[time_column],
        threshold_c,
        frame[monthly_column],
        where=frame[monthly_column] >= threshold_c,
        color="#ff7f0e",
        alpha=0.18,
        interpolate=True,
        label=f"meses >= P{percentile:g}",
    )

    if not peaks.empty:
        peak_frame = peaks.copy()
        peak_frame["peak_time"] = pd.to_datetime(peak_frame["peak_time"])
        ax.scatter(
            peak_frame["peak_time"],
            peak_frame["peak_nino34_anom_c"],
            s=60,
            color="#d62728",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label=f"picos >= P{percentile:g}",
        )

    ax.set_title("Nino 3.4 OISST mensal - picos acima do P90")
    ax.set_ylabel("anomalia Nino 3.4 (C)")
    ax.set_xlabel("tempo")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def export_nino34_sst_p90_peak_analysis(
    index: pd.DataFrame,
    *,
    peaks_csv_path: Path,
    peaks_zarr_path: Path,
    plot_path: Path,
    percentile: float = 90.0,
) -> Nino34SstP90Output:
    """Write Phase 3 P90 CSV/Zarr/PNG artifacts from the local OISST monthly table."""
    peaks = build_nino34_sst_p90_peaks(index, percentile=percentile)
    complete_index = _filter_complete_months(index)
    threshold_c = (
        float(peaks["percentile_threshold_c"].iloc[0])
        if not peaks.empty
        else float(
            np.nanpercentile(
                pd.to_numeric(complete_index["nino34_ssta_c"], errors="coerce").dropna(),
                percentile,
            )
        )
    )

    peaks_csv_path.parent.mkdir(parents=True, exist_ok=True)
    peaks.to_csv(peaks_csv_path, index=False)
    dataframe_to_zarr(
        peaks,
        peaks_zarr_path,
        overwrite=True,
        attrs={
            "artifact": "nino34_oisst_p90_peaks",
            "percentile": float(percentile),
            "percentile_threshold_c": threshold_c,
            "source_contract": "local OISST-derived Nino 3.4 monthly SSTA",
        },
    )
    save_nino34_sst_p90_plot(
        index,
        peaks,
        plot_path,
        percentile=percentile,
        threshold_c=threshold_c,
    )
    return Nino34SstP90Output(
        peaks_csv_path=peaks_csv_path,
        peaks_zarr_path=peaks_zarr_path,
        plot_path=plot_path,
        percentile=float(percentile),
        threshold_c=threshold_c,
        peaks=int(len(peaks)),
    )


def export_nino34_sst_reference(
    daily: pd.DataFrame,
    *,
    csv_path: Path,
    zarr_path: Path,
    peaks_csv_path: Path,
    peaks_zarr_path: Path,
    p90_peaks_csv_path: Path | None = None,
    p90_peaks_zarr_path: Path | None = None,
    p90_plot_path: Path | None = None,
    p90_percentile: float = 90.0,
    event_threshold_c: float = 0.5,
    min_duration_months: int = 5,
) -> Nino34SstReferenceOutput:
    """Export monthly and event-reference products derived only from local OISST SST."""
    monthly = build_monthly_nino34_sst_reference(daily)
    peaks = build_nino34_sst_peak_reference(
        monthly,
        threshold_c=event_threshold_c,
        min_duration_months=min_duration_months,
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    peaks_csv_path.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(csv_path, index=False)
    peaks.to_csv(peaks_csv_path, index=False)
    dataframe_to_zarr(
        monthly,
        zarr_path,
        overwrite=True,
        attrs={
            "artifact": "nino34_monthly_oisst",
            "source_contract": "derived from local downloaded OISST daily SST",
        },
    )
    dataframe_to_zarr(
        peaks,
        peaks_zarr_path,
        overwrite=True,
        attrs={
            "artifact": "nino34_oisst_event_reference",
            "event_threshold_c": float(event_threshold_c),
            "min_duration_months": int(min_duration_months),
        },
    )

    p90_output: Nino34SstP90Output | None = None
    if p90_peaks_csv_path is not None and p90_peaks_zarr_path is not None and p90_plot_path is not None:
        p90_output = export_nino34_sst_p90_peak_analysis(
            monthly,
            peaks_csv_path=p90_peaks_csv_path,
            peaks_zarr_path=p90_peaks_zarr_path,
            plot_path=p90_plot_path,
            percentile=p90_percentile,
        )

    return Nino34SstReferenceOutput(
        csv_path=csv_path,
        zarr_path=zarr_path,
        peaks_csv_path=peaks_csv_path,
        peaks_zarr_path=peaks_zarr_path,
        rows=int(len(monthly)),
        peaks=int(len(peaks)),
        p90_peaks_csv_path=p90_output.peaks_csv_path if p90_output else None,
        p90_peaks_zarr_path=p90_output.peaks_zarr_path if p90_output else None,
        p90_plot_path=p90_output.plot_path if p90_output else None,
        p90_threshold_c=p90_output.threshold_c if p90_output else None,
        p90_peaks=p90_output.peaks if p90_output else None,
    )
