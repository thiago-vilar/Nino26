"""Native-grid CHIRPS targets for the Brazil teleconnection experiments.

This module is the single target contract for Phases 4, 6 and 8.  It never
interpolates CHIRPS.  Annual daily stores are first concatenated on their
original coordinates and are resampled exactly once afterwards.  A regular
rectangular native grid is retained for convolutional models; ``brazil_fraction``
and ``brazil_center`` explicitly identify the cells on which scientific losses,
statistics and scores may be calculated.

The primary response variables are weekly precipitation, anomaly in mm,
robust standardized anomaly and a robust normal-score percentile.  Gamma SPI
at 1/3/6-month approximations and weekly diagnostic versions of common ETCCDI
indices are included with explicit method/status attributes.  The ETCCDI spell
indices are deliberately named ``*_within_week`` because official CDD/CWD are
normally reported over longer periods; they must not be cited as official
annual ETCCDI indices.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy import special


BRAZIL_BBOX: Mapping[str, float] = {
    "lat_min": -35.0,
    "lat_max": 7.0,
    "lon_min": -75.0,
    "lon_max": -32.0,
}
CLIMATOLOGY_BASE = ("1991-01-01", "2020-12-31")
TARGET_CONTRACT_VERSION = "chirps-native-weekly-v4"
# Frozen namespace used by existing daily/block/grid lineage.  Grid identity is
# a property of the native coordinates and must not change when the statistical
# target method is versioned.  The historical value is retained intentionally
# so the already-audited native grid keeps hash 4422ba2d... across target v3+.
NATIVE_GRID_HASH_NAMESPACE = "chirps-native-weekly-v2"


@dataclass(frozen=True)
class TargetValidation:
    """Machine-readable result of the native-target contract validation."""

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    grid_hash: str
    n_time: int
    n_latitude: int
    n_longitude: int

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "grid_hash": self.grid_hash,
            "n_time": self.n_time,
            "n_latitude": self.n_latitude,
            "n_longitude": self.n_longitude,
        }


def _coordinate_name(obj: xr.DataArray | xr.Dataset, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in obj.coords or name in obj.dims:
            return name
    raise KeyError(f"None of the coordinate names {tuple(candidates)!r} is present.")


def canonicalize_chirps_daily(
    obj: xr.DataArray | xr.Dataset,
    *,
    variable: str = "precip",
    bbox: Mapping[str, float] | None = BRAZIL_BBOX,
) -> xr.DataArray:
    """Return daily CHIRPS with canonical names and untouched native cells.

    Selection is coordinate slicing only.  No call to ``interp``, ``reindex`` or
    a regridding package is made.  Negative values (CHIRPS missing sentinels
    that escaped CF decoding) are converted to missing values, never to zero.
    """

    if isinstance(obj, xr.Dataset):
        if variable not in obj:
            raise KeyError(f"CHIRPS dataset has no variable {variable!r}.")
        da = obj[variable]
    else:
        da = obj
    lat_name = _coordinate_name(da, ("latitude", "lat", "y"))
    lon_name = _coordinate_name(da, ("longitude", "lon", "x"))
    time_name = _coordinate_name(da, ("time", "date"))
    rename = {
        old: new
        for old, new in (
            (lat_name, "latitude"),
            (lon_name, "longitude"),
            (time_name, "time"),
        )
        if old != new
    }
    da = da.rename(rename).transpose("time", "latitude", "longitude")
    if bbox is not None:
        lat = da["latitude"].values
        lon = da["longitude"].values
        lat_slice = (
            slice(bbox["lat_min"], bbox["lat_max"])
            if lat[0] <= lat[-1]
            else slice(bbox["lat_max"], bbox["lat_min"])
        )
        lon_slice = (
            slice(bbox["lon_min"], bbox["lon_max"])
            if lon[0] <= lon[-1]
            else slice(bbox["lon_max"], bbox["lon_min"])
        )
        da = da.sel(latitude=lat_slice, longitude=lon_slice)
    if da.sizes["latitude"] < 2 or da.sizes["longitude"] < 2:
        raise ValueError("The CHIRPS spatial selection is empty or degenerate.")
    da = da.sortby("time")
    da = da.where(da >= 0)
    da.name = "precip_daily_mm"
    da.attrs.update(
        {
            "units": "mm day-1",
            "source": "CHIRPS v2.x native grid",
            "spatial_operation": "native coordinate slice; no interpolation",
        }
    )
    return da


def native_grid_hash(latitude: Iterable[float], longitude: Iterable[float]) -> str:
    """Stable SHA-256 fingerprint of ordered native latitude/longitude cells."""

    lat = np.asarray(list(latitude), dtype=np.float64)
    lon = np.asarray(list(longitude), dtype=np.float64)
    payload = (
        f"{NATIVE_GRID_HASH_NAMESPACE}|lat={lat.size}|lon={lon.size}|"
        + ",".join(f"{value:.8f}" for value in lat)
        + "|"
        + ",".join(f"{value:.8f}" for value in lon)
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _assert_regular_coordinate(values: np.ndarray, name: str) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError(f"{name} must be a one-dimensional coordinate.")
    spacing = np.diff(values)
    if not np.all(np.isfinite(spacing)) or np.any(spacing == 0):
        raise ValueError(f"{name} has duplicate or non-finite coordinates.")
    if not (np.all(spacing > 0) or np.all(spacing < 0)):
        raise ValueError(f"{name} is not monotonic.")
    step = float(np.median(spacing))
    if not np.allclose(spacing, step, rtol=0.0, atol=1e-7):
        raise ValueError(f"{name} is not a regular native grid.")
    return step


def _assert_same_grid(reference: xr.DataArray, candidate: xr.DataArray) -> None:
    for coordinate in ("latitude", "longitude"):
        left = np.asarray(reference[coordinate].values)
        right = np.asarray(candidate[coordinate].values)
        if left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(
                f"CHIRPS grid changed between stores at {coordinate}; refusing "
                "implicit interpolation or an ambiguous concatenation."
            )


def concat_daily_native(arrays: Sequence[xr.DataArray]) -> xr.DataArray:
    """Concatenate all daily stores before any weekly aggregation.

    Duplicate dates are rejected rather than averaged.  This makes an overlap
    between annual inputs visible and prevents the old year-boundary behaviour
    where two partial weekly sums were averaged together.
    """

    if not arrays:
        raise ValueError("At least one CHIRPS daily array is required.")
    canonical = [canonicalize_chirps_daily(array, bbox=None) for array in arrays]
    reference = canonical[0]
    _assert_regular_coordinate(reference["latitude"].values, "latitude")
    _assert_regular_coordinate(reference["longitude"].values, "longitude")
    for candidate in canonical[1:]:
        _assert_same_grid(reference, candidate)
    daily = xr.concat(
        canonical,
        dim="time",
        coords="minimal",
        compat="override",
        join="exact",
    ).sortby("time")
    index = pd.DatetimeIndex(daily["time"].values)
    duplicates = index[index.duplicated(keep=False)]
    if len(duplicates):
        examples = ", ".join(str(ts.date()) for ts in duplicates[:5])
        raise ValueError(f"Duplicate CHIRPS daily timestamps: {examples}.")
    if not index.is_monotonic_increasing:
        raise ValueError("CHIRPS daily timestamps are not chronological.")
    daily.attrs.update(reference.attrs)
    daily.attrs["temporal_operation"] = (
        "annual native daily stores concatenated first; unique daily timestamps"
    )
    return daily


def open_concat_chirps_stores(
    stores: Sequence[str | Path],
    *,
    bbox: Mapping[str, float] = BRAZIL_BBOX,
    variable: str = "precip",
) -> xr.DataArray:
    """Open annual Zarr stores lazily and concatenate their native daily cells."""

    opened: list[xr.DataArray] = []
    for store in stores:
        path = Path(store)
        dataset = xr.open_zarr(path, consolidated=None)
        opened.append(canonicalize_chirps_daily(dataset, variable=variable, bbox=bbox))
    return concat_daily_native(opened)


def weekly_native_precipitation(
    daily: xr.DataArray,
    *,
    minimum_valid_days: int = 7,
) -> xr.Dataset:
    """Aggregate daily native pixels exactly once to W-SUN weekly totals.

    ``valid_day_count`` is pixel-specific. ``expected_day_count`` is the number
    of calendar days represented by the input at each edge.  Primary weekly
    precipitation is retained only for complete seven-day weeks by default;
    incomplete edge weeks remain in the dataset, are flagged and contain NaN.
    """

    if not 1 <= minimum_valid_days <= 7:
        raise ValueError("minimum_valid_days must lie in [1, 7].")
    daily = canonicalize_chirps_daily(daily, bbox=None)
    index = pd.DatetimeIndex(daily["time"].values)
    if index.has_duplicates:
        raise ValueError("Daily timestamps must be unique before resampling.")
    if len(index) > 1:
        delta = np.diff(index.values).astype("timedelta64[D]")
        if np.any(delta != np.timedelta64(1, "D")):
            # Gaps are allowed because valid_day_count exposes them, but they
            # must remain visible in metadata and never be silently filled.
            has_daily_gaps = True
        else:
            has_daily_gaps = False
    else:
        has_daily_gaps = False

    valid_day_count = daily.notnull().resample(time="W-SUN").sum().astype("uint8")
    expected_day_count = xr.DataArray(
        np.ones(len(index), dtype="uint8"),
        coords={"time": daily["time"]},
        dims=("time",),
    ).resample(time="W-SUN").sum().astype("uint8")
    raw_sum = daily.resample(time="W-SUN").sum(skipna=True, min_count=1)
    full_calendar_week = expected_day_count == 7
    valid_for_analysis = (valid_day_count >= minimum_valid_days) & full_calendar_week
    weekly = raw_sum.where(valid_for_analysis).astype("float32")
    weekly.name = "precip_weekly_mm"
    weekly.attrs.update(
        {
            "units": "mm week-1",
            "aggregation": "sum of native daily CHIRPS pixels; W-SUN; one resample",
            "minimum_valid_days": minimum_valid_days,
            "partial_edge_weeks": "retained as coordinates, flagged, precipitation NaN",
        }
    )
    dataset = xr.Dataset(
        {
            "precip_weekly_mm": weekly,
            "valid_day_count": valid_day_count,
            "expected_day_count": expected_day_count,
            "week_complete": valid_for_analysis,
            "week_is_partial_edge": expected_day_count < 7,
        }
    )
    dataset.attrs.update(
        {
            "weekly_anchor": "W-SUN",
            "daily_gaps_present": bool(has_daily_gaps),
            "resampling_contract": "concatenate daily first, resample exactly once",
            "spatial_contract": "CHIRPS native grid; no interpolation",
        }
    )
    return dataset


def _iso_week(index: xr.DataArray) -> xr.DataArray:
    values = pd.DatetimeIndex(index.values).isocalendar().week.to_numpy(dtype=np.int16)
    return xr.DataArray(values, coords={"time": index}, dims=("time",), name="week_of_year")


def _rechunk_for_time_statistics(
    array: xr.DataArray,
    *,
    spatial_chunk: int = 24,
) -> xr.DataArray:
    """Bound memory while making the statistical time axis a single chunk.

    Dask quantile/median needs all reference years for a given pixel in one
    task.  Annual source chunks otherwise create a very large groupby graph and
    can exhaust memory even for a small coordinate slice.  Spatial chunks keep
    each full-time task around tens of MB on the canonical grid.
    """

    if array.chunks is None:
        return array
    chunks: dict[str, int] = {"time": -1}
    for dimension in array.dims:
        if dimension == "time":
            continue
        if dimension == "pixel":
            chunks[dimension] = min(1024, array.sizes[dimension])
        else:
            chunks[dimension] = min(spatial_chunk, array.sizes[dimension])
    return array.chunk(chunks)


def robust_weekly_anomalies(
    weekly: xr.DataArray,
    *,
    baseline: tuple[str, str] = CLIMATOLOGY_BASE,
    minimum_baseline_years: int = 20,
    absolute_scale_floor_mm: float = 0.10,
    quantity_units: str | None = None,
    zero_inflated_tail: bool = False,
    minimum_positive_weeks: int = 20,
    threshold_floor: xr.DataArray | None = None,
) -> xr.Dataset:
    """Create anomaly, robust z and audit parameters on the native quantity.

    ISO-week centres/scales require at least ``minimum_baseline_years``.  Week
    53, which cannot meet that count directly in a 30-year climatology, uses the
    explicitly flagged circular neighbour window 52/53/1.  Other insufficient
    centres remain undefined.  The denominator is the largest supported scale
    among seasonal MAD, pooled residual MAD and a pooled residual L1-equivalent
    scale.  The latter equals ``mean(abs(residual))*sqrt(pi/2)`` and is a
    data-derived lower bound for ordinary zero-inflated rainfall, where a
    seasonal MAD can be positive yet implausibly close to zero.

    ``R95p``/``R99p`` weekly sums are a more extreme hurdle distribution.  If
    ``zero_inflated_tail`` is true, their magnitude scale is fitted conditional
    on a positive weekly amount.  This prevents the occurrence probability
    from shrinking the denominator: for a zero-centred non-negative response,
    the unconditional L1 scale is exactly ``P(Y>0)`` times the conditional
    scale.  At least ``minimum_positive_weeks`` are required.  Otherwise the
    supplied, independently supported wet-day percentile is used as an
    explicitly labelled physical ``threshold_floor``.  No observation is
    imputed, clipped or spatially pooled.  Every candidate, support count,
    selected source and fallback code is retained for audit.
    """

    if absolute_scale_floor_mm <= 0:
        raise ValueError("Robust absolute scale floor must be positive.")
    if minimum_positive_weeks < 1:
        raise ValueError("minimum_positive_weeks must be positive.")
    if zero_inflated_tail and threshold_floor is None:
        raise ValueError(
            "zero_inflated_tail requires an auditable threshold_floor."
        )
    weekly = _rechunk_for_time_statistics(weekly.astype("float64")).assign_coords(
        week_of_year=_iso_week(weekly.time)
    )
    baseline_data = weekly.sel(time=slice(*baseline))
    if baseline_data.sizes.get("time", 0) < minimum_baseline_years * 40:
        raise ValueError("Insufficient weekly coverage in the climatology baseline.")

    centre = baseline_data.groupby("week_of_year").median("time", skipna=True)
    direct_n = baseline_data.groupby("week_of_year").count("time")
    effective_n = direct_n.copy()
    fallback_code = xr.zeros_like(direct_n, dtype="uint8")
    week_values = set(np.asarray(centre["week_of_year"].values, dtype=int).tolist())
    week53_source: xr.DataArray | None = None
    week53_centre: xr.DataArray | None = None
    if 53 in week_values:
        week53_source = baseline_data.where(
            baseline_data["week_of_year"].isin([52, 53, 1]), drop=True
        )
        week53_centre = week53_source.median("time", skipna=True)
        week53_n = week53_source.count("time")
        needs_week53_fallback = (
            (centre["week_of_year"] == 53)
            & (direct_n < minimum_baseline_years)
        )
        centre = xr.where(needs_week53_fallback, week53_centre, centre)
        effective_n = xr.where(needs_week53_fallback, week53_n, direct_n)
        fallback_code = xr.where(
            needs_week53_fallback & (week53_n >= minimum_baseline_years),
            1,
            fallback_code,
        )
    centre = centre.where(effective_n >= minimum_baseline_years)
    baseline_anomaly = baseline_data.groupby("week_of_year") - centre
    seasonal_mad = (
        np.abs(baseline_anomaly)
        .groupby("week_of_year")
        .median("time", skipna=True)
        * 1.4826
    )
    if week53_source is not None and week53_centre is not None:
        week53_mad = (
            np.abs(week53_source - week53_centre).median("time", skipna=True)
            * 1.4826
        )
        seasonal_mad = xr.where(
            (seasonal_mad["week_of_year"] == 53)
            & (direct_n < minimum_baseline_years),
            week53_mad,
            seasonal_mad,
        )
    n_baseline = direct_n
    # Pool residuals *after* removing each ISO-week centre.  Pooling raw rain
    # would reintroduce the seasonal cycle into the variance fallback.
    pooled_mad = np.abs(baseline_anomaly).median("time", skipna=True) * 1.4826
    pooled_l1 = (
        np.abs(baseline_anomaly).mean("time", skipna=True) * np.sqrt(np.pi / 2.0)
    )
    positive_week_count: xr.DataArray | None = None
    pooled_positive_l1: xr.DataArray | None = None
    audited_threshold_floor: xr.DataArray | None = None
    tail_fallback_code: xr.DataArray | None = None
    seasonal_candidate = seasonal_mad.where(
        (effective_n >= minimum_baseline_years)
        & np.isfinite(seasonal_mad)
        & (seasonal_mad >= absolute_scale_floor_mm)
    )
    pooled_mad_candidate = pooled_mad.where(
        np.isfinite(pooled_mad) & (pooled_mad >= absolute_scale_floor_mm)
    ).broadcast_like(seasonal_mad)
    pooled_l1_candidate = pooled_l1.where(
        np.isfinite(pooled_l1) & (pooled_l1 >= absolute_scale_floor_mm)
    ).broadcast_like(seasonal_mad)
    candidate_names = ["seasonal_mad", "pooled_mad", "pooled_l1"]
    candidate_arrays = [
        seasonal_candidate,
        pooled_mad_candidate,
        pooled_l1_candidate,
    ]
    if zero_inflated_tail:
        if threshold_floor is None:  # guarded above; narrows the type for mypy
            raise AssertionError("threshold_floor unexpectedly missing")
        positive_week_count = baseline_data.where(baseline_data > 0).count("time")
        pooled_positive_l1 = (
            np.abs(baseline_anomaly)
            .where(baseline_data > 0)
            .mean("time", skipna=True)
            * np.sqrt(np.pi / 2.0)
        )
        positive_supported = (
            (positive_week_count >= minimum_positive_weeks)
            & np.isfinite(pooled_positive_l1)
            & (pooled_positive_l1 >= absolute_scale_floor_mm)
        )
        positive_candidate = pooled_positive_l1.where(
            positive_supported
        ).broadcast_like(seasonal_mad)

        if set(threshold_floor.dims) != set(pooled_l1.dims):
            raise ValueError(
                "threshold_floor must have the exact non-time dimensions of the "
                f"weekly target; got {threshold_floor.dims}, expected {pooled_l1.dims}."
            )
        audited_threshold_floor = threshold_floor.transpose(*pooled_l1.dims).astype(
            "float64"
        )
        for dimension in pooled_l1.dims:
            if not np.array_equal(
                audited_threshold_floor[dimension].values,
                pooled_l1[dimension].values,
            ):
                raise ValueError(
                    f"threshold_floor coordinate mismatch on {dimension!r}."
                )
        threshold_supported = (
            (~positive_supported)
            & np.isfinite(audited_threshold_floor)
            & (audited_threshold_floor >= absolute_scale_floor_mm)
        )
        threshold_candidate = audited_threshold_floor.where(
            threshold_supported
        ).broadcast_like(seasonal_mad)
        tail_fallback_code = xr.full_like(positive_week_count, 2, dtype="uint8")
        tail_fallback_code = xr.where(
            positive_supported, 0, tail_fallback_code
        )
        tail_fallback_code = xr.where(
            threshold_supported, 1, tail_fallback_code
        )
        candidate_names.extend(["pooled_positive_l1", "threshold_floor"])
        candidate_arrays.extend([positive_candidate, threshold_candidate])

    candidates = xr.concat(
        candidate_arrays,
        dim=xr.IndexVariable(
            "scale_candidate", candidate_names
        ),
    )
    scale = candidates.max("scale_candidate", skipna=True).where(
        candidates.notnull().any("scale_candidate")
    )
    scale_source_code = xr.full_like(
        scale, len(candidate_arrays), dtype="uint8"
    )
    # Reverse traversal makes exact ties prefer the earlier, more local
    # candidate in ``candidate_names``.
    for source_code, candidate in reversed(list(enumerate(candidate_arrays))):
        scale_source_code = xr.where(
            scale.notnull() & (scale == candidate),
            source_code,
            scale_source_code,
        )
    anomaly = weekly.groupby("week_of_year") - centre
    robust_z = anomaly.groupby("week_of_year") / scale
    percentile = xr.apply_ufunc(
        special.ndtr,
        robust_z,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    anomaly = anomaly.astype("float32").rename("precip_anomaly_mm")
    robust_z = robust_z.astype("float32").rename("precip_robust_z")
    percentile = percentile.astype("float32").rename("precip_robust_percentile")
    centre = centre.astype("float32").rename("climatology_median_mm")
    scale = scale.astype("float32").rename("climatology_robust_scale_mm")
    pooled_mad = pooled_mad.astype("float32").rename(
        "climatology_pooled_residual_mad_scale"
    )
    pooled_l1 = pooled_l1.astype("float32").rename(
        "climatology_pooled_residual_l1_scale"
    )
    scale_source_code = scale_source_code.astype("uint8").rename(
        "climatology_scale_source_code"
    )
    n_baseline = n_baseline.astype("uint8").rename("climatology_sample_count")
    effective_n = effective_n.astype("uint16").rename(
        "climatology_effective_sample_count"
    )
    fallback_code = fallback_code.where(
        effective_n >= minimum_baseline_years, 2
    ).astype("uint8").rename("climatology_fallback_code")
    units = quantity_units or str(weekly.attrs.get("units", "native units"))
    tail_audit: dict[str, xr.DataArray] = {}
    if zero_inflated_tail:
        if any(
            value is None
            for value in (
                positive_week_count,
                pooled_positive_l1,
                audited_threshold_floor,
                tail_fallback_code,
            )
        ):
            raise AssertionError("zero-inflated tail audit layers are incomplete")
        positive_week_count = positive_week_count.astype("uint16").rename(
            "climatology_positive_week_count"
        )
        pooled_positive_l1 = pooled_positive_l1.astype("float32").rename(
            "climatology_pooled_positive_l1_scale"
        )
        audited_threshold_floor = audited_threshold_floor.astype("float32").rename(
            "climatology_tail_threshold_floor"
        )
        tail_fallback_code = tail_fallback_code.astype("uint8").rename(
            "climatology_tail_fallback_code"
        )
        positive_week_count.attrs.update(
            {
                "units": "weeks",
                "baseline": f"{baseline[0]}/{baseline[1]}",
                "definition": "baseline weeks with finite weekly tail amount > 0",
                "minimum_required_for_conditional_scale": minimum_positive_weeks,
            }
        )
        pooled_positive_l1.attrs.update(
            {
                "units": units,
                "baseline": f"{baseline[0]}/{baseline[1]}",
                "method": (
                    "sqrt(pi/2) * mean absolute ISO-week-centred residual "
                    "conditional on weekly tail amount > 0"
                ),
                "selection_support": (
                    f"used only with at least {minimum_positive_weeks} positive weeks"
                ),
            }
        )
        audited_threshold_floor.attrs.update(
            {
                "units": str(threshold_floor.attrs.get("units", units)),
                "baseline": f"{baseline[0]}/{baseline[1]}",
                "role": (
                    "physical wet-day percentile floor used only when the weekly "
                    "positive-magnitude sample is insufficient"
                ),
                "minimum_positive_weeks": minimum_positive_weeks,
            }
        )
        tail_fallback_code.attrs.update(
            {
                "units": "code",
                "codes": (
                    "0=conditional_positive_l1_supported;"
                    "1=threshold_floor_due_to_insufficient_positive_weeks;"
                    "2=no_supported_tail_scale"
                ),
                "no_imputation": "true",
            }
        )
        tail_audit = {
            positive_week_count.name: positive_week_count,
            pooled_positive_l1.name: pooled_positive_l1,
            audited_threshold_floor.name: audited_threshold_floor,
            tail_fallback_code.name: tail_fallback_code,
        }
    anomaly.attrs.update({"units": units, "baseline": f"{baseline[0]}/{baseline[1]}"})
    centre.attrs.update({"units": units, "baseline": f"{baseline[0]}/{baseline[1]}"})
    scale.attrs.update(
        {
            "units": units,
            "baseline": f"{baseline[0]}/{baseline[1]}",
            "zero_inflated_tail": bool(zero_inflated_tail),
        }
    )
    pooled_mad.attrs.update(
        {
            "units": units,
            "baseline": f"{baseline[0]}/{baseline[1]}",
            "method": "1.4826 * pooled median absolute residual after ISO-week centering",
        }
    )
    pooled_l1.attrs.update(
        {
            "units": units,
            "baseline": f"{baseline[0]}/{baseline[1]}",
            "method": "sqrt(pi/2) * pooled mean absolute residual after ISO-week centering",
        }
    )
    scale_source_code.attrs.update(
        {
            "units": "code",
            "codes": ";".join(
                [
                    *(f"{code}={name}" for code, name in enumerate(candidate_names)),
                    f"{len(candidate_names)}=undefined",
                ]
            ),
            "selection": (
                "largest supported candidate; exact ties prefer the earlier code"
            ),
        }
    )
    scale_description = (
        "max(seasonal 1.4826*MAD, pooled residual 1.4826*MAD, "
        "pooled residual sqrt(pi/2)*mean absolute residual"
    )
    if zero_inflated_tail:
        scale_description += (
            ", pooled positive-week sqrt(pi/2)*mean absolute residual when "
            f"N+>={minimum_positive_weeks}, audited threshold floor otherwise"
        )
    scale_description += (
        f"); undefined below {absolute_scale_floor_mm:g} {units}"
    )
    robust_z.attrs.update(
        {
            "units": "1",
            "method": "(weekly quantity - audited ISO-week median) / robust positive scale",
            "scale": scale_description,
            "baseline": f"{baseline[0]}/{baseline[1]}",
            "zero_inflated_tail": bool(zero_inflated_tail),
        }
    )
    percentile.attrs.update(
        {
            "units": "0-1",
            "method": "standard normal CDF of precip_robust_z",
            "interpretation": "robust normal-score percentile; not an empirical rank",
        }
    )
    variables = {
        anomaly.name: anomaly,
        robust_z.name: robust_z,
        percentile.name: percentile,
        centre.name: centre,
        scale.name: scale,
        pooled_mad.name: pooled_mad,
        pooled_l1.name: pooled_l1,
        scale_source_code.name: scale_source_code,
        n_baseline.name: n_baseline,
        effective_n.name: effective_n,
        fallback_code.name: fallback_code,
        **tail_audit,
    }
    return xr.Dataset(variables)


def gamma_spi_weekly(
    weekly: xr.DataArray,
    *,
    accumulation_weeks: int,
    baseline: tuple[str, str] = CLIMATOLOGY_BASE,
    minimum_baseline_years: int = 20,
    return_parameters: bool = False,
) -> xr.DataArray | xr.Dataset:
    """Moment-fit zero-adjusted gamma SPI on weekly rolling origins.

    This is an auditable weekly-origin approximation to SPI: 4, 13 and 26
    weeks represent approximately 1, 3 and 6 months.  Parameters are fitted
    separately by ISO week on the baseline.  Outputs carry a provisional flag
    because publication should additionally compare them with a validated
    monthly SPI implementation.
    """

    if accumulation_weeks < 1:
        raise ValueError("accumulation_weeks must be positive.")
    accumulated = weekly.rolling(
        time=accumulation_weeks, min_periods=accumulation_weeks
    ).sum()
    accumulated = _rechunk_for_time_statistics(accumulated)
    accumulated = accumulated.assign_coords(week_of_year=_iso_week(accumulated.time))
    base = accumulated.sel(time=slice(*baseline))
    positive = base.where(base > 0)
    positive_count = positive.groupby("week_of_year").count("time")
    total_count = base.groupby("week_of_year").count("time")
    zero_probability = 1.0 - positive_count / total_count
    mean = positive.groupby("week_of_year").mean("time", skipna=True)
    second_moment = (positive**2).groupby("week_of_year").mean(
        "time", skipna=True
    )
    variance_population = (second_moment - mean**2).clip(min=0.0)
    variance = variance_population * positive_count / (positive_count - 1).where(
        positive_count > 1
    )
    shape = (mean**2 / variance).where(variance > 0)
    theta = (variance / mean).where(mean > 0)
    week = accumulated["week_of_year"]
    shape_t = shape.sel(week_of_year=week)
    theta_t = theta.sel(week_of_year=week)
    q_t = zero_probability.sel(week_of_year=week)
    n_t = total_count.sel(week_of_year=week)
    scaled = xr.where(accumulated > 0, accumulated / theta_t, 0.0)
    gamma_cdf = xr.apply_ufunc(
        special.gammainc,
        shape_t,
        scaled,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    probability = xr.where(accumulated <= 0, q_t * 0.5, q_t + (1.0 - q_t) * gamma_cdf)
    probability = probability.clip(min=1e-6, max=1.0 - 1e-6)
    spi = xr.apply_ufunc(
        special.ndtri,
        probability,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    spi = spi.where(
        (n_t >= minimum_baseline_years)
        & np.isfinite(shape_t)
        & np.isfinite(theta_t)
    ).astype("float32")
    months = {4: 1, 13: 3, 26: 6}.get(accumulation_weeks)
    suffix = f"{months}m" if months is not None else f"{accumulation_weeks}w"
    spi.name = f"spi_gamma_{suffix}_weekly_origin"
    spi.attrs.update(
        {
            "units": "1",
            "accumulation_weeks": accumulation_weeks,
            "baseline": f"{baseline[0]}/{baseline[1]}",
            "distribution": "zero-adjusted gamma, method-of-moments, by ISO week",
            "status": "provisional; validate against an independent monthly SPI implementation",
        }
    )
    if return_parameters:
        parameter_names = {
            "shape": f"spi_gamma_{suffix}_shape_by_iso_week",
            "scale": f"spi_gamma_{suffix}_scale_by_iso_week",
            "zero_probability": f"spi_gamma_{suffix}_zero_probability_by_iso_week",
            "sample_count": f"spi_gamma_{suffix}_sample_count_by_iso_week",
        }
        parameters = {
            parameter_names["shape"]: shape.astype("float32"),
            parameter_names["scale"]: theta.astype("float32"),
            parameter_names["zero_probability"]: zero_probability.astype("float32"),
            parameter_names["sample_count"]: total_count.astype("uint8"),
        }
        parameters[parameter_names["shape"]].attrs.update(
            {"units": "1", "method": "gamma method-of-moments shape"}
        )
        parameters[parameter_names["scale"]].attrs.update(
            {"units": "mm", "method": "gamma method-of-moments scale"}
        )
        parameters[parameter_names["zero_probability"]].attrs["units"] = "0-1"
        parameters[parameter_names["sample_count"]].attrs["units"] = "weeks"
        output = xr.Dataset({spi.name: spi, **parameters})
        output.attrs.update(
            {
                "status": "provisional SPI diagnostics with fitted parameters retained",
                "baseline": f"{baseline[0]}/{baseline[1]}",
            }
        )
        return output
    return spi


def _longest_true_run(values: np.ndarray, axis: int | tuple[int, ...] = 0) -> np.ndarray:
    if isinstance(axis, tuple):
        if len(axis) != 1:
            raise ValueError("Run length reduction accepts one time axis.")
        axis = axis[0]
    array = np.moveaxis(np.asarray(values), axis, 0)
    truth = np.isfinite(array) & array.astype(bool)
    current = np.zeros(array.shape[1:], dtype=np.int16)
    longest = np.zeros(array.shape[1:], dtype=np.int16)
    for step in truth:
        current = np.where(step, current + 1, 0)
        longest = np.maximum(longest, current)
    return longest


def _within_week_longest_run(condition: xr.DataArray) -> xr.DataArray:
    """Lazy longest run (0..7) reset at each W-SUN boundary.

    ``GroupBy.reduce`` with a Python/NumPy reducer materialised thousands of
    Dask groups at graph-construction time and exhausted RAM.  Repeated shifted
    resamples have the same problem on the complete 1981--present native grid.
    Instead, construct one explicit seven-day window for every W-SUN label and
    apply the bounded NumPy reducer along that seven-element core dimension.
    The operation remains lazy, chunkable and cannot join runs across weeks.
    """

    index = pd.DatetimeIndex(condition.time.values)
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Daily timestamps must be unique and ordered for run lengths.")
    if index.empty:
        raise ValueError("At least one daily timestamp is required for run lengths.")
    first_sunday = (
        index[0] + pd.Timedelta(days=int((6 - index[0].dayofweek) % 7))
    ).normalize()
    last_sunday = (
        index[-1] + pd.Timedelta(days=int((6 - index[-1].dayofweek) % 7))
    ).normalize()
    week_ends = pd.date_range(first_sunday, last_sunday, freq="7D")
    days: list[xr.DataArray] = []
    # Monday..Sunday ordering. Reindex deliberately inserts False outside the
    # observed range for partial edge weeks; completeness is audited separately.
    for days_before_sunday in range(6, -1, -1):
        requested = week_ends - pd.Timedelta(days=days_before_sunday)
        selected = condition.reindex(time=requested, fill_value=False)
        selected = selected.assign_coords(time=week_ends)
        days.append(selected.fillna(False).astype("uint8"))
    window = xr.concat(
        days,
        dim=xr.IndexVariable("run_day", np.arange(7, dtype=np.uint8)),
    )
    longest = xr.apply_ufunc(
        _longest_true_run,
        window,
        input_core_dims=[["run_day"]],
        output_core_dims=[[]],
        kwargs={"axis": -1},
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.uint8],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    return longest.astype("uint8")


def _supported_nanquantile(
    values: np.ndarray,
    *,
    quantile: float,
    minimum_count: int,
) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < minimum_count:
        return float("nan")
    return float(np.quantile(finite, quantile))


def weekly_extreme_indices(
    daily: xr.DataArray,
    *,
    baseline: tuple[str, str] = CLIMATOLOGY_BASE,
    wet_day_threshold_mm: float = 1.0,
    minimum_baseline_wet_days: int = 100,
) -> xr.Dataset:
    """Weekly diagnostics derived directly from daily native CHIRPS cells."""

    daily = canonicalize_chirps_daily(daily, bbox=None)
    base_source = _rechunk_for_time_statistics(
        daily.sel(time=slice(*baseline)), spatial_chunk=16
    )
    base_wet = base_source.where(base_source >= wet_day_threshold_mm)
    wet_day_count = base_wet.count("time")
    threshold_usable = wet_day_count >= minimum_baseline_wet_days
    def supported_quantile(value: float) -> xr.DataArray:
        return xr.apply_ufunc(
            _supported_nanquantile,
            base_wet,
            input_core_dims=[["time"]],
            output_core_dims=[[]],
            kwargs={
                "quantile": value,
                "minimum_count": minimum_baseline_wet_days,
            },
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float64],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )

    q95 = supported_quantile(0.95).where(threshold_usable)
    q99 = supported_quantile(0.99).where(threshold_usable)
    rx1 = daily.resample(time="W-SUN").max(skipna=True).astype("float32")
    rx5 = (
        daily.rolling(time=5, min_periods=5).sum()
        .resample(time="W-SUN")
        .max(skipna=True)
        .astype("float32")
    )
    r95p = (
        daily.where(daily > q95, 0.0)
        .resample(time="W-SUN")
        .sum()
        .where(q95.notnull())
        .astype("float32")
    )
    r99p = (
        daily.where(daily > q99, 0.0)
        .resample(time="W-SUN")
        .sum()
        .where(q99.notnull())
        .astype("float32")
    )
    dry = ((daily < wet_day_threshold_mm) & daily.notnull()).astype("uint8")
    wet = ((daily >= wet_day_threshold_mm) & daily.notnull()).astype("uint8")
    cdd = _within_week_longest_run(dry)
    cwd = _within_week_longest_run(wet)
    variables = {
        "rx1day_weekly_mm": rx1,
        "rx5day_weekly_mm": rx5,
        "r95p_weekly_mm": r95p,
        "r99p_weekly_mm": r99p,
        "cdd_within_week_days": cdd,
        "cwd_within_week_days": cwd,
        "baseline_wet_day_count": wet_day_count.astype("uint16"),
        "baseline_wet_day_p95_mm": q95.astype("float32"),
        "baseline_wet_day_p99_mm": q99.astype("float32"),
    }
    for name, variable in variables.items():
        variable.name = name
        variable.attrs["native_grid"] = "true"
    rx1.attrs.update({"units": "mm", "definition": "maximum 1-day precipitation in W-SUN week"})
    rx5.attrs.update({"units": "mm", "definition": "maximum trailing 5-day precipitation in W-SUN week"})
    r95p.attrs.update(
        {
            "units": "mm",
            "definition": "weekly amount above baseline wet-day pixel p95",
            "status": "weekly diagnostic; threshold is pixelwise baseline wet-day quantile",
        }
    )
    r99p.attrs.update(
        {
            "units": "mm",
            "definition": "weekly amount above baseline wet-day pixel p99",
            "status": "weekly diagnostic; threshold is pixelwise baseline wet-day quantile",
        }
    )
    cdd.attrs.update(
        {
            "units": "days",
            "definition": f"maximum run < {wet_day_threshold_mm:g} mm within each week",
            "status": "weekly diagnostic, not official annual ETCCDI CDD",
        }
    )
    cwd.attrs.update(
        {
            "units": "days",
            "definition": f"maximum run >= {wet_day_threshold_mm:g} mm within each week",
            "status": "weekly diagnostic, not official annual ETCCDI CWD",
        }
    )
    variables["baseline_wet_day_count"].attrs.update(
        {
            "units": "days",
            "baseline": f"{baseline[0]}/{baseline[1]}",
            "minimum_required": minimum_baseline_wet_days,
        }
    )
    variables["baseline_wet_day_p95_mm"].attrs.update(
        {"units": "mm day-1", "quantile": 0.95, "wet_day_threshold_mm": wet_day_threshold_mm}
    )
    variables["baseline_wet_day_p99_mm"].attrs.update(
        {"units": "mm day-1", "quantile": 0.99, "wet_day_threshold_mm": wet_day_threshold_mm}
    )
    return xr.Dataset(variables)


def native_pixel_table(
    latitude: Iterable[float],
    longitude: Iterable[float],
) -> pd.DataFrame:
    """Create immutable row-major IDs for every cell of the retained grid."""

    lat = np.asarray(list(latitude))
    lon = np.asarray(list(longitude))
    _assert_regular_coordinate(lat, "latitude")
    _assert_regular_coordinate(lon, "longitude")
    grid = native_grid_hash(lat, lon)
    row, column = np.meshgrid(
        np.arange(len(lat), dtype=np.int32),
        np.arange(len(lon), dtype=np.int32),
        indexing="ij",
    )
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    table = pd.DataFrame(
        {
            "pixel_id": (row.astype(np.int64) * len(lon) + column).ravel(),
            "grid_row": row.ravel(),
            "grid_column": column.ravel(),
            "lat": lat_grid.ravel(),
            "lon": lon_grid.ravel(),
            "grid_hash": grid,
            "native_pixel": True,
            "interpolated": False,
        }
    )
    return table


def add_brazil_mask(
    dataset: xr.Dataset,
    pixels: pd.DataFrame,
) -> xr.Dataset:
    """Attach explicit Brazil fraction/centre masks without dropping any cell."""

    required = {"pixel_id", "grid_row", "grid_column", "brazil_fraction", "brazil_center"}
    missing = required.difference(pixels.columns)
    if missing:
        raise KeyError(f"Pixel table is missing Brazil-mask columns {sorted(missing)}.")
    expected = dataset.sizes["latitude"] * dataset.sizes["longitude"]
    if len(pixels) != expected or pixels["pixel_id"].duplicated().any():
        raise ValueError("Pixel table does not map one-to-one onto the native grid.")
    ordered = pixels.sort_values(["grid_row", "grid_column"], kind="mergesort")
    shape = (dataset.sizes["latitude"], dataset.sizes["longitude"])
    fraction = ordered["brazil_fraction"].to_numpy(dtype=np.float32).reshape(shape)
    centre = ordered["brazil_center"].astype(bool).to_numpy().reshape(shape)
    out = dataset.copy()
    out["pixel_id"] = xr.DataArray(
        ordered["pixel_id"].to_numpy(dtype=np.int64).reshape(shape),
        dims=("latitude", "longitude"),
    )
    out["brazil_fraction"] = xr.DataArray(
        np.clip(fraction, 0.0, 1.0), dims=("latitude", "longitude")
    )
    out["brazil_center"] = xr.DataArray(centre, dims=("latitude", "longitude"))
    out["brazil_fraction"].attrs.update(
        {
            "units": "0-1",
            "definition": "equal-area overlap of native CHIRPS cell with official Brazil regions",
        }
    )
    out["brazil_center"].attrs["definition"] = "native cell centre lies in official Brazil geometry"
    return out


def build_native_weekly_targets(
    daily: xr.DataArray,
    *,
    pixels: pd.DataFrame | None = None,
    baseline: tuple[str, str] = CLIMATOLOGY_BASE,
    minimum_valid_days: int = 7,
    include_spi: bool = True,
    include_extremes: bool = True,
) -> xr.Dataset:
    """Build the canonical F4/F6/F8 target cube on exact CHIRPS cells."""

    daily = canonicalize_chirps_daily(daily, bbox=None)
    weekly = weekly_native_precipitation(daily, minimum_valid_days=minimum_valid_days)
    anomaly = robust_weekly_anomalies(weekly["precip_weekly_mm"], baseline=baseline)
    target = xr.merge([weekly, anomaly], compat="override")
    if include_spi:
        for accumulation in (4, 13, 26):
            spi_diagnostics = gamma_spi_weekly(
                weekly["precip_weekly_mm"],
                accumulation_weeks=accumulation,
                baseline=baseline,
                return_parameters=True,
            )
            if not isinstance(spi_diagnostics, xr.Dataset):
                raise TypeError("SPI diagnostics contract must return an xarray Dataset.")
            target = xr.merge([target, spi_diagnostics], compat="override")
    if include_extremes:
        extremes = weekly_extreme_indices(daily, baseline=baseline)
        # Edge weeks and pixel-specific daily gaps must never become usable
        # inferential targets merely because an extreme index can be computed
        # from a partial collection of days.
        for variable in extremes.data_vars:
            if "time" in extremes[variable].dims:
                extremes[variable] = extremes[variable].where(
                    weekly["week_complete"]
                )
        target = xr.merge([target, extremes])
        # Extreme indices remain on the same native pixels.  Their robust
        # seasonal anomalies are the inferential targets; raw amounts/runs are
        # retained alongside them for physical interpretation and audit.
        anomaly_names = {
            "rx1day_weekly_mm": ("rx1day_weekly_anomaly_mm", "rx1day_weekly_robust_z"),
            "rx5day_weekly_mm": ("rx5day_weekly_anomaly_mm", "rx5day_weekly_robust_z"),
            "r95p_weekly_mm": ("r95p_weekly_anomaly_mm", "r95p_weekly_robust_z"),
            "r99p_weekly_mm": ("r99p_weekly_anomaly_mm", "r99p_weekly_robust_z"),
            "cdd_within_week_days": (
                "cdd_within_week_anomaly_days",
                "cdd_within_week_robust_z",
            ),
            "cwd_within_week_days": (
                "cwd_within_week_anomaly_days",
                "cwd_within_week_robust_z",
            ),
        }
        for source_name, (anomaly_name, z_name) in anomaly_names.items():
            source_units = str(extremes[source_name].attrs.get("units", "native units"))
            scale_floor = 0.25 if source_units == "days" else 0.10
            tail_threshold_name = {
                "r95p_weekly_mm": "baseline_wet_day_p95_mm",
                "r99p_weekly_mm": "baseline_wet_day_p99_mm",
            }.get(source_name)
            transformed = robust_weekly_anomalies(
                extremes[source_name],
                baseline=baseline,
                absolute_scale_floor_mm=scale_floor,
                quantity_units=source_units,
                zero_inflated_tail=tail_threshold_name is not None,
                minimum_positive_weeks=20,
                threshold_floor=(
                    extremes[tail_threshold_name]
                    if tail_threshold_name is not None
                    else None
                ),
            )
            target[anomaly_name] = transformed["precip_anomaly_mm"].rename(anomaly_name)
            target[z_name] = transformed["precip_robust_z"].rename(z_name)
            audit_prefix = source_name.removesuffix("_mm").removesuffix("_days")
            audit_layers = {
                "climatology_median_mm": f"{audit_prefix}_climatology_median",
                "climatology_robust_scale_mm": f"{audit_prefix}_climatology_robust_scale",
                "climatology_pooled_residual_mad_scale": (
                    f"{audit_prefix}_climatology_pooled_residual_mad_scale"
                ),
                "climatology_pooled_residual_l1_scale": (
                    f"{audit_prefix}_climatology_pooled_residual_l1_scale"
                ),
                "climatology_scale_source_code": (
                    f"{audit_prefix}_climatology_scale_source_code"
                ),
                "climatology_sample_count": f"{audit_prefix}_climatology_sample_count",
                "climatology_effective_sample_count": (
                    f"{audit_prefix}_climatology_effective_sample_count"
                ),
                "climatology_fallback_code": f"{audit_prefix}_climatology_fallback_code",
            }
            if tail_threshold_name is not None:
                audit_layers.update(
                    {
                        "climatology_positive_week_count": (
                            f"{audit_prefix}_climatology_positive_week_count"
                        ),
                        "climatology_pooled_positive_l1_scale": (
                            f"{audit_prefix}_climatology_pooled_positive_l1_scale"
                        ),
                        "climatology_tail_threshold_floor": (
                            f"{audit_prefix}_climatology_tail_threshold_floor"
                        ),
                        "climatology_tail_fallback_code": (
                            f"{audit_prefix}_climatology_tail_fallback_code"
                        ),
                    }
                )
            for original, renamed in audit_layers.items():
                target[renamed] = transformed[original].rename(renamed)
            target[anomaly_name].attrs.update(
                {
                    "source_index": source_name,
                    "method": "ISO-week median anomaly on original CHIRPS pixels",
                }
            )
            target[z_name].attrs.update(
                {
                    "source_index": source_name,
                    "method": (
                        "ISO-week robust anomaly with positive-week conditional "
                        "L1 scale and audited threshold fallback"
                        if tail_threshold_name is not None
                        else "ISO-week median/MAD robust anomaly; undefined at zero variability"
                    ),
                }
            )
    grid = native_grid_hash(target.latitude.values, target.longitude.values)
    target.attrs.update(
        {
            "target_contract_version": TARGET_CONTRACT_VERSION,
            "grid_hash_sha256": grid,
            "source": "CHIRPS native daily precipitation",
            "spatial_operation": "coordinate subset only; interpolation=false",
            "weekly_operation": "daily concatenation before single W-SUN resample",
            "climatology_baseline": f"{baseline[0]}/{baseline[1]}",
            "primary_analysis_mask": "brazil_center (or area-weight brazil_fraction)",
            "include_spi": bool(include_spi),
            "include_extremes": bool(include_extremes),
            "robust_climatology_method": (
                "ISO-week median; scale=max(seasonal MAD, pooled residual MAD, "
                "pooled residual L1-equivalent); R95p/R99p additionally use a "
                "positive-week conditional L1 floor with N+>=20, otherwise the "
                "audited wet-day percentile threshold floor; week53 circular "
                "52/53/1 fallback; "
                "all scale candidates, source codes, counts and fallback codes retained"
            ),
            "extreme_threshold_parameters_retained": bool(include_extremes),
            "spi_fit_parameters_retained": bool(include_spi),
        }
    )
    if pixels is not None:
        target = add_brazil_mask(target, pixels)
    return target


def target_to_frame(
    dataset: xr.Dataset,
    *,
    variable: str = "precip_robust_z",
    brazil_only: bool = True,
    mask_rule: str = "center",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return weeks x original pixels without changing coordinates or values."""

    if variable not in dataset:
        raise KeyError(f"Target dataset has no variable {variable!r}.")
    if mask_rule not in {"center", "overlap"}:
        raise ValueError("mask_rule must be 'center' or 'overlap'.")
    pixel_table = native_pixel_table(dataset.latitude.values, dataset.longitude.values)
    if "pixel_id" in dataset:
        pixel_table["pixel_id"] = dataset["pixel_id"].values.ravel().astype(np.int64)
    if "brazil_fraction" in dataset:
        pixel_table["brazil_fraction"] = dataset["brazil_fraction"].values.ravel()
    if "brazil_center" in dataset:
        pixel_table["brazil_center"] = dataset["brazil_center"].values.ravel().astype(bool)
    if brazil_only:
        mask_name = "brazil_center" if mask_rule == "center" else "brazil_fraction"
        if mask_name not in pixel_table:
            raise KeyError(f"Target dataset lacks required mask {mask_name!r}.")
        keep = (
            pixel_table[mask_name].astype(bool)
            if mask_rule == "center"
            else pixel_table[mask_name].astype(float) > 0
        )
    else:
        keep = pd.Series(True, index=pixel_table.index)
    stacked = dataset[variable].stack(pixel=("latitude", "longitude"))
    indices = np.flatnonzero(keep.to_numpy())
    selected_pixels = pixel_table.loc[keep].reset_index(drop=True)
    selected = stacked.isel(pixel=indices).transpose("time", "pixel")
    frame = pd.DataFrame(
        selected.values,
        index=pd.DatetimeIndex(selected.time.values, name="week_ending_sunday"),
        columns=selected_pixels["pixel_id"].astype(str),
    )
    return frame, selected_pixels


def validate_native_target(
    dataset: xr.Dataset,
    *,
    maximum_abs_robust_z: float = 100.0,
    deep: bool = True,
) -> TargetValidation:
    """Validate grid lineage, weekly completeness and anomaly magnitude."""

    errors: list[str] = []
    warnings: list[str] = []
    required = {
        "precip_weekly_mm",
        "valid_day_count",
        "expected_day_count",
        "week_complete",
        "precip_anomaly_mm",
        "precip_robust_z",
        "precip_robust_percentile",
        "climatology_median_mm",
        "climatology_robust_scale_mm",
        "climatology_pooled_residual_mad_scale",
        "climatology_pooled_residual_l1_scale",
        "climatology_scale_source_code",
    }
    missing = required.difference(dataset.data_vars)
    if missing:
        errors.append(f"missing variables: {sorted(missing)}")
    if dataset.attrs.get("target_contract_version") != TARGET_CONTRACT_VERSION:
        errors.append(
            "target_contract_version missing or incompatible with "
            f"{TARGET_CONTRACT_VERSION}"
        )
    if bool(dataset.attrs.get("include_spi", False)):
        spi_required = {
            f"spi_gamma_{suffix}_{parameter}_by_iso_week"
            for suffix in ("1m", "3m", "6m")
            for parameter in ("shape", "scale", "zero_probability", "sample_count")
        } | {
            "spi_gamma_1m_weekly_origin",
            "spi_gamma_3m_weekly_origin",
            "spi_gamma_6m_weekly_origin",
        }
        if missing_spi := spi_required.difference(dataset.data_vars):
            errors.append(f"missing auditable SPI variables: {sorted(missing_spi)}")
    if bool(dataset.attrs.get("include_extremes", False)):
        extreme_required = {
            "rx1day_weekly_mm",
            "rx5day_weekly_mm",
            "r95p_weekly_mm",
            "r99p_weekly_mm",
            "cdd_within_week_days",
            "cwd_within_week_days",
            "baseline_wet_day_count",
            "baseline_wet_day_p95_mm",
            "baseline_wet_day_p99_mm",
        } | {
            f"{prefix}_climatology_{suffix}"
            for prefix in ("r95p_weekly", "r99p_weekly")
            for suffix in (
                "positive_week_count",
                "pooled_positive_l1_scale",
                "tail_threshold_floor",
                "tail_fallback_code",
            )
        }
        if missing_extreme := extreme_required.difference(dataset.data_vars):
            errors.append(
                f"missing auditable extreme variables: {sorted(missing_extreme)}"
            )
    try:
        _assert_regular_coordinate(dataset.latitude.values, "latitude")
        _assert_regular_coordinate(dataset.longitude.values, "longitude")
        grid = native_grid_hash(dataset.latitude.values, dataset.longitude.values)
    except Exception as exc:  # validation must report all detectable failures
        errors.append(str(exc))
        grid = ""
    recorded = str(dataset.attrs.get("grid_hash_sha256", ""))
    if not recorded:
        errors.append("grid_hash_sha256 is mandatory")
    elif grid and recorded != grid:
        errors.append("recorded grid_hash_sha256 does not match coordinates")
    if "interpolation=false" not in str(dataset.attrs.get("spatial_operation", "")):
        errors.append("spatial_operation does not explicitly prohibit interpolation")
    if deep and "valid_day_count" in dataset:
        max_days = float(dataset["valid_day_count"].max().compute())
        if max_days > 7:
            errors.append(f"valid_day_count exceeds seven ({max_days:g})")
    if deep:
        robust_targets = sorted(
            name
            for name in dataset.data_vars
            if name == "precip_robust_z" or name.endswith("_weekly_robust_z")
            or name.endswith("_within_week_robust_z")
        )
        for name in robust_targets:
            maximum = float(np.abs(dataset[name]).max(skipna=True).compute())
            if not np.isfinite(maximum):
                errors.append(f"{name} has no finite values")
            elif maximum > maximum_abs_robust_z:
                errors.append(
                    f"{name} maximum |z|={maximum:.3g} exceeds "
                    f"{maximum_abs_robust_z:g}"
                )
    masks_missing = "brazil_fraction" not in dataset or "brazil_center" not in dataset
    if masks_missing:
        if str(dataset.attrs.get("build_status", "")).startswith("canonical"):
            errors.append("canonical target lacks Brazil overlap/centre masks")
        else:
            warnings.append("Brazil overlap/centre masks are not attached")
    elif deep:
        fraction_min = float(dataset["brazil_fraction"].min(skipna=True).compute())
        fraction_max = float(dataset["brazil_fraction"].max(skipna=True).compute())
        if fraction_min < 0.0 or fraction_max > 1.0:
            errors.append(
                f"brazil_fraction lies outside [0,1]: {fraction_min:g}..{fraction_max:g}"
            )
    if deep and {"week_complete", "valid_day_count", "expected_day_count"}.issubset(dataset):
        expected_complete = (
            (dataset["valid_day_count"] >= 7) & (dataset["expected_day_count"] == 7)
        )
        mismatch = bool((dataset["week_complete"] != expected_complete).any().compute())
        if mismatch:
            errors.append("week_complete disagrees with valid/expected day counts")
        incomplete_has_target = bool(
            dataset["precip_weekly_mm"].where(~dataset["week_complete"]).notnull().any().compute()
        )
        if incomplete_has_target:
            errors.append("incomplete weeks contain primary precipitation target values")
    if deep and "precip_robust_percentile" in dataset:
        percentile_min = float(dataset["precip_robust_percentile"].min(skipna=True).compute())
        percentile_max = float(dataset["precip_robust_percentile"].max(skipna=True).compute())
        if percentile_min < 0.0 or percentile_max > 1.0:
            errors.append("precip_robust_percentile lies outside [0,1]")
    if "pixel_id" in dataset:
        ids = np.asarray(dataset["pixel_id"].values).ravel()
        n_latitude = int(dataset.sizes.get("latitude", 0))
        n_longitude = int(dataset.sizes.get("longitude", 0))
        row_offset = int(dataset.attrs.get("block_latitude_start", 0))
        expected_matrix = (
            (np.arange(n_latitude, dtype=np.int64) + row_offset)[:, None]
            * n_longitude
            + np.arange(n_longitude, dtype=np.int64)[None, :]
        )
        if ids.shape != (n_latitude * n_longitude,) or not np.array_equal(
            ids, expected_matrix.ravel()
        ):
            errors.append("pixel_id is not the exact global row-major native-grid identity")
    elif not masks_missing or str(dataset.attrs.get("build_status", "")).startswith("canonical"):
        errors.append("pixel_id is mandatory for masked/canonical targets")
    time_index = pd.DatetimeIndex(dataset.indexes.get("time", pd.Index([])))
    if time_index.has_duplicates:
        errors.append("weekly time coordinate has duplicates")
    if not time_index.is_monotonic_increasing:
        errors.append("weekly time coordinate is not monotonic")
    if len(time_index) and not bool((time_index.dayofweek == 6).all()):
        errors.append("weekly time coordinate is not W-SUN")
    if len(time_index) > 1 and not bool(
        (np.diff(time_index.values) == np.timedelta64(7, "D")).all()
    ):
        errors.append("weekly time coordinate has missing/non-weekly steps")
    if deep and {
        "baseline_wet_day_count",
        "baseline_wet_day_p95_mm",
        "baseline_wet_day_p99_mm",
    }.issubset(dataset):
        wet_count = dataset["baseline_wet_day_count"]
        threshold_invalid = wet_count < 100
        leaked_threshold = bool(
            (
                dataset["baseline_wet_day_p95_mm"].where(threshold_invalid).notnull().any()
                | dataset["baseline_wet_day_p99_mm"].where(threshold_invalid).notnull().any()
            ).compute()
        )
        if leaked_threshold:
            errors.append("wet-day p95/p99 is finite below 100 baseline wet days")
    if deep and {
        "climatology_median_mm",
        "climatology_effective_sample_count",
    }.issubset(dataset):
        invalid_centre = bool(
            dataset["climatology_median_mm"]
            .where(dataset["climatology_effective_sample_count"] < 20)
            .notnull()
            .any()
            .compute()
        )
        if invalid_centre:
            errors.append("climatology centre is finite below 20 effective samples")
    return TargetValidation(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        grid_hash=grid,
        n_time=int(dataset.sizes.get("time", 0)),
        n_latitude=int(dataset.sizes.get("latitude", 0)),
        n_longitude=int(dataset.sizes.get("longitude", 0)),
    )
