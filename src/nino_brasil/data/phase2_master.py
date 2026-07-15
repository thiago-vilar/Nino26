"""Scientific contract and pure transformations for the Phase 2 weekly master.

The legacy/raw master remains the compatibility artefact.  Source-aware
ocean anomalies are written to a separate, explicitly named artefact so that
an absolute depth (for example ``d20_m``) is never silently reinterpreted.
Predictive cross-validation must still fit climatology and detrending inside
each training fold; the source-adjusted product is intended for descriptive
statistics and seam diagnostics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


RHO_AIR_KG_M3 = 1.2
CD_NEUTRAL = 1.3e-3
ERA5_ACCUMULATION_SECONDS = 3600.0
CLIMATOLOGY_START = "1991-01-01"
CLIMATOLOGY_END = "2020-12-31"


@dataclass(frozen=True)
class VariableSpec:
    name: str
    source: str
    units: str
    positive: str
    representation_raw: str
    representation_source_adjusted: str


VARIABLE_SPECS: tuple[VariableSpec, ...] = (
    VariableSpec("nino34_ssta", "NOAA OISST", "degC", "warmer", "seasonal_anomaly", "unchanged_independent_anomaly"),
    VariableSpec("d20_m", "UFS/GLORYS/GLO12", "m", "deeper", "absolute", "source_seasonal_anomaly_detrended"),
    VariableSpec("tilt_m", "UFS/GLORYS/GLO12", "m", "east_minus_west", "absolute_difference", "source_seasonal_anomaly_detrended"),
    VariableSpec("tilt_slope", "UFS/GLORYS/GLO12", "m degree-1", "increasing_eastward", "linear_slope", "source_seasonal_anomaly_detrended"),
    VariableSpec("ohc_0_100", "UFS/GLORYS/GLO12", "J m-2", "more_heat", "absolute", "source_seasonal_anomaly_detrended"),
    VariableSpec("ohc_0_300", "UFS/GLORYS/GLO12", "J m-2", "more_heat", "absolute", "source_seasonal_anomaly_detrended"),
    VariableSpec("ohc_0_700", "UFS/GLORYS/GLO12", "J m-2", "more_heat", "absolute", "source_seasonal_anomaly_detrended"),
    VariableSpec("ohc_300_700", "UFS/GLORYS/GLO12", "J m-2", "more_heat", "absolute", "source_seasonal_anomaly_detrended"),
    VariableSpec("ssh_m", "UFS/GLORYS/GLO12", "m", "higher_sea_surface", "absolute", "source_seasonal_anomaly_detrended"),
    VariableSpec("wwv", "UFS/GLORYS/GLO12", "m3", "more_warm_water", "absolute", "source_seasonal_anomaly_detrended"),
    VariableSpec("t50m", "UFS/GLORYS/GLO12", "degC", "warmer", "absolute", "source_seasonal_anomaly_detrended"),
    VariableSpec("t100m", "UFS/GLORYS/GLO12", "degC", "warmer", "absolute", "source_seasonal_anomaly_detrended"),
    VariableSpec("t150m", "UFS/GLORYS/GLO12", "degC", "warmer", "absolute", "source_seasonal_anomaly_detrended"),
    VariableSpec("t200m", "UFS/GLORYS/GLO12", "degC", "warmer", "absolute", "source_seasonal_anomaly_detrended"),
    VariableSpec("t300m", "UFS/GLORYS/GLO12", "degC", "warmer", "absolute", "source_seasonal_anomaly_detrended"),
    VariableSpec("t500m", "UFS/GLORYS/GLO12", "degC", "warmer", "absolute", "source_seasonal_anomaly_detrended"),
    VariableSpec("t700m", "UFS/GLORYS/GLO12", "degC", "warmer", "absolute", "source_seasonal_anomaly_detrended"),
    VariableSpec("tau_x_anom", "ERA5", "Pa", "eastward", "day_of_year_anomaly", "same_as_raw"),
    VariableSpec("u10_anom", "ERA5", "m s-1", "eastward", "day_of_year_anomaly", "same_as_raw"),
    VariableSpec("v10_anom", "ERA5", "m s-1", "northward", "day_of_year_anomaly", "same_as_raw"),
    VariableSpec("mslp_anom", "ERA5", "Pa", "higher_pressure", "day_of_year_anomaly", "same_as_raw"),
    VariableSpec("tcwv_anom", "ERA5", "kg m-2", "more_water_vapour", "day_of_year_anomaly", "same_as_raw"),
    VariableSpec("slhf_anom", "ERA5", "W m-2", "upward_surface_cooling", "day_of_year_anomaly", "same_as_raw"),
    VariableSpec("sshf_anom", "ERA5", "W m-2", "upward_surface_cooling", "day_of_year_anomaly", "same_as_raw"),
    VariableSpec("ssr_anom", "ERA5", "W m-2", "downward_surface_warming", "day_of_year_anomaly", "same_as_raw"),
    VariableSpec("str_anom", "ERA5", "W m-2", "downward_surface_warming", "day_of_year_anomaly", "same_as_raw"),
    VariableSpec("u850_anom", "ERA5", "m s-1", "eastward", "day_of_year_anomaly", "same_as_raw"),
    VariableSpec("u200_anom", "ERA5", "m s-1", "eastward", "day_of_year_anomaly", "same_as_raw"),
    VariableSpec("omega850_anom", "ERA5", "Pa s-1", "downward_motion", "day_of_year_anomaly", "same_as_raw"),
    VariableSpec("omega500_anom", "ERA5", "Pa s-1", "downward_motion", "day_of_year_anomaly", "same_as_raw"),
    VariableSpec("div850_anom", "ERA5", "s-1", "horizontal_divergence", "day_of_year_anomaly", "same_as_raw"),
)

PHYSICAL_COLUMNS: tuple[str, ...] = tuple(spec.name for spec in VARIABLE_SPECS)
OCEAN_COLUMNS: tuple[str, ...] = PHYSICAL_COLUMNS[:17]
ATMOSPHERIC_COLUMNS: tuple[str, ...] = PHYSICAL_COLUMNS[17:]
METADATA_COLUMNS: tuple[str, ...] = ("ocean_source_code",)

# GLO12 is the operational continuation of the GLORYS system.  It has too
# short a record to estimate its own 366-day climatology, so both use the same
# climatological family while their source codes remain distinct.
SOURCE_CLIMATOLOGY_FAMILY: Mapping[int, str] = {1: "ufs", 2: "glorys", 3: "glorys"}


def variable_contract_frame() -> pd.DataFrame:
    frame = pd.DataFrame(asdict(spec) for spec in VARIABLE_SPECS)
    frame.insert(0, "ordinal", np.arange(1, len(frame) + 1))
    frame["is_physical_predictor"] = True
    metadata = pd.DataFrame(
        [
            {
                "ordinal": pd.NA,
                "name": "ocean_source_code",
                "source": "pipeline provenance",
                "units": "1=NOAA_UFS; 2=GLORYS12; 3=GLO12_operational",
                "positive": "not_applicable",
                "representation_raw": "categorical_metadata",
                "representation_source_adjusted": "categorical_metadata",
                "is_physical_predictor": False,
            }
        ]
    )
    return pd.concat([frame, metadata], ignore_index=True)


def vector_wind_stress_x(
    u10: pd.Series | np.ndarray,
    v10: pd.Series | np.ndarray,
    *,
    rho_air: float = RHO_AIR_KG_M3,
    drag_coefficient: float = CD_NEUTRAL,
) -> pd.Series | np.ndarray:
    """Return eastward zonal wind stress ``rho * Cd * |V| * u`` in Pa."""
    speed = np.hypot(u10, v10)
    return rho_air * drag_coefficient * speed * u10


def normalize_era5_daily_units(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert ERA5 one-hour accumulated surface fluxes to daily-mean W m-2.

    The project's daily Zarr stores average the 24 hourly ERA5 records.  ERA5
    flux records retain their native ``J m-2`` one-hour accumulation, therefore
    conversion is by 3600 s (not 86400 s).  Turbulent fluxes are flipped to an
    upward-positive convention; radiative fluxes retain ERA5's downward-positive
    convention.  Instantaneous variables are copied unchanged.
    """
    out = frame.copy()
    for name in ("slhf", "sshf"):
        if name in out:
            out[name] = -pd.to_numeric(out[name], errors="coerce") / ERA5_ACCUMULATION_SECONDS
    for name in ("ssr", "str"):
        if name in out:
            out[name] = pd.to_numeric(out[name], errors="coerce") / ERA5_ACCUMULATION_SECONDS
    return out


def day_of_year_anomaly(
    series: pd.Series,
    *,
    climatology_start: str = CLIMATOLOGY_START,
    climatology_end: str = CLIMATOLOGY_END,
    smooth_days: int = 15,
) -> pd.Series:
    """Return a circularly smoothed daily anomaly on a 366-day calendar."""
    values = pd.to_numeric(series, errors="coerce").sort_index()
    if not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError("day_of_year_anomaly requires a DatetimeIndex")
    base = values.loc[climatology_start:climatology_end]
    if base.notna().sum() < 366:
        base = values
    climatology = base.groupby(base.index.dayofyear).mean().reindex(range(1, 367))
    climatology = climatology.interpolate(limit_direction="both")
    pad = max(1, smooth_days // 2)
    circular = pd.concat([climatology.iloc[-pad:], climatology, climatology.iloc[:pad]])
    smooth = circular.rolling(smooth_days, center=True, min_periods=1).mean().iloc[pad:-pad]
    mapped = pd.Series(values.index.dayofyear.map(smooth), index=values.index, dtype=float)
    return (values - mapped).astype(float)


def _climatology_family(source: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(source, errors="coerce").astype("Int64")
    return numeric.map(SOURCE_CLIMATOLOGY_FAMILY).astype("string")


def _detrend_with_reference(
    anomaly: pd.Series,
    reference_mask: pd.Series,
    *,
    min_points: int = 365,
) -> pd.Series:
    result = anomaly.copy()
    finite = pd.Series(
        np.isfinite(pd.to_numeric(anomaly, errors="coerce").to_numpy(dtype=float)),
        index=anomaly.index,
    )
    fit_mask = finite & reference_mask.reindex(anomaly.index, fill_value=False)
    if int(fit_mask.sum()) < min_points:
        fit_mask = finite
    if int(fit_mask.sum()) < 3:
        return result
    origin = anomaly.index.min()
    x_all = (anomaly.index - origin).total_seconds().to_numpy(dtype=float) / 86400.0
    x_fit = x_all[fit_mask.to_numpy()]
    y_fit = anomaly.loc[fit_mask].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    centre = float(np.nanmean(x_fit))
    trend = slope * (x_all - centre)
    result.loc[finite] = anomaly.loc[finite].to_numpy(dtype=float) - trend[finite.to_numpy()]
    return result


def source_aware_ocean_adjustment(
    ocean_daily: pd.DataFrame,
    source_code: pd.Series,
    *,
    independent_anomaly_columns: Iterable[str] = ("nino34_ssta",),
    climatology_start: str = CLIMATOLOGY_START,
    climatology_end: str = CLIMATOLOGY_END,
) -> pd.DataFrame:
    """Seasonally adjust and detrend ocean variables within source families.

    UFS and GLORYS are never pooled.  GLORYS and its short operational GLO12
    continuation share a family.  Already independent anomalies (OISST SSTA)
    pass through unchanged.  This operation uses the full/reference record and
    must not be precomputed outside training folds for predictive evaluation.
    """
    if not isinstance(ocean_daily.index, pd.DatetimeIndex):
        raise TypeError("source_aware_ocean_adjustment requires a DatetimeIndex")
    source = source_code.reindex(ocean_daily.index)
    family = _climatology_family(source)
    adjusted = pd.DataFrame(index=ocean_daily.index)
    independent = set(independent_anomaly_columns)
    for column in ocean_daily.columns:
        values = pd.to_numeric(ocean_daily[column], errors="coerce")
        if column in independent:
            adjusted[column] = values
            continue
        combined = pd.Series(np.nan, index=values.index, dtype=float)
        finite_values = pd.Series(np.isfinite(values.to_numpy(dtype=float)), index=values.index)
        for family_name in family.dropna().unique():
            mask = family.eq(family_name) & finite_values
            if not mask.any():
                continue
            subset = values.where(mask)
            base_mask = mask & values.index.to_series().between(climatology_start, climatology_end).to_numpy()
            base = subset.loc[base_mask]
            if base.notna().sum() < 2 * 365:
                base = subset.loc[mask]
            climatology = base.groupby(base.index.dayofyear).mean().reindex(range(1, 367))
            climatology = climatology.interpolate(limit_direction="both")
            circular = pd.concat([climatology.iloc[-7:], climatology, climatology.iloc[:7]])
            climatology = circular.rolling(15, center=True, min_periods=1).mean().iloc[7:-7]
            mapped = pd.Series(values.index.dayofyear.map(climatology), index=values.index, dtype=float)
            anomaly = (values - mapped).where(mask)
            detrended = _detrend_with_reference(anomaly, base_mask)
            combined.loc[mask] = detrended.loc[mask]
        adjusted[column] = combined
    return adjusted


def weekly_source_mode(source: pd.Series) -> pd.Series:
    def mode_or_nan(values: pd.Series) -> float:
        valid = values.dropna()
        return float(valid.mode().iloc[0]) if len(valid) else np.nan

    return source.resample("W-SUN").agg(mode_or_nan)


def seam_audit(
    raw_weekly: pd.DataFrame,
    adjusted_weekly: pd.DataFrame,
    source_weekly: pd.Series,
    *,
    window_weeks: int = 26,
) -> pd.DataFrame:
    """Quantify each observed source boundary without treating climate as bias.

    ``standardized_jump`` is a review diagnostic, not an automatic correction:
    a real ENSO state can cross a calendar boundary.  The raw and adjusted
    representations are both reported to make that distinction auditable.
    """
    source = pd.to_numeric(source_weekly, errors="coerce")
    transitions = source.notna() & source.shift().notna() & source.ne(source.shift())
    rows: list[dict[str, object]] = []
    for when in source.index[transitions]:
        old_code = int(source.shift().loc[when])
        new_code = int(source.loc[when])
        before_candidates = source.index[(source.index < when) & source.eq(old_code)]
        after_candidates = source.index[(source.index >= when) & source.eq(new_code)]
        before = before_candidates[-window_weeks:]
        after = after_candidates[:window_weeks]
        for representation, frame in (("raw", raw_weekly), ("source_adjusted_v1", adjusted_weekly)):
            for variable in OCEAN_COLUMNS:
                if variable not in frame:
                    continue
                pre = pd.to_numeric(frame[variable].reindex(before), errors="coerce").dropna()
                post = pd.to_numeric(frame[variable].reindex(after), errors="coerce").dropna()
                mean_pre = float(pre.mean()) if len(pre) else np.nan
                mean_post = float(post.mean()) if len(post) else np.nan
                pooled = float(np.sqrt((pre.var(ddof=1) + post.var(ddof=1)) / 2.0)) if len(pre) > 1 and len(post) > 1 else np.nan
                jump = mean_post - mean_pre
                z = jump / pooled if np.isfinite(pooled) and pooled > 0 else np.nan
                rows.append(
                    {
                        "transition_week": when.date().isoformat(),
                        "source_from": old_code,
                        "source_to": new_code,
                        "representation": representation,
                        "variable": variable,
                        "window_weeks": window_weeks,
                        "n_before": int(len(pre)),
                        "n_after": int(len(post)),
                        "mean_before": mean_pre,
                        "mean_after": mean_post,
                        "jump": jump,
                        "pooled_sd": pooled,
                        "standardized_jump": z,
                        "review_flag_abs_z_gt_2": bool(np.isfinite(z) and abs(z) > 2.0),
                    }
                )
    return pd.DataFrame(rows)


def coverage_audit(master: pd.DataFrame, *, representation: str = "raw") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in PHYSICAL_COLUMNS:
        if column not in master:
            continue
        series = pd.to_numeric(master[column], errors="coerce")
        missing = series.isna()
        if missing.any():
            groups = missing.ne(missing.shift(fill_value=False)).cumsum()
            max_gap = int(missing.groupby(groups).sum().max())
        else:
            max_gap = 0
        rows.append(
            {
                "representation": representation,
                "variavel": column,
                "inicio": series.first_valid_index().date() if series.notna().any() else None,
                "fim": series.last_valid_index().date() if series.notna().any() else None,
                "semanas_validas": int(series.notna().sum()),
                "cobertura_%": round(100.0 * float(series.notna().mean()), 2),
                "maior_lacuna_semanas": max_gap,
            }
        )
    return pd.DataFrame(rows)


def validate_master(master: pd.DataFrame, *, require_full_axis: bool = True) -> pd.DataFrame:
    """Validate schema, temporal axis, provenance metadata, and broad physics."""
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: str, severity: str = "error") -> None:
        checks.append({"checagem": name, "passou": bool(passed), "severidade": severity, "detalhe": detail})

    index = master.index
    add("indice_datetime", isinstance(index, pd.DatetimeIndex), str(type(index).__name__))
    if isinstance(index, pd.DatetimeIndex) and len(index):
        add("indice_monotonico_crescente", index.is_monotonic_increasing, f"rows={len(index)}")
        add("sem_semanas_duplicadas", not index.duplicated().any(), f"duplicadas={int(index.duplicated().sum())}")
        regular = bool((index.to_series().diff().dropna() == pd.Timedelta(weeks=1)).all())
        add("grade_semanal_W-SUN_regular", regular and bool((index.dayofweek == 6).all()), f"{index.min().date()}..{index.max().date()}")
        if require_full_axis:
            add("eixo_1981_ao_ano_atual", index.min().year <= 1981 and index.max().year >= pd.Timestamp.today().year, f"{index.min().year}..{index.max().year}")
    actual_physical = [column for column in master.columns if column not in METADATA_COLUMNS]
    add("contrato_31_variaveis_fisicas", tuple(actual_physical) == PHYSICAL_COLUMNS, f"esperadas=31; encontradas={len(actual_physical)}")
    add("source_code_apenas_metadado", "ocean_source_code" in master and "ocean_source_code" not in PHYSICAL_COLUMNS, "source_code nao e preditor fisico")
    empty = [column for column in PHYSICAL_COLUMNS if column not in master or master[column].notna().sum() == 0]
    add("nenhuma_variavel_totalmente_vazia", not empty, "vazias=" + (",".join(empty) if empty else "nenhuma"))
    stale: list[str] = []
    if isinstance(index, pd.DatetimeIndex) and len(index):
        for column in PHYSICAL_COLUMNS:
            if column not in master or not master[column].notna().any():
                continue
            age = index.max() - master[column].last_valid_index()
            if age > pd.Timedelta(weeks=12):
                stale.append(f"{column}:{age.days // 7}w")
    add(
        "cobertura_final_alinhada_12_semanas",
        not stale,
        "defasadas=" + (",".join(stale) if stale else "nenhuma"),
    )
    numeric = all(column in master and pd.api.types.is_numeric_dtype(master[column]) for column in PHYSICAL_COLUMNS)
    add("variaveis_fisicas_numericas", numeric, "todas as 31 colunas devem ser numericas")
    finite = True
    offending: list[str] = []
    for column in PHYSICAL_COLUMNS:
        if column not in master:
            continue
        values = pd.to_numeric(master[column], errors="coerce").dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            finite = False
            offending.append(column)
    add("sem_inf_nas_variaveis", finite, "colunas=" + (",".join(offending) if offending else "nenhuma"))
    if "ocean_source_code" in master:
        codes = {
            int(value)
            for value in pd.to_numeric(master["ocean_source_code"], errors="coerce").dropna().astype(int).unique()
        }
        add("codigos_fonte_validos", codes.issubset({1, 2, 3}), f"codigos={sorted(codes)}")
        ordered = list(pd.to_numeric(master["ocean_source_code"], errors="coerce").dropna())
        add("fontes_oceanicas_nao_regridem_no_tempo", ordered == sorted(ordered), "ordem esperada UFS->GLORYS->GLO12")
    bounds = {
        "tau_x_anom": 5.0,
        "u10_anom": 100.0,
        "v10_anom": 100.0,
        "mslp_anom": 30_000.0,
        "tcwv_anom": 100.0,
        "slhf_anom": 2_000.0,
        "sshf_anom": 2_000.0,
        "ssr_anom": 2_000.0,
        "str_anom": 2_000.0,
        "u850_anom": 150.0,
        "u200_anom": 200.0,
        "omega850_anom": 20.0,
        "omega500_anom": 20.0,
        "div850_anom": 1.0,
    }
    for column, bound in bounds.items():
        if column in master and master[column].notna().any():
            maximum = float(pd.to_numeric(master[column], errors="coerce").abs().max())
            add(f"escala_plausivel:{column}", maximum <= bound, f"max_abs={maximum:.6g}; limite={bound:g}")
    return pd.DataFrame(checks)
