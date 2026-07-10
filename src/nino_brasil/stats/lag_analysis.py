"""Exact lagged signal/response analysis for Phase 4C.

The central convention is explicit: for a response observed at week ``t`` and
lag ``L``, the predictor and its ENSO phase are read at ``t - L``.  Predictors
are analysed one at a time.  The implementation vectorises the independent
pixel calculations inside one predictor/lag only; it never batches predictors
or changes the Pearson statistic.

Serial-correlation corrections are estimated only from truly consecutive
weekly pairs.  For phase-conditioned analyses the two weeks must additionally
belong to the same 4A ``event_id``.  This prevents the end of one ENSO event
from being treated as adjacent to the beginning of another event.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy import stats


PHASE_ORDER: tuple[str, ...] = ("genese", "crescimento", "pico", "decaimento")
ALREADY_ANOMALOUS_DEFAULT: frozenset[str] = frozenset(
    {
        "nino34_ssta",
        "tau_x_anom",
        "u10_anom",
        "v10_anom",
        "mslp_anom",
        "tcwv_anom",
        "slhf_anom",
        "sshf_anom",
        "ssr_anom",
        "str_anom",
        "u850_anom",
        "u200_anom",
        "omega850_anom",
        "omega500_anom",
        "div850_anom",
    }
)


@dataclass(frozen=True)
class SourceCondition:
    """A condition evaluated at the predictor week, never at response week."""

    name: str
    source_mask: pd.Series
    tipo_enso_fonte: str
    fase_fonte_em_t_menos_lag: str
    require_same_event_for_ar1: bool
    description: str


def _validate_weekly_index(index: pd.Index, *, name: str) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(index)
    if out.has_duplicates:
        raise ValueError(f"{name} must have a unique weekly DatetimeIndex.")
    if not out.is_monotonic_increasing:
        raise ValueError(f"{name} must be sorted chronologically.")
    return out


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y", "sim", "s"})


def load_selected_predictors(
    statistics_dir: str | Path,
    available_predictors: Iterable[str],
    *,
    explicit_predictors: Sequence[str] | None = None,
    allow_all_fallback: bool = False,
) -> tuple[list[str], str]:
    """Load the Phase 4 variable selection contract with an auditable fallback.

    Priority:
    1. 4.0 stable selection (``phase40_variaveis_selecionadas.csv``);
    2. 4B stable selection, when present;
    3. 4.0 redundancy audit, for compatibility while notebooks migrate;
    4. an explicit list supplied by the caller;
    5. all available predictors only when ``allow_all_fallback=True``.

    The default is deliberately strict: Phase 4C must not silently undo the
    mandatory dimensionality reduction from 4.0.
    """

    root = Path(statistics_dir)
    available = list(dict.fromkeys(str(v) for v in available_predictors))
    available_set = set(available)
    contracts = (
        ("phase40_variaveis_selecionadas.csv", "selecionada", "4.0:selecao_estavel"),
        ("phase4B_selecao_estavel.csv", "selecionada_estavel", "4B:selecao_estavel"),
        ("phase40_reducao_variaveis.csv", "retida_redundancia", "4.0:reducao_redundancia"),
    )
    for filename, flag_column, source in contracts:
        path = root / filename
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if "variavel" not in table or flag_column not in table:
            warnings.warn(
                f"Ignoring {path.name}: expected columns 'variavel' and "
                f"'{flag_column}'.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        selected = table.loc[_as_bool(table[flag_column]), "variavel"].astype(str)
        selected = [v for v in selected if v in available_set]
        selected = list(dict.fromkeys(selected))
        if selected:
            return selected, source
        warnings.warn(
            f"{path.name} did not select any predictor available to Phase 4C.",
            RuntimeWarning,
            stacklevel=2,
        )

    if explicit_predictors is not None:
        selected = [str(v) for v in explicit_predictors if str(v) in available_set]
        selected = list(dict.fromkeys(selected))
        if not selected:
            raise ValueError("explicit_predictors contains no available predictor.")
        warnings.warn(
            "Phase 4C is using an explicit predictor list because no valid 4.0/4B "
            "selection contract was found.",
            RuntimeWarning,
            stacklevel=2,
        )
        return selected, "lista_explicita_fallback"

    if not available:
        raise ValueError("No predictor is available for Phase 4C.")
    if not allow_all_fallback:
        raise FileNotFoundError(
            "Phase 4C requires a non-empty variable-selection contract from 4.0 "
            "(phase40_variaveis_selecionadas.csv). Set allow_all_fallback=True "
            "only for an explicitly documented sensitivity run or a unit test."
        )
    warnings.warn(
        "No valid 4.0/4B selection contract was found; Phase 4C will analyse all "
        "available predictors. This fallback is recorded in every output.",
        RuntimeWarning,
        stacklevel=2,
    )
    return available, "todas_disponiveis_fallback"


def build_source_conditions(
    phase_table: pd.DataFrame,
    *,
    phase_order: Sequence[str] = PHASE_ORDER,
    include_all_weeks: bool = True,
    include_event_aggregates: bool = True,
    include_la_nina: bool = True,
) -> dict[str, SourceCondition]:
    """Build conditions from 4A, evaluated at source week ``t - lag``."""

    required = {"fase", "tipo", "event_id"}
    missing = required.difference(phase_table.columns)
    if missing:
        raise KeyError(f"phase_table is missing columns: {sorted(missing)}")
    phases = phase_table.copy()
    phases.index = _validate_weekly_index(phases.index, name="phase_table")
    phases["fase"] = phases["fase"].fillna("neutro").astype(str)
    phases["tipo"] = phases["tipo"].fillna("neutro").astype(str)
    phases["event_id"] = phases["event_id"].fillna("").astype(str)

    out: dict[str, SourceCondition] = {}
    if include_all_weeks:
        out["todas"] = SourceCondition(
            name="todas",
            source_mask=pd.Series(True, index=phases.index),
            tipo_enso_fonte="todos",
            fase_fonte_em_t_menos_lag="todas",
            require_same_event_for_ar1=False,
            description="todas as semanas-fonte validas",
        )

    event_types = ["el_nino", *( ["la_nina"] if include_la_nina else [] )]
    for event_type in event_types:
        pretty = "El Nino" if event_type == "el_nino" else "La Nina"
        active = phases["tipo"].eq(event_type) & phases["fase"].isin(phase_order)
        if include_event_aggregates:
            name = f"{event_type}_todas_fases"
            out[name] = SourceCondition(
                name=name,
                source_mask=active,
                tipo_enso_fonte=event_type,
                fase_fonte_em_t_menos_lag="todas_fases_ativas",
                require_same_event_for_ar1=True,
                description=f"semana-fonte em qualquer fase ativa de {pretty}",
            )
        for phase in phase_order:
            name = f"{event_type}_{phase}"
            out[name] = SourceCondition(
                name=name,
                source_mask=phases["tipo"].eq(event_type) & phases["fase"].eq(phase),
                tipo_enso_fonte=event_type,
                fase_fonte_em_t_menos_lag=str(phase),
                require_same_event_for_ar1=True,
                description=f"fase-fonte {phase} de {pretty} em t-lag",
            )
    return out


def harmonic_deseasonalize_predictors(
    predictors: pd.DataFrame,
    *,
    already_anomalous: Iterable[str] = ALREADY_ANOMALOUS_DEFAULT,
    baseline: tuple[str, str] = ("1991-01-01", "2020-12-31"),
    n_harmonics: int = 3,
    min_baseline_observations: int = 104,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove the fitted weekly seasonal cycle from physical-level predictors.

    Predictors explicitly supplied as anomalies are preserved byte-for-byte
    (apart from numeric coercion).  Other series are regressed on an intercept
    and annual sine/cosine pairs fitted only on the stated climatological base.
    """

    if n_harmonics < 1:
        raise ValueError("n_harmonics must be at least 1.")
    frame = predictors.copy()
    frame.index = _validate_weekly_index(frame.index, name="predictors")
    anomaly_names = set(already_anomalous)
    day = frame.index.dayofyear.to_numpy(dtype=float) - 1.0
    angle = 2.0 * np.pi * day / 365.2425
    design_parts = [np.ones(len(frame), dtype=float)]
    for harmonic in range(1, n_harmonics + 1):
        design_parts.extend((np.sin(harmonic * angle), np.cos(harmonic * angle)))
    design = np.column_stack(design_parts)
    baseline_mask = (frame.index >= pd.Timestamp(baseline[0])) & (
        frame.index <= pd.Timestamp(baseline[1])
    )

    out = pd.DataFrame(index=frame.index)
    metadata: list[dict[str, object]] = []
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        valid_baseline = baseline_mask & np.isfinite(values)
        if column in anomaly_names or str(column).endswith("_anom"):
            out[column] = values
            treatment = "previamente_anomalia"
            fitted_n = int(valid_baseline.sum())
        else:
            if int(valid_baseline.sum()) < max(min_baseline_observations, design.shape[1] + 2):
                raise ValueError(
                    f"Predictor {column!r} has only {int(valid_baseline.sum())} valid "
                    "baseline weeks; harmonic deseasonalisation is not defensible."
                )
            coefficients, *_ = np.linalg.lstsq(
                design[valid_baseline], values[valid_baseline], rcond=None
            )
            seasonal_climatology = design @ coefficients
            out[column] = values - seasonal_climatology
            treatment = f"anomalia_harmonica_{n_harmonics}"
            fitted_n = int(valid_baseline.sum())
        metadata.append(
            {
                "variavel": str(column),
                "tratamento_sazonal_predictor": treatment,
                "climatologia_inicio": baseline[0],
                "climatologia_fim": baseline[1],
                "n_harmonicos": 0 if treatment == "previamente_anomalia" else n_harmonics,
                "n_semanas_ajuste": fitted_n,
            }
        )
    return out, pd.DataFrame(metadata)


def _source_dates(target_index: pd.DatetimeIndex, lag_weeks: int) -> pd.DatetimeIndex:
    if int(lag_weeks) != lag_weeks or lag_weeks < 0:
        raise ValueError("lags must be non-negative integer weeks.")
    return target_index - pd.to_timedelta(int(lag_weeks), unit="W")


def _align_at_source(
    series: pd.Series,
    target_index: pd.DatetimeIndex,
    lag_weeks: int,
) -> np.ndarray:
    source_index = _source_dates(target_index, lag_weeks)
    return series.reindex(source_index).to_numpy()


def _columnwise_corr(
    left: np.ndarray,
    right: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Pearson r by column for an explicitly supplied validity mask."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim == 1:
        left = left[:, None]
    if right.ndim == 1:
        right = right[:, None]
    valid = np.asarray(valid, dtype=bool)
    n_columns = max(left.shape[1], right.shape[1], valid.shape[1])
    if left.shape[1] == 1 and n_columns > 1:
        left = np.broadcast_to(left, (left.shape[0], n_columns))
    if right.shape[1] == 1 and n_columns > 1:
        right = np.broadcast_to(right, (right.shape[0], n_columns))
    if valid.shape[1] == 1 and n_columns > 1:
        valid = np.broadcast_to(valid, (valid.shape[0], n_columns))
    if left.shape != right.shape or left.shape != valid.shape:
        raise ValueError("left, right and valid masks are not broadcast-compatible.")

    count = valid.sum(axis=0).astype(float)
    safe_count = np.where(count > 0, count, 1.0)
    left_sum = np.where(valid, left, 0.0).sum(axis=0)
    right_sum = np.where(valid, right, 0.0).sum(axis=0)
    left_centered = np.where(valid, left - left_sum / safe_count, 0.0)
    right_centered = np.where(valid, right - right_sum / safe_count, 0.0)
    numerator = (left_centered * right_centered).sum(axis=0)
    denominator = np.sqrt(
        (left_centered**2).sum(axis=0) * (right_centered**2).sum(axis=0)
    )
    correlation = np.divide(
        numerator,
        denominator,
        out=np.full(n_columns, np.nan, dtype=float),
        where=denominator > 0,
    )
    correlation[count < 2] = np.nan
    return np.clip(correlation, -1.0, 1.0), count.astype(int)


def _pooled_within_segment_corr(
    left: np.ndarray,
    right: np.ndarray,
    valid: np.ndarray,
    segments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool covariance after centring each event segment independently."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if left.ndim == 1:
        left = left[:, None]
    if right.ndim == 1:
        right = right[:, None]
    n_columns = max(left.shape[1], right.shape[1], valid.shape[1])
    if left.shape[1] == 1 and n_columns > 1:
        left = np.broadcast_to(left, (left.shape[0], n_columns))
    if right.shape[1] == 1 and n_columns > 1:
        right = np.broadcast_to(right, (right.shape[0], n_columns))
    if valid.shape[1] == 1 and n_columns > 1:
        valid = np.broadcast_to(valid, (valid.shape[0], n_columns))
    if left.shape != right.shape or left.shape != valid.shape:
        raise ValueError("left, right and valid masks are not broadcast-compatible.")
    segments = pd.Series(segments, dtype="string").fillna("").to_numpy(dtype=str)
    if len(segments) != left.shape[0]:
        raise ValueError("segments length differs from the paired arrays.")

    numerator = np.zeros(n_columns, dtype=float)
    left_ss = np.zeros(n_columns, dtype=float)
    right_ss = np.zeros(n_columns, dtype=float)
    total_count = np.zeros(n_columns, dtype=int)
    for segment in pd.unique(segments):
        if not segment:
            continue
        segment_valid = valid & (segments == segment)[:, None]
        count = segment_valid.sum(axis=0).astype(int)
        usable = count >= 2
        if not usable.any():
            continue
        safe_count = np.where(count > 0, count, 1)
        left_mean = np.where(segment_valid, left, 0.0).sum(axis=0) / safe_count
        right_mean = np.where(segment_valid, right, 0.0).sum(axis=0) / safe_count
        left_centered = np.where(segment_valid, left - left_mean, 0.0)
        right_centered = np.where(segment_valid, right - right_mean, 0.0)
        numerator += np.where(usable, (left_centered * right_centered).sum(axis=0), 0.0)
        left_ss += np.where(usable, (left_centered**2).sum(axis=0), 0.0)
        right_ss += np.where(usable, (right_centered**2).sum(axis=0), 0.0)
        total_count += np.where(usable, count, 0)
    denominator = np.sqrt(left_ss * right_ss)
    correlation = np.divide(
        numerator,
        denominator,
        out=np.full(n_columns, np.nan, dtype=float),
        where=denominator > 0,
    )
    correlation[total_count < 2] = np.nan
    return np.clip(correlation, -1.0, 1.0), total_count


def pearson_columns_with_segmented_neff(
    predictor: np.ndarray,
    response: np.ndarray,
    target_index: pd.DatetimeIndex,
    *,
    source_event_id: np.ndarray | None = None,
    min_pairs: int = 30,
    min_ar1_pairs: int = 3,
    confidence: float = 0.95,
) -> dict[str, np.ndarray]:
    """Pearson correlation and Bretherton N_eff for one predictor vs columns.

    ``source_event_id`` activates event segmentation for the AR(1) estimate.
    Correlation itself uses every valid conditioned pair; only the AR(1) pairs
    are restricted to consecutive weeks within the same event.
    """

    times = _validate_weekly_index(target_index, name="target_index")
    x = np.asarray(predictor, dtype=float).reshape(-1)
    y = np.asarray(response, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if len(times) != len(x) or y.shape[0] != len(x):
        raise ValueError("predictor, response and target_index lengths differ.")
    valid = np.isfinite(x)[:, None] & np.isfinite(y)
    r, n_pairs = _columnwise_corr(x, y, valid)

    consecutive = np.zeros(len(times) - 1, dtype=bool)
    if len(times) > 1:
        consecutive = np.diff(times.values).astype("timedelta64[D]") == np.timedelta64(7, "D")
    if source_event_id is not None:
        segments = pd.Series(source_event_id, dtype="string").fillna("").to_numpy(dtype=str)
        if len(segments) != len(times):
            raise ValueError("source_event_id length differs from target_index.")
        consecutive &= (segments[:-1] != "") & (segments[:-1] == segments[1:])

    ar_valid = (
        consecutive[:, None]
        & valid[:-1]
        & valid[1:]
        & np.isfinite(x[:-1])[:, None]
        & np.isfinite(x[1:])[:, None]
    )
    if source_event_id is None:
        rho_x, n_ar_pairs = _columnwise_corr(x[:-1], x[1:], ar_valid)
        rho_y, _ = _columnwise_corr(y[:-1], y[1:], ar_valid)
    else:
        # Each AR(1) covariance is centred inside its own ENSO event.  Pooling
        # globally would mistake between-event mean differences for persistence.
        pair_segments = segments[1:]
        rho_x, n_ar_pairs = _pooled_within_segment_corr(
            x[:-1], x[1:], ar_valid, pair_segments
        )
        rho_y, _ = _pooled_within_segment_corr(
            y[:-1], y[1:], ar_valid, pair_segments
        )

    product = np.clip(rho_x * rho_y, -0.99, 0.99)
    n_eff = n_pairs.astype(float) * (1.0 - product) / (1.0 + product)
    n_eff = np.minimum(n_eff, n_pairs.astype(float))  # conservative for negative AR products
    n_eff = np.maximum(n_eff, 4.0)
    inferentially_valid = (
        (n_pairs >= min_pairs)
        & (n_ar_pairs >= min_ar1_pairs)
        & np.isfinite(r)
        & np.isfinite(n_eff)
    )
    n_eff[~inferentially_valid] = np.nan
    r[n_pairs < min_pairs] = np.nan

    denominator = np.clip(1.0 - r**2, 1e-15, None)
    t_statistic = r * np.sqrt((n_eff - 2.0) / denominator)
    p_value = 2.0 * stats.t.sf(np.abs(t_statistic), n_eff - 2.0)
    p_value[~inferentially_valid] = np.nan

    alpha = 1.0 - confidence
    critical = stats.norm.ppf(1.0 - alpha / 2.0)
    r_clipped = np.clip(r, -0.999999, 0.999999)
    z = np.arctanh(r_clipped)
    se = 1.0 / np.sqrt(n_eff - 3.0)
    ci_low = np.tanh(z - critical * se)
    ci_high = np.tanh(z + critical * se)
    ci_low[~inferentially_valid] = np.nan
    ci_high[~inferentially_valid] = np.nan

    return {
        "r": r,
        "p": p_value,
        "n_pairs": n_pairs.astype(np.int32),
        "n_eff": n_eff,
        "rho1_predictor": rho_x,
        "rho1_response": rho_y,
        "n_ar1_pairs": n_ar_pairs.astype(np.int32),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def lagged_correlation_exact(
    predictor: pd.Series,
    response: pd.DataFrame,
    lags: Sequence[int],
    condition: SourceCondition,
    phase_table: pd.DataFrame,
    *,
    min_pairs: int = 30,
    min_ar1_pairs: int = 3,
    confidence: float = 0.95,
) -> dict[str, np.ndarray]:
    """Calculate one predictor's exact lag correlation against response columns."""

    target_index = _validate_weekly_index(response.index, name="response")
    predictor = pd.to_numeric(predictor, errors="coerce").copy()
    predictor.index = _validate_weekly_index(predictor.index, name="predictor")
    response_values = response.to_numpy(dtype=float)
    phase_events = phase_table["event_id"].fillna("").astype(str)

    fields = (
        "r",
        "p",
        "n_pairs",
        "n_eff",
        "rho1_predictor",
        "rho1_response",
        "n_ar1_pairs",
        "ci_low",
        "ci_high",
    )
    collected: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    n_condition_weeks: list[int] = []
    for lag in lags:
        source_x = _align_at_source(predictor, target_index, int(lag)).astype(float)
        source_mask = _align_at_source(
            condition.source_mask.astype(bool), target_index, int(lag)
        ).astype(object)
        source_mask = pd.Series(source_mask).fillna(False).astype(bool).to_numpy()
        source_x[~source_mask] = np.nan
        source_event = None
        if condition.require_same_event_for_ar1:
            source_event = _align_at_source(phase_events, target_index, int(lag))
            source_event = pd.Series(source_event, dtype="string").fillna("").to_numpy(dtype=str)
        result = pearson_columns_with_segmented_neff(
            source_x,
            response_values,
            target_index,
            source_event_id=source_event,
            min_pairs=min_pairs,
            min_ar1_pairs=min_ar1_pairs,
            confidence=confidence,
        )
        for field in fields:
            collected[field].append(result[field])
        n_condition_weeks.append(int(np.isfinite(source_x).sum()))

    out = {field: np.stack(values, axis=0) for field, values in collected.items()}
    out["lags"] = np.asarray(lags, dtype=np.int16)
    out["columns"] = np.asarray(response.columns, dtype=object)
    out["n_condition_source_weeks"] = np.asarray(n_condition_weeks, dtype=np.int32)
    return out


def fdr_bh(p_values: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg mask over all finite tests supplied by the caller."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    p = np.asarray(p_values, dtype=float)
    flat = p.ravel()
    finite = np.isfinite(flat)
    rejected = np.zeros(flat.shape, dtype=bool)
    if not finite.any():
        return rejected.reshape(p.shape)
    values = flat[finite]
    order = np.argsort(values, kind="mergesort")
    sorted_p = values[order]
    threshold = alpha * np.arange(1, len(sorted_p) + 1) / len(sorted_p)
    accepted = sorted_p <= threshold
    if accepted.any():
        cutoff = sorted_p[np.flatnonzero(accepted)[-1]]
        rejected[finite] = values <= cutoff
    return rejected.reshape(p.shape)


def _take_at_index(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    columns = np.arange(values.shape[1])
    safe = np.where(indices >= 0, indices, 0)
    out = values[safe, columns].astype(float, copy=True)
    out[indices < 0] = np.nan
    return out


def best_lag_fields(
    result: Mapping[str, np.ndarray],
    *,
    significant: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Select the smallest lag tied for maximum absolute effect by response column."""

    r = np.asarray(result["r"], dtype=float)
    candidates = np.isfinite(r)
    if significant is not None:
        candidates &= np.asarray(significant, dtype=bool)
    score = np.where(candidates, np.abs(r), -np.inf)
    valid = candidates.any(axis=0)
    indices = np.argmax(score, axis=0).astype(int)
    indices[~valid] = -1
    out: dict[str, np.ndarray] = {
        "best_lag_sem": _take_at_index(
            np.broadcast_to(np.asarray(result["lags"])[:, None], r.shape), indices
        ),
        "r_no_best_lag": _take_at_index(r, indices),
        "p_no_best_lag": _take_at_index(np.asarray(result["p"]), indices),
        "n_no_best_lag": _take_at_index(np.asarray(result["n_pairs"]), indices),
        "n_eff_no_best_lag": _take_at_index(np.asarray(result["n_eff"]), indices),
        "ci_low_no_best_lag": _take_at_index(np.asarray(result["ci_low"]), indices),
        "ci_high_no_best_lag": _take_at_index(np.asarray(result["ci_high"]), indices),
        "n_ar1_pairs_no_best_lag": _take_at_index(
            np.asarray(result["n_ar1_pairs"]), indices
        ),
    }
    out["best_index"] = indices
    return out


def lag_window_inventory(
    predictor: pd.Series,
    target_index: pd.DatetimeIndex,
    lags: Sequence[int],
    conditions: Mapping[str, SourceCondition],
) -> pd.DataFrame:
    """Audit every moving source/response window and its source-phase rule."""

    target = _validate_weekly_index(target_index, name="target_index")
    predictor = pd.to_numeric(predictor, errors="coerce")
    rows: list[dict[str, object]] = []
    for condition in conditions.values():
        for lag in lags:
            source_dates = _source_dates(target, int(lag))
            x = predictor.reindex(source_dates).to_numpy(dtype=float)
            mask = condition.source_mask.reindex(source_dates).fillna(False).to_numpy(dtype=bool)
            valid = mask & np.isfinite(x)
            target_valid = target[valid]
            source_valid = source_dates[valid]
            rows.append(
                {
                    "condicao_fonte": condition.name,
                    "tipo_enso_fonte": condition.tipo_enso_fonte,
                    "fase_fonte_em_t_menos_lag": condition.fase_fonte_em_t_menos_lag,
                    "lag_sem": int(lag),
                    "janela_resposta_inicio": target_valid.min().date() if valid.any() else "",
                    "janela_resposta_fim": target_valid.max().date() if valid.any() else "",
                    "janela_fonte_inicio": source_valid.min().date() if valid.any() else "",
                    "janela_fonte_fim": source_valid.max().date() if valid.any() else "",
                    "n_semanas_fonte_validas": int(valid.sum()),
                    "segmentacao_ar1_por_event_id": condition.require_same_event_for_ar1,
                    "regra_pareamento": "Pacifico(t-lag) versus chuva(t)",
                }
            )
    return pd.DataFrame(rows)


def result_to_long_table(
    result: Mapping[str, np.ndarray],
    *,
    predictor_name: str,
    condition: SourceCondition,
    column_name: str,
    alpha_fdr: float = 0.10,
) -> pd.DataFrame:
    """Convert a lag result to an auditable long table and apply FDR once."""

    rejected = fdr_bh(np.asarray(result["p"]), alpha=alpha_fdr)
    lags = np.asarray(result["lags"])
    columns = np.asarray(result["columns"], dtype=object)
    lag_grid, column_grid = np.meshgrid(lags, columns, indexing="ij")
    frame = pd.DataFrame(
        {
            "variavel": predictor_name,
            "condicao_fonte": condition.name,
            "tipo_enso_fonte": condition.tipo_enso_fonte,
            "fase_fonte_em_t_menos_lag": condition.fase_fonte_em_t_menos_lag,
            "lag_sem": lag_grid.ravel(),
            column_name: column_grid.ravel(),
            "r": np.asarray(result["r"]).ravel(),
            "p": np.asarray(result["p"]).ravel(),
            "fdr_bh_0_10": rejected.ravel(),
            "n_pares": np.asarray(result["n_pairs"]).ravel(),
            "n_eff_bretherton": np.asarray(result["n_eff"]).ravel(),
            "rho1_predictor": np.asarray(result["rho1_predictor"]).ravel(),
            "rho1_resposta": np.asarray(result["rho1_response"]).ravel(),
            "n_pares_ar1_consecutivos_mesmo_evento": np.asarray(
                result["n_ar1_pairs"]
            ).ravel(),
            "ic95_r_inferior": np.asarray(result["ci_low"]).ravel(),
            "ic95_r_superior": np.asarray(result["ci_high"]).ravel(),
        }
    )
    frame["metodo"] = (
        "Pearson exato por variavel; N_eff Bretherton com AR1 apenas em semanas "
        "consecutivas do mesmo event_id; FDR BH"
    )
    return frame


def best_from_long_table(
    table: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    require_fdr: bool,
) -> pd.DataFrame:
    """Select maximum |r| per group, resolving ties toward the shortest lag."""

    work = table.copy()
    work = work[np.isfinite(pd.to_numeric(work["r"], errors="coerce"))]
    if require_fdr:
        work = work[_as_bool(work["fdr_bh_0_10"])]
    if work.empty:
        return work
    work["abs_r"] = work["r"].abs()
    work = work.sort_values(
        [*group_columns, "abs_r", "lag_sem"],
        ascending=[*[True] * len(group_columns), False, True],
        kind="mergesort",
    )
    best = work.groupby(list(group_columns), sort=False, as_index=False).head(1).copy()
    best = best.drop(columns="abs_r")
    best["criterio_melhor_lag"] = (
        "maior_abs_r_FDR" if require_fdr else "maior_abs_r_descritivo"
    )
    return best.reset_index(drop=True)
