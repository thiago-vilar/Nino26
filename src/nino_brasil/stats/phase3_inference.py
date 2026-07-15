"""Auditable lag inference for the Phase 3 ENSO signal analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from nino_brasil.events.enso import ENSO_ACTIVE_PHASES
from nino_brasil.stats.significance import benjamini_hochberg_fdr, correlation_p_value


PRECURSOR_TARGET_EXCLUSION_POLICY = (
    "exclude selected signal and aliases nino34_sst*, nino34_ssta*, "
    "nino34_anom*, oni and oni_local*; retain physically localized predictors"
)


def confirmed_friedman_discriminants(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only F3 phase effects confirmed by the registered BH-FDR family."""

    required = {
        "kendall_w_entre_fases",
        "q_friedman_bh",
        "significativo_friedman_fdr",
        "friedman_fdr_alpha",
    }
    if missing := required.difference(frame.columns):
        raise KeyError(
            "phase3 discriminants are missing the confirmatory Friedman FDR "
            f"contract: {sorted(missing)}"
        )
    flags = (
        frame["significativo_friedman_fdr"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .isin({"true", "1", "yes", "sim"})
    )
    q_values = pd.to_numeric(frame["q_friedman_bh"], errors="coerce")
    alpha = pd.to_numeric(frame["friedman_fdr_alpha"], errors="coerce")
    return frame.loc[
        flags & q_values.notna() & alpha.notna() & q_values.le(alpha)
    ].copy()


def is_phase3_target_alias(
    variable: object,
    *,
    selected_signal: object | None = None,
) -> bool:
    """Return whether ``variable`` is the target or an ENSO-target alias.

    The prefix rules are deliberately narrow.  They remove aliases of the
    scalar SST/ONI target while retaining physically localized predictors such
    as ``d20_nino34*``, ``ohc_*_nino34_*`` and
    ``tau_x_anom_nino34_pa``.
    """

    name = str(variable).strip().casefold()
    selected = str(selected_signal).strip().casefold() if selected_signal is not None else ""
    return bool(
        name
        and (
            (selected and name == selected)
            or name.startswith("nino34_sst")
            or name.startswith("nino34_anom")
            or name == "oni"
            or name.startswith("oni_local")
        )
    )


def phase3_precursor_columns(
    columns: Iterable[object],
    *,
    selected_signal: object | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split an ordered column catalogue into precursors and target aliases."""

    names = tuple(str(value) for value in columns)
    excluded = tuple(
        name
        for name in names
        if is_phase3_target_alias(name, selected_signal=selected_signal)
    )
    excluded_set = set(excluded)
    candidates = tuple(name for name in names if name not in excluded_set)
    return candidates, excluded


def _time_index(index: pd.Index, *, name: str) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(index)
    if out.has_duplicates or not out.is_monotonic_increasing:
        raise ValueError(f"{name} must have a unique, increasing DatetimeIndex")
    return out


def _segmented_lag1(values: np.ndarray, times: pd.DatetimeIndex, events: np.ndarray) -> float:
    adjacent = (
        (times[1:] - times[:-1] == pd.Timedelta(days=7))
        & (events[1:] == events[:-1])
        & (events[1:] != "")
    )
    if int(adjacent.sum()) < 3:
        return 0.0
    left = values[:-1][adjacent]
    right = values[1:][adjacent]
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _effective_n_segmented(pairs: pd.DataFrame) -> float:
    n = len(pairs)
    if n < 3:
        return float(n)
    times = pd.DatetimeIndex(pairs["target_time"])
    events = pairs["event_id"].fillna("").astype(str).to_numpy()
    rho_x = _segmented_lag1(pairs["x"].to_numpy(dtype=float), times, events)
    rho_y = _segmented_lag1(pairs["y"].to_numpy(dtype=float), times, events)
    product = float(np.clip(rho_x * rho_y, -0.99, 0.99))
    return float(np.clip(n * (1.0 - product) / (1.0 + product), 3.0, float(n)))


def lagged_pairs(
    predictor: pd.Series,
    target: pd.Series,
    lifecycle: pd.DataFrame,
    *,
    lag_weeks: int,
    event_type: str,
    phase: str,
    condition_time: str = "source",
) -> pd.DataFrame:
    """Return exact weekly pairs for one registered lag and ENSO condition."""

    if int(lag_weeks) != lag_weeks or lag_weeks < 0:
        raise ValueError("lag_weeks must be a non-negative integer")
    if event_type not in {"el_nino", "la_nina"}:
        raise ValueError("event_type must be 'el_nino' or 'la_nina'")
    if phase not in ENSO_ACTIVE_PHASES:
        raise ValueError(f"phase must be one of {ENSO_ACTIVE_PHASES}")
    if condition_time not in {"source", "target"}:
        raise ValueError("condition_time must be 'source' or 'target'")

    x = pd.Series(predictor, copy=True).astype(float)
    y = pd.Series(target, copy=True).astype(float)
    x.index = _time_index(x.index, name="predictor")
    y.index = _time_index(y.index, name="target")
    life = lifecycle.copy()
    life.index = _time_index(life.index, name="lifecycle")
    required = {"tipo", "fase", "event_id"}
    missing = required.difference(life.columns)
    if missing:
        raise KeyError(f"lifecycle is missing columns: {sorted(missing)}")

    target_time = y.index
    source_time = target_time - pd.Timedelta(weeks=int(lag_weeks))
    condition_index = source_time if condition_time == "source" else target_time
    labels = life.reindex(condition_index)
    selected = labels["tipo"].eq(event_type).to_numpy() & labels["fase"].eq(phase).to_numpy()
    frame = pd.DataFrame(
        {
            "source_time": source_time,
            "target_time": target_time,
            "x": x.reindex(source_time).to_numpy(dtype=float),
            "y": y.to_numpy(dtype=float),
            "event_id": labels["event_id"].fillna("").astype(str).to_numpy(),
            "selected": selected,
        }
    )
    valid = frame["selected"] & np.isfinite(frame["x"]) & np.isfinite(frame["y"])
    return frame.loc[valid, ["source_time", "target_time", "x", "y", "event_id"]].reset_index(drop=True)


def simes_field_p_value(p_values: Iterable[float]) -> float:
    """Simes global-null p-value for one pre-registered test family."""

    p = np.asarray(list(p_values), dtype=float)
    p = np.sort(p[np.isfinite(p)])
    if p.size == 0:
        return float("nan")
    ranks = np.arange(1, p.size + 1, dtype=float)
    return float(np.clip(np.min(p * p.size / ranks), 0.0, 1.0))


def scan_lagged_correlations(
    predictors: pd.DataFrame,
    target: pd.Series,
    lifecycle: pd.DataFrame,
    *,
    target_name: str | None = None,
    lags_weeks: Sequence[int] = tuple(range(0, 53)),
    event_types: Sequence[str] = ("el_nino", "la_nina"),
    phases: Sequence[str] = ENSO_ACTIVE_PHASES,
    condition_time: str = "source",
    min_pairs: int = 12,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Scan variables/lags and correct each ENSO type-phase family.

    Serial dependence is estimated only across consecutive pairs belonging to
    the same event.  Benjamini-Hochberg q-values and a Simes field test are
    added within each explicit ``tipo x fase x condition_time`` family.
    """

    frame = predictors.copy()
    frame.index = _time_index(frame.index, name="predictors")
    selected_target = str(target_name or target.name or "").strip()
    candidate_columns, excluded_aliases = phase3_precursor_columns(
        frame.columns,
        selected_signal=selected_target,
    )
    if not candidate_columns:
        raise ValueError("no precursor candidate remains after target-alias exclusion")
    frame = frame.loc[:, list(candidate_columns)]
    excluded_aliases_text = "|".join(excluded_aliases)
    if min_pairs < 4:
        raise ValueError("min_pairs must be at least 4")
    if not 0 < fdr_alpha < 1:
        raise ValueError("fdr_alpha must be in (0, 1)")
    lags = tuple(dict.fromkeys(int(value) for value in lags_weeks))
    if not lags or any(value < 0 for value in lags):
        raise ValueError("lags_weeks must contain non-negative integers")

    rows: list[dict[str, object]] = []
    for event_type in event_types:
        for phase in phases:
            family_id = f"F3_lags_{condition_time}_{event_type}_{phase}"
            for variable in frame.columns:
                predictor = frame[variable]
                for lag in lags:
                    pairs = lagged_pairs(
                        predictor,
                        target,
                        lifecycle,
                        lag_weeks=lag,
                        event_type=event_type,
                        phase=phase,
                        condition_time=condition_time,
                    )
                    n = len(pairs)
                    r = float("nan")
                    n_eff = float("nan")
                    p = float("nan")
                    if n >= min_pairs and np.ptp(pairs["x"]) > 0 and np.ptp(pairs["y"]) > 0:
                        r = float(np.corrcoef(pairs["x"], pairs["y"])[0, 1])
                        n_eff = _effective_n_segmented(pairs)
                        p = correlation_p_value(r, n_eff)
                    rows.append(
                        {
                            "tipo": event_type,
                            "fase": phase,
                            "variavel": str(variable),
                            "variavel_alvo": selected_target,
                            "n_precursores_candidatos": len(candidate_columns),
                            "aliases_alvo_excluidos": excluded_aliases_text,
                            "precursor_screening_policy": PRECURSOR_TARGET_EXCLUSION_POLICY,
                            "lag_semanas": lag,
                            "r_pearson": r,
                            "n_pares": n,
                            "n_eventos": int(pairs["event_id"].nunique()) if n else 0,
                            "n_efetivo_ar1_segmentado": n_eff,
                            "p_efetivo": p,
                            "condition_time": condition_time,
                            "convencao_lag": "X(t-lag) versus alvo(t)",
                            "family_id": family_id,
                            "min_pairs": min_pairs,
                            "fdr_alpha": fdr_alpha,
                            "evaluation_mode": "diagnostico_retrospectivo_inferencial",
                        }
                    )
    result = pd.DataFrame(rows)
    corrected: list[pd.DataFrame] = []
    for family_id, group in result.groupby("family_id", sort=False):
        group = group.copy()
        significant, q_values = benjamini_hochberg_fdr(group["p_efetivo"].to_numpy(), alpha=fdr_alpha)
        field_p = simes_field_p_value(group["p_efetivo"])
        group["q_fdr_bh"] = q_values
        group["significativo_fdr"] = significant
        group["field_p_simes"] = field_p
        group["campo_significativo"] = bool(np.isfinite(field_p) and field_p <= fdr_alpha)
        group["field_test_method"] = "Simes global-null sobre familia pre-registrada"
        group["field_test_assumption"] = (
            "independencia ou dependencia positiva (PRDS); publicar junto da sensibilidade por evento"
        )
        group["rank_abs_r_na_familia"] = group["r_pearson"].abs().rank(method="min", ascending=False)
        corrected.append(group)
    return pd.concat(corrected, ignore_index=True)


def select_best_lags(
    scan: pd.DataFrame,
    *,
    require_fdr: bool = True,
) -> pd.DataFrame:
    """Select one lag per type/phase/variable after registered inference."""

    required = {
        "tipo",
        "fase",
        "variavel",
        "lag_semanas",
        "r_pearson",
        "q_fdr_bh",
        "significativo_fdr",
        "family_id",
    }
    missing = required.difference(scan.columns)
    if missing:
        raise KeyError(f"scan is missing columns: {sorted(missing)}")
    if "variavel_alvo" in scan:
        target_names = scan["variavel_alvo"].dropna().astype(str).unique()
        for target_name in target_names:
            leaked = scan["variavel"].map(
                lambda value: is_phase3_target_alias(value, selected_signal=target_name)
            )
            if leaked.any():
                raise ValueError(
                    "scan contains the selected target or one of its aliases as a precursor"
                )
    rows = []
    for keys, group in scan.groupby(["tipo", "fase", "variavel"], sort=False):
        candidates = group[np.isfinite(group["r_pearson"])].copy()
        if require_fdr:
            candidates = candidates[candidates["significativo_fdr"]]
        if candidates.empty:
            continue
        best = candidates.loc[candidates["r_pearson"].abs().idxmax()].to_dict()
        best["selection_rule"] = "max_abs_r_entre_lags_significativos_fdr" if require_fdr else "max_abs_r_sensibilidade_sem_filtro"
        best["lag_selected_after_fdr"] = bool(require_fdr)
        rows.append(best)
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class EventBootstrapLagResult:
    replicates: pd.DataFrame
    summary: pd.DataFrame


def bootstrap_lag_selection_by_event(
    predictor: pd.Series,
    target: pd.Series,
    lifecycle: pd.DataFrame,
    *,
    predictor_name: str | None = None,
    target_name: str | None = None,
    n_precursor_candidates: int | None = None,
    excluded_target_aliases: Sequence[str] = (),
    screening_rank: int | None = None,
    screening_top_k: int | None = None,
    lags_weeks: Sequence[int],
    event_type: str,
    phase: str,
    condition_time: str = "source",
    n_boot: int = 1_000,
    min_pairs: int = 8,
    random_state: int | np.random.Generator | None = None,
) -> EventBootstrapLagResult:
    """Bootstrap whole events and repeat lag selection in every replicate.

    ``predictor_name`` is checked against the same target-alias policy used by
    the full lag scan.  The screening fields make explicit that this bootstrap
    measures stability only for a predeclared top-k subset selected after the
    original BH-FDR scan; it is not a second variable-selection procedure.
    """

    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    selected_target = str(target_name or target.name or "").strip()
    selected_predictor = str(predictor_name or predictor.name or "").strip()
    if is_phase3_target_alias(selected_predictor, selected_signal=selected_target):
        raise ValueError(
            "the selected target or one of its aliases cannot enter the precursor bootstrap"
        )
    excluded_aliases_text = "|".join(str(value) for value in excluded_target_aliases)
    screening_rule = "top_k_por_tipo_fase_apos_bh_fdr_no_scan_original"
    lags = tuple(dict.fromkeys(int(value) for value in lags_weeks))
    pools: dict[int, dict[str, pd.DataFrame]] = {}
    all_events: set[str] = set()
    for lag in lags:
        pairs = lagged_pairs(
            predictor,
            target,
            lifecycle,
            lag_weeks=lag,
            event_type=event_type,
            phase=phase,
            condition_time=condition_time,
        )
        pools[lag] = {
            event_id: group[["x", "y"]].reset_index(drop=True)
            for event_id, group in pairs.groupby("event_id")
            if event_id
        }
        all_events.update(pools[lag])
    events = sorted(all_events)
    if len(events) < 2:
        raise ValueError("at least two events are required for event bootstrap")
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    for replicate in range(int(n_boot)):
        sampled = rng.choice(events, size=len(events), replace=True)
        candidates: list[tuple[int, float, int]] = []
        for lag in lags:
            blocks = [pools[lag][event] for event in sampled if event in pools[lag]]
            if not blocks:
                continue
            pairs = pd.concat(blocks, ignore_index=True)
            if len(pairs) < min_pairs or np.ptp(pairs["x"]) == 0 or np.ptp(pairs["y"]) == 0:
                continue
            r = float(np.corrcoef(pairs["x"], pairs["y"])[0, 1])
            candidates.append((lag, r, len(pairs)))
        if not candidates:
            rows.append(
                {
                    "bootstrap_replicate": replicate,
                    "variavel_alvo": selected_target,
                    "n_precursores_candidatos": n_precursor_candidates,
                    "aliases_alvo_excluidos": excluded_aliases_text,
                    "bootstrap_screening_rank": screening_rank,
                    "bootstrap_screening_top_k": screening_top_k,
                    "bootstrap_screening_rule": screening_rule,
                    "lag_selecionado_semanas": np.nan,
                    "r_selecionado": np.nan,
                    "n_pares": 0,
                    "eventos_amostrados": "|".join(sampled),
                }
            )
            continue
        lag, r, n_pairs = max(candidates, key=lambda value: abs(value[1]))
        rows.append(
            {
                "bootstrap_replicate": replicate,
                "variavel_alvo": selected_target,
                "n_precursores_candidatos": n_precursor_candidates,
                "aliases_alvo_excluidos": excluded_aliases_text,
                "bootstrap_screening_rank": screening_rank,
                "bootstrap_screening_top_k": screening_top_k,
                "bootstrap_screening_rule": screening_rule,
                "lag_selecionado_semanas": lag,
                "r_selecionado": r,
                "n_pares": n_pairs,
                "eventos_amostrados": "|".join(sampled),
            }
        )
    replicates = pd.DataFrame(rows)
    valid = replicates.dropna(subset=["lag_selecionado_semanas", "r_selecionado"])
    summary_rows = []
    for lag in lags:
        selected = valid[valid["lag_selecionado_semanas"].eq(lag)]
        summary_rows.append(
            {
                "tipo": event_type,
                "fase": phase,
                "variavel_alvo": selected_target,
                "n_precursores_candidatos": n_precursor_candidates,
                "aliases_alvo_excluidos": excluded_aliases_text,
                "bootstrap_screening_rank": screening_rank,
                "bootstrap_screening_top_k": screening_top_k,
                "bootstrap_screening_rule": screening_rule,
                "lag_semanas": lag,
                "n_eventos_independentes": len(events),
                "bootstrap_n": int(n_boot),
                "bootstrap_validos": int(len(valid)),
                "vezes_selecionado": int(len(selected)),
                "frequencia_selecao": float(len(selected) / len(valid)) if len(valid) else np.nan,
                "r_mediano_quando_selecionado": float(selected["r_selecionado"].median()) if len(selected) else np.nan,
                "r_ic95_inf_quando_selecionado": float(selected["r_selecionado"].quantile(0.025)) if len(selected) else np.nan,
                "r_ic95_sup_quando_selecionado": float(selected["r_selecionado"].quantile(0.975)) if len(selected) else np.nan,
                "resampling_unit": "evento_enso_completo",
                "lag_selection_repeated_inside_bootstrap": True,
                "condition_time": condition_time,
            }
        )
    return EventBootstrapLagResult(replicates=replicates, summary=pd.DataFrame(summary_rows))
