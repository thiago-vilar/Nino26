"""Canonical ENSO event and lifecycle contracts.

The labels in this module are deliberately split into two products:

* retrospective lifecycle labels, which use the completed event (including
  its peak) and are suitable for diagnostics and composites; and
* rolling-origin targets, whose feature cut-off is the forecast origin and
  whose future values are represented only as prediction targets.

This distinction prevents a peak date learned from the future from silently
becoming a predictor in Phase 3 or in downstream Phase 5/7 experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


ENSO_ACTIVE_PHASES: tuple[str, ...] = (
    "genese",
    "crescimento",
    "pico",
    "decaimento",
)
ENSO_TYPES: tuple[str, ...] = ("el_nino", "la_nina")
ENSO_STATE_ORDER: tuple[str, ...] = (
    "neutro",
    *(f"{event_type}_{phase}" for event_type in ENSO_TYPES for phase in ENSO_ACTIVE_PHASES),
)


@dataclass(frozen=True)
class EnsoLifecycleConfig:
    """Parameters defining the canonical local-ONI lifecycle.

    ``peak_fraction`` defines a peak *band*, not a single month.  A month is
    in the peak band when the signed event magnitude is at least this fraction
    of that event's maximum magnitude.  The project default is 0.90 and every
    official result must also publish the configured sensitivity fractions.
    """

    threshold_c: float = 0.5
    min_consecutive_seasons: int = 5
    genesis_weeks: int = 26
    peak_fraction: float = 0.90
    peak_sensitivity_fractions: tuple[float, ...] = (0.80, 0.90, 0.95)

    def __post_init__(self) -> None:
        if self.threshold_c <= 0:
            raise ValueError("threshold_c must be positive")
        if self.min_consecutive_seasons < 1:
            raise ValueError("min_consecutive_seasons must be positive")
        if self.genesis_weeks < 1:
            raise ValueError("genesis_weeks must be positive")
        if not 0 < self.peak_fraction <= 1:
            raise ValueError("peak_fraction must be in (0, 1]")
        if not self.peak_sensitivity_fractions:
            raise ValueError("peak_sensitivity_fractions cannot be empty")
        if any(not 0 < value <= 1 for value in self.peak_sensitivity_fractions):
            raise ValueError("every sensitivity fraction must be in (0, 1]")


def _monthly_series(values: pd.Series) -> pd.Series:
    series = pd.Series(values, copy=True).astype(float).dropna()
    index = pd.DatetimeIndex(series.index)
    if index.has_duplicates:
        raise ValueError("monthly signal must have a unique DatetimeIndex")
    index = index.to_period("M").to_timestamp(how="start")
    if index.has_duplicates:
        raise ValueError("monthly signal contains more than one value per month")
    series.index = index
    return series.sort_index().rename(values.name or "oni_local_c")


def _weekly_index(values: Iterable[pd.Timestamp] | pd.DatetimeIndex) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(values)
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("weekly_index must be unique and increasing")
    if len(index) > 1 and not np.all(index[1:] - index[:-1] == pd.Timedelta(days=7)):
        raise ValueError("weekly_index must be a complete seven-day axis")
    return index


def _intensity_class(magnitude_c: float) -> str:
    if magnitude_c >= 2.0:
        return "muito_forte"
    if magnitude_c >= 1.5:
        return "forte"
    if magnitude_c >= 1.0:
        return "moderado"
    if magnitude_c >= 0.5:
        return "fraco"
    return "neutro"


def _true_monthly_runs(mask: pd.Series) -> list[pd.DatetimeIndex]:
    runs: list[list[pd.Timestamp]] = []
    current: list[pd.Timestamp] = []
    previous: pd.Timestamp | None = None
    for timestamp, selected in mask.items():
        timestamp = pd.Timestamp(timestamp)
        consecutive = previous is not None and timestamp == previous + pd.offsets.MonthBegin(1)
        if bool(selected):
            if current and not consecutive:
                runs.append(current)
                current = []
            current.append(timestamp)
        elif current:
            runs.append(current)
            current = []
        previous = timestamp
    if current:
        runs.append(current)
    return [pd.DatetimeIndex(run) for run in runs]


def _peak_band(signal: pd.Series, sign: float, fraction: float) -> tuple[pd.Timestamp, pd.Timestamp]:
    magnitude = sign * signal
    selected = magnitude >= float(fraction) * float(magnitude.max())
    # Retain the contiguous component containing the absolute peak.  This
    # avoids joining two separated shoulders into a fictitious long plateau.
    peak_time = pd.Timestamp(magnitude.idxmax())
    candidates = _true_monthly_runs(selected)
    component = next(run for run in candidates if peak_time in run)
    return pd.Timestamp(component.min()), pd.Timestamp(component.max()) + pd.offsets.MonthEnd(1)


def detect_enso_events(
    oni_monthly: pd.Series,
    *,
    config: EnsoLifecycleConfig = EnsoLifecycleConfig(),
) -> pd.DataFrame:
    """Detect El Nino and La Nina events symmetrically from a local ONI.

    Missing months always break a run.  The event table stores the canonical
    90% (configurable) peak-band boundaries so all phases use one definition.
    """

    oni = _monthly_series(oni_monthly)
    rows: list[dict[str, object]] = []
    for event_type, sign in (("el_nino", 1.0), ("la_nina", -1.0)):
        active = sign * oni >= config.threshold_c
        for run in _true_monthly_runs(active):
            if len(run) < config.min_consecutive_seasons:
                continue
            values = oni.loc[run]
            magnitude = sign * values
            peak_time = pd.Timestamp(magnitude.idxmax())
            peak_signed = float(values.loc[peak_time])
            peak_magnitude = float(magnitude.loc[peak_time])
            start = pd.Timestamp(run.min())
            end = pd.Timestamp(run.max()) + pd.offsets.MonthEnd(1)
            peak_start, peak_end = _peak_band(values, sign, config.peak_fraction)
            suffix = f"{start.year}_{end.year}" if start.year != end.year else str(start.year)
            rows.append(
                {
                    "event_id": f"{event_type}_{suffix}",
                    "tipo": event_type,
                    "classe": _intensity_class(peak_magnitude),
                    "onset": start,
                    "pico": peak_time,
                    "fim": end,
                    "faixa_pico_inicio": peak_start,
                    "faixa_pico_fim": peak_end,
                    "duracao_meses": int(len(run)),
                    "oni_pico_c": peak_signed,
                    "magnitude_pico_c": peak_magnitude,
                    "limiar_evento_c": float(config.threshold_c),
                    "min_estacoes_consecutivas": int(config.min_consecutive_seasons),
                    "fracao_faixa_pico": float(config.peak_fraction),
                    "definicao_faixa_pico": (
                        "componente mensal contiguo que contem o extremo e satisfaz "
                        f"|ONI| >= {config.peak_fraction:.2f} * |ONI_extremo_evento|"
                    ),
                    "modo_rotulo": "diagnostico_retrospectivo",
                }
            )
    columns = [
        "event_id",
        "tipo",
        "classe",
        "onset",
        "pico",
        "fim",
        "faixa_pico_inicio",
        "faixa_pico_fim",
        "duracao_meses",
        "oni_pico_c",
        "magnitude_pico_c",
        "limiar_evento_c",
        "min_estacoes_consecutivas",
        "fracao_faixa_pico",
        "definicao_faixa_pico",
        "modo_rotulo",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values("onset").reset_index(drop=True)


def build_enso_lifecycle(
    events: pd.DataFrame,
    weekly_index: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    *,
    config: EnsoLifecycleConfig = EnsoLifecycleConfig(),
) -> pd.DataFrame:
    """Expand canonical events to the nine weekly ENSO states.

    Genesis is a fixed pre-onset diagnostic window and is assigned only where
    no active event phase exists.  All labels are marked retrospective because
    their boundaries depend on a completed event and its peak band.
    """

    index = _weekly_index(weekly_index)
    required = {
        "event_id",
        "tipo",
        "onset",
        "pico",
        "fim",
        "faixa_pico_inicio",
        "faixa_pico_fim",
    }
    missing = required.difference(events.columns)
    if missing:
        raise KeyError(f"events is missing columns: {sorted(missing)}")
    out = pd.DataFrame(
        {
            "fase": "neutro",
            "tipo": "neutro",
            "event_id": "",
            "estado_enso": "neutro",
            "semana_relativa_pico": np.nan,
            "semana_no_evento": np.nan,
        },
        index=index,
    )

    def assign(event: pd.Series, phase: str, start: pd.Timestamp, end: pd.Timestamp, *, neutral_only: bool) -> None:
        mask = (index >= start) & (index <= end)
        if neutral_only:
            mask &= out["fase"].eq("neutro").to_numpy()
        if not mask.any():
            return
        selected = index[mask]
        out.loc[selected, "fase"] = phase
        out.loc[selected, "tipo"] = str(event["tipo"])
        out.loc[selected, "event_id"] = str(event["event_id"])
        out.loc[selected, "estado_enso"] = f"{event['tipo']}_{phase}"
        out.loc[selected, "semana_relativa_pico"] = (
            (selected - pd.Timestamp(event["pico"])).days / 7.0
        )
        out.loc[selected, "semana_no_evento"] = (
            (selected - pd.Timestamp(event["onset"])).days / 7.0
        )

    ordered = events.sort_values("onset")
    # Active phases have priority over genesis windows.
    for _, event in ordered.iterrows():
        onset = pd.Timestamp(event["onset"])
        peak_start = pd.Timestamp(event["faixa_pico_inicio"])
        peak_end = pd.Timestamp(event["faixa_pico_fim"])
        end = pd.Timestamp(event["fim"])
        assign(event, "crescimento", onset, peak_start - pd.Timedelta(days=1), neutral_only=False)
        assign(event, "pico", peak_start, peak_end, neutral_only=False)
        assign(event, "decaimento", peak_end + pd.Timedelta(days=1), end, neutral_only=False)
    for _, event in ordered.iterrows():
        onset = pd.Timestamp(event["onset"])
        assign(
            event,
            "genese",
            onset - pd.Timedelta(weeks=config.genesis_weeks),
            onset - pd.Timedelta(days=1),
            neutral_only=True,
        )

    out["rotulo_disponivel_na_origem"] = False
    out["modo_rotulo"] = "diagnostico_retrospectivo"
    out["fracao_faixa_pico"] = float(config.peak_fraction)
    out.index.name = "week_ending_sunday"
    return out


def peak_band_sensitivity(
    oni_monthly: pd.Series,
    events: pd.DataFrame,
    *,
    fractions: Sequence[float] = (0.80, 0.90, 0.95),
    canonical_fraction: float = 0.90,
) -> pd.DataFrame:
    """Recompute the peak band for every event and requested fraction."""

    if any(not 0 < float(fraction) <= 1 for fraction in fractions):
        raise ValueError("fractions must be in (0, 1]")
    oni = _monthly_series(oni_monthly)
    required = {"event_id", "tipo", "onset", "fim", "pico"}
    missing = required.difference(events.columns)
    if missing:
        raise KeyError(f"events is missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        start = pd.Timestamp(event["onset"]).to_period("M").to_timestamp()
        end = pd.Timestamp(event["fim"]).to_period("M").to_timestamp()
        values = oni.loc[start:end]
        sign = 1.0 if event["tipo"] == "el_nino" else -1.0
        for fraction in fractions:
            peak_start, peak_end = _peak_band(values, sign, float(fraction))
            rows.append(
                {
                    "event_id": event["event_id"],
                    "tipo": event["tipo"],
                    "pico": pd.Timestamp(event["pico"]),
                    "fracao_faixa_pico": float(fraction),
                    "faixa_pico_inicio": peak_start,
                    "faixa_pico_fim": peak_end,
                    "duracao_faixa_pico_meses": int(
                        (peak_end.to_period("M") - peak_start.to_period("M")).n + 1
                    ),
                    "configuracao_canonica": bool(np.isclose(fraction, canonical_fraction)),
                    "modo_rotulo": "diagnostico_retrospectivo_sensibilidade",
                }
            )
    return pd.DataFrame(rows)


def build_rolling_origin_targets(
    weekly_signal: pd.Series,
    *,
    horizons_weeks: Sequence[int],
    threshold_c: float = 0.5,
    lifecycle: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create leakage-safe, long-format targets for true rolling origins.

    The current signal is the last value observable by the feature pipeline.
    Lifecycle labels may be attached to the *future target* but are never
    exposed as an origin feature by this function.
    """

    signal = pd.Series(weekly_signal, copy=True).astype(float)
    signal.index = _weekly_index(signal.index)
    if threshold_c <= 0:
        raise ValueError("threshold_c must be positive")
    horizons = tuple(dict.fromkeys(int(value) for value in horizons_weeks))
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError("horizons_weeks must contain positive integers")
    if lifecycle is not None:
        life = lifecycle.reindex(signal.index)
    else:
        life = None

    rows: list[pd.DataFrame] = []
    for horizon in horizons:
        origin = signal.index
        target_time = origin + pd.Timedelta(weeks=horizon)
        target = signal.reindex(target_time).to_numpy(dtype=float)
        target_type = np.full(len(origin), "neutro", dtype=object)
        target_type[target >= threshold_c] = "el_nino"
        target_type[target <= -threshold_c] = "la_nina"
        target_type[~np.isfinite(target)] = "fora_da_amostra"
        frame = pd.DataFrame(
            {
                "origin_time": origin,
                "information_cutoff": origin,
                "target_time": target_time,
                "horizon_weeks": horizon,
                "signal_at_origin_c": signal.to_numpy(dtype=float),
                "target_signal_c": target,
                "target_tipo": target_type,
                "uses_future_features": False,
                "evaluation_mode": "rolling_origin_operacional",
            }
        )
        if life is not None:
            future = life.reindex(target_time)
            frame["target_fase"] = future["fase"].fillna("fora_da_amostra").to_numpy()
            frame["target_estado_enso"] = future["estado_enso"].fillna("fora_da_amostra").to_numpy()
            frame["target_event_id"] = future["event_id"].fillna("").to_numpy()
            frame["target_label_is_retrospective"] = True
        else:
            frame["target_fase"] = np.where(
                target_type == "fora_da_amostra",
                "fora_da_amostra",
                np.where(target_type == "neutro", "neutro", "ativa_nao_segmentada"),
            )
            frame["target_estado_enso"] = target_type
            frame["target_event_id"] = ""
            frame["target_label_is_retrospective"] = False
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(["origin_time", "horizon_weeks"]).reset_index(drop=True)
