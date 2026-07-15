"""Event-aware validation and conservative augmentation for phases 5--8.

Weekly samples from one ENSO event are strongly dependent.  This module keeps
all samples from an event (and each neutral time block) in the same fold,
applies a purge/embargo around the test interval, and records every augmented
sample's parent.  Augmentation therefore improves optimisation only; it never
increases the reported number of independent events.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ENSO_TYPES: tuple[str, ...] = ("el_nino", "la_nina")
ENSO_PHASES: tuple[str, ...] = ("genese", "crescimento", "pico", "decaimento")
ENSO_STATES: tuple[str, ...] = (
    "neutro",
    *(f"{event_type}_{phase}" for event_type in ENSO_TYPES for phase in ENSO_PHASES),
)


def assert_continuous_weekly_index(
    index: pd.DatetimeIndex | Sequence[object],
    *,
    name: str = "weekly_index",
    require_sunday: bool = True,
) -> pd.DatetimeIndex:
    """Validate the positional-lag contract used throughout F3--F8.

    ``shift(1)`` and sequence windows mean one week only when timestamps are
    unique, ordered and exactly seven days apart.  Intersections of sources can
    silently hide a missing partition, so every modeling entry point calls this
    guard after alignment.
    """

    values = pd.DatetimeIndex(index)
    if values.empty:
        raise ValueError(f"{name} esta vazio.")
    if not values.is_monotonic_increasing or not values.is_unique:
        raise ValueError(f"{name} deve ser cronologico e unico.")
    if require_sunday and not bool((values.dayofweek == 6).all()):
        raise ValueError(f"{name} deve usar domingos W-SUN.")
    if len(values) > 1:
        gaps = np.diff(values.to_numpy(dtype="datetime64[ns]"))
        if not bool((gaps == np.timedelta64(7, "D")).all()):
            bad = np.flatnonzero(gaps != np.timedelta64(7, "D"))[:5]
            examples = [
                f"{values[position].date()}->{values[position + 1].date()}"
                for position in bad
            ]
            raise ValueError(
                f"{name} contem semanas ausentes/irregulares: {examples}"
            )
    return values


def parse_condition(condition: str) -> tuple[str | None, str | None]:
    """Parse ``el_nino_pico``/``la_nina_genese`` without splitting the type.

    ``todas`` maps to ``(None, None)`` and ``neutro`` to
    ``("neutro", "neutro")``.  Invalid conditions fail loudly instead of
    silently selecting zero rows.
    """

    value = str(condition).strip().lower()
    if value in {"todas", "all", "todos"}:
        return None, None
    if value in {"neutro", "neutral"}:
        return "neutro", "neutro"
    for event_type in ENSO_TYPES:
        prefix = f"{event_type}_"
        if value.startswith(prefix):
            phase = value[len(prefix) :]
            if phase not in ENSO_PHASES:
                raise ValueError(f"Fase ENSO invalida em {condition!r}: {phase!r}.")
            return event_type, phase
    raise ValueError(
        f"Condicao ENSO invalida {condition!r}; use 'todas', 'neutro' ou "
        "'<el_nino|la_nina>_<genese|crescimento|pico|decaimento>'."
    )


def condition_mask(phase_table: pd.DataFrame, condition: str) -> pd.Series:
    """Return a boolean mask for a canonical ENSO condition."""

    event_type, phase = parse_condition(condition)
    if event_type is None:
        return pd.Series(True, index=phase_table.index, dtype=bool)
    if not {"tipo", "fase"}.issubset(phase_table.columns):
        raise KeyError("phase_table precisa das colunas 'tipo' e 'fase'.")
    return (
        phase_table["tipo"].astype(str).str.lower().eq(event_type)
        & phase_table["fase"].astype(str).str.lower().eq(phase)
    )


def canonical_state_labels(phase_table: pd.DataFrame) -> pd.Series:
    """Build the nine-state target: neutral + EN/LN x four life-cycle phases."""

    if not {"tipo", "fase"}.issubset(phase_table.columns):
        raise KeyError("phase_table precisa das colunas 'tipo' e 'fase'.")
    event_type = phase_table["tipo"].fillna("neutro").astype(str).str.lower()
    phase = phase_table["fase"].fillna("neutro").astype(str).str.lower()
    labels = pd.Series("neutro", index=phase_table.index, name="estado_enso", dtype="object")
    active = event_type.isin(ENSO_TYPES) & phase.isin(ENSO_PHASES)
    labels.loc[active] = event_type.loc[active] + "_" + phase.loc[active]
    unknown = ~(active | (event_type.eq("neutro") & phase.eq("neutro")))
    if unknown.any():
        examples = sorted(set(zip(event_type[unknown], phase[unknown])))[:5]
        raise ValueError(f"Pares tipo/fase fora do contrato canonico: {examples}")
    return labels


def parse_state(state: str) -> tuple[str, str]:
    """Parse the canonical single-underscore state without breaking event type."""

    value = str(state).lower()
    if value == "neutro":
        return "neutro", "neutro"
    for event_type in ENSO_TYPES:
        prefix = f"{event_type}_"
        if value.startswith(prefix) and value[len(prefix) :] in ENSO_PHASES:
            return event_type, value[len(prefix) :]
    raise ValueError(f"Estado ENSO invalido: {state!r}")


def canonical_event_ids(phase_table: pd.DataFrame, *, neutral_block: str = "QS") -> pd.Series:
    """Return independent grouping ids, assigning neutral weeks to time blocks.

    True event ids are preserved.  Neutral samples receive calendar-quarter ids
    so a temporally adjacent neutral window cannot be split across train/test.
    """

    if not isinstance(phase_table.index, pd.DatetimeIndex):
        raise TypeError("phase_table deve usar DatetimeIndex.")
    if "event_id" not in phase_table:
        raise KeyError("phase_table precisa da coluna 'event_id'.")
    labels = canonical_state_labels(phase_table)
    ids = phase_table["event_id"].fillna("").astype(str).copy()
    active = labels.ne("neutro")
    if (active & ids.eq("")).any():
        raise ValueError("Semanas ENSO ativas sem event_id impedem validacao por evento.")
    # Quarter-start grouping is deliberately explicit and deterministic.
    if neutral_block != "QS":
        raise ValueError("neutral_block suportado: 'QS'.")
    quarter = phase_table.index.to_period("Q").astype(str)
    ids.loc[~active] = "neutral_" + pd.Series(quarter, index=ids.index).loc[~active]
    ids.name = "validation_group"
    return ids


def balanced_active_event_test_start(
    phase_table: pd.DataFrame,
    *,
    min_train_active_events_per_type: int,
    purge_weeks: int,
) -> pd.Timestamp | None:
    """Return the first admissible test week after balanced ENSO support.

    Neutral calendar-quarter groups are useful for preventing leakage, but
    they must not satisfy an El Nino/La Nina sample-size prerequisite.  For an
    official expanding-window experiment, the first test group therefore
    starts only after at least ``min_train_active_events_per_type`` complete
    events of *each* sign can remain in training after the temporal embargo.

    ``None`` disables this additional boundary (used by smoke diagnostics).
    The extra week makes the strict ``time < test_start - embargo`` predicate
    in :func:`make_event_folds` retain the final week of the support event.
    """

    minimum = int(min_train_active_events_per_type)
    purge = int(purge_weeks)
    if minimum < 0:
        raise ValueError("min_train_active_events_per_type deve ser >= 0.")
    if purge < 0:
        raise ValueError("purge_weeks deve ser >= 0.")
    if minimum == 0:
        return None
    if not isinstance(phase_table.index, pd.DatetimeIndex):
        raise TypeError("phase_table deve usar DatetimeIndex.")
    required = {"tipo", "fase", "event_id"}
    if missing := required.difference(phase_table.columns):
        raise KeyError(f"phase_table sem colunas de suporte: {sorted(missing)}")

    labels = canonical_state_labels(phase_table)
    active = labels.ne("neutro")
    event_ids = phase_table["event_id"].fillna("").astype(str).str.strip()
    event_types = phase_table["tipo"].fillna("neutro").astype(str).str.lower()
    cutoffs: list[pd.Timestamp] = []
    for event_type in ENSO_TYPES:
        selected = active & event_types.eq(event_type) & event_ids.ne("")
        event_ends = (
            pd.DataFrame(
                {"event_id": event_ids.loc[selected]},
                index=phase_table.index[selected],
            )
            .reset_index(names="time")
            .groupby("event_id", sort=False)["time"]
            .max()
            .sort_values()
        )
        if len(event_ends) < minimum:
            raise ValueError(
                f"Suporte insuficiente para desenhar folds: {event_type} tem "
                f"{len(event_ends)} eventos, requer {minimum}."
            )
        cutoffs.append(pd.Timestamp(event_ends.iloc[minimum - 1]))
    support_end = max(cutoffs)
    return support_end + pd.Timedelta(weeks=purge + 1)


@dataclass(frozen=True)
class EventFold:
    """One expanding, event-grouped and purged validation fold."""

    name: str
    train_index: np.ndarray
    test_index: np.ndarray
    train_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge_weeks: int


def make_event_folds(
    index: pd.DatetimeIndex,
    groups: Sequence[str] | pd.Series,
    *,
    n_splits: int = 5,
    min_train_groups: int = 8,
    purge_weeks: int = 52,
    earliest_test_start: pd.Timestamp | str | None = None,
) -> list[EventFold]:
    """Create expanding folds whose groups never cross a split boundary.

    Test groups are consecutive chronological blocks.  Training uses only
    earlier groups and excludes samples within ``purge_weeks`` before the first
    test origin.  This covers the maximum predictor history/horizon and prevents
    overlapping windows from sharing raw weeks.
    """

    time = pd.DatetimeIndex(index)
    if not time.is_monotonic_increasing or not time.is_unique:
        raise ValueError("index deve ser cronologico e unico.")
    group_values = pd.Series(np.asarray(groups, dtype=object), index=time)
    if group_values.isna().any() or group_values.astype(str).eq("").any():
        raise ValueError("groups nao pode conter valores vazios.")
    bounds = (
        pd.DataFrame({"time": time, "group": group_values.to_numpy()})
        .groupby("group", sort=False)["time"]
        .agg(["min", "max"])
        .sort_values(["min", "max"])
    )
    ordered = bounds.index.astype(str).tolist()
    if len(ordered) <= min_train_groups:
        raise ValueError("Poucos grupos independentes para formar folds event-aware.")
    candidates = ordered[min_train_groups:]
    if earliest_test_start is not None:
        earliest = pd.Timestamp(earliest_test_start)
        if earliest.tzinfo is not None:
            earliest = earliest.tz_localize(None)
        candidates = [
            group
            for group in candidates
            if pd.Timestamp(bounds.loc[group, "min"]) >= earliest
        ]
    if not candidates:
        raise ValueError(
            "Nenhum grupo de teste resta depois do piso de suporte/embargo."
        )
    blocks = [list(b) for b in np.array_split(candidates, min(n_splits, len(candidates))) if len(b)]
    folds: list[EventFold] = []
    group_as_str = group_values.astype(str)
    embargo = pd.Timedelta(weeks=int(purge_weeks))
    for fold_no, test_groups in enumerate(blocks, start=1):
        test_start = pd.Timestamp(bounds.loc[test_groups, "min"].min())
        test_end = pd.Timestamp(bounds.loc[test_groups, "max"].max())
        prior = [g for g in ordered if pd.Timestamp(bounds.loc[g, "max"]) < test_start]
        train_mask = group_as_str.isin(prior) & (time < test_start - embargo)
        test_mask = group_as_str.isin(test_groups)
        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if not len(train_idx) or not len(test_idx):
            continue
        train_groups = tuple(pd.unique(group_as_str.iloc[train_idx]))
        if set(train_groups).intersection(test_groups):
            raise AssertionError("Um evento atravessou treino e teste.")
        folds.append(
            EventFold(
                name=f"event_fold_{fold_no:02d}",
                train_index=train_idx,
                test_index=test_idx,
                train_groups=train_groups,
                test_groups=tuple(test_groups),
                train_end=pd.Timestamp(time[train_idx].max()),
                test_start=test_start,
                test_end=test_end,
                purge_weeks=int(purge_weeks),
            )
        )
    if not folds:
        raise ValueError("Nenhum fold restou apos o embargo; reduza splits ou purge_weeks.")
    return folds


def event_phase_sample_weights(phase_table: pd.DataFrame) -> pd.Series:
    """Data-driven balancing without pretending that weeks are new events.

    Active weights follow ``1 / (N_types * N_events(type) * N_weeks(event,phase))``.
    Neutral quarters are handled as a third type.  The result is normalised to
    mean one and can be passed directly as ``sample_weight``.
    """

    labels = canonical_state_labels(phase_table)
    event_ids = canonical_event_ids(phase_table)
    event_type = phase_table["tipo"].fillna("neutro").astype(str).str.lower()
    phase = phase_table["fase"].fillna("neutro").astype(str).str.lower()
    work = pd.DataFrame(
        {"type": event_type, "phase": phase, "event": event_ids, "state": labels},
        index=phase_table.index,
    )
    n_types = int(work["type"].nunique())
    events_per_type = work.groupby("type")["event"].nunique()
    weeks_per_event_phase = work.groupby(["event", "phase"]).size()
    raw = []
    for row in work.itertuples():
        denom = n_types * int(events_per_type.loc[row.type]) * int(
            weeks_per_event_phase.loc[(row.event, row.phase)]
        )
        raw.append(1.0 / denom)
    weights = pd.Series(raw, index=work.index, name="sample_weight", dtype=float)
    return weights / weights.mean()


def independent_fold_support(
    train_phase_table: pd.DataFrame,
    test_phase_table: pd.DataFrame,
    *,
    fold: str,
    min_train_active_events_per_type: int,
) -> dict[str, object]:
    """Summarise genuinely independent ENSO support for one model fold.

    Neutral calendar-quarter groups are useful for keeping adjacent neutral
    windows together, but they are not El Nino or La Nina events.  This audit
    therefore reports them separately and applies the minimum only to distinct
    active event ids of each ENSO type.  The helper operates on the rows that
    actually survived fold-specific input availability checks.
    """

    minimum = int(min_train_active_events_per_type)
    if minimum < 0:
        raise ValueError("min_train_active_events_per_type deve ser >= 0.")

    def summarise(frame: pd.DataFrame, prefix: str) -> dict[str, object]:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError("Tabelas de suporte por fold devem usar DatetimeIndex.")
        labels = canonical_state_labels(frame)
        validation_groups = canonical_event_ids(frame)
        event_ids = frame["event_id"].fillna("").astype(str).str.strip()
        event_types = frame["tipo"].fillna("neutro").astype(str).str.lower()
        active = labels.ne("neutro")
        output: dict[str, object] = {
            f"n_{prefix}_rows": int(len(frame)),
            f"n_{prefix}_active_rows": int(active.sum()),
            f"n_{prefix}_neutral_rows": int((~active).sum()),
            f"n_{prefix}_neutral_blocks": int(validation_groups.loc[~active].nunique()),
        }
        active_ids: set[str] = set()
        for event_type in ENSO_TYPES:
            selected = active & event_types.eq(event_type)
            ids = set(event_ids.loc[selected])
            ids.discard("")
            output[f"n_{prefix}_{event_type}_events"] = int(len(ids))
            active_ids.update(ids)
        output[f"n_{prefix}_active_events"] = int(len(active_ids))
        return output

    row: dict[str, object] = {"fold": str(fold)}
    row.update(summarise(train_phase_table, "train"))
    row.update(summarise(test_phase_table, "test"))
    row["min_train_active_events_per_type_required"] = minimum
    row["independent_support_gate_pass"] = bool(
        int(row["n_train_el_nino_events"]) >= minimum
        and int(row["n_train_la_nina_events"]) >= minimum
    )
    row["independent_unit"] = "distinct complete ENSO event; neutral quarters reported separately"
    return row


@dataclass
class AugmentedTrainingData:
    X: pd.DataFrame
    y: pd.Series
    sample_weight: pd.Series
    provenance: pd.DataFrame


def augment_training_rows(
    X: pd.DataFrame,
    y: pd.Series,
    phase_table: pd.DataFrame,
    *,
    n_noise_copies: int = 0,
    noise_scale: float = 0.02,
    mixup_alpha: float | None = None,
    random_state: int = 42,
) -> AugmentedTrainingData:
    """Conservative, train-only augmentation with complete lineage.

    Noise uses the regularised covariance estimated from *this training set*.
    Mixup, when requested, pairs only rows of the same ENSO type and phase.
    Synthetic rows retain ``original_event_id`` and receive ``augmentation_id``.
    The function must be called after selecting a fold's training indices.
    """

    if not X.index.equals(y.index) or not X.index.equals(phase_table.index):
        raise ValueError("X, y e phase_table precisam do mesmo indice na mesma ordem.")
    if n_noise_copies < 0 or noise_scale < 0:
        raise ValueError("Parametros de augmentation devem ser nao negativos.")
    rng = np.random.default_rng(random_state)
    numeric = X.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Augmentation exige X numerico e sem NaN apos preprocessing do fold.")
    event_ids = canonical_event_ids(phase_table)
    weights = event_phase_sample_weights(phase_table)
    states = canonical_state_labels(phase_table)
    parsed_states = [parse_state(state) for state in states]
    base_prov = pd.DataFrame(
        {
            "sample_time": X.index.astype(str),
            "original_event_id": event_ids.to_numpy(),
            "state": states.to_numpy(),
            "event_type": [value[0] for value in parsed_states],
            "phase": [value[1] for value in parsed_states],
            "augmentation_id": "original",
            "augmentation_method": "none",
            "independent_event": True,
        }
    )
    x_parts = [numeric.reset_index(drop=True)]
    y_parts = [y.reset_index(drop=True)]
    w_parts = [weights.reset_index(drop=True)]
    p_parts = [base_prov]

    values = numeric.to_numpy(dtype=float)
    if len(numeric) > 1 and n_noise_copies:
        covariance = np.cov(values, rowvar=False)
        covariance = np.atleast_2d(covariance)
        diagonal = np.diag(np.diag(covariance))
        covariance = 0.15 * covariance + 0.85 * diagonal
        covariance += np.eye(covariance.shape[0]) * 1e-10
        for copy in range(1, int(n_noise_copies) + 1):
            perturb = rng.multivariate_normal(
                np.zeros(values.shape[1]), covariance, size=len(values), check_valid="ignore"
            )
            x_parts.append(pd.DataFrame(values + noise_scale * perturb, columns=numeric.columns))
            y_parts.append(y.reset_index(drop=True))
            w_parts.append(weights.reset_index(drop=True))
            prov = base_prov.copy()
            prov["augmentation_id"] = f"covnoise_{copy:02d}"
            prov["augmentation_method"] = "covariance_noise_train_only"
            prov["independent_event"] = False
            p_parts.append(prov)

    if mixup_alpha is not None:
        if mixup_alpha <= 0:
            raise ValueError("mixup_alpha deve ser positivo.")
        mixed_x: list[np.ndarray] = []
        mixed_y: list[object] = []
        mixed_w: list[float] = []
        mixed_p: list[dict[str, object]] = []
        for state in pd.unique(states):
            positions = np.flatnonzero(states.to_numpy() == state)
            if len(positions) < 2:
                continue
            partners = rng.permutation(positions)
            lam = rng.beta(mixup_alpha, mixup_alpha, size=len(positions))
            for pos, other, coefficient in zip(positions, partners, lam):
                # Classification labels are unchanged because pairing is same-state.
                mixed_x.append(coefficient * values[pos] + (1.0 - coefficient) * values[other])
                mixed_y.append(y.iloc[pos])
                mixed_w.append(float((weights.iloc[pos] + weights.iloc[other]) / 2.0))
                mixed_p.append(
                    {
                        "sample_time": str(X.index[pos]),
                        "original_event_id": str(event_ids.iloc[pos]),
                        "state": str(states.iloc[pos]),
                        "event_type": parse_state(str(states.iloc[pos]))[0],
                        "phase": parse_state(str(states.iloc[pos]))[1],
                        "augmentation_id": f"mixup_{pos:06d}",
                        "augmentation_method": "same_type_phase_mixup_train_only",
                        "mixup_parent_sample_time": str(X.index[other]),
                        "mixup_parent_event_id": str(event_ids.iloc[other]),
                        "mixup_parent_state": str(states.iloc[other]),
                        "mixup_lambda": float(coefficient),
                        "independent_event": False,
                    }
                )
        if mixed_x:
            x_parts.append(pd.DataFrame(mixed_x, columns=numeric.columns))
            y_parts.append(pd.Series(mixed_y, name=y.name))
            w_parts.append(pd.Series(mixed_w, name="sample_weight"))
            p_parts.append(pd.DataFrame(mixed_p))

    X_out = pd.concat(x_parts, ignore_index=True)
    y_out = pd.concat(y_parts, ignore_index=True).rename(y.name)
    w_out = pd.concat(w_parts, ignore_index=True).rename("sample_weight")
    provenance = pd.concat(p_parts, ignore_index=True).fillna("")
    return AugmentedTrainingData(X=X_out, y=y_out, sample_weight=w_out, provenance=provenance)


def assert_fold_independence(folds: Iterable[EventFold]) -> None:
    """Raise when any validation group appears on both sides of a fold."""

    for fold in folds:
        overlap = set(fold.train_groups).intersection(fold.test_groups)
        if overlap:
            raise AssertionError(f"{fold.name}: grupos vazaram entre treino/teste: {sorted(overlap)}")
