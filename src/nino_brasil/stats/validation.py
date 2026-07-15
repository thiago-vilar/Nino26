"""Event-grouped validation contracts for ENSO experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventFold:
    """One auditable split represented by positional row indices."""

    fold_id: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    test_event_id: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_end: pd.Timestamp
    train_event_ids: tuple[str, ...]
    purge_weeks: int
    evaluation_mode: str

    def as_dict(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "test_event_id": self.test_event_id,
            "train_rows": int(len(self.train_indices)),
            "test_rows": int(len(self.test_indices)),
            "n_train_events": int(len(self.train_event_ids)),
            "train_event_ids": "|".join(self.train_event_ids),
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "gap_days": int((self.test_start - self.train_end).days),
            "purge_weeks": self.purge_weeks,
            "evaluation_mode": self.evaluation_mode,
            "future_events_in_training": False,
            "test_event_in_training": False,
            "status": "ok",
        }


def required_purge_weeks(
    *,
    max_feature_lag_weeks: int = 0,
    target_horizon_weeks: int = 0,
    sequence_length_weeks: int = 1,
) -> int:
    """Return a conservative embargo preventing shared raw weeks."""

    values = (max_feature_lag_weeks, target_horizon_weeks, sequence_length_weeks)
    if any(int(value) != value or value < 0 for value in values):
        raise ValueError("lag, horizon and sequence length must be non-negative integers")
    lookback = max(int(max_feature_lag_weeks), max(int(sequence_length_weeks) - 1, 0))
    return lookback + int(target_horizon_weeks)


def _validate_samples(
    samples: pd.DataFrame,
    *,
    time_column: str,
    event_column: str,
    target_time_column: str | None,
) -> pd.DataFrame:
    required = {time_column, event_column}
    if target_time_column:
        required.add(target_time_column)
    missing = required.difference(samples.columns)
    if missing:
        raise KeyError(f"samples is missing columns: {sorted(missing)}")
    frame = samples.copy().reset_index(drop=True)
    frame[time_column] = pd.to_datetime(frame[time_column])
    if target_time_column:
        frame[target_time_column] = pd.to_datetime(frame[target_time_column])
    frame[event_column] = frame[event_column].fillna("").astype(str)
    if frame[time_column].isna().any():
        raise ValueError("sample origins cannot be missing")
    return frame


def event_purged_rolling_origin_folds(
    samples: pd.DataFrame,
    *,
    time_column: str = "origin_time",
    event_column: str = "event_id",
    target_time_column: str | None = "target_time",
    purge_weeks: int,
    min_train_events: int = 3,
) -> list[EventFold]:
    """Create expanding-window tests in which every event is indivisible.

    Test folds contain one complete non-neutral event.  A candidate training
    event is retained only when *all* its rows end before the embargo cut-off;
    rows from a boundary event are never partially admitted.  Neutral rows are
    permitted before the same cut-off because they do not represent repeated
    windows from an event, but future target timestamps must also precede the
    test event.
    """

    if purge_weeks < 0:
        raise ValueError("purge_weeks cannot be negative")
    if min_train_events < 1:
        raise ValueError("min_train_events must be positive")
    frame = _validate_samples(
        samples,
        time_column=time_column,
        event_column=event_column,
        target_time_column=target_time_column,
    )
    active = frame.loc[frame[event_column].ne("")]
    if active.empty:
        raise ValueError("no labelled events are available for grouped validation")
    event_bounds = (
        active.groupby(event_column, sort=False)[time_column]
        .agg(["min", "max"])
        .sort_values("min")
    )
    folds: list[EventFold] = []
    for event_id, bounds in event_bounds.iterrows():
        test_start = pd.Timestamp(bounds["min"])
        test_end = pd.Timestamp(bounds["max"])
        cutoff = test_start - pd.Timedelta(weeks=int(purge_weeks))
        prior_bounds = event_bounds.loc[event_bounds["max"] < cutoff]
        prior_event_ids = set(prior_bounds.index.astype(str))
        if target_time_column:
            valid_ids: set[str] = set()
            for prior_id in prior_event_ids:
                group = frame.loc[frame[event_column].eq(prior_id)]
                if group[target_time_column].max() < test_start:
                    valid_ids.add(prior_id)
            prior_event_ids = valid_ids
        if len(prior_event_ids) < min_train_events:
            continue

        neutral = frame[event_column].eq("") & frame[time_column].lt(cutoff)
        if target_time_column:
            neutral &= frame[target_time_column].lt(test_start)
        train_mask = neutral | frame[event_column].isin(prior_event_ids)
        test_mask = frame[event_column].eq(str(event_id))
        train_indices = np.flatnonzero(train_mask.to_numpy())
        test_indices = np.flatnonzero(test_mask.to_numpy())
        if train_indices.size == 0 or test_indices.size == 0:
            continue
        train_end = pd.Timestamp(frame.loc[train_indices, time_column].max())
        if not train_end < cutoff:
            raise RuntimeError("internal purge contract violation")
        folds.append(
            EventFold(
                fold_id=f"rolling_event_{len(folds) + 1:02d}",
                train_indices=train_indices,
                test_indices=test_indices,
                test_event_id=str(event_id),
                test_start=test_start,
                test_end=test_end,
                train_end=train_end,
                train_event_ids=tuple(sorted(prior_event_ids)),
                purge_weeks=int(purge_weeks),
                evaluation_mode="rolling_origin_evento_agrupado_purgado",
            )
        )
    if not folds:
        raise ValueError(
            "no fold satisfies min_train_events and the purge; expand the record "
            "or reduce only a scientifically pre-specified embargo"
        )
    return folds


def grouped_event_loo_folds(
    samples: pd.DataFrame,
    *,
    event_column: str = "event_id",
) -> Iterator[tuple[np.ndarray, np.ndarray, dict[str, object]]]:
    """Yield diagnostic leave-one-event-out folds, explicitly non-operational."""

    if event_column not in samples:
        raise KeyError(event_column)
    labels = samples[event_column].fillna("").astype(str)
    events = [value for value in pd.unique(labels) if value]
    if len(events) < 2:
        raise ValueError("at least two events are required")
    for fold_number, event_id in enumerate(events, start=1):
        test = np.flatnonzero(labels.eq(event_id).to_numpy())
        train = np.flatnonzero(labels.ne(event_id).to_numpy())
        yield train, test, {
            "fold_id": f"diagnostic_loo_{fold_number:02d}",
            "test_event_id": event_id,
            "evaluation_mode": "diagnostico_loo_evento_nao_operacional",
            "future_events_in_training": True,
            "test_event_in_training": False,
        }


def folds_audit_table(folds: list[EventFold]) -> pd.DataFrame:
    """Materialise fold boundaries for numeric-table auditing."""

    return pd.DataFrame([fold.as_dict() for fold in folds])
