"""Breakpoint-free sensitivity diagnostics for lagged correlations.

The helpers in this module deliberately quantify sampling and event influence
without declaring a structural climate break.  The caller chooses an already
specified lag, then evaluates the paired series with a moving-block bootstrap
and/or by leaving one named event out at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapCorrelationResult:
    """Result of a paired moving-block bootstrap for a correlation."""

    observed_r: float
    bootstrap_r: np.ndarray
    ci_low: float
    ci_high: float
    median_r: float
    sign_agreement: float
    n_samples: int
    block_length: int
    n_boot: int

    def as_dict(self) -> dict[str, float | int]:
        """Return scalar diagnostics suitable for a CSV row."""

        return {
            "r_full": self.observed_r,
            "bootstrap_r_mediana": self.median_r,
            "bootstrap_ic95_inf": self.ci_low,
            "bootstrap_ic95_sup": self.ci_high,
            "bootstrap_fracao_mesmo_sinal": self.sign_agreement,
            "n_semanas": self.n_samples,
            "bootstrap_bloco_semanas": self.block_length,
            "bootstrap_n": self.n_boot,
        }


def _paired_segments(
    x: Iterable[float] | pd.Series,
    y: Iterable[float] | pd.Series,
    *,
    expected_step: str | pd.Timedelta | None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Return all finite pairs plus finite, contiguous time segments."""

    if isinstance(x, pd.Series) or isinstance(y, pd.Series):
        xs = x if isinstance(x, pd.Series) else pd.Series(np.asarray(list(x), dtype=float))
        ys = y if isinstance(y, pd.Series) else pd.Series(np.asarray(list(y), dtype=float))
        pair = pd.concat([xs.rename("x"), ys.rename("y")], axis=1, join="inner")
        if not pair.index.is_monotonic_increasing or not pair.index.is_unique:
            raise ValueError("time-indexed inputs must have a unique, increasing index")
        if expected_step is not None and not isinstance(pair.index, pd.DatetimeIndex):
            raise ValueError("expected_step requires a DatetimeIndex for Series inputs")
        raw_x = pair["x"].to_numpy(dtype=float)
        raw_y = pair["y"].to_numpy(dtype=float)
        adjacent = np.ones(len(pair), dtype=bool)
        if len(pair) > 1 and expected_step is not None:
            step = pd.Timedelta(expected_step)
            adjacent[1:] = np.asarray(pair.index[1:] - pair.index[:-1] == step, dtype=bool)
    else:
        raw_x = np.asarray(list(x), dtype=float)
        raw_y = np.asarray(list(y), dtype=float)
        if raw_x.shape != raw_y.shape:
            raise ValueError("x and y must have the same shape")
        adjacent = np.ones(raw_x.size, dtype=bool)

    finite = np.isfinite(raw_x) & np.isfinite(raw_y)
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start: int | None = None
    for i, is_finite in enumerate(finite):
        continues = bool(is_finite and (i == 0 or (finite[i - 1] and adjacent[i])))
        if is_finite and (start is None or not continues):
            if start is not None:
                segments.append((raw_x[start:i], raw_y[start:i]))
            start = i
        elif not is_finite and start is not None:
            segments.append((raw_x[start:i], raw_y[start:i]))
            start = None
    if start is not None:
        segments.append((raw_x[start:], raw_y[start:]))

    xv = raw_x[finite]
    yv = raw_y[finite]
    return xv, yv, segments


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def moving_block_bootstrap_correlation(
    x: Iterable[float] | pd.Series,
    y: Iterable[float] | pd.Series,
    *,
    block_length: int,
    n_boot: int = 2_000,
    random_state: int | np.random.Generator | None = None,
    confidence: float = 0.95,
    expected_step: str | pd.Timedelta | None = "7D",
) -> BootstrapCorrelationResult:
    """Bootstrap a paired correlation by resampling contiguous time blocks.

    Blocks are sampled with replacement from every admissible starting
    position and concatenated until the original sample length is restored.
    For time-indexed Series, a block is admissible only when every pair is
    finite and timestamps are separated by ``expected_step`` throughout; a
    missing week or NaN therefore splits the series and is never crossed by a
    block. This preserves within-block serial dependence and never presupposes
    a breakpoint. ``x`` and ``y`` must already encode the desired lag.
    """

    xv, yv, segments = _paired_segments(x, y, expected_step=expected_step)
    n = int(xv.size)
    if n < 3:
        raise ValueError("at least three finite pairs are required")
    if not 2 <= int(block_length) <= n:
        raise ValueError("block_length must be between 2 and the paired sample size")
    if int(n_boot) < 1:
        raise ValueError("n_boot must be positive")
    if not 0 < float(confidence) < 1:
        raise ValueError("confidence must be between 0 and 1")

    block_length = int(block_length)
    n_boot = int(n_boot)
    observed = _pearson(xv, yv)
    if not np.isfinite(observed):
        raise ValueError("correlation is undefined for a constant input")

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    candidates = [
        (segment_id, start)
        for segment_id, (segment_x, _) in enumerate(segments)
        for start in range(len(segment_x) - block_length + 1)
    ]
    if not candidates:
        raise ValueError(
            "no finite contiguous segment is long enough for block_length; "
            "blocks cannot cross gaps or non-finite pairs"
        )
    n_blocks = ceil(n / block_length)
    boot = np.full(n_boot, np.nan, dtype=float)
    for i in range(n_boot):
        chosen = rng.integers(0, len(candidates), size=n_blocks)
        sampled_x = []
        sampled_y = []
        for candidate_index in chosen:
            segment_id, start = candidates[int(candidate_index)]
            segment_x, segment_y = segments[segment_id]
            sampled_x.append(segment_x[start : start + block_length])
            sampled_y.append(segment_y[start : start + block_length])
        bx = np.concatenate(sampled_x)[:n]
        by = np.concatenate(sampled_y)[:n]
        boot[i] = _pearson(bx, by)

    valid = boot[np.isfinite(boot)]
    if valid.size == 0:
        raise ValueError("all bootstrap correlations were undefined")
    alpha = (1.0 - float(confidence)) / 2.0
    ci_low, ci_high = np.quantile(valid, [alpha, 1.0 - alpha])
    sign_agreement = float(np.mean(np.sign(valid) == np.sign(observed)))
    return BootstrapCorrelationResult(
        observed_r=observed,
        bootstrap_r=boot,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        median_r=float(np.median(valid)),
        sign_agreement=sign_agreement,
        n_samples=n,
        block_length=block_length,
        n_boot=n_boot,
    )


def leave_one_event_out_correlation(
    x: pd.Series,
    y: pd.Series,
    event_labels: pd.Series,
    *,
    min_samples: int = 30,
) -> pd.DataFrame:
    """Re-estimate a correlation after excluding each labelled event.

    Unlabelled dates remain in every estimate.  Event labels therefore define
    the intervals whose influence is audited, not strata or climate regimes.
    The returned table has one row per non-null event label.
    """

    if min_samples < 3:
        raise ValueError("min_samples must be at least 3")
    frame = pd.concat(
        [x.rename("x"), y.rename("y"), event_labels.rename("event")],
        axis=1,
        join="inner",
    )
    finite = np.isfinite(frame["x"].to_numpy(dtype=float)) & np.isfinite(frame["y"].to_numpy(dtype=float))
    frame = frame.loc[finite]
    events = pd.unique(frame["event"].dropna())
    rows: list[dict[str, object]] = []
    for event in events:
        kept = frame.loc[frame["event"].isna() | (frame["event"] != event)]
        r = _pearson(kept["x"].to_numpy(dtype=float), kept["y"].to_numpy(dtype=float))
        rows.append(
            {
                "evento_removido": event,
                "n_removido": int((frame["event"] == event).sum()),
                "n_restante": int(len(kept)),
                "r_sem_evento": r if len(kept) >= min_samples else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=["evento_removido", "n_removido", "n_restante", "r_sem_evento"])


def summarize_correlation_stability(
    bootstrap: BootstrapCorrelationResult,
    leave_one_event_out: pd.DataFrame,
) -> dict[str, float | int]:
    """Combine breakpoint-free bootstrap and event-influence diagnostics."""

    values = leave_one_event_out.get("r_sem_evento", pd.Series(dtype=float)).dropna().to_numpy(dtype=float)
    if values.size:
        loo_min = float(np.min(values))
        loo_max = float(np.max(values))
        loo_amplitude = float(loo_max - loo_min)
        loo_sign_agreement = float(np.mean(np.sign(values) == np.sign(bootstrap.observed_r)))
    else:
        loo_min = loo_max = loo_amplitude = loo_sign_agreement = float("nan")
    return {
        **bootstrap.as_dict(),
        "loo_eventos_n": int(values.size),
        "loo_eventos_r_min": loo_min,
        "loo_eventos_r_max": loo_max,
        "loo_eventos_amplitude_r": loo_amplitude,
        "loo_eventos_fracao_mesmo_sinal": loo_sign_agreement,
    }
