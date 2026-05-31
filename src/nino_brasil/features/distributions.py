from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class TailFit:
    variable: str
    tail: str
    n: int
    xmin: float
    alpha: float
    ks_distance: float
    loglik_power_law: float
    loglik_lognormal: float
    loglik_exponential: float
    llr_power_law_vs_lognormal: float
    llr_power_law_vs_exponential: float
    preferred_distribution: str


def _tail_values(values: np.ndarray, tail: str) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if tail == "absolute":
        x = np.abs(x)
    elif tail != "upper":
        raise ValueError("tail must be 'upper' or 'absolute'.")
    return np.sort(x[x > 0])


def _power_law_alpha(x: np.ndarray, xmin: float) -> float:
    tail = x[x >= xmin]
    return 1.0 + tail.size / np.log(tail / xmin).sum()


def _power_law_ks(x: np.ndarray, xmin: float, alpha: float) -> float:
    tail = x[x >= xmin]
    empirical = np.arange(1, tail.size + 1, dtype=float) / tail.size
    theoretical = 1.0 - (tail / xmin) ** (1.0 - alpha)
    return float(np.max(np.abs(empirical - theoretical)))


def _loglik_power_law(x: np.ndarray, xmin: float, alpha: float) -> float:
    tail = x[x >= xmin]
    return float(tail.size * np.log((alpha - 1.0) / xmin) - alpha * np.log(tail / xmin).sum())


def _loglik_exponential(x: np.ndarray, xmin: float) -> float:
    tail = x[x >= xmin] - xmin
    rate = 1.0 / np.mean(tail) if np.mean(tail) > 0 else np.inf
    if not np.isfinite(rate):
        return float("-inf")
    return float(tail.size * np.log(rate) - rate * tail.sum())


def _loglik_lognormal(x: np.ndarray, xmin: float) -> float:
    tail = x[x >= xmin]
    log_tail = np.log(tail)
    sigma = float(np.std(log_tail, ddof=0))
    if sigma == 0:
        return float("-inf")
    mu = float(np.mean(log_tail))
    return float(
        np.sum(
            -np.log(tail)
            - np.log(sigma)
            - 0.5 * np.log(2.0 * np.pi)
            - ((log_tail - mu) ** 2) / (2.0 * sigma**2)
        )
    )


def fit_power_law_tail(
    values: np.ndarray,
    *,
    variable: str = "value",
    tail: str = "upper",
    min_tail: int = 50,
    xmin_candidates: Iterable[float] | None = None,
) -> TailFit:
    """Fit a positive tail with power-law MLE and compare with lognormal/exponential."""
    x = _tail_values(values, tail)
    if x.size < min_tail:
        raise ValueError(f"Need at least {min_tail} positive tail values; received {x.size}.")

    candidates = np.asarray(list(xmin_candidates) if xmin_candidates is not None else [])
    if candidates.size == 0:
        quantiles = np.linspace(0.50, 0.95, 24)
        candidates = np.unique(np.quantile(x, quantiles))
    candidates = candidates[np.isfinite(candidates)]

    best: tuple[float, float, float] | None = None
    for xmin in candidates:
        tail_count = int(np.sum(x >= xmin))
        if tail_count < min_tail:
            continue
        alpha = _power_law_alpha(x, float(xmin))
        if not np.isfinite(alpha) or alpha <= 1.0:
            continue
        ks = _power_law_ks(x, float(xmin), alpha)
        if best is None or ks < best[2]:
            best = (float(xmin), float(alpha), float(ks))
    if best is None:
        raise ValueError("No valid xmin candidate produced enough tail samples.")

    xmin, alpha, ks = best
    tail_count = int(np.sum(x >= xmin))
    ll_power = _loglik_power_law(x, xmin, alpha)
    ll_lognormal = _loglik_lognormal(x, xmin)
    ll_exponential = _loglik_exponential(x, xmin)
    likelihoods = {
        "power_law": ll_power,
        "lognormal": ll_lognormal,
        "exponential": ll_exponential,
    }
    preferred = max(likelihoods, key=likelihoods.get)
    return TailFit(
        variable=variable,
        tail=tail,
        n=tail_count,
        xmin=xmin,
        alpha=alpha,
        ks_distance=ks,
        loglik_power_law=ll_power,
        loglik_lognormal=ll_lognormal,
        loglik_exponential=ll_exponential,
        llr_power_law_vs_lognormal=ll_power - ll_lognormal,
        llr_power_law_vs_exponential=ll_power - ll_exponential,
        preferred_distribution=preferred,
    )


def sample_dataarray(
    da: xr.DataArray,
    *,
    sample_size: int = 200_000,
    random_state: int = 42,
) -> np.ndarray:
    """Return a finite random sample from a DataArray."""
    values = np.asarray(da.values).ravel()
    values = values[np.isfinite(values)]
    if values.size <= sample_size:
        return values
    rng = np.random.default_rng(random_state)
    idx = rng.choice(values.size, size=sample_size, replace=False)
    return values[idx]


def diagnose_dataset_distributions(
    ds: xr.Dataset,
    *,
    variables: Iterable[str] | None = None,
    tail: str = "upper",
    sample_size: int = 200_000,
    min_tail: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit tail diagnostics for selected variables in a Dataset."""
    selected = list(variables) if variables is not None else list(ds.data_vars)
    rows = []
    for variable in selected:
        values = sample_dataarray(ds[variable], sample_size=sample_size, random_state=random_state)
        fit = fit_power_law_tail(values, variable=variable, tail=tail, min_tail=min_tail)
        rows.append(fit.__dict__)
    return pd.DataFrame(rows)
