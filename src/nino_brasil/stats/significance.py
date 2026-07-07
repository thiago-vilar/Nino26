from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _finite_pair(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    x_arr = np.asarray(x, dtype="float64")
    if y is None:
        return x_arr[np.isfinite(x_arr)], None
    y_arr = np.asarray(y, dtype="float64")
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]


def lag1_autocorrelation(x: pd.Series | np.ndarray) -> float:
    """Return lag-1 autocorrelation after dropping non-finite values."""
    arr, _ = _finite_pair(x)
    if arr.size < 3:
        return float("nan")
    if np.nanstd(arr[:-1]) == 0 or np.nanstd(arr[1:]) == 0:
        return 0.0
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


def effective_sample_size(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray | None = None) -> float:
    """Estimate effective N under lag-1 autocorrelation.

    For one series, use N * (1-rho)/(1+rho). For a correlation pair, use the
    Bretherton-style N * (1-rho_x*rho_y)/(1+rho_x*rho_y). The result is bounded
    to [3, N] so downstream t tests never claim more information than exists.
    """
    x_arr, y_arr = _finite_pair(x, y)
    n = x_arr.size if y_arr is None else min(x_arr.size, y_arr.size)
    if n < 3:
        return float(n)
    rho_x = lag1_autocorrelation(x_arr)
    rho_y = rho_x if y_arr is None else lag1_autocorrelation(y_arr)
    if not np.isfinite(rho_x) or not np.isfinite(rho_y):
        return float(n)
    rho_product = float(np.clip(rho_x * rho_y, -0.99, 0.99))
    n_eff = n * (1.0 - rho_product) / (1.0 + rho_product)
    return float(np.clip(n_eff, 3.0, float(n)))


def correlation_p_value(r: float, n_eff: float) -> float:
    """Two-sided p-value for a correlation using effective degrees of freedom."""
    if not np.isfinite(r) or not np.isfinite(n_eff) or n_eff <= 2:
        return float("nan")
    r = float(np.clip(r, -0.999999, 0.999999))
    t_value = r * np.sqrt((n_eff - 2.0) / (1.0 - r * r))
    return float(2.0 * stats.t.sf(abs(t_value), df=n_eff - 2.0))


def _residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(controls.shape[0]), controls])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def partial_correlation(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    controls: pd.DataFrame | np.ndarray,
) -> dict[str, float]:
    """Partial correlation between x and y after linear control residualization."""
    x_arr = np.asarray(x, dtype="float64")
    y_arr = np.asarray(y, dtype="float64")
    c_arr = np.asarray(controls, dtype="float64")
    if c_arr.ndim == 1:
        c_arr = c_arr[:, None]
    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & np.all(np.isfinite(c_arr), axis=1)
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]
    c_clean = c_arr[mask]
    n = int(x_clean.size)
    if n <= c_clean.shape[1] + 3:
        return {"r": float("nan"), "n": float(n), "n_eff": float("nan"), "p_effective": float("nan")}
    x_res = _residualize(x_clean, c_clean)
    y_res = _residualize(y_clean, c_clean)
    if np.nanstd(x_res) == 0 or np.nanstd(y_res) == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(x_res, y_res)[0, 1])
    n_eff = effective_sample_size(x_res, y_res)
    p_eff = correlation_p_value(r, n_eff)
    return {"r": r, "n": float(n), "n_eff": n_eff, "p_effective": p_eff}


def benjamini_hochberg_fdr(
    p_values: pd.Series | np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (significant_mask, q_values) using Benjamini-Hochberg FDR."""
    p = np.asarray(p_values, dtype="float64")
    flat = p.ravel()
    valid = np.isfinite(flat)
    q = np.full(flat.shape, np.nan, dtype="float64")
    significant = np.zeros(flat.shape, dtype=bool)
    if valid.any():
        valid_p = flat[valid]
        order = np.argsort(valid_p)
        ordered = valid_p[order]
        ranks = np.arange(1, ordered.size + 1, dtype="float64")
        ordered_q = ordered * ordered.size / ranks
        ordered_q = np.minimum.accumulate(ordered_q[::-1])[::-1]
        ordered_q = np.clip(ordered_q, 0.0, 1.0)
        valid_q = np.empty_like(ordered_q)
        valid_q[order] = ordered_q
        q[valid] = valid_q
        significant[valid] = valid_q <= alpha
    return significant.reshape(p.shape), q.reshape(p.shape)
