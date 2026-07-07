from __future__ import annotations

import numpy as np
import pandas as pd


def _fourier_design(day_of_year: np.ndarray, harmonics: int) -> np.ndarray:
    angle = 2.0 * np.pi * (day_of_year.astype(float) - 1.0) / 365.2425
    columns = [np.ones_like(angle)]
    for harmonic in range(1, harmonics + 1):
        columns.append(np.sin(harmonic * angle))
        columns.append(np.cos(harmonic * angle))
    return np.column_stack(columns)


def harmonic_weekly_climatology(
    series: pd.Series,
    *,
    harmonics: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Fit a smooth annual Fourier climatology for weekly analysis.

    The caller is responsible for passing only the training block when the
    result is used in inference or skill estimates. This avoids the noisy
    52-bin week-of-year climatology while preserving the annual cycle.
    """
    if harmonics < 1:
        raise ValueError("harmonics must be at least 1.")
    s = series.sort_index()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("series index must be a DatetimeIndex.")
    observed = s.dropna()
    minimum = 1 + 2 * harmonics
    if observed.size < minimum:
        raise ValueError(f"at least {minimum} finite samples are required.")

    x_fit = _fourier_design(observed.index.dayofyear.to_numpy(), harmonics)
    coef, *_ = np.linalg.lstsq(x_fit, observed.to_numpy(dtype=float), rcond=None)
    x_all = _fourier_design(s.index.dayofyear.to_numpy(), harmonics)
    clim = pd.Series(x_all @ coef, index=s.index, name=f"{s.name}_harmonic_clim")
    anom = (s - clim).rename(f"{s.name}_harmonic_anom")
    return anom, clim
