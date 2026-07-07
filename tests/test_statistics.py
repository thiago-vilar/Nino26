from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.stats.significance import (
    benjamini_hochberg_fdr,
    correlation_p_value,
    effective_sample_size,
    partial_correlation,
)
from nino_brasil.stats.climatology import harmonic_weekly_climatology


class StatisticsTests(unittest.TestCase):
    def test_effective_sample_size_drops_for_autocorrelated_series(self) -> None:
        x = np.arange(200, dtype=float)
        y = x + 0.01

        n_eff = effective_sample_size(x, y)

        self.assertGreaterEqual(n_eff, 3.0)
        self.assertLess(n_eff, float(x.size))

    def test_correlation_p_value_uses_effective_n(self) -> None:
        p_small_n = correlation_p_value(0.5, 5.0)
        p_large_n = correlation_p_value(0.5, 100.0)

        self.assertGreater(p_small_n, p_large_n)

    def test_benjamini_hochberg_fdr_returns_q_values(self) -> None:
        mask, q = benjamini_hochberg_fdr(np.array([0.001, 0.02, 0.2, np.nan]), alpha=0.05)

        self.assertTrue(bool(mask[0]))
        self.assertTrue(bool(mask[1]))
        self.assertFalse(bool(mask[2]))
        self.assertTrue(np.isnan(q[3]))

    def test_partial_correlation_removes_linear_control_signal(self) -> None:
        control = np.linspace(-2.0, 2.0, 80)
        x = control + 0.01 * np.sin(np.arange(control.size))
        y = control + 0.01 * np.cos(np.arange(control.size))

        raw_r = float(np.corrcoef(x, y)[0, 1])
        partial = partial_correlation(x, y, control)

        self.assertGreater(raw_r, 0.99)
        self.assertLess(abs(partial["r"]), 0.2)
        self.assertEqual(partial["n"], 80.0)

    def test_harmonic_weekly_climatology_recovers_smooth_annual_cycle(self) -> None:
        index = pd.date_range("2001-01-07", periods=104, freq="W-SUN")
        angle = 2.0 * np.pi * (index.dayofyear.to_numpy() - 1.0) / 365.2425
        values = 2.0 + 1.5 * np.sin(angle) - 0.5 * np.cos(angle)
        series = pd.Series(values, index=index, name="ssta")

        anom, clim = harmonic_weekly_climatology(series, harmonics=1)

        self.assertLess(float(np.nanmax(np.abs(anom.to_numpy()))), 1e-10)
        self.assertAlmostEqual(float(clim.iloc[0]), float(series.iloc[0]), places=10)


if __name__ == "__main__":
    unittest.main()
