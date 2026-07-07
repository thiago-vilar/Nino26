"""Statistical safeguards used by NINO-BRASIL analyses."""

from nino_brasil.stats.climatology import harmonic_weekly_climatology
from nino_brasil.stats.significance import (
    benjamini_hochberg_fdr,
    correlation_p_value,
    effective_sample_size,
    lag1_autocorrelation,
    partial_correlation,
)

__all__ = [
    "benjamini_hochberg_fdr",
    "correlation_p_value",
    "effective_sample_size",
    "harmonic_weekly_climatology",
    "lag1_autocorrelation",
    "partial_correlation",
]
