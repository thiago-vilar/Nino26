"""Statistical safeguards used by NINO-BRASIL analyses."""

from nino_brasil.stats.climatology import harmonic_weekly_climatology
from nino_brasil.stats.significance import (
    benjamini_hochberg_fdr,
    correlation_p_value,
    effective_sample_size,
    lag1_autocorrelation,
    partial_correlation,
)
from nino_brasil.stats.temporal_stability import (
    BootstrapCorrelationResult,
    leave_one_event_out_correlation,
    moving_block_bootstrap_correlation,
    summarize_correlation_stability,
)

__all__ = [
    "BootstrapCorrelationResult",
    "benjamini_hochberg_fdr",
    "correlation_p_value",
    "effective_sample_size",
    "harmonic_weekly_climatology",
    "lag1_autocorrelation",
    "leave_one_event_out_correlation",
    "moving_block_bootstrap_correlation",
    "partial_correlation",
    "summarize_correlation_stability",
]
