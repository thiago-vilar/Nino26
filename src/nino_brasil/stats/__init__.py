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
from nino_brasil.stats.phase3_inference import (
    EventBootstrapLagResult,
    PRECURSOR_TARGET_EXCLUSION_POLICY,
    bootstrap_lag_selection_by_event,
    is_phase3_target_alias,
    phase3_precursor_columns,
    scan_lagged_correlations,
    select_best_lags,
    simes_field_p_value,
)
from nino_brasil.stats.preprocessing import (
    SeasonalTrendConfig,
    SeasonalTrendTransformer,
    full_sample_diagnostic_transform,
)
from nino_brasil.stats.validation import (
    EventFold,
    event_purged_rolling_origin_folds,
    folds_audit_table,
    required_purge_weeks,
)
from nino_brasil.stats.semantic_tables import (
    SemanticTableContract,
    SemanticTableOutput,
    sha256_file,
    verify_semantic_csv,
    write_semantic_csv,
)

__all__ = [
    "BootstrapCorrelationResult",
    "EventBootstrapLagResult",
    "EventFold",
    "SeasonalTrendConfig",
    "SeasonalTrendTransformer",
    "SemanticTableContract",
    "SemanticTableOutput",
    "benjamini_hochberg_fdr",
    "bootstrap_lag_selection_by_event",
    "correlation_p_value",
    "effective_sample_size",
    "event_purged_rolling_origin_folds",
    "folds_audit_table",
    "full_sample_diagnostic_transform",
    "harmonic_weekly_climatology",
    "is_phase3_target_alias",
    "lag1_autocorrelation",
    "leave_one_event_out_correlation",
    "moving_block_bootstrap_correlation",
    "partial_correlation",
    "phase3_precursor_columns",
    "PRECURSOR_TARGET_EXCLUSION_POLICY",
    "required_purge_weeks",
    "scan_lagged_correlations",
    "select_best_lags",
    "sha256_file",
    "simes_field_p_value",
    "summarize_correlation_stability",
    "verify_semantic_csv",
    "write_semantic_csv",
]
