"""Model training and validation modules."""

from nino_brasil.models.feature_matrix import (
    FeatureMatrix,
    PredictorMatrix,
    add_causal_monthly_ocean_features,
    align_target_to_predictor_matrix,
    build_feature_matrix,
    build_predictor_matrix,
)
from nino_brasil.models.walk_forward import WalkForwardFold, WalkForwardOutput, run_walk_forward
from nino_brasil.models.ablation import (
    OCEAN_FREQUENCY_ABLATIONS,
    PHASE1_ABLATIONS,
    PHASE2_ABLATIONS,
    PHYSICS_ABLATIONS,
    select_feature_groups,
)
from nino_brasil.models.progression import (
    DEFAULT_NINO_THRESHOLDS,
    NinoPeakThresholds,
    build_daily_enso_peak_progression_table,
    build_enso_peak_progression_table,
    build_nino34_cluster_progression_table,
    classify_nino_peak,
    cluster_pixel_events,
)

__all__ = [
    "DEFAULT_NINO_THRESHOLDS",
    "FeatureMatrix",
    "NinoPeakThresholds",
    "PHASE1_ABLATIONS",
    "PHASE2_ABLATIONS",
    "PHYSICS_ABLATIONS",
    "OCEAN_FREQUENCY_ABLATIONS",
    "PredictorMatrix",
    "WalkForwardFold",
    "WalkForwardOutput",
    "align_target_to_predictor_matrix",
    "add_causal_monthly_ocean_features",
    "build_daily_enso_peak_progression_table",
    "build_enso_peak_progression_table",
    "build_feature_matrix",
    "build_nino34_cluster_progression_table",
    "build_predictor_matrix",
    "classify_nino_peak",
    "cluster_pixel_events",
    "run_walk_forward",
    "select_feature_groups",
]
