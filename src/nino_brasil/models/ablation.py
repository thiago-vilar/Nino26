from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from nino_brasil.models.feature_matrix import FeatureMatrix


PHASE1_ABLATIONS = {
    "A_ocean": {"ocean"},
    "B_atmosphere": {"atmosphere"},
    "C_ocean_atmosphere": {"ocean", "atmosphere"},
}

PHASE2_ABLATIONS = {
    "D_without_subsurface": {"ocean", "atmosphere"},
    "E_without_upper_levels": {"ocean", "atmosphere"},
    "F_without_humidity": {"ocean", "atmosphere"},
}

PHYSICS_ABLATIONS = {
    "G_with_physics": {"ocean", "atmosphere", "physics_precalc"},
    "H_without_physics": {"ocean", "atmosphere"},
}

OCEAN_FREQUENCY_ABLATIONS = {
    "I_surface_atmosphere_only": {"ocean_surface", "atmosphere"},
    "J_plus_daily_ocean": {"ocean_surface", "atmosphere", "ocean_daily"},
    "K_plus_monthly_oras5": {"ocean_surface", "atmosphere", "ocean_monthly"},
    "L_daily_plus_monthly_ocean": {"ocean_surface", "atmosphere", "ocean_daily", "ocean_monthly"},
    "M_daily_monthly_plus_physics": {
        "ocean_surface",
        "atmosphere",
        "ocean_daily",
        "ocean_monthly",
        "physics_precalc",
    },
}


def select_feature_groups(
    matrix: FeatureMatrix,
    allowed_groups: Iterable[str],
    *,
    keep_calendar: bool = True,
) -> FeatureMatrix:
    """Return a FeatureMatrix restricted to selected physical feature groups."""
    allowed = set(allowed_groups)
    if keep_calendar:
        allowed.add("calendar")
    columns = [col for col in matrix.X.columns if matrix.feature_groups.get(col, "other") in allowed]
    return replace(
        matrix,
        X=matrix.X.loc[:, columns],
        feature_groups={col: matrix.feature_groups[col] for col in columns},
    )
