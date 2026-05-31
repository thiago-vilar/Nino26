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
