from __future__ import annotations

import numpy as np
import pytest

from nino_brasil.maps.spatial_support import _regular_lattice_step


def test_regular_lattice_accepts_missing_native_columns() -> None:
    coordinates = np.array([-74.875, -74.625, -74.375, -71.875])

    step = _regular_lattice_step(coordinates, axis_name="Longitude")

    assert step == pytest.approx(0.25)


def test_regular_lattice_rejects_true_irregular_spacing() -> None:
    coordinates = np.array([-74.875, -74.625, -74.34, -74.09])

    with pytest.raises(ValueError, match="regular lattice"):
        _regular_lattice_step(coordinates, axis_name="Longitude")
