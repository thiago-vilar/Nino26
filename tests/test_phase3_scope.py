from __future__ import annotations

import pandas as pd

from nino_brasil.features.phase3_diagnostics import _drop_phase3_excluded_columns
from scripts.run_fase4c_regional import PACIFIC_VARS


def test_phase3_public_predictor_contract_excludes_salinity() -> None:
    assert len(PACIFIC_VARS) == 31
    assert not any("salinity" in value.lower() or "sss" in value.lower() for value in PACIFIC_VARS)


def test_phase3_public_signal_drops_salinity_columns() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-01-01"]),
            "nino34_ssta": [0.7],
            "sss_nino34_mean": [35.1],
            "absolute_salinity_nino34": [35.2],
        }
    )
    output = _drop_phase3_excluded_columns(frame)
    assert "nino34_ssta" in output
    assert "sss_nino34_mean" not in output
    assert "absolute_salinity_nino34" not in output


def test_phase3_wind_contract_uses_anomaly_not_raw_proxy() -> None:
    assert "tau_x_anom" in PACIFIC_VARS
    assert "tau_x_proxy_nino34_pa" not in PACIFIC_VARS
