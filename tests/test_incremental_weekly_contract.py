from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.build_master_weekly import _complete_weekly_mean


def test_weekly_product_requires_seven_valid_daily_values_per_variable() -> None:
    index = pd.date_range("2026-06-01", periods=11, freq="D")
    frame = pd.DataFrame(
        {
            "complete": np.arange(11, dtype=float),
            "one_day_missing": np.arange(11, dtype=float),
        },
        index=index,
    )
    frame.loc[pd.Timestamp("2026-06-03"), "one_day_missing"] = np.nan

    weekly = _complete_weekly_mean(frame)

    assert weekly.loc[pd.Timestamp("2026-06-07"), "complete"] == 3.0
    assert np.isnan(weekly.loc[pd.Timestamp("2026-06-07"), "one_day_missing"])
    assert np.isnan(weekly.loc[pd.Timestamp("2026-06-14"), "complete"])
