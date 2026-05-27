from __future__ import annotations

import pandas as pd


def temporal_split(
    times: pd.DatetimeIndex,
    train_end: str,
    validation_end: str,
) -> dict[str, pd.DatetimeIndex]:
    """Create chronological train, validation and test splits."""
    train = times[times <= pd.Timestamp(train_end)]
    validation = times[(times > pd.Timestamp(train_end)) & (times <= pd.Timestamp(validation_end))]
    test = times[times > pd.Timestamp(validation_end)]
    return {"train": train, "validation": validation, "test": test}
