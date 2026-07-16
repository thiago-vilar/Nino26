from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from nino_brasil.data.download_validation_insitu import validation_csv_to_zarr


def test_incremental_validation_tail_merges_and_replaces_overlap(tmp_path: Path) -> None:
    raw = tmp_path / "tail.csv"
    output = tmp_path / "validation.zarr"
    pd.DataFrame(
        {
            "time": ["2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z"],
            "station": ["A", "A"],
            "depth": [10.0, 10.0],
            "T_20": [25.0, 25.1],
        }
    ).to_csv(raw, index=False)
    validation_csv_to_zarr(raw, output, source="TAO test")

    pd.DataFrame(
        {
            "time": ["2026-07-02T00:00:00Z", "2026-07-03T00:00:00Z"],
            "station": ["A", "A"],
            "depth": [10.0, 10.0],
            "T_20": [26.1, 25.2],
        }
    ).to_csv(raw, index=False)
    validation_csv_to_zarr(raw, output, source="TAO test", merge_existing=True)

    with xr.open_zarr(output, consolidated=True) as dataset:
        frame = dataset.to_dataframe().reset_index(drop=True)
    assert len(frame) == 3
    revised = frame.loc[pd.to_datetime(frame["time"]).eq(pd.Timestamp("2026-07-02")), "T_20"]
    assert revised.iloc[0] == 26.1
