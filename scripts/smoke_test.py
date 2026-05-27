from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.data.anomalies import daily_anomaly
from nino_brasil.data.build_lagged_dataset import align_predictor_target
from nino_brasil.features.ocean_heat import layer_mean_temperature, ocean_heat_content
from nino_brasil.features.precipitation_events import event_mask, rolling_accumulation
from nino_brasil.features.thermocline import d20_depth
from nino_brasil.maps.plot_pixel_maps import save_pixel_map


def synthetic_fields() -> tuple[xr.Dataset, xr.DataArray, xr.DataArray]:
    rng = np.random.default_rng(42)
    time = pd.date_range("2001-01-01", "2003-12-31", freq="D")
    lat_p = np.linspace(-10, 10, 6)
    lon_p = np.linspace(180, 240, 8)
    lat_b = np.linspace(-34, 5, 10)
    lon_b = np.linspace(-74, -34, 12)
    depth = np.array([0, 50, 100, 200, 300, 500, 700])

    seasonal = np.sin(2 * np.pi * np.arange(time.size) / 365.25)
    sst = 27 + seasonal[:, None, None] + 0.2 * rng.normal(size=(time.size, lat_p.size, lon_p.size))
    slp = 1010 + 2 * seasonal[:, None, None] + rng.normal(size=(time.size, lat_p.size, lon_p.size))
    precip = 5 + 0.8 * np.roll(seasonal, 30)[:, None, None] + rng.gamma(2, 1, size=(time.size, lat_b.size, lon_b.size))

    temp_profile = (
        28
        - 0.018 * depth[None, :, None, None]
        + 0.4 * seasonal[:, None, None, None]
        + 0.05 * rng.normal(size=(time.size, depth.size, lat_p.size, lon_p.size))
    )

    predictors = xr.Dataset(
        {
            "sst": (("time", "lat", "lon"), sst),
            "slp": (("time", "lat", "lon"), slp),
        },
        coords={"time": time, "lat": lat_p, "lon": lon_p},
    )
    precipitation = xr.DataArray(
        precip,
        coords={"time": time, "lat": lat_b, "lon": lon_b},
        dims=("time", "lat", "lon"),
        name="precipitation",
    )
    temperature = xr.DataArray(
        temp_profile,
        coords={"time": time, "depth": depth, "lat": lat_p, "lon": lon_p},
        dims=("time", "depth", "lat", "lon"),
        name="temperature",
    )
    return predictors, precipitation, temperature


def main() -> None:
    predictors, precipitation, temperature = synthetic_fields()

    sst_anom = daily_anomaly(predictors["sst"])
    precip_anom = daily_anomaly(precipitation)
    precip_30d = rolling_accumulation(precipitation, 30)
    dry_mask = event_mask(precipitation, 10, "dry")
    x_lag, y_lag = align_predictor_target(predictors[["sst", "slp"]], precip_anom, lag_days=30)

    t300 = layer_mean_temperature(temperature, 300)
    ohc300 = ocean_heat_content(temperature, 300)
    d20 = d20_depth(temperature)

    output = ROOT / "docs" / "assets" / "maps" / "smoke_precip_anomaly.png"
    save_pixel_map(precip_anom.isel(time=100), output, "Synthetic precipitation anomaly")

    print("Smoke test OK")
    print(f"sst_anom shape: {sst_anom.shape}")
    print(f"precip_30d shape: {precip_30d.shape}")
    print(f"dry_mask true count: {int(dry_mask.sum().values)}")
    print(f"lagged X time: {x_lag.sizes['time']}; lagged Y time: {y_lag.sizes['time']}")
    print(f"t300 shape: {t300.shape}; ohc300 shape: {ohc300.shape}; d20 shape: {d20.shape}")
    print(f"map: {output}")


if __name__ == "__main__":
    main()
