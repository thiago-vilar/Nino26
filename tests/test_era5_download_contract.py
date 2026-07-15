from __future__ import annotations

from nino_brasil.data.download_cds import era5_daily_aggregation


def test_era5_instantaneous_fields_use_daily_mean() -> None:
    assert era5_daily_aggregation("mean_sea_level_pressure") == "mean"
    assert era5_daily_aggregation("u_component_of_wind") == "mean"


def test_era5_accumulated_fluxes_use_daily_sum() -> None:
    assert era5_daily_aggregation("surface_latent_heat_flux") == "sum"
    assert era5_daily_aggregation("surface_net_solar_radiation") == "sum"
