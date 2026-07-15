from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.build_phase4_chirps_targets import (
    _pixel_mask_fingerprint,
    materialize_daily_native,
    validate_canonical_target,
)
from nino_brasil.targets.chirps_native import (
    TARGET_CONTRACT_VERSION,
    add_brazil_mask,
    build_native_weekly_targets,
    concat_daily_native,
    gamma_spi_weekly,
    native_grid_hash,
    native_pixel_table,
    robust_weekly_anomalies,
    target_to_frame,
    weekly_extreme_indices,
    weekly_native_precipitation,
    validate_native_target,
)


LAT = np.array([-1.875, -1.625], dtype="float32")
LON = np.array([-40.875, -40.625, -40.375], dtype="float32")


def test_native_grid_hash_is_frozen_independently_of_target_method_version() -> None:
    assert native_grid_hash(LAT, LON) == (
        "c617781e7723624f3869d01146fe368ca80afd587932e7eca41de5d90fbfff5d"
    )


def test_pixel_mask_fingerprint_survives_csv_float_promotion(tmp_path) -> None:
    pixels = native_pixel_table(LAT, LON)
    pixels["brazil_fraction"] = np.array(
        [0.0, 0.12345679, 0.50000006, 0.7777778, 0.99999994, 1.0],
        dtype="float32",
    )
    pixels["brazil_center"] = pixels["brazil_fraction"] >= 0.5
    pixels["brazil_mask_method"] = "test_fraction"
    path = tmp_path / "pixels.csv"

    pixels.to_csv(path, index=False)
    reloaded = pd.read_csv(path)

    assert pixels["brazil_fraction"].dtype == np.dtype("float32")
    assert reloaded["brazil_fraction"].dtype == np.dtype("float64")
    assert _pixel_mask_fingerprint(reloaded) == _pixel_mask_fingerprint(pixels)


def daily_array(start: str, periods: int, value: float = 1.0) -> xr.DataArray:
    time = pd.date_range(start, periods=periods, freq="D")
    values = np.full((periods, len(LAT), len(LON)), value, dtype="float32")
    return xr.DataArray(
        values,
        coords={"time": time, "latitude": LAT, "longitude": LON},
        dims=("time", "latitude", "longitude"),
        name="precip",
    )


def test_daily_is_concatenated_before_one_weekly_sum_across_year_boundary() -> None:
    first = daily_array("2000-12-25", 7, 1.0)
    second = daily_array("2001-01-01", 7, 2.0)
    daily = concat_daily_native([first, second])
    weekly = weekly_native_precipitation(daily)

    assert daily.sizes["time"] == 14
    assert weekly.sizes["time"] == 2
    np.testing.assert_allclose(weekly["precip_weekly_mm"].isel(time=0), 7.0)
    np.testing.assert_allclose(weekly["precip_weekly_mm"].isel(time=1), 14.0)
    assert bool(weekly["week_complete"].all())
    assert int(weekly["valid_day_count"].min()) == 7


def test_duplicate_daily_dates_are_rejected_instead_of_averaged() -> None:
    first = daily_array("2001-01-01", 4)
    second = daily_array("2001-01-04", 4)
    with pytest.raises(ValueError, match="Duplicate CHIRPS daily timestamps"):
        concat_daily_native([first, second])


def test_persistent_daily_concat_is_a_pre_resample_memory_boundary(tmp_path) -> None:
    stores = []
    for year, start, value in (
        (2000, "2000-12-25", 1.0),
        (2001, "2001-01-01", 2.0),
    ):
        path = tmp_path / f"chirps_p25_{year}.zarr"
        daily_array(start, 7, value).to_dataset(name="precip").to_zarr(
            path, mode="w", zarr_format=2
        )
        stores.append(path)
    destination = tmp_path / "daily_native.zarr"
    daily = materialize_daily_native(stores, destination, rebuild=False)
    assert destination.exists()
    assert daily.sizes["time"] == 14
    assert not pd.DatetimeIndex(daily.time.values).has_duplicates
    assert daily.attrs["source_store_count"] == 2
    assert len(daily.attrs["daily_data_content_sha256"]) == 64
    weekly = weekly_native_precipitation(daily).compute()
    np.testing.assert_allclose(weekly["precip_weekly_mm"].isel(time=0), 7.0)
    np.testing.assert_allclose(weekly["precip_weekly_mm"].isel(time=1), 14.0)


def test_partial_edge_week_is_retained_flagged_and_not_used_as_target() -> None:
    daily = daily_array("2001-01-03", 12)
    weekly = weekly_native_precipitation(daily)
    assert bool(weekly["week_is_partial_edge"].isel(time=0))
    assert np.isnan(weekly["precip_weekly_mm"].isel(time=0)).all()
    assert int(weekly["expected_day_count"].isel(time=0)) == 5
    assert bool(weekly["week_complete"].isel(time=-1).all())


def synthetic_weekly() -> xr.DataArray:
    time = pd.date_range("1991-01-06", "2021-12-26", freq="W-SUN")
    rng = np.random.default_rng(42)
    week = pd.DatetimeIndex(time).isocalendar().week.to_numpy()
    seasonal = 20 + 8 * np.sin(2 * np.pi * week / 52.1775)
    values = seasonal[:, None, None] + rng.normal(0, 2, (len(time), 2, 3))
    values[:, 0, 0] = 0.0  # undefined variance must not become a huge z score
    return xr.DataArray(
        values.astype("float32"),
        coords={"time": time, "latitude": LAT, "longitude": LON},
        dims=("time", "latitude", "longitude"),
        name="precip_weekly_mm",
    )


def test_robust_anomaly_never_uses_near_zero_scale() -> None:
    result = robust_weekly_anomalies(synthetic_weekly())
    assert {
        "climatology_pooled_residual_mad_scale",
        "climatology_pooled_residual_l1_scale",
        "climatology_scale_source_code",
    }.issubset(result.data_vars)
    assert np.isnan(result["precip_robust_z"].sel(latitude=LAT[0], longitude=LON[0])).all()
    finite = np.abs(result["precip_robust_z"].values)
    finite = finite[np.isfinite(finite)]
    assert finite.size
    assert float(finite.max()) < 20
    positive_scale = result["climatology_robust_scale_mm"].values
    positive_scale = positive_scale[np.isfinite(positive_scale)]
    assert float(positive_scale.min()) >= 0.10


def test_zero_inflated_rainfall_uses_data_derived_l1_scale_lower_bound() -> None:
    time = pd.date_range("1991-01-06", "2021-12-26", freq="W-SUN")
    rng = np.random.default_rng(20260713)
    wet = rng.random(len(time)) < 0.18
    values = np.where(wet, rng.gamma(shape=1.4, scale=12.0, size=len(time)), 0.0)
    target_time = pd.Timestamp("2021-11-14")
    values[np.flatnonzero(time == target_time)[0]] = 120.0
    weekly = xr.DataArray(
        values[:, None, None].astype("float32"),
        coords={"time": time, "latitude": [-9.125], "longitude": [-40.125]},
        dims=("time", "latitude", "longitude"),
        name="precip_weekly_mm",
    )

    raw_sha256 = hashlib.sha256(weekly.values.tobytes()).hexdigest()
    threshold_floor = xr.DataArray(
        [[10.0]],
        coords={"latitude": weekly.latitude, "longitude": weekly.longitude},
        dims=("latitude", "longitude"),
        attrs={"units": "mm day-1"},
    )
    result = robust_weekly_anomalies(
        weekly,
        zero_inflated_tail=True,
        threshold_floor=threshold_floor,
    ).compute()
    observed = float(result["precip_robust_z"].sel(time=target_time).item())
    week = int(target_time.isocalendar().week)
    source_code = int(
        result["climatology_scale_source_code"].sel(week_of_year=week).item()
    )
    selected_scale = float(
        result["climatology_robust_scale_mm"].sel(week_of_year=week).item()
    )
    pooled_positive_l1 = float(
        result["climatology_pooled_positive_l1_scale"].item()
    )

    assert source_code == 3
    assert selected_scale == pytest.approx(pooled_positive_l1)
    assert int(result["climatology_positive_week_count"].item()) >= 20
    assert int(result["climatology_tail_fallback_code"].item()) == 0
    assert 0.0 < observed < 20.0
    assert hashlib.sha256(weekly.values.tobytes()).hexdigest() == raw_sha256

    event_week = int(target_time.isocalendar().week)
    reconstructed = (
        observed
        * float(
            result["climatology_robust_scale_mm"]
            .sel(week_of_year=event_week)
            .item()
        )
        + float(
            result["climatology_median_mm"]
            .sel(week_of_year=event_week)
            .item()
        )
    )
    assert reconstructed == pytest.approx(
        float(weekly.sel(time=target_time).item()), rel=1e-6
    )


def test_conditional_positive_scale_is_invariant_to_padding_with_zeros() -> None:
    short_time = pd.date_range("1991-01-06", "2011-12-25", freq="W-SUN")
    long_time = pd.date_range("1991-01-06", "2020-12-27", freq="W-SUN")
    short_values = np.zeros(len(short_time), dtype="float32")
    positive_positions = np.linspace(5, len(short_time) - 5, 24, dtype=int)
    short_values[positive_positions] = np.linspace(8.0, 48.0, 24, dtype="float32")
    long_values = np.zeros(len(long_time), dtype="float32")
    long_values[: len(short_values)] = short_values

    def transform(time: pd.DatetimeIndex, values: np.ndarray, baseline_end: str) -> xr.Dataset:
        weekly = xr.DataArray(
            values[:, None, None],
            coords={"time": time, "latitude": [-9.125], "longitude": [-40.125]},
            dims=("time", "latitude", "longitude"),
        )
        threshold = xr.DataArray(
            [[8.0]],
            coords={"latitude": weekly.latitude, "longitude": weekly.longitude},
            dims=("latitude", "longitude"),
        )
        return robust_weekly_anomalies(
            weekly,
            baseline=("1991-01-01", baseline_end),
            zero_inflated_tail=True,
            threshold_floor=threshold,
        ).compute()

    short = transform(short_time, short_values, "2011-12-31")
    padded = transform(long_time, long_values, "2020-12-31")

    assert int(short["climatology_positive_week_count"].item()) == 24
    assert int(padded["climatology_positive_week_count"].item()) == 24
    assert float(short["climatology_pooled_residual_l1_scale"].item()) > float(
        padded["climatology_pooled_residual_l1_scale"].item()
    )
    assert float(short["climatology_pooled_positive_l1_scale"].item()) == pytest.approx(
        float(padded["climatology_pooled_positive_l1_scale"].item()), rel=1e-7
    )


def test_sparse_tail_uses_audited_threshold_floor_without_imputation() -> None:
    time = pd.date_range("1991-01-06", "2021-12-26", freq="W-SUN")
    values = np.zeros(len(time), dtype="float32")
    baseline_positions = np.linspace(10, 52 * 29, 10, dtype=int)
    values[baseline_positions] = np.linspace(12.0, 30.0, 10, dtype="float32")
    event_time = pd.Timestamp("2021-11-14")
    values[np.flatnonzero(time == event_time)[0]] = 96.0
    weekly = xr.DataArray(
        values[:, None, None],
        coords={"time": time, "latitude": [-9.125], "longitude": [-40.125]},
        dims=("time", "latitude", "longitude"),
    )
    threshold = xr.DataArray(
        [[12.0]],
        coords={"latitude": weekly.latitude, "longitude": weekly.longitude},
        dims=("latitude", "longitude"),
    )

    result = robust_weekly_anomalies(
        weekly,
        zero_inflated_tail=True,
        threshold_floor=threshold,
    ).compute()
    week = int(event_time.isocalendar().week)

    assert int(result["climatology_positive_week_count"].item()) == 10
    assert int(result["climatology_tail_fallback_code"].item()) == 1
    assert int(
        result["climatology_scale_source_code"]
        .sel(week_of_year=week)
        .item()
    ) == 4
    assert float(
        result["climatology_robust_scale_mm"].sel(week_of_year=week).item()
    ) == 12.0
    assert float(result["precip_robust_z"].sel(time=event_time).item()) == pytest.approx(
        8.0
    )
    np.testing.assert_array_equal(weekly.values[:, 0, 0], values)


def test_target_v4_preserves_raw_extremes_coordinates_pixels_and_masks_bitwise() -> None:
    time = pd.date_range("1991-01-01", "2021-12-31", freq="D")
    values = np.full((len(time), len(LAT), len(LON)), 2.0, dtype="float32")
    values[::90, :, :] = 30.0
    values[45::180, :, :] = 55.0
    daily = xr.DataArray(
        values,
        coords={"time": time, "latitude": LAT, "longitude": LON},
        dims=("time", "latitude", "longitude"),
        name="precip",
        attrs={"units": "mm"},
    )
    weekly = weekly_native_precipitation(daily)
    expected = weekly_extreme_indices(daily)
    expected_r95 = expected["r95p_weekly_mm"].where(weekly["week_complete"])
    expected_r99 = expected["r99p_weekly_mm"].where(weekly["week_complete"])
    expected_hashes = {
        "r95p_weekly_mm": hashlib.sha256(expected_r95.values.tobytes()).hexdigest(),
        "r99p_weekly_mm": hashlib.sha256(expected_r99.values.tobytes()).hexdigest(),
    }
    pixels = native_pixel_table(LAT, LON)
    pixels["brazil_fraction"] = np.linspace(0.0, 1.0, len(pixels), dtype="float32")
    pixels["brazil_center"] = pixels["brazil_fraction"] >= 0.5

    target = build_native_weekly_targets(
        daily,
        pixels=pixels,
        include_spi=False,
        include_extremes=True,
    ).compute()

    for name, expected_array in (
        ("r95p_weekly_mm", expected_r95),
        ("r99p_weekly_mm", expected_r99),
    ):
        np.testing.assert_array_equal(
            target[name].values,
            expected_array.values,
        )
        assert hashlib.sha256(target[name].values.tobytes()).hexdigest() == expected_hashes[name]
    np.testing.assert_array_equal(target.latitude.values, LAT)
    np.testing.assert_array_equal(target.longitude.values, LON)
    np.testing.assert_array_equal(
        target["pixel_id"].values.ravel(), pixels["pixel_id"].to_numpy()
    )
    np.testing.assert_array_equal(
        target["brazil_fraction"].values.ravel(),
        pixels["brazil_fraction"].to_numpy(),
    )
    np.testing.assert_array_equal(
        target["brazil_center"].values.ravel(),
        pixels["brazil_center"].to_numpy(),
    )
    assert target.attrs["target_contract_version"] == "chirps-native-weekly-v4"
    assert {
        "r95p_weekly_climatology_positive_week_count",
        "r95p_weekly_climatology_pooled_positive_l1_scale",
        "r95p_weekly_climatology_tail_threshold_floor",
        "r95p_weekly_climatology_tail_fallback_code",
        "r99p_weekly_climatology_positive_week_count",
        "r99p_weekly_climatology_pooled_positive_l1_scale",
        "r99p_weekly_climatology_tail_threshold_floor",
        "r99p_weekly_climatology_tail_fallback_code",
    }.issubset(target.data_vars)


def test_robust_anomaly_executes_from_chunked_persistent_input(tmp_path) -> None:
    path = tmp_path / "weekly.zarr"
    synthetic_weekly().to_dataset(name="precip_weekly_mm").to_zarr(
        path,
        mode="w",
        zarr_format=2,
        encoding={"precip_weekly_mm": {"chunks": (52, 1, 1)}},
    )
    chunked = xr.open_zarr(path)["precip_weekly_mm"]
    result = robust_weekly_anomalies(chunked).compute()
    finite = np.abs(result["precip_robust_z"].values)
    finite = finite[np.isfinite(finite)]
    assert finite.size
    assert float(finite.max()) < 20


def test_gamma_spi_scaffold_has_auditable_provisional_contract() -> None:
    spi = gamma_spi_weekly(synthetic_weekly(), accumulation_weeks=13)
    assert spi.name == "spi_gamma_3m_weekly_origin"
    assert "provisional" in spi.attrs["status"]
    finite = spi.values[np.isfinite(spi.values)]
    assert finite.size
    assert np.nanmax(np.abs(finite)) < 5
    diagnostics = gamma_spi_weekly(
        synthetic_weekly(), accumulation_weeks=13, return_parameters=True
    )
    assert isinstance(diagnostics, xr.Dataset)
    assert {
        "spi_gamma_3m_shape_by_iso_week",
        "spi_gamma_3m_scale_by_iso_week",
        "spi_gamma_3m_zero_probability_by_iso_week",
        "spi_gamma_3m_sample_count_by_iso_week",
    }.issubset(diagnostics)


def test_native_pixel_ids_and_grid_hash_are_stable_and_masks_do_not_drop_cells() -> None:
    first = native_pixel_table(LAT, LON)
    second = native_pixel_table(LAT.copy(), LON.copy())
    pd.testing.assert_frame_equal(first, second)
    assert first["pixel_id"].tolist() == list(range(6))
    assert first["grid_hash"].nunique() == 1
    assert first["grid_hash"].iloc[0] == native_grid_hash(LAT, LON)

    time = pd.date_range("2001-01-07", periods=2, freq="W-SUN")
    dataset = xr.Dataset(
        {
            "precip_robust_z": xr.DataArray(
                np.arange(12, dtype="float32").reshape(2, 2, 3),
                coords={"time": time, "latitude": LAT, "longitude": LON},
                dims=("time", "latitude", "longitude"),
            )
        }
    )
    first["brazil_fraction"] = [0.0, 0.2, 1.0, 0.0, 1.0, 1.0]
    first["brazil_center"] = [False, False, True, False, True, True]
    masked = add_brazil_mask(dataset, first)
    assert masked.sizes["latitude"] * masked.sizes["longitude"] == 6
    frame, pixels = target_to_frame(masked, brazil_only=True, mask_rule="center")
    assert frame.shape == (2, 3)
    assert pixels["pixel_id"].tolist() == [2, 4, 5]
    np.testing.assert_allclose(frame.iloc[0].to_numpy(), [2, 4, 5])


def test_weekly_extreme_scaffold_uses_daily_native_values() -> None:
    daily = daily_array("1991-01-01", 365 * 2, 2.0)
    # Add one clearly extreme day after enough wet-day baseline samples.
    daily.loc[{"time": pd.Timestamp("1992-06-01")}] = 30.0
    result = weekly_extreme_indices(
        daily,
        baseline=("1991-01-01", "1992-12-31"),
    )
    assert {
        "rx1day_weekly_mm",
        "rx5day_weekly_mm",
        "r95p_weekly_mm",
        "r99p_weekly_mm",
        "cdd_within_week_days",
        "cwd_within_week_days",
    }.issubset(result.data_vars)
    assert float(result["rx1day_weekly_mm"].max()) == 30.0
    assert int(result["cwd_within_week_days"].max()) == 7
    assert {
        "baseline_wet_day_count",
        "baseline_wet_day_p95_mm",
        "baseline_wet_day_p99_mm",
    }.issubset(result)


def test_wet_day_percentiles_remain_missing_without_baseline_support() -> None:
    dry = daily_array("1991-01-01", 365 * 2, 0.0)
    result = weekly_extreme_indices(
        dry,
        baseline=("1991-01-01", "1992-12-31"),
        minimum_baseline_wet_days=100,
    ).compute()
    assert int(result["baseline_wet_day_count"].max()) == 0
    assert result["baseline_wet_day_p95_mm"].isnull().all()
    assert result["r95p_weekly_mm"].isnull().all()


def test_iso_week_53_uses_explicit_effective_sample_fallback() -> None:
    result = robust_weekly_anomalies(synthetic_weekly()).compute().sel(week_of_year=53)
    assert int(result["climatology_sample_count"].max()) < 20
    assert int(result["climatology_effective_sample_count"].min()) >= 20
    assert (result["climatology_fallback_code"] == 1).all()
    assert result["climatology_median_mm"].notnull().all()


def _minimal_validated_target() -> xr.Dataset:
    time = pd.date_range("2001-01-07", periods=2, freq="W-SUN")
    shape = (len(time), len(LAT), len(LON))
    pixels = native_pixel_table(LAT, LON)
    dataset = xr.Dataset(
        {
            "precip_weekly_mm": (("time", "latitude", "longitude"), np.ones(shape)),
            "valid_day_count": (("time", "latitude", "longitude"), np.full(shape, 7)),
            "expected_day_count": (("time",), np.full(len(time), 7)),
            "week_complete": (("time", "latitude", "longitude"), np.ones(shape, dtype=bool)),
            "precip_anomaly_mm": (("time", "latitude", "longitude"), np.zeros(shape)),
            "precip_robust_z": (("time", "latitude", "longitude"), np.zeros(shape)),
            "precip_robust_percentile": (("time", "latitude", "longitude"), np.full(shape, 0.5)),
            "climatology_median_mm": (
                ("week_of_year", "latitude", "longitude"),
                np.zeros((1, len(LAT), len(LON))),
            ),
            "climatology_robust_scale_mm": (
                ("week_of_year", "latitude", "longitude"),
                np.ones((1, len(LAT), len(LON))),
            ),
            "climatology_pooled_residual_mad_scale": (
                ("latitude", "longitude"),
                np.ones((len(LAT), len(LON))),
            ),
            "climatology_pooled_residual_l1_scale": (
                ("latitude", "longitude"),
                np.ones((len(LAT), len(LON))),
            ),
            "climatology_scale_source_code": (
                ("week_of_year", "latitude", "longitude"),
                np.zeros((1, len(LAT), len(LON)), dtype="uint8"),
            ),
            "pixel_id": (("latitude", "longitude"), pixels["pixel_id"].to_numpy().reshape(len(LAT), len(LON))),
            "brazil_fraction": (("latitude", "longitude"), np.ones((len(LAT), len(LON)))),
            "brazil_center": (("latitude", "longitude"), np.ones((len(LAT), len(LON)), dtype=bool)),
        },
        coords={"time": time, "week_of_year": [1], "latitude": LAT, "longitude": LON},
        attrs={
            "target_contract_version": TARGET_CONTRACT_VERSION,
            "grid_hash_sha256": native_grid_hash(LAT, LON),
            "spatial_operation": "coordinate subset only; interpolation=false",
            "build_status": "canonical",
            "include_spi": False,
            "include_extremes": False,
        },
    )
    return dataset


def test_native_target_validator_rejects_missing_hash_reversed_ids_and_week_gaps() -> None:
    valid = _minimal_validated_target()
    assert validate_native_target(valid, deep=False).valid
    missing_hash = valid.copy()
    missing_hash.attrs.pop("grid_hash_sha256")
    assert not validate_native_target(missing_hash, deep=False).valid
    reversed_ids = valid.copy()
    reversed_ids["pixel_id"] = xr.DataArray(
        reversed_ids["pixel_id"].values[::-1, ::-1],
        dims=("latitude", "longitude"),
    )
    assert not validate_native_target(reversed_ids, deep=False).valid
    gap = valid.assign_coords(time=[pd.Timestamp("2001-01-07"), pd.Timestamp("2001-01-21")])
    assert not validate_native_target(gap, deep=False).valid


def test_canonical_validator_rejects_unscaled_zero_inflated_tail_diagnostics() -> None:
    target = _minimal_validated_target()
    shape = target["precip_robust_z"].shape
    r99 = np.zeros(shape, dtype="float32")
    r99[0, 0, 0] = 500.0
    target["r99p_weekly_robust_z"] = xr.DataArray(
        r99,
        coords={
            "time": target.time,
            "latitude": target.latitude,
            "longitude": target.longitude,
        },
        dims=("time", "latitude", "longitude"),
    )

    validation = validate_canonical_target(target)

    assert not validation.valid
    assert float(target["r99p_weekly_robust_z"].max()) == 500.0
    assert any(
        "r99p_weekly_robust_z maximum |z|=500" in item
        for item in validation.errors
    )


def test_canonical_validator_keeps_primary_robust_targets_below_one_hundred() -> None:
    target = _minimal_validated_target()
    values = target["precip_robust_z"].values.copy()
    values[0, 0, 0] = 101.0
    target["precip_robust_z"] = xr.DataArray(
        values,
        coords={
            "time": target.time,
            "latitude": target.latitude,
            "longitude": target.longitude,
        },
        dims=("time", "latitude", "longitude"),
    )

    validation = validate_canonical_target(target)

    assert not validation.valid
    assert any("precip_robust_z maximum |z|=101" in item for item in validation.errors)


def test_non_precipitation_robust_anomaly_keeps_native_units() -> None:
    weekly = synthetic_weekly().clip(0, 7)
    weekly.attrs["units"] = "days"
    result = robust_weekly_anomalies(
        weekly,
        quantity_units="days",
        absolute_scale_floor_mm=0.25,
    )
    assert result["precip_anomaly_mm"].attrs["units"] == "days"
    assert "0.25 days" in result["precip_robust_z"].attrs["scale"]
