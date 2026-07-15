from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import scripts.run_fase4c_regional as phase4c
import scripts.run_fase4d_targets as phase4d
from scripts.run_fase4c_regional import compute_pixel_atlas
from scripts.run_fase4d_targets import CONFIRMATORY_TARGETS, build_gate
from nino_brasil.config import confirmatory_fdr_alpha
from nino_brasil.stats.lag_analysis import (
    PHASE_ORDER,
    SourceCondition,
    _columnwise_corr,
    _pooled_within_segment_corr,
    best_lag_fields,
    build_source_conditions,
    fdr_bh_adjusted,
    lagged_correlation_exact,
    pearson_columns_with_segmented_neff,
    result_to_long_table,
    spatial_field_significance_from_null,
)


def weekly_index(n: int = 260) -> pd.DatetimeIndex:
    return pd.date_range("2000-01-02", periods=n, freq="W-SUN")


def phase_table(index: pd.DatetimeIndex) -> pd.DataFrame:
    phases = np.resize(np.array(PHASE_ORDER, dtype=object), len(index))
    event_type = np.where(np.arange(len(index)) % 2 == 0, "el_nino", "la_nina")
    return pd.DataFrame(
        {
            "fase": phases,
            "tipo": event_type,
            "event_id": [f"event_{i // 20:02d}" for i in range(len(index))],
        },
        index=index,
    )


def test_segmented_ar1_optimized_rows_match_explicit_event_oracle() -> None:
    rng = np.random.default_rng(20260713)
    n_rows, n_pixels = 37, 9
    left = rng.normal(size=n_rows)
    right = rng.normal(size=(n_rows, n_pixels))
    right[rng.random(right.shape) < 0.12] = np.nan
    segments = np.array(
        ["event_a"] * 8
        + [""] * 3
        + ["event_b"] * 11
        + ["event_c"] * 15,
        dtype=str,
    )
    valid = np.isfinite(left)[:, None] & np.isfinite(right)

    observed_r, observed_n = _pooled_within_segment_corr(
        left, right, valid, segments
    )

    numerator = np.zeros(n_pixels)
    left_ss = np.zeros(n_pixels)
    right_ss = np.zeros(n_pixels)
    expected_n = np.zeros(n_pixels, dtype=int)
    for segment in ("event_a", "event_b", "event_c"):
        rows = segments == segment
        for pixel in range(n_pixels):
            usable = rows & valid[:, pixel]
            if usable.sum() < 2:
                continue
            x = left[usable]
            y = right[usable, pixel]
            x_centered = x - x.mean()
            y_centered = y - y.mean()
            numerator[pixel] += np.sum(x_centered * y_centered)
            left_ss[pixel] += np.sum(x_centered**2)
            right_ss[pixel] += np.sum(y_centered**2)
            expected_n[pixel] += len(x)
    expected_r = numerator / np.sqrt(left_ss * right_ss)

    np.testing.assert_array_equal(observed_n, expected_n)
    np.testing.assert_allclose(observed_r, expected_r, rtol=1e-12, atol=1e-12)


def test_sparse_source_row_selection_matches_full_field_reference() -> None:
    rng = np.random.default_rng(260713)
    n_rows, n_pixels = 48, 7
    dates = pd.date_range("2001-01-07", periods=n_rows, freq="W-SUN").to_numpy().copy()
    dates[25:] += np.timedelta64(7, "D")  # one real discontinuity
    index = pd.DatetimeIndex(dates)
    x = rng.normal(size=n_rows)
    x[[0, 2, 3, 8, 9, 14, 21, 22, 31, 37, 38, 44]] = np.nan
    y = rng.normal(size=(n_rows, n_pixels))
    y[rng.random(y.shape) < 0.09] = np.nan
    segments = np.array(
        ["event_a"] * 12
        + [""] * 4
        + ["event_b"] * 15
        + ["event_c"] * 17,
        dtype=str,
    )

    observed = pearson_columns_with_segmented_neff(
        x, y, index, source_event_id=segments, min_pairs=4, min_ar1_pairs=2
    )

    valid = np.isfinite(x)[:, None] & np.isfinite(y)
    expected_r, expected_n = _columnwise_corr(x, y, valid)
    consecutive = (
        np.diff(index.values).astype("timedelta64[D]")
        == np.timedelta64(7, "D")
    )
    consecutive &= (segments[:-1] != "") & (segments[:-1] == segments[1:])
    ar_valid = (
        consecutive[:, None]
        & valid[:-1]
        & valid[1:]
        & np.isfinite(x[:-1])[:, None]
        & np.isfinite(x[1:])[:, None]
    )
    expected_rho_x, expected_n_ar = _pooled_within_segment_corr(
        x[:-1], x[1:], ar_valid, segments[1:]
    )
    expected_rho_y, _ = _pooled_within_segment_corr(
        y[:-1], y[1:], ar_valid, segments[1:]
    )

    np.testing.assert_allclose(observed["r"], expected_r, equal_nan=True)
    np.testing.assert_array_equal(observed["n_pairs"], expected_n)
    np.testing.assert_allclose(
        observed["rho1_predictor"], expected_rho_x, equal_nan=True
    )
    np.testing.assert_allclose(
        observed["rho1_response"], expected_rho_y, equal_nan=True
    )
    np.testing.assert_array_equal(observed["n_ar1_pairs"], expected_n_ar)


def test_conditions_cover_el_nino_and_la_nina_four_phases_at_source() -> None:
    index = weekly_index()
    conditions = build_source_conditions(phase_table(index))
    expected = {
        f"{event_type}_{phase}"
        for event_type in ("el_nino", "la_nina")
        for phase in PHASE_ORDER
    }
    assert expected.issubset(conditions)
    for name in expected:
        condition = conditions[name]
        assert "t-lag" in condition.description


def test_lag_is_predictor_at_t_minus_lag_versus_response_at_t() -> None:
    index = weekly_index(400)
    rng = np.random.default_rng(7)
    predictor = pd.Series(rng.normal(size=len(index)), index=index)
    response = pd.DataFrame({"pixel_1": predictor.shift(4)}, index=index)
    phases = pd.DataFrame(
        {"fase": "neutro", "tipo": "neutro", "event_id": ""}, index=index
    )
    condition = SourceCondition(
        name="todas",
        source_mask=pd.Series(True, index=index),
        tipo_enso_fonte="todos",
        fase_fonte_em_t_menos_lag="todas",
        require_same_event_for_ar1=False,
        description="source at t-lag",
    )
    result = lagged_correlation_exact(
        predictor,
        response,
        [0, 2, 4, 6],
        condition,
        phases,
    )
    best = best_lag_fields(result)
    assert best["best_lag_sem"].tolist() == [4.0]
    assert best["r_no_best_lag"][0] > 0.99


def test_fdr_table_records_q_values_and_the_exact_test_family() -> None:
    index = weekly_index(400)
    rng = np.random.default_rng(11)
    predictor = pd.Series(rng.normal(size=len(index)), index=index)
    response = pd.DataFrame(
        {"p0": predictor.shift(2), "p1": rng.normal(size=len(index))}, index=index
    )
    phases = pd.DataFrame(
        {"fase": "neutro", "tipo": "neutro", "event_id": ""}, index=index
    )
    condition = SourceCondition(
        name="todas",
        source_mask=pd.Series(True, index=index),
        tipo_enso_fonte="todos",
        fase_fonte_em_t_menos_lag="todas",
        require_same_event_for_ar1=False,
        description="all source weeks",
    )
    result = lagged_correlation_exact(
        predictor, response, [0, 2, 4], condition, phases
    )
    table = result_to_long_table(
        result,
        predictor_name="nino34_ssta",
        condition=condition,
        column_name="pixel_id",
    )
    assert table["fdr_family_id"].nunique() == 1
    assert table["fdr_family_n_tests"].nunique() == 1
    assert int(table["fdr_family_n_tests"].iloc[0]) == int(table["p"].notna().sum())
    assert set(table["fdr_alpha"]) == {0.05}
    assert "fdr_bh_reject" in table
    np.testing.assert_allclose(
        table["q_fdr_bh"].to_numpy(),
        fdr_bh_adjusted(np.asarray(result["p"])).ravel(),
        equal_nan=True,
    )


def test_field_significance_uses_whole_map_null_and_max_lag_correction() -> None:
    observed = np.full((2, 100), 0.5)
    observed[0, :80] = 0.01
    null = np.full((99, 2, 100), 0.5)
    for permutation in range(99):
        null[permutation, :, : 5 + permutation % 5] = 0.01
    table = spatial_field_significance_from_null(observed, null, [0, 4])
    first = table.set_index("lag_sem").loc[0]
    assert first["fracao_campo_ponto_significativo"] == 0.8
    assert first["p_field_max_lag"] == 0.01
    assert bool(first["field_significant_confirmatory_max_lag"])
    assert first["point_alpha"] == 0.05


def test_native_pixel_atlas_appends_predictors_and_keeps_condition_contract(tmp_path) -> None:
    index = weekly_index(400)
    rng = np.random.default_rng(19)
    predictors = pd.DataFrame(
        {
            "v1": rng.normal(size=len(index)),
            "v2": rng.normal(size=len(index)),
        },
        index=index,
    )
    response = pd.DataFrame(
        {
            str(pixel): predictors["v1"].shift(2).to_numpy()
            + rng.normal(0, 0.2, len(index))
            for pixel in range(3)
        },
        index=index,
    )
    phases = pd.DataFrame(
        {"fase": "neutro", "tipo": "neutro", "event_id": ""}, index=index
    )
    conditions = {
        name: SourceCondition(
            name=name,
            source_mask=pd.Series(True, index=index),
            tipo_enso_fonte="todos",
            fase_fonte_em_t_menos_lag=name,
            require_same_event_for_ar1=False,
            description="test source at t-lag",
        )
        for name in ("condition_a", "condition_b")
    }
    pixels = pd.DataFrame(
        {
            "pixel_id": [0, 1, 2],
            "lat": [-10.0, -9.75, -9.5],
            "lon": [-45.0, -44.75, -44.5],
            "grid_hash": "synthetic-grid",
            "brazil_fraction": 1.0,
            "brazil_center": True,
        }
    )
    destination = tmp_path / "pixel_atlas.zarr"
    best = compute_pixel_atlas(
        predictors,
        response,
        pixels,
        phases,
        conditions,
        destination=destination,
        replace_existing=False,
        grid_hash="synthetic-grid",
        run_id="F4C_TEST",
        contract_metadata={
            "target_build_id": "F4TARGET_TEST",
            "target_block_signature_sha256": "a" * 64,
            "target_contract_version": "chirps-native-weekly-v4",
            "parent_f3_run_id": "F3_TEST",
        },
    )
    atlas = xr.open_zarr(destination, consolidated=False)
    assert atlas.attrs["analysis_run_id"] == "F4C_TEST"
    assert atlas.attrs["target_build_id"] == "F4TARGET_TEST"
    assert atlas.attrs["target_block_signature_sha256"] == "a" * 64
    assert atlas.attrs["parent_f3_run_id"] == "F3_TEST"
    assert atlas.sizes["variavel"] == 2
    assert atlas.sizes["condicao_fonte"] == 2
    assert atlas.sizes["pixel"] == 3
    assert set(atlas["variavel"].values.tolist()) == {"v1", "v2"}
    assert len(best) == 2 * 2 * 3
    assert best["lag_rule"].str.contains("t-lag", regex=False).all()
    assert "fdr_bh_reject" in atlas
    assert float(atlas.attrs["fdr_alpha"]) == 0.05


def test_failed_replacement_preserves_last_valid_pixel_atlas(tmp_path, monkeypatch) -> None:
    index = weekly_index(60)
    predictors = pd.DataFrame({"v1": np.arange(len(index), dtype=float)}, index=index)
    response = pd.DataFrame({"0": np.arange(len(index), dtype=float)}, index=index)
    phases = pd.DataFrame(
        {"fase": "pico", "tipo": "el_nino", "event_id": "event_1"}, index=index
    )
    condition = SourceCondition(
        name="el_nino_pico",
        source_mask=pd.Series(True, index=index),
        tipo_enso_fonte="el_nino",
        fase_fonte_em_t_menos_lag="pico",
        require_same_event_for_ar1=True,
        description="synthetic source phase",
    )
    pixels = pd.DataFrame(
        {
            "pixel_id": [0],
            "lat": [-10.0],
            "lon": [-45.0],
            "grid_hash": ["synthetic-grid"],
            "brazil_fraction": [1.0],
            "brazil_center": [True],
        }
    )
    destination = tmp_path / "pixel_atlas.zarr"
    destination.mkdir()
    marker = destination / "last_valid_atlas.marker"
    marker.write_text("preserve", encoding="utf-8")

    def fail_during_replacement(*args, **kwargs):
        raise RuntimeError("synthetic calculation failure")

    monkeypatch.setattr(phase4c, "lagged_correlation_exact", fail_during_replacement)
    with pytest.raises(RuntimeError, match="synthetic calculation failure"):
        phase4c.compute_pixel_atlas(
            predictors,
            response,
            pixels,
            phases,
            {condition.name: condition},
            destination=destination,
            replace_existing=True,
            grid_hash="synthetic-grid",
            run_id="F4C_TEST_REPLACE",
        )
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_confirmatory_fdr_alpha_reads_config_and_has_explicit_legacy_fallback() -> None:
    configured = {"modeling": {"phase_3_diagnostics": {"fdr_alpha": 0.05}}}
    assert confirmatory_fdr_alpha(configured) == 0.05
    with pytest.warns(RuntimeWarning, match="explicit compatibility fallback"):
        assert confirmatory_fdr_alpha({}, fallback=0.05) == 0.05
    with pytest.raises(ValueError, match="must lie"):
        confirmatory_fdr_alpha(
            {"modeling": {"phase_3_diagnostics": {"fdr_alpha": 1.5}}}
        )


def test_phase4d_gate_reports_every_region_phase_when_no_lag_passes_fdr() -> None:
    index = weekly_index(260)
    phase_frame = pd.DataFrame(
        {"fase": "pico", "tipo": "el_nino", "event_id": "event_1"}, index=index
    )
    conditions = {}
    for event_type in ("el_nino", "la_nina"):
        for phase in PHASE_ORDER:
            name = f"{event_type}_{phase}"
            conditions[name] = SourceCondition(
                name=name,
                source_mask=pd.Series(True, index=index),
                tipo_enso_fonte=event_type,
                fase_fonte_em_t_menos_lag=phase,
                require_same_event_for_ar1=True,
                description="synthetic source phase",
            )
    region_names = ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]
    units = pd.DataFrame(
        {
            "id_unidade": [f"region_{i}" for i in range(5)],
            "tipo_unidade": "regiao",
            "nome_unidade": region_names,
        }
    )
    unit_lags = pd.DataFrame(
        [
            {
                "variavel": "nino34_ssta",
                "condicao_fonte": condition,
                "id_unidade": unit,
                "lag_sem": 4,
                "r": 0.1,
                "p": 0.8,
                "fdr_bh_reject": False,
            }
            for unit in units["id_unidade"]
            for condition in conditions
        ]
    )
    field = pd.DataFrame(
        columns=["variavel", "condicao_fonte", "lag_sem", "p_field_max_lag"]
    )
    target_series = {
        name: pd.DataFrame(index=index, columns=units["id_unidade"], dtype=float)
        for name in CONFIRMATORY_TARGETS
    }
    gate, summary = build_gate(
        target_series,
        unit_lags,
        field,
        units,
        pd.Series(np.zeros(len(index)), index=index),
        phase_frame,
        conditions,
    )
    assert len(gate) == 5 * 8 * len(CONFIRMATORY_TARGETS)
    assert gate["primary_F4C_status"].eq("no_lag_passed_F4C_BH").all()
    assert not gate["gate_supports_direction"].any()
    assert set(gate["fdr_alpha_confirmatory"]) == {0.05}
    assert "fdr_F4D_confirmatory" in gate
    assert gate["target_family"].nunique() == 4
    assert set(
        gate.loc[
            gate["target_chirps"].isin(
                ["r95p_weekly_robust_z", "r99p_weekly_robust_z"]
            ),
            "target_family",
        ]
    ) == {"heavy_rainfall_extremes"}
    assert "n_target_families_support" in summary
    assert summary["hypothesis_gate"].eq("no_target_specific_support").all()
    assert summary["aggregation_policy"].str.contains("nao se exige").all()
    assert len(summary) == 2 * 2 * 4


def test_peak_region_summary_is_computed_after_native_pixel_lags() -> None:
    key = pd.DataFrame(
        {
            "pixel_id": ["p1", "p2", "p3", "p1", "p2", "p3"],
            "condicao_fonte": ["el_nino_pico"] * 3 + ["el_nino_genese"] * 3,
            "best_lag_sem_fdr": [2.0, 6.0, np.nan, 4.0, 8.0, 10.0],
            "r_no_best_lag_fdr": [-0.3, -0.5, np.nan, -0.1, -0.2, -0.3],
        }
    )
    membership = pd.DataFrame(
        {
            "pixel_id": ["p1", "p2", "p3"],
            "tipo_unidade": ["regiao"] * 3,
            "id_unidade": ["regiao_sul"] * 3,
            "nome_unidade": ["Sul"] * 3,
            "peso_agregacao": [1.0, 3.0, 2.0],
        }
    )
    result = phase4c.summarize_peak_pixel_lags_by_region(key, membership)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["fase_fonte_em_t_menos_lag"] == "pico"
    assert row["n_pixels_originais"] == 3
    assert row["n_pixels_com_lag_fdr"] == 2
    assert row["fracao_area_com_lag_fdr"] == pytest.approx(4 / 6)
    assert row["lag_medio_semanas_pixel_fdr"] == pytest.approx(5.0)


def test_expected_direction_uses_signed_nino34_ssta_for_both_event_types() -> None:
    for event_type in ("el_nino", "la_nina"):
        assert phase4d._expected_direction(event_type, "Nordeste", 1) == -1
        assert phase4d._expected_direction(event_type, "Sul", 1) == 1
        assert phase4d._expected_direction(event_type, "Nordeste", -1) == 1
        assert phase4d._expected_direction(event_type, "Sul", -1) == -1
    with pytest.raises(ValueError, match="Unsupported ENSO"):
        phase4d._expected_direction("neutral", "Nordeste", 1)
