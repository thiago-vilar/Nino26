from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nino_brasil.models.event_validation import (
    ENSO_STATES,
    assert_continuous_weekly_index,
    assert_fold_independence,
    augment_training_rows,
    balanced_active_event_test_start,
    canonical_event_ids,
    canonical_state_labels,
    condition_mask,
    make_event_folds,
    event_phase_sample_weights,
    independent_fold_support,
)
from nino_brasil.models.phase5_cycle_ml import (
    FoldStateEncoder,
    FoldHarmonicPreprocessor,
    build_rolling_origin_table,
    configure_serial_prediction,
    paired_state_permutation_deltas,
    physical_predictor_columns,
    valid_phase_target_mask,
)
from nino_brasil.models.phase6_brazil_ml import validate_native_pixel_contract
from nino_brasil.models.phase6_brazil_ml import (
    causal_persistence_target,
    fit_fold_target_transformer,
    fold_target_transform,
)
from nino_brasil.models.phase7_convlstm import (
    PacificCube,
    augment_sequence_batch,
    build_convlstm_classifier,
    build_sequence_event_targets,
    event_level_target_table,
    gaussian_event_equal_audit,
    make_sequence_dataset,
    make_sequences,
    sequence_source_phase_table,
)
from scripts.run_fase7_cycle_convlstm import (
    _f5_gate_passes,
    _f7_event_dimension_gate,
    _paired_f5_comparison,
)
from scripts.run_fase5_cycle_ml import _classification_gate_pass
from nino_brasil.models.phase8_convlstm_brazil import (
    align_pacific_to_brazil,
    build_convlstm_encoder_decoder,
    masked_area_weighted_gaussian_nll,
    masked_area_weighted_quantile_loss,
    native_chirps_grid,
)


def phase_table(n_quarters: int = 16) -> pd.DataFrame:
    index = pd.date_range("2000-01-02", periods=n_quarters * 13, freq="W-SUN")
    table = pd.DataFrame(
        {"tipo": "neutro", "fase": "neutro", "event_id": ""}, index=index
    )
    phases = ("genese", "crescimento", "pico", "decaimento")
    for event_no in range(1, n_quarters, 2):
        start = event_no * 13
        event_type = "el_nino" if event_no % 4 == 1 else "la_nina"
        for offset in range(12):
            if start + offset >= len(table):
                break
            table.iloc[start + offset] = (
                event_type,
                phases[min(offset // 3, 3)],
                f"{event_type}_{event_no:02d}",
            )
    return table


def test_nine_state_contract_and_multitoken_condition_parser() -> None:
    table = phase_table()
    labels = canonical_state_labels(table)
    assert set(labels).issubset(ENSO_STATES)
    mask = condition_mask(table, "el_nino_pico")
    assert mask.any()
    assert table.loc[mask, "tipo"].eq("el_nino").all()
    assert table.loc[mask, "fase"].eq("pico").all()
    with pytest.raises(ValueError):
        condition_mask(table, "el_pico")


def test_neutral_rows_remain_valid_and_fold_encoder_expands_missing_classes() -> None:
    table = phase_table(8)
    table.loc[table["tipo"].eq("neutro"), "event_id"] = np.nan
    assert valid_phase_target_mask(table).all()
    labels = canonical_state_labels(table)
    assert labels.eq("neutro").any()
    encoder = FoldStateEncoder.fit(["neutro", "el_nino_genese"])
    local = encoder.transform(["neutro", "el_nino_genese"])
    expanded = encoder.expand_probabilities(
        np.asarray([[0.8, 0.2], [0.1, 0.9]]), np.asarray([0, 1])
    )
    assert local.tolist() == [0, 1]
    assert expanded.shape == (2, 9)
    assert np.allclose(expanded.sum(axis=1), 1.0)
    float32_expanded = encoder.expand_probabilities(
        np.asarray([[0.8000001, 0.1999998]], dtype=np.float32),
        np.asarray([0, 1]),
    )
    np.testing.assert_allclose(
        float32_expanded.sum(axis=1), 1.0, rtol=0.0, atol=1e-15
    )


def test_state_permutation_reuses_one_prediction_across_all_nine_states() -> None:
    class CountingEstimator:
        classes_ = np.arange(len(ENSO_STATES), dtype=int)

        def __init__(self) -> None:
            self.calls = 0

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            self.calls += 1
            logits = np.column_stack(
                [
                    (state_id + 1) * values[:, state_id % values.shape[1]]
                    for state_id in range(len(ENSO_STATES))
                ]
            )
            logits -= logits.max(axis=1, keepdims=True)
            probability = np.exp(logits)
            return probability / probability.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(713)
    reference = rng.normal(size=(27, 4))
    truth_ids = np.resize(np.arange(len(ENSO_STATES), dtype=int), len(reference))
    estimator = CountingEstimator()
    base = estimator.predict_proba(reference)
    estimator.calls = 0
    encoder = FoldStateEncoder.fit(ENSO_STATES)
    counts, deltas = paired_state_permutation_deltas(
        estimator,
        encoder,
        reference,
        base,
        truth_ids,
        {
            "predictor_a": np.asarray([0, 1]),
            "predictor_b": np.asarray([2, 3]),
            "unavailable": np.asarray([], dtype=int),
        },
        ("predictor_a", "predictor_b", "unavailable"),
        repeats=3,
        rng=np.random.default_rng(42),
    )

    assert estimator.calls == 2 * 3
    assert counts.sum() == len(reference)
    assert deltas["predictor_a"].shape == (3, len(ENSO_STATES))
    assert deltas["predictor_b"].shape == (3, len(ENSO_STATES))
    assert deltas["unavailable"].shape == (0, len(ENSO_STATES))
    assert np.isfinite(deltas["predictor_a"]).all()


def test_serial_prediction_configuration_does_not_change_fitted_rf() -> None:
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(2613)
    X = rng.normal(size=(80, 5))
    y = np.resize(np.arange(4, dtype=int), len(X))
    estimator = RandomForestClassifier(
        n_estimators=25, random_state=42, n_jobs=-1
    ).fit(X, y)
    expected = estimator.predict_proba(X[:12])

    returned = configure_serial_prediction(estimator)
    observed = estimator.predict_proba(X[:12])

    assert returned is estimator
    assert estimator.get_params(deep=False)["n_jobs"] == 1
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)


def test_future_events_cannot_change_train_only_weights() -> None:
    table = phase_table(12)
    train = table.iloc[:78]
    before = event_phase_sample_weights(train)
    future = table.iloc[78:].copy()
    future["tipo"] = "el_nino"
    future["fase"] = "pico"
    future["event_id"] = "future_event"
    combined = pd.concat([train, future])
    after = event_phase_sample_weights(combined.iloc[: len(train)])
    pd.testing.assert_series_equal(before, after)


def test_folds_keep_events_whole_and_apply_embargo() -> None:
    table = phase_table(24)
    groups = canonical_event_ids(table)
    folds = make_event_folds(
        table.index, groups, n_splits=4, min_train_groups=6, purge_weeks=4
    )
    assert_fold_independence(folds)
    for fold in folds:
        assert fold.train_end < fold.test_start - pd.Timedelta(weeks=4)
        train_group_values = set(groups.iloc[fold.train_index])
        test_group_values = set(groups.iloc[fold.test_index])
        assert train_group_values.isdisjoint(test_group_values)


def test_independent_support_does_not_count_neutral_quarters_as_enso_events() -> None:
    table = phase_table(12)
    active_ids = table.loc[table["tipo"].isin(("el_nino", "la_nina")), "event_id"].unique()
    train_ids = set(active_ids[:2])
    train = table.loc[
        table["event_id"].isin(train_ids) | table["tipo"].eq("neutro")
    ].iloc[:100]
    test = table.iloc[100:]
    support = independent_fold_support(
        train,
        test,
        fold="event_fold_01",
        min_train_active_events_per_type=2,
    )
    assert support["n_train_neutral_blocks"] > 2
    assert support["n_train_el_nino_events"] == 1
    assert support["n_train_la_nina_events"] == 1
    assert support["n_train_active_events"] == 2
    assert support["independent_support_gate_pass"] is False
    assert not _classification_gate_pass(
        0.2, False, support_required_for_gate=True
    )
    assert _classification_gate_pass(
        0.2, False, support_required_for_gate=False
    )


def test_official_fold_start_waits_for_balanced_active_event_support() -> None:
    table = phase_table(28)
    purge_weeks = 4
    earliest = balanced_active_event_test_start(
        table,
        min_train_active_events_per_type=3,
        purge_weeks=purge_weeks,
    )
    assert earliest is not None
    groups = canonical_event_ids(table)
    folds = make_event_folds(
        table.index,
        groups,
        n_splits=4,
        min_train_groups=6,
        purge_weeks=purge_weeks,
        earliest_test_start=earliest,
    )
    assert folds
    for fold in folds:
        support = independent_fold_support(
            table.iloc[fold.train_index],
            table.iloc[fold.test_index],
            fold=fold.name,
            min_train_active_events_per_type=3,
        )
        assert support["n_train_el_nino_events"] >= 3
        assert support["n_train_la_nina_events"] >= 3
        assert support["independent_support_gate_pass"] is True
        assert fold.test_start >= earliest


def test_balanced_support_boundary_fails_when_one_enso_type_is_too_rare() -> None:
    table = phase_table(12)
    table.loc[table["tipo"].eq("la_nina"), ["tipo", "fase", "event_id"]] = (
        "neutro",
        "neutro",
        "",
    )
    with pytest.raises(ValueError, match="la_nina"):
        balanced_active_event_test_start(
            table,
            min_train_active_events_per_type=2,
            purge_weeks=4,
        )


def test_augmentation_keeps_lineage_and_does_not_claim_new_events() -> None:
    table = phase_table(8).iloc[:70]
    X = pd.DataFrame(
        np.arange(len(table) * 3, dtype=float).reshape(len(table), 3),
        index=table.index,
        columns=["a", "b", "c"],
    )
    y = canonical_state_labels(table)
    augmented = augment_training_rows(
        X,
        y,
        table,
        n_noise_copies=1,
        mixup_alpha=0.4,
        random_state=7,
    )
    assert len(augmented.X) > len(X)
    assert augmented.provenance["original_event_id"].ne("").all()
    synthetic = augmented.provenance["augmentation_id"].ne("original")
    assert (~augmented.provenance.loc[synthetic, "independent_event"]).all()
    assert np.isclose(augmented.sample_weight.mean(), 1.0, atol=0.2)
    assert {"state", "event_type", "phase"}.issubset(augmented.provenance.columns)
    mixup = augmented.provenance["augmentation_method"].eq(
        "same_type_phase_mixup_train_only"
    )
    assert mixup.any()
    assert augmented.provenance.loc[mixup, "mixup_parent_event_id"].ne("").all()
    assert augmented.provenance.loc[mixup, "mixup_parent_sample_time"].ne("").all()
    assert (
        augmented.provenance.loc[mixup, "state"]
        == augmented.provenance.loc[mixup, "mixup_parent_state"]
    ).all()


def test_sequence_augmentation_has_explicit_neutral_lineage_and_times() -> None:
    sequences = np.arange(3 * 2 * 2 * 2, dtype="float32").reshape(3, 2, 2, 2, 1)
    sample_times = pd.date_range("2000-02-06", periods=3, freq="W-SUN")
    origin_times = sample_times - pd.Timedelta(weeks=1)
    augmented = augment_sequence_batch(
        sequences,
        ["neutral_2000Q1", "el_nino_2000", "la_nina_2001"],
        sample_times=sample_times,
        origin_times=origin_times,
        states=["neutro", "el_nino_genese", "la_nina_pico"],
        random_state=7,
    )
    assert len(augmented.provenance) == len(sequences)
    assert augmented.provenance["original_event_id"].ne("").all()
    assert augmented.provenance.loc[0, "original_event_id"] == "neutral_2000Q1"
    assert augmented.provenance["sample_time"].ne("").all()
    assert augmented.provenance["origin_time"].ne("").all()
    assert augmented.provenance["state"].tolist() == [
        "neutro",
        "el_nino_genese",
        "la_nina_pico",
    ]
    with pytest.raises(ValueError, match="neutral_YYYYQn"):
        augment_sequence_batch(
            sequences,
            ["", "el_nino_2000", "la_nina_2001"],
            sample_times=sample_times,
            origin_times=origin_times,
            states=["neutro", "el_nino_genese", "la_nina_pico"],
        )


def test_rolling_origin_uses_all_31_physical_variables() -> None:
    table = phase_table(8)
    master = pd.DataFrame(
        {f"v{i:02d}": np.arange(len(table), dtype=float) + i for i in range(31)},
        index=table.index,
    )
    master["ocean_source_code"] = 1
    predictors = physical_predictor_columns(master)
    assert len(predictors) == 31
    rolling = build_rolling_origin_table(master, table, horizons=(1, 4), ssta_column="v00")
    assert "target_state_h01" in rolling
    assert "target_ssta_h04" in rolling
    assert (rolling["dataset_role"] == "rolling_origin_predictive").all()


def test_fold_preprocessor_is_source_aware_and_rejects_unseen_family() -> None:
    index = pd.date_range("1981-01-04", periods=52 * 14, freq="W-SUN")
    annual = np.sin(2 * np.pi * index.dayofyear.to_numpy() / 365.2425)
    source = pd.Series(np.where(index.year <= 1992, 1, 2), index=index)
    frame = pd.DataFrame(
        {"d20_m": 100 + 10 * annual + np.where(source.eq(2), 40, 0)}, index=index
    )
    train = frame.loc[:"1992-12-31"]
    transformer = FoldHarmonicPreprocessor(("d20_m",)).fit(
        train, source_codes=source.loc[train.index]
    )
    transformed = transformer.transform(frame, source_codes=source)
    assert transformed.loc[:"1992-12-31", "d20_m"].notna().all()
    assert transformed.loc["1993-01-01":, "d20_m"].isna().all()

    transformer_both = FoldHarmonicPreprocessor(("d20_m",)).fit(
        frame, source_codes=source
    )
    adjusted = transformer_both.transform(frame, source_codes=source)
    assert abs(adjusted[source.eq(1)]["d20_m"].mean()) < 1e-6
    assert abs(adjusted[source.eq(2)]["d20_m"].mean()) < 1e-6


def test_native_pixel_contract_rejects_regridded_or_missing_pixels() -> None:
    time = pd.date_range("2000-01-02", periods=4, freq="W-SUN")
    response = pd.DataFrame({"p0": 0.0, "p1": 1.0}, index=time)
    pixels = pd.DataFrame(
        {"pixel_id": ["p0", "p1"], "lat": [-10.0, -10.0], "lon": [-40.0, -39.75]}
    )
    contract = validate_native_pixel_contract(response, pixels)
    assert bool(contract.loc[0, "pixel_identity_preserved"])
    assert len(contract.loc[0, "grid_sha256"]) == 64
    with pytest.raises(ValueError):
        validate_native_pixel_contract(response[["p0"]], pixels)


def test_rainfall_target_climatology_is_fit_only_through_fold_end() -> None:
    index = pd.date_range("2000-01-02", periods=52 * 6, freq="W-SUN")
    values = (
        np.sin(2 * np.pi * index.isocalendar().week.to_numpy() / 52.0)
        + 0.2 * np.sin(2 * np.pi * np.arange(len(index)) / 17.0)
    )
    response = pd.DataFrame({"p0": values}, index=index)
    fit_end = index[52 * 3 - 1]
    baseline = fold_target_transform(response, fit_end=fit_end, method="train_robust_z")
    shifted = response.copy()
    shifted.loc[index > fit_end, "p0"] += 1000.0
    changed = fold_target_transform(shifted, fit_end=fit_end, method="train_robust_z")
    pd.testing.assert_series_equal(
        baseline.loc[:fit_end, "p0"], changed.loc[:fit_end, "p0"]
    )
    transformer = fit_fold_target_transformer(
        response, fit_end=fit_end, method="train_robust_z"
    )
    restored = transformer.inverse_array(
        baseline.to_numpy(), index=baseline.index, columns=baseline.columns
    )
    finite = np.isfinite(baseline.to_numpy())
    assert finite.any()
    np.testing.assert_allclose(
        restored[finite], response.to_numpy()[finite], rtol=1e-5, atol=1e-6
    )


def test_fold_rainfall_scale_uses_train_only_pooled_l1_lower_bound() -> None:
    index = pd.date_range("1991-01-06", periods=52 * 8, freq="W-SUN")
    rng = np.random.default_rng(20260713)
    values = np.where(
        rng.random(len(index)) < 0.15,
        rng.gamma(shape=1.3, scale=10.0, size=len(index)),
        0.0,
    )
    response = pd.DataFrame({"p0": values}, index=index)
    fit_end = index[52 * 6 - 1]
    transformer = fit_fold_target_transformer(
        response, fit_end=fit_end, method="train_robust_z"
    )

    history = response.loc[:fit_end]
    weeks = history.index.isocalendar().week.to_numpy()
    centres = history.groupby(weeks).median()
    history_centre = np.vstack([centres.loc[week].to_numpy() for week in weeks])
    residual = history.to_numpy(dtype=float) - history_centre
    pooled_l1 = float(np.nanmean(np.abs(residual)) * np.sqrt(np.pi / 2.0))
    finite_scale = transformer.scale["p0"].dropna()

    assert pooled_l1 >= 0.10
    assert float(finite_scale.min()) >= pooled_l1 - 1e-6


def test_fold_tail_scale_is_conditional_positive_audited_and_leakage_safe() -> None:
    index = pd.date_range("1991-01-06", periods=52 * 8, freq="W-SUN")
    fit_end = index[52 * 6 - 1]
    supported = np.zeros(len(index), dtype=float)
    sparse = np.zeros(len(index), dtype=float)
    supported_positions = np.linspace(3, 52 * 6 - 3, 30, dtype=int)
    sparse_positions = np.linspace(7, 52 * 6 - 7, 10, dtype=int)
    supported[supported_positions] = np.linspace(18.0, 48.0, 30)
    sparse[sparse_positions] = np.linspace(12.0, 30.0, 10)
    supported[52 * 6 :] = 120.0
    sparse[52 * 6 :] = 96.0
    response = pd.DataFrame(
        {"supported": supported, "sparse": sparse}, index=index
    )

    transformer = fit_fold_target_transformer(
        response,
        fit_end=fit_end,
        method="train_robust_z",
        target_name="r99p_weekly_mm",
    )
    changed = response.copy()
    changed.loc[index > fit_end, :] = 1_000_000.0
    changed_transformer = fit_fold_target_transformer(
        changed,
        fit_end=fit_end,
        method="train_robust_z",
        target_name="r99p_weekly_mm",
    )

    pd.testing.assert_frame_equal(transformer.centre, changed_transformer.centre)
    pd.testing.assert_frame_equal(transformer.scale, changed_transformer.scale)
    pd.testing.assert_series_equal(
        transformer.positive_week_count,
        changed_transformer.positive_week_count,
    )
    pd.testing.assert_series_equal(
        transformer.pooled_positive_l1_scale,
        changed_transformer.pooled_positive_l1_scale,
    )
    assert transformer.positive_week_count is not None
    assert transformer.tail_fallback_code is not None
    assert transformer.tail_threshold_floor is not None
    assert int(transformer.positive_week_count["supported"]) == 30
    assert int(transformer.positive_week_count["sparse"]) == 10
    assert int(transformer.tail_fallback_code["supported"]) == 0
    assert int(transformer.tail_fallback_code["sparse"]) == 1
    assert float(transformer.tail_threshold_floor["sparse"]) == pytest.approx(12.0)
    assert transformer.scale_source is not None
    assert "pooled_positive_l1" in set(
        transformer.scale_source["supported"].astype(str)
    )
    assert set(transformer.scale_source["sparse"].astype(str)) == {
        "threshold_floor"
    }

    encoded = transformer.transform(response)
    restored = transformer.inverse_array(
        encoded.to_numpy(), index=encoded.index, columns=encoded.columns
    )
    finite = np.isfinite(encoded.to_numpy())
    np.testing.assert_allclose(
        restored[finite], response.to_numpy()[finite], rtol=1e-5, atol=1e-6
    )


def test_persistence_uses_raw_value_at_forecast_origin_before_transform() -> None:
    index = pd.date_range("1991-01-06", periods=52 * 8, freq="W-SUN")
    seasonal = 20.0 + 12.0 * np.sin(
        2 * np.pi * index.isocalendar().week.to_numpy(dtype=float) / 52.0
    )
    response = pd.DataFrame({"p0": seasonal + np.arange(len(index)) * 0.01}, index=index)
    transformer = fit_fold_target_transformer(
        response,
        fit_end=index[52 * 5 - 1],
        method="train_robust_z",
    )
    horizon = 4
    encoded = causal_persistence_target(
        response, transformer, horizon_weeks=horizon
    )
    decoded = transformer.inverse_array(
        encoded.to_numpy(), index=encoded.index, columns=encoded.columns
    )
    expected = response.shift(horizon).to_numpy(dtype=float)
    finite = np.isfinite(encoded.to_numpy())
    assert finite.any()
    np.testing.assert_allclose(decoded[finite], expected[finite], rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match=">= 1"):
        causal_persistence_target(response, transformer, horizon_weeks=0)


def test_pytorch_convlstm_real_forward_and_exact_brazil_shape() -> None:
    torch = pytest.importorskip("torch")
    sequence = torch.randn(2, 4, 5, 6, 3)
    cycle = build_convlstm_classifier((4, 5, 6, 3), filters=(4,), n_classes=9)
    cycle_output = cycle(sequence)
    assert cycle_output["state_logits"].shape == (2, 9)
    assert cycle_output["event_mean"].shape == (2, 3)

    brazil = build_convlstm_encoder_decoder(
        (4, 5, 6, 3), (7, 9), filters=(4,), distribution="gaussian"
    )
    output = brazil(sequence)
    assert output["mean"].shape == (2, 7, 9)
    target = torch.randn(2, 7, 9)
    loss = masked_area_weighted_gaussian_nll(
        output["mean"],
        output["log_scale"],
        target,
        brazil_mask=torch.ones(7, 9),
        area_weight=torch.ones(7, 9),
    )
    assert torch.isfinite(loss)
    quantile_model = build_convlstm_encoder_decoder(
        (4, 5, 6, 3), (7, 9), filters=(4,), distribution="quantile"
    )
    quantile_output = quantile_model(sequence)["quantiles"]
    assert quantile_output.shape == (2, 3, 7, 9)
    assert torch.all(quantile_output[:, 1:] >= quantile_output[:, :-1])
    quantile_loss = masked_area_weighted_quantile_loss(
        quantile_output,
        target,
        (0.1, 0.5, 0.9),
        brazil_mask=torch.ones(7, 9),
        area_weight=torch.ones(7, 9),
        sample_weight=torch.ones(2),
    )
    assert torch.isfinite(quantile_loss)
    with pytest.raises(ValueError, match="quantiles"):
        build_convlstm_encoder_decoder(
            (4, 5, 6, 3),
            (7, 9),
            filters=(4,),
            distribution="quantile",
            quantiles=(0.1, 0.5, 0.5),
        )


def test_sequence_and_phase8_alignment_include_sequence_offset() -> None:
    cube = np.arange(10, dtype="float32")[:, None, None, None]
    labels = np.arange(10)
    sequences, targets = make_sequences(cube, labels, seq_len=3, horizon=2)
    assert targets[0] == 4  # window ends at index 2, target is index 4
    fields = np.arange(10, dtype="float32")[:, None, None]
    _, aligned = align_pacific_to_brazil(
        sequences,
        fields,
        horizon_weeks=2,
        seq_len=3,
    )
    assert aligned[0, 0, 0] == 4
    with pytest.raises(ValueError):
        align_pacific_to_brazil(sequences, fields, horizon_weeks=2)


def test_native_grid_has_stable_original_pixel_ids() -> None:
    first = native_chirps_grid([-10.0, -9.75], [-40.0, -39.75, -39.5])
    second = native_chirps_grid([-10.0, -9.75], [-40.0, -39.75, -39.5])
    assert len(first.pixel_ids) == 6
    assert first.grid_sha256 == second.grid_sha256
    assert first.pixel_ids[0] == 0
    with pytest.raises(ValueError, match="Hash CHIRPS"):
        native_chirps_grid(
            [-10.0, -9.75],
            [-40.0, -39.75, -39.5],
            expected_grid_sha256="not-the-coordinate-hash",
        )
    with pytest.raises(ValueError, match="row-major"):
        native_chirps_grid(
            [-10.0, -9.75],
            [-40.0, -39.75, -39.5],
            pixel_ids=[1, 0, 2, 3, 4, 5],
        )


def test_missing_week_invalidates_positional_lag_and_sequence_contract() -> None:
    index = pd.date_range("2001-01-07", periods=8, freq="W-SUN").delete(3)
    with pytest.raises(ValueError, match="semanas ausentes"):
        assert_continuous_weekly_index(index)
    values = np.zeros((len(index), 2, 2, 1), dtype="float32")
    cube = PacificCube(
        values=values,
        times=index,
        lat=np.asarray([-1.0, 1.0]),
        lon=np.asarray([150.0, 151.0]),
        channel_names=("sst",),
        finite_mask=np.isfinite(values),
        source_path="synthetic",
    )
    with pytest.raises(ValueError, match="semanas ausentes"):
        make_sequence_dataset(cube, None, seq_len=2)


def test_sequence_event_targets_keep_official_targets_constant_per_event() -> None:
    index = pd.date_range("2000-01-02", periods=12, freq="W-SUN")
    values = np.zeros((12, 2, 2, 1), dtype="float32")
    cube = PacificCube(
        values=values,
        times=index,
        lat=np.asarray([-1.0, 1.0]),
        lon=np.asarray([150.0, 151.0]),
        channel_names=("sst",),
        finite_mask=np.isfinite(values),
        source_path="synthetic",
    )
    phase = pd.DataFrame(
        {"tipo": "el_nino", "fase": "crescimento", "event_id": "event_1"},
        index=index,
    )
    dataset = make_sequence_dataset(
        cube,
        canonical_state_labels(phase).to_numpy(),
        seq_len=3,
        horizon=1,
        event_ids=phase["event_id"],
    )
    events = pd.DataFrame(
        {
            "event_id": ["event_1"],
            "onset": [index[0]],
            "pico": [index[8]],
            "fim": [index[-1]],
            "oni_pico_c": [1.7],
        }
    )
    targets = build_sequence_event_targets(dataset, phase, events)
    assert np.isclose(targets.iloc[0]["peak_magnitude_c"], 1.7)
    expected_event_time_to_peak = (index[8] - index[0]).days / 7
    assert targets["event_time_to_peak_weeks"].eq(expected_event_time_to_peak).all()
    assert targets["event_duration_weeks"].nunique() == 1
    assert np.isclose(
        targets.iloc[0]["weeks_origin_to_peak"],
        (index[8] - dataset.origin_times[0]).days / 7,
    )
    assert targets["weeks_origin_to_peak"].nunique() > 1
    independent = event_level_target_table(
        targets,
        ("peak_magnitude_c", "event_time_to_peak_weeks", "event_duration_weeks"),
    )
    assert independent.index.tolist() == ["event_1"]
    inconsistent = targets.copy()
    inconsistent.iloc[-1, inconsistent.columns.get_loc("event_time_to_peak_weeks")] += 1
    with pytest.raises(ValueError, match="variam dentro"):
        event_level_target_table(
            inconsistent,
            ("peak_magnitude_c", "event_time_to_peak_weeks", "event_duration_weeks"),
        )
    phase.loc[index[4]:, ["tipo", "fase", "event_id"]] = (
        "la_nina",
        "pico",
        "event_2",
    )
    source_phase = sequence_source_phase_table(dataset, phase)
    # First sequence ends at index 2 and targets index 3: source remains El Nino.
    assert source_phase.iloc[0]["tipo"] == "el_nino"


def test_gaussian_event_audit_weights_events_not_sequence_rows() -> None:
    by_event, summary = gaussian_event_equal_audit(
        observed=np.asarray([0.0, 0.0, 10.0]),
        predicted_mean=np.asarray([0.0, 3.0, 10.0]),
        predicted_scale=np.ones(3),
        event_ids=np.asarray(["event_a", "event_a", "event_b"]),
        nominal_coverage=0.90,
    )
    assert by_event["event_id"].tolist() == ["event_a", "event_b"]
    assert by_event["n_sequence_rows"].tolist() == [2, 1]
    # Event A has 50% coverage and event B 100%; event-equal coverage is 75%,
    # not the row-weighted 2/3 result.
    assert np.isclose(summary["interval_coverage_event_equal"], 0.75)
    assert np.isclose(summary["interval_calibration_error"], -0.15)
    assert np.isfinite(summary["gaussian_nll_event_equal"])
    assert summary["n_events"] == 2


def test_f7_event_dimension_gate_requires_skill_and_calibrated_intervals() -> None:
    fold_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    for fold, n_events, coverage in (("f1", 2, 0.85), ("f2", 3, 0.93)):
        row: dict[str, object] = {"fold": fold}
        for target in (
            "peak_magnitude_c",
            "event_time_to_peak_weeks",
            "event_duration_weeks",
        ):
            row[f"n_test_events_{target}"] = n_events
            row[f"mae_event_equal_{target}"] = 0.8
            row[f"mae_baseline_event_equal_{target}"] = 1.0
            probability_rows.append(
                {
                    "fold": fold,
                    "target": target,
                    "n_events": n_events,
                    "interval_coverage_event_equal": coverage,
                }
            )
        fold_rows.append(row)
    metrics = pd.DataFrame(fold_rows)
    probability = pd.DataFrame(probability_rows)
    passed = _f7_event_dimension_gate(
        metrics,
        probability,
        minimum_skill=0.0,
        maximum_absolute_calibration_error=0.15,
    )
    assert passed["event_dimension_gate_pass"] is True
    assert passed["interval_calibration_gate_pass"] is True
    assert np.isclose(passed["skill_mae_event_equal_peak_magnitude_c"], 0.2)

    metrics.loc[0, "mae_event_equal_event_duration_weeks"] = 2.0
    probability.loc[probability["target"].eq("peak_magnitude_c"), "interval_coverage_event_equal"] = 0.4
    failed = _f7_event_dimension_gate(
        metrics,
        probability,
        minimum_skill=0.0,
        maximum_absolute_calibration_error=0.15,
    )
    assert failed["event_dimension_gate_pass"] is False
    assert failed["interval_calibration_gate_pass"] is False


def _write_f5_gate_fixture(
    root,
    *,
    run_id: str,
    mode: str = "official",
    status: str = "complete",
    declared_horizons: tuple[int, ...] = (4,),
    metric_horizons: tuple[int, ...] = (4,),
    skills: tuple[float, ...] = (0.1,),
) -> None:
    run = root / run_id
    (run / "tables").mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "phase": 5,
                "mode": mode,
                "status": status,
                "parameters": {"horizons": list(declared_horizons), "model": "rf"},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "experiment_horizon_weeks": metric_horizons,
            "f1_macro_9state": np.repeat(0.4, len(metric_horizons)),
            "skill_f1_vs_best_baseline": skills,
        }
    ).to_csv(run / "tables" / "fold_metrics.csv", index=False)
    pd.DataFrame(
        {
            "origin_time": pd.date_range(
                "2000-01-02", periods=len(metric_horizons), freq="W-SUN"
            ),
            "observed_state": np.repeat("neutro", len(metric_horizons)),
            "predicted_state": np.repeat("neutro", len(metric_horizons)),
            "experiment_horizon_weeks": metric_horizons,
            "is_original_observation": True,
        }
    ).to_csv(run / "tables" / "oos_predictions.csv", index=False)


def test_f7_gate_requires_complete_official_f5_at_exact_horizon(tmp_path) -> None:
    smoke_root = tmp_path / "smoke_manifest"
    _write_f5_gate_fixture(
        smoke_root,
        run_id="F5_20260101T000000Z_smoke",
        mode="smoke",
    )
    assert not _f5_gate_passes(4, root=smoke_root)

    incomplete_root = tmp_path / "incomplete_manifest"
    _write_f5_gate_fixture(
        incomplete_root,
        run_id="F5_20260101T000000Z_incomplete",
        status="running",
    )
    assert not _f5_gate_passes(4, root=incomplete_root)

    wrong_horizon_root = tmp_path / "wrong_horizon"
    _write_f5_gate_fixture(
        wrong_horizon_root,
        run_id="F5_20260101T000000Z_h8",
        declared_horizons=(8,),
        metric_horizons=(8,),
    )
    assert not _f5_gate_passes(4, root=wrong_horizon_root)

    exact_root = tmp_path / "exact_horizon"
    _write_f5_gate_fixture(
        exact_root,
        run_id="F5_20260101T000000Z_h4",
        declared_horizons=(4,),
        metric_horizons=(4, 8),
        skills=(0.2, -0.9),
    )
    _write_f5_gate_fixture(
        exact_root,
        run_id="F5_20260102T000000Z_newer_but_negative",
        declared_horizons=(4,),
        metric_horizons=(4,),
        skills=(-0.1,),
    )
    assert _f5_gate_passes(4, root=exact_root)
    assert not _f5_gate_passes(8, root=exact_root)


def test_f7_paired_f5_comparison_uses_exact_oos_origins(tmp_path) -> None:
    root = tmp_path / "paired"
    _write_f5_gate_fixture(
        root,
        run_id="F5_20260101T000000Z_h4",
        metric_horizons=(4, 4),
        skills=(0.2, 0.2),
    )
    f7 = pd.DataFrame(
        {
            "origin_time": pd.date_range("2000-01-02", periods=2, freq="W-SUN"),
            "observed_state": ["neutro", "neutro"],
            "predicted_state": ["neutro", "el_nino_genese"],
        }
    )
    from scripts.run_fase7_cycle_convlstm import _official_f5_references

    comparison = _paired_f5_comparison(
        f7,
        _official_f5_references(4, root=root),
        horizon_weeks=4,
    )
    assert comparison.loc[0, "n_paired_oos_origins"] == 2
    assert comparison.loc[0, "paired_coverage"] == 1.0
    assert comparison.loc[0, "skill_f1_f7_minus_f5_paired"] < 0.0
    assert not bool(comparison.loc[0, "paired_gate_pass"])
