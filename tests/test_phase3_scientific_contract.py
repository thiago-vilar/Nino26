from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import friedmanchisquare

from nino_brasil.events.enso import (
    ENSO_ACTIVE_PHASES,
    ENSO_STATE_ORDER,
    EnsoLifecycleConfig,
    build_enso_lifecycle,
    build_rolling_origin_targets,
    detect_enso_events,
    peak_band_sensitivity,
)
from nino_brasil.stats.phase3_inference import (
    bootstrap_lag_selection_by_event,
    is_phase3_target_alias,
    phase3_precursor_columns,
    scan_lagged_correlations,
    select_best_lags,
)
from nino_brasil.stats.preprocessing import SeasonalTrendConfig, SeasonalTrendTransformer
from nino_brasil.stats.semantic_tables import (
    SemanticTableContract,
    sha256_file,
    verify_semantic_csv,
    write_semantic_csv,
)
from nino_brasil.stats.validation import (
    event_purged_rolling_origin_folds,
    folds_audit_table,
    required_purge_weeks,
)
from scripts.phase3_en_ln import (
    percent_influence,
    phase_boundary_sensitivity,
    phase_statistics,
    phase_variable_sets,
    transition_guides,
)
from nino_brasil.stats.phase3_inference import confirmed_friedman_discriminants


def _synthetic_oni() -> pd.Series:
    index = pd.date_range("2000-01-01", periods=36, freq="MS")
    values = np.zeros(len(index), dtype=float)
    values[8:13] = [0.6, 0.8, 1.0, 0.88, 0.55]
    values[24:29] = [-0.55, -0.82, -1.1, -0.9, -0.6]
    return pd.Series(values, index=index, name="oni_local_c")


def test_canonical_events_are_symmetric_and_peak_band_has_one_definition() -> None:
    oni = _synthetic_oni()
    events = detect_enso_events(oni)

    assert events["tipo"].tolist() == ["el_nino", "la_nina"]
    assert set(events["fracao_faixa_pico"]) == {0.9}
    assert events["event_id"].is_unique
    assert np.allclose(events["magnitude_pico_c"], [1.0, 1.1])
    assert (events["modo_rotulo"] == "diagnostico_retrospectivo").all()

    sensitivity = peak_band_sensitivity(oni, events, fractions=(0.8, 0.9, 0.95))
    assert sensitivity.groupby("event_id").size().eq(3).all()
    assert sensitivity[sensitivity["configuracao_canonica"]]["fracao_faixa_pico"].eq(0.9).all()
    warm = sensitivity[sensitivity["event_id"].str.startswith("el_nino")].set_index("fracao_faixa_pico")
    assert warm.loc[0.8, "duracao_faixa_pico_meses"] > warm.loc[0.95, "duracao_faixa_pico_meses"]


def test_weekly_lifecycle_has_nine_state_contract_and_declares_future_information() -> None:
    events = detect_enso_events(_synthetic_oni())
    weekly = pd.date_range("2000-01-02", "2002-12-29", freq="W-SUN")
    lifecycle = build_enso_lifecycle(events, weekly)

    assert set(lifecycle["estado_enso"]).issubset(set(ENSO_STATE_ORDER))
    for event_type in ("el_nino", "la_nina"):
        assert set(lifecycle.loc[lifecycle["tipo"].eq(event_type), "fase"]) == {
            "genese",
            "crescimento",
            "pico",
            "decaimento",
        }
    assert not lifecycle["rotulo_disponivel_na_origem"].any()
    assert lifecycle["modo_rotulo"].eq("diagnostico_retrospectivo").all()


def test_phase_boundary_sensitivity_covers_genesis_and_peak_choices() -> None:
    oni = _synthetic_oni()
    events = detect_enso_events(oni)
    weekly = pd.date_range("2000-01-02", "2002-12-29", freq="W-SUN")
    result = phase_boundary_sensitivity(oni, events, weekly)

    assert set(result["janela_genese_semanas"]) == {13, 26, 39}
    assert set(result["fracao_faixa_pico"]) == {0.8, 0.9, 0.95}
    assert result.groupby(["event_id", "janela_genese_semanas", "fracao_faixa_pico"]).size().eq(4).all()
    assert result.loc[result["configuracao_canonica"]].groupby("event_id").size().eq(4).all()
    genesis = result[result["fase"].eq("genese")]
    medians = genesis.groupby("janela_genese_semanas")["duracao_semanas"].median()
    assert medians.loc[13] < medians.loc[26] < medians.loc[39]


def test_transition_guides_measure_adjacent_changes_and_exclude_target() -> None:
    phases = list(ENSO_ACTIVE_PHASES)
    rows: list[dict[str, object]] = []
    values: list[dict[str, float]] = []
    index: list[pd.Timestamp] = []
    week = pd.Timestamp("2000-01-02")
    for event_number in range(8):
        for phase_number, phase in enumerate(phases):
            index.append(week)
            rows.append({"tipo": "el_nino", "fase": phase, "event_id": f"e{event_number}"})
            values.append(
                {
                    "nino34_ssta": float(phase_number),
                    "d20_m": phase_number * (1.0 + 0.03 * event_number),
                    "tau_x_anom": phase_number * (0.7 + 0.02 * event_number),
                    "mslp_anom": -phase_number * (0.5 + 0.01 * event_number),
                }
            )
            week += pd.Timedelta(weeks=1)
    transformed = pd.DataFrame(values, index=index)
    lifecycle = pd.DataFrame(rows, index=index)
    discrimination = pd.DataFrame(
        {
            "tipo": "el_nino",
            "variavel": ["nino34_ssta", "d20_m", "tau_x_anom", "mslp_anom"],
            "kendall_w_entre_fases": [0.95, 0.8, 0.7, 0.6],
            "significativo_friedman_fdr": True,
        }
    )
    guides = transition_guides(transformed, lifecycle, discrimination)

    assert set(guides["transicao_monitorada"]) == {
        "genese_para_crescimento",
        "crescimento_para_pico",
        "pico_para_decaimento",
    }
    assert "nino34_ssta" not in set(guides["variavel"])
    assert guides.groupby("transicao_monitorada")["guia_principal"].sum().ge(1).all()
    assert guides["tipo_guia"].eq("marcador_empirico_retrospectivo_da_transicao").all()


def test_phase_variable_sets_exclude_target_and_limit_redundant_family() -> None:
    variables = ["nino34_ssta", "d20_m", "wwv", "t50m", "ssh_m", "tau_x_anom", "mslp_anom"]
    phase_stats = pd.DataFrame(
        [
            {
                "tipo": "el_nino",
                "fase": phase,
                "variavel": variable,
                "media_z_entre_eventos": 2.0 - 0.1 * number,
                "desvio_z_entre_eventos": 0.5,
                "n_eventos_independentes": 8,
            }
            for phase in ENSO_ACTIVE_PHASES
            for number, variable in enumerate(variables)
        ]
    )
    discrimination = pd.DataFrame(
        {
            "tipo": "el_nino",
            "variavel": variables,
            "kendall_w_entre_fases": np.linspace(0.95, 0.65, len(variables)),
            "significativo_friedman_fdr": True,
        }
    )
    result = phase_variable_sets(phase_stats, discrimination)
    selected = result[result["integra_conjunto_descritor"]]

    assert "nino34_ssta" not in set(selected["variavel"])
    assert selected.groupby("fase").size().le(5).all()
    assert selected[selected["familia"].eq("oceano_subsuperficie")].groupby("fase").size().le(2).all()


def test_percent_influence_excludes_target_and_closes_at_one_hundred() -> None:
    variables = ["nino34_ssta", "d20_m", "ssh_m", "tau_x_anom"]
    phase_stats = pd.DataFrame(
        [
            {
                "tipo": "el_nino",
                "fase": phase,
                "variavel": variable,
                "media_z_entre_eventos": float(number + 1),
                "n_eventos_independentes": 8,
            }
            for phase in ENSO_ACTIVE_PHASES
            for number, variable in enumerate(variables)
        ]
    )
    discrimination = pd.DataFrame(
        {
            "tipo": "el_nino",
            "variavel": variables,
            "kendall_w_entre_fases": [0.95, 0.8, 0.7, 0.6],
            "significativo_friedman_fdr": True,
            "n_eventos_completos": 8,
        }
    )

    result = percent_influence(phase_stats, discrimination)

    assert "nino34_ssta" not in set(result["variavel"])
    totals = result.groupby(["fase", "metrica"])["valor_pct"].sum()
    assert np.allclose(totals.to_numpy(), 100.0)
    assert result["base_metrica"].str.contains("alvo excluida").all()


def test_rolling_origin_targets_never_expose_future_label_as_feature() -> None:
    index = pd.date_range("2000-01-02", periods=20, freq="W-SUN")
    signal = pd.Series(np.linspace(-1, 1, len(index)), index=index)
    targets = build_rolling_origin_targets(signal, horizons_weeks=(1, 4))

    assert not targets["uses_future_features"].any()
    assert (targets["information_cutoff"] == targets["origin_time"]).all()
    assert (
        targets["target_time"] - targets["origin_time"]
        == pd.to_timedelta(targets["horizon_weeks"], unit="W")
    ).all()
    assert targets.loc[targets["target_signal_c"].isna(), "target_tipo"].eq("fora_da_amostra").all()
    assert "fase" not in targets.columns


def test_preprocessing_fit_is_unchanged_by_future_values() -> None:
    index = pd.date_range("2000-01-02", periods=220, freq="W-SUN")
    angle = 2 * np.pi * (index.dayofyear.to_numpy() - 1) / 365.2425
    frame = pd.DataFrame(
        {
            "raw_ocean": 3 * np.sin(angle) + np.linspace(0, 2, len(index)),
            "nino34_ssta": np.cos(angle) + np.linspace(0, 0.2, len(index)),
        },
        index=index,
    )
    train = frame.iloc[:160]
    config = SeasonalTrendConfig(harmonics=2, min_observations=104)
    first = SeasonalTrendTransformer(config=config, already_anomalous={"nino34_ssta"}).fit(train)
    first_values = first.transform(train)

    changed = frame.copy()
    changed.iloc[160:, :] += 1_000_000
    second = SeasonalTrendTransformer(config=config, already_anomalous={"nino34_ssta"}).fit(changed.iloc[:160])
    second_values = second.transform(changed.iloc[:160])

    assert np.allclose(first_values, second_values)
    assert first.metadata()["fit_uses_evaluation_rows"].eq(False).all()
    assert first.metadata()["ajuste_fim"].max() == train.index.max()


def _known_lag_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(44)
    index = pd.date_range("2000-01-02", periods=240, freq="W-SUN")
    predictor = pd.Series(rng.normal(size=len(index)), index=index)
    target = predictor.shift(2) + pd.Series(rng.normal(scale=0.05, size=len(index)), index=index)
    lifecycle = pd.DataFrame(
        {"tipo": "el_nino", "fase": "crescimento", "event_id": ""},
        index=index,
    )
    for event_number, start in enumerate((0, 60, 120, 180), start=1):
        lifecycle.iloc[start : start + 40, lifecycle.columns.get_loc("event_id")] = f"event_{event_number}"
    return predictor.to_frame("precursor"), target, lifecycle


def test_precursor_catalogue_excludes_target_aliases_but_keeps_local_physics() -> None:
    columns = [
        "nino34_ssta",
        "nino34_sst_c",
        "nino34_ssta_weekly",
        "nino34_anom_c",
        "oni",
        "oni_local_c",
        "d20_nino34_m",
        "ohc_0_300_nino34_jm2",
        "tau_x_anom_nino34_pa",
        "wwv",
    ]
    candidates, excluded = phase3_precursor_columns(
        columns,
        selected_signal="nino34_ssta",
    )

    assert excluded == tuple(columns[:6])
    assert candidates == tuple(columns[6:])
    assert not any(
        is_phase3_target_alias(name, selected_signal="nino34_ssta")
        for name in candidates
    )


def test_lag_scan_excludes_aliases_and_registers_target_catalogue() -> None:
    predictors, target, lifecycle = _known_lag_data()
    predictors["nino34_ssta"] = target
    predictors["nino34_sst_weekly"] = target
    predictors["oni"] = target
    predictors["oni_local_c"] = target
    predictors["d20_nino34_m"] = predictors["precursor"]
    scan = scan_lagged_correlations(
        predictors,
        target.rename("nino34_ssta"),
        lifecycle,
        target_name="nino34_ssta",
        lags_weeks=range(3),
        event_types=("el_nino",),
        phases=("crescimento",),
        min_pairs=20,
    )

    assert set(scan["variavel"]) == {"precursor", "d20_nino34_m"}
    assert scan["variavel_alvo"].eq("nino34_ssta").all()
    assert scan["n_precursores_candidatos"].eq(2).all()
    expected_excluded = {
        "nino34_ssta",
        "nino34_sst_weekly",
        "oni",
        "oni_local_c",
    }
    assert scan["aliases_alvo_excluidos"].map(
        lambda value: set(str(value).split("|")) == expected_excluded
    ).all()
    assert scan["precursor_screening_policy"].str.contains("oni_local").all()


def test_lag_scan_recovers_known_lag_with_effective_n_fdr_and_field_test() -> None:
    predictors, target, lifecycle = _known_lag_data()
    scan = scan_lagged_correlations(
        predictors,
        target,
        lifecycle,
        lags_weeks=range(6),
        event_types=("el_nino",),
        phases=("crescimento",),
        min_pairs=20,
    )
    best = select_best_lags(scan)

    assert best.iloc[0]["lag_semanas"] == 2
    assert best.iloc[0]["significativo_fdr"]
    assert scan["n_efetivo_ar1_segmentado"].dropna().le(scan["n_pares"]).all()
    assert scan["field_test_method"].str.contains("Simes").all()
    assert scan["family_id"].nunique() == 1


def test_event_bootstrap_repeats_lag_selection_and_keeps_event_as_unit() -> None:
    predictors, target, lifecycle = _known_lag_data()
    result = bootstrap_lag_selection_by_event(
        predictors["precursor"],
        target,
        lifecycle,
        predictor_name="precursor",
        target_name="nino34_ssta",
        n_precursor_candidates=30,
        excluded_target_aliases=("nino34_ssta", "oni_local_c"),
        screening_rank=1,
        screening_top_k=5,
        lags_weeks=range(6),
        event_type="el_nino",
        phase="crescimento",
        n_boot=100,
        min_pairs=20,
        random_state=9,
    )

    top = result.summary.sort_values("frequencia_selecao", ascending=False).iloc[0]
    assert top["lag_semanas"] == 2
    assert top["frequencia_selecao"] > 0.9
    assert result.summary["lag_selection_repeated_inside_bootstrap"].all()
    assert result.summary["resampling_unit"].eq("evento_enso_completo").all()
    assert result.summary["bootstrap_screening_top_k"].eq(5).all()
    assert result.summary["bootstrap_screening_rank"].eq(1).all()
    assert result.summary["bootstrap_screening_rule"].str.contains("scan_original").all()
    assert result.summary["n_precursores_candidatos"].eq(30).all()


def test_event_bootstrap_refuses_target_alias_as_predictor() -> None:
    predictors, target, lifecycle = _known_lag_data()
    with pytest.raises(ValueError, match="aliases"):
        bootstrap_lag_selection_by_event(
            predictors["precursor"].rename("oni_local_c"),
            target.rename("nino34_ssta"),
            lifecycle,
            predictor_name="oni_local_c",
            target_name="nino34_ssta",
            lags_weeks=range(3),
            event_type="el_nino",
            phase="crescimento",
            n_boot=5,
            min_pairs=20,
            random_state=9,
        )


def _phase_statistics_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260713)
    rows: list[dict[str, object]] = []
    values: list[dict[str, float]] = []
    index: list[pd.Timestamp] = []
    week = pd.Timestamp("2000-01-02")
    for event_type in ("el_nino", "la_nina"):
        for event_number in range(6):
            for phase_number, phase in enumerate(ENSO_ACTIVE_PHASES):
                index.append(week)
                rows.append(
                    {
                        "tipo": event_type,
                        "fase": phase,
                        "event_id": f"{event_type}_{event_number}",
                    }
                )
                values.append(
                    {
                        "strong_phase_signal": 10.0 * phase_number + event_number / 10,
                        "noise": float(rng.normal()),
                        "missing_phase": np.nan if phase == "decaimento" else float(phase_number),
                    }
                )
                week += pd.Timedelta(weeks=1)
    return pd.DataFrame(values, index=index), pd.DataFrame(rows, index=index)


def test_phase_statistics_preserves_friedman_and_corrects_each_enso_type() -> None:
    transformed, lifecycle = _phase_statistics_fixture()
    _, discrimination = phase_statistics(transformed, lifecycle, fdr_alpha=0.05)

    expected_columns = {
        "p_friedman",
        "kendall_w_entre_fases",
        "q_friedman_bh",
        "significativo_friedman_fdr",
        "friedman_family_id",
        "friedman_family_size",
        "friedman_valid_p_count",
        "friedman_fdr_alpha",
    }
    assert expected_columns.issubset(discrimination.columns)
    assert discrimination.groupby("tipo")["friedman_family_id"].nunique().eq(1).all()
    assert discrimination["friedman_family_id"].nunique() == 2
    assert discrimination["friedman_family_size"].eq(3).all()
    assert discrimination["friedman_valid_p_count"].eq(2).all()
    assert discrimination.loc[
        discrimination["variavel"].eq("missing_phase"), "q_friedman_bh"
    ].isna().all()

    warm = transformed.join(lifecycle).query("tipo == 'el_nino'")
    pivot = warm.pivot(index="event_id", columns="fase", values="strong_phase_signal")
    pivot = pivot.reindex(columns=ENSO_ACTIVE_PHASES)
    expected = friedmanchisquare(*(pivot[phase].to_numpy() for phase in ENSO_ACTIVE_PHASES))
    observed = discrimination[
        discrimination["tipo"].eq("el_nino")
        & discrimination["variavel"].eq("strong_phase_signal")
    ].iloc[0]
    assert observed["p_friedman"] == pytest.approx(expected.pvalue)
    assert observed["kendall_w_entre_fases"] == pytest.approx(
        expected.statistic / (len(pivot) * (len(ENSO_ACTIVE_PHASES) - 1))
    )
    assert observed["significativo_friedman_fdr"]
    assert observed["q_friedman_bh"] <= observed["friedman_fdr_alpha"]


def test_phase_statistics_can_isolate_one_enso_signal() -> None:
    transformed, lifecycle = _phase_statistics_fixture()
    phase_stats, discrimination = phase_statistics(
        transformed,
        lifecycle,
        fdr_alpha=0.05,
        event_types=("la_nina",),
    )

    assert set(phase_stats["tipo"]) == {"la_nina"}
    assert set(discrimination["tipo"]) == {"la_nina"}
    assert discrimination["friedman_family_id"].str.contains("la_nina").all()


def test_semantic_renderer_fails_closed_and_only_keeps_fdr_confirmed_rows() -> None:
    frame = pd.DataFrame(
        {
            "variavel": ["confirmed", "raw_only", "flag_mismatch"],
            "kendall_w_entre_fases": [0.8, 0.9, 0.7],
            "q_friedman_bh": [0.01, 0.20, 0.01],
            "significativo_friedman_fdr": [True, False, False],
            "friedman_fdr_alpha": [0.05, 0.05, 0.05],
        }
    )
    confirmed = confirmed_friedman_discriminants(frame)
    assert confirmed["variavel"].tolist() == ["confirmed"]
    with pytest.raises(KeyError, match="confirmatory Friedman FDR"):
        confirmed_friedman_discriminants(frame.drop(columns="q_friedman_bh"))


def test_rolling_folds_keep_whole_events_and_apply_purge() -> None:
    index = pd.date_range("2000-01-02", periods=80, freq="W-SUN")
    event_id = np.full(len(index), "", dtype=object)
    for number, start in enumerate((5, 22, 39, 56), start=1):
        event_id[start : start + 6] = f"event_{number}"
    samples = pd.DataFrame(
        {
            "origin_time": index,
            "target_time": index + pd.Timedelta(weeks=1),
            "event_id": event_id,
        }
    )
    purge = required_purge_weeks(
        max_feature_lag_weeks=3,
        target_horizon_weeks=1,
        sequence_length_weeks=2,
    )
    folds = event_purged_rolling_origin_folds(samples, purge_weeks=purge, min_train_events=2)

    assert purge == 4
    for fold in folds:
        train = samples.iloc[fold.train_indices]
        test = samples.iloc[fold.test_indices]
        assert set(test["event_id"]) == {fold.test_event_id}
        assert fold.test_event_id not in set(train["event_id"])
        for train_event in set(train["event_id"]) - {""}:
            expected = samples.index[samples["event_id"].eq(train_event)]
            observed = train.index[train["event_id"].eq(train_event)]
            assert set(expected) == set(observed)
        assert train["origin_time"].max() < fold.test_start - pd.Timedelta(weeks=purge)
    audit = folds_audit_table(folds)
    assert not audit["future_events_in_training"].any()
    assert audit["n_train_events"].ge(2).all()
    assert audit["status"].eq("ok").all()


def test_semantic_table_has_full_hash_schema_and_lineage(tmp_path) -> None:
    table = pd.DataFrame({"tipo": ["el_nino", "la_nina"], "valor": [1.0, -1.0]})
    output = tmp_path / "table.csv"
    contract = SemanticTableContract(
        table_id="phase3_test",
        phase="F3",
        method="synthetic",
        description="unit test",
        evaluation_mode="diagnostico_retrospectivo",
        primary_keys=("tipo",),
        allowed_values={"tipo": ("el_nino", "la_nina")},
        units={"valor": "degC"},
    )
    result = write_semantic_csv(
        table,
        output,
        contract=contract,
        run_id="test_run",
        project_root=tmp_path,
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert len(result.sha256) == 64
    assert result.sha256 == sha256_file(output)
    assert metadata["artifact"]["sha256"] == result.sha256
    assert metadata["run_id"] == "test_run"
    assert metadata["contract"]["primary_keys"] == ["tipo"]
    assert metadata["runtime"]["python"]
    assert "dirty" in metadata["git_worktree"]
    assert verify_semantic_csv(output)["valid"]
    output.write_text("tipo,valor\nel_nino,999\n", encoding="utf-8")
    verification = verify_semantic_csv(output)
    assert not verification["valid"]
    assert not verification["artifact_hash_ok"]
