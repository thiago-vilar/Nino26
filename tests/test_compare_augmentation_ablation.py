from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import compare_augmentation_ablation as ablation


TARGETS = (
    "peak_magnitude_c",
    "event_time_to_peak_weeks",
    "event_duration_weeks",
)


def _lineage(active: bool, *, phase: int) -> pd.DataFrame:
    if active:
        return pd.DataFrame(
            {
                "original_event_id": ["el_nino_2000_2001"],
                "augmentation_id": ["sequence_aug_000001" if phase == 7 else "covnoise_01"],
                "augmentation_method": [
                    "train_only_covscale_noise_channel_time_mask"
                    if phase == 7
                    else "covariance_noise_train_only"
                ],
                "independent_event": [False],
            }
        )
    if phase == 7:
        # This is the exact shape emitted by the F7 runner with
        # --no-augmentation: an intentionally empty, zero-column table.
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "original_event_id": ["el_nino_2000_2001"],
            "augmentation_id": ["original"],
            "augmentation_method": ["none"],
            "independent_event": [True],
        }
    )


def _common_tables(
    active: bool, *, phase: int, fold_variant: str = "standard"
) -> dict[str, pd.DataFrame]:
    if phase == 5:
        contract = pd.DataFrame(
            {
                "model": ["rf"],
                "experiment_horizon_weeks": [4],
                "fold": ["event_fold_01"],
                "train_end": ["1999-01-03"],
                "test_start": ["2000-01-02"],
                "test_end": ["2002-01-06"],
                "purge_weeks": [52],
                "test_groups": [
                    "el_nino_2000_2001;la_nina_2001_2002;neutral_2001Q2"
                    if fold_variant == "standard"
                    else "el_nino_2000_2001;neutral_2001Q2"
                ],
                "augmentation_train_only": [True],
            }
        )
        metrics = pd.DataFrame(
            {
                "model": ["rf"],
                "experiment_horizon_weeks": [4],
                "fold": ["event_fold_01"],
                "f1_macro_9state": [0.60 if active else 0.55],
                "skill_f1_vs_best_baseline": [0.10 if active else 0.05],
                "brier_multiclass": [0.18 if active else 0.20],
                "n_train_rows_original": [100],
                "n_train_rows_optimisation": [200 if active else 100],
            }
        )
        support = pd.DataFrame(
            {
                "experiment_horizon_weeks": [4],
                "fold": ["event_fold_01"],
                "n_test_active_events": [2],
                "n_test_neutral_blocks": [1],
            }
        )
        predictions = pd.DataFrame(
            {
                "model": ["rf", "rf", "rf"],
                "experiment_horizon_weeks": [4, 4, 4],
                "fold": ["event_fold_01"] * 3,
                "origin_time": ["2000-01-02", "2001-01-07", "2001-04-01"],
                "target_time": ["2000-01-30", "2001-02-04", "2001-04-29"],
                "event_id": [
                    "el_nino_2000_2001",
                    "la_nina_2001_2002",
                    "neutral_2001Q2",
                ],
                "observed_state": ["el_nino_genese", "la_nina_genese", "neutro"],
                "predicted_state": ["neutro", "neutro", "neutro"],
                "is_original_observation": [True, True, True],
            }
        )
        return {
            "fold_contract.csv": contract,
            "fold_metrics.csv": metrics,
            "independent_support_by_fold.csv": support,
            "oos_predictions.csv": predictions,
            "augmentation_provenance.csv": _lineage(active, phase=phase),
        }

    contract = pd.DataFrame(
        {
            "fold": ["event_fold_01"],
            "train_end": ["1999-01-03"],
            "test_start": ["2000-01-02"],
            "test_end": ["2002-01-06" if fold_variant == "standard" else "2001-01-07"],
            "purge_weeks": [28],
            "whole_event_split": [True],
            "event_regression_metric_unit": ["event_equal"],
        }
    )
    metric_values: dict[str, list[object]] = {
        "fold": ["event_fold_01"],
        "f1_macro_9state": [0.50 if active else 0.45],
        "skill_f1_vs_best_baseline": [0.08 if active else 0.03],
        "n_train_sequences_original": [100],
        "n_train_sequences_optimisation": [200 if active else 100],
    }
    for index, target in enumerate(TARGETS):
        metric_values[f"n_test_events_{target}"] = [2]
        metric_values[f"mae_event_equal_{target}"] = [0.5 + index + (0.0 if active else 0.1)]
        metric_values[f"mae_baseline_event_equal_{target}"] = [1.0 + index]
        metric_values[f"skill_mae_{target}"] = [0.2 + index * 0.05 if active else 0.1]
    metrics = pd.DataFrame(metric_values)
    support = pd.DataFrame(
        {
            "fold": ["event_fold_01"],
            "n_test_active_events": [2],
            "n_test_neutral_blocks": [1],
        }
    )
    predictions = pd.DataFrame(
        {
            "fold": ["event_fold_01"] * 3,
            "origin_time": ["2000-01-02", "2001-01-07", "2001-04-01"],
            "target_time": ["2000-01-30", "2001-02-04", "2001-04-29"],
            "event_id": ["el_nino_2000_2001", "la_nina_2001_2002", np.nan],
            "observed_state": ["el_nino_genese", "la_nina_genese", "neutro"],
            "observed_peak_magnitude_c": [1.5, 1.2, np.nan],
            "observed_event_time_to_peak_weeks": [20.0, 24.0, np.nan],
            "observed_event_duration_weeks": [40.0, 50.0, np.nan],
            "predicted_state": ["neutro", "neutro", "neutro"],
        }
    )
    return {
        "fold_contract.csv": contract,
        "fold_metrics.csv": metrics,
        "independent_support_by_fold.csv": support,
        "oos_predictions.csv": predictions,
        "augmentation_provenance.csv": _lineage(active, phase=phase),
    }


def _phase5_tables(active: bool, *, fold_variant: str = "standard") -> dict[str, pd.DataFrame]:
    tables = _common_tables(active, phase=5, fold_variant=fold_variant)
    rows: list[dict[str, object]] = []
    for target_index, target in enumerate(("Y_pico", "Y_tempo_para_pico_sem", "Y_duracao_sem")):
        for event_index, (event_id, tipo) in enumerate(
            (("el_nino_2000_2001", "el_nino"), ("la_nina_2001_2002", "la_nina"))
        ):
            observed = float(target_index * 10 + event_index + 2)
            rows.append(
                {
                    "fold": "event_fold_01",
                    "event_id": event_id,
                    "tipo": tipo,
                    "origin_time": "2000-01-02" if event_index == 0 else "2001-01-07",
                    "target": target,
                    "observed": observed,
                    "predicted": observed + (0.1 if active else 0.2),
                    "baseline_global_train": observed - 0.5,
                    "baseline_type_train": observed - 0.4,
                    "preprocessing_fit_end": "1999-01-03",
                    "augmentation_train_only": True,
                }
            )
    tables["event_dimension_oos_predictions.csv"] = pd.DataFrame(rows)
    tables["event_dimension_augmentation_provenance.csv"] = _lineage(active, phase=5)
    return tables


def _phase7_tables(active: bool, *, fold_variant: str = "standard") -> dict[str, pd.DataFrame]:
    tables = _common_tables(active, phase=7, fold_variant=fold_variant)
    calibration_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for target_index, target in enumerate(TARGETS):
        coverage = 0.85 if active else 0.80
        calibration_rows.append(
            {
                "fold": "event_fold_01",
                "target": target,
                "n_events": 2,
                "n_sequence_rows": 20,
                "gaussian_nll_event_equal": 1.0 + target_index,
                "interval_coverage_event_equal": coverage,
                "interval_nominal_coverage": 0.9,
                "interval_calibration_error": coverage - 0.9,
                "interval_absolute_calibration_error": abs(coverage - 0.9),
                "mean_interval_width_event_equal": 2.0,
                "aggregation_unit": "independent_event_equal",
                "distribution": "gaussian",
            }
        )
        for event_id in ("el_nino_2000_2001", "la_nina_2001_2002"):
            event_rows.append(
                {
                    "fold": "event_fold_01",
                    "target": target,
                    "event_id": event_id,
                    "n_sequence_rows": 10,
                    "gaussian_nll_mean": 1.0,
                    "interval_coverage": coverage,
                    "mean_interval_width": 2.0,
                    "mean_absolute_error": 0.5,
                    "interval_nominal_coverage": 0.9,
                    "interval_calibration_error": coverage - 0.9,
                }
            )
    tables["event_probabilistic_metrics.csv"] = pd.DataFrame(calibration_rows)
    tables["event_probabilistic_metrics_by_event.csv"] = pd.DataFrame(event_rows)
    return tables


def _parameters(phase: int, *, active: bool) -> dict[str, object]:
    if phase == 5:
        return {
            "model": "rf",
            "horizons": [4],
            "lags": [4, 8],
            "n_estimators": 20,
            "n_splits": 2,
            "n_physical_predictors": 31,
            "noise_copies": 1 if active else 0,
            "noise_scale": 0.02,
            "mixup_alpha": 0.4 if active else None,
            "min_train_active_events_per_type": 3,
            "validation_unit": "whole_event_and_neutral_quarter",
        }
    return {
        "seq_len": 24,
        "horizon": 4,
        "epochs": 1,
        "batch_size": 16,
        "hidden": 8,
        "augmentation": active,
        "device": "cpu",
        "scalar_channels": [f"v{index}" for index in range(31)],
        "spatial_channels": ["thetao_surface_c"],
    }


def _input_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "exists": True,
        "is_directory": False,
        "size_bytes": path.stat().st_size,
        "modified_ns": path.stat().st_mtime_ns,
        "sha256": ablation.sha256_file(path),
    }


def _make_run(
    root: Path,
    *,
    phase: int,
    active: bool,
    suffix: str,
    status: str = "complete",
    fold_variant: str = "standard",
    input_path: Path | None = None,
    parameter_override: dict[str, object] | None = None,
) -> Path:
    run_id = f"F{phase}_20260713T120000Z_{suffix}"
    directory = root / "runs" / run_id
    table_dir = directory / "tables"
    table_dir.mkdir(parents=True)
    tables = (
        _phase5_tables(active, fold_variant=fold_variant)
        if phase == 5
        else _phase7_tables(active, fold_variant=fold_variant)
    )
    catalog_rows: list[dict[str, object]] = []
    for name, frame in tables.items():
        path = table_dir / name
        frame.to_csv(path, index=False, lineterminator="\n")
        catalog_rows.append(
            {
                "table": name,
                "path": f"tables/{name}",
                "sha256": ablation.sha256_file(path),
            }
        )
    tables_manifest = directory / "tables_manifest.csv"
    pd.DataFrame(catalog_rows).to_csv(tables_manifest, index=False, lineterminator="\n")
    parameters = _parameters(phase, active=active)
    if parameter_override:
        parameters.update(parameter_override)
    manifest = {
        "schema_version": "nino26-run-v1",
        "run_id": run_id,
        "phase": phase,
        "mode": "smoke",
        "seed": 42,
        "status": status,
        "finished_at": "2026-07-13T12:01:00+00:00",
        "parameters": parameters,
        "parameters_sha256": ablation._json_hash(parameters),
        "git": {"commit": "abc123", "status_sha256": "d" * 64},
        "environment": {"python": "3.12.3", "packages": {"pandas": "3.0.3"}},
        "inputs": [] if input_path is None else [_input_record(input_path)],
        "configs": [],
        "files": [],
        "n_tables": len(tables),
        "tables_manifest_sha256": ablation.sha256_file(tables_manifest),
    }
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return directory


def _audit_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data" / "audit" / "augmentation_ablation"
    monkeypatch.setattr(ablation, "AUDIT_ROOT", root)
    return root / "comparison_001"


def test_f5_dry_run_is_read_only_and_event_equal(tmp_path, monkeypatch, capsys) -> None:
    with_run = _make_run(tmp_path, phase=5, active=True, suffix="on")
    without_run = _make_run(tmp_path, phase=5, active=False, suffix="off")
    output = _audit_output(tmp_path, monkeypatch)

    assert (
        ablation.main(
            [
                "--with-augmentation",
                str(with_run),
                "--without-augmentation",
                str(without_run),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    assert not output.exists()
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "validated"
    assert summary["write"] is False
    frame, manifest = ablation.compare_runs(with_run, without_run)
    assert set(frame["metric_scope"]) == {
        "classification_by_fold",
        "event_dimension_by_fold_target",
    }
    assert "brier_multiclass" in set(frame["metric"])
    assert frame["n_original_events"].eq(2).all()
    assert frame["augmentation_does_not_change_independent_n"].eq(True).all()
    assert manifest["augmentation_does_not_change_independent_n"] is True


def test_f7_write_has_paired_ic90_and_preserves_hash_receipts(
    tmp_path, monkeypatch
) -> None:
    with_run = _make_run(tmp_path, phase=7, active=True, suffix="on")
    without_run = _make_run(tmp_path, phase=7, active=False, suffix="off")
    output = _audit_output(tmp_path, monkeypatch)

    assert (
        ablation.main(
            [
                "--with-augmentation",
                str(with_run),
                "--without-augmentation",
                str(without_run),
                "--output-dir",
                str(output),
                "--write",
            ]
        )
        == 0
    )

    csv_path = output / ablation.CSV_NAME
    manifest_path = output / ablation.MANIFEST_NAME
    frame = pd.read_csv(csv_path)
    calibration = frame.loc[
        frame["metric_scope"].eq("ic90_calibration_by_fold_target")
    ]
    assert len(calibration) == 3 * len(TARGETS)
    assert calibration["n_original_events"].eq(2).all()
    assert calibration["aggregation_unit"].eq(
        "independent_event_equal_within_fold"
    ).all()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["outputs"][ablation.CSV_NAME]["sha256"] == ablation.sha256_file(
        csv_path
    )
    assert manifest["runs"]["with_augmentation"]["run_manifest_sha256"]
    assert manifest["runs"]["without_augmentation"]["synthetic_lineage_rows"] == 0


def test_refuses_incomplete_run(tmp_path) -> None:
    with_run = _make_run(
        tmp_path, phase=5, active=True, suffix="on", status="failed"
    )
    without_run = _make_run(tmp_path, phase=5, active=False, suffix="off")

    with pytest.raises(ablation.AblationComparisonError, match="Incomplete"):
        ablation.compare_runs(with_run, without_run)


def test_refuses_incompatible_source_or_data(tmp_path) -> None:
    source_a = tmp_path / "input_a.csv"
    source_b = tmp_path / "input_b.csv"
    source_a.write_text("x\n1\n", encoding="utf-8")
    source_b.write_text("x\n2\n", encoding="utf-8")
    with_run = _make_run(
        tmp_path, phase=5, active=True, suffix="on", input_path=source_a
    )
    without_run = _make_run(
        tmp_path, phase=5, active=False, suffix="off", input_path=source_b
    )

    with pytest.raises(ablation.AblationComparisonError, match="source/data"):
        ablation.compare_runs(with_run, without_run)


def test_refuses_different_fold_contract(tmp_path) -> None:
    with_run = _make_run(tmp_path, phase=7, active=True, suffix="on")
    without_run = _make_run(
        tmp_path,
        phase=7,
        active=False,
        suffix="off",
        fold_variant="different",
    )

    with pytest.raises(ablation.AblationComparisonError, match="fold contract"):
        ablation.compare_runs(with_run, without_run)


def test_refuses_on_on_and_nonaugmentation_parameter_changes(tmp_path) -> None:
    on_a = _make_run(tmp_path, phase=5, active=True, suffix="on_a")
    on_b = _make_run(tmp_path, phase=5, active=True, suffix="on_b")
    with pytest.raises(ablation.AblationComparisonError, match="ON/ON"):
        ablation.compare_runs(on_a, on_b)

    off_changed = _make_run(
        tmp_path,
        phase=5,
        active=False,
        suffix="off_changed",
        parameter_override={"horizons": [8]},
    )
    with pytest.raises(ablation.AblationComparisonError, match="Scientific parameters"):
        ablation.compare_runs(on_a, off_changed)


def test_write_refuses_nonempty_destination_and_outside_path(
    tmp_path, monkeypatch
) -> None:
    with_run = _make_run(tmp_path, phase=5, active=True, suffix="on")
    without_run = _make_run(tmp_path, phase=5, active=False, suffix="off")
    output = _audit_output(tmp_path, monkeypatch)
    output.mkdir(parents=True)
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    frame, manifest = ablation.compare_runs(with_run, without_run)

    with pytest.raises(ablation.AblationComparisonError, match="preserving"):
        ablation.write_outputs(frame, manifest, output)
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    with pytest.raises(ablation.AblationComparisonError, match="must be"):
        ablation._output_directory(tmp_path / "outside")
