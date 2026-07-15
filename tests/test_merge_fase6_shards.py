from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.merge_fase6_shards import (
    MergeProducts,
    SourceShard,
    assert_homogeneous_shards,
    build_field_gate,
    manifest_fingerprint_payload,
    pool_oos_pixel_metrics,
    validate_prediction_coverage,
    validate_source_manifest,
    write_merge_artifacts,
)


def _pixel_meta(pixel_ids: tuple[str, ...], fractions: tuple[float, ...] | None = None) -> pd.DataFrame:
    fractions = fractions or tuple(1.0 for _ in pixel_ids)
    latitude = np.linspace(-20.0, -10.0, len(pixel_ids))
    frame = pd.DataFrame(
        {
            "pixel_id": pixel_ids,
            "lat": latitude,
            "lon": np.linspace(-55.0, -45.0, len(pixel_ids)),
            "brazil_fraction": fractions,
        }
    )
    frame["area_weight"] = np.cos(np.deg2rad(frame["lat"])) * frame["brazil_fraction"]
    return frame


def _predictions(
    pixel_ids: tuple[str, ...] = ("0", "1"),
    *,
    model_errors: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    model_errors = model_errors or tuple(1.0 for _ in pixel_ids)
    rows: list[dict[str, object]] = []
    for pixel_id, error in zip(pixel_ids, model_errors):
        for fold, time in (("event_fold_01", "2010-01-03"), ("event_fold_02", "2012-01-01")):
            rows.append(
                {
                    "target_variable": "precip_weekly_mm",
                    "target_transform": "train_robust_z",
                    "target_units": "mm week-1",
                    "model": "rf",
                    "fold": fold,
                    "time": pd.Timestamp(time),
                    "pixel_id": pixel_id,
                    "condition": "el_nino_pico",
                    "lag_weeks": 4,
                    "observed": 0.0,
                    "predicted": error,
                    "baseline_climatology_train_mean": 2.0,
                    "baseline_climatology_week_of_year_train": 2.5,
                    "baseline_persistence": 3.0,
                    "baseline_phase4_statistical_ridge": 3.5,
                    "observed_native_value": 10.0,
                    "predicted_native_value": 10.0 + error,
                    "baseline_persistence_native_value": 13.0,
                    "source_run_id": f"shard_{pixel_id}",
                }
            )
    return pd.DataFrame(rows)


def _manifest(*, seed: int = 42, start: int = 0, stop: int = 1) -> dict[str, object]:
    return {
        "schema_version": "nino26-run-v1",
        "run_id": f"F6_test_{start}_{stop}",
        "phase": 6,
        "mode": "official",
        "status": "complete",
        "seed": seed,
        "command": "python scripts/run_fase6_brazil_ml.py --n-estimators 10",
        "parameters": {
            "model": "rf",
            "target_variable": "precip_weekly_mm",
            "target_transform": "train_robust_z",
            "conditions": ["el_nino_pico"],
            "lags": [4],
            "pixel_start": start,
            "pixel_stop": stop,
            "n_pixels": stop - start,
            "phase4_statistical_baseline_required": True,
        },
        "git": {"commit": "abc", "status_sha256": "clean"},
        "environment": {
            "packages": {
                "numpy": "2",
                "pandas": "2",
                "scikit-learn": "1",
                "xgboost": "3",
                "xarray": "2026",
            }
        },
        "configs": [{"path": "project.yaml", "exists": True, "sha256": "cfg"}],
        "inputs": [
            {"path": "target.zarr", "exists": True, "is_directory": True, "tree_sha256": "target"},
            {"path": "master.csv", "exists": True, "is_directory": False, "sha256": "master"},
        ],
    }


def _source_shard(manifest: dict[str, object], directory: Path) -> SourceShard:
    payload = manifest_fingerprint_payload(
        manifest,
        target_units="mm week-1",
        native_schema=True,
        phase4_comparator=True,
    )
    from scripts.merge_fase6_shards import _json_hash

    return SourceShard(
        directory=directory,
        manifest=manifest,
        fingerprint=_json_hash(payload),
        fingerprint_payload=payload,
        metrics=pd.DataFrame(),
        predictions=pd.DataFrame(),
        inventory=pd.DataFrame(),
        importance=pd.DataFrame(),
        predictor_contract=pd.DataFrame({"variable": [f"v{i}" for i in range(31)]}),
        native_schema=True,
        phase4_comparator=True,
        source_files=(),
    )


def test_pool_oos_metrics_combines_folds_before_skill() -> None:
    predictions = _predictions(model_errors=(1.0, 2.0))
    meta = _pixel_meta(("0", "1"))
    validate_prediction_coverage(
        predictions,
        expected_ids={"0", "1"},
        declared_conditions=("el_nino_pico",),
        declared_lags=(4,),
        phase4_comparator=True,
    )
    pooled = pool_oos_pixel_metrics(
        predictions,
        meta,
        native_schema=True,
        phase4_comparator=True,
    ).set_index("pixel_id")
    assert len(pooled) == 2
    assert pooled.loc["0", "n_oos_folds"] == 2
    assert pooled.loc["0", "rmse_model_oos"] == pytest.approx(1.0)
    assert pooled.loc["1", "rmse_model_oos"] == pytest.approx(2.0)
    assert pooled.loc["0", "skill_vs_phase4_statistical_ridge"] == pytest.approx(
        1.0 - 1.0 / 3.5
    )
    assert bool(pooled.loc["0", "gate_pass"])


def test_prediction_coverage_requires_exact_pixel_set_and_all_finite_baselines() -> None:
    predictions = _predictions()
    with pytest.raises(ValueError, match="nao cobre exatamente"):
        validate_prediction_coverage(
            predictions.loc[predictions["pixel_id"].ne("1")].copy(),
            expected_ids={"0", "1"},
            declared_conditions=("el_nino_pico",),
            declared_lags=(4,),
            phase4_comparator=True,
        )
    broken = predictions.copy()
    broken.loc[0, "baseline_phase4_statistical_ridge"] = np.nan
    with pytest.raises(ValueError, match="sem avaliacao comum"):
        validate_prediction_coverage(
            broken,
            expected_ids={"0", "1"},
            declared_conditions=("el_nino_pico",),
            declared_lags=(4,),
            phase4_comparator=True,
        )


def test_field_gate_uses_one_pooled_vote_per_pixel_and_area_weighted_errors() -> None:
    predictions = _predictions(("0", "1", "2"), model_errors=(1.0, 1.0, 10.0))
    meta = _pixel_meta(("0", "1", "2"), fractions=(1.0, 1.0, 0.01))
    pooled = pool_oos_pixel_metrics(
        predictions,
        meta,
        native_schema=True,
        phase4_comparator=True,
    )
    gate = build_field_gate(
        predictions,
        pooled,
        expected_ids={"0", "1", "2"},
        native_schema=True,
        phase4_comparator=True,
    ).iloc[0]
    assert gate["fraction_pixels_positive_skill"] == pytest.approx(2.0 / 3.0)
    assert gate["pixel_coverage_fraction"] == 1.0
    assert bool(gate["pixel_set_exact"])
    assert gate["field_skill_vs_phase4_statistical_ridge"] > 0
    assert bool(gate["gate_pass"])


def test_legacy_raw_mm_schema_is_readable_but_cannot_pass_new_gate() -> None:
    predictions = _predictions()
    predictions = predictions.drop(
        columns=[
            "target_units",
            "observed_native_value",
            "predicted_native_value",
            "baseline_persistence_native_value",
            "baseline_phase4_statistical_ridge",
        ]
    )
    predictions["target_units"] = "mm (legacy raw_mm schema)"
    predictions["observed_raw_mm"] = 10.0
    predictions["predicted_raw_mm"] = 11.0
    predictions["baseline_persistence_raw_mm"] = 13.0
    meta = _pixel_meta(("0", "1"))
    validate_prediction_coverage(
        predictions,
        expected_ids={"0", "1"},
        declared_conditions=("el_nino_pico",),
        declared_lags=(4,),
        phase4_comparator=False,
    )
    pooled = pool_oos_pixel_metrics(
        predictions,
        meta,
        native_schema=False,
        phase4_comparator=False,
    )
    gate = build_field_gate(
        predictions,
        pooled,
        expected_ids={"0", "1"},
        native_schema=False,
        phase4_comparator=False,
    )
    assert not pooled["gate_eligible"].any()
    assert not gate["gate_eligible"].any()
    assert not gate["gate_pass"].any()


def test_fingerprint_ignores_only_shard_extent_and_rejects_seed_mismatch(tmp_path: Path) -> None:
    first_manifest = _manifest(start=0, stop=10)
    second_manifest = _manifest(start=10, stop=20)
    first = _source_shard(first_manifest, tmp_path / str(first_manifest["run_id"]))
    second = _source_shard(second_manifest, tmp_path / str(second_manifest["run_id"]))
    assert assert_homogeneous_shards([first, second]) == first.fingerprint

    changed = deepcopy(second_manifest)
    changed["seed"] = 99
    mismatched = _source_shard(changed, tmp_path / "F6_seed_99")
    with pytest.raises(ValueError, match="fingerprints"):
        assert_homogeneous_shards([first, mismatched])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"status": "failed"}, "status deve ser complete"),
        ({"mode": "smoke"}, "somente shards oficiais"),
        ({"command": "python run.py --research-override-gate"}, "override"),
    ],
)
def test_source_manifest_rejects_invalid_or_overridden_run(
    tmp_path: Path,
    change: dict[str, object],
    message: str,
) -> None:
    manifest = _manifest()
    manifest.update(change)
    directory = tmp_path / str(manifest["run_id"])
    with pytest.raises(ValueError, match=message):
        validate_source_manifest(directory, manifest)


def test_write_failure_finalizes_artifact_as_failed() -> None:
    class BrokenRun:
        def __init__(self) -> None:
            self.finalizations: list[dict[str, object]] = []

        def write_table(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("disk failure")

        def finalize(self, **kwargs: object) -> None:
            self.finalizations.append(dict(kwargs))

    products = MergeProducts(
        fold_metrics=pd.DataFrame(),
        pooled_pixel_metrics=pd.DataFrame(),
        predictions=pd.DataFrame(),
        gate=pd.DataFrame(),
        pixel_inventory=pd.DataFrame(),
        pixel_variable_importance=pd.DataFrame(),
        predictor_contract=pd.DataFrame(),
        source_files=[],
        source_runs=[],
        fingerprint="fingerprint",
        native_schema=True,
        phase4_comparator=True,
        coverage=1.0,
    )
    run = BrokenRun()
    with pytest.raises(RuntimeError, match="disk failure"):
        write_merge_artifacts(run, products)
    assert run.finalizations == [{"status": "failed", "notes": "RuntimeError: disk failure"}]


def test_merge_artifact_persists_importance_and_predictor_contract() -> None:
    class RecordingRun:
        def __init__(self) -> None:
            self.tables: list[tuple[str, dict[str, object]]] = []
            self.finalizations: list[dict[str, object]] = []

        def write_table(self, name: str, _frame: pd.DataFrame, **kwargs: object) -> None:
            self.tables.append((name, dict(kwargs)))

        def finalize(self, **kwargs: object) -> None:
            self.finalizations.append(dict(kwargs))

    products = MergeProducts(
        fold_metrics=pd.DataFrame(),
        pooled_pixel_metrics=pd.DataFrame(),
        predictions=pd.DataFrame(),
        gate=pd.DataFrame(),
        pixel_inventory=pd.DataFrame(),
        pixel_variable_importance=pd.DataFrame(),
        predictor_contract=pd.DataFrame(
            {"variable": [f"v{i:02d}" for i in range(31)]}
        ),
        source_files=[],
        source_runs=[],
        fingerprint="fingerprint",
        native_schema=True,
        phase4_comparator=True,
        coverage=1.0,
    )
    run = RecordingRun()
    write_merge_artifacts(run, products)

    names = [name for name, _ in run.tables]
    assert "pixel_variable_importance" in names
    assert "predictor_contract" in names
    importance_kwargs = dict(run.tables)["pixel_variable_importance"]
    assert tuple(importance_kwargs["primary_keys"]) == (
        "target_variable",
        "target_transform",
        "model",
        "fold",
        "pixel_id",
        "condition",
        "lag_weeks",
        "variable",
    )
    assert run.finalizations and run.finalizations[0].get("status", "complete") == "complete"
