#!/usr/bin/env python3
"""Compare paired F5/F7 ArtifactRuns with augmentation ON versus OFF.

The comparison is deliberately stricter than a visual/model-score comparison.
Both runs must be complete and integral ArtifactRuns produced from the same
source, data, configuration, seed, model/backend, folds, evaluation groups and
scientific parameters.  Only a small, phase-specific allow-list of training
augmentation parameters may differ.

The default operation validates and prints a read-only summary.  ``--write``
writes ``augmentation_ablation.csv`` and
``augmentation_ablation_manifest.json`` to a new/empty directory explicitly
chosen below ``data/audit/augmentation_ablation``.  Synthetic rows are never
reported as independent observations: every comparison row uses an
independent-event or whole-event-fold denominator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from nino_brasil.artifacts import (  # noqa: E402
    sha256_file,
    validate_artifact_run,
)


AUDIT_ROOT = ROOT / "data" / "audit" / "augmentation_ablation"
CSV_NAME = "augmentation_ablation.csv"
MANIFEST_NAME = "augmentation_ablation_manifest.json"
ALLOWED_AUGMENTATION_PARAMETERS: Mapping[int, frozenset[str]] = {
    5: frozenset({"noise_copies", "noise_scale", "mixup_alpha"}),
    7: frozenset({"augmentation"}),
}
REQUIRED_COMMON_TABLES = frozenset(
    {
        "fold_contract.csv",
        "fold_metrics.csv",
        "independent_support_by_fold.csv",
        "oos_predictions.csv",
        "augmentation_provenance.csv",
    }
)
REQUIRED_PHASE_TABLES: Mapping[int, frozenset[str]] = {
    5: frozenset(
        {
            "event_dimension_oos_predictions.csv",
            "event_dimension_augmentation_provenance.csv",
        }
    ),
    7: frozenset(
        {
            "event_probabilistic_metrics.csv",
            "event_probabilistic_metrics_by_event.csv",
        }
    ),
}
F5_EVENT_TARGETS = frozenset(
    {"Y_pico", "Y_tempo_para_pico_sem", "Y_duracao_sem"}
)
F7_EVENT_TARGETS = frozenset(
    {"peak_magnitude_c", "event_time_to_peak_weeks", "event_duration_weeks"}
)


class AblationComparisonError(ValueError):
    """Raised when two runs cannot support a scientific ON/OFF ablation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")


def _json_hash(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AblationComparisonError(f"Invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise AblationComparisonError(f"JSON object required: {path}")
    return payload


def _normalise_scalar(value: object) -> object:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    return str(value)


def _normalise_frame(frame: pd.DataFrame, columns: Sequence[str]) -> list[list[object]]:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise AblationComparisonError(f"Table is missing columns: {missing}")
    return [
        [_normalise_scalar(value) for value in row]
        for row in frame.loc[:, list(columns)].itertuples(index=False, name=None)
    ]


def _assert_same_frame(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    label: str,
    columns: Sequence[str] | None = None,
) -> None:
    selected = list(columns) if columns is not None else list(left.columns)
    if columns is None and list(left.columns) != list(right.columns):
        raise AblationComparisonError(f"{label} schemas differ")
    if _normalise_frame(left, selected) != _normalise_frame(right, selected):
        raise AblationComparisonError(f"{label} differ")


def _stable_provenance(records: object, *, section: str) -> list[dict[str, object]]:
    if not isinstance(records, list):
        raise AblationComparisonError(f"Manifest {section} catalog is missing")
    stable: list[dict[str, object]] = []
    for raw in records:
        if not isinstance(raw, Mapping):
            raise AblationComparisonError(f"Manifest {section} record is invalid")
        path = str(raw.get("path") or "")
        digest = raw.get("tree_sha256") if raw.get("is_directory") else raw.get("sha256")
        if not path or not digest:
            raise AblationComparisonError(f"Manifest {section} record lacks path/hash")
        stable.append(
            {
                "path": os.path.normcase(os.path.abspath(path)),
                "is_directory": bool(raw.get("is_directory")),
                "sha256": str(digest),
            }
        )
    return sorted(stable, key=lambda item: (str(item["path"]), str(item["sha256"])))


def _load_table_catalog(directory: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    path = directory / "tables_manifest.csv"
    try:
        catalog = pd.read_csv(path)
    except Exception as exc:  # pandas exposes parser/encoding subclasses here
        raise AblationComparisonError(f"Cannot read table manifest: {path}") from exc
    if not {"table", "path", "sha256"}.issubset(catalog.columns):
        raise AblationComparisonError(f"Invalid table manifest schema: {path}")
    if catalog["table"].duplicated().any():
        raise AblationComparisonError(f"Duplicate table names: {path}")
    paths: dict[str, Path] = {}
    root = directory.resolve()
    for row in catalog.itertuples(index=False):
        member = (directory / str(row.path)).resolve()
        if member != root and root not in member.parents:
            raise AblationComparisonError(f"Unsafe table path in {path}: {row.path}")
        paths[str(row.table)] = member
    return catalog, paths


def _table(paths: Mapping[str, Path], name: str) -> pd.DataFrame:
    try:
        path = paths[name]
    except KeyError as exc:
        raise AblationComparisonError(f"Required table missing: {name}") from exc
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # F7 --no-augmentation intentionally has no lineage rows or columns.
        # The empty file remains hash-audited through its ArtifactRun manifest.
        return pd.DataFrame()
    except Exception as exc:
        raise AblationComparisonError(f"Cannot read required table: {path}") from exc


def _validate_run(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    if not directory.is_dir():
        raise AblationComparisonError(f"ArtifactRun directory not found: {directory}")
    problems = validate_artifact_run(directory)
    if not problems.empty:
        detail = "; ".join(
            f"{row.get('type', 'integrity')}:{row.get('item', '')}"
            for row in problems.to_dict(orient="records")
        )
        raise AblationComparisonError(
            f"ArtifactRun integrity validation failed for {directory}: {detail}"
        )
    manifest_path = directory / "run_manifest.json"
    manifest = _read_json_mapping(manifest_path)
    if manifest.get("schema_version") != "nino26-run-v1":
        raise AblationComparisonError(f"Unsupported ArtifactRun schema: {directory}")
    if str(manifest.get("run_id") or "") != directory.name:
        raise AblationComparisonError(f"run_id/directory mismatch: {directory}")
    if manifest.get("status") != "complete":
        raise AblationComparisonError(f"Incomplete ArtifactRun: {directory}")
    if not manifest.get("finished_at"):
        raise AblationComparisonError(f"ArtifactRun has no completion timestamp: {directory}")
    try:
        phase = int(manifest.get("phase"))
        seed = int(manifest.get("seed"))
    except (TypeError, ValueError) as exc:
        raise AblationComparisonError(f"Invalid phase/seed: {directory}") from exc
    if phase not in (5, 7):
        raise AblationComparisonError(f"Only F5 and F7 are supported, got F{phase}")
    if manifest.get("mode") not in {"official", "smoke"}:
        raise AblationComparisonError(f"Invalid ArtifactRun mode: {directory}")
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise AblationComparisonError(f"ArtifactRun parameters are missing: {directory}")
    if str(manifest.get("parameters_sha256") or "") != _json_hash(parameters):
        raise AblationComparisonError(f"parameters_sha256 mismatch: {directory}")
    catalog, paths = _load_table_catalog(directory)
    expected = REQUIRED_COMMON_TABLES | REQUIRED_PHASE_TABLES[phase]
    missing = sorted(expected.difference(paths))
    if missing:
        raise AblationComparisonError(
            f"ArtifactRun lacks ablation audit tables {missing}: {directory}"
        )
    return {
        "directory": directory,
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "tables_manifest_sha256": sha256_file(directory / "tables_manifest.csv"),
        "catalog": catalog,
        "paths": paths,
        "phase": phase,
        "seed": seed,
    }


def _augmentation_active(run: Mapping[str, Any]) -> bool:
    parameters = run["manifest"]["parameters"]
    if run["phase"] == 5:
        try:
            copies = int(parameters.get("noise_copies", 0) or 0)
            alpha = float(parameters.get("mixup_alpha") or 0.0)
        except (TypeError, ValueError) as exc:
            raise AblationComparisonError("Invalid F5 augmentation parameters") from exc
        if copies < 0 or alpha < 0:
            raise AblationComparisonError("Negative F5 augmentation parameters")
        return copies > 0 or alpha > 0.0
    value = parameters.get("augmentation")
    if not isinstance(value, bool):
        raise AblationComparisonError("F7 requires boolean parameter 'augmentation'")
    return value


def _synthetic_rows(run: Mapping[str, Any]) -> int:
    names = ["augmentation_provenance.csv"]
    if run["phase"] == 5:
        names.append("event_dimension_augmentation_provenance.csv")
    count = 0
    for name in names:
        frame = _table(run["paths"], name)
        if frame.empty:
            continue
        if "augmentation_method" not in frame or "augmentation_id" not in frame:
            raise AblationComparisonError(f"{name} lacks augmentation lineage columns")
        method = frame["augmentation_method"].fillna("").astype(str).str.lower()
        augmentation_id = frame["augmentation_id"].fillna("").astype(str).str.lower()
        synthetic = ~method.isin({"", "none", "original"}) | ~augmentation_id.isin(
            {"", "none", "original"}
        )
        if "independent_event" in frame:
            independent = frame["independent_event"].astype(str).str.lower().isin(
                {"true", "1"}
            )
            if (synthetic & independent).any():
                raise AblationComparisonError(
                    f"{name} incorrectly marks synthetic rows as independent"
                )
        count += int(synthetic.sum())
    return count


def _scientific_parameters(run: Mapping[str, Any]) -> dict[str, Any]:
    excluded = ALLOWED_AUGMENTATION_PARAMETERS[run["phase"]]
    return {
        key: value
        for key, value in run["manifest"]["parameters"].items()
        if key not in excluded
    }


def _backend_identity(run: Mapping[str, Any]) -> str:
    parameters = run["manifest"]["parameters"]
    for key in ("model", "backend", "architecture"):
        if parameters.get(key) is not None:
            return str(parameters[key])
    return "convlstm_gru_pytorch" if run["phase"] == 7 else "unknown"


def _evaluation_identity(run: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    paths = run["paths"]
    phase = run["phase"]
    contract = _table(paths, "fold_contract.csv")
    support = _table(paths, "independent_support_by_fold.csv")
    predictions = _table(paths, "oos_predictions.csv")
    if phase == 5:
        prediction_columns = [
            "model",
            "experiment_horizon_weeks",
            "fold",
            "origin_time",
            "target_time",
            "event_id",
            "observed_state",
            "is_original_observation",
        ]
        event_predictions = _table(paths, "event_dimension_oos_predictions.csv")
        event_columns = [
            "fold",
            "event_id",
            "tipo",
            "origin_time",
            "target",
            "observed",
            "baseline_global_train",
            "baseline_type_train",
            "preprocessing_fit_end",
        ]
        return {
            "fold contract": contract,
            "independent support": support,
            "classification evaluation rows": predictions.loc[:, prediction_columns],
            "event-dimension evaluation events": event_predictions.loc[:, event_columns],
        }
    prediction_columns = [
        "fold",
        "origin_time",
        "target_time",
        "event_id",
        "observed_state",
        "observed_peak_magnitude_c",
        "observed_event_time_to_peak_weeks",
        "observed_event_duration_weeks",
    ]
    by_event = _table(paths, "event_probabilistic_metrics_by_event.csv")
    by_event_columns = [
        "fold",
        "target",
        "event_id",
        "n_sequence_rows",
        "interval_nominal_coverage",
    ]
    return {
        "fold contract": contract,
        "independent support": support,
        "classification evaluation rows": predictions.loc[:, prediction_columns],
        "event-dimension evaluation events": by_event.loc[:, by_event_columns],
    }


def _assert_pair_compatible(with_run: Mapping[str, Any], without_run: Mapping[str, Any]) -> None:
    if with_run["directory"] == without_run["directory"]:
        raise AblationComparisonError("Two distinct ArtifactRun directories are required")
    for field in ("phase", "seed"):
        if with_run[field] != without_run[field]:
            raise AblationComparisonError(f"Runs have different {field}")
    for field in ("mode",):
        if with_run["manifest"].get(field) != without_run["manifest"].get(field):
            raise AblationComparisonError(f"Runs have different {field}")
    if _backend_identity(with_run) != _backend_identity(without_run):
        raise AblationComparisonError("Runs have different model/backend")
    if _scientific_parameters(with_run) != _scientific_parameters(without_run):
        raise AblationComparisonError(
            "Scientific parameters differ beyond the augmentation allow-list"
        )
    for section in ("inputs", "configs"):
        left = _stable_provenance(with_run["manifest"].get(section), section=section)
        right = _stable_provenance(without_run["manifest"].get(section), section=section)
        if left != right:
            raise AblationComparisonError(f"Runs have incompatible source/data {section}")
    if with_run["manifest"].get("environment") != without_run["manifest"].get(
        "environment"
    ):
        raise AblationComparisonError("Runs have incompatible execution environments")
    left_git = with_run["manifest"].get("git", {})
    right_git = without_run["manifest"].get("git", {})
    for key in ("commit", "status_sha256"):
        if left_git.get(key) != right_git.get(key):
            raise AblationComparisonError(f"Runs have incompatible git {key}")

    if not _augmentation_active(with_run):
        raise AblationComparisonError("--with-augmentation is not an augmentation-ON run")
    if _augmentation_active(without_run):
        raise AblationComparisonError(
            "--without-augmentation is augmentation-ON; ON/ON comparisons are forbidden"
        )
    with_synthetic = _synthetic_rows(with_run)
    without_synthetic = _synthetic_rows(without_run)
    if with_synthetic <= 0:
        raise AblationComparisonError("Augmentation-ON run has no synthetic lineage rows")
    if without_synthetic != 0:
        raise AblationComparisonError("Augmentation-OFF run contains synthetic lineage rows")

    left_identity = _evaluation_identity(with_run)
    right_identity = _evaluation_identity(without_run)
    if left_identity.keys() != right_identity.keys():
        raise AblationComparisonError("Evaluation identity catalogs differ")
    for label in left_identity:
        _assert_same_frame(left_identity[label], right_identity[label], label=label)


def _row_key(frame: pd.DataFrame, phase: int) -> list[str]:
    keys = ["fold"]
    if phase == 5:
        keys.insert(0, "experiment_horizon_weeks")
    missing = [key for key in keys if key not in frame]
    if missing:
        raise AblationComparisonError(f"fold_metrics lacks keys: {missing}")
    return keys


def _paired_frames(
    with_frame: pd.DataFrame,
    without_frame: pd.DataFrame,
    *,
    keys: Sequence[str],
    label: str,
) -> pd.DataFrame:
    if with_frame.duplicated(list(keys)).any() or without_frame.duplicated(list(keys)).any():
        raise AblationComparisonError(f"{label} keys are not unique")
    left_keys = _normalise_frame(with_frame, keys)
    right_keys = _normalise_frame(without_frame, keys)
    if left_keys != right_keys:
        raise AblationComparisonError(f"{label} fold/target keys differ")
    return with_frame.merge(
        without_frame,
        on=list(keys),
        how="inner",
        suffixes=("_with", "_without"),
        validate="one_to_one",
        sort=False,
    )


def _support_counts(run: Mapping[str, Any]) -> dict[tuple[object, ...], tuple[int, int]]:
    support = _table(run["paths"], "independent_support_by_fold.csv")
    key_columns = ["fold"]
    if run["phase"] == 5:
        key_columns.insert(0, "experiment_horizon_weeks")
    required = {
        *key_columns,
        "n_test_active_events",
        "n_test_neutral_blocks",
    }
    if missing := required.difference(support.columns):
        raise AblationComparisonError(
            f"independent_support_by_fold lacks {sorted(missing)}"
        )
    result: dict[tuple[object, ...], tuple[int, int]] = {}
    for row in support.itertuples(index=False):
        key = tuple(_normalise_scalar(getattr(row, column)) for column in key_columns)
        active = int(getattr(row, "n_test_active_events"))
        groups = active + int(getattr(row, "n_test_neutral_blocks"))
        result[key] = (active, groups)
    return result


def _metric_row(
    *,
    run: Mapping[str, Any],
    scope: str,
    metric: str,
    fold: object,
    with_value: object,
    without_value: object,
    target: object = None,
    horizon: object = None,
    n_events: int,
    n_groups: int | None,
    aggregation_unit: str,
    direction: str,
) -> dict[str, object]:
    with_number = float(with_value)
    without_number = float(without_value)
    return {
        "phase": int(run["phase"]),
        "mode": str(run["manifest"]["mode"]),
        "model_backend": _backend_identity(run),
        "metric_scope": scope,
        "metric": metric,
        "target": None if target is None else str(target),
        "horizon_weeks": _normalise_scalar(horizon),
        "fold": str(fold),
        "aggregation_unit": aggregation_unit,
        "n_original_events": int(n_events),
        "n_original_test_groups": None if n_groups is None else int(n_groups),
        "with_augmentation_value": with_number,
        "without_augmentation_value": without_number,
        "delta_with_minus_without": with_number - without_number,
        "direction": direction,
        "augmentation_does_not_change_independent_n": True,
    }


def _classification_rows(
    with_run: Mapping[str, Any], without_run: Mapping[str, Any]
) -> list[dict[str, object]]:
    with_frame = _table(with_run["paths"], "fold_metrics.csv")
    without_frame = _table(without_run["paths"], "fold_metrics.csv")
    keys = _row_key(with_frame, with_run["phase"])
    if _row_key(without_frame, without_run["phase"]) != keys:
        raise AblationComparisonError("Classification key schema differs")
    paired = _paired_frames(with_frame, without_frame, keys=keys, label="classification")
    standard = ["skill_f1_vs_best_baseline", "f1_macro_9state"]
    brier = sorted(
        column
        for column in set(with_frame.columns).intersection(without_frame.columns)
        if "brier" in column.lower()
        and pd.api.types.is_numeric_dtype(with_frame[column])
        and pd.api.types.is_numeric_dtype(without_frame[column])
    )
    metrics = [metric for metric in standard if metric in with_frame and metric in without_frame]
    metrics.extend(metric for metric in brier if metric not in metrics)
    if not {"skill_f1_vs_best_baseline", "f1_macro_9state"}.issubset(metrics):
        raise AblationComparisonError("Classification fold metrics lack skill/F1")
    support = _support_counts(with_run)
    rows: list[dict[str, object]] = []
    for record in paired.to_dict(orient="records"):
        key = tuple(_normalise_scalar(record[column]) for column in keys)
        try:
            n_events, n_groups = support[key]
        except KeyError as exc:
            raise AblationComparisonError(f"No independent support for fold key {key}") from exc
        horizon = record.get(
            "experiment_horizon_weeks",
            with_run["manifest"]["parameters"].get("horizon"),
        )
        for metric in metrics:
            direction = "lower_is_better" if "brier" in metric.lower() else "higher_is_better"
            rows.append(
                _metric_row(
                    run=with_run,
                    scope="classification_by_fold",
                    metric=metric,
                    fold=record["fold"],
                    horizon=horizon,
                    with_value=record[f"{metric}_with"],
                    without_value=record[f"{metric}_without"],
                    n_events=n_events,
                    n_groups=n_groups,
                    aggregation_unit="whole_event_test_fold; windows_not_independent_n",
                    direction=direction,
                )
            )
    return rows


def _phase5_event_rows(
    with_run: Mapping[str, Any], without_run: Mapping[str, Any]
) -> list[dict[str, object]]:
    def summarise(run: Mapping[str, Any]) -> pd.DataFrame:
        frame = _table(run["paths"], "event_dimension_oos_predictions.csv").copy()
        required = {
            "fold",
            "target",
            "event_id",
            "observed",
            "predicted",
            "baseline_type_train",
        }
        if missing := required.difference(frame.columns):
            raise AblationComparisonError(f"F5 event predictions lack {sorted(missing)}")
        frame["absolute_error"] = (
            pd.to_numeric(frame["observed"], errors="coerce")
            - pd.to_numeric(frame["predicted"], errors="coerce")
        ).abs()
        frame["baseline_absolute_error"] = (
            pd.to_numeric(frame["observed"], errors="coerce")
            - pd.to_numeric(frame["baseline_type_train"], errors="coerce")
        ).abs()
        if frame[["absolute_error", "baseline_absolute_error"]].isna().any().any():
            raise AblationComparisonError("F5 event predictions contain invalid numeric values")
        if frame.duplicated(["fold", "target", "event_id"]).any():
            raise AblationComparisonError(
                "F5 event table must contain one row per fold/target/original event"
            )
        observed_targets = frozenset(frame["target"].astype(str).unique())
        if observed_targets != F5_EVENT_TARGETS:
            raise AblationComparisonError(
                "F5 event comparison requires exactly peak magnitude, time-to-peak "
                "and duration targets"
            )
        summary = (
            frame.groupby(["fold", "target"], sort=False, as_index=False)
            .agg(
                n_original_events=("event_id", "nunique"),
                mae_event_equal=("absolute_error", "mean"),
                mae_baseline_event_equal=("baseline_absolute_error", "mean"),
            )
        )
        summary["skill_mae_vs_type_climatology"] = np.where(
            summary["mae_baseline_event_equal"] > 0,
            1.0
            - summary["mae_event_equal"] / summary["mae_baseline_event_equal"],
            np.nan,
        )
        return summary

    left = summarise(with_run)
    right = summarise(without_run)
    paired = _paired_frames(
        left,
        right,
        keys=["fold", "target"],
        label="F5 event dimensions",
    )
    rows: list[dict[str, object]] = []
    for record in paired.to_dict(orient="records"):
        left_n = int(record["n_original_events_with"])
        right_n = int(record["n_original_events_without"])
        if left_n != right_n:
            raise AblationComparisonError("Augmentation changed F5 independent event N")
        for metric, direction in (
            ("skill_mae_vs_type_climatology", "higher_is_better"),
            ("mae_event_equal", "lower_is_better"),
        ):
            rows.append(
                _metric_row(
                    run=with_run,
                    scope="event_dimension_by_fold_target",
                    metric=metric,
                    fold=record["fold"],
                    target=record["target"],
                    with_value=record[f"{metric}_with"],
                    without_value=record[f"{metric}_without"],
                    n_events=left_n,
                    n_groups=None,
                    aggregation_unit="independent_event_equal_within_fold",
                    direction=direction,
                )
            )
    return rows


def _phase7_event_and_calibration_rows(
    with_run: Mapping[str, Any], without_run: Mapping[str, Any]
) -> list[dict[str, object]]:
    left_metrics = _table(with_run["paths"], "fold_metrics.csv")
    right_metrics = _table(without_run["paths"], "fold_metrics.csv")
    paired = _paired_frames(left_metrics, right_metrics, keys=["fold"], label="F7 event metrics")
    target_prefix = "skill_mae_"
    targets = sorted(
        column[len(target_prefix) :]
        for column in set(left_metrics.columns).intersection(right_metrics.columns)
        if column.startswith(target_prefix)
    )
    if frozenset(targets) != F7_EVENT_TARGETS:
        raise AblationComparisonError(
            "F7 comparison requires exactly peak magnitude, time-to-peak and "
            "duration event skills"
        )
    by_event = _table(
        with_run["paths"], "event_probabilistic_metrics_by_event.csv"
    )
    required_by_event = {"fold", "target", "event_id"}
    if missing := required_by_event.difference(by_event.columns):
        raise AblationComparisonError(
            f"F7 event-level calibration table lacks {sorted(missing)}"
        )
    if by_event.duplicated(["fold", "target", "event_id"]).any():
        raise AblationComparisonError(
            "F7 calibration must contain one row per fold/target/original event"
        )
    event_counts = (
        by_event.groupby(["fold", "target"], sort=False)["event_id"]
        .nunique()
        .to_dict()
    )
    rows: list[dict[str, object]] = []
    horizon = with_run["manifest"]["parameters"].get("horizon")
    for record in paired.to_dict(orient="records"):
        for target in targets:
            n_column = f"n_test_events_{target}"
            mae_column = f"mae_event_equal_{target}"
            skill_column = f"skill_mae_{target}"
            for column in (n_column, mae_column, skill_column):
                if f"{column}_with" not in record or f"{column}_without" not in record:
                    raise AblationComparisonError(f"F7 event metric missing {column}")
            left_n = int(record[f"{n_column}_with"])
            right_n = int(record[f"{n_column}_without"])
            if left_n != right_n:
                raise AblationComparisonError("Augmentation changed F7 independent event N")
            if event_counts.get((record["fold"], target)) != left_n:
                raise AblationComparisonError(
                    "F7 fold metric event N disagrees with event-level audit table"
                )
            for metric, direction in (
                (skill_column, "higher_is_better"),
                (mae_column, "lower_is_better"),
            ):
                rows.append(
                    _metric_row(
                        run=with_run,
                        scope="event_dimension_by_fold_target",
                        metric=metric.removesuffix(f"_{target}"),
                        fold=record["fold"],
                        target=target,
                        horizon=horizon,
                        with_value=record[f"{metric}_with"],
                        without_value=record[f"{metric}_without"],
                        n_events=left_n,
                        n_groups=None,
                        aggregation_unit="independent_event_equal_within_fold",
                        direction=direction,
                    )
                )

    left_calibration = _table(with_run["paths"], "event_probabilistic_metrics.csv")
    right_calibration = _table(without_run["paths"], "event_probabilistic_metrics.csv")
    calibration = _paired_frames(
        left_calibration,
        right_calibration,
        keys=["fold", "target"],
        label="F7 IC90 calibration",
    )
    for record in calibration.to_dict(orient="records"):
        if record.get("aggregation_unit_with") != "independent_event_equal" or record.get(
            "aggregation_unit_without"
        ) != "independent_event_equal":
            raise AblationComparisonError("F7 calibration is not event-equal")
        if float(record["interval_nominal_coverage_with"]) != float(
            record["interval_nominal_coverage_without"]
        ):
            raise AblationComparisonError("F7 nominal interval coverage differs")
        if not np.isclose(
            float(record["interval_nominal_coverage_with"]), 0.90, atol=1e-12
        ):
            raise AblationComparisonError("F7 calibration table is not IC90")
        left_n = int(record["n_events_with"])
        right_n = int(record["n_events_without"])
        if left_n != right_n:
            raise AblationComparisonError("Augmentation changed F7 calibration event N")
        if event_counts.get((record["fold"], record["target"])) != left_n:
            raise AblationComparisonError(
                "F7 calibration event N disagrees with event-level audit table"
            )
        for metric, direction in (
            ("interval_coverage_event_equal", "descriptive_against_nominal_0.90"),
            ("interval_calibration_error", "closer_to_zero_is_better"),
            ("interval_absolute_calibration_error", "lower_is_better"),
        ):
            rows.append(
                _metric_row(
                    run=with_run,
                    scope="ic90_calibration_by_fold_target",
                    metric=metric,
                    fold=record["fold"],
                    target=record["target"],
                    horizon=horizon,
                    with_value=record[f"{metric}_with"],
                    without_value=record[f"{metric}_without"],
                    n_events=left_n,
                    n_groups=None,
                    aggregation_unit="independent_event_equal_within_fold",
                    direction=direction,
                )
            )
    return rows


def compare_runs(
    with_directory: Path, without_directory: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and build a paired, event-aware augmentation ablation."""

    with_run = _validate_run(with_directory)
    without_run = _validate_run(without_directory)
    _assert_pair_compatible(with_run, without_run)
    rows = _classification_rows(with_run, without_run)
    if with_run["phase"] == 5:
        rows.extend(_phase5_event_rows(with_run, without_run))
    else:
        rows.extend(_phase7_event_and_calibration_rows(with_run, without_run))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise AblationComparisonError("Ablation produced no comparable metrics")
    if int(frame["n_original_events"].min()) <= 0:
        raise AblationComparisonError("Ablation has no independent original events")
    if not frame["augmentation_does_not_change_independent_n"].eq(True).all():
        raise AblationComparisonError("Independent-N invariant failed")
    frame.insert(0, "with_augmentation_run_id", with_run["manifest"]["run_id"])
    frame.insert(1, "without_augmentation_run_id", without_run["manifest"]["run_id"])
    frame.insert(2, "with_augmentation_manifest_sha256", with_run["manifest_sha256"])
    frame.insert(3, "without_augmentation_manifest_sha256", without_run["manifest_sha256"])
    comparison_digest = _json_hash(frame.to_dict(orient="records"))
    manifest = {
        "schema_version": "nino26-augmentation-ablation-v1",
        "created_at": _utc_now(),
        "status": "validated",
        "phase": int(with_run["phase"]),
        "mode": str(with_run["manifest"]["mode"]),
        "model_backend": _backend_identity(with_run),
        "seed": int(with_run["seed"]),
        "augmentation_parameter_allowlist": sorted(
            ALLOWED_AUGMENTATION_PARAMETERS[with_run["phase"]]
        ),
        "scientific_parameters_sha256": _json_hash(_scientific_parameters(with_run)),
        "comparison_rows": int(len(frame)),
        "comparison_rows_sha256": comparison_digest,
        "n_original_events_min": int(frame["n_original_events"].min()),
        "n_original_events_max": int(frame["n_original_events"].max()),
        "augmentation_does_not_change_independent_n": True,
        "independent_unit_policy": (
            "independent ENSO event or whole-event test fold; synthetic rows and "
            "weekly/sequence windows never increase N"
        ),
        "runs": {
            "with_augmentation": {
                "run_id": with_run["manifest"]["run_id"],
                "directory": str(with_run["directory"]),
                "run_manifest_sha256": with_run["manifest_sha256"],
                "tables_manifest_sha256": with_run["tables_manifest_sha256"],
                "parameters": with_run["manifest"]["parameters"],
                "synthetic_lineage_rows": _synthetic_rows(with_run),
            },
            "without_augmentation": {
                "run_id": without_run["manifest"]["run_id"],
                "directory": str(without_run["directory"]),
                "run_manifest_sha256": without_run["manifest_sha256"],
                "tables_manifest_sha256": without_run["tables_manifest_sha256"],
                "parameters": without_run["manifest"]["parameters"],
                "synthetic_lineage_rows": _synthetic_rows(without_run),
            },
        },
    }
    return frame, manifest


def _output_directory(raw: Path) -> Path:
    path = raw if raw.is_absolute() else ROOT / raw
    path = path.resolve(strict=False)
    audit_root = AUDIT_ROOT.resolve(strict=False)
    if path == audit_root or audit_root not in path.parents:
        raise AblationComparisonError(
            f"--output-dir must be a named subdirectory below {AUDIT_ROOT}"
        )
    return path


def _write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_outputs(frame: pd.DataFrame, manifest: dict[str, Any], output_dir: Path) -> None:
    """Write a comparison to a new/empty audit directory without overwrites."""

    output_dir = _output_directory(output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise AblationComparisonError(f"Output path is not a directory: {output_dir}")
        existing = list(output_dir.iterdir())
        if existing:
            raise AblationComparisonError(
                f"Output directory must be empty; preserving existing files: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / CSV_NAME
    manifest_path = output_dir / MANIFEST_NAME
    csv_payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    manifest = dict(manifest)
    manifest.update(
        {
            "status": "complete",
            "output_directory": str(output_dir),
            "outputs": {
                CSV_NAME: {
                    "rows": int(len(frame)),
                    "sha256": hashlib.sha256(csv_payload).hexdigest(),
                }
            },
        }
    )
    _write_atomic(csv_path, csv_payload)
    _write_atomic(
        manifest_path,
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=str,
        ).encode("utf-8"),
    )
    if sha256_file(csv_path) != manifest["outputs"][CSV_NAME]["sha256"]:
        raise RuntimeError("Post-write CSV hash verification failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-augmentation",
        required=True,
        type=Path,
        help="complete augmentation-ON F5 or F7 ArtifactRun directory",
    )
    parser.add_argument(
        "--without-augmentation",
        required=True,
        type=Path,
        help="complete augmentation-OFF paired ArtifactRun directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="named destination below data/audit/augmentation_ablation",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write outputs; default validates only and performs no filesystem mutation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = _output_directory(args.output_dir)
    frame, manifest = compare_runs(args.with_augmentation, args.without_augmentation)
    summary = {
        "status": "validated",
        "write": bool(args.write),
        "phase": manifest["phase"],
        "mode": manifest["mode"],
        "rows": len(frame),
        "output_directory": str(output_dir),
        "augmentation_does_not_change_independent_n": True,
    }
    if args.write:
        write_outputs(frame, manifest, output_dir)
        summary["status"] = "written"
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AblationComparisonError as exc:
        print(f"augmentation ablation refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
