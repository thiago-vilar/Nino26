#!/usr/bin/env python3
"""Merge strictly compatible F6 shards and compute a pooled OOS field gate.

The merger is an inferential boundary, not a CSV concatenator.  Every accepted
source must be a complete, non-overridden official shard from the same data and
model contract.  Skills are recomputed from the shared out-of-sample rows,
first per native pixel and then for the area-weighted Brazil field.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nino_brasil.artifacts import (  # noqa: E402
    start_artifact_run,
    validate_artifact_run,
)
from nino_brasil.targets.chirps_native import (  # noqa: E402
    native_pixel_table,
    validate_native_target,
)
from nino_brasil.data.phase2_master import PHYSICAL_COLUMNS  # noqa: E402

RUNS = ROOT / "data" / "processed" / "runs" / "official" / "fase6"
TARGET = ROOT / "data" / "processed" / "zarr" / "features" / "chirps_native_weekly_targets.zarr"

BASE_GROUP_COLUMNS = ["target_variable", "target_transform", "model"]
GROUP_COLUMNS = [*BASE_GROUP_COLUMNS, "target_units", "condition", "lag_weeks"]
PIXEL_KEY = [*GROUP_COLUMNS, "pixel_id"]
FOLD_KEY = [*BASE_GROUP_COLUMNS, "fold", "pixel_id", "condition", "lag_weeks"]
PREDICTION_KEY = [
    *BASE_GROUP_COLUMNS,
    "fold",
    "time",
    "pixel_id",
    "condition",
    "lag_weeks",
]
IMPORTANCE_KEY = [
    *BASE_GROUP_COLUMNS,
    "fold",
    "pixel_id",
    "condition",
    "lag_weeks",
    "variable",
]

BASELINE_COLUMNS: dict[str, str] = {
    "climatology_train_mean": "baseline_climatology_train_mean",
    "climatology_week_of_year_train": "baseline_climatology_week_of_year_train",
    "persistence_week": "baseline_persistence",
}
PHASE4_BASELINE_NAME = "phase4_statistical_ridge"
PHASE4_BASELINE_COLUMN = "baseline_phase4_statistical_ridge"

NEW_NATIVE_COLUMNS = {
    "observed_native_value",
    "predicted_native_value",
    "baseline_persistence_native_value",
    "target_units",
}
LEGACY_NATIVE_COLUMNS = {
    "observed_raw_mm",
    "predicted_raw_mm",
    "baseline_persistence_raw_mm",
}


@dataclass
class SourceShard:
    directory: Path
    manifest: dict[str, Any]
    fingerprint: str
    fingerprint_payload: dict[str, Any]
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    inventory: pd.DataFrame
    importance: pd.DataFrame
    predictor_contract: pd.DataFrame
    native_schema: bool
    phase4_comparator: bool
    source_files: tuple[Path, ...]


@dataclass
class MergeProducts:
    fold_metrics: pd.DataFrame
    pooled_pixel_metrics: pd.DataFrame
    predictions: pd.DataFrame
    gate: pd.DataFrame
    pixel_inventory: pd.DataFrame
    pixel_variable_importance: pd.DataFrame
    predictor_contract: pd.DataFrame
    source_files: list[Path]
    source_runs: list[str]
    fingerprint: str
    native_schema: bool
    phase4_comparator: bool
    coverage: float


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "sim"}


def _single_value(frame: pd.DataFrame, column: str, *, label: str) -> str:
    if column not in frame:
        raise KeyError(f"{label}: coluna obrigatoria ausente: {column}")
    values = frame[column].dropna().astype(str).unique()
    if len(values) != 1 or not values[0].strip():
        raise ValueError(f"{label}: {column} deve ter exatamente um valor nao vazio; encontrei {values!r}")
    return str(values[0])


def _command_option(command: str, option: str, *, default: str | None = None) -> str | None:
    try:
        parts = shlex.split(str(command), posix=False)
    except ValueError:
        parts = str(command).split()
    if option not in parts:
        return default
    position = parts.index(option)
    return parts[position + 1] if position + 1 < len(parts) else "__flag_without_value__"


def _record_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep content identity while ignoring timestamps and file sizes."""

    return {
        "path": record.get("path"),
        "exists": record.get("exists"),
        "is_directory": record.get("is_directory"),
        "sha256": record.get("sha256"),
        "tree_sha256": record.get("tree_sha256"),
    }


def manifest_fingerprint_payload(
    manifest: Mapping[str, Any],
    *,
    target_units: str,
    native_schema: bool,
    phase4_comparator: bool,
) -> dict[str, Any]:
    """Return the shard-invariant scientific/configuration fingerprint."""

    parameters = dict(manifest.get("parameters", {}))
    for shard_specific in ("pixel_start", "pixel_stop", "n_pixels"):
        parameters.pop(shard_specific, None)
    git = dict(manifest.get("git", {}))
    packages = dict(manifest.get("environment", {}).get("packages", {}))
    relevant_packages = {
        name: packages.get(name)
        for name in ("numpy", "pandas", "scikit-learn", "xgboost", "xarray")
    }
    return {
        "schema_version": manifest.get("schema_version"),
        "phase": manifest.get("phase"),
        "mode": manifest.get("mode"),
        "seed": manifest.get("seed"),
        "parameters": parameters,
        "git_commit": git.get("commit"),
        "git_status_sha256": git.get("status_sha256"),
        "packages": relevant_packages,
        "configs": [_record_identity(record) for record in manifest.get("configs", [])],
        "inputs": [_record_identity(record) for record in manifest.get("inputs", [])],
        # Older runner manifests did not record this hyperparameter.  The CLI
        # value is therefore included as a migration bridge.
        "n_estimators": _command_option(
            str(manifest.get("command", "")), "--n-estimators", default="250"
        ),
        "native_value_schema": bool(native_schema),
        "target_units": target_units,
        "phase4_statistical_comparator": bool(phase4_comparator),
    }


def validate_source_manifest(directory: Path, manifest: Mapping[str, Any]) -> None:
    """Reject incomplete, exploratory, overridden or merged runs as shards."""

    if int(manifest.get("phase", -1)) != 6:
        raise ValueError(f"{directory}: manifest nao pertence a Fase 6.")
    if manifest.get("mode") != "official":
        raise ValueError(f"{directory}: somente shards oficiais podem entrar no gate.")
    if manifest.get("status") != "complete":
        raise ValueError(f"{directory}: status deve ser complete, recebido {manifest.get('status')!r}.")
    parameters = dict(manifest.get("parameters", {}))
    if parameters.get("role") == "merge_pixel_shards_and_field_gate":
        raise ValueError(f"{directory}: um merge anterior nao pode ser usado como shard.")
    override_fields = [
        name
        for name, value in parameters.items()
        if "override" in str(name).lower() and _as_bool(value)
    ]
    command = str(manifest.get("command", ""))
    if override_fields or "--research-override-gate" in command:
        raise ValueError(
            f"{directory}: run oficial usa override de pesquisa ({override_fields or ['command']})."
        )
    if "--no-predictions" in command:
        raise ValueError(f"{directory}: shard oficial sem contrato de predicoes nao e agregavel.")
    recorded_id = str(manifest.get("run_id", ""))
    if recorded_id and recorded_id != directory.name:
        raise ValueError(
            f"{directory}: run_id do manifest ({recorded_id}) difere do diretorio ({directory.name})."
        )


def _fill_contract_columns(
    frame: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    model: str | None,
    label: str,
) -> pd.DataFrame:
    out = frame.copy()
    parameters = dict(manifest.get("parameters", {}))
    defaults = {
        "model": parameters.get("model"),
        "target_variable": parameters.get("target_variable"),
        "target_transform": parameters.get("target_transform"),
    }
    for column, default in defaults.items():
        if column not in out:
            if default is None:
                raise KeyError(f"{label}: {column} ausente na tabela e no manifest.")
            out[column] = str(default)
        value = _single_value(out, column, label=label)
        if default is not None and value != str(default):
            raise ValueError(
                f"{label}: {column}={value!r} diverge do manifest ({default!r})."
            )
    if model is not None and _single_value(out, "model", label=label) != model:
        raise ValueError(f"{label}: model nao corresponde a --model={model}.")
    return out


def load_source_shard(directory: Path, *, model: str | None = None) -> SourceShard:
    """Load one source shard after strict provenance and schema validation."""

    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Shard F6 inexistente ou nao e diretorio: {directory}")
    paths = {
        "manifest": directory / "run_manifest.json",
        "metrics": directory / "tables" / "pixel_metrics.csv",
        "predictions": directory / "tables" / "pixel_oos_predictions.csv",
        "inventory": directory / "tables" / "pixel_shard_inventory.csv",
        "importance": directory / "tables" / "pixel_variable_importance.csv",
        "predictor_contract": directory / "tables" / "predictor_contract.csv",
        "tables_manifest": directory / "tables_manifest.csv",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{directory}: arquivos obrigatorios ausentes: {missing}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    validate_source_manifest(directory, manifest)
    integrity = validate_artifact_run(directory)
    if not integrity.empty:
        raise ValueError(
            f"{directory}: integridade do artifact run falhou:\n"
            + integrity.to_string(index=False)
        )

    metrics = _fill_contract_columns(
        pd.read_csv(paths["metrics"]), manifest, model=model, label=f"{directory}: metrics"
    )
    predictions = _fill_contract_columns(
        pd.read_csv(paths["predictions"]),
        manifest,
        model=model,
        label=f"{directory}: predictions",
    )
    inventory = pd.read_csv(paths["inventory"])
    importance = _fill_contract_columns(
        pd.read_csv(paths["importance"]),
        manifest,
        model=model,
        label=f"{directory}: importance",
    )
    predictor_contract = pd.read_csv(paths["predictor_contract"])
    required_inventory = {"pixel_id", "lat", "lon", "brazil_fraction"}
    if missing_inventory := required_inventory.difference(inventory.columns):
        raise KeyError(f"{directory}: inventario sem {sorted(missing_inventory)}")
    if inventory.empty or metrics.empty or predictions.empty or importance.empty:
        raise ValueError(
            f"{directory}: metrics, predictions, importance e inventory devem ser nao vazios."
        )
    if "variable" not in predictor_contract or predictor_contract["variable"].isna().any():
        raise KeyError(f"{directory}: predictor_contract sem variable completa.")
    predictor_names = predictor_contract["variable"].astype(str).tolist()
    expected_count = len(PHYSICAL_COLUMNS)
    if len(predictor_names) != expected_count or len(set(predictor_names)) != expected_count:
        raise ValueError(f"{directory}: predictor_contract deve conter {expected_count} variaveis unicas.")

    for frame in (metrics, predictions, inventory, importance):
        if "pixel_id" not in frame:
            raise KeyError(f"{directory}: tabela sem pixel_id.")
        frame["pixel_id"] = frame["pixel_id"].astype(str)
    if inventory["pixel_id"].duplicated().any():
        raise ValueError(f"{directory}: pixel_id duplicado dentro do inventario do shard.")
    inventory_ids = set(inventory["pixel_id"])
    for label, frame in (("metrics", metrics), ("predictions", predictions)):
        table_ids = set(frame["pixel_id"])
        if table_ids != inventory_ids:
            raise ValueError(
                f"{directory}: conjunto de pixels de {label} difere do inventario; "
                f"faltam={sorted(inventory_ids - table_ids)[:5]}, "
                f"extras={sorted(table_ids - inventory_ids)[:5]}."
            )
    required_importance = {
        "fold",
        "pixel_id",
        "condition",
        "lag_weeks",
        "variable",
        "importance_gain_train",
        "delta_rmse_permutation_oos",
        "n_permutation_repeats",
    }
    if missing_importance := required_importance.difference(importance.columns):
        raise KeyError(f"{directory}: importance sem {sorted(missing_importance)}")
    importance["lag_weeks"] = pd.to_numeric(
        importance["lag_weeks"], errors="raise"
    ).astype(int)
    if importance.duplicated(IMPORTANCE_KEY).any():
        raise ValueError(f"{directory}: importance duplicada na chave canonica.")
    importance_ids = set(importance["pixel_id"])
    if importance_ids != inventory_ids:
        raise ValueError(f"{directory}: pixels de importance diferem do inventario.")
    importance_groups = importance.groupby(
        [*BASE_GROUP_COLUMNS, "fold", "pixel_id", "condition", "lag_weeks"],
        sort=False,
        dropna=False,
    )["variable"].agg(lambda values: set(values.astype(str)))
    expected_predictors = set(predictor_names)
    if not all(variables == expected_predictors for variables in importance_groups):
        raise ValueError(
            f"{directory}: importance nao cobre as {expected_count} variaveis em cada fold/pixel/condicao/lag."
        )

    required_predictions = {
        "fold",
        "time",
        "pixel_id",
        "condition",
        "lag_weeks",
        "observed",
        "predicted",
        *BASELINE_COLUMNS.values(),
    }
    if missing_predictions := required_predictions.difference(predictions.columns):
        raise KeyError(f"{directory}: predicoes sem {sorted(missing_predictions)}")
    predictions["time"] = pd.to_datetime(predictions["time"], errors="raise")
    predictions["lag_weeks"] = pd.to_numeric(
        predictions["lag_weeks"], errors="raise"
    ).astype(int)
    metrics["lag_weeks"] = pd.to_numeric(metrics["lag_weeks"], errors="raise").astype(int)

    native_columns_present = NEW_NATIVE_COLUMNS.intersection(predictions.columns)
    native_schema = bool(native_columns_present)
    if native_schema:
        missing_native = NEW_NATIVE_COLUMNS.difference(predictions.columns)
        if missing_native:
            raise KeyError(
                f"{directory}: schema native_value parcial; faltam {sorted(missing_native)}"
            )
        target_units = _single_value(predictions, "target_units", label=f"{directory}: predictions")
        for column in NEW_NATIVE_COLUMNS.difference({"target_units"}):
            values = pd.to_numeric(predictions[column], errors="coerce")
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f"{directory}: {column} contem valores OOS nao finitos.")
    else:
        missing_legacy = LEGACY_NATIVE_COLUMNS.difference(predictions.columns)
        if missing_legacy:
            raise KeyError(
                f"{directory}: nem native_value nem legado raw_mm completo; faltam "
                f"{sorted(missing_legacy)}"
            )
        target_units = "mm (legacy raw_mm schema)"
        predictions["target_units"] = target_units

    phase4_comparator = PHASE4_BASELINE_COLUMN in predictions
    parameters = dict(manifest.get("parameters", {}))
    phase4_required = native_schema or _as_bool(
        parameters.get("phase4_statistical_baseline_required", False)
    )
    if phase4_required and not phase4_comparator:
        raise KeyError(
            f"{directory}: run novo exige {PHASE4_BASELINE_COLUMN} nas predicoes OOS."
        )

    predictions["source_run_id"] = directory.name
    metrics["source_run_id"] = directory.name
    inventory["source_run_id"] = directory.name
    importance["source_run_id"] = directory.name
    payload = manifest_fingerprint_payload(
        manifest,
        target_units=target_units,
        native_schema=native_schema,
        phase4_comparator=phase4_comparator,
    )
    return SourceShard(
        directory=directory,
        manifest=dict(manifest),
        fingerprint=_json_hash(payload),
        fingerprint_payload=payload,
        metrics=metrics,
        predictions=predictions,
        inventory=inventory,
        importance=importance,
        predictor_contract=predictor_contract,
        native_schema=native_schema,
        phase4_comparator=phase4_comparator,
        source_files=tuple(paths.values()),
    )


def assert_homogeneous_shards(shards: Sequence[SourceShard]) -> str:
    if not shards:
        raise FileNotFoundError("Nenhum shard F6 completo foi encontrado.")
    fingerprints = {shard.fingerprint for shard in shards}
    if len(fingerprints) != 1:
        summary = "\n".join(
            f"- {shard.directory.name}: {shard.fingerprint}"
            for shard in shards
        )
        raise ValueError(f"Shards F6 com fingerprints cientificos/configuracionais distintos:\n{summary}")
    native = {shard.native_schema for shard in shards}
    phase4 = {shard.phase4_comparator for shard in shards}
    if len(native) != 1 or len(phase4) != 1:
        raise ValueError("Shards misturam geracoes de schema native_value/F4 comparator.")
    return shards[0].fingerprint


def expected_native_pixels(dataset: xr.Dataset) -> pd.DataFrame:
    pixels = native_pixel_table(dataset.latitude.values, dataset.longitude.values)
    if "pixel_id" in dataset:
        pixels["pixel_id"] = np.asarray(dataset["pixel_id"].values).ravel().astype(np.int64)
    if "brazil_fraction" not in dataset:
        raise KeyError("Alvo CHIRPS sem brazil_fraction.")
    pixels["brazil_fraction"] = np.asarray(dataset["brazil_fraction"].values).ravel()
    pixels = pixels.loc[pd.to_numeric(pixels["brazil_fraction"], errors="coerce").gt(0)].copy()
    pixels["pixel_id"] = pixels["pixel_id"].astype(str)
    return pixels.reset_index(drop=True)


def _validate_pixel_inventory(
    inventory: pd.DataFrame,
    expected_pixels: pd.DataFrame,
) -> pd.DataFrame:
    inventory = inventory.copy()
    inventory["pixel_id"] = inventory["pixel_id"].astype(str)
    expected = expected_pixels.copy()
    expected["pixel_id"] = expected["pixel_id"].astype(str)
    if inventory["pixel_id"].duplicated().any():
        duplicate = inventory.loc[
            inventory["pixel_id"].duplicated(keep=False), ["pixel_id", "source_run_id"]
        ].head()
        raise ValueError(
            "Inventarios de shards sobrepostos:\n" + duplicate.to_string(index=False)
        )
    actual_ids = set(inventory["pixel_id"])
    expected_ids = set(expected["pixel_id"])
    if actual_ids != expected_ids:
        raise ValueError(
            "Uniao dos inventarios nao e exatamente o grid Brasil esperado; "
            f"faltam={sorted(expected_ids - actual_ids)[:10]}, "
            f"extras={sorted(actual_ids - expected_ids)[:10]}."
        )
    columns = ["pixel_id", "lat", "lon", "brazil_fraction"]
    actual = inventory[columns].sort_values("pixel_id").reset_index(drop=True)
    reference = expected[columns].sort_values("pixel_id").reset_index(drop=True)
    for column in ("lat", "lon", "brazil_fraction"):
        left = pd.to_numeric(actual[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(reference[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(left)) or not np.allclose(left, right, rtol=0.0, atol=1e-7):
            raise ValueError(f"Inventario diverge do alvo CHIRPS em {column}.")
    meta = reference.copy()
    meta["area_weight"] = (
        np.cos(np.deg2rad(pd.to_numeric(meta["lat"], errors="raise")))
        * pd.to_numeric(meta["brazil_fraction"], errors="raise")
    )
    if not np.isfinite(meta["area_weight"]).all() or not meta["area_weight"].gt(0).all():
        raise ValueError("Pesos de area dos pixels Brasil devem ser finitos e positivos.")
    return meta


def _required_baselines(*, phase4_comparator: bool) -> dict[str, str]:
    baselines = dict(BASELINE_COLUMNS)
    if phase4_comparator:
        baselines[PHASE4_BASELINE_NAME] = PHASE4_BASELINE_COLUMN
    return baselines


def validate_prediction_coverage(
    predictions: pd.DataFrame,
    *,
    expected_ids: set[str],
    declared_conditions: Sequence[str],
    declared_lags: Sequence[int],
    phase4_comparator: bool,
) -> None:
    """Require every declared group and exact/evaluable native-pixel coverage."""

    predictions["pixel_id"] = predictions["pixel_id"].astype(str)
    if predictions.duplicated(PREDICTION_KEY).any():
        duplicate = predictions.loc[
            predictions.duplicated(PREDICTION_KEY, keep=False),
            PREDICTION_KEY + ["source_run_id"],
        ].head()
        raise ValueError(f"Predicoes OOS F6 sobrepostas:\n{duplicate.to_string(index=False)}")
    base_values = predictions[BASE_GROUP_COLUMNS].drop_duplicates()
    if len(base_values) != 1:
        raise ValueError("Merge deve conter exatamente um target/transform/model.")
    expected_groups = {
        (str(condition), int(lag))
        for condition in declared_conditions
        for lag in declared_lags
    }
    actual_groups = set(
        zip(predictions["condition"].astype(str), predictions["lag_weeks"].astype(int))
    )
    if actual_groups != expected_groups:
        raise ValueError(
            "Condicoes/lags OOS diferem do contrato declarado; "
            f"faltam={sorted(expected_groups - actual_groups)[:10]}, "
            f"extras={sorted(actual_groups - expected_groups)[:10]}."
        )
    baselines = _required_baselines(phase4_comparator=phase4_comparator)
    numeric_columns = ["observed", "predicted", *baselines.values()]
    for keys, group in predictions.groupby(
        [*BASE_GROUP_COLUMNS, "condition", "lag_weeks"], sort=False, dropna=False
    ):
        actual_ids = set(group["pixel_id"])
        if actual_ids != expected_ids:
            raise ValueError(
                f"Grupo {keys} nao cobre exatamente os pixels esperados; "
                f"faltam={sorted(expected_ids - actual_ids)[:10]}, "
                f"extras={sorted(actual_ids - expected_ids)[:10]}."
            )
        numeric = group[numeric_columns].apply(pd.to_numeric, errors="coerce")
        finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
        if not bool(finite.all()):
            bad = group.loc[~finite, ["pixel_id", "time", "fold"]].head()
            raise ValueError(
                f"Grupo {keys} possui {int((~finite).sum())} linhas sem avaliacao comum "
                f"modelo+baselines:\n{bad.to_string(index=False)}"
            )
        counts = group.groupby("pixel_id", sort=False).size()
        if not counts.index.astype(str).isin(expected_ids).all() or counts.le(0).any():
            raise ValueError(f"Grupo {keys} contem pixel sem amostra OOS avaliavel.")


def pool_oos_pixel_metrics(
    predictions: pd.DataFrame,
    pixel_meta: pd.DataFrame,
    *,
    native_schema: bool,
    phase4_comparator: bool,
) -> pd.DataFrame:
    """Pool all OOS weeks/folds before computing one metric row per pixel."""

    meta = pixel_meta.set_index(pixel_meta["pixel_id"].astype(str))
    baselines = _required_baselines(phase4_comparator=phase4_comparator)
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(PIXEL_KEY, sort=False, dropna=False):
        values = dict(zip(PIXEL_KEY, keys))
        observed = pd.to_numeric(group["observed"], errors="coerce").to_numpy(dtype=float)
        predicted = pd.to_numeric(group["predicted"], errors="coerce").to_numpy(dtype=float)
        rmse_model = float(np.sqrt(np.mean((predicted - observed) ** 2)))
        rmse_baselines: dict[str, float] = {}
        skills: dict[str, float] = {}
        for name, column in baselines.items():
            baseline = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            rmse = float(np.sqrt(np.mean((baseline - observed) ** 2)))
            rmse_baselines[name] = rmse
            skills[name] = 1.0 - rmse_model / rmse if rmse > 0 else np.nan
        finite_skills = np.asarray(list(skills.values()), dtype=float)
        available_all = bool(np.isfinite(finite_skills).all())
        gate_eligible = bool(native_schema and phase4_comparator and available_all)
        pixel_id = str(values["pixel_id"])
        pixel = meta.loc[pixel_id]
        row: dict[str, Any] = {
            **values,
            "lat": float(pixel["lat"]),
            "lon": float(pixel["lon"]),
            "brazil_fraction": float(pixel["brazil_fraction"]),
            "area_weight": float(pixel["area_weight"]),
            "n_oos_rows": int(len(group)),
            "n_oos_folds": int(group["fold"].nunique()),
            "rmse_model_oos": rmse_model,
            "minimum_skill_vs_required_baselines": (
                float(np.min(finite_skills)) if available_all else np.nan
            ),
            "all_required_baselines_finite": available_all,
            "native_value_schema": bool(native_schema),
            "phase4_comparator_available": bool(phase4_comparator),
            "gate_eligible": gate_eligible,
            "gate_pass": bool(gate_eligible and np.all(finite_skills > 0)),
        }
        for name in (*BASELINE_COLUMNS, PHASE4_BASELINE_NAME):
            row[f"rmse_{name}"] = rmse_baselines.get(name, np.nan)
            row[f"skill_vs_{name}"] = skills.get(name, np.nan)
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty or result.duplicated(PIXEL_KEY).any():
        raise ValueError("Metricas OOS pooled por pixel vazias ou duplicadas.")
    return result


def build_field_gate(
    predictions: pd.DataFrame,
    pooled: pd.DataFrame,
    *,
    expected_ids: set[str],
    native_schema: bool,
    phase4_comparator: bool,
) -> pd.DataFrame:
    """Compute paired area-weighted field errors from all OOS pixel-weeks."""

    baselines = _required_baselines(phase4_comparator=phase4_comparator)
    area_map = pooled.drop_duplicates("pixel_id").set_index("pixel_id")["area_weight"]
    rows: list[dict[str, Any]] = []
    group_columns = [*GROUP_COLUMNS]
    for keys, group in predictions.groupby(group_columns, sort=False, dropna=False):
        values = dict(zip(group_columns, keys))
        pixel_ids = group["pixel_id"].astype(str)
        actual_ids = set(pixel_ids)
        coverage = len(actual_ids.intersection(expected_ids)) / len(expected_ids)
        exact_coverage = actual_ids == expected_ids
        weights = pixel_ids.map(area_map).to_numpy(dtype=float)
        observed = pd.to_numeric(group["observed"], errors="coerce").to_numpy(dtype=float)
        predicted = pd.to_numeric(group["predicted"], errors="coerce").to_numpy(dtype=float)
        denominator = float(np.sum(weights))
        rmse_model = float(np.sqrt(np.sum(weights * (predicted - observed) ** 2) / denominator))
        rmse_baselines: dict[str, float] = {}
        skills: dict[str, float] = {}
        for name, column in baselines.items():
            baseline = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            rmse = float(np.sqrt(np.sum(weights * (baseline - observed) ** 2) / denominator))
            rmse_baselines[name] = rmse
            skills[name] = 1.0 - rmse_model / rmse if rmse > 0 else np.nan
        pixel_group = pooled.loc[
            np.logical_and.reduce(
                [
                    pooled[column].astype(str).eq(str(values[column])).to_numpy()
                    for column in group_columns
                ]
            )
        ]
        pixel_set_exact = set(pixel_group["pixel_id"].astype(str)) == expected_ids
        all_finite = bool(np.isfinite(np.asarray(list(skills.values()), dtype=float)).all())
        positive_fraction = float(pixel_group["gate_pass"].astype(bool).mean())
        weighted_pixel_skill = float(
            np.average(
                pixel_group["minimum_skill_vs_required_baselines"].to_numpy(dtype=float),
                weights=pixel_group["area_weight"].to_numpy(dtype=float),
            )
        )
        gate_eligible = bool(
            native_schema
            and phase4_comparator
            and exact_coverage
            and pixel_set_exact
            and all_finite
            and pixel_group["gate_eligible"].astype(bool).all()
        )
        row: dict[str, Any] = {
            **values,
            "n_pixels": int(pixel_group["pixel_id"].nunique()),
            "n_oos_pixel_weeks": int(len(group)),
            "pixel_coverage_fraction": float(coverage),
            "pixel_set_exact": bool(exact_coverage and pixel_set_exact),
            "evaluable_oos_fraction": 1.0,
            "field_rmse_model_oos": rmse_model,
            "area_weighted_mean_minimum_pixel_skill": weighted_pixel_skill,
            # Backward-compatible alias, now computed from one pooled row/pixel.
            "area_weighted_mean_skill": weighted_pixel_skill,
            "fraction_pixels_positive_skill": positive_fraction,
            "native_value_schema": bool(native_schema),
            "phase4_comparator_available": bool(phase4_comparator),
            "gate_eligible": gate_eligible,
        }
        for name in (*BASELINE_COLUMNS, PHASE4_BASELINE_NAME):
            row[f"field_rmse_{name}"] = rmse_baselines.get(name, np.nan)
            row[f"field_skill_vs_{name}"] = skills.get(name, np.nan)
        row["gate_pass"] = bool(
            gate_eligible
            and all(value > 0 for value in skills.values())
            and positive_fraction > 0.5
        )
        rows.append(row)
    gate = pd.DataFrame(rows)
    if gate.empty or gate.duplicated(group_columns).any():
        raise ValueError("Gate de campo vazio ou com chaves duplicadas.")
    return gate


def prepare_merge_products(
    shards: Sequence[SourceShard],
    *,
    target_path: Path,
) -> tuple[MergeProducts, str]:
    fingerprint = assert_homogeneous_shards(shards)
    native_schema = shards[0].native_schema
    phase4_comparator = shards[0].phase4_comparator
    fold_metrics = pd.concat([shard.metrics for shard in shards], ignore_index=True)
    predictions = pd.concat([shard.predictions for shard in shards], ignore_index=True)
    inventory = pd.concat([shard.inventory for shard in shards], ignore_index=True)
    importance = pd.concat([shard.importance for shard in shards], ignore_index=True)
    predictor_contract = shards[0].predictor_contract.copy()
    if fold_metrics.duplicated(FOLD_KEY).any():
        duplicate = fold_metrics.loc[
            fold_metrics.duplicated(FOLD_KEY, keep=False), FOLD_KEY + ["source_run_id"]
        ].head()
        raise ValueError(f"Metricas fold/pixel F6 sobrepostas:\n{duplicate.to_string(index=False)}")
    if importance.duplicated(IMPORTANCE_KEY).any():
        duplicate = importance.loc[
            importance.duplicated(IMPORTANCE_KEY, keep=False),
            IMPORTANCE_KEY + ["source_run_id"],
        ].head()
        raise ValueError(
            "Importancias fold/pixel F6 sobrepostas:\n" + duplicate.to_string(index=False)
        )
    expected_fold_groups = set(
        map(tuple, fold_metrics[FOLD_KEY].astype(str).drop_duplicates().to_numpy())
    )
    importance_fold_groups = set(
        map(
            tuple,
            importance[[*BASE_GROUP_COLUMNS, "fold", "pixel_id", "condition", "lag_weeks"]]
            .astype(str)
            .drop_duplicates()
            .to_numpy(),
        )
    )
    if importance_fold_groups != expected_fold_groups:
        raise ValueError(
            "Importancia F6 nao cobre exatamente os mesmos grupos fold/pixel das metricas."
        )

    target = xr.open_zarr(target_path, consolidated=None)
    try:
        validation = validate_native_target(target)
        if not validation.valid:
            raise ValueError(f"Alvo CHIRPS invalido: {validation.errors}")
        expected_pixels = expected_native_pixels(target)
    finally:
        target.close()
    pixel_meta = _validate_pixel_inventory(inventory, expected_pixels)
    expected_ids = set(pixel_meta["pixel_id"].astype(str))

    parameters = dict(shards[0].manifest.get("parameters", {}))
    declared_conditions = tuple(map(str, parameters.get("conditions", ())))
    declared_lags = tuple(int(value) for value in parameters.get("lags", ()))
    if not declared_conditions or not declared_lags:
        raise ValueError("Manifest do shard deve declarar conditions e lags completos.")
    validate_prediction_coverage(
        predictions,
        expected_ids=expected_ids,
        declared_conditions=declared_conditions,
        declared_lags=declared_lags,
        phase4_comparator=phase4_comparator,
    )
    pooled = pool_oos_pixel_metrics(
        predictions,
        pixel_meta,
        native_schema=native_schema,
        phase4_comparator=phase4_comparator,
    )
    pooled_groups = pooled.groupby(
        [*BASE_GROUP_COLUMNS, "condition", "lag_weeks"], dropna=False
    )["pixel_id"].agg(lambda values: set(values.astype(str)))
    if not all(value == expected_ids for value in pooled_groups):
        raise ValueError("Metricas pooled nao preservaram a cobertura exata por grupo.")
    gate = build_field_gate(
        predictions,
        pooled,
        expected_ids=expected_ids,
        native_schema=native_schema,
        phase4_comparator=phase4_comparator,
    )
    source_files = [path for shard in shards for path in shard.source_files]
    products = MergeProducts(
        fold_metrics=fold_metrics,
        pooled_pixel_metrics=pooled,
        predictions=predictions,
        gate=gate,
        pixel_inventory=pixel_meta,
        pixel_variable_importance=importance,
        predictor_contract=predictor_contract,
        source_files=source_files,
        source_runs=[shard.directory.name for shard in shards],
        fingerprint=fingerprint,
        native_schema=native_schema,
        phase4_comparator=phase4_comparator,
        coverage=1.0,
    )
    return products, validation.grid_hash


def write_merge_artifacts(run: Any, products: MergeProducts) -> None:
    """Write/finalize atomically at the run-status level; failures stay failed."""

    try:
        run.write_table(
            "pixel_fold_metrics",
            products.fold_metrics,
            description="Metricas originais por fold/pixel dos shards F6, preservadas para auditoria.",
            methods={"target_interpolation": False, "role": "source fold diagnostics"},
            primary_keys=tuple(FOLD_KEY),
        )
        run.write_table(
            "pixel_metrics",
            products.pooled_pixel_metrics,
            description="Metricas OOS recomputadas apos combinar todos os folds, uma linha por pixel.",
            methods={
                "pooling": "all paired OOS rows before RMSE",
                "three_simple_baselines_required": True,
                "phase4_statistical_ridge_required_for_new_gate": True,
            },
            primary_keys=tuple(PIXEL_KEY),
        )
        run.write_table(
            "pixel_oos_predictions",
            products.predictions,
            description="Predicoes F6 OOS alinhaveis por alvo/modelo/pixel/semana/condicao/lag.",
            methods={
                "target_interpolation": False,
                "comparison_role": "paired F8 and pooled F6 gate",
                "native_value_schema": products.native_schema,
            },
            primary_keys=tuple(PREDICTION_KEY),
        )
        run.write_table(
            "field_gate",
            products.gate,
            description=(
                "Gate pareado area-ponderado: pixels exatos, OOS integral, tres baselines "
                "simples e comparador ridge F4 no schema novo."
            ),
            methods={
                "coverage_required": 1.0,
                "evaluable_oos_required": 1.0,
                "positive_pixel_fraction_required_strictly_greater_than": 0.5,
                "field_metric": "paired area-weighted RMSE over all OOS pixel-weeks",
            },
            primary_keys=tuple(GROUP_COLUMNS),
        )
        run.write_table(
            "native_pixel_inventory",
            products.pixel_inventory,
            description="Inventario exato de pixels nativos CHIRPS no Brasil e pesos de area.",
            primary_keys=("pixel_id",),
        )
        run.write_table(
            "pixel_variable_importance",
            products.pixel_variable_importance,
            description=(
                "Importancia das 31 variaveis preservada por fold, pixel nativo, "
                "condicao ENSO e lag; permutacao circular OOS dentro do evento."
            ),
            methods={
                "importance_scope": "out_of_sample_circular_within_event_permutation",
                "aggregation": "none; shard rows concatenated after exact-key validation",
            },
            primary_keys=tuple(IMPORTANCE_KEY),
        )
        run.write_table(
            "predictor_contract",
            products.predictor_contract,
            description="Catalogo ordenado das 31 variaveis F2 usado por todos os shards.",
            primary_keys=("variable",),
        )
        run.finalize(
            notes=(
                "Merge validado por fingerprint, cobertura exata e metricas OOS pareadas. "
                "Somente gate_eligible=true pode liberar F8."
            )
        )
    except Exception as exc:
        try:
            run.finalize(status="failed", notes=f"{type(exc).__name__}: {exc}")
        finally:
            raise


def _discover_run_dirs() -> list[Path]:
    directories: list[Path] = []
    for directory in sorted(RUNS.glob("F6_*")) if RUNS.exists() else []:
        manifest_path = directory / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Run F6 sem manifest: {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("parameters", {}).get("role") == "merge_pixel_shards_and_field_gate":
            continue
        directories.append(directory)
    return directories


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", type=Path, default=None)
    parser.add_argument("--target", type=Path, default=TARGET)
    parser.add_argument("--model", choices=("rf", "xgb"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.runs is not None and not args.runs:
        parser.error("--runs foi informado sem nenhum diretorio")
    run_dirs = args.runs if args.runs is not None else _discover_run_dirs()
    if not run_dirs:
        raise FileNotFoundError("Nenhum shard F6 candidato foi encontrado.")
    shards = [load_source_shard(path, model=args.model) for path in run_dirs]
    products, grid_hash = prepare_merge_products(shards, target_path=args.target)
    parameters = {
        "role": "merge_pixel_shards_and_field_gate",
        "source_runs": products.source_runs,
        "source_fingerprint_sha256": products.fingerprint,
        "parent_grid_sha256": grid_hash,
        "model": _single_value(products.predictions, "model", label="merged predictions"),
        "target_variable": _single_value(
            products.predictions, "target_variable", label="merged predictions"
        ),
        "target_transform": _single_value(
            products.predictions, "target_transform", label="merged predictions"
        ),
        "target_units": _single_value(
            products.predictions, "target_units", label="merged predictions"
        ),
        "native_value_schema": products.native_schema,
        "phase4_statistical_comparator": products.phase4_comparator,
        "coverage_required": 1.0,
    }
    run = start_artifact_run(
        6,
        mode="official",
        inputs=[args.target, *products.source_files, Path(__file__).resolve()],
        seed=args.seed,
        parameters=parameters,
        command=" ".join([sys.executable, *sys.argv]),
    )
    write_merge_artifacts(run, products)
    print(
        f"[F6-merge] coverage={products.coverage:.3f} | "
        f"gate_pass={int(products.gate['gate_pass'].sum())}/{len(products.gate)}"
    )
    print(run.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
