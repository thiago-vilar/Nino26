from __future__ import annotations

import base64
import csv
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "update_painel_executivo.py"
SPEC = importlib.util.spec_from_file_location("nino26_project_status_mirror", SCRIPT)
assert SPEC and SPEC.loader
mirror = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mirror
SPEC.loader.exec_module(mirror)

FINAL_VALIDATION_SCRIPT = ROOT / "scripts" / "run_final_validation.py"
FINAL_VALIDATION_SPEC = importlib.util.spec_from_file_location(
    "nino26_final_validation", FINAL_VALIDATION_SCRIPT
)
assert FINAL_VALIDATION_SPEC and FINAL_VALIDATION_SPEC.loader
final_validation = importlib.util.module_from_spec(FINAL_VALIDATION_SPEC)
sys.modules[FINAL_VALIDATION_SPEC.name] = final_validation
FINAL_VALIDATION_SPEC.loader.exec_module(final_validation)


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _input_record(path: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "path": str(path.resolve()),
        "exists": True,
        "is_directory": path.is_dir(),
    }
    if path.is_dir():
        record["tree_sha256"] = mirror._tree_sha256(path)
    else:
        record["sha256"] = mirror._sha256_file(path)
        record["size_bytes"] = path.stat().st_size
    return record


def _write_phase4_quick_fixture(
    root: Path,
    *,
    omit: frozenset[str] = frozenset(),
    condition_catalog: frozenset[str] = mirror.ENSO_PHASE_CONDITIONS,
    selection_contract: str = "quick:key-predictor",
    run_overrides: dict[str, str] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Materialize a hash-consistent F4C quick run for mirror tests."""

    f4c_runner = _write(root / "scripts/run_fase4c_regional.py", "# f4c quick fixture\n")
    lag_module = _write(root / "src/nino_brasil/stats/lag_analysis.py", "# lag fixture\n")
    for shapefile in (
        root / "data/interim/ibge/BR_Regioes_2024/BR_Regioes_2024.shp",
        root / "data/interim/ibge/Biomas_2025/lml_bioma_e250k_v20250911_A.shp",
    ):
        for suffix in (".shp", ".shx", ".dbf", ".prj"):
            _write(shapefile.with_suffix(suffix), f"{shapefile.stem}:{suffix}\n")
    geometry_contract, geometry_components, geometry_problems = (
        mirror._current_ibge_geometry_contract(root)
    )
    assert geometry_problems == []
    master = _write(
        root / "data/processed/parquet/features/nino34_master_weekly.csv",
        "week_ending_sunday,nino34_ssta\n2026-01-04,1.0\n",
    )
    phase_table = _write(
        root / "data/processed/parquet/statistics/phase3_fases_semanais_en_ln.csv",
        "week_ending_sunday,tipo,fase,event_id\n2026-01-04,el_nino,genese,E1\n",
    )
    phase_manifest = _write(
        Path(f"{phase_table}.manifest.json"),
        json.dumps({"run_id": "f3-current"}),
    )
    target_signature = "4" * 64
    target_manifest = _write(
        root
        / "data/interim/chirps_weekly_native_blocks"
        / target_signature[:16]
        / "manifest.json",
        json.dumps({"build_id": "target-current", "signature_sha256": target_signature}),
    )
    variable_contract = _write(
        root / "data/processed/parquet/statistics/phase4_chirps_target_variable_contract.csv",
        "variable,units\nprecip_robust_z,robust_z\n",
    )
    pixels = _write(
        root / "data/processed/parquet/features/phase4_chirps_native_pixels.csv",
        "pixel_id,latitude,longitude\n1,-10,-50\n",
    )
    membership = root / "data/processed/parquet/statistics/phase4C_native_pixel_membership_exact.parquet"
    membership.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "pixel_id": 1,
                "grid_hash": mirror.CHIRPS_FROZEN_GRID_HASH,
                "membership_status": "canonical_equal_area_official_ibge",
                **geometry_contract,
            },
            {
                "pixel_id": 2,
                "grid_hash": mirror.CHIRPS_FROZEN_GRID_HASH,
                "membership_status": "canonical_equal_area_official_ibge",
                **geometry_contract,
            },
        ]
    ).to_parquet(membership, index=False)
    inputs = [
        master,
        phase_table,
        phase_manifest,
        target_manifest,
        variable_contract,
        pixels,
        membership,
        *geometry_components,
        f4c_runner,
        lag_module,
    ]
    phase_hash = mirror._sha256_file(phase_table)
    common = {
        "parent_f3_run_id": "f3-current",
        "parent_f3_artifact_sha256": phase_hash,
        "target_build_id": "target-current",
        "target_block_signature_sha256": target_signature,
        "target_contract_version": mirror.CHIRPS_TARGET_CONTRACT,
        "grid_hash_sha256": mirror.CHIRPS_FROZEN_GRID_HASH,
        **geometry_contract,
        "selection_contract": selection_contract,
        "predictor_count": 1,
        "predictor_catalog_sha256": mirror.hashlib.sha256(b"nino34_ssta").hexdigest(),
        "f4c_runner_sha256": mirror._sha256_file(f4c_runner),
        "lag_analysis_module_sha256": mirror._sha256_file(lag_module),
    }
    quick_root = root / "data/processed/parquet/statistics/quick"
    run_overrides = run_overrides or {}
    for name in mirror.F4C_QUICK_TABULAR_NAMES:
        if name in omit:
            continue
        run_id = run_overrides.get(name, "F4C_QUICK_fixture")
        identity = {"analysis_run_id": run_id, **common}
        if name in mirror.F4C_QUICK_CONDITION_TABLE_NAMES:
            rows = [
                {
                    **identity,
                    "condicao_fonte": condition,
                    "pixel_id": 1 + index % 2,
                    "value": index,
                }
                for index, condition in enumerate(sorted(condition_catalog))
            ]
        elif name == "phase4C_native_predictor_treatment_quick.csv":
            rows = [
                {
                    **identity,
                    "variavel": "nino34_ssta",
                }
            ]
        else:
            rows = [{**identity, "week_ending_sunday": "2026-01-04", "coverage": 1.0}]
        output = quick_root / name
        if output.suffix == ".csv":
            _write_csv(output, rows)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(output, index=False)
        sidecar = {
            "schema_version": "nino-brasil-phase4-semantic-output-v1",
            "run_id": run_id,
            "created_utc": "2026-07-13T10:00:00+00:00",
            "contract": {
                "phase": "F4",
                "stage": "F4C_QUICK",
                "artifact_type": "numeric_table",
                **common,
                "predictor_names": ["nino34_ssta"],
                "field_permutations": 19,
            },
            "artifact": {
                "path": str(output.relative_to(root)).replace("\\", "/"),
                "sha256": mirror._sha256_file(output),
                "size_bytes": output.stat().st_size,
                "rows": len(rows),
                "columns": len(rows[0]),
                "column_names": list(rows[0]),
            },
            "inputs": [_input_record(path) for path in inputs],
        }
        _write(Path(f"{output}.manifest.json"), json.dumps(sidecar))

    atlas = root / mirror.F4C_QUICK_ATLAS
    if atlas.name not in omit:
        atlas.parent.mkdir(parents=True, exist_ok=True)
        coords = {
            "variavel": ["nino34_ssta"],
            "condicao_fonte": sorted(condition_catalog),
            "lag_sem": np.arange(0, 79, 2, dtype="int16"),
            "pixel": np.array([1, 2], dtype="int64"),
        }
        shape = (1, len(condition_catalog), 40, 2)
        dataset = xr.Dataset(
            {
                "r": (("variavel", "condicao_fonte", "lag_sem", "pixel"), np.zeros(shape, dtype="float32")),
                "p": (("variavel", "condicao_fonte", "lag_sem", "pixel"), np.ones(shape, dtype="float32")),
                "q_fdr_bh": (("variavel", "condicao_fonte", "lag_sem", "pixel"), np.ones(shape, dtype="float32")),
                "fdr_bh_reject": (("variavel", "condicao_fonte", "lag_sem", "pixel"), np.zeros(shape, dtype=bool)),
                "n_eff_bretherton": (("variavel", "condicao_fonte", "lag_sem", "pixel"), np.full(shape, 10, dtype="float32")),
            },
            coords=coords,
            attrs={
                "analysis_run_id": "F4C_QUICK_fixture",
                **common,
                "spatial_contract": "original CHIRPS pixels with Brazil overlap; no interpolation",
            },
        )
        dataset.to_zarr(atlas, mode="w", consolidated=False, zarr_format=2)
        opened = xr.open_zarr(atlas, consolidated=False)
        try:
            dimensions = {name: int(size) for name, size in opened.sizes.items()}
            arrays = {
                name: {
                    "dims": list(array.dims),
                    "shape": [int(value) for value in array.shape],
                    "dtype": str(array.dtype),
                    "chunks": [list(map(int, axis_chunks)) for axis_chunks in array.chunks]
                    if array.chunks is not None
                    else None,
                }
                for name, array in opened.data_vars.items()
            }
        finally:
            opened.close()
        atlas_files = [path for path in atlas.rglob("*") if path.is_file()]
        atlas_manifest = {
            "schema_version": "nino-brasil-phase4-semantic-output-v1",
            "run_id": "F4C_QUICK_fixture",
            "created_utc": "2026-07-13T10:00:00+00:00",
            "contract": {
                "phase": "F4",
                "stage": "F4C_QUICK",
                "artifact_type": "numeric_array",
                **common,
                "predictor_names": ["nino34_ssta"],
                "field_permutations": 19,
            },
            "artifact": {
                "path": str(atlas.relative_to(root)).replace("\\", "/"),
                "is_directory": True,
                "tree_sha256": mirror._tree_sha256(atlas),
                "n_files": len(atlas_files),
                "dimensions": dimensions,
                "arrays": arrays,
            },
            "inputs": [_input_record(path) for path in inputs],
        }
        _write(Path(f"{atlas}.manifest.json"), json.dumps(atlas_manifest))

    chirps = mirror._status(
        "promoted",
        "fixture",
        promoted=True,
        build_id="target-current",
        block_signature_sha256=target_signature,
        grid_hash_sha256=mirror.CHIRPS_FROZEN_GRID_HASH,
    )
    phase3 = mirror._status(
        "complete",
        "fixture",
        complete=True,
        run_id="f3-current",
        phase_table_sha256=phase_hash,
    )
    return chirps, phase3


def _gate_rows(phase: int, passed: bool) -> tuple[str, list[dict[str, object]]]:
    if phase == 5:
        components = (
            ("classification_h04", "mean_skill_f1_vs_best_persistence_or_seasonal"),
            ("event_dimension_Y_pico", "skill_mae_vs_type_climatology"),
            ("event_dimension_Y_tempo_para_pico_sem", "skill_mae_vs_type_climatology"),
            ("event_dimension_Y_duracao_sem", "skill_mae_vs_type_climatology"),
        )
        return "scientific_gate.csv", [
            {
                "component": component,
                "metric": metric,
                "value": 0.1 if passed else -0.1,
                "threshold_rule": ">0",
                "n_oos_units": 5,
                "oos_unit": "whole-event fold",
                "min_train_el_nino_events": 3,
                "min_train_la_nina_events": 3,
                "min_train_active_events_per_type_required": 3,
                "independent_support_gate_pass": passed,
                "support_required_for_gate": True,
                "gate_pass": passed,
            }
            for component, metric in components
        ]
    if phase == 6:
        return "field_gate.csv", [
            {
                "target_variable": "precip_weekly_mm",
                "target_transform": "train_only_robust_z",
                "model": "rf",
                "target_units": "mm/week",
                "condition": condition,
                "lag_weeks": 4,
                "gate_pass": passed,
                "gate_eligible": True,
                "pixel_set_exact": True,
                "pixel_coverage_fraction": 1.0,
            }
            for condition in sorted(mirror.ENSO_PHASE_CONDITIONS)
        ]
    if phase == 7:
        return "scientific_gate.csv", [
            {
                "horizon_weeks": 4,
                "mean_skill_f1_vs_best_persistence_or_seasonal": 0.1,
                "baseline_gate_pass": passed,
                "best_f5_reference_run_id": "F5_reference",
                "paired_coverage_best_f5": 1.0,
                "skill_f1_f7_minus_best_f5_paired": 0.1,
                "paired_f5_gate_pass": passed,
                "f5_comparison_required": True,
                "min_train_active_events_per_type_required": 3,
                "min_train_el_nino_events": 3,
                "min_train_la_nina_events": 3,
                "independent_support_gate_pass": passed,
                "support_required_for_gate": True,
                "minimum_event_dimension_skill_required": 0.0,
                "max_interval90_absolute_calibration_error": 0.15,
                "event_dimension_gate_pass": passed,
                "interval_calibration_gate_pass": passed,
                "skill_mae_event_equal_peak_magnitude_c": 0.1,
                "skill_mae_event_equal_event_time_to_peak_weeks": 0.1,
                "skill_mae_event_equal_event_duration_weeks": 0.1,
                "interval90_absolute_calibration_error_peak_magnitude_c": 0.05,
                "interval90_absolute_calibration_error_event_time_to_peak_weeks": 0.05,
                "interval90_absolute_calibration_error_event_duration_weeks": 0.05,
                "scientific_gate_pass": passed,
            }
        ]
    if phase == 8:
        rows = [
            {
                "condition": condition,
                "n_independent_events": 3,
                "minimum_independent_events_required": 3,
                "minimum_event_support_gate_pass": passed,
                "persistence_gate_pass": passed,
                "f6_comparison_gate_pass": passed,
                "n_events_with_interval_coverage": 3,
                "n_calibration_folds": 3,
                "interval90_coverage_event_equal": 0.90,
                "interval90_nominal_coverage": 0.90,
                "interval90_absolute_calibration_error": 0.0,
                "maximum_fold_interval90_absolute_calibration_error": 0.05,
                "max_interval90_absolute_calibration_error": 0.15,
                "complete_native_interval_values": passed,
                "all_fold_interval_calibration_gate_pass": passed,
                "interval_calibration_gate_pass": passed,
                "interval_aggregation_unit": (
                    "area_weighted_native_pixel_week_within_event_then_"
                    "independent_event_equal"
                ),
                "gate_pass": passed,
                "overall_eight_condition_gate_pass": passed,
            }
            for condition in sorted(mirror.ENSO_PHASE_CONDITIONS)
        ]
        return "confirmatory_gate_by_condition.csv", rows
    raise AssertionError(phase)


def _make_run(
    root: Path,
    *,
    phase: int,
    mode: str,
    run_id: str,
    model: str = "default",
    status: str = "complete",
    finished_at: str = "2026-07-13T10:00:00+00:00",
    gate_pass: bool = True,
    role: str | None = None,
    shard_contract: bool = False,
) -> Path:
    source = root / "src/nino_brasil"
    _write(source / "core.py", "VALUE = 1\n")
    runner_name = mirror.RUNNER_BY_PHASE[phase]
    if shard_contract:
        runner_name = "run_fase6_brazil_ml.py"
    runner = _write(root / "scripts" / runner_name, "# runner fixture\n")
    directory = root / f"data/processed/runs/{mode}/fase{phase}/{run_id}"
    table_name, gate_rows = _gate_rows(phase, gate_pass)
    if shard_contract:
        table_name = "pixel_metrics.csv"
        gate_rows = [{"pixel_id": 1, "rmse": 1.0}]
    table_products: dict[str, list[dict[str, object]]] = {table_name: gate_rows}
    if not shard_contract:
        for required_name in mirror.REQUIRED_TABLES_BY_PHASE_MODE.get(
            (phase, mode), frozenset()
        ):
            if required_name == table_name:
                continue
            if required_name == "predictor_contract.csv":
                table_products[required_name] = [
                    {"variable": variable} for variable in mirror.PHASE2_PHYSICAL_COLUMNS
                ]
            else:
                table_products[required_name] = [{"value": 1}]
    manifest_rows = []
    for product_name, product_rows in sorted(table_products.items()):
        product = _write_csv(directory / "tables" / product_name, product_rows)
        manifest_rows.append(
            {
                "table": product_name,
                "path": f"tables/{product_name}",
                "rows": len(product_rows),
                "columns": len(product_rows[0]),
                "sha256": mirror._sha256_file(product),
                "schema_sha256": "b" * 64,
                "description": "fixture semântica",
                "units_json": "{}",
                "dimensions_json": "{}",
                "methods_json": "{}",
                "primary_keys": "",
            }
        )
    table_manifest = _write_csv(directory / "tables_manifest.csv", manifest_rows)
    parameters: dict[str, object] = {"model": model}
    if phase == 5:
        parameters["horizons"] = [4]
    if role is not None:
        parameters["role"] = role
    config = _write(root / "configs/project.yaml", "seed: 42\n")
    manifest = {
        "schema_version": "nino26-run-v1",
        "run_id": run_id,
        "phase": phase,
        "mode": mode,
        "status": status,
        "started_at": "2026-07-13T09:00:00+00:00",
        "finished_at": finished_at,
        "parameters": parameters,
        "parameters_sha256": mirror._json_hash(parameters),
        "seed": 42,
        "command": f"python scripts/{runner_name}",
        "environment": {"python": "3.12"},
        "git": {"commit": "a" * 40, "branch": "test", "dirty": False},
        "inputs": [_input_record(source), _input_record(runner)],
        "configs": [_input_record(config)],
        "files": [],
        "n_tables": len(manifest_rows),
        "tables_manifest_sha256": mirror._sha256_file(table_manifest),
    }
    _write(directory / "run_manifest.json", json.dumps(manifest))
    return directory


def _make_f3_catalog(
    root: Path,
    run_id: str,
    codes: tuple[str, ...] | list[str] = mirror.PHASE3_FIGURE_CODES,
) -> None:
    rows: list[dict[str, object]] = []
    for code in codes:
        source = _write_csv(
            root / f"data/processed/parquet/statistics/{code}_source.csv",
            [{"week": "2026-01-04", "value": 1.0}],
        )
        source_manifest = _write(
            Path(f"{source}.manifest.json"),
            json.dumps(
                {
                    "schema_version": "nino-brasil.semantic-table.v1",
                    "run_id": run_id,
                    "artifact": {
                        "path": str(source.resolve()),
                        "sha256": mirror._sha256_file(source),
                    },
                    "inputs": [],
                }
            ),
        )
        numeric = _write_csv(
            root / f"data/processed/numeric-tables/fase3/{code}/source.csv",
            [{"week": "2026-01-04", "value": 1.0}],
        )
        _write_csv(
            numeric.parent / "manifest.csv",
            [
                {
                    "tabela": numeric.name,
                    "linhas": 1,
                    "colunas": 2,
                    "sha256": mirror._sha256_file(numeric),
                    "semantic_source": True,
                    "audit_level": "semantic_source",
                    "run_id": run_id,
                    "source_path": str(source.resolve()),
                    "source_manifest_path": str(source_manifest.resolve()),
                    "source_manifest_sha256": mirror._sha256_file(source_manifest),
                    "source_run_id": run_id,
                }
            ],
        )
        figure_rel = f"fase3/{code}.png"
        _write_bytes(root / "data/processed/figures" / figure_rel, PNG_1X1)
        rows.append(
            {
                "codigo": code,
                "fase": 3,
                "arquivo": figure_rel,
                "tabelas": numeric.name,
                "n_tabelas": 1,
                "audit_level": "semantic_source",
                "run_id": run_id,
            }
        )
    _write_csv(root / "data/processed/figuras_manifesto.csv", rows)


def _make_phase2_contract(root: Path) -> tuple[Path, Path]:
    created: dict[str, Path] = {}
    for section, paths in mirror.PHASE2_REQUIRED_PATHS.items():
        for relative in paths:
            path = root / relative
            if relative.endswith("nino34_master_weekly.csv"):
                row = {
                    "week_ending_sunday": "1981-01-04",
                    **{column: 0.0 for column in mirror.PHASE2_PHYSICAL_COLUMNS},
                    "ocean_source_code": 1,
                }
                _write_csv(path, [row])
            elif relative.endswith("phase2_master_validation.csv"):
                _write_csv(
                    path,
                    [
                        {
                            "checagem": check,
                            "passou": True,
                            "severidade": "error",
                            "detalhe": "fixture validada",
                        }
                        for check in mirror.PHASE2_VALIDATION_CHECKS
                    ],
                )
            else:
                _write(path, "fixture\n")
            created[f"{section}:{relative}"] = path
    manifest = {
        "schema_version": "phase2-master-manifest/1.0",
        "run_id": "phase2_fixture",
        "started_at_utc": "2026-07-13T09:00:00+00:00",
        "completed_at_utc": "2026-07-13T10:00:00+00:00",
        "contract": {
            "physical_variable_count": 31,
            "physical_columns": list(mirror.PHASE2_PHYSICAL_COLUMNS),
            "metadata_columns": ["ocean_source_code"],
        },
        "options": {"strict": True, "ocean_only": False, "skip_ctd": False},
        "raw_shape": [1, 32],
        "source_adjusted_shape": [1, 32],
    }
    for section, paths in mirror.PHASE2_REQUIRED_PATHS.items():
        manifest[section] = [_input_record(created[f"{section}:{relative}"]) for relative in paths]
    manifest_path = _write(
        root / "data/processed/parquet/statistics/phase2_master_run_manifest.json",
        json.dumps(manifest),
    )
    validation_path = root / "data/processed/parquet/statistics/phase2_master_validation.csv"
    return manifest_path, validation_path


def _write_final_validation_receipt(root: Path) -> Path:
    records = [
        mirror._validation_context_record(root, relative)
        for relative in mirror.FINAL_VALIDATION_CONTEXT_PATHS
    ]
    context_hash = mirror._compact_json_hash(records)
    command_catalog = [
        {"id": check_id, "argv": ["python", check_id]}
        for check_id in sorted(mirror.FINAL_VALIDATION_CHECK_IDS)
    ]
    empty_hash = mirror.hashlib.sha256(b"").hexdigest()
    checks = [
        {
            **item,
            "command_sha256": mirror.hashlib.sha256(
                "\0".join(item["argv"]).encode("utf-8")
            ).hexdigest(),
            "started_at_utc": "2026-07-13T10:00:00+00:00",
            "finished_at_utc": "2026-07-13T10:00:01+00:00",
            "exit_code": 0,
            "passed": True,
            "stdout_sha256": empty_hash,
            "stderr_sha256": empty_hash,
            "stdout_tail": "",
            "stderr_tail": "",
        }
        for item in command_catalog
    ]
    payload = {
        "schema_version": "nino26-final-validation-v1",
        "workspace": str(root.resolve()),
        "python": sys.executable,
        "started_at_utc": "2026-07-13T10:00:00+00:00",
        "finished_at_utc": "2026-07-13T10:01:00+00:00",
        "status": "passed",
        "all_passed": True,
        "check_count": len(checks),
        "expected_check_count": len(checks),
        "check_catalog_sha256": mirror._compact_json_hash(command_catalog),
        "validation_context": {
            "schema_version": "nino26-final-validation-context/v1",
            "paths": list(mirror.FINAL_VALIDATION_CONTEXT_PATHS),
            "records": records,
            "inputs_sha256": context_hash,
        },
        "validation_context_finished_sha256": context_hash,
        "validation_context_unchanged": True,
        "checks": checks,
    }
    return _write(
        root / "data/audit/final_validation_summary.json",
        json.dumps(payload),
    )


def test_run_selection_validates_before_using_recency(tmp_path: Path) -> None:
    older_rf = _make_run(
        tmp_path,
        phase=5,
        mode="official",
        run_id="F5_older_valid_rf",
        model="rf",
        finished_at="2026-07-13T10:00:00+00:00",
    )
    newer_failed = _make_run(
        tmp_path,
        phase=5,
        mode="official",
        run_id="F5_newer_failed_rf",
        model="rf",
        status="failed",
        finished_at="2026-07-13T12:00:00+00:00",
    )
    _make_run(
        tmp_path,
        phase=5,
        mode="official",
        run_id="F5_valid_xgb",
        model="xgb",
        finished_at="2026-07-13T11:00:00+00:00",
    )
    os.utime(newer_failed, (2_000_000_000, 2_000_000_000))
    os.utime(older_rf, (1_000_000_000, 1_000_000_000))

    selection = mirror._select_artifact_runs(tmp_path, 5, "official")

    assert selection["state"] == "complete"
    selected = {run["model"]: run["run_id"] for run in selection["selected"]}
    assert selected == {"rf": "F5_older_valid_rf", "xgb": "F5_valid_xgb"}
    assert "F5_newer_failed_rf" in {run["run_id"] for run in selection["rejected"]}


def test_selected_runs_publish_manifest_table_and_source_fingerprints(
    tmp_path: Path,
) -> None:
    _make_run(
        tmp_path,
        phase=5,
        mode="official",
        run_id="F5_hashes_rf",
        model="rf",
    )
    _make_run(
        tmp_path,
        phase=5,
        mode="official",
        run_id="F5_hashes_xgb",
        model="xgb",
    )

    selection = mirror._select_artifact_runs(tmp_path, 5, "official")

    assert selection["state"] == "complete"
    for run in selection["selected"]:
        assert mirror._is_sha256(run["run_manifest_sha256"])
        assert mirror._is_sha256(run["tables_manifest_sha256"])
        assert mirror._is_sha256(run["parameters_sha256"])
        assert mirror._is_sha256(run["input_catalog_sha256"])
        assert run["git_commit"] == "a" * 40
        assert run["table_count"] == len(run["table_catalog"])
        assert {
            Path(record["path"]).name for record in run["source_fingerprints"]
        } == {"nino_brasil", "run_fase5_cycle_ml.py"}


def test_final_validation_receipt_is_invalidated_by_later_source_drift(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path / "src/nino_brasil/core.py", "VALUE = 1\n")
    _write(tmp_path / "scripts/tool.py", "print('ok')\n")
    report_path = _write_final_validation_receipt(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    passed = mirror.audit_final_validation_summary(tmp_path, report, report_path)
    assert passed["state"] == "passed"

    _write(source, "VALUE = 2\n")
    failed = mirror.audit_final_validation_summary(tmp_path, report, report_path)

    assert failed["state"] == "failed"
    assert "final_validation_context_current_workspace_mismatch" in failed["problems"]
    assert "final_validation_context_hash_mismatch" in failed["problems"]


def test_final_validation_writer_and_mirror_share_context_fingerprint(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "src/nino_brasil/core.py", "VALUE = 1\n")
    _write(tmp_path / "scripts/tool.py", "print('ok')\n")
    _write(tmp_path / "tests/test_fixture.py", "def test_ok(): assert True\n")

    context = final_validation.collect_validation_context(tmp_path)
    expected_records = [
        mirror._validation_context_record(tmp_path, relative)
        for relative in mirror.FINAL_VALIDATION_CONTEXT_PATHS
    ]

    assert tuple(context["paths"]) == mirror.FINAL_VALIDATION_CONTEXT_PATHS
    assert context["records"] == expected_records
    assert context["inputs_sha256"] == mirror._compact_json_hash(expected_records)


def test_final_validation_writer_persists_unchanged_context_receipt(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "src/nino_brasil/core.py", "VALUE = 1\n")
    output = tmp_path / "data/audit/final_validation_summary.json"

    payload = final_validation.run_checks(
        [("fixture", [sys.executable, "-c", "print('ok')"])],
        output=output,
        root=tmp_path,
    )

    assert payload["status"] == "passed"
    assert payload["validation_context_unchanged"] is True
    assert payload["validation_context_finished_sha256"] == payload[
        "validation_context"
    ]["inputs_sha256"]
    assert output.is_file()


def test_workspace_hygiene_reports_quarantine_separately_from_active_residue(
    tmp_path: Path,
) -> None:
    quarantine_manifest = _write(
        tmp_path / "data/quarantine/stale_derived/20260713T120000Z/manifest.json",
        json.dumps(
            {
                "schema": "nino26-derived-quarantine-v2",
                "created_at_utc": "2026-07-13T12:00:00+00:00",
                "records": [{"source": "legacy.csv", "status": "moved"}],
            }
        ),
    )
    legacy_destination = _write(
        tmp_path / "data/quarantine/phase5_runs/20260713T120100Z/F5_old/file.txt",
        "old run\n",
    )
    _write_bytes(
        tmp_path / "data/quarantine/phase5_runs/20260713T120100Z/manifest.json",
        b"\xef\xbb\xbf"
        + json.dumps(
            {
                "operation": "move; no deletion",
                "source": str(tmp_path / "old/F5_old"),
                "destination": str(legacy_destination.parent),
            }
        ).encode("utf-8"),
    )
    legacy = _write(
        tmp_path / "data/processed/parquet/statistics/phase4A_legacy.csv",
        "value\n1\n",
    )
    incomplete_run = tmp_path / "data/processed/runs/official/fase5/F5_incomplete"
    incomplete_run.mkdir(parents=True)
    details = {
        "F5": {
            "official": {
                "rejected": [
                    {
                        "path": str(incomplete_run.relative_to(tmp_path)).replace("\\", "/"),
                        "status": "missing",
                        "artifact_valid": False,
                        "problems": ["missing_or_invalid_run_manifest"],
                    }
                ]
            },
            "smoke": {"rejected": []},
        }
    }

    status = mirror.collect_workspace_hygiene(
        tmp_path,
        details,
        mirror._status("missing", "fixture"),
    )

    assert status["state"] == "needs_attention"
    assert status["active_residual_candidate_count"] == 2
    assert {item["path"] for item in status["active_residual_candidates"]} == {
        str(legacy.relative_to(tmp_path)).replace("\\", "/"),
        str(incomplete_run.relative_to(tmp_path)).replace("\\", "/"),
    }
    assert status["quarantine"]["batch_count"] == 2
    assert status["quarantine"]["manifest_invalid_count"] == 0
    assert status["quarantine"]["quarantined_record_count"] == 2
    assert status["mutation_performed"] is False
    assert quarantine_manifest.is_file()


def test_workspace_hygiene_flags_only_superseded_chirps_block_roots(
    tmp_path: Path,
) -> None:
    block_root = tmp_path / "data/interim/chirps_weekly_native_blocks"
    active_signature = "a" * 64
    _write(
        block_root / "active/manifest.json",
        json.dumps(
            {
                "build_id": "F4TARGET_active",
                "signature_sha256": active_signature,
            }
        ),
    )
    inactive = _write(
        block_root / "inactive/manifest.json",
        json.dumps(
            {
                "build_id": "F4TARGET_old",
                "signature_sha256": "b" * 64,
            }
        ),
    ).parent

    status = mirror.collect_workspace_hygiene(
        tmp_path,
        {},
        mirror._status(
            "promoted",
            "fixture",
            build_id="F4TARGET_active",
            block_signature_sha256=active_signature,
        ),
    )

    assert status["active_residual_candidate_count"] == 1
    assert status["active_residual_candidates"][0]["path"] == str(
        inactive.relative_to(tmp_path)
    ).replace("\\", "/")
    assert status["active_residual_candidates"][0]["category"] == (
        "superseded_chirps_block_root"
    )


def test_run_in_wrong_phase_mode_folder_is_never_eligible(tmp_path: Path) -> None:
    directory = _make_run(
        tmp_path,
        phase=5,
        mode="official",
        run_id="F5_wrong_identity",
        model="rf",
    )
    manifest_path = directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"phase": 7, "mode": "smoke"})
    _write(manifest_path, json.dumps(manifest))

    selection = mirror._select_artifact_runs(tmp_path, 5, "official")

    assert selection["state"] == "invalidated"
    assert not selection["selected"]
    problems = selection["rejected"][0]["problems"]
    assert "manifest_phase_mismatch:expected=5:actual=7" in problems
    assert "manifest_mode_mismatch:expected=official:actual=smoke" in problems


def test_recorded_paths_are_root_relative_and_cannot_escape(tmp_path: Path) -> None:
    outside = _write(tmp_path.parent / "outside-mirror-fixture.txt", "outside\n")
    record = {
        "path": str(Path("..") / outside.name),
        "sha256": mirror._sha256_file(outside),
    }

    problems = mirror._verify_record(tmp_path, record, "fixture")

    assert len(problems) == 1
    assert "invalid_path:path_outside_root" in problems[0]


def test_phase2_rejects_a_single_arbitrary_passing_check_even_with_fresh_hash(
    tmp_path: Path,
) -> None:
    manifest_path, validation_path = _make_phase2_contract(tmp_path)
    spec = next(spec for spec in mirror.PHASE_SPECS if spec.number == 2)
    assert mirror._phase2(tmp_path, spec)["official"]["state"] == "complete"

    _write_csv(
        validation_path,
        [
            {
                "checagem": "checagem_arbitraria",
                "passou": True,
                "severidade": "error",
                "detalhe": "não representa o catálogo científico",
            }
        ],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["outputs"]
        if str(item["path"]).replace("\\", "/").endswith("phase2_master_validation.csv")
    )
    record["sha256"] = mirror._sha256_file(validation_path)
    record["size_bytes"] = validation_path.stat().st_size
    _write(manifest_path, json.dumps(manifest))

    phase = mirror._phase2(tmp_path, spec)

    assert phase["official"]["state"] == "invalidated"
    assert any("validation_count_mismatch" in item for item in phase["data"]["problems"])
    assert any("validation_catalog_mismatch" in item for item in phase["data"]["problems"])


def test_phase2_rejects_empty_code_inputs_and_outputs(tmp_path: Path) -> None:
    manifest_path, _ = _make_phase2_contract(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"code": [], "inputs": [], "outputs": []})
    _write(manifest_path, json.dumps(manifest))
    spec = next(spec for spec in mirror.PHASE_SPECS if spec.number == 2)

    phase = mirror._phase2(tmp_path, spec)

    assert phase["official"]["state"] == "invalidated"
    assert "missing_required_code_records" in phase["data"]["problems"]
    assert "missing_required_inputs_records" in phase["data"]["problems"]
    assert "missing_required_outputs_records" in phase["data"]["problems"]


def test_smoke_never_becomes_official_or_promoted(tmp_path: Path) -> None:
    _make_run(
        tmp_path,
        phase=7,
        mode="smoke",
        run_id="F7_smoke_only",
    )
    spec = next(spec for spec in mirror.PHASE_SPECS if spec.number == 7)
    chirps = mirror._status("missing", "fixture")
    phase3 = mirror._status("missing", "fixture", complete=False)

    phase, _ = mirror._artifact_phase(tmp_path, spec, chirps, phase3)

    assert phase["smoke"]["state"] == "complete"
    assert phase["official"]["state"] == "not_run"
    assert phase["gate"]["state"] == "pending"
    assert phase["promotion"]["state"] == "blocked"


def test_complete_official_negative_result_is_not_relabelled_pending(tmp_path: Path) -> None:
    _make_run(
        tmp_path,
        phase=5,
        mode="official",
        run_id="F5_negative_rf",
        model="rf",
        gate_pass=False,
    )
    _make_run(
        tmp_path,
        phase=5,
        mode="official",
        run_id="F5_negative_xgb",
        model="xgb",
        gate_pass=False,
    )
    spec = next(spec for spec in mirror.PHASE_SPECS if spec.number == 5)
    phase, _ = mirror._artifact_phase(
        tmp_path,
        spec,
        mirror._status("missing", "fixture"),
        mirror._status("missing", "fixture", complete=False),
    )

    assert phase["official"]["state"] == "complete"
    assert phase["gate"]["state"] == "failed"
    assert phase["gate"]["state"] != "pending"
    assert phase["promotion"]["state"] == "audit_ready"
    assert phase["promotion"]["state"] != "promoted"


def test_scientific_predecessor_gate_blocks_downstream_execution(tmp_path: Path) -> None:
    spec = next(spec for spec in mirror.PHASE_SPECS if spec.number == 6)
    chirps = mirror._status("promoted", "fixture", promoted=True)
    phase3 = mirror._status("complete", "fixture", complete=True)
    predecessor = mirror._status("failed", "negative F4", passed=False)

    phase, _ = mirror._artifact_phase(
        tmp_path,
        spec,
        chirps,
        phase3,
        scientific_prerequisites=(
            ("gate F4", predecessor, frozenset({"passed"})),
        ),
    )

    assert phase["official"]["state"] == "not_run"
    assert phase["data"]["state"] == "blocked"
    assert phase["gate"]["state"] == "blocked"
    assert phase["promotion"]["state"] == "blocked"
    assert "Não executar" in phase["next_action"]


def test_phase4_negative_gate_remains_auditable_but_not_promoted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mirror,
        "collect_phase4_outputs",
        lambda *_: {
            "official": mirror._status("complete", "fixture"),
            "gate": mirror._status("failed", "negative result", passed=False),
            "run_id": "F4D_fixture",
        },
    )
    monkeypatch.setattr(
        mirror,
        "collect_phase4_smoke",
        lambda *_: mirror._status("complete", "fixture", passed=True),
    )
    monkeypatch.setattr(
        mirror,
        "collect_figure_lineage",
        lambda *_: mirror._status("passed", "fixture", passed=True),
    )
    monkeypatch.setattr(
        mirror,
        "collect_notebook_contract",
        lambda *_: mirror._status("passed", "fixture", passed=True),
    )
    spec = next(spec for spec in mirror.PHASE_SPECS if spec.number == 4)

    phase, _ = mirror._phase4(
        tmp_path,
        spec,
        mirror._status("promoted", "fixture", promoted=True),
        mirror._status("complete", "fixture", complete=True),
    )

    assert phase["official"]["state"] == "complete"
    assert phase["gate"]["state"] == "failed"
    assert phase["promotion"]["state"] == "audit_ready"


def test_chirps_rejects_attributes_without_materialized_zarr_arrays(tmp_path: Path) -> None:
    block_root = tmp_path / "data/interim/chirps_weekly_native_blocks/signature"
    manifest_path = block_root / "manifest.json"
    partial = {
        "build_status": "canonical",
        "build_complete": False,
        "contract": "chirps-native-weekly-spatial-blocks-v4",
        "target_contract_version": "chirps-native-weekly-v4",
        "signature_sha256": "s" * 64,
        "full_grid_hash_sha256": mirror.CHIRPS_FROZEN_GRID_HASH,
        "latitude_count": 8,
        "latitude_block_size": 4,
        "completed_blocks": ["latitude_000_004.zarr"],
    }
    _write(manifest_path, json.dumps(partial))
    assert mirror.collect_chirps_status(tmp_path)["state"] == "in_progress"

    builder = _write(tmp_path / "scripts/build_phase4_chirps_targets.py", "# builder\n")
    target_module = _write(tmp_path / "src/nino_brasil/targets/chirps_native.py", "# target\n")
    target = tmp_path / "data/processed/zarr/features/chirps_native_weekly_targets.zarr"
    attrs = {
        "deep_validation_passed": True,
        "deep_validation_timestamp_utc": "2026-07-13T10:00:00+00:00",
        "build_id": "F4TARGET_fixture",
        "block_signature_sha256": "s" * 64,
        "target_contract_version": "chirps-native-weekly-v4",
        "grid_hash_sha256": mirror.CHIRPS_FROZEN_GRID_HASH,
        "build_status": "canonical",
        "spatial_operation": "coordinate subset only; interpolation=false",
    }
    _write(target / ".zattrs", json.dumps(attrs))
    complete = {
        **partial,
        "build_complete": True,
        "promotion_status": "promoted_after_deep_validation",
        "promoted_utc": "2026-07-13T10:00:00+00:00",
        "promoted_target": "data/processed/zarr/features/chirps_native_weekly_targets.zarr",
        "build_id": "F4TARGET_fixture",
        "completed_blocks": ["latitude_000_004.zarr", "latitude_004_008.zarr"],
        "deep_validation": {
            "valid": True,
            "errors": [],
            "grid_hash": mirror.CHIRPS_FROZEN_GRID_HASH,
            "n_time": 1,
            "n_latitude": 8,
            "n_longitude": 1,
        },
        "builder_script_sha256": mirror._sha256_file(builder),
        "target_module_sha256": mirror._sha256_file(target_module),
    }
    contract = _write(tmp_path / "data/processed/parquet/statistics/phase4_chirps_target_variable_contract.csv", "variable,units\nprecip,mm\n")
    pixels = _write(tmp_path / "data/processed/parquet/features/phase4_chirps_native_pixels.csv", "pixel_id,lat,lon\n1,-10,-50\n")
    complete.update(
        {
            "target_variable_contract": str(contract.relative_to(tmp_path)).replace("\\", "/"),
            "target_variable_contract_sha256": mirror._sha256_file(contract),
            "promoted_pixel_inventory": str(pixels.relative_to(tmp_path)).replace("\\", "/"),
        }
    )
    _write(manifest_path, json.dumps(complete))

    result = mirror.collect_chirps_status(tmp_path)

    assert result["state"] == "invalid"
    assert result["promoted"] is False
    assert any("target_zarr_metadata_missing" in item for item in result["problems"])
    assert "target_zarr_has_no_arrays_or_chunks" in result["problems"]


def test_chirps_v4_requires_zero_inflated_tail_audit_layers(tmp_path: Path) -> None:
    target = tmp_path / "data/processed/zarr/features/chirps_native_weekly_targets.zarr"
    missing = "r99p_weekly_climatology_pooled_positive_l1_scale"
    coordinates = {
        "time": pd.DatetimeIndex(["2026-01-04"]),
        "latitude": np.asarray([-10.0], dtype="float32"),
        "longitude": np.asarray([-50.0], dtype="float32"),
    }
    variables: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for name in sorted(mirror.CHIRPS_REQUIRED_VARIABLES - {missing}):
        if name == "pixel_id":
            variables[name] = (("latitude", "longitude"), np.asarray([[0]], dtype="int64"))
        elif name == "brazil_center":
            variables[name] = (("latitude", "longitude"), np.asarray([[True]], dtype=bool))
        elif name == "brazil_fraction":
            variables[name] = (("latitude", "longitude"), np.asarray([[1.0]], dtype="float32"))
        else:
            variables[name] = (
                ("time", "latitude", "longitude"),
                np.ones((1, 1, 1), dtype="float32"),
            )
    xr.Dataset(variables, coords=coordinates).to_zarr(
        target, mode="w", consolidated=True, zarr_format=2
    )

    problems, _ = mirror._inspect_chirps_zarr(target)

    assert f"target_variables_missing:{missing}" in problems


def test_chirps_inspection_records_optional_variable_shapes(tmp_path: Path) -> None:
    target = tmp_path / "data/processed/zarr/features/chirps_native_weekly_targets.zarr"
    coordinates = {
        "time": pd.DatetimeIndex(["2026-01-04"]),
        "latitude": np.asarray([-10.0], dtype="float32"),
        "longitude": np.asarray([-50.0], dtype="float32"),
    }
    variables: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for name in sorted(mirror.CHIRPS_REQUIRED_VARIABLES):
        if name == "pixel_id":
            variables[name] = (("latitude", "longitude"), np.asarray([[0]], dtype="int64"))
        elif name == "brazil_center":
            variables[name] = (("latitude", "longitude"), np.asarray([[True]], dtype=bool))
        elif name == "brazil_fraction":
            variables[name] = (("latitude", "longitude"), np.asarray([[1.0]], dtype="float32"))
        else:
            variables[name] = (
                ("time", "latitude", "longitude"),
                np.ones((1, 1, 1), dtype="float32"),
            )
    variables["optional_spi_audit_layer"] = (
        ("latitude", "longitude"),
        np.ones((1, 1), dtype="float32"),
    )
    xr.Dataset(variables, coords=coordinates).to_zarr(
        target, mode="w", consolidated=True, zarr_format=2
    )

    _, details = mirror._inspect_chirps_zarr(target)

    assert details["variable_contract"]["optional_spi_audit_layer"] == {
        "dimensions": "latitude;longitude",
        "shape": "1x1",
    }


def test_phase5_mixed_component_gate_is_not_global_pass(tmp_path: Path) -> None:
    run = tmp_path / "data/processed/runs/official/fase5/F5_mixed"
    rows = []
    for horizon, passed in ((0, False), (4, False), (8, False), (12, False), (24, True)):
        rows.append(
            {
                "component": f"classification_h{horizon:02d}",
                "gate_pass": passed,
            }
        )
    for target in ("Y_pico", "Y_tempo_para_pico_sem", "Y_duracao_sem"):
        rows.append({"component": f"event_dimension_{target}", "gate_pass": True})
    gate_path = _write_csv(run / "tables/scientific_gate.csv", rows)
    audit = {
        "phase": 5,
        "path": str(run.relative_to(tmp_path)).replace("\\", "/"),
        "table_names": [gate_path.name],
        "artifact_valid": True,
    }

    gate = mirror._gate_from_run(tmp_path, audit)

    assert gate["state"] == "failed"
    assert gate["passed"] is False
    assert "classificação=1/5" in gate["detail"]


def test_chirps_deep_receipt_binds_current_content_manifest_and_code(tmp_path: Path) -> None:
    builder = _write(tmp_path / "scripts/build_phase4_chirps_targets.py", "# builder\n")
    target_module = _write(
        tmp_path / "src/nino_brasil/targets/chirps_native.py", "# target\n"
    )
    target = tmp_path / "data/processed/zarr/features/chirps_native_weekly_targets.zarr"
    _write(target / ".zattrs", "{}\n")
    manifest_path = _write(
        tmp_path / "data/interim/chirps_weekly_native_blocks/abcd/manifest.json",
        "{}\n",
    )
    manifest = {
        "build_id": "F4TARGET_fixture",
        "signature_sha256": "a" * 64,
        "promoted_target_data_content_sha256": "b" * 64,
        "promoted_target_data_state_sha256": "c" * 64,
        "builder_script_sha256": mirror._sha256_file(builder),
        "target_module_sha256": mirror._sha256_file(target_module),
        "promoted_utc": "2026-07-13T10:00:00+00:00",
    }
    _write(manifest_path, json.dumps(manifest))
    report = {
        "schema_version": "nino26-chirps-deep-validation/v1",
        "checked_at_utc": "2026-07-13T11:00:00+00:00",
        "target_path": str(target.relative_to(tmp_path)).replace("\\", "/"),
        "target_data_state_sha256": "c" * 64,
        "builder_script_sha256": mirror._sha256_file(builder),
        "target_module_sha256": mirror._sha256_file(target_module),
        "build_manifest_sha256": mirror._sha256_file(manifest_path),
        "validation": {
            "valid": True,
            "problems": [],
            "manifest": str(manifest_path.relative_to(tmp_path)).replace("\\", "/"),
            "build_id": "F4TARGET_fixture",
            "block_signature_sha256": "a" * 64,
            "target_data_content_sha256": "b" * 64,
        },
    }
    _write(
        tmp_path / mirror.CHIRPS_DEEP_VALIDATION_REPORT,
        json.dumps(report),
    )

    assert mirror._audit_chirps_deep_receipt(tmp_path, manifest, target) == []

    report["validation"]["target_data_content_sha256"] = "d" * 64
    _write(
        tmp_path / mirror.CHIRPS_DEEP_VALIDATION_REPORT,
        json.dumps(report),
    )
    problems = mirror._audit_chirps_deep_receipt(tmp_path, manifest, target)
    assert "chirps_deep_validation_identity_mismatch:target_data_content_sha256" in problems


def test_phase4_requires_eight_conditions_and_seven_hashed_sidecars(tmp_path: Path) -> None:
    f4c_runner = _write(tmp_path / "scripts/run_fase4c_regional.py", "# f4c\n")
    f4d_runner = _write(tmp_path / "scripts/run_fase4d_targets.py", "# f4d\n")
    lag_module = _write(tmp_path / "src/nino_brasil/stats/lag_analysis.py", "# lag\n")
    stats = tmp_path / "data/processed/parquet/statistics"
    common = {
        "parent_f3_run_id": "f3-run",
        "parent_f3_artifact_sha256": "3" * 64,
        "target_build_id": "target-build",
        "target_block_signature_sha256": "4" * 64,
        "target_contract_version": mirror.CHIRPS_TARGET_CONTRACT,
        "grid_hash_sha256": mirror.CHIRPS_FROZEN_GRID_HASH,
        "f4c_runner_sha256": mirror._sha256_file(f4c_runner),
        "lag_analysis_module_sha256": mirror._sha256_file(lag_module),
    }
    f4c_row = {
        **common,
        "analysis_run_id": "f4c-run",
        "condicao_fonte": "el_nino_genese",
    }
    for name in (
        "phase4C_native_lags_por_unidade.csv",
        "phase4C_native_predictor_treatment.csv",
        "phase4C_native_best_lag_pixel_key.csv",
        "phase4C_native_field_significance.csv",
    ):
        _write_csv(stats / name, [f4c_row])
    f4d_row = {
        **common,
        "analysis_run_id": "f4d-run",
        "parent_f4c_run_id": "f4c-run",
        "f4d_runner_sha256": mirror._sha256_file(f4d_runner),
        "tipo_enso_fonte": "el_nino",
        "fase_fonte_em_t_menos_lag": "genese",
        "coverage_gate_pass": True,
        "hypothesis_gate": "supports_with_at_least_3_complementary_target_families",
    }
    for name in (
        "phase4D_native_gate_event_jackknife.csv",
        "phase4D_native_hypothesis_summary.csv",
        "phase4D_native_target_coverage.csv",
    ):
        _write_csv(stats / name, [f4d_row])

    result = mirror.collect_phase4_outputs(
        tmp_path,
        mirror._status(
            "promoted",
            "fixture",
            build_id="target-build",
            block_signature_sha256="4" * 64,
            grid_hash_sha256=mirror.CHIRPS_FROZEN_GRID_HASH,
        ),
        {"run_id": "f3-run", "phase_table_sha256": "3" * 64},
    )

    assert result["official"]["state"] == "invalidated"
    assert result["gate"]["state"] == "failed"
    assert result["f4c_conditions"] == ["el_nino_genese"]
    assert any("f4c_condition_catalog_mismatch" in item for item in result["official"]["problems"])
    assert any("f4d_condition_catalog_mismatch" in item for item in result["official"]["problems"])
    assert sum("missing_output_manifest" in item for item in result["official"]["problems"]) == 7


def test_phase4_sidecar_schema_is_hash_verifiable_by_the_mirror(tmp_path: Path) -> None:
    source = _write(tmp_path / "data/source.csv", "source\n")
    output = _write_csv(tmp_path / "data/output.csv", [{"value": 1}])
    _write(
        Path(f"{output}.manifest.json"),
        json.dumps(
            {
                "run_id": "f4-run",
                "contract": {"phase": "F4"},
                "artifact": {
                    "path": str(output.relative_to(tmp_path)).replace("\\", "/"),
                    "sha256": mirror._sha256_file(output),
                },
                "inputs": [_input_record(source)],
            }
        ),
    )

    assert mirror._audit_semantic_csv_sidecar(tmp_path, output, "f4-run") == []


def test_phase4_smoke_requires_complete_current_quick_contract(tmp_path: Path) -> None:
    chirps, phase3 = _write_phase4_quick_fixture(tmp_path)

    result = mirror.collect_phase4_smoke(tmp_path, chirps, phase3)

    assert result["state"] == "complete"
    assert result["passed"] is True
    assert result["valid_outputs"] == 7
    assert result["valid_csv_outputs"] == 4
    assert result["run_id"] == "F4C_QUICK_fixture"
    assert result["enso_phase_condition_count"] == 8
    assert result["problems"] == []


def test_phase4_smoke_rejects_partial_output_catalog(tmp_path: Path) -> None:
    missing_name = "phase4C_native_best_lag_pixel_key_quick.csv"
    chirps, phase3 = _write_phase4_quick_fixture(
        tmp_path,
        omit=frozenset({missing_name}),
    )

    result = mirror.collect_phase4_smoke(tmp_path, chirps, phase3)

    assert result["state"] == "invalidated"
    assert result["passed"] is False
    assert result["valid_outputs"] == 6
    assert result["valid_csv_outputs"] == 3
    assert f"missing_quick_output:{missing_name}" in result["problems"]
    assert f"missing_quick_output_manifest:{missing_name}" in result["problems"]


@pytest.mark.parametrize(
    ("missing_name", "expected_problem"),
    [
        (
            "phase4C_native_best_lag_pixel_quick.parquet",
            "missing_quick_output:phase4C_native_best_lag_pixel_quick.parquet",
        ),
        (
            Path(mirror.F4C_QUICK_ATLAS).name,
            "missing_quick_atlas",
        ),
    ],
)
def test_phase4_smoke_requires_parquet_and_atlas_outputs(
    tmp_path: Path,
    missing_name: str,
    expected_problem: str,
) -> None:
    chirps, phase3 = _write_phase4_quick_fixture(
        tmp_path,
        omit=frozenset({missing_name}),
    )

    result = mirror.collect_phase4_smoke(tmp_path, chirps, phase3)

    assert result["state"] == "invalidated"
    assert expected_problem in result["problems"]


def test_phase4_smoke_rejects_atlas_tree_or_ibge_bundle_drift(tmp_path: Path) -> None:
    chirps, phase3 = _write_phase4_quick_fixture(tmp_path)
    atlas = tmp_path / mirror.F4C_QUICK_ATLAS
    _write(atlas / "tampered-after-manifest.txt", "drift\n")
    _write(
        tmp_path / "data/interim/ibge/BR_Regioes_2024/BR_Regioes_2024.prj",
        "geometry drift\n",
    )

    result = mirror.collect_phase4_smoke(tmp_path, chirps, phase3)

    assert result["state"] == "invalidated"
    assert "quick_atlas_tree_hash_mismatch" in result["problems"]
    assert "quick_current_identity_mismatch:ibge_regions_bundle_sha256" in result["problems"]
    assert any("sha256_mismatch" in item for item in result["problems"])


@pytest.mark.parametrize(
    ("condition_catalog", "selection_contract", "expected_problem"),
    [
        (
            frozenset(set(mirror.ENSO_PHASE_CONDITIONS) - {"la_nina_decaimento"}),
            "quick:key-predictor",
            "quick_condition_catalog_mismatch",
        ),
        (
            mirror.ENSO_PHASE_CONDITIONS,
            "canonical_all_31_physical_variables",
            "quick_selection_contract_mismatch",
        ),
    ],
)
def test_phase4_smoke_rejects_condition_or_selection_contradiction(
    tmp_path: Path,
    condition_catalog: frozenset[str],
    selection_contract: str,
    expected_problem: str,
) -> None:
    chirps, phase3 = _write_phase4_quick_fixture(
        tmp_path,
        condition_catalog=condition_catalog,
        selection_contract=selection_contract,
    )

    result = mirror.collect_phase4_smoke(tmp_path, chirps, phase3)

    assert result["state"] == "invalidated"
    assert any(item.startswith(expected_problem) for item in result["problems"])


def test_phase4_smoke_rejects_mixed_run_and_current_code_drift(tmp_path: Path) -> None:
    drifted_name = "phase4C_native_field_significance_quick.csv"
    chirps, phase3 = _write_phase4_quick_fixture(
        tmp_path,
        run_overrides={drifted_name: "F4C_QUICK_other"},
    )
    _write(tmp_path / "scripts/run_fase4c_regional.py", "# code drift after quick run\n")

    result = mirror.collect_phase4_smoke(tmp_path, chirps, phase3)

    assert result["state"] == "invalidated"
    assert "quick_identity_catalog_mismatch:analysis_run_id:2" in result["problems"]
    assert "quick_current_identity_mismatch:f4c_runner_sha256" in result["problems"]
    assert any("output_manifest_input" in item and "sha256_mismatch" in item for item in result["problems"])


def test_phase7_cube_directory_without_arrays_is_not_a_data_dependency(tmp_path: Path) -> None:
    cube = tmp_path / "data/processed/zarr/modeling/phase7_pacific_weekly.zarr"
    _write(cube / ".zattrs", json.dumps({"phase": 7}))

    status = mirror.collect_phase7_cube_status(tmp_path)

    assert status["state"] == "failed"
    assert "phase7_cube_metadata_missing:.zgroup" in status["problems"]
    assert "phase7_cube_has_no_arrays_or_chunks" in status["problems"]


def test_phase3_command_defers_global_figure_lineage_to_final_validation() -> None:
    commands = mirror._commands()

    assert all("--require-semantic-lineage" not in command for command in commands["F3"])
    assert any("run_final_validation.py" in command for command in commands["validação final"])


def test_source_tree_drift_invalidates_artifact_run(tmp_path: Path) -> None:
    directory = _make_run(
        tmp_path,
        phase=7,
        mode="official",
        run_id="F7_before_drift",
    )
    assert mirror.audit_artifact_run(tmp_path, directory)["eligible"] is True

    _write(tmp_path / "src/nino_brasil/core.py", "VALUE = 2\n")
    audit = mirror.audit_artifact_run(tmp_path, directory)

    assert audit["eligible"] is False
    assert any("tree_sha256_mismatch" in problem for problem in audit["problems"])


def test_official_artifact_run_requires_full_semantic_table_catalog(tmp_path: Path) -> None:
    directory = _make_run(
        tmp_path,
        phase=5,
        mode="official",
        run_id="F5_missing_importance",
        model="rf",
    )
    table_manifest_path = directory / "tables_manifest.csv"
    rows = [
        row
        for row in _read_csv_fixture(table_manifest_path)
        if row["table"] != "state_importance_oos.csv"
    ]
    _write_csv(table_manifest_path, rows)
    (directory / "tables/state_importance_oos.csv").unlink()
    manifest_path = directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["n_tables"] = len(rows)
    manifest["tables_manifest_sha256"] = mirror._sha256_file(table_manifest_path)
    _write(manifest_path, json.dumps(manifest))

    audit = mirror.audit_artifact_run(tmp_path, directory)

    assert audit["eligible"] is False
    assert any(
        problem.startswith("missing_required_tables:")
        and "state_importance_oos.csv" in problem
        for problem in audit["problems"]
    )


def _read_csv_fixture(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_artifact_run_input_hashes_are_not_optional(tmp_path: Path) -> None:
    directory = _make_run(
        tmp_path,
        phase=7,
        mode="smoke",
        run_id="F7_missing_source_hash",
    )
    manifest_path = directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"][0].pop("tree_sha256")
    _write(manifest_path, json.dumps(manifest))

    audit = mirror.audit_artifact_run(tmp_path, directory)

    assert audit["eligible"] is False
    assert any("missing_or_invalid_tree_sha256" in problem for problem in audit["problems"])


def test_rf_xgb_pair_requires_common_comparison_signature(tmp_path: Path) -> None:
    _make_run(
        tmp_path,
        phase=5,
        mode="official",
        run_id="F5_comparable_rf",
        model="rf",
    )
    xgb = _make_run(
        tmp_path,
        phase=5,
        mode="official",
        run_id="F5_incompatible_xgb",
        model="xgb",
    )
    manifest_path = xgb / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parameters"]["noise_scale"] = 0.99
    manifest["parameters_sha256"] = mirror._json_hash(manifest["parameters"])
    _write(manifest_path, json.dumps(manifest))

    selection = mirror._select_artifact_runs(tmp_path, 5, "official")

    assert selection["state"] == "invalidated"
    assert selection["comparison_compatible"] is False
    assert selection["comparison_problem"] == "comparison_signature_mismatch"


def test_f6_shard_is_not_an_official_merge(tmp_path: Path) -> None:
    directory = _make_run(
        tmp_path,
        phase=6,
        mode="official",
        run_id="F6_shard_only",
        model="rf",
        shard_contract=True,
    )

    audit = mirror.audit_artifact_run(tmp_path, directory)

    assert audit["eligible"] is False
    assert "official_f6_is_shard_not_merge" in audit["problems"]
    assert "missing_required_gate_table:field_gate.csv" in audit["problems"]


def test_figure_lineage_requires_exact_expected_run_set(tmp_path: Path) -> None:
    _make_f3_catalog(tmp_path, "run-b", ["Fig_3A01", "Fig_3B01"])

    lineage = mirror.collect_figure_lineage(tmp_path, 3, ["run-a"])

    assert lineage["state"] == "failed"
    assert any("run_id_set_mismatch" in problem for problem in lineage["problems"])


def test_f3_figure_catalog_must_be_complete(tmp_path: Path) -> None:
    _make_f3_catalog(tmp_path, "run-a", ["Fig_3A01"])

    lineage = mirror.collect_figure_lineage(tmp_path, 3, ["run-a"])

    assert lineage["state"] == "failed"
    assert lineage["expected_figure_count"] == len(mirror.PHASE3_FIGURE_CODES)
    assert "Fig_3L04" in lineage["missing_codes"]
    assert any("figure_catalog_count_mismatch" in problem for problem in lineage["problems"])


def test_f3_figure_lineage_revalidates_numeric_and_source_hashes(tmp_path: Path) -> None:
    run_id = "run-a"
    _make_f3_catalog(tmp_path, run_id)
    assert mirror.collect_figure_lineage(tmp_path, 3, [run_id])["state"] == "passed"

    numeric = tmp_path / "data/processed/numeric-tables/fase3/Fig_3A01/source.csv"
    _write(numeric, "week,value\n2026-01-04,999.0\nextra,row\n")
    numeric_result = mirror.collect_figure_lineage(tmp_path, 3, [run_id])
    assert numeric_result["state"] == "failed"
    assert "numeric_table_hash_mismatch:Fig_3A01:source.csv" in numeric_result["problems"]

    _make_f3_catalog(tmp_path, run_id)
    source = tmp_path / "data/processed/parquet/statistics/Fig_3A01_source.csv"
    _write(source, "week,value\n2026-01-04,999.0\nextra,row\n")
    source_result = mirror.collect_figure_lineage(tmp_path, 3, [run_id])
    assert source_result["state"] == "failed"
    assert "source_artifact_hash_mismatch:Fig_3A01:source.csv" in source_result["problems"]


def test_atomic_pair_uses_same_generation_and_leaves_no_temp_files(tmp_path: Path) -> None:
    snapshot = mirror.collect_project_snapshot(
        tmp_path,
        generated_at=datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc),
    )
    snapshot_path = tmp_path / "data/audit/project_status_snapshot.json"
    panel_path = tmp_path / "painel_executivo.md"

    snapshot_hash, _ = mirror.write_outputs_atomic(snapshot, snapshot_path, panel_path)

    written = json.loads(snapshot_path.read_text(encoding="utf-8"))
    panel = panel_path.read_text(encoding="utf-8")
    assert written["generation_id"] == snapshot["generation_id"]
    assert snapshot["generation_id"] in panel
    assert snapshot_hash in panel
    assert [phase["phase"] for phase in written["phases"]] == [
        f"F{number}" for number in range(1, 9)
    ]
    assert all("audit_surfaces" in phase for phase in written["phases"])
    assert "workspace_hygiene" in written
    assert "## Superfícies de auditoria por fase" in panel
    assert "## Runs selecionados F5–F8" in panel
    assert "## Validação final e higiene" in panel
    assert "Código pronto" in panel
    assert "não implica execução" in panel
    assert "chirps-native-weekly-v4" in panel
    assert not list(tmp_path.rglob("*.tmp"))
    assert any(
        issue["phase"] == "GLOBAL"
        and issue["dimension"] == "final_validation"
        and issue["state"] == "missing"
        for issue in snapshot["quality_issues"]
    )


def test_atomic_pair_rolls_back_when_second_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_path = _write(
        tmp_path / "data/audit/project_status_snapshot.json",
        '{"generation_id":"old"}\n',
    )
    panel_path = _write(tmp_path / "painel_executivo.md", "old panel\n")
    old_snapshot = snapshot_path.read_text(encoding="utf-8")
    old_panel = panel_path.read_text(encoding="utf-8")
    real_replace = mirror.os.replace
    calls = 0

    def fail_second_forward_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(mirror.os, "replace", fail_second_forward_replace)
    new_snapshot = mirror.collect_project_snapshot(
        tmp_path,
        generated_at=datetime(2026, 7, 13, 13, 0, tzinfo=timezone.utc),
    )

    with pytest.raises(OSError, match="simulated second replace failure"):
        mirror.write_outputs_atomic(
            new_snapshot,
            snapshot_path,
            panel_path,
        )

    assert snapshot_path.read_text(encoding="utf-8") == old_snapshot
    assert panel_path.read_text(encoding="utf-8") == old_panel
    assert not list(tmp_path.rglob("*.tmp"))


def test_notebook_contract_rejects_unexecuted_code(tmp_path: Path) -> None:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"language_info": {"version": "3.12"}},
        "cells": [
            {
                "id": "header",
                "cell_type": "markdown",
                "metadata": {},
                "source": "<!-- NINO26-CABECALHO v1 -->",
            },
            {
                "id": "code",
                "cell_type": "code",
                "metadata": {},
                "source": "print('not executed')",
                "execution_count": None,
                "outputs": [],
            },
            {
                "id": "footer",
                "cell_type": "markdown",
                "metadata": {},
                "source": "<!-- NINO26-REFERENCIAS v1 -->",
            },
        ],
    }
    _write(tmp_path / "notebooks/fase3/3A_fixture.ipynb", json.dumps(notebook))

    status = mirror.collect_notebook_contract(tmp_path, 3)

    assert status["state"] == "failed"
    assert any("unexecuted_code_cells" in problem for problem in status["problems"])


def test_phase4_notebook_contract_excludes_declared_historical_viewers(tmp_path: Path) -> None:
    canonical = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "id": "header",
                "cell_type": "markdown",
                "metadata": {},
                "source": "<!-- NINO26-CABECALHO v1 -->",
            },
            {
                "id": "code",
                "cell_type": "code",
                "metadata": {},
                "source": "print('executed')",
                "execution_count": 1,
                "outputs": [],
            },
            {
                "id": "footer",
                "cell_type": "markdown",
                "metadata": {},
                "source": "<!-- NINO26-REFERENCIAS v1 -->",
            },
        ],
    }
    historical = json.loads(json.dumps(canonical))
    historical["metadata"]["nino26"] = {
        "canonical": False,
        "promotion_status": "excluded",
        "execution_policy": "read-only-viewer",
    }
    historical["cells"][1]["execution_count"] = None
    for name in ("4C_sinal_pixel_lags.ipynb", "4D_clusters_alvo.ipynb"):
        _write(tmp_path / "notebooks/fase4" / name, json.dumps(canonical))
    for name in (
        "4_0_fase4_abertura.ipynb",
        "4A_ciclo_enso_fases.ipynb",
        "4B_variaveis_determinantes_fases.ipynb",
    ):
        _write(tmp_path / "notebooks/fase4" / name, json.dumps(historical))

    status = mirror.collect_notebook_contract(tmp_path, 4)

    assert status["state"] == "passed"
    assert status["notebook_count"] == 2
    assert status["excluded_notebook_count"] == 3
    assert status["problems"] == []


def test_final_validation_reuses_fresh_chirps_deep_receipt() -> None:
    commands = dict(final_validation._commands(sys.executable, ROOT))

    argv = commands["chirps_native_contract"]
    assert "--reuse-valid-receipt" in argv
    assert "data/audit/chirps_deep_validation.json" in argv
