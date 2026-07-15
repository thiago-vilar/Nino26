#!/usr/bin/env python3
"""Gera o espelho vivo e auditável do projeto NINO-BRASIL.

O painel não infere autoridade científica por nome de pasta ou data de
modificação. Primeiro valida contrato, hashes, proveniência e modalidade; só
então usa ``finished_at`` para escolher entre candidatos igualmente elegíveis.

Saídas (gravadas atomicamente, com o mesmo ``generation_id``):

* ``data/audit/project_status_snapshot.json`` — fonte estruturada do painel;
* ``painel_executivo.md`` — leitura humana local, ignorada pelo Git.

O coletor é deliberadamente somente leitura. Ele não executa notebooks,
modelos, downloads ou validadores caros.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "painel_executivo.md"
SNAPSHOT_PATH = ROOT / "data/audit/project_status_snapshot.json"
SCHEMA_VERSION = "nino26-project-status-snapshot/v1"
SEMANTIC_LEVEL = "semantic_source"
FINAL_VALIDATION_CONTEXT_PATHS: tuple[str, ...] = (
    "src",
    "scripts",
    "tests",
    "notebooks",
    "configs",
    "requirements.txt",
    "requirements.lock.txt",
    "pyproject.toml",
)
FINAL_VALIDATION_CHECK_IDS: frozenset[str] = frozenset(
    {
        "python_compile",
        "dependency_check",
        "pytest",
        "notebook_contract",
        "figure_contract",
        "phase2_contract",
        "phase3_semantic",
        "chirps_native_contract",
        "phase7_cube_contract",
        "git_diff_check",
    }
)
_IGNORED_VALIDATION_CONTEXT_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
}


@dataclass(frozen=True)
class PhaseSpec:
    number: int
    title: str
    code_paths: tuple[str, ...]
    next_action: str


PHASE_SPECS: tuple[PhaseSpec, ...] = (
    PhaseSpec(
        1,
        "Ingestão local auditável",
        ("scripts/data_pipeline.py", "src/nino_brasil/data/audit.py"),
        "Auditar o ledger e retomar somente tarefas explicitamente incompletas.",
    ),
    PhaseSpec(
        2,
        "Master semanal de 31 variáveis",
        ("scripts/build_master_weekly.py", "src/nino_brasil/data/phase2_master.py"),
        "Manter as 31 variáveis físicas e o metadado de fonte sob o mesmo contrato.",
    ),
    PhaseSpec(
        3,
        "Estatística do ciclo EN/LN",
        (
            "scripts/phase3_en_ln.py",
            "scripts/run_fase3_all.py",
            "scripts/verify_phase3_semantic_tables.py",
        ),
        "Fechar tabelas, notebooks e figuras no mesmo run semântico.",
    ),
    PhaseSpec(
        4,
        "Estatística CHIRPS pixel-a-pixel",
        (
            "scripts/build_phase4_chirps_targets.py",
            "scripts/run_fase4c_regional.py",
            "scripts/run_fase4d_targets.py",
        ),
        "Promover o alvo nativo; depois executar 4C/4D e auditar cobertura e linhagem.",
    ),
    PhaseSpec(
        5,
        "Ciclo ENSO com RF/XGBoost",
        ("scripts/run_fase5_cycle_ml.py",),
        "Executar RF e XGBoost oficiais; preservar inclusive gates negativos.",
    ),
    PhaseSpec(
        6,
        "Brasil pixel-a-pixel com RF/XGBoost",
        ("scripts/run_fase6_brazil_ml.py", "scripts/merge_fase6_shards.py"),
        "Executar shards RF/XGBoost e aceitar somente merges com cobertura nativa integral.",
    ),
    PhaseSpec(
        7,
        "Ciclo ENSO com ConvLSTM",
        ("scripts/build_phase7_pacific_cube.py", "scripts/run_fase7_cycle_convlstm.py"),
        "Executar oficialmente somente após uma referência F5 compatível e elegível.",
    ),
    PhaseSpec(
        8,
        "Brasil nativo com ConvLSTM probabilística",
        ("scripts/run_fase8_brazil_convlstm.py",),
        "Executar oficialmente somente contra merge F6 compatível e cobertura pareada.",
    ),
)

PHASE3_TABLES: tuple[str, ...] = (
    "phase3_events_en_ln",
    "phase3_fases_semanais_en_ln",
    "phase3_event_lifecycle_en_ln",
    "phase3_duracao_por_tipo_classe",
    "phase3_peak_band_sensitivity",
    "phase3_preprocessing_contract",
    "phase3_fase_stats_variaveis",
    "phase3_discriminantes_por_periodo",
    "phase3_pca_por_fase",
    "phase3_pca_loadings_por_fase",
    "phase3_rolling_origin_targets",
    "phase3_rolling_origin_folds",
    "phase3_lag_scan_en_ln_fases",
    "phase3_best_lags_fdr",
    "phase3_lag_event_bootstrap_summary",
    "phase3_lag_event_bootstrap_replicates",
)

# Catálogo canônico produzido pelos notebooks F3. A promoção visual só é
# completa quando o conjunto é exato; uma amostra de figuras não representa a
# análise científica inteira.
PHASE3_FIGURE_CODES: tuple[str, ...] = (
    "Fig_3A01",
    "Fig_3A02",
    "Fig_3A03",
    "Fig_3B01",
    "Fig_3B02",
    "Fig_3B03",
    "Fig_3B04",
    "Fig_3C01",
    "Fig_3C02",
    "Fig_3D01",
    "Fig_3D02",
    "Fig_3E01",
    "Fig_3E02",
    "Fig_3F01",
    "Fig_3F02",
    "Fig_3G01",
    "Fig_3G02",
    "Fig_3G03",
    "Fig_3H01",
    "Fig_3H02",
    "Fig_3H03",
    "Fig_3I01",
    "Fig_3I02",
    "Fig_3I03",
    "Fig_3I04",
    "Fig_3K01",
    "Fig_3K02",
    "Fig_3K03",
    "Fig_3L01",
    "Fig_3L02",
    "Fig_3L03",
    "Fig_3L04",
)
FIGURE_CODES_BY_PHASE: Mapping[int, frozenset[str]] = {
    3: frozenset(PHASE3_FIGURE_CODES),
    4: frozenset(
        {
            "Fig_4C01_lags_regiao_bioma_el_nino",
            "Fig_4C02_lags_regiao_bioma_la_nina",
            "Fig_4C03_mapa_pixel_el_nino_pico",
            "Fig_4C04_mapa_pixel_la_nina_pico",
            "Fig_4D01_mapa_clusters_pixels_nativos",
            "Fig_4D02_perfis_clusters_fdr",
            "Fig_4D03_gate_multialvo_eventos",
        }
    ),
    5: frozenset(
        {"Fig_5A01_rf", "Fig_5A01_xgb", "Fig_5A02_rf", "Fig_5A02_xgb"}
    ),
    6: frozenset({"Fig_6A01_rf", "Fig_6A01_xgb"}),
    7: frozenset({"Fig_7A01", "Fig_7A02"}),
    8: frozenset({"Fig_8A01", "Fig_8A02"}),
}

PHASE2_VALIDATION_CHECKS: tuple[str, ...] = (
    "indice_datetime",
    "indice_monotonico_crescente",
    "sem_semanas_duplicadas",
    "grade_semanal_W-SUN_regular",
    "eixo_1981_ao_ano_atual",
    "contrato_31_variaveis_fisicas",
    "source_code_apenas_metadado",
    "nenhuma_variavel_totalmente_vazia",
    "cobertura_final_alinhada_12_semanas",
    "variaveis_fisicas_numericas",
    "sem_inf_nas_variaveis",
    "codigos_fonte_validos",
    "fontes_oceanicas_nao_regridem_no_tempo",
    "escala_plausivel:tau_x_anom",
    "escala_plausivel:u10_anom",
    "escala_plausivel:v10_anom",
    "escala_plausivel:mslp_anom",
    "escala_plausivel:tcwv_anom",
    "escala_plausivel:slhf_anom",
    "escala_plausivel:sshf_anom",
    "escala_plausivel:ssr_anom",
    "escala_plausivel:str_anom",
    "escala_plausivel:u850_anom",
    "escala_plausivel:u200_anom",
    "escala_plausivel:omega850_anom",
    "escala_plausivel:omega500_anom",
    "escala_plausivel:div850_anom",
)

PHASE2_PHYSICAL_COLUMNS: tuple[str, ...] = (
    "nino34_ssta",
    "d20_m",
    "tilt_m",
    "tilt_slope",
    "ohc_0_100",
    "ohc_0_300",
    "ohc_0_700",
    "ohc_300_700",
    "ssh_m",
    "wwv",
    "t50m",
    "t100m",
    "t150m",
    "t200m",
    "t300m",
    "t500m",
    "t700m",
    "tau_x_anom",
    "u10_anom",
    "v10_anom",
    "mslp_anom",
    "tcwv_anom",
    "slhf_anom",
    "sshf_anom",
    "ssr_anom",
    "str_anom",
    "u850_anom",
    "u200_anom",
    "omega850_anom",
    "omega500_anom",
    "div850_anom",
)

PHASE2_REQUIRED_PATHS: Mapping[str, frozenset[str]] = {
    "code": frozenset(
        {
            "scripts/build_master_weekly.py",
            "src/nino_brasil/data/phase2_master.py",
            "src/nino_brasil/data/audit.py",
        }
    ),
    "inputs": frozenset(
        {
            "data/processed/parquet/features/nino34_physical_signal.csv",
            "data/processed/parquet/features/era5_nino34_daily_cache.parquet",
        }
    ),
    "outputs": frozenset(
        {
            "data/processed/parquet/features/nino34_master_weekly.csv",
            "data/processed/parquet/features/nino34_master_weekly_source_adjusted_v1.csv",
            "data/processed/parquet/statistics/phase2_master_audit.csv",
            "data/processed/parquet/statistics/phase2_master_source_adjusted_v1_audit.csv",
            "data/processed/parquet/statistics/phase2_master_validation.csv",
            "data/processed/parquet/statistics/phase2_ocean_source_seam_audit.csv",
            "data/processed/parquet/statistics/phase2_variable_contract.csv",
            "data/processed/parquet/statistics/phase2_ctd_validation.csv",
        }
    ),
}

CHIRPS_TARGET_CONTRACT = "chirps-native-weekly-v4"
CHIRPS_BLOCK_CONTRACT = "chirps-native-weekly-spatial-blocks-v4"
CHIRPS_FROZEN_GRID_HASH = "4422ba2d57f6d8665401ae2d437c12e18b07fc9dbc76c613f90c4ccb158d3c1d"
CHIRPS_DEEP_VALIDATION_REPORT = "data/audit/chirps_deep_validation.json"
CHIRPS_REQUIRED_VARIABLES: frozenset[str] = frozenset(
    {
        "precip_weekly_mm",
        "valid_day_count",
        "expected_day_count",
        "week_complete",
        "precip_anomaly_mm",
        "precip_robust_z",
        "precip_robust_percentile",
        "climatology_median_mm",
        "climatology_robust_scale_mm",
        "climatology_pooled_residual_mad_scale",
        "climatology_pooled_residual_l1_scale",
        "climatology_scale_source_code",
        "r95p_weekly_climatology_positive_week_count",
        "r95p_weekly_climatology_pooled_positive_l1_scale",
        "r95p_weekly_climatology_tail_threshold_floor",
        "r95p_weekly_climatology_tail_fallback_code",
        "r99p_weekly_climatology_positive_week_count",
        "r99p_weekly_climatology_pooled_positive_l1_scale",
        "r99p_weekly_climatology_tail_threshold_floor",
        "r99p_weekly_climatology_tail_fallback_code",
        "pixel_id",
        "brazil_fraction",
        "brazil_center",
    }
)
ENSO_PHASE_CONDITIONS: frozenset[str] = frozenset(
    f"{event_type}_{phase}"
    for event_type in ("el_nino", "la_nina")
    for phase in ("genese", "crescimento", "pico", "decaimento")
)
F4C_IDENTITY_COLUMNS: tuple[str, ...] = (
    "analysis_run_id",
    "parent_f3_run_id",
    "parent_f3_artifact_sha256",
    "target_build_id",
    "target_block_signature_sha256",
    "target_contract_version",
    "grid_hash_sha256",
    "ibge_regions_bundle_sha256",
    "ibge_biomes_bundle_sha256",
    "ibge_geometry_bundle_sha256",
    "f4c_runner_sha256",
    "lag_analysis_module_sha256",
)
F4C_PREDICTOR_CONTRACT_COLUMNS: tuple[str, ...] = (
    "selection_contract",
    "predictor_count",
    "predictor_catalog_sha256",
)
F4C_QUICK_CSV_NAMES: tuple[str, ...] = (
    "phase4C_native_lags_por_unidade_quick.csv",
    "phase4C_native_predictor_treatment_quick.csv",
    "phase4C_native_best_lag_pixel_key_quick.csv",
    "phase4C_native_field_significance_quick.csv",
)
F4C_QUICK_PARQUET_NAMES: tuple[str, ...] = (
    "phase4C_native_cobertura_unidades_quick.parquet",
    "phase4C_native_best_lag_pixel_quick.parquet",
)
F4C_QUICK_TABULAR_NAMES: tuple[str, ...] = (
    *F4C_QUICK_CSV_NAMES,
    *F4C_QUICK_PARQUET_NAMES,
)
F4C_QUICK_CONDITION_TABLE_NAMES: frozenset[str] = frozenset(
    {
        "phase4C_native_lags_por_unidade_quick.csv",
        "phase4C_native_best_lag_pixel_key_quick.csv",
        "phase4C_native_field_significance_quick.csv",
        "phase4C_native_best_lag_pixel_quick.parquet",
    }
)
F4C_QUICK_ATLAS = "data/processed/zarr/statistics/phase4C_native_pixel_lags_quick.zarr"
F4C_ATLAS_VARIABLES: frozenset[str] = frozenset(
    {"r", "p", "q_fdr_bh", "fdr_bh_reject", "n_eff_bretherton"}
)
F4D_CONFIRMATORY_TARGETS: frozenset[str] = frozenset(
    {
        "precip_robust_z",
        "spi_gamma_3m_weekly_origin",
        "r95p_weekly_robust_z",
        "r99p_weekly_robust_z",
        "cdd_within_week_robust_z",
    }
)
F4D_REGIONS: frozenset[str] = frozenset(
    {"Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"}
)
PHASE7_CUBE_VARIABLES: frozenset[str] = frozenset(
    {
        "thetao_surface_c",
        "thetao_50m_c",
        "thetao_100m_c",
        "thetao_150m_c",
        "thetao_200m_c",
        "thetao_300m_c",
        "salinity_surface_psu",
        "ssh_m",
        "valid_day_count",
        "expected_day_count",
        "complete_week",
    }
)
PHASE7_CUBE_SIZES = {"time": 1723, "lat": 10, "lon": 160}

RUNNER_BY_PHASE = {
    5: "run_fase5_cycle_ml.py",
    6: "merge_fase6_shards.py",
    7: "run_fase7_cycle_convlstm.py",
    8: "run_fase8_brazil_convlstm.py",
}
GATE_TABLE_BY_PHASE = {
    5: ("scientific_gate.csv", "gate_pass"),
    6: ("field_gate.csv", "gate_pass"),
    7: ("scientific_gate.csv", "scientific_gate_pass"),
    8: ("confirmatory_gate_by_condition.csv", "gate_pass"),
}
REQUIRED_MODELS = {5: ("rf", "xgb"), 6: ("rf", "xgb")}
REQUIRED_TABLES_BY_PHASE_MODE: Mapping[tuple[int, str], frozenset[str]] = {
    (5, "official"): frozenset(
        {
            "predictor_contract.csv",
            "rolling_origin_targets.csv",
            "fold_metrics.csv",
            "oos_predictions.csv",
            "global_importance.csv",
            "state_importance_oos.csv",
            "augmentation_provenance.csv",
            "fold_contract.csv",
            "independent_support_by_fold.csv",
            "event_dimension_metrics.csv",
            "event_dimension_oos_predictions.csv",
            "event_dimension_importance_oos.csv",
            "event_dimension_augmentation_provenance.csv",
            "scientific_gate.csv",
        }
    ),
    (6, "official"): frozenset(
        {
            "pixel_fold_metrics.csv",
            "pixel_metrics.csv",
            "pixel_oos_predictions.csv",
            "field_gate.csv",
            "native_pixel_inventory.csv",
            "pixel_variable_importance.csv",
            "predictor_contract.csv",
        }
    ),
    (7, "official"): frozenset(
        {
            "fold_metrics.csv",
            "oos_predictions.csv",
            "scalar_variable_importance_oos.csv",
            "spatial_channel_importance_oos.csv",
            "augmentation_provenance.csv",
            "event_probabilistic_metrics.csv",
            "event_probabilistic_metrics_by_event.csv",
            "fold_contract.csv",
            "independent_support_by_fold.csv",
            "paired_f5_comparison.csv",
            "scientific_gate.csv",
        }
    ),
    (8, "official"): frozenset(
        {
            "fold_metrics.csv",
            "pixel_metrics.csv",
            "input_importance_oos.csv",
            "probabilistic_metrics.csv",
            "confirmatory_metrics_by_event.csv",
            "probabilistic_gate_by_condition_fold.csv",
            "confirmatory_gate_by_condition.csv",
            "fold_contract.csv",
            "native_pixel_inventory.csv",
            "predictor_contract.csv",
        }
    ),
}
TABLE_MANIFEST_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "table",
        "path",
        "rows",
        "columns",
        "sha256",
        "schema_sha256",
        "description",
        "units_json",
        "dimensions_json",
        "methods_json",
        "primary_keys",
    }
)
_FILE_HASH_CACHE: dict[Path, tuple[tuple[int, int], str]] = {}
_TREE_HASH_CACHE: dict[Path, tuple[tuple[tuple[str, int, int], ...], str]] = {}


def _status(
    state: str,
    detail: str,
    evidence: Iterable[str] = (),
    **metadata: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "state": state,
        "detail": detail,
        "evidence": list(dict.fromkeys(str(item) for item in evidence if item)),
    }
    result.update(metadata)
    return result


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    except (OSError, UnicodeDecodeError, csv.Error):
        return []


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "sim", "yes", "pass", "passed"}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    path = path.resolve()
    stat = path.stat()
    state = (stat.st_size, stat.st_mtime_ns)
    cached = _FILE_HASH_CACHE.get(path)
    if cached and cached[0] == state:
        return cached[1]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    result = digest.hexdigest()
    _FILE_HASH_CACHE[path] = (state, result)
    return result


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _compact_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tree_sha256(path: Path) -> str:
    path = path.resolve()
    files = []
    for item in sorted(path.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if not item.is_file() or "__pycache__" in item.parts or item.suffix.lower() in {".pyc", ".pyo"}:
            continue
        files.append(item)
    state = tuple(
        (item.relative_to(path).as_posix(), item.stat().st_size, item.stat().st_mtime_ns)
        for item in files
    )
    cached = _TREE_HASH_CACHE.get(path)
    if cached and cached[0] == state:
        return cached[1]
    listing = []
    for item in files:
        listing.append(
            {
                "relative": item.relative_to(path).as_posix(),
                "size": item.stat().st_size,
                "sha256": _sha256_file(item),
            }
        )
    result = _json_hash(listing)
    _TREE_HASH_CACHE[path] = (state, result)
    return result


def _validation_context_record(root: Path, relative: str) -> dict[str, Any]:
    """Recompute one final-validation input using the receipt's stable rules."""

    path = root / relative
    if path.is_dir():
        files = [
            item
            for item in sorted(path.rglob("*"), key=lambda candidate: candidate.as_posix())
            if item.is_file()
            and not any(
                part in _IGNORED_VALIDATION_CONTEXT_PARTS for part in item.parts
            )
            and item.suffix.lower() not in {".pyc", ".pyo"}
            and not item.name.endswith(".tmp")
        ]
        catalogue = [
            {
                "relative": item.relative_to(path).as_posix(),
                "size": item.stat().st_size,
                "sha256": _sha256_file(item),
            }
            for item in files
        ]
        return {
            "path": relative,
            "kind": "directory",
            "tree_sha256": _compact_json_hash(catalogue),
            "file_count": len(files),
            "size_bytes": sum(item.stat().st_size for item in files),
        }
    if path.is_file():
        return {
            "path": relative,
            "kind": "file",
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return {"path": relative, "kind": "missing"}


def _audit_final_validation_context(
    root: Path,
    report: Mapping[str, Any],
) -> list[str]:
    """Reject a once-green receipt after any validated source surface drifts."""

    context = report.get("validation_context")
    if not isinstance(context, Mapping):
        return ["final_validation_context_missing"]
    problems: list[str] = []
    warnings: list[str] = []
    if context.get("schema_version") != "nino26-final-validation-context/v1":
        problems.append("final_validation_context_schema_mismatch")
    paths = context.get("paths")
    if paths != list(FINAL_VALIDATION_CONTEXT_PATHS):
        problems.append("final_validation_context_path_catalog_mismatch")
    rows = context.get("records")
    records = rows if isinstance(rows, list) else []
    if len(records) != len(FINAL_VALIDATION_CONTEXT_PATHS) or not all(
        isinstance(record, Mapping) for record in records
    ):
        problems.append("final_validation_context_records_invalid")
        records = []
    expected_records = [
        _validation_context_record(root, relative)
        for relative in FINAL_VALIDATION_CONTEXT_PATHS
    ]
    if records and records != expected_records:
        problems.append("final_validation_context_current_workspace_mismatch")
    expected_hash = _compact_json_hash(expected_records)
    if str(context.get("inputs_sha256") or "") != expected_hash:
        problems.append("final_validation_context_hash_mismatch")
    if report.get("validation_context_unchanged") is not True:
        problems.append("final_validation_context_changed_during_checks")
    if str(report.get("validation_context_finished_sha256") or "") != expected_hash:
        problems.append("final_validation_finished_context_hash_mismatch")
    return problems


def audit_final_validation_summary(
    root: Path,
    report: Mapping[str, Any] | None,
    report_path: Path,
) -> dict[str, Any]:
    """Validate the persisted receipt, including freshness against this tree."""

    evidence = [_relative(root, report_path)]
    if report is None:
        return _status(
            "missing",
            "Resumo persistido da validação final está ausente.",
            evidence,
            passed=False,
            problems=["final_validation_summary_missing"],
        )
    problems: list[str] = []
    if report.get("schema_version") != "nino26-final-validation-v1":
        problems.append("final_validation_schema_mismatch")
    workspace_value = str(report.get("workspace") or "").strip()
    try:
        workspace = Path(workspace_value).resolve() if workspace_value else None
    except (OSError, ValueError):
        workspace = None
    if workspace is None or workspace != root.resolve():
        problems.append("final_validation_workspace_mismatch")
    rows = report.get("checks")
    checks = rows if isinstance(rows, list) else []
    check_ids = [
        str(check.get("id") or "")
        for check in checks
        if isinstance(check, Mapping)
    ]
    if (
        len(checks) != len(FINAL_VALIDATION_CHECK_IDS)
        or len(check_ids) != len(checks)
        or len(set(check_ids)) != len(check_ids)
        or set(check_ids) != FINAL_VALIDATION_CHECK_IDS
    ):
        problems.append("final_validation_check_catalog_mismatch")
    command_catalog: list[dict[str, Any]] = []
    for index, check in enumerate(checks):
        if not isinstance(check, Mapping):
            problems.append(f"final_validation_check_invalid:{index}")
            continue
        argv = check.get("argv")
        if not isinstance(argv, list) or not argv or not all(
            isinstance(value, str) and value for value in argv
        ):
            problems.append(f"final_validation_argv_invalid:{index}")
            continue
        command_catalog.append({"id": str(check.get("id") or ""), "argv": argv})
        command_hash = hashlib.sha256(
            "\0".join(argv).encode("utf-8", errors="replace")
        ).hexdigest()
        if str(check.get("command_sha256") or "") != command_hash:
            problems.append(f"final_validation_command_hash_mismatch:{check.get('id')}")
        if not (
            check.get("passed") is True
            and _safe_int(check.get("exit_code"), -1) == 0
            and _is_sha256(check.get("stdout_sha256"))
            and _is_sha256(check.get("stderr_sha256"))
        ):
            problems.append(f"final_validation_check_not_passed:{check.get('id')}")
    if str(report.get("check_catalog_sha256") or "") != _compact_json_hash(
        command_catalog
    ):
        problems.append("final_validation_command_catalog_hash_mismatch")
    if not (
        report.get("status") == "passed"
        and report.get("all_passed") is True
        and _safe_int(report.get("check_count"), -1)
        == len(FINAL_VALIDATION_CHECK_IDS)
        and _safe_int(report.get("expected_check_count"), -1)
        == len(FINAL_VALIDATION_CHECK_IDS)
    ):
        problems.append("final_validation_completion_contract_failed")
    started = _parse_time(report.get("started_at_utc"))
    finished = _parse_time(report.get("finished_at_utc"))
    minimum = datetime.min.replace(tzinfo=timezone.utc)
    if started == minimum or finished == minimum or finished < started:
        problems.append("final_validation_timestamps_invalid")
    problems.extend(_audit_final_validation_context(root, report))
    passed = not problems
    return _status(
        "passed" if passed else "failed",
        (
            f"{len(checks)}/{len(FINAL_VALIDATION_CHECK_IDS)} checks vinculados ao código atual."
            if passed
            else f"Recibo final falhou em {len(problems)} verificação(ões)."
        ),
        evidence,
        passed=passed,
        problems=problems,
        report_sha256=_sha256_file(report_path) if report_path.is_file() else "",
        checked_at_utc=report.get("finished_at_utc"),
        validation_context_sha256=(report.get("validation_context") or {}).get(
            "inputs_sha256"
        )
        if isinstance(report.get("validation_context"), Mapping)
        else "",
    )


def _resolve_recorded_path(root: Path, value: Any) -> Path:
    """Resolve um caminho gravado sem escapar do workspace auditado.

    Caminhos relativos pertencem sempre a ``root``; nunca ao diretório de
    trabalho do processo. Caminhos absolutos também precisam apontar para
    dentro do workspace, evitando que ``--root`` misture evidências externas.
    """

    text = str(value or "").strip()
    if not text:
        raise ValueError("empty_recorded_path")
    root_resolved = root.resolve()
    path = Path(text)
    candidate = path if path.is_absolute() else root_resolved / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"path_outside_root:{text}") from exc
    return resolved


def _parse_time(value: Any) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _git_state(root: Path) -> dict[str, Any]:
    def command(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-c", f"safe.directory={root.resolve().as_posix()}", *arguments],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
        return result.stdout.strip() or ""

    porcelain = command("status", "--porcelain")
    return {
        "revision": command("rev-parse", "HEAD") or "unknown",
        "branch": command("branch", "--show-current") or "unknown",
        "dirty": bool(porcelain and porcelain != "unknown"),
        "status_sha256": hashlib.sha256(porcelain.encode("utf-8")).hexdigest(),
        "changed_paths": len(porcelain.splitlines()) if porcelain != "unknown" else None,
    }


def _code_status(root: Path, spec: PhaseSpec) -> dict[str, Any]:
    missing = [path for path in spec.code_paths if not (root / path).is_file()]
    found = [path for path in spec.code_paths if path not in missing]
    if missing:
        return _status(
            "incomplete",
            f"{len(found)}/{len(spec.code_paths)} componentes esperados presentes; faltam {', '.join(missing)}.",
            found,
            required=len(spec.code_paths),
            present=len(found),
        )
    return _status(
        "implemented",
        f"{len(found)}/{len(found)} componentes de implementação presentes.",
        found,
        required=len(found),
        present=len(found),
    )


def _verify_record(root: Path, record: Mapping[str, Any], label: str) -> list[str]:
    problems: list[str] = []
    try:
        path = _resolve_recorded_path(root, record.get("path"))
    except (OSError, ValueError) as exc:
        return [f"{label}:invalid_path:{exc}"]
    if record.get("exists") is False:
        return [f"{label}:recorded_missing_input:{path}"]
    if not path.exists():
        return [f"{label}:missing:{path}"]
    if "is_directory" in record and bool(record.get("is_directory")) != path.is_dir():
        problems.append(f"{label}:path_type_mismatch:{path}")
    expected_file = str(record.get("sha256") or "").strip()
    expected_tree = str(record.get("tree_sha256") or "").strip()
    try:
        if expected_file and (not path.is_file() or _sha256_file(path) != expected_file):
            problems.append(f"{label}:sha256_mismatch:{path}")
        if expected_tree and (not path.is_dir() or _tree_sha256(path) != expected_tree):
            problems.append(f"{label}:tree_sha256_mismatch:{path}")
    except OSError as exc:
        problems.append(f"{label}:unreadable:{path}:{type(exc).__name__}")
    return problems


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _audit_required_file_records(
    root: Path,
    manifest: Mapping[str, Any],
    section: str,
    expected_paths: frozenset[str],
) -> tuple[list[str], set[str]]:
    """Valida catálogo, contenção, tamanho e hash dos arquivos de uma seção."""

    problems: list[str] = []
    raw_records = manifest.get(section)
    if not isinstance(raw_records, list) or not raw_records:
        return [f"missing_required_{section}_records"], set()
    observed: set[str] = set()
    for index, record in enumerate(raw_records):
        if not isinstance(record, Mapping):
            problems.append(f"invalid_{section}_record:{index}")
            continue
        if not _is_sha256(record.get("sha256")):
            problems.append(f"missing_or_invalid_{section}_sha256:{index}")
        try:
            path = _resolve_recorded_path(root, record.get("path"))
        except (OSError, ValueError) as exc:
            problems.append(f"invalid_{section}_path:{index}:{exc}")
            continue
        relative = _relative(root, path)
        if relative in observed:
            problems.append(f"duplicate_{section}_path:{relative}")
        observed.add(relative)
        problems.extend(_verify_record(root, record, f"phase2_{section}"))
        if path.is_file():
            try:
                recorded_size = int(record.get("size_bytes"))
            except (TypeError, ValueError):
                recorded_size = -1
            if recorded_size <= 0 or recorded_size != path.stat().st_size:
                problems.append(f"{section}_size_mismatch:{relative}")
    missing = sorted(expected_paths - observed)
    unexpected = sorted(observed - expected_paths)
    if missing:
        problems.append(f"missing_required_{section}:{','.join(missing)}")
    if unexpected:
        problems.append(f"unexpected_{section}:{','.join(unexpected)}")
    return problems, observed


def _csv_fieldnames(path: Path) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return tuple(csv.DictReader(stream).fieldnames or ())
    except (OSError, UnicodeDecodeError, csv.Error):
        return ()


def _require_record_content_hash(
    root: Path,
    record: Mapping[str, Any],
    label: str,
) -> list[str]:
    """Make ArtifactRun provenance fail closed instead of trusting optional hashes."""

    problems: list[str] = []
    if record.get("exists") is not True:
        problems.append(f"{label}:exists_not_true")
    try:
        path = _resolve_recorded_path(root, record.get("path"))
    except (OSError, ValueError) as exc:
        return [*problems, f"{label}:invalid_path:{exc}"]
    if bool(record.get("is_directory")):
        if not _is_sha256(record.get("tree_sha256")):
            problems.append(f"{label}:missing_or_invalid_tree_sha256:{_relative(root, path)}")
    elif not _is_sha256(record.get("sha256")):
        problems.append(f"{label}:missing_or_invalid_sha256:{_relative(root, path)}")
    problems.extend(_verify_record(root, record, label))
    return problems


def _audit_predictor_contract(directory: Path) -> list[str]:
    path = directory / "tables" / "predictor_contract.csv"
    if not path.is_file():
        return []
    rows = _read_csv(path)
    variables = [str(row.get("variable") or "").strip() for row in rows]
    problems: list[str] = []
    if tuple(variables) != PHASE2_PHYSICAL_COLUMNS:
        problems.append(
            "predictor_contract_not_exact_ordered_31:"
            f"observed={len(variables)}:unique={len(set(variables))}"
        )
    return problems


def _audit_gate_table_contract(
    directory: Path,
    manifest: Mapping[str, Any],
    phase: int,
) -> list[str]:
    spec = GATE_TABLE_BY_PHASE.get(phase)
    if spec is None:
        return []
    table_name, boolean_column = spec
    path = directory / "tables" / table_name
    if not path.is_file():
        return []
    rows = _read_csv(path)
    problems: list[str] = []
    fields = set(_csv_fieldnames(path))
    if boolean_column not in fields:
        problems.append(f"gate_boolean_column_missing:{table_name}:{boolean_column}")
        return problems
    if not rows:
        problems.append(f"empty_gate_table:{table_name}")
        return problems
    if phase == 5:
        required = {
            "component",
            "metric",
            "value",
            "threshold_rule",
            "n_oos_units",
            "oos_unit",
            "gate_pass",
            "min_train_el_nino_events",
            "min_train_la_nina_events",
            "min_train_active_events_per_type_required",
            "independent_support_gate_pass",
            "support_required_for_gate",
        }
        missing = sorted(required - fields)
        if missing:
            problems.append(f"f5_gate_schema_missing:{','.join(missing)}")
        components = [str(row.get("component") or "").strip() for row in rows]
        if len(components) != len(set(components)) or any(not value for value in components):
            problems.append("f5_gate_component_key_not_unique")
        horizons = (manifest.get("parameters") or {}).get("horizons")
        if isinstance(horizons, list):
            expected_classification = {f"classification_h{int(value):02d}" for value in horizons}
            observed_classification = {
                value for value in components if value.startswith("classification_")
            }
            if observed_classification != expected_classification:
                problems.append("f5_gate_horizon_catalog_mismatch")
        expected_dimensions = {
            "event_dimension_Y_pico",
            "event_dimension_Y_tempo_para_pico_sem",
            "event_dimension_Y_duracao_sem",
        }
        observed_dimensions = {
            value for value in components if value.startswith("event_dimension_")
        }
        if observed_dimensions != expected_dimensions:
            problems.append("f5_gate_event_dimension_catalog_mismatch")
    elif phase == 6:
        required = {
            "target_variable",
            "target_transform",
            "model",
            "target_units",
            "condition",
            "lag_weeks",
            "pixel_coverage_fraction",
            "pixel_set_exact",
            "gate_eligible",
            "gate_pass",
        }
        missing = sorted(required - fields)
        if missing:
            problems.append(f"f6_gate_schema_missing:{','.join(missing)}")
        keys = [
            tuple(str(row.get(column) or "") for column in (
                "target_variable",
                "target_transform",
                "model",
                "target_units",
                "condition",
                "lag_weeks",
            ))
            for row in rows
        ]
        if len(keys) != len(set(keys)):
            problems.append("f6_gate_primary_key_not_unique")
        conditions = {str(row.get("condition") or "").strip() for row in rows}
        conditions.discard("")
        if conditions != ENSO_PHASE_CONDITIONS:
            problems.append("f6_gate_condition_catalog_mismatch")
    elif phase == 7:
        required = {
            "horizon_weeks",
            "mean_skill_f1_vs_best_persistence_or_seasonal",
            "baseline_gate_pass",
            "best_f5_reference_run_id",
            "paired_coverage_best_f5",
            "skill_f1_f7_minus_best_f5_paired",
            "paired_f5_gate_pass",
            "f5_comparison_required",
            "min_train_active_events_per_type_required",
            "min_train_el_nino_events",
            "min_train_la_nina_events",
            "independent_support_gate_pass",
            "support_required_for_gate",
            "minimum_event_dimension_skill_required",
            "max_interval90_absolute_calibration_error",
            "event_dimension_gate_pass",
            "interval_calibration_gate_pass",
            "skill_mae_event_equal_peak_magnitude_c",
            "skill_mae_event_equal_event_time_to_peak_weeks",
            "skill_mae_event_equal_event_duration_weeks",
            "interval90_absolute_calibration_error_peak_magnitude_c",
            "interval90_absolute_calibration_error_event_time_to_peak_weeks",
            "interval90_absolute_calibration_error_event_duration_weeks",
            "scientific_gate_pass",
        }
        missing = sorted(required - fields)
        if missing:
            problems.append(f"f7_gate_schema_missing:{','.join(missing)}")
        horizons = [str(row.get("horizon_weeks") or "").strip() for row in rows]
        if len(horizons) != len(set(horizons)) or any(not value for value in horizons):
            problems.append("f7_gate_horizon_key_not_unique")
    elif phase == 8:
        required = {
            "condition",
            "n_independent_events",
            "minimum_independent_events_required",
            "minimum_event_support_gate_pass",
            "persistence_gate_pass",
            "f6_comparison_gate_pass",
            "n_events_with_interval_coverage",
            "n_calibration_folds",
            "interval90_coverage_event_equal",
            "interval90_nominal_coverage",
            "interval90_absolute_calibration_error",
            "maximum_fold_interval90_absolute_calibration_error",
            "max_interval90_absolute_calibration_error",
            "complete_native_interval_values",
            "all_fold_interval_calibration_gate_pass",
            "interval_calibration_gate_pass",
            "interval_aggregation_unit",
            "gate_pass",
            "overall_eight_condition_gate_pass",
        }
        missing = sorted(required - fields)
        if missing:
            problems.append(f"f8_gate_schema_missing:{','.join(missing)}")
        conditions = [str(row.get("condition") or "").strip() for row in rows]
        if len(conditions) != len(set(conditions)):
            problems.append("f8_gate_condition_key_not_unique")
        if set(conditions) != ENSO_PHASE_CONDITIONS:
            problems.append("f8_gate_condition_catalog_mismatch")
        for row in rows:
            condition = str(row.get("condition") or "").strip()
            try:
                nominal = float(row.get("interval90_nominal_coverage"))
                error = float(row.get("interval90_absolute_calibration_error"))
                maximum_fold_error = float(
                    row.get("maximum_fold_interval90_absolute_calibration_error")
                )
                tolerance = float(
                    row.get("max_interval90_absolute_calibration_error")
                )
            except (TypeError, ValueError):
                problems.append(f"f8_gate_invalid_calibration_values:{condition}")
                continue
            if abs(nominal - 0.90) > 1e-12:
                problems.append(f"f8_gate_nominal_coverage_not_090:{condition}")
            if not 0.0 <= tolerance <= 0.15 + 1e-12:
                problems.append(f"f8_gate_tolerance_above_predeclared_max:{condition}")
            calibration_flags = (
                _truthy(row.get("complete_native_interval_values"))
                and _truthy(row.get("all_fold_interval_calibration_gate_pass"))
                and _truthy(row.get("interval_calibration_gate_pass"))
            )
            if _truthy(row.get("gate_pass")) and (
                not calibration_flags
                or error > tolerance + 1e-12
                or maximum_fold_error > tolerance + 1e-12
            ):
                problems.append(f"f8_gate_pass_without_valid_ic90:{condition}")
    return problems


def audit_artifact_run(
    root: Path,
    directory: Path,
    *,
    expected_phase: int | None = None,
    expected_mode: str | None = None,
) -> dict[str, Any]:
    """Audita um ArtifactRun sem confiar em timestamp ou nome de pasta."""

    manifest_path = directory / "run_manifest.json"
    table_manifest_path = directory / "tables_manifest.csv"
    run_manifest_sha256 = (
        _sha256_file(manifest_path) if manifest_path.is_file() else ""
    )
    current_tables_manifest_sha256 = (
        _sha256_file(table_manifest_path) if table_manifest_path.is_file() else ""
    )
    manifest = _read_json(manifest_path)
    problems: list[str] = []
    warnings: list[str] = []
    if manifest is None:
        problems.append("missing_or_invalid_run_manifest")
        manifest = {}
    if manifest.get("schema_version") != "nino26-run-v1":
        problems.append("invalid_or_missing_run_schema_version")
    if not table_manifest_path.is_file():
        problems.append("missing_tables_manifest")
        table_rows: list[dict[str, str]] = []
    else:
        table_rows = _read_csv(table_manifest_path)
        expected = str(manifest.get("tables_manifest_sha256") or "").strip()
        if not _is_sha256(expected):
            problems.append("missing_or_invalid_tables_manifest_sha256")
        elif _sha256_file(table_manifest_path) != expected:
            problems.append("tables_manifest_hash_mismatch")
        fields = set(_csv_fieldnames(table_manifest_path))
        missing_fields = sorted(TABLE_MANIFEST_REQUIRED_COLUMNS - fields)
        if missing_fields:
            problems.append(f"tables_manifest_schema_missing:{','.join(missing_fields)}")

    table_names: set[str] = set()
    for index, row in enumerate(table_rows):
        name = str(row.get("table") or "").strip()
        if not name or name != Path(name).name or not name.endswith(".csv"):
            problems.append(f"invalid_table_name:{index}:{name or 'missing'}")
            continue
        if name in table_names:
            problems.append(f"duplicate_table_manifest_entry:{name}")
        table_names.add(name)
        recorded_path = str(row.get("path") or "").replace("\\", "/").strip()
        expected_path = f"tables/{name}"
        if recorded_path != expected_path:
            problems.append(
                f"noncanonical_table_path:{name}:expected={expected_path}:actual={recorded_path or 'missing'}"
            )
        path = (directory / recorded_path).resolve()
        try:
            path.relative_to(directory.resolve())
        except ValueError:
            problems.append(f"table_path_outside_run:{name}")
            continue
        if not _is_sha256(row.get("sha256")):
            problems.append(f"missing_or_invalid_table_sha256:{name}")
        if not _is_sha256(row.get("schema_sha256")):
            problems.append(f"missing_or_invalid_table_schema_sha256:{name}")
        try:
            rows_count = int(row.get("rows") or -1)
            columns_count = int(row.get("columns") or -1)
        except (TypeError, ValueError):
            rows_count = columns_count = -1
        if rows_count < 0 or columns_count <= 0:
            problems.append(f"invalid_table_shape_metadata:{name}")
        if not str(row.get("description") or "").strip():
            problems.append(f"missing_table_description:{name}")
        for json_column in ("units_json", "dimensions_json", "methods_json"):
            try:
                decoded = json.loads(str(row.get(json_column) or ""))
            except json.JSONDecodeError:
                decoded = None
            if not isinstance(decoded, dict):
                problems.append(f"invalid_{json_column}:{name}")
        if not path.is_file():
            problems.append(f"missing_table:{name}")
        elif _is_sha256(row.get("sha256")) and _sha256_file(path) != row["sha256"]:
            problems.append(f"table_hash_mismatch:{name}")

    if table_rows and _safe_int(manifest.get("n_tables"), -1) != len(table_rows):
        problems.append("manifest_n_tables_mismatch")

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        problems.append("missing_or_invalid_registered_files_catalog")
        raw_files = []
    for index, record in enumerate(raw_files):
        if not isinstance(record, Mapping):
            problems.append(f"invalid_registered_file_record:{index}")
            continue
        path = (directory / str(record.get("path") or "")).resolve()
        try:
            path.relative_to(directory.resolve())
        except ValueError:
            problems.append(f"registered_path_outside_run:{record.get('path')}")
            continue
        if path.is_file() and not _is_sha256(record.get("sha256")):
            problems.append(f"registered_file_missing_sha256:{record.get('path')}")
        if path.is_dir() and not _is_sha256(record.get("tree_sha256")):
            problems.append(f"registered_directory_missing_tree_sha256:{record.get('path')}")
        if not path.exists():
            problems.append(f"missing_registered_file:{record.get('path')}")
        elif path.is_file() and record.get("sha256") and _sha256_file(path) != record["sha256"]:
            problems.append(f"registered_file_hash_mismatch:{record.get('path')}")
        elif path.is_dir() and record.get("tree_sha256") and _tree_sha256(path) != record["tree_sha256"]:
            problems.append(f"registered_tree_hash_mismatch:{record.get('path')}")

    raw_inputs = manifest.get("inputs", []) if isinstance(manifest.get("inputs"), list) else []
    raw_configs = manifest.get("configs", []) if isinstance(manifest.get("configs"), list) else []
    inputs = [record for record in raw_inputs if isinstance(record, Mapping)]
    configs = [record for record in raw_configs if isinstance(record, Mapping)]
    if not raw_inputs or len(inputs) != len(raw_inputs):
        problems.append("missing_or_invalid_input_records")
    if not raw_configs or len(configs) != len(raw_configs):
        problems.append("missing_or_invalid_config_records")
    for label, records in (("input", inputs), ("config", configs)):
        for index, record in enumerate(records):
            issues = _require_record_content_hash(root, record, f"{label}:{index}")
            recorded = str(record.get("path") or "").replace("\\", "/").rstrip("/")
            if recorded == "src/nino_brasil" or recorded.endswith("/src/nino_brasil"):
                if issues:
                    warnings.append("Warning: Fingerprint modificado")
            else:
                problems.extend(issues)

    input_fingerprints: list[dict[str, Any]] = []
    for section, records in (("inputs", inputs), ("configs", configs)):
        for record in records:
            try:
                fingerprint_path = _resolve_recorded_path(root, record.get("path"))
            except (OSError, ValueError):
                continue
            input_fingerprints.append(
                {
                    "section": section,
                    "path": _relative(root, fingerprint_path),
                    "sha256": str(record.get("sha256") or ""),
                    "tree_sha256": str(record.get("tree_sha256") or ""),
                }
            )
    input_fingerprints.sort(key=lambda item: (item["section"], item["path"]))

    try:
        phase = int(manifest.get("phase"))
    except (TypeError, ValueError):
        phase = 0
        problems.append("invalid_manifest_phase")
    mode = str(manifest.get("mode") or "").strip()
    run_id = str(manifest.get("run_id") or "").strip()
    if not run_id:
        problems.append("missing_manifest_run_id")
    elif run_id != directory.name:
        problems.append(f"run_id_directory_mismatch:manifest={run_id}:directory={directory.name}")
    if mode not in {"smoke", "official"}:
        problems.append(f"invalid_manifest_mode:{mode or 'missing'}")
    if str(manifest.get("status") or "") not in {"complete", "failed"}:
        problems.append("invalid_manifest_status")
    parameters = manifest.get("parameters")
    if not isinstance(parameters, Mapping):
        problems.append("missing_or_invalid_parameters")
        parameters = {}
    parameters_sha256 = str(manifest.get("parameters_sha256") or "").strip()
    if not _is_sha256(parameters_sha256):
        problems.append("missing_or_invalid_parameters_sha256")
    elif parameters_sha256 != _json_hash(dict(parameters)):
        problems.append("parameters_sha256_mismatch")
    if not isinstance(manifest.get("environment"), Mapping):
        problems.append("missing_environment_contract")
    if not str(manifest.get("command") or "").strip():
        problems.append("missing_run_command")
    try:
        int(manifest.get("seed"))
    except (TypeError, ValueError):
        problems.append("missing_or_invalid_seed")
    if expected_phase is not None and phase != expected_phase:
        problems.append(f"manifest_phase_mismatch:expected={expected_phase}:actual={phase}")
    if expected_mode is not None and mode != expected_mode:
        problems.append(f"manifest_mode_mismatch:expected={expected_mode}:actual={mode or 'missing'}")
    started_at = _parse_time(manifest.get("started_at"))
    finished_at = _parse_time(manifest.get("finished_at"))
    minimum_time = datetime.min.replace(tzinfo=timezone.utc)
    if str(manifest.get("status") or "") == "complete" and (
        started_at == minimum_time or finished_at == minimum_time or finished_at < started_at
    ):
        problems.append("invalid_run_timestamps")
    normalized_inputs = [str(record.get("path") or "").replace("\\", "/") for record in inputs]
    source_fingerprint = any(
        (
            path.rstrip("/") == "src/nino_brasil"
            or path.rstrip("/").endswith("/src/nino_brasil")
        )
        and record.get("tree_sha256")
        for path, record in zip(normalized_inputs, inputs)
    )
    expected_runner = RUNNER_BY_PHASE.get(phase)
    runner_fingerprint = bool(expected_runner) and any(
        Path(path).name == expected_runner and record.get("sha256")
        for path, record in zip(normalized_inputs, inputs)
    )
    git_commit = str((manifest.get("git") or {}).get("commit") or "").strip()
    provenance_complete = bool(
        # Política conservadora intencional: enquanto não houver um grafo de
        # dependências congelado por fase, qualquer deriva em src/nino_brasil
        # invalida a reprodução. Isso pode gerar falso negativo entre fases,
        # mas nunca promove um run executado com código divergente.
        source_fingerprint
        and runner_fingerprint
        and git_commit
        and git_commit.lower() not in {"unknown", "none", "null"}
    )
    source_fingerprints = [
        record
        for record in input_fingerprints
        if (
            record["path"].rstrip("/") == "src/nino_brasil"
            or record["path"].rstrip("/").endswith("/src/nino_brasil")
        )
        or (expected_runner and Path(record["path"]).name == expected_runner)
    ]

    if phase == 6 and mode == "official":
        if parameters.get("role") != "merge_pixel_shards_and_field_gate":
            problems.append("official_f6_is_shard_not_merge")
    required_gate = GATE_TABLE_BY_PHASE.get(phase)
    if mode == "official" and required_gate and required_gate[0] not in table_names:
        problems.append(f"missing_required_gate_table:{required_gate[0]}")
    required_tables = REQUIRED_TABLES_BY_PHASE_MODE.get((phase, mode), frozenset())
    missing_tables = sorted(required_tables - table_names)
    if missing_tables:
        problems.append(f"missing_required_tables:{','.join(missing_tables)}")
    if mode == "official" and phase in {5, 6, 8}:
        problems.extend(_audit_predictor_contract(directory))
    if mode == "official":
        problems.extend(_audit_gate_table_contract(directory, manifest, phase))

    return {
        "path": _relative(root, directory),
        "run_id": run_id or directory.name,
        "phase": phase,
        "mode": mode,
        "status": str(manifest.get("status") or "missing"),
        "started_at": str(manifest.get("started_at") or ""),
        "finished_at": str(manifest.get("finished_at") or ""),
        "model": str((manifest.get("parameters") or {}).get("model") or "default"),
        "run_manifest_sha256": run_manifest_sha256,
        "tables_manifest_sha256": current_tables_manifest_sha256,
        "parameters_sha256": parameters_sha256,
        "input_catalog_sha256": _compact_json_hash(input_fingerprints),
        "source_fingerprints": source_fingerprints,
        "git_commit": git_commit,
        "artifact_valid": not problems,
        "provenance_complete": provenance_complete,
        "eligible": bool(
            str(manifest.get("status")) == "complete"
            and not problems
            and provenance_complete
        ),
        "problems": problems,
        "warnings": sorted(set(warnings)),
        "table_names": sorted(table_names),
        "table_rows": table_rows,
        "manifest": manifest,
    }


def _gate_from_run(root: Path, audit: Mapping[str, Any]) -> dict[str, Any]:
    phase = int(audit.get("phase") or 0)
    gate_spec = GATE_TABLE_BY_PHASE.get(phase)
    if gate_spec is None:
        return _status("not_applicable", "Esta fase não usa gate de ArtifactRun.")
    table_name, boolean_column = gate_spec
    directory = root / str(audit["path"])
    path = directory / "tables" / table_name
    if table_name not in set(audit.get("table_names") or ()):
        return _status(
            "missing",
            f"Gate não pertence ao catálogo de tabelas auditado: {table_name}.",
            [_relative(root, path)],
            passed=False,
        )
    if not audit.get("artifact_valid"):
        return _status(
            "invalidated",
            "ArtifactRun inválido; o gate não pode ser lido como evidência científica.",
            [_relative(root, path)],
            passed=False,
        )
    rows = _read_csv(path)
    if not rows or boolean_column not in rows[0]:
        return _status("missing", f"Tabela/coluna de gate ausente: {table_name}:{boolean_column}.", [_relative(root, path)])
    values = [_truthy(row.get(boolean_column)) for row in rows]
    passed = all(values)
    component_detail = ""
    if phase == 5:
        classification = [
            _truthy(row.get(boolean_column))
            for row in rows
            if str(row.get("component") or "").startswith("classification_")
        ]
        dimensions = [
            _truthy(row.get(boolean_column))
            for row in rows
            if str(row.get("component") or "").startswith("event_dimension_")
        ]
        # Um horizonte positivo isolado é um resultado parcial, não um gate
        # global. A promoção exige todos os horizontes classificatórios e as
        # três dimensões independentes de evento acima dos seus baselines.
        passed = (
            bool(classification)
            and all(classification)
            and bool(dimensions)
            and all(dimensions)
        )
        component_detail = (
            f" classificação={sum(classification)}/{len(classification)};"
            f" dimensões={sum(dimensions)}/{len(dimensions)};"
        )
    if phase == 6:
        eligible_values = [_truthy(row.get("gate_eligible")) for row in rows]
        exact_values = []
        for row in rows:
            try:
                coverage = float(row.get("pixel_coverage_fraction") or 0.0)
            except (TypeError, ValueError):
                coverage = 0.0
            exact_values.append(_truthy(row.get("pixel_set_exact")) and math.isclose(coverage, 1.0))
        conditions = {str(row.get("condition") or "").strip() for row in rows}
        conditions.discard("")
        passed = (
            passed
            and all(eligible_values)
            and all(exact_values)
            and conditions == ENSO_PHASE_CONDITIONS
        )
        component_detail = f" condições={len(conditions)}/8;"
    if phase == 8:
        overall = [_truthy(row.get("overall_eight_condition_gate_pass")) for row in rows]
        conditions = {str(row.get("condition") or row.get("condicao") or "").strip() for row in rows}
        conditions.discard("")
        passed = passed and all(overall) and conditions == ENSO_PHASE_CONDITIONS
    return _status(
        "passed" if passed else "failed",
        f"{sum(values)}/{len(values)} linhas satisfazem {boolean_column};{component_detail} regra agregada={'passou' if passed else 'falhou'}.",
        [_relative(root, path)],
        passed=passed,
        passed_rows=sum(values),
        total_rows=len(values),
    )


def _run_sort_key(audit: Mapping[str, Any]) -> tuple[datetime, str]:
    return (_parse_time(audit.get("finished_at") or audit.get("started_at")), str(audit.get("run_id")))


def _discover_run_audits(root: Path, phase: int, mode: str) -> list[dict[str, Any]]:
    base = root / f"data/processed/runs/{mode}/fase{phase}"
    if not base.is_dir():
        return []
    audits: list[dict[str, Any]] = []
    for path in base.iterdir():
        if not path.is_dir():
            continue
        manifest = _read_json(path / "run_manifest.json") or {}
        if phase == 6 and mode == "official" and (manifest.get("parameters") or {}).get("role") != "merge_pixel_shards_and_field_gate":
            try:
                actual_phase = int(manifest.get("phase"))
            except (TypeError, ValueError):
                actual_phase = 0
            actual_mode = str(manifest.get("mode") or "").strip()
            actual_run_id = str(manifest.get("run_id") or "").strip()
            problems = ["official_f6_is_shard_not_merge"]
            if actual_phase != phase:
                problems.append(f"manifest_phase_mismatch:expected={phase}:actual={actual_phase}")
            if actual_mode != mode:
                problems.append(f"manifest_mode_mismatch:expected={mode}:actual={actual_mode or 'missing'}")
            if not actual_run_id or actual_run_id != path.name:
                problems.append(
                    f"run_id_directory_mismatch:manifest={actual_run_id or 'missing'}:directory={path.name}"
                )
            started_at = _parse_time(manifest.get("started_at"))
            finished_at = _parse_time(manifest.get("finished_at"))
            minimum_time = datetime.min.replace(tzinfo=timezone.utc)
            if str(manifest.get("status") or "") == "complete" and (
                started_at == minimum_time or finished_at == minimum_time or finished_at < started_at
            ):
                problems.append("invalid_run_timestamps")
            audits.append(
                {
                    "path": _relative(root, path),
                    "run_id": actual_run_id or path.name,
                    "phase": actual_phase,
                    "mode": actual_mode,
                    "status": str(manifest.get("status") or "missing"),
                    "started_at": str(manifest.get("started_at") or ""),
                    "finished_at": str(manifest.get("finished_at") or ""),
                    "model": str((manifest.get("parameters") or {}).get("model") or "default"),
                    "artifact_valid": False,
                    "provenance_complete": False,
                    "eligible": False,
                    "problems": problems,
                    "table_names": [],
                    "manifest": manifest,
                }
            )
            continue
        audits.append(
            audit_artifact_run(
                root,
                path,
                expected_phase=phase,
                expected_mode=mode,
            )
        )
    return sorted(audits, key=_run_sort_key, reverse=True)


def _comparison_signature(root: Path, audit: Mapping[str, Any]) -> str:
    """Hash the protocol shared by model variants, excluding the model itself."""

    phase = int(audit.get("phase") or 0)
    manifest = audit.get("manifest") if isinstance(audit.get("manifest"), Mapping) else {}
    parameters = dict(manifest.get("parameters") or {})
    parameters.pop("model", None)
    if phase == 6:
        # Shard IDs and their output hashes are model-specific. The scientific
        # pairing is defined by target/protocol/catalogs, not by those IDs.
        parameters.pop("source_runs", None)
        parameters.pop("source_fingerprint_sha256", None)

    record_catalog: list[dict[str, Any]] = []
    for section in ("inputs", "configs"):
        records = manifest.get(section) if isinstance(manifest.get(section), list) else []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            try:
                path = _resolve_recorded_path(root, record.get("path"))
            except (OSError, ValueError):
                continue
            relative = _relative(root, path)
            if phase == 6 and relative.startswith("data/processed/runs/"):
                continue
            record_catalog.append(
                {
                    "section": section,
                    "path": relative,
                    "sha256": record.get("sha256"),
                    "tree_sha256": record.get("tree_sha256"),
                }
            )

    directory = root / str(audit.get("path") or "")
    table_hashes = {
        str(row.get("table")): str(row.get("sha256"))
        for row in audit.get("table_rows") or []
        if isinstance(row, Mapping)
    }
    shared_tables: dict[str, Any] = {}
    for name in (
        "predictor_contract.csv",
        "rolling_origin_targets.csv",
        "native_pixel_inventory.csv",
    ):
        if name in table_hashes:
            shared_tables[name] = table_hashes[name]

    structural_catalog: dict[str, Any] = {}
    if phase == 5:
        fold_rows = _read_csv(directory / "tables" / "fold_contract.csv")
        structural_catalog["fold_contract"] = sorted(
            {
                (
                    str(row.get("experiment_horizon_weeks") or ""),
                    str(row.get("fold") or ""),
                    str(row.get("train_end") or ""),
                    str(row.get("test_start") or ""),
                    str(row.get("test_end") or ""),
                    str(row.get("purge_weeks") or ""),
                    str(row.get("test_groups") or ""),
                    str(row.get("preprocessing_fit_end") or ""),
                )
                for row in fold_rows
            }
        )
    elif phase == 6:
        gate_rows = _read_csv(directory / "tables" / "field_gate.csv")
        structural_catalog["field_gate_groups"] = sorted(
            {
                (
                    str(row.get("target_variable") or ""),
                    str(row.get("target_transform") or ""),
                    str(row.get("target_units") or ""),
                    str(row.get("condition") or ""),
                    str(row.get("lag_weeks") or ""),
                )
                for row in gate_rows
            }
        )

    payload = {
        "phase": phase,
        "mode": audit.get("mode"),
        "seed": manifest.get("seed"),
        "parameters_without_model": parameters,
        "provenance": sorted(record_catalog, key=lambda item: (item["section"], item["path"])),
        "shared_table_hashes": shared_tables,
        "structural_catalog": structural_catalog,
    }
    return _json_hash(payload)


def _public_run(
    audit: Mapping[str, Any],
    gate: Mapping[str, Any] | None = None,
    comparison_signature: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": audit.get("run_id"),
        "path": audit.get("path"),
        "model": audit.get("model"),
        "status": audit.get("status"),
        "eligible": audit.get("eligible"),
        "finished_at": audit.get("finished_at"),
        "artifact_valid": audit.get("artifact_valid"),
        "provenance_complete": audit.get("provenance_complete"),
        "run_manifest_sha256": audit.get("run_manifest_sha256"),
        "tables_manifest_sha256": audit.get("tables_manifest_sha256"),
        "parameters_sha256": audit.get("parameters_sha256"),
        "input_catalog_sha256": audit.get("input_catalog_sha256"),
        "source_fingerprints": list(audit.get("source_fingerprints") or []),
        "git_commit": audit.get("git_commit"),
        "table_count": len(audit.get("table_names") or []),
        "table_catalog": list(audit.get("table_names") or []),
        "problems": list(audit.get("problems") or []),
        "gate": gate,
        "comparison_signature_sha256": comparison_signature,
    }


def _select_artifact_runs(root: Path, phase: int, mode: str) -> dict[str, Any]:
    audits = _discover_run_audits(root, phase, mode)
    eligible = [audit for audit in audits if audit["eligible"]]
    selected: list[dict[str, Any]] = []
    missing_variants: list[str] = []
    required_models = REQUIRED_MODELS.get(phase, ()) if mode == "official" else ()
    if required_models:
        for model in required_models:
            candidates = [audit for audit in eligible if audit["model"] == model]
            if candidates:
                selected.append(candidates[0])
            else:
                missing_variants.append(model)
    elif eligible:
        selected.append(eligible[0])

    comparison_signatures: dict[str, str] = {}
    comparison_compatible: bool | None = None
    if required_models and len(selected) == len(required_models) and not missing_variants:
        comparison_signatures = {
            str(audit["run_id"]): _comparison_signature(root, audit) for audit in selected
        }
        comparison_compatible = len(set(comparison_signatures.values())) == 1

    rejected = [audit for audit in audits if audit not in selected]
    if selected and not missing_variants and comparison_compatible is not False:
        state = "complete"
    elif selected and comparison_compatible is False:
        state = "invalidated"
    elif selected:
        state = "partial"
    elif audits:
        state = "invalidated" if any(audit["status"] == "complete" for audit in audits) else "failed"
    else:
        state = "not_run"

    gate_by_run = {audit["run_id"]: _gate_from_run(root, audit) for audit in selected}
    selected_public = [
        _public_run(
            audit,
            gate_by_run[audit["run_id"]],
            comparison_signatures.get(str(audit["run_id"])),
        )
        for audit in selected
    ]
    detail = {
        "complete": f"{len(selected)} run(s) elegível(is), contrato oficial completo.",
        "partial": f"Runs elegíveis presentes; faltam variantes oficiais: {', '.join(missing_variants)}.",
        "invalidated": (
            "As variantes existem, mas não formam um par científico comparável."
            if comparison_compatible is False
            else "Há runs complete, mas nenhum conserva integridade, proveniência e contrato atuais."
        ),
        "failed": "Há tentativas registradas, mas nenhuma terminou com status complete elegível.",
        "not_run": "Nenhum run registrado nesta modalidade.",
    }[state]
    status = _status(
        state,
        detail,
        [run["path"] for run in selected_public],
        selected=selected_public,
        required_models=list(required_models),
        missing_models=missing_variants,
        candidate_count=len(audits),
        rejected=[_public_run(audit) for audit in rejected[:10]],
        comparison_compatible=comparison_compatible,
        comparison_signatures=comparison_signatures,
        comparison_problem=(
            "comparison_signature_mismatch" if comparison_compatible is False else None
        ),
    )
    status["_selected_audits"] = selected
    return status


def _strip_internal(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_internal(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, list):
        return [_strip_internal(item) for item in value]
    return value


def collect_notebook_contract(root: Path, phase: int | None = None) -> dict[str, Any]:
    notebooks_root = root / "notebooks"

    def notebook_phase(path: Path) -> int | None:
        folder = path.parent.name
        if not folder.startswith("fase"):
            return None
        digits = "".join(character for character in folder[4:] if character.isdigit())
        return int(digits) if digits else None

    paths = (
        sorted(
            path
            for path in notebooks_root.rglob("*.ipynb")
            if "executed" not in path.parts
            and notebook_phase(path) in range(2, 9)
        )
        if notebooks_root.exists()
        else []
    )
    catalogue_problems: list[str] = []
    excluded: list[Path] = []
    if phase is not None:
        paths = [path for path in paths if notebook_phase(path) == phase]

    canonical_source_paths: list[Path] = []
    for path in paths:
        notebook = _read_json(path) or {}
        metadata = (notebook.get("metadata") or {}).get("nino26") or {}
        if (
            metadata.get("canonical") is True
            and metadata.get("execution_policy") == "numeric-core-first-viewer-publisher"
        ):
            canonical_source_paths.append(path)
    uses_source_contract = bool(canonical_source_paths)
    if uses_source_contract:
        paths = canonical_source_paths

    if phase == 4 and not uses_source_contract:
        canonical_names = {
            "4C_sinal_pixel_lags.ipynb",
            "4D_clusters_alvo.ipynb",
        }
        discovered = {path.name: path for path in paths}
        for name in sorted(canonical_names - set(discovered)):
            catalogue_problems.append(f"missing_canonical_notebook:notebooks/fase4/{name}")
        for name, path in discovered.items():
            if name in canonical_names:
                continue
            notebook = _read_json(path) or {}
            project_metadata = (notebook.get("metadata") or {}).get("nino26") or {}
            if (
                project_metadata.get("canonical") is False
                and project_metadata.get("promotion_status") == "excluded"
                and project_metadata.get("execution_policy") == "read-only-viewer"
            ):
                excluded.append(path)
            else:
                catalogue_problems.append(f"unexpected_promotable_notebook:{_relative(root, path)}")
        paths = [discovered[name] for name in sorted(canonical_names & set(discovered))]
    if not paths:
        return _status(
            "failed" if catalogue_problems else "not_run",
            "Nenhum notebook canônico aplicável encontrado.",
            passed=False,
            excluded_notebook_count=len(excluded),
            problems=catalogue_problems,
        )
    problems: list[str] = list(catalogue_problems)
    for path in paths:
        relative = _relative(root, path)
        notebook = _read_json(path)
        if notebook is None:
            problems.append(f"invalid_json:{relative}")
            continue
        cells = notebook.get("cells", []) if isinstance(notebook.get("cells"), list) else []
        project_metadata = (notebook.get("metadata") or {}).get("nino26") or {}
        source_contract = (
            project_metadata.get("canonical") is True
            and project_metadata.get("execution_policy")
            == "numeric-core-first-viewer-publisher"
        )
        persist_inline_outputs = bool(project_metadata.get("persist_inline_outputs"))
        if int(notebook.get("nbformat") or 0) < 4:
            problems.append(f"nbformat_contract:{relative}")
        sources = []
        ids = []
        executable_cells = 0
        unexecuted_cells = 0
        for cell in cells:
            source = cell.get("source", "")
            sources.append(source if isinstance(source, str) else "".join(source))
            ids.append(str(cell.get("id") or "").strip())
            if cell.get("cell_type") == "code" and sources[-1].strip():
                executable_cells += 1
                if cell.get("execution_count") is None:
                    unexecuted_cells += 1
                if source_contract and not persist_inline_outputs and cell.get("execution_count") is not None:
                    problems.append(f"source_execution_count:{relative}")
                if source_contract and not persist_inline_outputs and cell.get("outputs"):
                    problems.append(f"source_embedded_outputs:{relative}")
            if cell.get("cell_type") == "code" and any(
                output.get("output_type") == "error" for output in cell.get("outputs", [])
            ):
                problems.append(f"stored_error:{relative}")
        if executable_cells and unexecuted_cells and (not source_contract or persist_inline_outputs):
            problems.append(f"unexecuted_code_cells:{relative}:{unexecuted_cells}/{executable_cells}")
        if any(not cell_id for cell_id in ids) or len(ids) != len(set(ids)):
            problems.append(f"cell_id_contract:{relative}")
        if source_contract:
            required_sections = (
                "## 1. Contexto e delimita",
                "## 2. Pergunta cient",
                "## 3. Hip",
                "## 4. Motiva",
                "## 5. Metodologia",
                "## 6. Resultados esperados e contrato de sa",
                "## Dados",
                "## Resultados",
                "## Conclus",
            )
            joined_source = "\n".join(sources)
            for section in required_sections:
                if section not in joined_source:
                    problems.append(f"source_section_contract:{relative}:{section}")
            phase_number = notebook_phase(path)
            notebook_code = str(project_metadata.get("notebook_code") or "")
            if phase_number is None or not notebook_code.startswith(f"F{phase_number}"):
                problems.append(f"notebook_code_contract:{relative}")
            if not str(project_metadata.get("figure_precode") or "").startswith("Fig" + notebook_code):
                problems.append(f"figure_precode_contract:{relative}")
            if not str(project_metadata.get("table_precode") or "").startswith("Tab" + notebook_code):
                problems.append(f"table_precode_contract:{relative}")
        else:
            if sum("<!-- NINO26-CABECALHO v1 -->" in source for source in sources) != 1:
                problems.append(f"header_contract:{relative}")
            if sum("<!-- NINO26-REFERENCIAS v1 -->" in source for source in sources) != 1:
                problems.append(f"footer_contract:{relative}")
        phase_number = notebook_phase(path)
        if not source_contract and phase_number is not None and phase_number >= 5 and not any(
            "parameters" in cell.get("metadata", {}).get("tags", []) for cell in cells
        ):
            problems.append(f"parameters_cell:{relative}")
    return _status(
        "passed" if not problems else "failed",
        f"{len(paths)} notebook(s) inspecionados; {len(problems)} problema(s) estático(s).",
        [_relative(root, path) for path in paths],
        passed=not problems,
        notebook_count=len(paths),
        excluded_notebook_count=len(excluded),
        excluded_notebooks=[_relative(root, path) for path in excluded],
        problems=problems,
    )


def collect_figure_lineage(
    root: Path,
    phase: int,
    expected_run_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    manifest_path = root / "data/processed/figuras_manifesto.csv"
    rows = [row for row in _read_csv(manifest_path) if str(row.get("fase")) == str(phase)]
    if not rows:
        return _status("not_run", "Nenhuma figura da fase está registrada.", [_relative(root, manifest_path)])
    problems: list[str] = []
    run_ids: set[str] = set()
    semantic_count = 0
    numeric_table_catalog: list[dict[str, str]] = []
    codes = [str(row.get("codigo") or "").strip() for row in rows]
    duplicate_codes = sorted({code for code in codes if code and codes.count(code) > 1})
    if duplicate_codes:
        problems.append(f"duplicate_figure_codes:{','.join(duplicate_codes)}")
    missing_codes: list[str] = []
    unexpected_codes: list[str] = []
    expected_codes = set(FIGURE_CODES_BY_PHASE.get(phase, frozenset()))
    if expected_codes:
        observed_codes = {code for code in codes if code}
        missing_codes = sorted(expected_codes - observed_codes)
        unexpected_codes = sorted(observed_codes - expected_codes)
        if len(rows) != len(expected_codes):
            problems.append(
                f"figure_catalog_count_mismatch:expected={len(expected_codes)}:actual={len(rows)}"
            )
        if missing_codes:
            problems.append(f"missing_figure_codes:{','.join(missing_codes)}")
        if unexpected_codes:
            problems.append(f"unexpected_figure_codes:{','.join(unexpected_codes)}")
    for row in rows:
        code = str(row.get("codigo") or "").strip()
        run_id = str(row.get("run_id") or "").strip()
        audit_level = str(row.get("audit_level") or "").strip()
        if audit_level == SEMANTIC_LEVEL and run_id:
            semantic_count += 1
            run_ids.add(run_id)
        else:
            problems.append(f"non_semantic:{code}")
        figures_root = (root / "data/processed/figures").resolve()
        figure = (figures_root / str(row.get("arquivo") or "")).resolve()
        try:
            figure.relative_to(figures_root)
        except ValueError:
            problems.append(f"figure_path_outside_root:{code}")
            continue
        if not figure.is_file():
            problems.append(f"missing_figure:{code}")
        else:
            try:
                with figure.open("rb") as stream:
                    signature = stream.read(8)
                if signature != b"\x89PNG\r\n\x1a\n":
                    problems.append(f"invalid_png_signature:{code}")
            except OSError as exc:
                problems.append(f"unreadable_figure:{code}:{type(exc).__name__}")
        local_manifest = root / f"data/processed/numeric-tables/fase{phase}/{code}/manifest.csv"
        local_rows = _read_csv(local_manifest)
        if not local_rows:
            problems.append(f"missing_numeric_manifest:{code}")
            continue
        local_table_names = {
            str(item.get("tabela") or "").strip()
            for item in local_rows
            if str(item.get("tabela") or "").strip()
        }
        declared_table_names = {
            value.strip()
            for value in str(row.get("tabelas") or "").split(";")
            if value.strip()
        }
        try:
            declared_count = int(row.get("n_tabelas"))
        except (TypeError, ValueError):
            declared_count = -1
        if declared_table_names != local_table_names or declared_count != len(local_rows):
            problems.append(f"figure_numeric_catalog_mismatch:{code}")
        for item in local_rows:
            table_name = str(item.get("tabela") or "").strip()
            if table_name:
                numeric_table_catalog.append(
                    {
                        "figure_code": code,
                        "table": table_name,
                        "sha256": str(item.get("sha256") or ""),
                        "source_run_id": str(item.get("source_run_id") or ""),
                    }
                )
            if not (
                _truthy(item.get("semantic_source"))
                and str(item.get("audit_level") or "") == SEMANTIC_LEVEL
                and str(item.get("run_id") or "").strip() == run_id
            ):
                problems.append(f"numeric_lineage_mismatch:{code}:{table_name or 'missing'}")
            if not table_name:
                problems.append(f"missing_numeric_table_name:{code}")
                continue
            numeric_root = local_manifest.parent.resolve()
            numeric_path = (numeric_root / table_name).resolve()
            try:
                numeric_path.relative_to(numeric_root)
            except ValueError:
                problems.append(f"numeric_table_outside_figure_dir:{code}:{table_name}")
                continue
            expected_numeric_hash = str(item.get("sha256") or "").strip()
            if not numeric_path.is_file():
                problems.append(f"missing_numeric_table:{code}:{table_name}")
            elif not expected_numeric_hash:
                problems.append(f"missing_numeric_table_hash:{code}:{table_name}")
            elif _sha256_file(numeric_path) != expected_numeric_hash:
                problems.append(f"numeric_table_hash_mismatch:{code}:{table_name}")

            source_run_id = str(item.get("source_run_id") or "").strip()
            source_path_text = str(item.get("source_path") or "").strip()
            source_manifest_text = str(item.get("source_manifest_path") or "").strip()
            source_manifest_hash = str(item.get("source_manifest_sha256") or "").strip()
            if not all((source_run_id, source_path_text, source_manifest_text, source_manifest_hash)):
                problems.append(f"incomplete_source_identity:{code}:{table_name}")
                continue
            if source_run_id != run_id:
                problems.append(f"source_run_id_mismatch:{code}:{table_name}")
            try:
                source_path = _resolve_recorded_path(root, source_path_text)
                source_manifest_path = _resolve_recorded_path(root, source_manifest_text)
            except (OSError, ValueError) as exc:
                problems.append(f"invalid_source_path:{code}:{table_name}:{exc}")
                continue
            if not source_manifest_path.is_file():
                problems.append(f"missing_source_manifest:{code}:{table_name}")
                continue
            if _sha256_file(source_manifest_path) != source_manifest_hash:
                problems.append(f"source_manifest_hash_mismatch:{code}:{table_name}")
                continue
            source_manifest = _read_json(source_manifest_path)
            if source_manifest is None:
                problems.append(f"invalid_source_manifest:{code}:{table_name}")
                continue
            if str(source_manifest.get("run_id") or "").strip() != source_run_id:
                problems.append(f"source_manifest_run_id_mismatch:{code}:{table_name}")
            artifact = source_manifest.get("artifact") or {}
            try:
                artifact_path = _resolve_recorded_path(root, artifact.get("path"))
            except (OSError, ValueError) as exc:
                problems.append(f"invalid_source_artifact_path:{code}:{table_name}:{exc}")
                continue
            if artifact_path != source_path:
                problems.append(f"source_artifact_path_mismatch:{code}:{table_name}")
            expected_source_hash = str(artifact.get("sha256") or "").strip()
            if not source_path.is_file():
                problems.append(f"missing_source_artifact:{code}:{table_name}")
            elif not expected_source_hash:
                problems.append(f"missing_source_artifact_hash:{code}:{table_name}")
            elif _sha256_file(source_path) != expected_source_hash:
                problems.append(f"source_artifact_hash_mismatch:{code}:{table_name}")
    expected = {str(item) for item in expected_run_ids or [] if str(item)}
    if expected and run_ids != expected:
        missing = ",".join(sorted(expected - run_ids)) or "none"
        unexpected = ",".join(sorted(run_ids - expected)) or "none"
        problems.append(f"run_id_set_mismatch:missing={missing}:unexpected={unexpected}")
    passed = not problems
    return _status(
        "passed" if passed else "failed",
        f"{semantic_count}/{len(rows)} figura(s) têm fonte semântica e run_id verificável.",
        [_relative(root, manifest_path)],
        passed=passed,
        figure_count=len(rows),
        expected_figure_count=len(expected_codes) if expected_codes else None,
        semantic_count=semantic_count,
        numeric_table_count=len(numeric_table_catalog),
        numeric_table_catalog_sha256=_compact_json_hash(
            sorted(
                numeric_table_catalog,
                key=lambda item: (item["figure_code"], item["table"]),
            )
        ),
        run_ids=sorted(run_ids),
        missing_codes=missing_codes,
        unexpected_codes=unexpected_codes,
        problems=problems,
    )


def collect_phase3_semantic_status(root: Path) -> dict[str, Any]:
    statistics = root / "data/processed/parquet/statistics"
    problems: list[str] = []
    run_ids: set[str] = set()
    valid_count = 0
    phase_table_sha256 = ""
    evidence: list[str] = []
    table_catalog: list[dict[str, str]] = []
    for table in PHASE3_TABLES:
        csv_path = statistics / f"{table}.csv"
        manifest_path = statistics / f"{table}.csv.manifest.json"
        evidence.extend([_relative(root, csv_path), _relative(root, manifest_path)])
        manifest = _read_json(manifest_path)
        if manifest is None:
            problems.append(f"missing_manifest:{table}")
            continue
        artifact = manifest.get("artifact") or {}
        try:
            artifact_path = _resolve_recorded_path(root, artifact.get("path") or csv_path)
        except (OSError, ValueError) as exc:
            problems.append(f"invalid_artifact_path:{table}:{exc}")
            continue
        if not artifact_path.is_file():
            problems.append(f"missing_artifact:{table}")
            continue
        if str(artifact.get("sha256") or "") != _sha256_file(artifact_path):
            problems.append(f"artifact_hash_mismatch:{table}")
            continue
        if table == "phase3_fases_semanais_en_ln":
            phase_table_sha256 = str(artifact.get("sha256") or "")
        input_problems = []
        for record in manifest.get("inputs", []):
            if isinstance(record, Mapping):
                input_problems.extend(_verify_record(root, record, f"semantic_input:{table}"))
        if input_problems:
            problems.extend(input_problems)
            continue
        run_id = str(manifest.get("run_id") or "").strip()
        if not run_id:
            problems.append(f"missing_run_id:{table}")
            continue
        run_ids.add(run_id)
        table_catalog.append(
            {
                "table": table,
                "path": _relative(root, artifact_path),
                "sha256": str(artifact.get("sha256") or ""),
                "manifest_path": _relative(root, manifest_path),
                "manifest_sha256": _sha256_file(manifest_path),
                "run_id": run_id,
            }
        )
        valid_count += 1
    if len(run_ids) > 1:
        problems.append(f"mixed_run_ids:{','.join(sorted(run_ids))}")
    complete = valid_count == len(PHASE3_TABLES) and len(run_ids) == 1 and not problems
    return _status(
        "complete" if complete else ("partial" if valid_count else "missing"),
        f"{valid_count}/{len(PHASE3_TABLES)} tabelas semânticas válidas; run_ids={len(run_ids)}.",
        evidence,
        complete=complete,
        valid_tables=valid_count,
        expected_tables=len(PHASE3_TABLES),
        run_id=next(iter(run_ids)) if len(run_ids) == 1 else "",
        phase_table_sha256=phase_table_sha256,
        table_catalog=table_catalog,
        table_catalog_sha256=_compact_json_hash(table_catalog),
        problems=problems,
    )


def _zarr_attrs(path: Path) -> dict[str, Any]:
    direct = _read_json(path / ".zattrs")
    if direct is not None:
        return direct
    zarr_v3 = _read_json(path / "zarr.json")
    if zarr_v3 and isinstance(zarr_v3.get("attributes"), dict):
        return dict(zarr_v3["attributes"])
    consolidated = _read_json(path / ".zmetadata")
    if consolidated:
        attrs = (consolidated.get("metadata") or {}).get(".zattrs")
        if isinstance(attrs, dict):
            return dict(attrs)
    return {}


def _zarr_storage_inventory(path: Path) -> dict[str, Any]:
    """Inventário barato dos arquivos que materializam arrays/chunks Zarr."""

    excluded = {".zattrs", ".zmetadata", ".zgroup", "zarr.json"}
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.name not in excluded
    )
    payload = [
        (
            item.relative_to(path).as_posix(),
            item.stat().st_size,
            item.stat().st_mtime_ns,
        )
        for item in files
    ]
    return {
        "files": files,
        "file_count": len(files),
        "size_bytes": sum(item.stat().st_size for item in files),
        "state_sha256": hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _native_grid_hash(latitude: Iterable[Any], longitude: Iterable[Any]) -> str:
    lat = [float(value) for value in latitude]
    lon = [float(value) for value in longitude]
    payload = (
        f"chirps-native-weekly-v2|lat={len(lat)}|lon={len(lon)}|"
        + ",".join(f"{value:.8f}" for value in lat)
        + "|"
        + ",".join(f"{value:.8f}" for value in lon)
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _inspect_chirps_zarr(target: Path) -> tuple[list[str], dict[str, Any]]:
    """Abre o Zarr consolidado e confere arrays, chunks e eixos canônicos."""

    problems: list[str] = []
    details: dict[str, Any] = {
        "data_vars": [],
        "variable_contract": {},
        "n_time": 0,
        "n_latitude": 0,
        "n_longitude": 0,
        "grid_hash": "",
    }
    for metadata_name in (".zgroup", ".zattrs", ".zmetadata"):
        if not (target / metadata_name).is_file():
            problems.append(f"target_zarr_metadata_missing:{metadata_name}")
    inventory = _zarr_storage_inventory(target) if target.is_dir() else {
        "files": [],
        "file_count": 0,
        "size_bytes": 0,
        "state_sha256": "",
    }
    details.update({key: value for key, value in inventory.items() if key != "files"})
    if not inventory["files"]:
        problems.append("target_zarr_has_no_arrays_or_chunks")
    if problems:
        return problems, details

    try:
        import xarray as xr

        dataset = xr.open_zarr(target, consolidated=True)
    except Exception as exc:
        problems.append(f"target_zarr_open_failed:{type(exc).__name__}")
        return problems, details
    try:
        data_vars = set(dataset.data_vars)
        details["data_vars"] = sorted(data_vars)
        missing_variables = sorted(CHIRPS_REQUIRED_VARIABLES - data_vars)
        if missing_variables:
            problems.append(f"target_variables_missing:{','.join(missing_variables)}")
        sizes = {
            "time": int(dataset.sizes.get("time", 0)),
            "latitude": int(dataset.sizes.get("latitude", 0)),
            "longitude": int(dataset.sizes.get("longitude", 0)),
        }
        details.update(
            {
                "n_time": sizes["time"],
                "n_latitude": sizes["latitude"],
                "n_longitude": sizes["longitude"],
            }
        )
        if any(value <= 0 for value in sizes.values()):
            problems.append("target_dimensions_empty")
        # The persisted contract inventories every data variable, not only
        # the required-core subset used to detect missing layers.  Recording
        # the full catalogue keeps optional SPI/extreme arrays auditable.
        for name in sorted(data_vars):
            variable = dataset[name]
            details["variable_contract"][name] = {
                "dimensions": ";".join(variable.dims),
                "shape": "x".join(str(value) for value in variable.shape),
            }
            array_dir = target / name
            if not (array_dir / ".zarray").is_file():
                problems.append(f"target_array_metadata_missing:{name}")
            else:
                chunk_files = [
                    item
                    for item in array_dir.iterdir()
                    if item.is_file() and not item.name.startswith(".")
                ]
                if not chunk_files:
                    problems.append(f"target_array_chunks_missing:{name}")
        for coordinate in ("time", "latitude", "longitude"):
            array_dir = target / coordinate
            if coordinate not in dataset.coords:
                problems.append(f"target_coordinate_missing:{coordinate}")
            elif not (array_dir / ".zarray").is_file() or not any(
                item.is_file() and not item.name.startswith(".")
                for item in array_dir.iterdir()
            ):
                problems.append(f"target_coordinate_not_materialized:{coordinate}")
        if sizes["latitude"] and sizes["longitude"]:
            grid_hash = _native_grid_hash(
                dataset["latitude"].values.tolist(),
                dataset["longitude"].values.tolist(),
            )
            details["grid_hash"] = grid_hash
            if grid_hash != CHIRPS_FROZEN_GRID_HASH:
                problems.append("target_frozen_grid_hash_mismatch")
        if "pixel_id" in dataset and sizes["latitude"] and sizes["longitude"]:
            pixel_ids = [int(value) for value in dataset["pixel_id"].values.ravel().tolist()]
            if pixel_ids != list(range(sizes["latitude"] * sizes["longitude"])):
                problems.append("target_pixel_id_not_exact_row_major")
        time_index = dataset.indexes.get("time")
        if time_index is None or len(time_index) == 0:
            problems.append("target_weekly_time_missing")
        else:
            if time_index.has_duplicates or not time_index.is_monotonic_increasing:
                problems.append("target_weekly_time_order_invalid")
            if not bool((time_index.dayofweek == 6).all()):
                problems.append("target_weekly_time_not_w_sun")
            if len(time_index) > 1:
                deltas = [
                    (time_index[index] - time_index[index - 1]).days
                    for index in range(1, len(time_index))
                ]
                if any(delta != 7 for delta in deltas):
                    problems.append("target_weekly_time_has_gaps")
    except Exception as exc:
        problems.append(f"target_zarr_inspection_failed:{type(exc).__name__}")
    finally:
        dataset.close()
    return problems, details


def _audit_chirps_variable_contract(
    root: Path,
    manifest: Mapping[str, Any],
    zarr_details: Mapping[str, Any],
) -> list[str]:
    problems: list[str] = []
    try:
        path = _resolve_recorded_path(root, manifest.get("target_variable_contract"))
    except (OSError, ValueError) as exc:
        return [f"target_variable_contract_invalid_path:{exc}"]
    if _relative(root, path) != "data/processed/parquet/statistics/phase4_chirps_target_variable_contract.csv":
        problems.append("target_variable_contract_noncanonical_path")
    expected_hash = str(manifest.get("target_variable_contract_sha256") or "")
    if not path.is_file() or not _is_sha256(expected_hash):
        return [*problems, "target_variable_contract_mismatch"]
    if _sha256_file(path) != expected_hash:
        return [*problems, "target_variable_contract_hash_mismatch"]
    rows = _read_csv(path)
    required_columns = {
        "build_id",
        "variable",
        "role",
        "dimensions",
        "shape",
        "dtype",
        "units",
        "method",
        "source_index",
        "grid_hash_sha256",
        "target_contract_version",
        "block_signature_sha256",
        "numeric_authority",
    }
    if not rows or not required_columns.issubset(rows[0]):
        return [*problems, "target_variable_contract_schema_mismatch"]
    variables = [str(row.get("variable") or "").strip() for row in rows]
    if len(variables) != len(set(variables)):
        problems.append("target_variable_contract_duplicate_variables")
    expected_variables = set(zarr_details.get("data_vars") or [])
    if set(variables) != expected_variables:
        problems.append("target_variable_contract_catalog_mismatch")
    expected_identity = {
        "build_id": str(manifest.get("build_id") or ""),
        "grid_hash_sha256": CHIRPS_FROZEN_GRID_HASH,
        "target_contract_version": CHIRPS_TARGET_CONTRACT,
        "block_signature_sha256": str(manifest.get("signature_sha256") or ""),
    }
    contracts = zarr_details.get("variable_contract") or {}
    for row in rows:
        name = str(row.get("variable") or "").strip()
        if any(str(row.get(key) or "") != value for key, value in expected_identity.items()):
            problems.append(f"target_variable_contract_identity_mismatch:{name}")
        expected = contracts.get(name) or {}
        if (
            str(row.get("dimensions") or "") != str(expected.get("dimensions") or "")
            or str(row.get("shape") or "") != str(expected.get("shape") or "")
        ):
            problems.append(f"target_variable_contract_shape_mismatch:{name}")
    return problems


def _audit_chirps_pixel_inventory(
    root: Path,
    manifest: Mapping[str, Any],
    zarr_details: Mapping[str, Any],
) -> list[str]:
    problems: list[str] = []
    try:
        path = _resolve_recorded_path(root, manifest.get("promoted_pixel_inventory"))
    except (OSError, ValueError) as exc:
        return [f"promoted_pixel_inventory_invalid_path:{exc}"]
    if _relative(root, path) != "data/processed/parquet/features/phase4_chirps_native_pixels.csv":
        problems.append("promoted_pixel_inventory_noncanonical_path")
    expected_hash = str(manifest.get("promoted_pixel_inventory_sha256") or "")
    if not path.is_file() or not _is_sha256(expected_hash):
        return [*problems, "promoted_pixel_inventory_hash_missing"]
    if _sha256_file(path) != expected_hash:
        return [*problems, "promoted_pixel_inventory_hash_mismatch"]
    rows = _read_csv(path)
    required_columns = {
        "pixel_id",
        "grid_row",
        "grid_column",
        "lat",
        "lon",
        "grid_hash",
        "native_pixel",
        "interpolated",
        "brazil_fraction",
        "brazil_center",
        "brazil_mask_method",
    }
    if not rows or not required_columns.issubset(rows[0]):
        return [*problems, "promoted_pixel_inventory_schema_mismatch"]
    n_latitude = int(zarr_details.get("n_latitude") or 0)
    n_longitude = int(zarr_details.get("n_longitude") or 0)
    expected_count = n_latitude * n_longitude
    if len(rows) != expected_count:
        problems.append(
            f"promoted_pixel_inventory_count_mismatch:expected={expected_count}:actual={len(rows)}"
        )
    try:
        ordered = sorted(rows, key=lambda row: int(row["pixel_id"]))
        ids = [int(row["pixel_id"]) for row in ordered]
        if ids != list(range(expected_count)):
            problems.append("promoted_pixel_inventory_ids_not_exact")
        for index, row in enumerate(ordered):
            if int(row["grid_row"]) != index // max(n_longitude, 1) or int(row["grid_column"]) != index % max(n_longitude, 1):
                problems.append("promoted_pixel_inventory_not_row_major")
                break
            fraction = float(row["brazil_fraction"])
            if not 0.0 <= fraction <= 1.0:
                problems.append("promoted_pixel_inventory_fraction_out_of_bounds")
                break
    except (KeyError, TypeError, ValueError):
        problems.append("promoted_pixel_inventory_values_invalid")
    if {str(row.get("grid_hash") or "") for row in rows} != {CHIRPS_FROZEN_GRID_HASH}:
        problems.append("promoted_pixel_inventory_grid_hash_mismatch")
    if not all(_truthy(row.get("native_pixel")) and not _truthy(row.get("interpolated")) for row in rows):
        problems.append("promoted_pixel_inventory_native_contract_mismatch")
    return problems


def _audit_chirps_deep_receipt(
    root: Path,
    manifest: Mapping[str, Any],
    target: Path,
) -> list[str]:
    """Bind a fresh byte-level validator receipt to the currently promoted store."""

    report_path = root / CHIRPS_DEEP_VALIDATION_REPORT
    report = _read_json(report_path)
    if report is None:
        return ["chirps_deep_validation_report_missing_or_invalid"]
    problems: list[str] = []
    if report.get("schema_version") != "nino26-chirps-deep-validation/v1":
        problems.append("chirps_deep_validation_report_schema_mismatch")
    if str(report.get("target_path") or "").replace("\\", "/") != _relative(root, target):
        problems.append("chirps_deep_validation_target_path_mismatch")
    validation = report.get("validation")
    if not isinstance(validation, Mapping) or validation.get("valid") is not True:
        return [*problems, "chirps_deep_validation_failed"]
    if validation.get("problems") not in ([], ()):
        problems.append("chirps_deep_validation_has_problems")
    expected_values = {
        "build_id": manifest.get("build_id"),
        "block_signature_sha256": manifest.get("signature_sha256"),
        "target_data_content_sha256": manifest.get(
            "promoted_target_data_content_sha256"
        ),
    }
    for key, expected in expected_values.items():
        if str(validation.get(key) or "") != str(expected or ""):
            problems.append(f"chirps_deep_validation_identity_mismatch:{key}")
    if str(report.get("target_data_state_sha256") or "") != str(
        manifest.get("promoted_target_data_state_sha256") or ""
    ):
        problems.append("chirps_deep_validation_state_mismatch")
    manifest_value = str(validation.get("manifest") or "").strip()
    try:
        manifest_path = _resolve_recorded_path(root, manifest_value)
    except (OSError, ValueError) as exc:
        problems.append(f"chirps_deep_validation_manifest_path_invalid:{exc}")
        manifest_path = None
    if manifest_path is not None:
        expected_manifest_hash = str(report.get("build_manifest_sha256") or "")
        if (
            not manifest_path.is_file()
            or not _is_sha256(expected_manifest_hash)
            or _sha256_file(manifest_path) != expected_manifest_hash
        ):
            problems.append("chirps_deep_validation_manifest_hash_mismatch")
    for report_key, relative in (
        ("builder_script_sha256", "scripts/build_phase4_chirps_targets.py"),
        ("target_module_sha256", "src/nino_brasil/targets/chirps_native.py"),
    ):
        value = str(report.get(report_key) or "")
        code_path = root / relative
        if (
            not _is_sha256(value)
            or not code_path.is_file()
            or _sha256_file(code_path) != value
            or value != str(manifest.get(report_key) or "")
        ):
            problems.append(f"chirps_deep_validation_{report_key}_mismatch")
    checked_at = _parse_time(report.get("checked_at_utc"))
    promoted_at = _parse_time(manifest.get("promoted_utc"))
    if (
        checked_at == datetime.min.replace(tzinfo=timezone.utc)
        or promoted_at == datetime.min.replace(tzinfo=timezone.utc)
        or checked_at < promoted_at
    ):
        problems.append("chirps_deep_validation_report_not_fresh")
    return problems


def collect_chirps_status(root: Path) -> dict[str, Any]:
    target = root / "data/processed/zarr/features/chirps_native_weekly_targets.zarr"
    manifests = sorted((root / "data/interim/chirps_weekly_native_blocks").glob("*/manifest.json"))
    attrs = _zarr_attrs(target) if target.is_dir() else {}
    zarr_problems, zarr_details = (
        _inspect_chirps_zarr(target)
        if target.is_dir()
        else (["target_missing"], {"data_vars": [], "variable_contract": {}})
    )
    candidates: list[dict[str, Any]] = []
    for path in manifests:
        manifest = _read_json(path)
        if manifest is not None and str(manifest.get("build_status") or "") == "canonical":
            candidates.append({"path": path, "manifest": manifest})
    eligible: list[dict[str, Any]] = []
    for candidate in candidates:
        manifest = candidate["manifest"]
        problems: list[str] = list(zarr_problems)
        if str(manifest.get("contract") or "") != CHIRPS_BLOCK_CONTRACT:
            problems.append("block_contract_version_mismatch")
        if str(manifest.get("full_grid_hash_sha256") or "") != CHIRPS_FROZEN_GRID_HASH:
            problems.append("manifest_frozen_grid_hash_mismatch")
        if str(manifest.get("target_contract_version") or "") != CHIRPS_TARGET_CONTRACT:
            problems.append("manifest_target_contract_version_mismatch")
        if str(manifest.get("promoted_target_grid_hash_sha256") or "") != CHIRPS_FROZEN_GRID_HASH:
            problems.append("manifest_promoted_grid_hash_mismatch")
        if not _is_sha256(manifest.get("signature_sha256")):
            problems.append("block_signature_invalid")
        if not manifest.get("build_complete"):
            problems.append("build_complete_false")
        if manifest.get("blocks_complete") is not True or _safe_int(manifest.get("remaining_blocks"), -1) != 0:
            problems.append("blocks_not_declared_complete")
        if manifest.get("promotion_status") != "promoted_after_deep_validation":
            problems.append("promotion_status_missing")
        if not target.is_dir():
            problems.append("target_missing")
        if attrs.get("deep_validation_passed") is not True:
            problems.append("target_deep_validation_missing")
        if not str(attrs.get("deep_validation_timestamp_utc") or "").strip():
            problems.append("target_deep_validation_timestamp_missing")
        if str(manifest.get("build_id") or "") != str(attrs.get("build_id") or ""):
            problems.append("build_id_mismatch")
        if str(manifest.get("signature_sha256") or "") != str(attrs.get("block_signature_sha256") or ""):
            problems.append("block_signature_mismatch")
        if str(attrs.get("target_contract_version") or "") != CHIRPS_TARGET_CONTRACT:
            problems.append("target_contract_version_mismatch")
        if str(attrs.get("grid_hash_sha256") or "") != CHIRPS_FROZEN_GRID_HASH:
            problems.append("target_grid_hash_attribute_mismatch")
        if not str(attrs.get("build_status") or "").startswith("canonical"):
            problems.append("target_build_status_not_canonical")
        if "interpolation=false" not in str(attrs.get("spatial_operation") or ""):
            problems.append("target_spatial_operation_allows_interpolation")
        if bool(attrs.get("include_spi")) != bool(manifest.get("include_spi")):
            problems.append("target_include_spi_mismatch")
        if bool(attrs.get("include_extremes")) != bool(manifest.get("include_extremes")):
            problems.append("target_include_extremes_mismatch")
        promoted_target = str(manifest.get("promoted_target") or "").replace("\\", "/")
        if promoted_target != _relative(root, target):
            problems.append("promoted_target_mismatch")
        deep_validation = manifest.get("deep_validation") or {}
        if not isinstance(deep_validation, Mapping):
            deep_validation = {}
        if not deep_validation.get("valid") or deep_validation.get("errors") not in ([], ()):
            problems.append("manifest_deep_validation_failed")
        dimension_pairs = (
            ("n_time", "n_time"),
            ("n_latitude", "n_latitude"),
            ("n_longitude", "n_longitude"),
        )
        for manifest_key, zarr_key in dimension_pairs:
            if _safe_int(deep_validation.get(manifest_key), -1) != _safe_int(zarr_details.get(zarr_key), 0):
                problems.append(f"deep_validation_dimension_mismatch:{manifest_key}")
        if str(deep_validation.get("grid_hash") or "") != CHIRPS_FROZEN_GRID_HASH:
            problems.append("deep_validation_grid_hash_mismatch")

        try:
            latitude_count = int(manifest.get("latitude_count") or 0)
            longitude_count = int(manifest.get("longitude_count") or 0)
            block_size = int(manifest.get("latitude_block_size") or 0)
        except (TypeError, ValueError):
            latitude_count = longitude_count = block_size = 0
        if (
            latitude_count != int(zarr_details.get("n_latitude") or 0)
            or longitude_count != int(zarr_details.get("n_longitude") or 0)
            or block_size <= 0
        ):
            problems.append("manifest_target_dimensions_mismatch")
        expected_blocks = [
            f"latitude_{start:03d}_{min(start + block_size, latitude_count):03d}.zarr"
            for start in range(0, latitude_count, block_size)
        ] if latitude_count > 0 and block_size > 0 else []
        completed_blocks = [str(value) for value in manifest.get("completed_blocks") or []]
        block_records = manifest.get("block_records") or {}
        if completed_blocks != sorted(set(completed_blocks)) or set(completed_blocks) != set(expected_blocks):
            problems.append("completed_block_catalog_mismatch")
        if not isinstance(block_records, Mapping) or set(block_records) != set(expected_blocks):
            problems.append("block_record_catalog_mismatch")
        elif expected_blocks:
            block_root = candidate["path"].parent
            for name in expected_blocks:
                record = block_records.get(name) or {}
                if not isinstance(record, Mapping):
                    problems.append(f"block_record_invalid:{name}")
                    continue
                try:
                    start = int(record.get("latitude_start"))
                    stop = int(record.get("latitude_stop"))
                except (TypeError, ValueError):
                    start = stop = -1
                expected_start = expected_blocks.index(name) * block_size
                expected_stop = min(expected_start + block_size, latitude_count)
                validated = record.get("validated") or {}
                if not isinstance(validated, Mapping):
                    validated = {}
                if start != expected_start or stop != expected_stop:
                    problems.append(f"block_extent_mismatch:{name}")
                if not (
                    _is_sha256(record.get("data_state_sha256"))
                    and _is_sha256(record.get("data_content_sha256"))
                    and validated.get("valid") is True
                    and validated.get("errors") in ([], ())
                    and _safe_int(validated.get("n_latitude"), -1) == expected_stop - expected_start
                    and _safe_int(validated.get("n_longitude"), -1) == longitude_count
                    and _safe_int(validated.get("n_time"), -1) == _safe_int(zarr_details.get("n_time"), 0)
                ):
                    problems.append(f"block_record_invalid:{name}")
                block_path = block_root / name
                if not block_path.is_dir():
                    problems.append(f"block_store_missing:{name}")
                else:
                    state = _zarr_storage_inventory(block_path)["state_sha256"]
                    if state != str(record.get("data_state_sha256") or ""):
                        problems.append(f"block_state_hash_mismatch:{name}")

        target_content_hash = str(manifest.get("promoted_target_data_content_sha256") or "")
        target_state_hash = str(manifest.get("promoted_target_data_state_sha256") or "")
        if not _is_sha256(target_content_hash):
            problems.append("promoted_target_content_hash_missing")
        if target_state_hash != str(zarr_details.get("state_sha256") or ""):
            problems.append("promoted_target_state_hash_mismatch")
        for key, relative_path in (
            ("builder_script_sha256", "scripts/build_phase4_chirps_targets.py"),
            ("target_module_sha256", "src/nino_brasil/targets/chirps_native.py"),
        ):
            expected = str(manifest.get(key) or "")
            path = root / relative_path
            if not expected or not path.is_file() or _sha256_file(path) != expected:
                problems.append(f"{key}_mismatch")
        problems.extend(_audit_chirps_variable_contract(root, manifest, zarr_details))
        problems.extend(_audit_chirps_pixel_inventory(root, manifest, zarr_details))
        problems.extend(_audit_chirps_deep_receipt(root, manifest, target))
        candidate["problems"] = problems
        if not problems:
            eligible.append(candidate)

    if eligible:
        selected = sorted(
            eligible,
            key=lambda item: (_parse_time(item["manifest"].get("promoted_utc")), str(item["manifest"].get("build_id"))),
            reverse=True,
        )[0]
        manifest = selected["manifest"]
        return _status(
            "promoted",
            "Alvo CHIRPS nativo promovido após validação profunda; identidade e código conferem.",
            [
                _relative(root, target),
                _relative(root, selected["path"]),
                CHIRPS_DEEP_VALIDATION_REPORT,
            ],
            promoted=True,
            build_id=manifest.get("build_id"),
            target_contract_version=CHIRPS_TARGET_CONTRACT,
            block_signature_sha256=manifest.get("signature_sha256"),
            completed_blocks=len(manifest.get("completed_blocks") or []),
            total_blocks=math.ceil(
                _safe_int(manifest.get("latitude_count"), 0)
                / max(_safe_int(manifest.get("latitude_block_size"), 1), 1)
            ),
            target_content_sha256=manifest.get("promoted_target_data_content_sha256"),
            target_state_sha256=manifest.get("promoted_target_data_state_sha256"),
            grid_hash_sha256=CHIRPS_FROZEN_GRID_HASH,
            n_time=_safe_int(zarr_details.get("n_time"), 0),
            n_latitude=_safe_int(zarr_details.get("n_latitude"), 0),
            n_longitude=_safe_int(zarr_details.get("n_longitude"), 0),
            native_pixel_count=(
                _safe_int(zarr_details.get("n_latitude"), 0)
                * _safe_int(zarr_details.get("n_longitude"), 0)
            ),
            native_pixel_preservation_verified=True,
            interpolation_applied=False,
            spatial_operation=str(attrs.get("spatial_operation") or ""),
            pixel_inventory_sha256=manifest.get("promoted_pixel_inventory_sha256"),
            variable_contract_sha256=manifest.get("target_variable_contract_sha256"),
            deep_validation_report_sha256=(
                _sha256_file(root / CHIRPS_DEEP_VALIDATION_REPORT)
                if (root / CHIRPS_DEEP_VALIDATION_REPORT).is_file()
                else ""
            ),
            problems=[],
        )

    if not candidates:
        state = "invalid" if target.exists() else "missing"
        return _status(
            state,
            "Nenhum manifesto canônico CHIRPS semanal foi encontrado.",
            [_relative(root, target)],
            promoted=False,
            target_contract_version=CHIRPS_TARGET_CONTRACT,
            native_pixel_preservation_verified=False,
            interpolation_applied=None,
        )
    selected = sorted(
        candidates,
        key=lambda item: (
            len(item["manifest"].get("completed_blocks") or []),
            str(item["manifest"].get("signature_sha256") or ""),
        ),
        reverse=True,
    )[0]
    manifest = selected["manifest"]
    completed = len(manifest.get("completed_blocks") or [])
    total = math.ceil(
        _safe_int(manifest.get("latitude_count"), 0)
        / max(_safe_int(manifest.get("latitude_block_size"), 1), 1)
    )
    state = "in_progress" if not manifest.get("build_complete") else "invalid"
    return _status(
        state,
        f"{completed}/{total or '?'} blocos completos; promoção ainda não é válida.",
        [_relative(root, selected["path"]), _relative(root, target)],
        promoted=False,
        build_id=manifest.get("build_id"),
        target_contract_version=CHIRPS_TARGET_CONTRACT,
        block_signature_sha256=manifest.get("signature_sha256"),
        completed_blocks=completed,
        total_blocks=total,
        native_pixel_preservation_verified=False,
        interpolation_applied=None,
        problems=selected.get("problems", ["build_incomplete"]),
    )


def _csv_column_sets(path: Path, columns: Sequence[str]) -> tuple[dict[str, set[str]], int, list[str]]:
    values = {column: set() for column in columns}
    problems: list[str] = []
    count = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = [column for column in columns if column not in (reader.fieldnames or [])]
            if missing:
                problems.append(f"missing_columns:{path.name}:{','.join(missing)}")
            for row in reader:
                count += 1
                for column in columns:
                    value = str(row.get(column) or "").strip()
                    if value:
                        values[column].add(value)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        problems.append(f"unreadable_csv:{path.name}:{type(exc).__name__}")
    return values, count, problems


def _table_column_sets(path: Path, columns: Sequence[str]) -> tuple[dict[str, set[str]], int, list[str]]:
    """Read unique contract values from CSV or Parquet without loading wide data."""

    if path.suffix.lower() == ".csv":
        return _csv_column_sets(path, columns)
    values = {column: set() for column in columns}
    if path.suffix.lower() != ".parquet":
        return values, 0, [f"unsupported_table_format:{path.name}"]
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        missing = [column for column in columns if column not in available]
        if missing:
            return values, int(parquet.metadata.num_rows), [
                f"missing_columns:{path.name}:{','.join(missing)}"
            ]
        table = parquet.read(columns=list(columns))
        for column in columns:
            unique = pc.unique(table[column]).to_pylist()
            values[column] = {
                str(value).strip()
                for value in unique
                if value is not None and str(value).strip()
            }
        return values, int(parquet.metadata.num_rows), []
    except Exception as exc:
        return values, 0, [f"unreadable_parquet:{path.name}:{type(exc).__name__}"]


def _audit_semantic_csv_sidecar(
    root: Path,
    path: Path,
    expected_run_id: str,
    *,
    expected_contract: Mapping[str, str] | None = None,
    required_inputs: frozenset[Path] | None = None,
) -> list[str]:
    """Confere o hash e a linhagem persistidos para uma saída CSV canônica."""

    sidecar_path = Path(f"{path}.manifest.json")
    manifest = _read_json(sidecar_path)
    if manifest is None:
        return [f"missing_output_manifest:{path.name}"]
    problems: list[str] = []
    if str(manifest.get("run_id") or "").strip() != expected_run_id:
        problems.append(f"output_manifest_run_id_mismatch:{path.name}")
    if expected_contract is not None and manifest.get("schema_version") != "nino-brasil-phase4-semantic-output-v1":
        problems.append(f"output_manifest_schema_mismatch:{path.name}")
    contract = manifest.get("contract") or {}
    if str(contract.get("phase") or "").upper() not in {"4", "F4"}:
        problems.append(f"output_manifest_phase_mismatch:{path.name}")
    for key, expected in (expected_contract or {}).items():
        if not expected or str(contract.get(key) or "") != expected:
            problems.append(f"output_manifest_contract_mismatch:{path.name}:{key}")
    artifact = manifest.get("artifact") or {}
    try:
        artifact_path = _resolve_recorded_path(root, artifact.get("path"))
    except (OSError, ValueError) as exc:
        return [*problems, f"output_manifest_artifact_path_invalid:{path.name}:{exc}"]
    if artifact_path != path.resolve():
        problems.append(f"output_manifest_artifact_path_mismatch:{path.name}")
    expected_hash = str(artifact.get("sha256") or "")
    if not path.is_file() or not _is_sha256(expected_hash) or _sha256_file(path) != expected_hash:
        problems.append(f"output_manifest_artifact_hash_mismatch:{path.name}")
    raw_inputs = manifest.get("inputs")
    observed_inputs: set[Path] = set()
    if not isinstance(raw_inputs, list) or not raw_inputs:
        problems.append(f"output_manifest_inputs_missing:{path.name}")
    else:
        for index, record in enumerate(raw_inputs):
            if isinstance(record, Mapping):
                problems.extend(_verify_record(root, record, f"output_manifest_input:{path.name}"))
                try:
                    input_path = _resolve_recorded_path(root, record.get("path"))
                except (OSError, ValueError):
                    continue
                if input_path in observed_inputs:
                    problems.append(f"output_manifest_input_duplicate:{path.name}:{_relative(root, input_path)}")
                observed_inputs.add(input_path)
                if input_path in (required_inputs or frozenset()):
                    fingerprint = record.get("tree_sha256") if input_path.is_dir() else record.get("sha256")
                    if not _is_sha256(fingerprint):
                        problems.append(f"output_manifest_required_input_sha256_invalid:{path.name}:{_relative(root, input_path)}")
            else:
                problems.append(f"output_manifest_input_invalid:{path.name}:{index}")
    missing_required = sorted(
        _relative(root, input_path)
        for input_path in (required_inputs or frozenset()) - observed_inputs
    )
    if missing_required:
        problems.append(f"output_manifest_required_inputs_missing:{path.name}:{','.join(missing_required)}")
    return problems


def _current_ibge_geometry_contract(
    root: Path,
) -> tuple[dict[str, str], tuple[Path, ...], list[str]]:
    """Fingerprint the complete current IBGE region and biome shapefile bundles."""

    shapefiles = {
        "regions": root / "data/interim/ibge/BR_Regioes_2024/BR_Regioes_2024.shp",
        "biomes": root
        / "data/interim/ibge/Biomas_2025/lml_bioma_e250k_v20250911_A.shp",
    }
    bundle_hashes: dict[str, str] = {}
    all_components: list[Path] = []
    problems: list[str] = []
    for label, shapefile in shapefiles.items():
        required = tuple(
            shapefile.with_suffix(suffix)
            for suffix in (".shp", ".shx", ".dbf", ".prj")
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            problems.extend(
                f"ibge_{label}_bundle_component_missing:{_relative(root, path)}"
                for path in missing
            )
            bundle_hashes[label] = ""
            continue
        components = tuple(
            sorted(
                (
                    candidate.resolve()
                    for candidate in shapefile.parent.glob(shapefile.stem + ".*")
                    if candidate.is_file()
                ),
                key=lambda candidate: candidate.name.casefold(),
            )
        )
        if not components:
            problems.append(f"ibge_{label}_bundle_empty:{_relative(root, shapefile)}")
            bundle_hashes[label] = ""
            continue
        records = [
            f"{component.name}\0{component.stat().st_size}\0{_sha256_file(component)}"
            for component in components
        ]
        bundle_hashes[label] = hashlib.sha256(
            "\n".join(records).encode("utf-8")
        ).hexdigest()
        all_components.extend(components)
    regions = bundle_hashes.get("regions", "")
    biomes = bundle_hashes.get("biomes", "")
    combined = (
        hashlib.sha256(
            f"regions={regions}\nbiomes={biomes}".encode("utf-8")
        ).hexdigest()
        if regions and biomes
        else ""
    )
    return (
        {
            "ibge_regions_bundle_sha256": regions,
            "ibge_biomes_bundle_sha256": biomes,
            "ibge_geometry_bundle_sha256": combined,
        },
        tuple(all_components),
        problems,
    )


def _phase4_quick_required_inputs(
    root: Path,
    target_block_signature_sha256: str,
    geometry_components: Sequence[Path],
) -> frozenset[Path]:
    """Return the exact file catalogue bound into every F4C quick sidecar."""

    return frozenset(
        path.resolve()
        for path in (
            root / "data/processed/parquet/features/nino34_master_weekly.csv",
            root / "data/processed/parquet/statistics/phase3_fases_semanais_en_ln.csv",
            root / "data/processed/parquet/statistics/phase3_fases_semanais_en_ln.csv.manifest.json",
            root
            / "data/interim/chirps_weekly_native_blocks"
            / target_block_signature_sha256[:16]
            / "manifest.json",
            root / "data/processed/parquet/statistics/phase4_chirps_target_variable_contract.csv",
            root / "data/processed/parquet/features/phase4_chirps_native_pixels.csv",
            root / "data/processed/parquet/statistics/phase4C_native_pixel_membership_exact.parquet",
            *geometry_components,
            root / "scripts/run_fase4c_regional.py",
            root / "src/nino_brasil/stats/lag_analysis.py",
        )
    )


def _audit_phase4_quick_sidecar(
    root: Path,
    path: Path,
    *,
    expected_run_id: str,
    expected_contract: Mapping[str, str],
    expected_inputs: frozenset[Path],
) -> list[str]:
    """Audit a quick-table sidecar beyond mere presence or filename.

    The quick run is operational evidence, not an official result.  Still, it
    must bind the exact table bytes, the canonical F3/CHIRPS identities, the
    current F4 code and the complete producer input catalogue.
    """

    sidecar_path = Path(f"{path}.manifest.json")
    manifest = _read_json(sidecar_path)
    if manifest is None:
        return [f"missing_quick_output_manifest:{path.name}"]
    problems = _audit_semantic_csv_sidecar(root, path, expected_run_id)
    if manifest.get("schema_version") != "nino-brasil-phase4-semantic-output-v1":
        problems.append(f"quick_output_manifest_schema_mismatch:{path.name}")
    contract = manifest.get("contract") or {}
    if str(contract.get("stage") or "") != "F4C_QUICK":
        problems.append(f"quick_output_manifest_stage_mismatch:{path.name}")
    if str(contract.get("selection_contract") or "") != "quick:key-predictor":
        problems.append(f"quick_output_manifest_selection_contract_mismatch:{path.name}")
    if contract.get("predictor_names") != ["nino34_ssta"]:
        problems.append(f"quick_output_manifest_predictor_catalog_mismatch:{path.name}")
    if _safe_int(contract.get("field_permutations"), -1) < 19:
        problems.append(f"quick_output_manifest_field_permutations_invalid:{path.name}")
    for key, expected in expected_contract.items():
        if not expected or str(contract.get(key) or "") != expected:
            problems.append(f"quick_output_manifest_contract_mismatch:{path.name}:{key}")

    artifact = manifest.get("artifact") or {}
    if not _is_sha256(artifact.get("sha256")):
        problems.append(f"quick_output_manifest_artifact_sha256_invalid:{path.name}")
    try:
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader, [])
                row_count = sum(1 for _ in reader)
        elif path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            header = parquet.schema_arrow.names
            row_count = int(parquet.metadata.num_rows)
        else:
            raise ValueError("unsupported table format")
    except Exception as exc:
        problems.append(f"quick_output_table_unreadable:{path.name}:{type(exc).__name__}")
    else:
        try:
            recorded_size = int(artifact.get("size_bytes"))
        except (TypeError, ValueError):
            recorded_size = -1
        try:
            recorded_rows = int(artifact.get("rows"))
            recorded_columns = int(artifact.get("columns"))
        except (TypeError, ValueError):
            recorded_rows = recorded_columns = -1
        if recorded_size != path.stat().st_size:
            problems.append(f"quick_output_manifest_artifact_size_mismatch:{path.name}")
        if recorded_rows != row_count:
            problems.append(f"quick_output_manifest_artifact_rows_mismatch:{path.name}")
        if recorded_columns != len(header):
            problems.append(f"quick_output_manifest_artifact_columns_mismatch:{path.name}")
        if artifact.get("column_names") != header:
            problems.append(f"quick_output_manifest_artifact_column_names_mismatch:{path.name}")

    raw_inputs = manifest.get("inputs")
    observed_inputs: set[Path] = set()
    if not isinstance(raw_inputs, list) or not raw_inputs:
        problems.append(f"quick_output_manifest_inputs_missing:{path.name}")
    else:
        for index, record in enumerate(raw_inputs):
            if not isinstance(record, Mapping):
                problems.append(f"quick_output_manifest_input_invalid:{path.name}:{index}")
                continue
            try:
                input_path = _resolve_recorded_path(root, record.get("path"))
            except (OSError, ValueError) as exc:
                problems.append(f"quick_output_manifest_input_path_invalid:{path.name}:{index}:{exc}")
                continue
            if input_path in observed_inputs:
                problems.append(f"quick_output_manifest_input_duplicate:{path.name}:{_relative(root, input_path)}")
            observed_inputs.add(input_path)
            if not _is_sha256(record.get("sha256")):
                problems.append(f"quick_output_manifest_input_sha256_invalid:{path.name}:{_relative(root, input_path)}")
            if input_path.is_file():
                try:
                    recorded_size = int(record.get("size_bytes"))
                except (TypeError, ValueError):
                    recorded_size = -1
                if recorded_size != input_path.stat().st_size:
                    problems.append(f"quick_output_manifest_input_size_mismatch:{path.name}:{_relative(root, input_path)}")
        missing_inputs = sorted(
            _relative(root, input_path)
            for input_path in expected_inputs - observed_inputs
        )
        unexpected_inputs = sorted(
            _relative(root, input_path)
            for input_path in observed_inputs - expected_inputs
        )
        if missing_inputs:
            problems.append(f"quick_output_manifest_required_inputs_missing:{path.name}:{','.join(missing_inputs)}")
        if unexpected_inputs:
            problems.append(f"quick_output_manifest_unexpected_inputs:{path.name}:{','.join(unexpected_inputs)}")
    return problems


def _audit_phase4_quick_atlas(
    root: Path,
    path: Path,
    *,
    expected_run_id: str,
    expected_contract: Mapping[str, str],
    expected_inputs: frozenset[Path],
    best_pixels_path: Path,
) -> list[str]:
    """Validate quick pixel-lag Zarr structure and its cryptographic sidecar."""

    problems: list[str] = []
    sidecar_path = Path(f"{path}.manifest.json")
    manifest = _read_json(sidecar_path)
    if manifest is None:
        return ["missing_quick_atlas_manifest"]
    if manifest.get("schema_version") != "nino-brasil-phase4-semantic-output-v1":
        problems.append("quick_atlas_manifest_schema_mismatch")
    if str(manifest.get("run_id") or "") != expected_run_id:
        problems.append("quick_atlas_manifest_run_id_mismatch")
    contract = manifest.get("contract") or {}
    if not (
        str(contract.get("phase") or "").upper() == "F4"
        and contract.get("stage") == "F4C_QUICK"
        and contract.get("artifact_type") == "numeric_array"
        and contract.get("selection_contract") == "quick:key-predictor"
        and contract.get("predictor_names") == ["nino34_ssta"]
        and _safe_int(contract.get("field_permutations"), -1) >= 19
    ):
        problems.append("quick_atlas_manifest_contract_mismatch")
    for key, expected in expected_contract.items():
        if not expected or str(contract.get(key) or "") != expected:
            problems.append(f"quick_atlas_manifest_identity_mismatch:{key}")

    artifact = manifest.get("artifact") or {}
    try:
        artifact_path = _resolve_recorded_path(root, artifact.get("path"))
    except (OSError, ValueError) as exc:
        artifact_path = None
        problems.append(f"quick_atlas_manifest_path_invalid:{exc}")
    if artifact_path != path.resolve() or artifact.get("is_directory") is not True:
        problems.append("quick_atlas_manifest_artifact_mismatch")
    if not path.is_dir():
        return [*problems, "missing_quick_atlas"]
    expected_tree_hash = str(artifact.get("tree_sha256") or "")
    if not _is_sha256(expected_tree_hash) or _tree_sha256(path) != expected_tree_hash:
        problems.append("quick_atlas_tree_hash_mismatch")
    atlas_files = [
        item
        for item in path.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix.lower() not in {".pyc", ".pyo"}
    ]
    if _safe_int(artifact.get("n_files"), -1) != len(atlas_files):
        problems.append("quick_atlas_manifest_file_count_mismatch")

    raw_inputs = manifest.get("inputs")
    observed_inputs: set[Path] = set()
    if not isinstance(raw_inputs, list) or not raw_inputs:
        problems.append("quick_atlas_manifest_inputs_missing")
    else:
        for index, record in enumerate(raw_inputs):
            if not isinstance(record, Mapping):
                problems.append(f"quick_atlas_manifest_input_invalid:{index}")
                continue
            problems.extend(_verify_record(root, record, "quick_atlas_manifest_input"))
            try:
                input_path = _resolve_recorded_path(root, record.get("path"))
            except (OSError, ValueError):
                continue
            if input_path in observed_inputs:
                problems.append(f"quick_atlas_manifest_input_duplicate:{_relative(root, input_path)}")
            observed_inputs.add(input_path)
            if not _is_sha256(record.get("sha256")):
                problems.append(f"quick_atlas_manifest_input_sha256_invalid:{_relative(root, input_path)}")
    missing_inputs = sorted(
        _relative(root, input_path)
        for input_path in expected_inputs - observed_inputs
    )
    unexpected_inputs = sorted(
        _relative(root, input_path)
        for input_path in observed_inputs - expected_inputs
    )
    if missing_inputs:
        problems.append(f"quick_atlas_manifest_required_inputs_missing:{','.join(missing_inputs)}")
    if unexpected_inputs:
        problems.append(f"quick_atlas_manifest_unexpected_inputs:{','.join(unexpected_inputs)}")

    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        import xarray as xr

        dataset = xr.open_zarr(path, consolidated=False)
    except Exception as exc:
        return [*problems, f"quick_atlas_open_failed:{type(exc).__name__}"]
    try:
        dimensions = {name: int(size) for name, size in dataset.sizes.items()}
        arrays = {
            name: {
                "dims": list(array.dims),
                "shape": [int(value) for value in array.shape],
                "dtype": str(array.dtype),
                "chunks": (
                    [list(map(int, axis_chunks)) for axis_chunks in array.chunks]
                    if array.chunks is not None
                    else None
                ),
            }
            for name, array in dataset.data_vars.items()
        }
        if artifact.get("dimensions") != dimensions:
            problems.append("quick_atlas_manifest_dimensions_mismatch")
        if artifact.get("arrays") != arrays:
            problems.append("quick_atlas_manifest_arrays_mismatch")
        if set(dataset.data_vars) != F4C_ATLAS_VARIABLES:
            problems.append("quick_atlas_variable_catalog_mismatch")
        expected_sizes = {
            "variavel": 1,
            "condicao_fonte": len(ENSO_PHASE_CONDITIONS),
            "lag_sem": 40,
        }
        if any(int(dataset.sizes.get(name, 0)) != size for name, size in expected_sizes.items()):
            problems.append("quick_atlas_dimension_contract_mismatch")
        if int(dataset.sizes.get("pixel", 0)) <= 0:
            problems.append("quick_atlas_pixel_dimension_empty")
        if set(map(str, dataset["variavel"].values.tolist())) != {"nino34_ssta"}:
            problems.append("quick_atlas_predictor_catalog_mismatch")
        if set(map(str, dataset["condicao_fonte"].values.tolist())) != ENSO_PHASE_CONDITIONS:
            problems.append("quick_atlas_condition_catalog_mismatch")
        if tuple(int(value) for value in dataset["lag_sem"].values.tolist()) != tuple(range(0, 79, 2)):
            problems.append("quick_atlas_lag_catalog_mismatch")
        for key, expected in {"analysis_run_id": expected_run_id, **expected_contract}.items():
            if not expected or str(dataset.attrs.get(key) or "") != expected:
                problems.append(f"quick_atlas_attribute_mismatch:{key}")
        if dataset.attrs.get("spatial_contract") != "original CHIRPS pixels with Brazil overlap; no interpolation":
            problems.append("quick_atlas_spatial_contract_mismatch")
        if best_pixels_path.is_file():
            best_parquet = pq.ParquetFile(best_pixels_path)
            if "pixel_id" not in best_parquet.schema_arrow.names:
                problems.append("quick_best_pixels_missing_pixel_id")
            else:
                best_pixel_ids = {
                    int(value)
                    for value in pc.unique(best_parquet.read(columns=["pixel_id"])["pixel_id"]).to_pylist()
                    if value is not None
                }
                atlas_pixel_ids = {int(value) for value in dataset["pixel"].values.tolist()}
                if best_pixel_ids != atlas_pixel_ids:
                    problems.append("quick_atlas_best_pixels_inventory_mismatch")
    except Exception as exc:
        problems.append(f"quick_atlas_contract_read_failed:{type(exc).__name__}")
    finally:
        dataset.close()
    return problems


def collect_phase4_smoke(
    root: Path,
    chirps: Mapping[str, Any],
    phase3: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the complete real-data F4C quick contract validates."""

    quick_root = root / "data/processed/parquet/statistics/quick"
    required = [quick_root / name for name in F4C_QUICK_TABULAR_NAMES]
    sidecars = [Path(f"{path}.manifest.json") for path in required]
    atlas = root / F4C_QUICK_ATLAS
    atlas_sidecar = Path(f"{atlas}.manifest.json")
    all_expected = (*required, *sidecars, atlas, atlas_sidecar)
    evidence = [_relative(root, path) for path in all_expected]
    if not any(path.exists() for path in all_expected):
        return _status(
            "not_run",
            "Quick F4C não encontrado.",
            evidence,
            passed=False,
            run_id="",
            expected_outputs=len(required) + 1,
            valid_outputs=0,
            expected_csv_outputs=len(F4C_QUICK_CSV_NAMES),
            valid_csv_outputs=0,
            problems=[],
        )

    problems: list[str] = []
    for path in required:
        if not path.is_file():
            problems.append(f"missing_quick_output:{path.name}")
        if not Path(f"{path}.manifest.json").is_file():
            problems.append(f"missing_quick_output_manifest:{path.name}")
    if not atlas.is_dir():
        problems.append("missing_quick_atlas")
    if not atlas_sidecar.is_file():
        problems.append("missing_quick_atlas_manifest")

    values_by_name: dict[str, dict[str, set[str]]] = {}
    conditions_by_name: dict[str, set[str]] = {}
    quick_identity_columns = (*F4C_IDENTITY_COLUMNS, *F4C_PREDICTOR_CONTRACT_COLUMNS)
    for path in required:
        if not path.is_file():
            continue
        extra_columns: tuple[str, ...]
        if path.name in F4C_QUICK_CONDITION_TABLE_NAMES:
            extra_columns = ("condicao_fonte",)
        elif path.name == "phase4C_native_predictor_treatment_quick.csv":
            extra_columns = ("variavel",)
        else:
            extra_columns = ()
        values, count, table_problems = _table_column_sets(
            path,
            (*quick_identity_columns, *extra_columns),
        )
        problems.extend(table_problems)
        if count == 0:
            problems.append(f"empty_quick_output:{path.name}")
        values_by_name[path.name] = values
        for column in quick_identity_columns:
            if len(values[column]) != 1:
                problems.append(f"quick_identity_not_singleton:{path.name}:{column}:{len(values[column])}")
        if path.name in F4C_QUICK_CONDITION_TABLE_NAMES:
            conditions = values["condicao_fonte"]
            conditions_by_name[path.name] = conditions
            if conditions != ENSO_PHASE_CONDITIONS:
                missing = ",".join(sorted(ENSO_PHASE_CONDITIONS - conditions)) or "none"
                unexpected = ",".join(sorted(conditions - ENSO_PHASE_CONDITIONS)) or "none"
                problems.append(
                    f"quick_condition_catalog_mismatch:{path.name}:missing={missing}:unexpected={unexpected}"
                )
        elif path.name == "phase4C_native_predictor_treatment_quick.csv":
            if values["selection_contract"] != {"quick:key-predictor"}:
                problems.append("quick_selection_contract_mismatch")
            if values["variavel"] != {"nino34_ssta"}:
                problems.append("quick_predictor_catalog_mismatch")

    merged = {
        column: set().union(
            *(values[column] for values in values_by_name.values())
        )
        if values_by_name
        else set()
        for column in quick_identity_columns
    }
    for column, values in merged.items():
        if len(values) != 1:
            problems.append(f"quick_identity_catalog_mismatch:{column}:{len(values)}")
    run_id = next(iter(merged["analysis_run_id"]), "")
    if not run_id.startswith("F4C_QUICK_"):
        problems.append("quick_run_id_contract_mismatch")

    expected_f3_run_id = str(phase3.get("run_id") or "")
    expected_f3_hash = str(phase3.get("phase_table_sha256") or "")
    if phase3.get("state") != "complete" or phase3.get("complete") is not True:
        problems.append("quick_parent_f3_not_current_complete")
    if not expected_f3_run_id or merged["parent_f3_run_id"] != {expected_f3_run_id}:
        problems.append("quick_parent_f3_run_id_mismatch")
    if not _is_sha256(expected_f3_hash) or merged["parent_f3_artifact_sha256"] != {expected_f3_hash}:
        problems.append("quick_parent_f3_artifact_mismatch")

    target_build_id = str(chirps.get("build_id") or "")
    target_signature = str(chirps.get("block_signature_sha256") or "")
    target_grid = str(chirps.get("grid_hash_sha256") or "")
    if chirps.get("state") != "promoted" or chirps.get("promoted") is not True:
        problems.append("quick_chirps_not_current_promoted")
    geometry_contract, geometry_components, geometry_problems = _current_ibge_geometry_contract(root)
    problems.extend(geometry_problems)
    current_identity = {
        "parent_f3_run_id": expected_f3_run_id,
        "parent_f3_artifact_sha256": expected_f3_hash,
        "target_build_id": target_build_id,
        "target_block_signature_sha256": target_signature,
        "target_contract_version": CHIRPS_TARGET_CONTRACT,
        "grid_hash_sha256": target_grid,
        **geometry_contract,
        "selection_contract": "quick:key-predictor",
        "predictor_count": "1",
        "predictor_catalog_sha256": hashlib.sha256(b"nino34_ssta").hexdigest(),
        "f4c_runner_sha256": _sha256_file(root / "scripts/run_fase4c_regional.py")
        if (root / "scripts/run_fase4c_regional.py").is_file()
        else "",
        "lag_analysis_module_sha256": _sha256_file(root / "src/nino_brasil/stats/lag_analysis.py")
        if (root / "src/nino_brasil/stats/lag_analysis.py").is_file()
        else "",
    }
    if not _is_sha256(target_signature):
        problems.append("quick_chirps_signature_invalid")
    if target_grid != CHIRPS_FROZEN_GRID_HASH:
        problems.append("quick_chirps_grid_mismatch")
    for column, expected in current_identity.items():
        if not expected or merged[column] != {expected}:
            problems.append(f"quick_current_identity_mismatch:{column}")

    membership = root / "data/processed/parquet/statistics/phase4C_native_pixel_membership_exact.parquet"
    if not membership.is_file():
        problems.append("quick_membership_missing")
    else:
        membership_values, membership_rows, membership_problems = _table_column_sets(
            membership,
            ("grid_hash", "membership_status", *geometry_contract),
        )
        problems.extend(membership_problems)
        if membership_rows == 0:
            problems.append("quick_membership_empty")
        if membership_values["grid_hash"] != {target_grid}:
            problems.append("quick_membership_grid_mismatch")
        if membership_values["membership_status"] != {"canonical_equal_area_official_ibge"}:
            problems.append("quick_membership_method_mismatch")
        for column, expected in geometry_contract.items():
            if not expected or membership_values[column] != {expected}:
                problems.append(f"quick_membership_geometry_mismatch:{column}")

    expected_inputs = _phase4_quick_required_inputs(
        root,
        target_signature,
        geometry_components,
    )
    sidecar_contract = dict(current_identity)
    for path in required:
        if path.is_file():
            problems.extend(
                _audit_phase4_quick_sidecar(
                    root,
                    path,
                    expected_run_id=run_id,
                    expected_contract=sidecar_contract,
                    expected_inputs=expected_inputs,
                )
            )
    if atlas.is_dir():
        problems.extend(
            _audit_phase4_quick_atlas(
                root,
                atlas,
                expected_run_id=run_id,
                expected_contract=sidecar_contract,
                expected_inputs=expected_inputs,
                best_pixels_path=quick_root / "phase4C_native_best_lag_pixel_quick.parquet",
            )
        )

    problems = sorted(set(problems))
    passed = not problems
    return _status(
        "complete" if passed else "invalidated",
        "Quick F4C completo, íntegro e alinhado às identidades correntes."
        if passed
        else f"Quick F4C encontrado, mas invalidado por {len(problems)} problema(s) de contrato.",
        evidence,
        passed=passed,
        run_id=run_id,
        expected_outputs=len(required) + 1,
        valid_outputs=sum(path.is_file() for path in required) + int(atlas.is_dir()),
        expected_csv_outputs=len(F4C_QUICK_CSV_NAMES),
        valid_csv_outputs=sum((quick_root / name).is_file() for name in F4C_QUICK_CSV_NAMES),
        enso_phase_condition_count=len(
            set().union(*conditions_by_name.values()) if conditions_by_name else set()
        ),
        enso_phase_conditions=sorted(
            set().union(*conditions_by_name.values()) if conditions_by_name else set()
        ),
        expected_enso_phase_conditions=sorted(ENSO_PHASE_CONDITIONS),
        problems=problems,
    )


def _audit_official_f4c_sidecar(
    root: Path,
    path: Path,
    *,
    expected_run_id: str,
    expected_contract: Mapping[str, str],
    expected_inputs: frozenset[Path],
) -> list[str]:
    problems = _audit_semantic_csv_sidecar(
        root,
        path,
        expected_run_id,
        expected_contract={"stage": "F4C", **expected_contract},
        required_inputs=expected_inputs,
    )
    manifest = _read_json(Path(f"{path}.manifest.json"))
    if manifest is None:
        return problems
    contract = manifest.get("contract") or {}
    expected_catalog = list(PHASE2_PHYSICAL_COLUMNS)
    expected_catalog_hash = hashlib.sha256(
        "\n".join(expected_catalog).encode("utf-8")
    ).hexdigest()
    if contract.get("artifact_type") != "numeric_table":
        problems.append(f"f4c_manifest_artifact_type_mismatch:{path.name}")
    if contract.get("selection_contract") != "canonical_all_31_physical_variables":
        problems.append(f"f4c_manifest_selection_contract_mismatch:{path.name}")
    if _safe_int(contract.get("predictor_count"), -1) != len(expected_catalog):
        problems.append(f"f4c_manifest_predictor_count_mismatch:{path.name}")
    if contract.get("predictor_names") != expected_catalog:
        problems.append(f"f4c_manifest_predictor_catalog_mismatch:{path.name}")
    if contract.get("predictor_catalog_sha256") != expected_catalog_hash:
        problems.append(f"f4c_manifest_predictor_catalog_hash_mismatch:{path.name}")
    if _safe_int(contract.get("field_permutations"), -1) < 199:
        problems.append(f"f4c_manifest_field_permutations_below_official:{path.name}")
    return problems


def _audit_official_f4c_atlas_manifest(
    root: Path,
    path: Path,
    *,
    expected_run_id: str,
    expected_contract: Mapping[str, str],
    expected_inputs: frozenset[Path],
) -> list[str]:
    problems: list[str] = []
    manifest = _read_json(Path(f"{path}.manifest.json"))
    if manifest is None:
        return ["f4c_official_atlas_manifest_missing"]
    if manifest.get("schema_version") != "nino-brasil-phase4-semantic-output-v1":
        problems.append("f4c_official_atlas_manifest_schema_mismatch")
    if str(manifest.get("run_id") or "") != expected_run_id:
        problems.append("f4c_official_atlas_manifest_run_id_mismatch")
    contract = manifest.get("contract") or {}
    canonical_catalog = list(PHASE2_PHYSICAL_COLUMNS)
    canonical_hash = hashlib.sha256("\n".join(canonical_catalog).encode("utf-8")).hexdigest()
    if not (
        str(contract.get("phase") or "").upper() == "F4"
        and contract.get("stage") == "F4C"
        and contract.get("artifact_type") == "numeric_array"
        and contract.get("selection_contract") == "canonical_all_31_physical_variables"
        and _safe_int(contract.get("predictor_count"), -1) == len(canonical_catalog)
        and contract.get("predictor_names") == canonical_catalog
        and contract.get("predictor_catalog_sha256") == canonical_hash
        and _safe_int(contract.get("field_permutations"), -1) >= 199
    ):
        problems.append("f4c_official_atlas_contract_mismatch")
    for key, expected in expected_contract.items():
        if not expected or str(contract.get(key) or "") != expected:
            problems.append(f"f4c_official_atlas_identity_mismatch:{key}")
    artifact = manifest.get("artifact") or {}
    try:
        artifact_path = _resolve_recorded_path(root, artifact.get("path"))
    except (OSError, ValueError) as exc:
        artifact_path = None
        problems.append(f"f4c_official_atlas_path_invalid:{exc}")
    if artifact_path != path.resolve() or artifact.get("is_directory") is not True:
        problems.append("f4c_official_atlas_artifact_mismatch")
    if path.is_dir():
        tree_hash = str(artifact.get("tree_sha256") or "")
        if not _is_sha256(tree_hash) or _tree_sha256(path) != tree_hash:
            problems.append("f4c_official_atlas_tree_hash_mismatch")
        dimensions = artifact.get("dimensions") or {}
        pixel_count = _safe_int(dimensions.get("pixel"), 0)
        expected_dimensions = {
            "variavel": len(canonical_catalog),
            "condicao_fonte": len(ENSO_PHASE_CONDITIONS),
            "lag_sem": 40,
            "pixel": pixel_count,
        }
        if pixel_count <= 0 or dimensions != expected_dimensions:
            problems.append("f4c_official_atlas_dimensions_mismatch")
        arrays = artifact.get("arrays") or {}
        if set(arrays) != F4C_ATLAS_VARIABLES:
            problems.append("f4c_official_atlas_arrays_mismatch")
        else:
            expected_shape = [
                len(canonical_catalog),
                len(ENSO_PHASE_CONDITIONS),
                40,
                pixel_count,
            ]
            for name, record in arrays.items():
                if not isinstance(record, Mapping) or record.get("dims") != [
                    "variavel",
                    "condicao_fonte",
                    "lag_sem",
                    "pixel",
                ] or record.get("shape") != expected_shape:
                    problems.append(f"f4c_official_atlas_array_contract_mismatch:{name}")
    raw_inputs = manifest.get("inputs")
    observed_inputs: set[Path] = set()
    if not isinstance(raw_inputs, list) or not raw_inputs:
        problems.append("f4c_official_atlas_inputs_missing")
    else:
        for record in raw_inputs:
            if not isinstance(record, Mapping):
                problems.append("f4c_official_atlas_input_invalid")
                continue
            problems.extend(_verify_record(root, record, "f4c_official_atlas_input"))
            try:
                input_path = _resolve_recorded_path(root, record.get("path"))
            except (OSError, ValueError):
                continue
            observed_inputs.add(input_path)
            fingerprint = record.get("tree_sha256") if input_path.is_dir() else record.get("sha256")
            if input_path in expected_inputs and not _is_sha256(fingerprint):
                problems.append(f"f4c_official_atlas_input_sha256_invalid:{_relative(root, input_path)}")
    missing_inputs = sorted(
        _relative(root, input_path)
        for input_path in expected_inputs - observed_inputs
    )
    if missing_inputs:
        problems.append(f"f4c_official_atlas_required_inputs_missing:{','.join(missing_inputs)}")
    return problems


def collect_phase4_outputs(root: Path, chirps: Mapping[str, Any], phase3: Mapping[str, Any]) -> dict[str, Any]:
    stats = root / "data/processed/parquet/statistics"
    f4c_names = (
        "phase4C_native_lags_por_unidade.csv",
        "phase4C_native_predictor_treatment.csv",
        "phase4C_native_best_lag_pixel_key.csv",
        "phase4C_native_field_significance.csv",
    )
    f4d_names = (
        "phase4D_native_gate_event_jackknife.csv",
        "phase4D_native_hypothesis_summary.csv",
        "phase4D_native_target_coverage.csv",
    )
    required = [stats / name for name in (*f4c_names, *f4d_names)]
    missing = [_relative(root, path) for path in required if not path.is_file()]
    if missing:
        return {
            "official": _status("blocked" if chirps.get("state") != "promoted" else "not_run", f"Faltam {len(missing)} saídas canônicas 4C/4D.", missing),
            "gate": _status("pending", "Gate F4D ainda não materializado de modo canônico."),
            "run_id": "",
        }

    identity_columns = F4C_IDENTITY_COLUMNS
    f4c_identity_columns = (*identity_columns, *F4C_PREDICTOR_CONTRACT_COLUMNS)
    problems: list[str] = []
    f4c_values: list[dict[str, set[str]]] = []
    f4c_conditions: set[str] = set()
    for path in required[:4]:
        extras = ("variavel",) if path.name == "phase4C_native_predictor_treatment.csv" else ()
        if path.name != "phase4C_native_predictor_treatment.csv":
            extras = (*extras, "condicao_fonte")
        values, count, table_problems = _csv_column_sets(
            path,
            (*f4c_identity_columns, *extras),
        )
        problems.extend(table_problems)
        if count == 0:
            problems.append(f"empty:{path.name}")
        f4c_values.append(values)
        if path.name == "phase4C_native_predictor_treatment.csv":
            if values["variavel"] != set(PHASE2_PHYSICAL_COLUMNS):
                problems.append("f4c_predictor_table_catalog_mismatch")
        else:
            conditions = values["condicao_fonte"]
            f4c_conditions.update(conditions)
            if conditions != ENSO_PHASE_CONDITIONS:
                missing_conditions = ",".join(sorted(ENSO_PHASE_CONDITIONS - conditions)) or "none"
                unexpected_conditions = ",".join(sorted(conditions - ENSO_PHASE_CONDITIONS)) or "none"
                problems.append(
                    f"f4c_condition_catalog_mismatch:{path.name}:missing={missing_conditions}:unexpected={unexpected_conditions}"
                )
    merged_f4c = {
        column: set().union(*(values[column] for values in f4c_values))
        for column in f4c_identity_columns
    }
    for column, values in merged_f4c.items():
        if len(values) != 1:
            problems.append(f"f4c_identity:{column}:{len(values)}")
    f4c_run_id = next(iter(merged_f4c["analysis_run_id"]), "")
    canonical_predictor_hash = hashlib.sha256(
        "\n".join(PHASE2_PHYSICAL_COLUMNS).encode("utf-8")
    ).hexdigest()
    canonical_predictor_identity = {
        "selection_contract": "canonical_all_31_physical_variables",
        "predictor_count": str(len(PHASE2_PHYSICAL_COLUMNS)),
        "predictor_catalog_sha256": canonical_predictor_hash,
    }
    for column, expected in canonical_predictor_identity.items():
        if merged_f4c[column] != {expected}:
            problems.append(f"f4c_predictor_contract_mismatch:{column}")
    if f4c_conditions != ENSO_PHASE_CONDITIONS:
        missing_conditions = ",".join(sorted(ENSO_PHASE_CONDITIONS - f4c_conditions)) or "none"
        unexpected_conditions = ",".join(sorted(f4c_conditions - ENSO_PHASE_CONDITIONS)) or "none"
        problems.append(f"f4c_condition_catalog_mismatch:missing={missing_conditions}:unexpected={unexpected_conditions}")

    f4d_columns = (*identity_columns, "parent_f4c_run_id", "f4d_runner_sha256")
    f4d_values: list[dict[str, set[str]]] = []
    f4d_conditions: set[str] = set()
    for path in required[4:]:
        row_identity_columns = (
            tuple(column for column in f4d_columns if column != "grid_hash_sha256")
            if path.name == "phase4D_native_target_coverage.csv"
            else f4d_columns
        )
        values, count, table_problems = _csv_column_sets(path, row_identity_columns)
        if path.name == "phase4D_native_target_coverage.csv":
            # Coverage is a long diagnostic table.  Its grid identity lives in
            # the hashed semantic sidecar to avoid repeating a 64-byte value
            # on every weekly target-region row.
            sidecar = _read_json(Path(f"{path}.manifest.json")) or {}
            contract = sidecar.get("contract") or {}
            grid_hash = str(contract.get("grid_hash_sha256") or "").strip()
            values["grid_hash_sha256"] = {grid_hash} if grid_hash else set()
        problems.extend(table_problems)
        if count == 0:
            problems.append(f"empty:{path.name}")
        f4d_values.append(values)
        if path.name != "phase4D_native_target_coverage.csv":
            table_conditions: set[str] = set()
            for row in _read_csv(path):
                event_type = str(row.get("tipo_enso_fonte") or "").strip()
                source_phase = str(row.get("fase_fonte_em_t_menos_lag") or "").strip()
                if event_type and source_phase:
                    table_conditions.add(f"{event_type}_{source_phase}")
            f4d_conditions.update(table_conditions)
            if table_conditions != ENSO_PHASE_CONDITIONS:
                missing_conditions = ",".join(sorted(ENSO_PHASE_CONDITIONS - table_conditions)) or "none"
                unexpected_conditions = ",".join(sorted(table_conditions - ENSO_PHASE_CONDITIONS)) or "none"
                problems.append(
                    f"f4d_condition_catalog_mismatch:{path.name}:missing={missing_conditions}:unexpected={unexpected_conditions}"
                )
    merged_f4d = {column: set().union(*(values[column] for values in f4d_values)) for column in f4d_columns}
    for column, values in merged_f4d.items():
        if len(values) != 1:
            problems.append(f"f4d_identity:{column}:{len(values)}")
    f4d_run_id = next(iter(merged_f4d["analysis_run_id"]), "")
    if f4d_conditions != ENSO_PHASE_CONDITIONS:
        missing_conditions = ",".join(sorted(ENSO_PHASE_CONDITIONS - f4d_conditions)) or "none"
        unexpected_conditions = ",".join(sorted(f4d_conditions - ENSO_PHASE_CONDITIONS)) or "none"
        problems.append(
            f"f4d_condition_catalog_mismatch:missing={missing_conditions}:unexpected={unexpected_conditions}"
        )
    if merged_f4d["parent_f4c_run_id"] != {f4c_run_id}:
        problems.append("f4d_parent_f4c_mismatch")
    for column in identity_columns[1:]:
        if merged_f4c[column] != merged_f4d[column]:
            problems.append(f"f4c_f4d_identity_mismatch:{column}")
    expected_f3 = str(phase3.get("run_id") or "")
    if phase3.get("state") != "complete" or phase3.get("complete") is not True:
        problems.append("f4_parent_f3_not_current_complete")
    if not expected_f3 or (merged_f4c["parent_f3_run_id"] != {expected_f3} or merged_f4d["parent_f3_run_id"] != {expected_f3}):
        problems.append("f4_parent_f3_mismatch")
    expected_f3_hash = str(phase3.get("phase_table_sha256") or "")
    if not expected_f3_hash or (
        merged_f4c["parent_f3_artifact_sha256"] != {expected_f3_hash}
        or merged_f4d["parent_f3_artifact_sha256"] != {expected_f3_hash}
    ):
        problems.append("f4_parent_f3_artifact_mismatch")
    chirps_expectations = {
        "target_build_id": str(chirps.get("build_id") or ""),
        "target_block_signature_sha256": str(chirps.get("block_signature_sha256") or ""),
        "target_contract_version": CHIRPS_TARGET_CONTRACT,
        "grid_hash_sha256": str(chirps.get("grid_hash_sha256") or ""),
    }
    if chirps.get("state") != "promoted" or chirps.get("promoted") is not True:
        problems.append("chirps_not_promoted")
    else:
        for column, expected in chirps_expectations.items():
            if not expected or merged_f4c[column] != {expected} or merged_f4d[column] != {expected}:
                problems.append(f"f4_chirps_identity_mismatch:{column}")
    geometry_contract, geometry_components, geometry_problems = _current_ibge_geometry_contract(root)
    problems.extend(geometry_problems)
    for column, expected in geometry_contract.items():
        if not expected or merged_f4c[column] != {expected} or merged_f4d[column] != {expected}:
            problems.append(f"f4_ibge_geometry_identity_mismatch:{column}")
    membership = stats / "phase4C_native_pixel_membership_exact.parquet"
    if not membership.is_file():
        problems.append("f4_membership_missing")
    else:
        membership_values, membership_rows, membership_problems = _table_column_sets(
            membership,
            ("grid_hash", "membership_status", *geometry_contract),
        )
        problems.extend(membership_problems)
        if membership_rows == 0:
            problems.append("f4_membership_empty")
        if membership_values["grid_hash"] != {CHIRPS_FROZEN_GRID_HASH}:
            problems.append("f4_membership_grid_mismatch")
        if membership_values["membership_status"] != {"canonical_equal_area_official_ibge"}:
            problems.append("f4_membership_method_mismatch")
        for column, expected in geometry_contract.items():
            if not expected or membership_values[column] != {expected}:
                problems.append(f"f4_membership_geometry_mismatch:{column}")
    code_expectations = {
        "f4c_runner_sha256": root / "scripts/run_fase4c_regional.py",
        "f4d_runner_sha256": root / "scripts/run_fase4d_targets.py",
        "lag_analysis_module_sha256": root / "src/nino_brasil/stats/lag_analysis.py",
    }
    current_code_hashes: dict[str, str] = {}
    for column, path in code_expectations.items():
        expected = _sha256_file(path) if path.is_file() else ""
        current_code_hashes[column] = expected
        f4d_match = merged_f4d.get(column, set()) == {expected}
        f4c_match = column == "f4d_runner_sha256" or merged_f4c.get(column, set()) == {expected}
        if not expected or not f4d_match or not f4c_match:
            problems.append(f"f4_code_fingerprint_mismatch:{column}")

    official_atlas = root / "data/processed/zarr/statistics/phase4C_native_pixel_lags.zarr"
    official_atlas_manifest = Path(f"{official_atlas}.manifest.json")
    if not official_atlas.is_dir():
        problems.append("f4c_official_atlas_missing")
    if not official_atlas_manifest.is_file():
        problems.append("f4c_official_atlas_manifest_missing")
    f4c_required_inputs = frozenset({membership.resolve(), *(path.resolve() for path in geometry_components)})
    f4d_required_inputs = frozenset(
        {
            *f4c_required_inputs,
            official_atlas.resolve(),
            official_atlas_manifest.resolve(),
        }
    )
    common_contract = {
        "parent_f3_run_id": expected_f3,
        "parent_f3_artifact_sha256": expected_f3_hash,
        **chirps_expectations,
        **geometry_contract,
        "f4c_runner_sha256": current_code_hashes.get("f4c_runner_sha256", ""),
        "lag_analysis_module_sha256": current_code_hashes.get("lag_analysis_module_sha256", ""),
    }
    f4c_contract = {**common_contract, **canonical_predictor_identity}
    f4d_contract = {
        **common_contract,
        "stage": "F4D",
        "artifact_type": "numeric_table",
        "parent_f4c_run_id": f4c_run_id,
        "f4d_runner_sha256": current_code_hashes.get("f4d_runner_sha256", ""),
    }
    for path in required[:4]:
        problems.extend(
            _audit_official_f4c_sidecar(
                root,
                path,
                expected_run_id=f4c_run_id,
                expected_contract=f4c_contract,
                expected_inputs=f4c_required_inputs,
            )
        )
    if official_atlas.is_dir():
        problems.extend(
            _audit_official_f4c_atlas_manifest(
                root,
                official_atlas,
                expected_run_id=f4c_run_id,
                expected_contract=f4c_contract,
                expected_inputs=f4c_required_inputs,
            )
        )
    for path in required[4:]:
        problems.extend(
            _audit_semantic_csv_sidecar(
                root,
                path,
                f4d_run_id,
                expected_contract=f4d_contract,
                required_inputs=f4d_required_inputs,
            )
        )

    gate_rows = _read_csv(stats / "phase4D_native_gate_event_jackknife.csv")
    gate_keys = [
        (
            row.get("regiao"),
            row.get("tipo_enso_fonte"),
            row.get("fase_fonte_em_t_menos_lag"),
            row.get("target_chirps"),
        )
        for row in gate_rows
    ]
    if len(gate_rows) != 200 or len(set(gate_keys)) != 200:
        problems.append(f"f4d_gate_row_contract_mismatch:expected=200:actual={len(gate_rows)}")
    if {str(row.get("regiao") or "") for row in gate_rows} != F4D_REGIONS:
        problems.append("f4d_gate_region_catalog_mismatch")
    if {str(row.get("target_chirps") or "") for row in gate_rows} != F4D_CONFIRMATORY_TARGETS:
        problems.append("f4d_gate_target_catalog_mismatch")

    summary_rows = _read_csv(stats / "phase4D_native_hypothesis_summary.csv")
    summary_keys = [
        (
            row.get("regiao"),
            row.get("tipo_enso_fonte"),
            row.get("fase_fonte_em_t_menos_lag"),
        )
        for row in summary_rows
    ]
    if len(summary_rows) != 16 or len(set(summary_keys)) != 16:
        problems.append(f"f4d_summary_row_contract_mismatch:expected=16:actual={len(summary_rows)}")
    if {str(row.get("regiao") or "") for row in summary_rows} != {"Nordeste", "Sul"}:
        problems.append("f4d_summary_region_catalog_mismatch")
    if not all(
        _safe_int(row.get("n_targets"), -1) == len(F4D_CONFIRMATORY_TARGETS)
        and _safe_int(row.get("n_target_families"), -1) == 4
        for row in summary_rows
    ):
        problems.append("f4d_summary_target_family_contract_mismatch")

    coverage_rows = _read_csv(stats / "phase4D_native_target_coverage.csv")
    if not coverage_rows:
        problems.append("f4_target_coverage_empty")
    if {str(row.get("target_chirps") or "") for row in coverage_rows} != F4D_CONFIRMATORY_TARGETS:
        problems.append("f4_target_coverage_catalog_mismatch")
    if len({str(row.get("id_unidade") or "") for row in coverage_rows}) != len(F4D_REGIONS):
        problems.append("f4_target_coverage_region_count_mismatch")
    coverage_keys = [
        (row.get("target_chirps"), row.get("week_ending_sunday"), row.get("id_unidade"))
        for row in coverage_rows
    ]
    if len(coverage_keys) != len(set(coverage_keys)):
        problems.append("f4_target_coverage_primary_key_duplicate")
    supports = sum(str(row.get("hypothesis_gate") or "").startswith("supports_") for row in summary_rows)
    coverage_pass_count = sum(
        _truthy(row.get("coverage_gate_pass")) for row in coverage_rows
    )
    coverage_failed_count = len(coverage_rows) - coverage_pass_count
    coverage_pass_fraction = (
        coverage_pass_count / len(coverage_rows) if coverage_rows else 0.0
    )
    gate_passed = bool(supports) and f4d_conditions == ENSO_PHASE_CONDITIONS and not problems
    gate = _status(
        "passed" if gate_passed else "failed",
        f"{supports}/{len(summary_rows)} condições agregadas sustentam ao menos três famílias complementares.",
        [_relative(root, stats / "phase4D_native_hypothesis_summary.csv")],
        passed=gate_passed,
        supporting_conditions=supports,
        total_conditions=len(summary_rows),
        enso_phase_condition_count=len(f4d_conditions),
        enso_phase_conditions=sorted(f4d_conditions),
        expected_enso_phase_conditions=sorted(ENSO_PHASE_CONDITIONS),
    )
    official = _status(
        "complete" if not problems else "invalidated",
        (
            "Saídas F4C/F4D canônicas e coerentes; falhas semanais de cobertura "
            "permanecem explicitamente registradas e são excluídas por valor ausente."
            if not problems and coverage_failed_count
            else "Saídas F4C/F4D canônicas e coerentes."
            if not problems
            else f"Saídas F4C/F4D têm {len(problems)} problema(s) de contrato."
        ),
        [_relative(root, path) for path in required],
        problems=problems,
        coverage_rows=len(coverage_rows),
        coverage_pass_rows=coverage_pass_count,
        coverage_failed_rows=coverage_failed_count,
        coverage_pass_fraction=coverage_pass_fraction,
    )
    return {
        "official": official,
        "gate": gate,
        "run_id": f4d_run_id,
        "parent_f4c_run_id": f4c_run_id,
        "f4c_conditions": sorted(f4c_conditions),
        "f4d_conditions": sorted(f4d_conditions),
    }


def _phase1(root: Path, spec: PhaseSpec) -> dict[str, Any]:
    path = root / "data/audit/ledger_audit_report.json"
    ledger_path = root / "data/audit/ledger.jsonl"
    report = _read_json(path)
    if report is None:
        data = _status("missing", "Auditoria estruturada do ledger ausente.", [_relative(root, path)])
        gate = _status("pending", "Sem auditoria do ledger não há gate operacional F1.")
        promotion = _status("blocked", "F1 ainda não é auditável no espelho.")
    else:
        critical = int(report.get("record_checksum_issue_count") or 0) + len(report.get("duplicate_event_ids") or [])
        warnings = int(report.get("malformed_lines") or 0) + int(report.get("incomplete_task_count") or 0)
        data = _status(
            "complete_with_warnings" if warnings else "complete",
            f"{report.get('valid_events', 0)} eventos válidos; {warnings} pendência(s) histórica(s); {critical} falha(s) crítica(s).",
            [_relative(root, path), "data/audit/ledger.jsonl"],
            valid_events=report.get("valid_events"),
            malformed_lines=report.get("malformed_lines"),
            incomplete_tasks=report.get("incomplete_task_count"),
        )
        passed = bool(report.get("original_preserved")) and critical == 0
        gate = _status("passed_with_warnings" if passed and warnings else ("passed" if passed else "failed"), "Ledger original preservado e sem colisões/checksums críticos." if passed else "Auditoria F1 encontrou falha crítica.", [_relative(root, path)], passed=passed)
        promotion = _status("operational" if passed else "blocked", "Base de ingestão auditável; pendências históricas permanecem explícitas." if passed else "Corrigir falhas críticas antes de uso operacional.")
    return {
        "phase": "F1",
        "title": spec.title,
        "implemented": _code_status(root, spec),
        "data": data,
        "smoke": _status("not_applicable", "F1 não usa modalidade smoke de modelagem."),
        "official": _status("not_applicable", "F1 é uma camada de ingestão, não um ArtifactRun oficial."),
        "gate": gate,
        "promotion": promotion,
        "run_id": "",
        "identity": {
            "kind": "ledger_audit",
            "artifacts": [
                {
                    "path": _relative(root, artifact),
                    "sha256": _sha256_file(artifact),
                }
                for artifact in (path, ledger_path)
                if artifact.is_file()
            ],
        },
        "audit_surfaces": {
            "numeric_tables": _status(
                "not_applicable",
                "F1 usa ledger e relatório estruturado; não produz tabela de insight.",
                [_relative(root, path), _relative(root, ledger_path)],
            ),
            "figures": _status("not_applicable", "F1 não possui figuras científicas."),
            "notebooks": _status("not_applicable", "F1 não depende de notebook para promoção."),
        },
        "next_action": spec.next_action,
    }


def _phase2(root: Path, spec: PhaseSpec) -> dict[str, Any]:
    manifest_path = root / "data/processed/parquet/statistics/phase2_master_run_manifest.json"
    validation_path = root / "data/processed/parquet/statistics/phase2_master_validation.csv"
    manifest = _read_json(manifest_path)
    validation_rows = _read_csv(validation_path)
    problems: list[str] = []
    if manifest is None:
        problems.append("missing_manifest")
    else:
        if str(manifest.get("schema_version") or "") != "phase2-master-manifest/1.0":
            problems.append("phase2_schema_version_mismatch")
        if not str(manifest.get("run_id") or "").strip():
            problems.append("phase2_run_id_missing")
        started = _parse_time(manifest.get("started_at_utc"))
        completed = _parse_time(manifest.get("completed_at_utc"))
        if started == datetime.min.replace(tzinfo=timezone.utc) or completed < started:
            problems.append("phase2_timestamps_invalid")
        contract = manifest.get("contract") or {}
        if int(contract.get("physical_variable_count") or 0) != 31:
            problems.append("physical_variable_count_not_31")
        if tuple(contract.get("physical_columns") or ()) != PHASE2_PHYSICAL_COLUMNS:
            problems.append("physical_columns_catalog_mismatch")
        if contract.get("metadata_columns") != ["ocean_source_code"]:
            problems.append("metadata_columns_contract_mismatch")
        options = manifest.get("options") or {}
        if not (
            options.get("strict") is True
            and options.get("ocean_only") is False
            and options.get("skip_ctd") is False
        ):
            problems.append("phase2_canonical_options_mismatch")
        for section, expected_paths in PHASE2_REQUIRED_PATHS.items():
            section_problems, _ = _audit_required_file_records(
                root,
                manifest,
                section,
                expected_paths,
            )
            problems.extend(section_problems)

        raw_shape = manifest.get("raw_shape")
        adjusted_shape = manifest.get("source_adjusted_shape")
        try:
            raw_rows = int(raw_shape[0]) if isinstance(raw_shape, list) and len(raw_shape) == 2 else -1
            raw_columns = int(raw_shape[1]) if isinstance(raw_shape, list) and len(raw_shape) == 2 else -1
        except (TypeError, ValueError):
            raw_rows = raw_columns = -1
        if not (
            isinstance(raw_shape, list)
            and len(raw_shape) == 2
            and raw_rows > 0
            and raw_columns == 32
        ):
            problems.append("phase2_raw_shape_invalid")
        if not (
            isinstance(adjusted_shape, list)
            and len(adjusted_shape) == 2
            and adjusted_shape == raw_shape
        ):
            problems.append("phase2_adjusted_shape_mismatch")

        master_path = root / "data/processed/parquet/features/nino34_master_weekly.csv"
        if master_path.is_file():
            try:
                with master_path.open("r", encoding="utf-8-sig", newline="") as stream:
                    reader = csv.reader(stream)
                    header = next(reader, [])
                    row_count = sum(1 for _ in reader)
                expected_header = [
                    "week_ending_sunday",
                    *PHASE2_PHYSICAL_COLUMNS,
                    "ocean_source_code",
                ]
                if header != expected_header:
                    problems.append("phase2_master_header_mismatch")
                expected_rows = raw_rows
                if row_count <= 0:
                    problems.append("phase2_master_empty")
                if row_count != expected_rows:
                    problems.append(
                        f"phase2_master_row_count_mismatch:expected={expected_rows}:actual={row_count}"
                    )
            except (OSError, UnicodeDecodeError, csv.Error) as exc:
                problems.append(f"phase2_master_unreadable:{type(exc).__name__}")
    required_validation_columns = {"checagem", "passou", "severidade", "detalhe"}
    if not validation_rows or not required_validation_columns.issubset(validation_rows[0]):
        problems.append("missing_validation")
    else:
        check_names = [str(row.get("checagem") or "").strip() for row in validation_rows]
        duplicate_checks = sorted({name for name in check_names if check_names.count(name) > 1})
        observed_checks = set(check_names)
        expected_checks = set(PHASE2_VALIDATION_CHECKS)
        if len(validation_rows) != len(PHASE2_VALIDATION_CHECKS):
            problems.append(
                f"validation_count_mismatch:expected={len(PHASE2_VALIDATION_CHECKS)}:actual={len(validation_rows)}"
            )
        if duplicate_checks:
            problems.append(f"duplicate_validation_checks:{','.join(duplicate_checks)}")
        if observed_checks != expected_checks:
            missing = ",".join(sorted(expected_checks - observed_checks)) or "none"
            unexpected = ",".join(sorted(observed_checks - expected_checks)) or "none"
            problems.append(f"validation_catalog_mismatch:missing={missing}:unexpected={unexpected}")
        if not all(_truthy(row.get("passou")) for row in validation_rows):
            problems.append("validation_failed")
        if any(not str(row.get("detalhe") or "").strip() for row in validation_rows):
            problems.append("validation_detail_missing")
    passed = not problems
    notebooks = collect_notebook_contract(root, 2)
    promoted = passed and notebooks.get("state") == "passed"
    data = _status(
        "complete" if passed else ("invalidated" if manifest else "missing"),
        f"{len(validation_rows)} validações; contrato de 31 variáveis {'íntegro' if passed else 'não íntegro'}.",
        [_relative(root, manifest_path), _relative(root, validation_path)],
        problems=problems,
    )
    gate = _status("passed" if passed else "failed", "Validação F2 aprovada." if passed else "Validação/hash F2 falhou.", data["evidence"], passed=passed)
    return {
        "phase": "F2",
        "title": spec.title,
        "implemented": _code_status(root, spec),
        "data": data,
        "smoke": _status("not_applicable", "F2 não usa modalidade smoke de modelagem."),
        "official": _status("complete" if passed else "invalidated", "Master canônico materializado." if passed else "Master não é elegível."),
        "gate": gate,
        "promotion": _status(
            "promoted" if promoted else ("audit_ready" if passed else "blocked"),
            "F2 promovida por contrato, hashes e notebook executado."
            if promoted
            else ("Master F2 íntegro; falta fechar o notebook de sanidade." if passed else "F2 bloqueada pela validação."),
            [*data["evidence"], *notebooks.get("evidence", [])],
        ),
        "run_id": str((manifest or {}).get("run_id") or ""),
        "identity": {
            "kind": "phase2_master_manifest",
            "run_id": str((manifest or {}).get("run_id") or ""),
            "manifest_path": _relative(root, manifest_path),
            "manifest_sha256": (
                _sha256_file(manifest_path) if manifest_path.is_file() else ""
            ),
        },
        "audit_surfaces": {
            "numeric_tables": _status(
                "passed" if passed else "failed",
                f"{len(validation_rows)} checks na tabela canônica F2.",
                [_relative(root, validation_path)],
                passed=passed,
                table_count=1 if validation_path.is_file() else 0,
                table_sha256=(
                    _sha256_file(validation_path) if validation_path.is_file() else ""
                ),
            ),
            "figures": _status(
                "not_applicable",
                "F2 é promovida pelo contrato numérico e notebook de sanidade.",
            ),
            "notebooks": notebooks,
        },
        "next_action": spec.next_action,
    }


def _phase3(root: Path, spec: PhaseSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    semantic = collect_phase3_semantic_status(root)
    run_id = str(semantic.get("run_id") or "")
    notebooks = collect_notebook_contract(root, 3)
    figures = collect_figure_lineage(root, 3, [run_id] if run_id else None)
    official_complete = bool(semantic.get("complete"))
    promoted = official_complete and notebooks.get("state") == "passed" and figures.get("state") == "passed"
    phase = {
        "phase": "F3",
        "title": spec.title,
        "implemented": _code_status(root, spec),
        "data": semantic,
        "smoke": _status("not_applicable", "A verificação quick F3 não substitui o núcleo semântico oficial."),
        "official": _status("complete" if official_complete else "partial", "Núcleo semântico oficial íntegro." if official_complete else "Núcleo semântico incompleto ou divergente.", semantic["evidence"]),
        "gate": _status("passed" if official_complete else "failed", "Gate de tabelas semânticas F3 aprovado." if official_complete else "Gate semântico F3 falhou.", semantic["evidence"], passed=official_complete),
        "promotion": _status("promoted" if promoted else "blocked", "Tabelas, notebooks e figuras F3 compartilham linhagem válida." if promoted else "Promoção aguarda tabelas, notebooks e figuras semanticamente coerentes.", [*notebooks.get("evidence", []), *figures.get("evidence", [])]),
        "run_id": run_id,
        "identity": {
            "kind": "phase3_semantic_catalog",
            "run_id": run_id,
            "table_catalog_sha256": semantic.get("table_catalog_sha256"),
            "phase_table_sha256": semantic.get("phase_table_sha256"),
        },
        "audit_surfaces": {
            "numeric_tables": _status(
                "passed" if official_complete and figures.get("state") == "passed" else "failed",
                (
                    f"{semantic.get('valid_tables', 0)} tabelas semânticas e "
                    f"{figures.get('numeric_table_count', 0)} tabelas-fonte de figuras."
                ),
                semantic.get("evidence", []),
                passed=official_complete and figures.get("state") == "passed",
                semantic_table_count=semantic.get("valid_tables", 0),
                figure_numeric_table_count=figures.get("numeric_table_count", 0),
                semantic_catalog_sha256=semantic.get("table_catalog_sha256"),
                figure_numeric_catalog_sha256=figures.get(
                    "numeric_table_catalog_sha256"
                ),
            ),
            "figures": figures,
            "notebooks": notebooks,
        },
        "next_action": spec.next_action,
    }
    return phase, {"semantic": semantic, "notebooks": notebooks, "figures": figures}


def _phase4(root: Path, spec: PhaseSpec, chirps: dict[str, Any], phase3: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    outputs = collect_phase4_outputs(root, chirps, phase3)
    smoke = collect_phase4_smoke(root, chirps, phase3)
    expected_figure_runs = [
        value
        for value in (outputs.get("parent_f4c_run_id"), outputs.get("run_id"))
        if value
    ]
    figures = collect_figure_lineage(
        root,
        4,
        expected_figure_runs if expected_figure_runs else None,
    )
    notebooks = collect_notebook_contract(root, 4)
    official = outputs["official"]
    gate = outputs["gate"]
    promoted = (
        official["state"] == "complete"
        and gate["state"] == "passed"
        and figures["state"] == "passed"
        and notebooks["state"] == "passed"
    )
    promotion_state = "promoted" if promoted else (
        "audit_ready" if official["state"] == "complete" else "blocked"
    )
    promotion_detail = (
        "F4C/F4D, gate, notebooks e figuras canônicas são auditáveis."
        if promoted
        else "Saídas oficiais executadas e auditáveis, porém sem promoção científica enquanto gate/notebooks/figuras não passarem."
        if promotion_state == "audit_ready"
        else "Promoção F4 aguarda alvo e saídas oficiais coerentes."
    )
    next_action = (
        "Preservar o resultado confirmatório negativo; não executar F6 nem repetir F4 sem nova hipótese predefinida."
        if official["state"] == "complete" and gate["state"] == "failed"
        else "Não repetir sem mudança documentada de dados, código ou hipótese."
        if promoted
        else spec.next_action
    )
    phase = {
        "phase": "F4",
        "title": spec.title,
        "implemented": _code_status(root, spec),
        "data": chirps,
        "smoke": smoke,
        "official": official,
        "gate": gate,
        "promotion": _status(
            promotion_state,
            promotion_detail,
            [*figures.get("evidence", []), *notebooks.get("evidence", [])],
        ),
        "run_id": outputs.get("run_id", ""),
        "identity": {
            "kind": "phase4_native_outputs",
            "run_id": outputs.get("run_id", ""),
            "parent_f4c_run_id": outputs.get("parent_f4c_run_id", ""),
            "target_build_id": chirps.get("build_id", ""),
            "target_content_sha256": chirps.get("target_content_sha256", ""),
            "grid_hash_sha256": chirps.get("grid_hash_sha256", ""),
        },
        "audit_surfaces": {
            "numeric_tables": _status(
                "passed" if official.get("state") == "complete" and figures.get("state") == "passed" else "failed",
                (
                    "Saídas F4C/F4D e "
                    f"{figures.get('numeric_table_count', 0)} tabelas-fonte de figuras."
                ),
                official.get("evidence", []),
                passed=official.get("state") == "complete" and figures.get("state") == "passed",
                official_output_count=len(official.get("evidence", [])),
                figure_numeric_table_count=figures.get("numeric_table_count", 0),
                figure_numeric_catalog_sha256=figures.get(
                    "numeric_table_catalog_sha256"
                ),
            ),
            "figures": figures,
            "notebooks": notebooks,
        },
        "next_action": next_action,
    }
    return phase, {
        "smoke": smoke,
        "outputs": outputs,
        "figures": figures,
        "notebooks": notebooks,
    }


def collect_phase7_cube_status(root: Path) -> dict[str, Any]:
    cube = root / "data/processed/zarr/modeling/phase7_pacific_weekly.zarr"
    evidence = [_relative(root, cube), "scripts/build_phase7_pacific_cube.py"]
    if not cube.is_dir():
        return _status("missing", "Cubo Pacífico F7 canônico ausente.", evidence, passed=False)
    problems: list[str] = []
    for metadata_name in (".zgroup", ".zattrs", ".zmetadata"):
        if not (cube / metadata_name).is_file():
            problems.append(f"phase7_cube_metadata_missing:{metadata_name}")
    inventory = _zarr_storage_inventory(cube)
    if not inventory["files"]:
        problems.append("phase7_cube_has_no_arrays_or_chunks")
    if not problems:
        try:
            import xarray as xr

            dataset = xr.open_zarr(cube, consolidated=True)
        except Exception as exc:
            problems.append(f"phase7_cube_open_failed:{type(exc).__name__}")
            dataset = None
        if dataset is not None:
            try:
                observed_variables = set(dataset.data_vars)
                if observed_variables != PHASE7_CUBE_VARIABLES:
                    missing = ",".join(sorted(PHASE7_CUBE_VARIABLES - observed_variables)) or "none"
                    unexpected = ",".join(sorted(observed_variables - PHASE7_CUBE_VARIABLES)) or "none"
                    problems.append(
                        f"phase7_cube_variable_catalog_mismatch:missing={missing}:unexpected={unexpected}"
                    )
                observed_sizes = {name: int(dataset.sizes.get(name, 0)) for name in PHASE7_CUBE_SIZES}
                if observed_sizes != PHASE7_CUBE_SIZES:
                    problems.append(
                        f"phase7_cube_size_mismatch:expected={PHASE7_CUBE_SIZES}:actual={observed_sizes}"
                    )
                attrs = dataset.attrs
                if not (
                    int(attrs.get("phase") or 0) == 7
                    and attrs.get("role") == "real_spatial_pacific_predictors"
                    and attrs.get("daily_concat_before_weekly_resample") is True
                    and attrs.get("weekly_anchor") == "W-SUN"
                    and attrs.get("spatial_operation") == "block_mean_4x4"
                    and str(attrs.get("scalar_fusion") or "").startswith("31 physical F2 variables")
                ):
                    problems.append("phase7_cube_attribute_contract_mismatch")
                for name in sorted(PHASE7_CUBE_VARIABLES):
                    variable = dataset.get(name)
                    expected_dims = ("time",) if name in {"expected_day_count", "complete_week"} else ("time", "lat", "lon")
                    if variable is None or variable.dims != expected_dims:
                        problems.append(f"phase7_cube_dimensions_mismatch:{name}")
                    array_dir = cube / name
                    if not (array_dir / ".zarray").is_file() or not any(
                        item.is_file() and not item.name.startswith(".")
                        for item in array_dir.iterdir()
                    ):
                        problems.append(f"phase7_cube_array_not_materialized:{name}")
                for coordinate in ("time", "lat", "lon"):
                    array_dir = cube / coordinate
                    if coordinate not in dataset.coords or not (array_dir / ".zarray").is_file() or not any(
                        item.is_file() and not item.name.startswith(".")
                        for item in array_dir.iterdir()
                    ):
                        problems.append(f"phase7_cube_coordinate_not_materialized:{coordinate}")
                time_index = dataset.indexes.get("time")
                if time_index is None or len(time_index) != PHASE7_CUBE_SIZES["time"]:
                    problems.append("phase7_cube_time_axis_missing")
                else:
                    if time_index.has_duplicates or not time_index.is_monotonic_increasing:
                        problems.append("phase7_cube_time_order_invalid")
                    if not bool((time_index.dayofweek == 6).all()):
                        problems.append("phase7_cube_time_not_w_sun")
                    if any(
                        (time_index[index] - time_index[index - 1]).days != 7
                        for index in range(1, len(time_index))
                    ):
                        problems.append("phase7_cube_time_has_gaps")
                try:
                    sources = json.loads(str(attrs.get("input_stores_json") or "[]"))
                except json.JSONDecodeError:
                    sources = []
                if not isinstance(sources, list) or len(sources) != 33:
                    problems.append("phase7_cube_source_catalog_mismatch")
                else:
                    source_years: set[int] = set()
                    for value in sources:
                        try:
                            source_path = _resolve_recorded_path(root, value)
                        except (OSError, ValueError) as exc:
                            problems.append(f"phase7_cube_source_path_invalid:{exc}")
                            continue
                        if not source_path.is_dir():
                            problems.append(f"phase7_cube_source_missing:{_relative(root, source_path)}")
                        try:
                            source_years.add(int(source_path.parent.name))
                        except ValueError:
                            problems.append(f"phase7_cube_source_year_invalid:{_relative(root, source_path)}")
                    if source_years != set(range(1993, 2026)):
                        problems.append("phase7_cube_source_years_mismatch")
            except Exception as exc:
                problems.append(f"phase7_cube_inspection_failed:{type(exc).__name__}")
            finally:
                dataset.close()
    passed = not problems
    return _status(
        "passed" if passed else "failed",
        "Cubo F7 materializado com 1723 semanas, grade 10×160 e 8 canais espaciais reais."
        if passed
        else f"Cubo F7 falhou em {len(problems)} verificação(ões) de contrato.",
        evidence,
        passed=passed,
        expected_sizes=PHASE7_CUBE_SIZES,
        storage_state_sha256=inventory.get("state_sha256"),
        storage_file_count=inventory.get("file_count"),
        problems=problems,
    )


def _data_dependency_status(
    root: Path,
    phase: int,
    chirps: Mapping[str, Any],
    phase3: Mapping[str, Any],
    phase2: Mapping[str, Any] | None = None,
    cube_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    phase2_complete = bool(phase2 and (phase2.get("official") or {}).get("state") == "complete")
    cube_complete = bool(cube_status and cube_status.get("state") == "passed")
    if phase == 5:
        complete = phase2_complete
        evidence = [*(phase2 or {}).get("data", {}).get("evidence", []), *phase3.get("evidence", [])[:2]]
        detail = "F2 e fases/eventos F3 disponíveis." if complete else "F5 aguarda F2/F3 canônicos."
    elif phase == 6:
        complete = chirps.get("state") == "promoted" and phase2_complete
        evidence = chirps.get("evidence", []) + phase3.get("evidence", [])[:2]
        detail = "CHIRPS nativo e fases F3 disponíveis." if complete else "F6 aguarda CHIRPS/F3 canônicos."
    elif phase == 7:
        complete = cube_complete and phase2_complete
        evidence = [*(cube_status or {}).get("evidence", []), *(phase2 or {}).get("data", {}).get("evidence", [])]
        detail = "Cubo Pacífico, F2 e F3 disponíveis." if complete else "F7 aguarda cubo Pacífico/F2/F3."
    else:
        complete = cube_complete and chirps.get("state") == "promoted" and phase2_complete
        evidence = [*(cube_status or {}).get("evidence", []), *chirps.get("evidence", [])]
        detail = "Cubo Pacífico, CHIRPS e F3 disponíveis." if complete else "F8 aguarda cubo Pacífico/CHIRPS/F3."
    return _status("complete" if complete else "blocked", detail, evidence)


def _artifact_phase(
    root: Path,
    spec: PhaseSpec,
    chirps: Mapping[str, Any],
    phase3: Mapping[str, Any],
    phase2: Mapping[str, Any] | None = None,
    cube_status: Mapping[str, Any] | None = None,
    scientific_prerequisites: Sequence[
        tuple[str, Mapping[str, Any], frozenset[str]]
    ] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    phase = spec.number
    blocked_prerequisites = [
        (label, status)
        for label, status, accepted_states in scientific_prerequisites
        if str(status.get("state") or "") not in accepted_states
    ]
    prerequisite_evidence = [
        item
        for _, status in blocked_prerequisites
        for item in status.get("evidence", []) or []
    ]
    prerequisite_labels = ", ".join(label for label, _ in blocked_prerequisites)
    smoke_internal = _select_artifact_runs(root, phase, "smoke")
    official_internal = _select_artifact_runs(root, phase, "official")
    selected_official = official_internal.get("_selected_audits", [])
    selected_run_ids = [str(audit["run_id"]) for audit in selected_official]
    official_gates = [_gate_from_run(root, audit) for audit in selected_official]
    passed_gates = [gate for gate in official_gates if gate.get("state") == "passed"]
    if blocked_prerequisites:
        gate = _status(
            "blocked",
            f"Execução oficial bloqueada pelo(s) predecessor(es): {prerequisite_labels}.",
            prerequisite_evidence,
            passed=False,
            blocked_by=[label for label, _ in blocked_prerequisites],
        )
    elif passed_gates:
        gate = _status("passed", f"{len(passed_gates)}/{len(official_gates)} run(s) oficial(is) elegível(is) passam o gate; qualquer referência compatível positiva pode liberar a fase seguinte.", [item for status in passed_gates for item in status.get("evidence", [])], passed=True)
    elif official_internal["state"] == "complete" and official_gates:
        gate = _status("failed", "Todos os runs oficiais exigidos foram executados, mas nenhum passou o gate.", [item for status in official_gates for item in status.get("evidence", [])], passed=False)
    elif selected_official:
        gate = _status("pending", "Gate ainda parcial: falta variante oficial ou tabela elegível.", [item for status in official_gates for item in status.get("evidence", [])], passed=False)
    else:
        gate = _status("pending", "Nenhum run oficial elegível para avaliar o gate.", passed=False)

    figures = collect_figure_lineage(root, phase, selected_run_ids if selected_run_ids else None)
    notebooks = collect_notebook_contract(root, phase)
    promoted = (
        not blocked_prerequisites
        and
        official_internal["state"] == "complete"
        and gate["state"] == "passed"
        and figures["state"] == "passed"
        and notebooks["state"] == "passed"
    )
    promotion_state = "promoted" if promoted else (
        "blocked"
        if blocked_prerequisites
        else "audit_ready"
        if official_internal["state"] == "complete"
        else "blocked"
    )
    promotion_detail = (
        "Runs, notebook e figuras oficiais têm linhagem semântica."
        if promoted
        else (
            "Runs oficiais íntegros e auditáveis; promoção científica exige gate positivo e notebook/figuras válidos."
            if promotion_state == "audit_ready"
            else f"Promoção bloqueada pelo(s) predecessor(es): {prerequisite_labels}."
            if blocked_prerequisites
            else "Promoção bloqueada até completar runs oficiais íntegros."
        )
    )
    public_smoke = _strip_internal(smoke_internal)
    public_official = _strip_internal(official_internal)
    official_tables = sum(
        _safe_int(run.get("table_count"), 0)
        for run in public_official.get("selected", [])
        if isinstance(run, Mapping)
    )
    smoke_tables = sum(
        _safe_int(run.get("table_count"), 0)
        for run in public_smoke.get("selected", [])
        if isinstance(run, Mapping)
    )
    data_status = _data_dependency_status(
        root, phase, chirps, phase3, phase2, cube_status
    )
    if blocked_prerequisites:
        data_status = _status(
            "blocked",
            f"Insumos materiais existem, mas o gate científico predecessor bloqueia {spec.title}: {prerequisite_labels}.",
            [*data_status.get("evidence", []), *prerequisite_evidence],
            passed=False,
            blocked_by=[label for label, _ in blocked_prerequisites],
        )
    notebook_surface = notebooks
    if blocked_prerequisites and official_internal["state"] != "complete":
        notebook_surface = _status(
            "blocked",
            "Notebook oficial não é exigível enquanto a execução estiver bloqueada pelo gate predecessor.",
            notebooks.get("evidence", []),
            passed=False,
            static_contract_state=notebooks.get("state"),
            static_contract_problems=notebooks.get("problems", []),
        )
    if blocked_prerequisites:
        next_action = (
            f"Não executar a fase oficial; preservar o bloqueio de {prerequisite_labels} "
            "até existir nova hipótese/referência predefinida e aprovada."
        )
    elif official_internal["state"] == "complete" and gate["state"] == "failed":
        next_action = (
            "Preservar os runs e o resultado negativo; reformular o alvo/modelo "
            "antes de qualquer nova campanha."
        )
    elif promoted:
        next_action = "Não repetir sem mudança documentada de dados, código ou hipótese."
    else:
        next_action = spec.next_action
    phase_row = {
        "phase": f"F{phase}",
        "title": spec.title,
        "implemented": _code_status(root, spec),
        "data": data_status,
        "smoke": public_smoke,
        "official": public_official,
        "gate": gate,
        "promotion": _status(promotion_state, promotion_detail, [*figures.get("evidence", []), *notebooks.get("evidence", [])]),
        "run_id": ", ".join(selected_run_ids),
        "identity": {
            "kind": "artifact_runs",
            "smoke_selected": public_smoke.get("selected", []),
            "official_selected": public_official.get("selected", []),
        },
        "audit_surfaces": {
            "numeric_tables": _status(
                "passed" if official_internal["state"] == "complete" else official_internal["state"],
                (
                    f"{official_tables} tabela(s) oficial(is) e "
                    f"{smoke_tables} tabela(s) smoke em manifests auditados."
                ),
                public_official.get("evidence", []),
                passed=official_internal["state"] == "complete",
                official_table_count=official_tables,
                smoke_table_count=smoke_tables,
                official_run_ids=selected_run_ids,
                smoke_run_ids=[
                    str(run.get("run_id") or "")
                    for run in public_smoke.get("selected", [])
                    if isinstance(run, Mapping)
                ],
            ),
            "figures": figures,
            "notebooks": notebook_surface,
        },
        "next_action": next_action,
    }
    return phase_row, {
        "smoke": public_smoke,
        "official": public_official,
        "figures": figures,
        "notebooks": notebooks,
    }


def collect_workspace_hygiene(
    root: Path,
    artifact_details: Mapping[str, Any],
    chirps: Mapping[str, Any],
) -> dict[str, Any]:
    """Inventory quarantine receipts and active residue candidates, read-only."""

    quarantine_root = root / "data/quarantine"
    manifest_paths: set[Path] = set()
    if quarantine_root.is_dir():
        manifest_paths.update(quarantine_root.glob("*/manifest.json"))
        manifest_paths.update(quarantine_root.glob("*/*/manifest.json"))
    batches: list[dict[str, Any]] = []
    invalid_manifest_count = 0
    failed_record_count = 0
    quarantined_record_count = 0
    for path in sorted(manifest_paths, key=lambda candidate: candidate.as_posix()):
        try:
            raw_payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raw_payload = None
        if not isinstance(raw_payload, (dict, list)):
            invalid_manifest_count += 1
            batches.append(
                {
                    "path": _relative(root, path),
                    "manifest_sha256": _sha256_file(path),
                    "valid_json": False,
                    "record_count": 0,
                    "moved_count": 0,
                    "failed_count": 0,
                }
            )
            continue
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        rows = payload.get("records") if payload else raw_payload
        if isinstance(rows, list):
            records = [record for record in rows if isinstance(record, Mapping)]
        elif payload.get("source") and payload.get("destination"):
            records = [payload]
        else:
            records = []
        statuses: list[str] = []
        for record in records:
            status = str(record.get("status") or "").strip()
            if not status and str(record.get("operation") or "").startswith("move"):
                destination_value = str(record.get("destination") or "").strip()
                destination = Path(destination_value)
                if not destination.is_absolute():
                    destination = root / destination
                status = "moved" if destination.exists() else "legacy_move_unverified"
            statuses.append(status)
        moved_count = sum(status in {"moved", "quarantined"} for status in statuses)
        failed_count = sum(status == "failed" for status in statuses)
        failed_record_count += failed_count
        quarantined_record_count += moved_count
        batches.append(
            {
                "path": _relative(root, path),
                "manifest_sha256": _sha256_file(path),
                "valid_json": True,
                "schema": payload.get("schema")
                or payload.get("schema_version")
                or "legacy-unversioned-quarantine",
                "created_at_utc": payload.get("created_at_utc"),
                "completed_at_utc": payload.get("completed_at_utc"),
                "record_count": len(records),
                "moved_count": moved_count,
                "failed_count": failed_count,
            }
        )

    active_candidates: list[dict[str, Any]] = []
    historical_invalidated_runs: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_candidate(path: Path, category: str, reason: str) -> None:
        if not path.exists():
            return
        relative = _relative(root, path)
        if relative in seen:
            return
        seen.add(relative)
        active_candidates.append(
            {
                "path": relative,
                "kind": "directory" if path.is_dir() else "file",
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "category": category,
                "reason": reason,
                "action": "review_or_quarantine; never auto-delete",
            }
        )

    statistics = root / "data/processed/parquet/statistics"
    if statistics.is_dir():
        for prefix in ("phase40_", "phase4A_", "phase4B_", "phase4C_", "phase4D_"):
            for path in statistics.glob(f"{prefix}*"):
                if "_native_" not in path.name:
                    add_candidate(
                        path,
                        "legacy_phase4_output",
                        "saída F4 anterior ao contrato CHIRPS nativo; revisar utilitário de quarentena",
                    )
    add_candidate(
        root / "data/processed/zarr/statistics/phase4C_atlas_pixel.zarr",
        "legacy_phase4_output",
        "atlas F4 legado sem identidade native",
    )
    feature_root = root / "data/processed/zarr/features"
    if feature_root.is_dir():
        for path in feature_root.glob("chirps_native_weekly_targets.zarr.staging-*"):
            add_candidate(
                path,
                "target_staging",
                "staging CHIRPS não promovido; pode ser build ativo ou interrompido",
            )
    block_root = root / "data/interim/chirps_weekly_native_blocks"
    if chirps.get("state") == "promoted" and block_root.is_dir():
        active_build_id = str(chirps.get("build_id") or "")
        active_signature = str(chirps.get("block_signature_sha256") or "")
        for manifest_path in block_root.glob("*/manifest.json"):
            manifest = _read_json(manifest_path) or {}
            if (
                str(manifest.get("build_id") or "") == active_build_id
                and str(manifest.get("signature_sha256") or "")
                == active_signature
            ):
                continue
            add_candidate(
                manifest_path.parent,
                "superseded_chirps_block_root",
                "blocos CHIRPS não pertencem ao build v4 promovido",
            )
    modeling_root = root / "data/processed/zarr/modeling"
    if modeling_root.is_dir():
        for pattern in (".phase7_pacific_weekly.zarr.tmp*", "*.staging-*"):
            for path in modeling_root.glob(pattern):
                add_candidate(
                    path,
                    "modeling_staging",
                    "staging de cubo/modelagem não promovido",
                )

    # Only invalid or incomplete run directories are residue candidates. Older
    # complete runs remain legitimate history even when not selected.
    for phase_key, detail in artifact_details.items():
        if not isinstance(detail, Mapping):
            continue
        for mode in ("smoke", "official"):
            status = detail.get(mode)
            if not isinstance(status, Mapping):
                continue
            for run in status.get("rejected", []) or []:
                if not isinstance(run, Mapping):
                    continue
                if run.get("artifact_valid") is True and run.get("status") == "complete":
                    continue
                run_path = str(run.get("path") or "").strip()
                if run.get("status") == "complete":
                    historical_invalidated_runs.append(
                        {
                            "phase": phase_key,
                            "mode": mode,
                            "run_id": run.get("run_id"),
                            "path": run_path,
                            "run_manifest_sha256": run.get("run_manifest_sha256"),
                            "problems": list(run.get("problems") or []),
                            "disposition": "preserve_as_history; not_an_active_residue",
                        }
                    )
                    continue
                if run_path:
                    add_candidate(
                        root / run_path,
                        "invalid_or_incomplete_run",
                        f"{phase_key} {mode}: "
                        + ", ".join(str(value) for value in run.get("problems", [])[:3]),
                    )

    quarantine_passed = invalid_manifest_count == 0 and failed_record_count == 0
    active_count = len(active_candidates)
    overall_state = "clean" if quarantine_passed and active_count == 0 else "needs_attention"
    return {
        "state": overall_state,
        "detail": (
            "Nenhum resíduo ativo conhecido; recibos de quarentena são legíveis."
            if overall_state == "clean"
            else (
                f"{active_count} candidato(s) ativo(s), {invalid_manifest_count} manifesto(s) "
                f"inválido(s) e {failed_record_count} movimento(s) falho(s)."
            )
        ),
        "active_residual_candidate_count": active_count,
        "active_residual_candidates": active_candidates,
        "historical_invalidated_run_count": len(historical_invalidated_runs),
        "historical_invalidated_runs": historical_invalidated_runs,
        "quarantine": {
            "root": _relative(root, quarantine_root),
            "batch_count": len(batches),
            "manifest_invalid_count": invalid_manifest_count,
            "quarantined_record_count": quarantined_record_count,
            "failed_record_count": failed_record_count,
            "batches": batches,
        },
        "chirps_promotion_state": chirps.get("state"),
        "mutation_performed": False,
    }


def _commands() -> dict[str, list[str]]:
    return {
        "ambiente": [
            'cd /c/DEV/NINO26',
            'PY=".venv/Scripts/python.exe"',
            'export PYTHONPATH="$PWD/src"',
        ],
        "F1": [
            '"$PY" scripts/run_full_download_pipeline.py --execute --stop-on-error',
        ],
        "F2": [
            '"$PY" scripts/run_fase2_all.py',
        ],
        "F3": [
            '"$PY" scripts/run_fase3_nino.py',
            '"$PY" scripts/run_fase3_nina.py',
        ],
        "F4": [
            '"$PY" scripts/run_fase4_nino.py',
            '"$PY" scripts/run_fase4_nina.py',
        ],
        "F5": [
            '"$PY" scripts/run_fase5_all.py',
        ],
        "F6": [
            '"$PY" scripts/run_fase6_all.py',
        ],
        "F7": [
            '"$PY" scripts/run_fase7_all.py --device cuda',
        ],
        "F8": [
            '"$PY" scripts/run_fase8_all.py --device cuda',
        ],
        "validação final": [
            '"$PY" scripts/run_final_validation.py --stop-on-failure',
            '"$PY" scripts/update_painel_executivo.py',
        ],
    }


def collect_project_snapshot(root: Path = ROOT, generated_at: datetime | None = None) -> dict[str, Any]:
    root = root.resolve()
    now = generated_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    generation_seed = f"{now.astimezone(timezone.utc).isoformat()}|{root.as_posix()}"
    generation_id = "mirror_" + hashlib.sha256(generation_seed.encode("utf-8")).hexdigest()[:16]
    specs = {spec.number: spec for spec in PHASE_SPECS}
    chirps = collect_chirps_status(root)
    phase1 = _phase1(root, specs[1])
    phase2 = _phase2(root, specs[2])
    phase3, phase3_details = _phase3(root, specs[3])
    phase4, phase4_details = _phase4(root, specs[4], chirps, phase3_details["semantic"])
    phase7_cube = collect_phase7_cube_status(root)
    phases = [phase1, phase2, phase3, phase4]
    artifact_details: dict[str, Any] = {}
    for number in (5, 6, 7, 8):
        phase_row, details = _artifact_phase(
            root,
            specs[number],
            chirps,
            phase3_details["semantic"],
            phase2,
            phase7_cube,
            # Gates de uma família anterior são evidência comparativa, não
            # bloqueio de execução para estatística, ML ou redes neurais.
            scientific_prerequisites=(),
        )
        phases.append(phase_row)
        artifact_details[f"F{number}"] = details

    reset_receipts = sorted((root / "data/audit/output_resets").glob("*_apply.json"))
    if reset_receipts:
        receipt = _relative(root, reset_receipts[-1])
        for phase_row in phases[1:]:
            phase_number = int(str(phase_row["phase"]).removeprefix("F"))
            figure_products = list(
                (root / "data/processed/figures").rglob(f"FigF{phase_number}*")
            )
            table_products = list(
                (root / "data/processed/numeric-tables").rglob(f"TabF{phase_number}*")
            )
            run_products = list(
                (
                    root
                    / "data/processed/runs/official"
                    / f"fase{phase_number}"
                ).glob("*/run_manifest.json")
            )
            has_public_products = bool(figure_products or table_products or run_products)
            if (
                not has_public_products
                and phase_row["official"].get("state") != "complete"
            ):
                phase_row["official"] = _status(
                    "not_run",
                    "Saídas derivadas foram limpas; a execução canônica ainda não foi iniciada.",
                    [receipt],
                )
                phase_row["gate"] = _status(
                    "pending",
                    "Gate científico será avaliado somente após a nova execução.",
                    [receipt],
                )
                phase_row["promotion"] = _status(
                    "pending",
                    "Promoção aguarda tabelas e figuras produzidas pela nova execução.",
                    [receipt],
                )

    final_validation_path = root / "data/audit/final_validation_summary.json"
    final_validation = _read_json(final_validation_path)
    final_validation_status = audit_final_validation_summary(
        root,
        final_validation,
        final_validation_path,
    )
    hygiene = collect_workspace_hygiene(root, artifact_details, chirps)
    validations = {
        "notebooks_static": collect_notebook_contract(root),
        "persisted_final_validation": final_validation,
        "persisted_final_validation_path": _relative(root, final_validation_path),
        "persisted_final_validation_status": final_validation_status,
    }
    quality_issues: list[dict[str, str]] = []
    if final_validation_status.get("state") != "passed":
        quality_issues.append(
            {
                "phase": "GLOBAL",
                "dimension": "final_validation",
                "state": str(final_validation_status.get("state") or "failed"),
                "detail": str(final_validation_status.get("detail") or ""),
            }
        )
    if hygiene.get("state") != "clean":
        quality_issues.append(
            {
                "phase": "GLOBAL",
                "dimension": "workspace_hygiene",
                "state": str(hygiene.get("state") or "needs_attention"),
                "detail": str(hygiene.get("detail") or ""),
            }
        )
    issue_states = {
        "failed",
        "invalid",
        "invalidated",
        "incomplete",
        "partial",
        "blocked",
        "missing",
        "pending",
        "not_run",
        "in_progress",
        "audit_ready",
    }
    for phase in phases:
        for dimension in ("implemented", "data", "smoke", "official", "gate", "promotion"):
            value = phase[dimension]
            if value["state"] in issue_states:
                quality_issues.append(
                    {
                        "phase": phase["phase"],
                        "dimension": dimension,
                        "state": value["state"],
                        "detail": value["detail"],
                    }
                )
    summary = {
        "implemented": sum(phase["implemented"]["state"] == "implemented" for phase in phases),
        "smoke_complete": sum(phase["smoke"]["state"] == "complete" for phase in phases),
        "official_complete": sum(phase["official"]["state"] == "complete" for phase in phases),
        "gate_passed": sum(phase["gate"]["state"] in {"passed", "passed_with_warnings"} for phase in phases),
        "promoted": sum(phase["promotion"]["state"] in {"promoted", "operational"} for phase in phases),
        "final_validation_passed": final_validation_status.get("state") == "passed",
        "active_residual_candidates": hygiene.get(
            "active_residual_candidate_count", 0
        ),
        "quarantine_batches": (hygiene.get("quarantine") or {}).get(
            "batch_count", 0
        ),
        "quality_issue_count": len(quality_issues),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generation_id": generation_id,
        "generated_at_utc": now.astimezone(timezone.utc).isoformat(),
        "workspace": str(root),
        "machine": {"node": platform.node(), "system": platform.system(), "release": platform.release()},
        "selection_policy": (
            "validate contract, hashes, provenance, mode and gate-table presence first; "
            "use finished_at only among equally eligible candidates"
        ),
        "provenance_policy": (
            "conservative by design: an ArtifactRun fingerprints the complete src/nino_brasil tree; "
            "any source drift invalidates it until phase-specific dependency snapshots are available"
        ),
        "status_vocabulary": {
            "implemented": "código/contrato pronto para execução pelo usuário; não implica run executado",
            "smoke": "execução reduzida em dados reais; nunca é oficial",
            "official": "run completo, íntegro, reprodutível e na modalidade official",
            "gate": "comparador científico pré-declarado",
            "promotion": "dados, tabelas, notebooks e figuras com linhagem compatível",
        },
        "git": _git_state(root),
        "summary": summary,
        "phases": phases,
        "chirps_target": chirps,
        "phase7_pacific_cube": phase7_cube,
        "validation": validations,
        "workspace_hygiene": hygiene,
        "quality_issues": quality_issues,
        "details": {"F3": phase3_details, "F4": phase4_details, **artifact_details},
        "commands_git_bash": _commands(),
    }


STATE_LABELS = {
    "implemented": "Implementado",
    "incomplete": "Incompleto",
    "complete": "Completo",
    "complete_with_warnings": "Completo c/ alertas",
    "partial": "Parcial",
    "missing": "Ausente",
    "invalid": "Inválido",
    "invalidated": "Invalidado",
    "in_progress": "Em construção",
    "not_run": "Não executado",
    "failed": "Reprovado",
    "passed": "Aprovado",
    "passed_with_warnings": "Aprovado c/ alertas",
    "pending": "Pendente",
    "blocked": "Bloqueado",
    "promoted": "Promovido",
    "operational": "Operacional",
    "audit_ready": "Numérico auditável",
    "not_applicable": "N/A",
    "clean": "Limpo",
    "needs_attention": "Requer revisão",
}


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "/").replace("\n", " ")


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_markdown_escape(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def _state_cell(status: Mapping[str, Any]) -> str:
    label = STATE_LABELS.get(str(status.get("state")), str(status.get("state")))
    return f"**{label}** — {status.get('detail', '')}"


def build_markdown(snapshot: Mapping[str, Any], snapshot_sha256: str = "") -> str:
    summary = snapshot["summary"]
    git = snapshot["git"]
    details = snapshot.get("details") if isinstance(snapshot.get("details"), Mapping) else {}
    audit_surface_rows: list[tuple[Any, ...]] = []
    identity_rows: list[tuple[Any, ...]] = []
    selected_run_rows: list[tuple[Any, ...]] = []
    for phase in snapshot["phases"]:
        surfaces = (
            phase.get("audit_surfaces")
            if isinstance(phase.get("audit_surfaces"), Mapping)
            else {}
        )
        audit_surface_rows.append(
            (
                phase["phase"],
                _state_cell(surfaces.get("numeric_tables") or _status("missing", "Não reportado.")),
                _state_cell(surfaces.get("figures") or _status("missing", "Não reportado.")),
                _state_cell(surfaces.get("notebooks") or _status("missing", "Não reportado.")),
            )
        )
        if phase["phase"] in {"F1", "F2", "F3", "F4"}:
            identity = phase.get("identity") if isinstance(phase.get("identity"), Mapping) else {}
            primary_hashes = [
                str(identity.get(key) or "")
                for key in (
                    "manifest_sha256",
                    "table_catalog_sha256",
                    "phase_table_sha256",
                    "target_content_sha256",
                    "grid_hash_sha256",
                )
                if identity.get(key)
            ]
            if not primary_hashes:
                primary_hashes = [
                    str(item.get("sha256") or "")
                    for item in identity.get("artifacts", [])
                    if isinstance(item, Mapping) and item.get("sha256")
                ]
            identity_rows.append(
                (
                    phase["phase"],
                    identity.get("kind") or "—",
                    identity.get("run_id")
                    or identity.get("target_build_id")
                    or phase.get("run_id")
                    or "—",
                    "; ".join(primary_hashes) or "—",
                )
            )
    for phase_key in ("F5", "F6", "F7", "F8"):
        phase_detail = details.get(phase_key) if isinstance(details, Mapping) else None
        phase_detail = phase_detail if isinstance(phase_detail, Mapping) else {}
        for mode in ("smoke", "official"):
            status = phase_detail.get(mode)
            status = status if isinstance(status, Mapping) else {}
            selected = [
                run
                for run in status.get("selected", []) or []
                if isinstance(run, Mapping)
            ]
            if not selected:
                selected_run_rows.append(
                    (
                        phase_key,
                        mode,
                        STATE_LABELS.get(str(status.get("state")), status.get("state") or "—"),
                        "—",
                        "—",
                        "—",
                        "—",
                        "—",
                    )
                )
                continue
            for run in selected:
                gate = run.get("gate") if isinstance(run.get("gate"), Mapping) else {}
                selected_run_rows.append(
                    (
                        phase_key,
                        mode,
                        run.get("model") or "default",
                        run.get("run_id") or "—",
                        run.get("run_manifest_sha256") or "—",
                        run.get("tables_manifest_sha256") or "—",
                        run.get("input_catalog_sha256") or "—",
                        STATE_LABELS.get(str(gate.get("state")), gate.get("state") or "N/A"),
                    )
                )
    lines = [
        "# Espelho vivo do projeto NINO-BRASIL",
        "",
        "> Painel local gerado exclusivamente de evidências em disco. Implementação, smoke,",
        "> execução oficial, gate e promoção são estados independentes.",
        "",
        f"- Geração: `{snapshot['generation_id']}` em `{snapshot['generated_at_utc']}`",
        f"- Snapshot estruturado: `data/audit/project_status_snapshot.json` (`sha256={snapshot_sha256 or 'calculado na gravação'}`)",
        f"- Git: `{git.get('branch')}` / `{git.get('revision')}`; alterações locais: `{git.get('changed_paths')}`",
        f"- Política de seleção: {snapshot['selection_policy']}.",
        "",
        "## Visão condensada",
        "",
        _table(
            ["Indicador", "Valor"],
            [
                ("Fases com código/contrato pronto (não implica run)", f"{summary['implemented']}/8"),
                ("Smokes elegíveis", summary["smoke_complete"]),
                ("Execuções oficiais completas", summary["official_complete"]),
                ("Gates aprovados", summary["gate_passed"]),
                ("Fases promovidas/operacionais", summary["promoted"]),
                ("Validação final vinculada ao código atual", "sim" if summary.get("final_validation_passed") else "não"),
                ("Candidatos a resíduo ativo", summary.get("active_residual_candidates", 0)),
                ("Lotes de quarentena inventariados", summary.get("quarantine_batches", 0)),
                ("Pendências/bloqueios verificáveis", summary["quality_issue_count"]),
            ],
        ),
        "",
        "## Painel F1–F8",
        "",
        "> A coluna **Código pronto** informa somente que o entrypoint/contrato pode ser executado; código pronto não implica execução nem run oficial.",
        "> A coluna **Oficial** é a evidência independente de que um run completo existe; `Não executado` continua ausente mesmo com código pronto.",
        "",
        _table(
            ["Fase", "Produto", "Código pronto", "Dados", "Smoke", "Oficial", "Gate", "Promoção", "Run/build"],
            [
                (
                    phase["phase"],
                    phase["title"],
                    _state_cell(phase["implemented"]),
                    _state_cell(phase["data"]),
                    _state_cell(phase["smoke"]),
                    _state_cell(phase["official"]),
                    _state_cell(phase["gate"]),
                    _state_cell(phase["promotion"]),
                    phase.get("run_id") or "—",
                )
                for phase in snapshot["phases"]
            ],
        ),
        "",
        "## Superfícies de auditoria por fase",
        "",
        _table(
            ["Fase", "Tabelas numéricas", "Figuras", "Notebooks"],
            audit_surface_rows,
        ),
        "",
        "## Identidade das evidências F1–F4",
        "",
        _table(["Fase", "Contrato", "Run/build", "Hash(es) primário(s)"], identity_rows),
        "",
        "## Runs selecionados F5–F8",
        "",
        _table(
            ["Fase", "Modo", "Modelo/estado", "Run", "Manifest SHA-256", "Tabelas SHA-256", "Fontes SHA-256", "Gate"],
            selected_run_rows,
        ),
        "",
        "## Próxima ação por fase",
        "",
        _table(["Fase", "Ação"], [(phase["phase"], phase["next_action"]) for phase in snapshot["phases"]]),
        "",
        "## Alvo CHIRPS nativo",
        "",
        _table(
            ["Estado", "Contrato", "Build", "Pixels nativos", "Interpolação", "Grid SHA-256", "Conteúdo SHA-256"],
            [
                (
                    _state_cell(snapshot["chirps_target"]),
                    snapshot["chirps_target"].get("target_contract_version") or "—",
                    snapshot["chirps_target"].get("build_id") or "—",
                    (
                        f"{snapshot['chirps_target'].get('native_pixel_count', '—')} / preservados="
                        f"{snapshot['chirps_target'].get('native_pixel_preservation_verified', False)}"
                    ),
                    snapshot["chirps_target"].get("interpolation_applied"),
                    snapshot["chirps_target"].get("grid_hash_sha256") or "—",
                    snapshot["chirps_target"].get("target_content_sha256") or "—",
                )
            ],
        ),
        "",
        "## Validação final e higiene",
        "",
        _table(
            ["Dimensão", "Estado", "Evidência"],
            [
                (
                    "Validação final",
                    _state_cell(snapshot["validation"]["persisted_final_validation_status"]),
                    "; ".join(snapshot["validation"]["persisted_final_validation_status"].get("evidence", [])),
                ),
                (
                    "Resíduos/quarentena",
                    f"**{STATE_LABELS.get(str(snapshot['workspace_hygiene'].get('state')), snapshot['workspace_hygiene'].get('state'))}** — {snapshot['workspace_hygiene'].get('detail', '')}",
                    snapshot["workspace_hygiene"].get("quarantine", {}).get("root", "data/quarantine"),
                ),
            ],
        ),
        "",
        "## Pendências verificáveis",
        "",
    ]
    if snapshot["quality_issues"]:
        lines.extend(
            f"- **{issue['phase']} / {issue['dimension']} / {issue['state']}**: {issue['detail']}"
            for issue in snapshot["quality_issues"]
        )
    else:
        lines.append("- Nenhuma pendência detectada pelo contrato do espelho.")
    lines.extend(["", "## Comandos Git Bash no VS Code", ""])
    for section, commands in snapshot["commands_git_bash"].items():
        lines.extend([f"### {section}", "", "```bash", *commands, "```", ""])
    lines.extend(
        [
            "## Critério de leitura",
            "",
            "- Um smoke prova funcionamento reduzido, não superioridade científica.",
            "- Código implementado/pronto não comprova execução: run oficial ausente permanece `Não executado`.",
            "- Um run oficial com gate negativo continua sendo evidência oficial negativa.",
            "- F5/F6 só aparecem completos quando RF e XGBoost elegíveis existem; um único modelo é parcial.",
            "- F6 oficial significa merge integral, nunca apenas shards.",
            "- Promoção exige linhagem de tabelas, notebooks e figuras; gate positivo não substitui auditoria.",
            "- Data augmentation regulariza o treino, mas nunca aumenta o número de eventos independentes.",
            "- O fingerprint de `src/nino_brasil` inteiro é conservador e intencional; qualquer deriva invalida o run até existir snapshot de dependências por fase.",
            "",
        ]
    )
    return "\n".join(lines)


def _prepare_temp(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def write_outputs_atomic(
    snapshot: Mapping[str, Any],
    snapshot_path: Path,
    panel_path: Path,
) -> tuple[str, str]:
    """Publica snapshot e painel como uma transação com rollback explícito."""

    snapshot_text = json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    snapshot_sha256 = hashlib.sha256(snapshot_text.encode("utf-8")).hexdigest()
    panel_text = build_markdown(snapshot, snapshot_sha256=snapshot_sha256)
    snapshot_temp: Path | None = None
    panel_temp: Path | None = None
    snapshot_backup: Path | None = None
    panel_backup: Path | None = None
    snapshot_existed = snapshot_path.is_file()
    panel_existed = panel_path.is_file()
    snapshot_replaced = False
    panel_replaced = False
    try:
        # Os dois novos conteúdos e os dois estados anteriores ficam duráveis
        # antes da primeira promoção. Se a segunda troca falhar, o primeiro
        # destino é restaurado e o par observável continua na geração antiga.
        snapshot_temp = _prepare_temp(snapshot_path, snapshot_text)
        panel_temp = _prepare_temp(panel_path, panel_text)
        if snapshot_existed:
            snapshot_backup = _prepare_temp(
                snapshot_path,
                snapshot_path.read_text(encoding="utf-8"),
            )
        if panel_existed:
            panel_backup = _prepare_temp(
                panel_path,
                panel_path.read_text(encoding="utf-8"),
            )
        os.replace(snapshot_temp, snapshot_path)
        snapshot_replaced = True
        os.replace(panel_temp, panel_path)
        panel_replaced = True
    except Exception as original_error:
        rollback_errors: list[str] = []
        if panel_replaced:
            try:
                if panel_existed and panel_backup is not None:
                    os.replace(panel_backup, panel_path)
                    panel_backup = None
                else:
                    panel_path.unlink(missing_ok=True)
            except OSError as exc:
                rollback_errors.append(f"panel:{type(exc).__name__}:{exc}")
        if snapshot_replaced:
            try:
                if snapshot_existed and snapshot_backup is not None:
                    os.replace(snapshot_backup, snapshot_path)
                    snapshot_backup = None
                else:
                    snapshot_path.unlink(missing_ok=True)
            except OSError as exc:
                rollback_errors.append(f"snapshot:{type(exc).__name__}:{exc}")
        if rollback_errors:
            raise RuntimeError(
                "falha ao publicar o espelho e ao restaurar o par anterior: "
                + "; ".join(rollback_errors)
            ) from original_error
        raise
    finally:
        for temporary in (snapshot_temp, panel_temp, snapshot_backup, panel_backup):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return snapshot_sha256, hashlib.sha256(panel_text.encode("utf-8")).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="workspace a inspecionar")
    parser.add_argument("--snapshot", type=Path, default=None, help="saída JSON; padrão sob --root")
    parser.add_argument("--output", type=Path, default=None, help="painel Markdown; padrão sob --root")
    parser.add_argument("--dry-run", action="store_true", help="coleta e imprime o Markdown sem gravar")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    snapshot = collect_project_snapshot(root)
    if args.dry_run:
        print(build_markdown(snapshot))
        return 0
    snapshot_path = args.snapshot or root / "data/audit/project_status_snapshot.json"
    panel_path = args.output or root / "painel_executivo.md"
    snapshot_hash, panel_hash = write_outputs_atomic(snapshot, snapshot_path, panel_path)
    print(f"espelho atualizado: {panel_path}")
    print(f"snapshot: {snapshot_path} sha256={snapshot_hash}")
    print(f"painel sha256={panel_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
