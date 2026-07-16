#!/usr/bin/env python3
"""Fase 4C: ENSO(t-lag) -> CHIRPS(t) on original native Brazil pixels.

Official outputs always separate El Nino and La Nina into genesis, growth,
peak and decay at the *source* week ``t-lag``.  Statistics use exact Pearson
correlations, segmented Bretherton effective sample size, one documented BH
family per predictor/condition over all lags and pixels, and whole-field
circular-shift significance over the five non-overlapping IBGE regions.

The response comes exclusively from ``chirps_native_weekly_targets.zarr``.
No regridding or interpolation is accepted.  The rectangular target grid is
retained for F8 while F4 evaluates only cells with official Brazil overlap.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.maps.figure_registry import (  # noqa: E402
    figure_code,
    save_registered_figure,
)
from nino_brasil.artifacts import scientific_input_record, sha256_file  # noqa: E402
from nino_brasil.artifact_codes import notebook_code_for, table_code  # noqa: E402
from nino_brasil.config import confirmatory_fdr_alpha  # noqa: E402
from nino_brasil.data.phase2_master import PHYSICAL_COLUMNS  # noqa: E402
from nino_brasil.maps.plot_pixel_maps import (  # noqa: E402
    plot_pixel_field,
    save_unit_lag_heatmap,
    yellow_neutral_diverging_cmap,
)
from nino_brasil.maps.spatial_support import (  # noqa: E402
    aggregate_area_weighted_response,
    build_analysis_units,
    build_pixel_membership,
    load_ibge_biomes,
    load_ibge_regions,
)
from nino_brasil.stats.lag_analysis import (  # noqa: E402
    PHASE_ORDER,
    best_from_long_table,
    best_lag_fields,
    build_source_conditions,
    circular_shift_field_significance,
    fdr_bh,
    fdr_bh_adjusted,
    harmonic_deseasonalize_predictors,
    lagged_correlation_exact,
    load_selected_predictors,
    result_to_long_table,
)
from nino_brasil.stats.semantic_tables import verify_semantic_csv  # noqa: E402
from nino_brasil.targets.chirps_native import (  # noqa: E402
    target_to_frame,
    validate_native_target,
)


FEAT = ROOT / "data/processed/parquet/features"
STATS = ROOT / "data/processed/parquet/statistics"
FIGS = ROOT / "data/processed/figures/fase4"
TARGET = ROOT / "data/processed/zarr/features/chirps_native_weekly_targets.zarr"
PHASES = STATS / "phase3_fases_semanais_en_ln.csv"
ATLAS = ROOT / "data/processed/zarr/statistics/phase4C_native_pixel_lags.zarr"
IBGE = ROOT / "data/interim/ibge"
REG_SHP = IBGE / "BR_Regioes_2024/BR_Regioes_2024.shp"
BIO_SHP = IBGE / "Biomas_2025/lml_bioma_e250k_v20250911_A.shp"
MEMBERSHIP_EXACT = STATS / "phase4C_native_pixel_membership_exact.parquet"
MEMBERSHIP_CENTROID_QUICK = STATS / "quick/phase4C_native_pixel_membership_centroid.parquet"
TARGET_VARIABLE_CONTRACT = STATS / "phase4_chirps_target_variable_contract.csv"
TARGET_PIXEL_INVENTORY = FEAT / "phase4_chirps_native_pixels.csv"
LAGS = tuple(range(0, 79, 2))
FDR_ALPHA = confirmatory_fdr_alpha(fallback=0.05)
KEY_PREDICTOR = "nino34_ssta"
PACIFIC_VARS = tuple(PHYSICAL_COLUMNS)
if len(set(PACIFIC_VARS)) != len(PACIFIC_VARS):
    raise RuntimeError("The canonical Pacific predictor contract must contain unique variables.")
OFFICIAL_CONDITIONS = tuple(
    f"{event_type}_{phase}"
    for event_type in ("el_nino", "la_nina")
    for phase in PHASE_ORDER
)
ENSO_TYPES = ("el_nino", "la_nina")

PHASE4_ARTIFACT_LAYOUT: dict[str, tuple[str, int, str, str]] = {
    "phase4C_native_lags_por_unidade": ("C", 1, "lags_unidades", "Tab"),
    "phase4C_native_cobertura_unidades": ("C", 2, "cobertura_unidades", "Tab"),
    "phase4C_native_predictor_treatment": ("C", 3, "tratamento_preditores", "Tab"),
    "phase4C_native_best_lag_pixel": ("C", 4, "melhor_lag_pixel", "Tab"),
    "phase4C_native_best_lag_pixel_key": ("C", 5, "chave_pixel", "Tab"),
    "phase4C_native_field_significance": ("C", 6, "significancia_campo", "Tab"),
    "phase4C_native_pixel_lags": ("C", 7, "atlas_lags_pixel", "Cube"),
    "phase4C_native_peak_lag_regional_summary": ("C", 8, "lag_pico_regioes", "Tab"),
    "phase4D_native_clusters_pixels": ("D", 1, "clusters_pixels", "Tab"),
    "phase4D_native_cluster_profiles": ("D", 2, "perfis_clusters", "Tab"),
    "phase4D_native_cluster_ranking": ("D", 3, "ranking_clusters", "Tab"),
    "phase4D_native_gate_event_jackknife": ("D", 4, "jackknife_eventos", "Tab"),
    "phase4D_native_hypothesis_summary": ("D", 5, "resumo_hipoteses", "Tab"),
    "phase4D_native_target_coverage": ("D", 6, "cobertura_alvos", "Tab"),
}


def conditions_for_enso_type(enso_type: str | None) -> tuple[str, ...]:
    """Return the four source-phase conditions for one ENSO sign, or all eight."""

    if enso_type is None:
        return OFFICIAL_CONDITIONS
    if enso_type not in ENSO_TYPES:
        raise ValueError(f"Unsupported ENSO type: {enso_type!r}")
    return tuple(f"{enso_type}_{phase}" for phase in PHASE_ORDER)


def scoped_artifact_path(path: Path, enso_type: str | None) -> Path:
    """Resolve a public F4 artifact with the notebook's Nino/Nina pre-code."""

    if enso_type is None:
        return path
    try:
        block, ordinal, slug, kind = PHASE4_ARTIFACT_LAYOUT[path.stem]
    except KeyError as exc:
        raise KeyError(f"F4 artifact without public pre-code: {path.stem}") from exc
    notebook = notebook_code_for(4, block, enso_type)
    stem = (
        table_code(notebook, ordinal, slug=slug)
        if kind == "Tab"
        else f"Cube{notebook}{ordinal}_{slug}"
    )
    return path.with_name(f"{stem}{path.suffix}")


def phase_table_path_for_enso_type(enso_type: str | None) -> Path:
    """Resolve the canonical F3 table for the requested independent analysis."""

    if enso_type is None:
        return PHASES
    scope = "F3Nino" if enso_type == "el_nino" else "F3Nina"
    notebook = notebook_code_for(3, "B", enso_type)
    return STATS / scope / f"{table_code(notebook, 2, slug='fases_semanais')}.csv"


def summarize_peak_pixel_lags_by_region(
    key_pixels: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Resume lags de pico somente depois da estimação por pixel original.

    O lag médio é condicionado aos pixels cujo lag sobreviveu ao FDR. A fração
    de área com suporte é publicada ao lado para impedir que uma média de poucos
    pixels seja interpretada como resposta regional completa.
    """

    required_key = {
        "pixel_id",
        "condicao_fonte",
        "best_lag_sem_fdr",
        "r_no_best_lag_fdr",
    }
    required_membership = {
        "pixel_id",
        "tipo_unidade",
        "id_unidade",
        "nome_unidade",
        "peso_agregacao",
    }
    if missing := required_key.difference(key_pixels.columns):
        raise KeyError(f"best-pixel key table missing {sorted(missing)}")
    if missing := required_membership.difference(membership.columns):
        raise KeyError(f"membership table missing {sorted(missing)}")
    peak = key_pixels.loc[
        key_pixels["condicao_fonte"].astype(str).str.endswith("_pico")
    ].copy()
    regions = membership.loc[
        membership["tipo_unidade"].eq("regiao"),
        ["pixel_id", "id_unidade", "nome_unidade", "peso_agregacao"],
    ].copy()
    joined = peak.merge(regions, on="pixel_id", how="inner", validate="many_to_many")
    rows: list[dict[str, object]] = []
    for (condition, unit_id, region_name), group in joined.groupby(
        ["condicao_fonte", "id_unidade", "nome_unidade"],
        sort=True,
        dropna=False,
    ):
        lag = pd.to_numeric(group["best_lag_sem_fdr"], errors="coerce")
        correlation = pd.to_numeric(group["r_no_best_lag_fdr"], errors="coerce")
        weight = pd.to_numeric(group["peso_agregacao"], errors="coerce")
        valid = lag.notna() & correlation.notna() & weight.gt(0)
        total_weight = float(weight.where(weight.gt(0)).sum())
        supported_weight = float(weight.where(valid, 0.0).sum())
        lag_values = lag.loc[valid].to_numpy(dtype=float)
        lag_weights = weight.loc[valid].to_numpy(dtype=float)
        event_type = str(condition).removesuffix("_pico")
        rows.append(
            {
                "condicao_fonte": str(condition),
                "tipo_enso_fonte": event_type,
                "fase_fonte_em_t_menos_lag": "pico",
                "id_unidade": str(unit_id),
                "regiao": str(region_name),
                "n_pixels_originais": int(group["pixel_id"].nunique()),
                "n_pixels_com_lag_fdr": int(group.loc[valid, "pixel_id"].nunique()),
                "fracao_area_com_lag_fdr": (
                    supported_weight / total_weight if total_weight > 0 else np.nan
                ),
                "lag_medio_semanas_pixel_fdr": (
                    float(np.average(lag_values, weights=lag_weights))
                    if lag_values.size
                    else np.nan
                ),
                "lag_mediano_semanas_pixel_fdr": (
                    float(np.median(lag_values)) if lag_values.size else np.nan
                ),
                "lag_p25_semanas_pixel_fdr": (
                    float(np.quantile(lag_values, 0.25)) if lag_values.size else np.nan
                ),
                "lag_p75_semanas_pixel_fdr": (
                    float(np.quantile(lag_values, 0.75)) if lag_values.size else np.nan
                ),
                "regra_resumo": (
                    "lags estimados pixel-a-pixel; media ponderada por cos(lat)*fracao "
                    "somente entre pixels com lag FDR; cobertura publicada separadamente"
                ),
            }
        )
    return pd.DataFrame(rows)


def shapefile_bundle_paths(path: Path) -> tuple[Path, ...]:
    """Return every component that defines one local shapefile dataset."""

    path = Path(path)
    required = tuple(path.with_suffix(suffix) for suffix in (".shp", ".shx", ".dbf", ".prj"))
    missing = [str(component) for component in required if not component.is_file()]
    if missing:
        raise FileNotFoundError(
            "Incomplete IBGE shapefile bundle; missing component(s): " + ", ".join(missing)
        )
    components = tuple(
        sorted(
            (
                candidate
                for candidate in path.parent.glob(path.stem + ".*")
                if candidate.is_file()
            ),
            key=lambda candidate: candidate.name.casefold(),
        )
    )
    if not components:
        raise FileNotFoundError(f"No shapefile components found for {path}")
    return components


def _shapefile_bundle_sha256(path: Path) -> str:
    records = [
        f"{component.name}\0{component.stat().st_size}\0{sha256_file(component)}"
        for component in shapefile_bundle_paths(path)
    ]
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def ibge_geometry_contract() -> dict[str, str]:
    """Fingerprint the exact IBGE region/biome bundles used by membership."""

    regions = _shapefile_bundle_sha256(REG_SHP)
    biomes = _shapefile_bundle_sha256(BIO_SHP)
    combined = hashlib.sha256(
        f"regions={regions}\nbiomes={biomes}".encode("utf-8")
    ).hexdigest()
    return {
        "ibge_regions_bundle_sha256": regions,
        "ibge_biomes_bundle_sha256": biomes,
        "ibge_geometry_bundle_sha256": combined,
    }


def ibge_geometry_input_paths() -> list[Path]:
    """List the complete, hashable geometry inputs for semantic sidecars."""

    return [*shapefile_bundle_paths(REG_SHP), *shapefile_bundle_paths(BIO_SHP)]


def predictor_catalog_sha256(names) -> str:
    """Hash an ordered predictor catalog without relying on a filename."""

    normalized = [str(name).strip() for name in names]
    if not normalized or any(not name for name in normalized):
        raise ValueError("Predictor catalog must contain non-empty names.")
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def target_build_manifest_path(target_contract: dict[str, str]) -> Path:
    """Locate the immutable builder manifest referenced by a target contract."""

    signature = target_contract["target_block_signature_sha256"]
    path = (
        ROOT
        / "data/interim/chirps_weekly_native_blocks"
        / signature[:16]
        / "manifest.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            "CHIRPS target build manifest is missing for signature "
            f"{signature}: {path}"
        )
    return path


def write_phase4_csv_manifests(
    outputs: list[tuple[Path, pd.DataFrame]],
    *,
    run_id: str,
    stage: str,
    contract: dict[str, object],
    inputs: list[Path],
) -> None:
    """Write hash-verifiable sidecars for canonical/quick Phase 4 CSVs.

    CSV columns carry the scientific identity for row-level inspection.  The
    sidecar independently binds the complete file bytes to the exact input
    files used by the run, so downstream gates do not need to trust filenames.
    """

    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot finalize Phase 4 provenance; input file(s) missing: "
            + ", ".join(missing)
        )
    input_records = [scientific_input_record(path) for path in inputs]
    created_utc = datetime.now(timezone.utc).isoformat()
    for path, frame in outputs:
        if not path.is_file():
            raise FileNotFoundError(f"Phase 4 output was not written: {path}")
        manifest = {
            "schema_version": "nino-brasil-phase4-semantic-output-v1",
            "run_id": run_id,
            "created_utc": created_utc,
            "contract": {
                "phase": "F4",
                "stage": stage,
                "artifact_type": "numeric_table",
                **contract,
            },
            "artifact": {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "rows": int(len(frame)),
                "columns": int(frame.shape[1]),
                "column_names": list(frame.columns),
            },
            "inputs": input_records,
        }
        sidecar = Path(f"{path}.manifest.json")
        temporary = sidecar.with_name(
            sidecar.name + f".tmp-{uuid.uuid4().hex[:8]}"
        )
        temporary.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, sidecar)


def write_phase4_directory_manifest(
    path: Path,
    *,
    run_id: str,
    stage: str,
    contract: dict[str, object],
    inputs: list[Path],
) -> None:
    """Bind a Zarr/directory output and its inputs through a tree hash."""

    if not path.is_dir():
        raise FileNotFoundError(f"Phase 4 directory output was not written: {path}")
    missing = [str(input_path) for input_path in inputs if not input_path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot finalize Phase 4 provenance; input file(s) missing: "
            + ", ".join(missing)
        )
    directory_record = scientific_input_record(path)
    dimensions: dict[str, int] = {}
    arrays: dict[str, dict[str, object]] = {}
    if path.suffix.lower() == ".zarr":
        dataset = xr.open_zarr(path, consolidated=None)
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
        finally:
            dataset.close()
    manifest = {
        "schema_version": "nino-brasil-phase4-semantic-output-v1",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "phase": "F4",
            "stage": stage,
            "artifact_type": "numeric_array",
            **contract,
        },
        "artifact": {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "is_directory": True,
            "tree_sha256": directory_record.get("tree_sha256"),
            "n_files": directory_record.get("n_files"),
            "dimensions": dimensions,
            "arrays": arrays,
        },
        "inputs": [scientific_input_record(input_path) for input_path in inputs],
    }
    sidecar = Path(f"{path}.manifest.json")
    temporary = sidecar.with_name(sidecar.name + f".tmp-{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, sidecar)


def verify_phase4_output_manifest(path: Path, *, expected_run_id: str) -> dict[str, object]:
    """Refuse a Phase 4 table whose bytes or recorded inputs have drifted."""

    sidecar = Path(f"{path}.manifest.json")
    try:
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid or missing Phase 4 sidecar {sidecar}: {exc}") from exc
    if str(manifest.get("run_id") or "").strip() != expected_run_id:
        raise ValueError(f"Phase 4 sidecar run_id mismatch for {path}")
    if str((manifest.get("contract") or {}).get("phase") or "").upper() != "F4":
        raise ValueError(f"Phase 4 sidecar contract mismatch for {path}")
    artifact = manifest.get("artifact") or {}
    recorded_path = Path(str(artifact.get("path") or ""))
    resolved_artifact = (
        recorded_path if recorded_path.is_absolute() else ROOT / recorded_path
    ).resolve()
    try:
        resolved_artifact.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Phase 4 artifact escapes project root: {recorded_path}") from exc
    is_directory = bool(artifact.get("is_directory"))
    if resolved_artifact != path.resolve() or not path.exists():
        raise ValueError(f"Phase 4 sidecar points to another artifact: {path}")
    if is_directory:
        if not path.is_dir():
            raise ValueError(f"Phase 4 sidecar expected a directory artifact: {path}")
        current = scientific_input_record(path)
        if str(artifact.get("tree_sha256") or "") != str(
            current.get("tree_sha256") or ""
        ):
            raise ValueError(f"Phase 4 artifact tree hash drift: {path}")
    else:
        if not path.is_file():
            raise ValueError(f"Phase 4 sidecar expected a file artifact: {path}")
        if str(artifact.get("sha256") or "") != sha256_file(path):
            raise ValueError(f"Phase 4 artifact hash drift: {path}")
    input_records = manifest.get("inputs")
    if not isinstance(input_records, list) or not input_records:
        raise ValueError(f"Phase 4 sidecar has no input records: {sidecar}")
    for record in input_records:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid Phase 4 input record in {sidecar}")
        recorded_input = Path(str(record.get("path") or ""))
        resolved_input = (
            recorded_input if recorded_input.is_absolute() else ROOT / recorded_input
        ).resolve()
        try:
            resolved_input.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Phase 4 input escapes project root: {recorded_input}"
            ) from exc
        if not resolved_input.exists():
            raise ValueError(f"Phase 4 recorded input is missing: {resolved_input}")
        if bool(record.get("is_directory")):
            if not resolved_input.is_dir():
                raise ValueError(
                    f"Phase 4 recorded input changed from directory: {resolved_input}"
                )
            current = scientific_input_record(resolved_input)
            expected_tree = str(record.get("tree_sha256") or "")
            if not expected_tree or str(current.get("tree_sha256") or "") != expected_tree:
                raise ValueError(f"Phase 4 input tree hash drift: {resolved_input}")
        else:
            if not resolved_input.is_file():
                raise ValueError(
                    f"Phase 4 recorded input changed from file: {resolved_input}"
                )
            expected_hash = str(record.get("sha256") or "")
            if not expected_hash or sha256_file(resolved_input) != expected_hash:
                raise ValueError(f"Phase 4 input hash drift: {resolved_input}")
    return manifest


def _archive_existing(path: Path, *, reason: str) -> Path:
    resolved = path.resolve()
    if ROOT.resolve() not in resolved.parents:
        raise ValueError(f"Refusing to archive outside project: {resolved}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "data/quarantine/phase4c" / stamp / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    (destination.parent / "README.txt").write_text(
        f"Archived, not deleted.\nSource: {path}\nReason: {reason}\n",
        encoding="utf-8",
    )
    return destination


def native_target_contract(target: xr.Dataset) -> dict[str, str]:
    """Return the immutable target identity required by downstream phases."""

    mapping = {
        "target_build_id": "build_id",
        "target_block_signature_sha256": "block_signature_sha256",
        "target_contract_version": "target_contract_version",
    }
    contract = {
        column: str(target.attrs.get(attribute, "")).strip()
        for column, attribute in mapping.items()
    }
    missing = [column for column, value in contract.items() if not value]
    if missing:
        raise ValueError(
            "Native CHIRPS target lacks immutable build identity fields: "
            f"{missing}. Rebuild/promote it with the audited builder."
        )
    return contract


def load_canonical_phase_table(
    index: pd.DatetimeIndex,
    *,
    phase_table_path: Path = PHASES,
) -> tuple[pd.DataFrame, str, str]:
    """Load F3 phases only when the semantic artifact hash and run ID validate."""

    if not phase_table_path.exists():
        raise FileNotFoundError(
            "Canonical F3 phase table is missing: "
            f"{phase_table_path}. Run scripts/run_fase3_all.py."
        )
    verification = verify_semantic_csv(phase_table_path, verify_inputs=False)
    if not bool(verification.get("valid")):
        raise ValueError(
            "Canonical F3 phase table or its semantic manifest failed hash validation: "
            f"{verification}"
        )
    parent_run_id = str(verification.get("run_id", "")).strip()
    artifact_sha256 = str(verification.get("current_sha256", "")).strip()
    if not parent_run_id or not artifact_sha256:
        raise ValueError("Canonical F3 phase contract lacks run_id or artifact SHA-256.")
    phases = pd.read_csv(
        phase_table_path, parse_dates=["week_ending_sunday"]
    ).set_index("week_ending_sunday")
    phases = phases.reindex(index).fillna(
        {"fase": "neutro", "tipo": "neutro", "event_id": ""}
    )
    return phases, parent_run_id, artifact_sha256


def load_native_response() -> tuple[
    xr.Dataset,
    pd.DataFrame,
    pd.DataFrame,
    str,
    dict[str, str],
]:
    if not TARGET.exists():
        raise FileNotFoundError(
            f"Native CHIRPS target not found: {TARGET}. Run "
            "python scripts/build_phase4_chirps_targets.py first."
        )
    target = xr.open_zarr(TARGET, consolidated=None)
    validation = validate_native_target(target, deep=False)
    if not validation.valid:
        raise ValueError(f"Native CHIRPS target is invalid: {validation.errors}")
    if target.attrs.get("deep_validation_passed") is not True:
        raise ValueError(
            "Native CHIRPS target has no successful deep-validation stamp; rerun the builder."
        )
    response, pixels = target_to_frame(
        target,
        variable="precip_robust_z",
        brazil_only=True,
        mask_rule="overlap",
    )
    return (
        target,
        response.sort_index(),
        pixels,
        validation.grid_hash,
        native_target_contract(target),
    )


def load_units():
    return build_analysis_units(load_ibge_regions(REG_SHP), load_ibge_biomes(BIO_SHP))


def load_membership(
    units,
    pixels: pd.DataFrame,
    *,
    centroid_quick: bool,
    grid_hash: str,
    geometry_contract: dict[str, str],
    replace_stale: bool = False,
) -> pd.DataFrame:
    path = MEMBERSHIP_CENTROID_QUICK if centroid_quick else MEMBERSHIP_EXACT
    if path.exists():
        membership = pd.read_parquet(path)
        recorded = set(membership.get("grid_hash", pd.Series(dtype=str)).astype(str))
        contract_errors: list[str] = []
        if recorded != {grid_hash}:
            contract_errors.append(f"grid_hash={recorded}, expected {grid_hash}")
        for column, expected in geometry_contract.items():
            values = set(
                membership.get(column, pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .str.strip()
            )
            values.discard("")
            if values != {expected}:
                contract_errors.append(f"{column}={values}, expected {expected}")
        if contract_errors and not replace_stale:
            raise ValueError(
                f"Membership cache {path} does not match the current grid/IBGE geometry; "
                + "; ".join(contract_errors)
                + ". Pass --replace-existing to archive and rebuild it explicitly."
            )
        if not contract_errors:
            return membership
        _archive_existing(
            path,
            reason="stale Phase 4 membership grid or IBGE geometry contract",
        )
    membership = build_pixel_membership(
        pixels[["pixel_id", "lat", "lon"]],
        units,
        boundary_method="centroid" if centroid_quick else "area",
    )
    membership["grid_hash"] = grid_hash
    membership["membership_status"] = (
        "quick_centroid_not_for_inference"
        if centroid_quick
        else "canonical_equal_area_official_ibge"
    )
    for column, value in geometry_contract.items():
        membership[column] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    membership.to_parquet(path, index=False)
    return membership


def official_source_conditions(
    phase_table: pd.DataFrame,
    *,
    enso_type: str | None = None,
) -> dict:
    requested = conditions_for_enso_type(enso_type)
    all_conditions = build_source_conditions(
        phase_table,
        include_all_weeks=False,
        include_event_aggregates=False,
        include_la_nina=enso_type != "el_nino",
    )
    missing = set(requested).difference(all_conditions)
    if missing:
        raise ValueError(f"Phase table cannot create conditions {sorted(missing)}")
    return {name: all_conditions[name] for name in requested}


def compute_unit_lags(
    deseasonalized: pd.DataFrame,
    unit_series: pd.DataFrame,
    phase_table: pd.DataFrame,
    conditions: dict,
    *,
    fdr_alpha: float = FDR_ALPHA,
) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for predictor_name in deseasonalized.columns:
        for condition in conditions.values():
            result = lagged_correlation_exact(
                deseasonalized[predictor_name],
                unit_series,
                LAGS,
                condition,
                phase_table,
            )
            tables.append(
                result_to_long_table(
                    result,
                    predictor_name=predictor_name,
                    condition=condition,
                    column_name="id_unidade",
                    alpha_fdr=fdr_alpha,
                )
            )
    return pd.concat(tables, ignore_index=True)


def _best_pixel_table(
    result: dict[str, np.ndarray],
    pixels: pd.DataFrame,
    *,
    predictor_name: str,
    condition,
    fdr_alpha: float = FDR_ALPHA,
) -> pd.DataFrame:
    p = np.asarray(result["p"])
    q = fdr_bh_adjusted(p)
    significant = fdr_bh(p, alpha=fdr_alpha)
    descriptive = best_lag_fields(result)
    inferential = best_lag_fields(result, significant=significant)
    desc_index = descriptive["best_index"]
    sig_index = inferential["best_index"]

    def take(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
        columns = np.arange(values.shape[1])
        safe = np.where(indices >= 0, indices, 0)
        out = values[safe, columns].astype(float, copy=True)
        out[indices < 0] = np.nan
        return out

    table = pixels.copy()
    table["variavel"] = predictor_name
    table["condicao_fonte"] = condition.name
    table["tipo_enso_fonte"] = condition.tipo_enso_fonte
    table["fase_fonte_em_t_menos_lag"] = condition.fase_fonte_em_t_menos_lag
    table["best_lag_sem_descritivo"] = descriptive["best_lag_sem"]
    table["r_no_best_lag_descritivo"] = descriptive["r_no_best_lag"]
    table["p_no_best_lag_descritivo"] = descriptive["p_no_best_lag"]
    table["q_no_best_lag_descritivo"] = take(q, desc_index)
    table["n_eff_no_best_lag_descritivo"] = descriptive["n_eff_no_best_lag"]
    table["best_lag_sem_fdr"] = inferential["best_lag_sem"]
    table["r_no_best_lag_fdr"] = inferential["r_no_best_lag"]
    table["p_no_best_lag_fdr"] = inferential["p_no_best_lag"]
    table["q_no_best_lag_fdr"] = take(q, sig_index)
    table["n_eff_no_best_lag_fdr"] = inferential["n_eff_no_best_lag"]
    table["fdr_family_id"] = (
        f"{predictor_name}|{condition.name}|todos_lags_x_pixels_brasil"
    )
    table["fdr_family_n_tests"] = int(np.isfinite(p).sum())
    table["fdr_alpha"] = fdr_alpha
    table["lag_rule"] = "predictor and ENSO phase at t-lag; CHIRPS response at t"
    return table


def compute_pixel_atlas(
    deseasonalized: pd.DataFrame,
    response: pd.DataFrame,
    pixels: pd.DataFrame,
    phase_table: pd.DataFrame,
    conditions: dict,
    *,
    destination: Path,
    replace_existing: bool,
    grid_hash: str,
    run_id: str,
    contract_metadata: dict[str, str] | None = None,
    fdr_alpha: float = FDR_ALPHA,
) -> pd.DataFrame:
    if destination.exists():
        if not replace_existing:
            raise FileExistsError(
                f"{destination} already exists. Pass --replace-existing to archive it first."
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(destination.name + f".staging-{uuid.uuid4().hex[:8]}")
    best_tables: list[pd.DataFrame] = []
    first = True
    for predictor_name in deseasonalized.columns:
        condition_datasets: list[xr.Dataset] = []
        for condition in conditions.values():
            print(f"[4C pixel] {predictor_name} | {condition.name}")
            result = lagged_correlation_exact(
                deseasonalized[predictor_name],
                response,
                LAGS,
                condition,
                phase_table,
            )
            q = fdr_bh_adjusted(result["p"])
            significant = fdr_bh(result["p"], alpha=fdr_alpha)
            condition_datasets.append(
                xr.Dataset(
                    {
                        "r": (("lag_sem", "pixel"), result["r"].astype("float32")),
                        "p": (("lag_sem", "pixel"), result["p"].astype("float32")),
                        "q_fdr_bh": (("lag_sem", "pixel"), q.astype("float32")),
                        "fdr_bh_reject": (("lag_sem", "pixel"), significant),
                        "n_eff_bretherton": (
                            ("lag_sem", "pixel"), result["n_eff"].astype("float32")
                        ),
                    },
                    coords={
                        "lag_sem": np.asarray(LAGS, dtype="int16"),
                        "pixel": pixels["pixel_id"].to_numpy(dtype="int64"),
                    },
                ).expand_dims(condicao_fonte=[condition.name])
            )
            best_tables.append(
                _best_pixel_table(
                    result,
                    pixels,
                    predictor_name=predictor_name,
                    condition=condition,
                    fdr_alpha=fdr_alpha,
                )
            )
        predictor_dataset = xr.concat(condition_datasets, dim="condicao_fonte")
        predictor_dataset = predictor_dataset.expand_dims(variavel=[predictor_name])
        predictor_dataset.attrs.update(
            {
                "grid_hash_sha256": grid_hash,
                "target": str(TARGET.relative_to(ROOT)),
                "spatial_contract": "original CHIRPS pixels with Brazil overlap; no interpolation",
                "lag_contract": "Pacific predictor and phase at t-lag versus CHIRPS at t",
                "fdr_contract": "BH per predictor x source-condition over all lags x Brazil pixels",
                "fdr_alpha": fdr_alpha,
                "analysis_run_id": run_id,
                **(contract_metadata or {}),
            }
        )
        encoding = {
            name: {"chunks": (1, 1, len(LAGS), min(2048, len(pixels)))}
            for name in predictor_dataset.data_vars
        }
        if first:
            predictor_dataset.to_zarr(
                staged,
                mode="w",
                consolidated=False,
                zarr_format=2,
                encoding=encoding,
            )
            first = False
        else:
            predictor_dataset.to_zarr(
                staged,
                mode="a",
                append_dim="variavel",
                consolidated=False,
                zarr_format=2,
            )
    opened = xr.open_zarr(staged, consolidated=False)
    if (
        opened.sizes.get("variavel") != len(deseasonalized.columns)
        or opened.sizes.get("condicao_fonte") != len(conditions)
        or opened.sizes.get("pixel") != len(pixels)
    ):
        raise ValueError(f"Staged pixel atlas has an invalid shape and was left at {staged}.")
    opened.close()
    # Preserve the last valid atlas throughout the expensive calculation.  It
    # is archived only after the replacement staging has passed its structural
    # checks, immediately before promotion.
    if destination.exists():
        archived = _archive_existing(
            destination, reason="replaced by a validated exact native-pixel Phase 4C run"
        )
        print(f"[archive] {archived.relative_to(ROOT)}")
    staged.replace(destination)
    return pd.concat(best_tables, ignore_index=True)


def compute_regional_field_significance(
    deseasonalized: pd.DataFrame,
    unit_series: pd.DataFrame,
    units,
    phase_table: pd.DataFrame,
    conditions: dict,
    *,
    n_permutations: int,
    all_predictors: bool,
) -> pd.DataFrame:
    region_ids = units.loc[units["tipo_unidade"].eq("regiao"), "id_unidade"].tolist()
    response_field = unit_series[[column for column in region_ids if column in unit_series]]
    predictors = list(deseasonalized.columns) if all_predictors else [KEY_PREDICTOR]
    tables: list[pd.DataFrame] = []
    for predictor_name in predictors:
        if predictor_name not in deseasonalized:
            continue
        for condition in conditions.values():
            print(
                f"[4C field] {predictor_name} | {condition.name} | "
                f"B={n_permutations} whole-field shifts"
            )
            _, field, shifts = circular_shift_field_significance(
                deseasonalized[predictor_name],
                response_field,
                LAGS,
                condition,
                phase_table,
                n_permutations=n_permutations,
                seed=42,
                point_alpha=FDR_ALPHA,
            )
            field["variavel"] = predictor_name
            field["field_domain"] = "five non-overlapping official IBGE regions"
            field["shift_min_sem"] = int(shifts.min())
            field["shift_max_sem"] = int(shifts.max())
            tables.append(field)
    return pd.concat(tables, ignore_index=True)


def heatmap_table_for(
    long_table: pd.DataFrame,
    predictor: str,
    event_type: str,
    names: pd.Series,
) -> pd.DataFrame:
    names_condition = [f"{event_type}_{phase}" for phase in PHASE_ORDER]
    subset = long_table[
        long_table["variavel"].eq(predictor)
        & long_table["condicao_fonte"].isin(names_condition)
    ]
    best = best_from_long_table(
        subset,
        group_columns=["id_unidade", "fase_fonte_em_t_menos_lag"],
        require_fdr=False,
    )
    if not best.empty:
        best = best.copy()
        best["nome_unidade"] = best["id_unidade"].map(names)
    return best


def save_figures(
    unit_lags: pd.DataFrame,
    best_pixels: pd.DataFrame,
    field_significance: pd.DataFrame,
    pixels: pd.DataFrame,
    units,
    *,
    quick: bool,
    sensitivity: bool = False,
    enso_type: str | None = None,
    run_id: str,
    source_paths: dict[str, Path],
    fdr_alpha: float = FDR_ALPHA,
) -> None:
    base_figure_dir = (
        FIGS / "quick" if quick else FIGS / "sensitivity" if sensitivity else FIGS
    )
    figure_dir = base_figure_dir / enso_type if enso_type else base_figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    names = units.set_index("id_unidade")["nome_unidade"]
    unit_order = list(
        units.sort_values(["tipo_unidade", "nome_unidade"])["nome_unidade"].unique()
    )
    event_types = ENSO_TYPES if enso_type is None else (enso_type,)
    event_indices = {"el_nino": 1, "la_nina": 2}
    for event_type in event_types:
        event_index = event_indices[event_type]
        table = heatmap_table_for(unit_lags, KEY_PREDICTOR, event_type, names)
        if table.empty:
            continue
        order = [unit for unit in unit_order if unit in set(table["nome_unidade"])]
        interpretation = (
            f"Lag da resposta CHIRPS ao {KEY_PREDICTOR} por fase-fonte de "
            f"{event_type.replace('_', ' ')}; fase e preditor sao medidos em t-lag. "
            "O marcador de significancia usa N_eff e BH dentro da familia declarada."
        )
        code = figure_code(4, "C", event_index, f"lags_regiao_bioma_{event_type}")
        output = figure_dir / f"{code}.png"
        metadata = (
            f"CHIRPS native 0.25 + master NINO26 | IBGE 2024/2025 | "
            f"lags {LAGS[0]}-{LAGS[-1]} weeks | source phase at t-lag"
        )
        save_unit_lag_heatmap(
            table,
            output,
            title=f"4C - Lag by region/biome | {KEY_PREDICTOR} | {event_type}",
            interpretation=interpretation,
            metadata=metadata,
            unit_order=order,
        )
        if not quick and not sensitivity:
            from nino_brasil.viz import registrar_figura

            registrar_figura(
                None,
                code,
                fase=4,
                bloco="C",
                titulo=f"Lag by region and biome ({event_type})",
                descricao=interpretation + " " + metadata,
                hipotese="HIP1",
                notebook="notebooks/fase4/4C_sinal_pixel_lags.ipynb",
                run_id=run_id,
                fontes={
                    "lags_unidades_fases": source_paths["unit_lags"],
                    "field_significance": source_paths["field_significance"],
                },
            )

    brazil = units[units["tipo_unidade"].eq("regiao")].dissolve()
    regions = units[units["tipo_unidade"].eq("regiao")]
    peak_conditions = {
        "el_nino": (3, "el_nino_pico"),
        "la_nina": (4, "la_nina_pico"),
    }
    for event_type in event_types:
        index, condition_name = peak_conditions[event_type]
        subset = best_pixels[
            best_pixels["variavel"].eq(KEY_PREDICTOR)
            & best_pixels["condicao_fonte"].eq(condition_name)
        ].set_index("pixel_id")
        if subset.empty:
            continue
        values = pixels["pixel_id"].map(subset["r_no_best_lag_fdr"]).to_numpy(dtype=float)
        fig, axis = plt.subplots(figsize=(13, 11))
        mesh = plot_pixel_field(
            axis,
            pixels[["lat", "lon"]],
            values,
            brazil_geometry=brazil,
            boundaries=regions,
            cmap=yellow_neutral_diverging_cmap(),
            vmin=-0.5,
            vmax=0.5,
            title=f"FDR-significant r at best lag | {KEY_PREDICTOR} | {condition_name}",
        )
        fig.colorbar(mesh, ax=axis, fraction=0.035, pad=0.02).set_label(
            "r (grey=no FDR-significant lag)"
        )
        if quick or sensitivity:
            prefix = "quick" if quick else "sensitivity"
            fig.savefig(
                figure_dir / f"{prefix}_pixel_{condition_name}.png",
                dpi=160,
                bbox_inches="tight",
            )
            plt.close(fig)
        else:
            output = save_registered_figure(
                fig,
                phase=4,
                block="C",
                index=index,
                slug=f"mapa_pixel_{condition_name}",
                interpretation=(
                    f"Original CHIRPS pixels with an FDR-significant lag for {condition_name}; "
                    "ENSO phase and predictor are evaluated at t-lag, response at t. "
                    f"Grey is not zero: it means no lag passed BH q<{fdr_alpha:.2f}."
                ),
                metadata=(
                    "CHIRPS native grid; official Brazil overlap; no interpolation | "
                    f"lags {LAGS[0]}-{LAGS[-1]} weeks"
                ),
                figures_dir=figure_dir,
                reserve_bottom=0.12,
            )
            from nino_brasil.viz import registrar_figura

            registrar_figura(
                None,
                output.stem,
                fase=4,
                bloco="C",
                titulo=f"Native-pixel FDR map: {condition_name}",
                descricao=(
                    "Original CHIRPS pixels; predictor and ENSO phase at t-lag, response at t."
                ),
                hipotese="HIP1",
                notebook="notebooks/fase4/4C_sinal_pixel_lags.ipynb",
                run_id=run_id,
                fontes={
                    "pixel_best_lag_FDR": source_paths["best_pixels"],
                    "field_significance": source_paths["field_significance"],
                },
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enso-type",
        choices=ENSO_TYPES,
        help="run one independent ENSO analysis (el_nino or la_nina)",
    )
    parser.add_argument("--quick", action="store_true", help="key predictor; separate quick outputs")
    parser.add_argument(
        "--centroid-membership-quick",
        action="store_true",
        help="quick-only centroid membership; canonical default is exact area overlap",
    )
    parser.add_argument("--field-permutations", type=int, default=199)
    parser.add_argument("--field-all-predictors", action="store_true")
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help="secondary dimensional-reduction sensitivity; canonical run uses all 31 variables",
    )
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.quick and args.selected_only:
        parser.error("--quick and --selected-only are mutually exclusive")
    if args.centroid_membership_quick and not args.quick:
        parser.error("--centroid-membership-quick is allowed only with --quick")
    if args.field_permutations < 19:
        parser.error("field significance requires at least 19 permutations")
    if not args.quick and not args.selected_only and args.field_permutations < 199:
        parser.error("canonical F4C requires at least 199 field permutations")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    STATS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    target, response, pixels, grid_hash, target_contract = load_native_response()
    geometry_contract = ibge_geometry_contract()
    analysis_contract = {
        **target_contract,
        **geometry_contract,
        "f4c_runner_sha256": sha256_file(Path(__file__).resolve()),
        "lag_analysis_module_sha256": sha256_file(
            ROOT / "src/nino_brasil/stats/lag_analysis.py"
        ),
    }
    target.close()
    phase_table_path = phase_table_path_for_enso_type(args.enso_type)
    phases, parent_f3_run_id, parent_f3_artifact_sha256 = load_canonical_phase_table(
        response.index,
        phase_table_path=phase_table_path,
    )
    scope_prefix = f"{args.enso_type.upper()}_" if args.enso_type else ""
    run_prefix = (
        f"F4C_{scope_prefix}QUICK_"
        if args.quick
        else f"F4C_{scope_prefix}SENSITIVITY_"
        if args.selected_only
        else f"F4C_{scope_prefix}"
    )
    run_id = (
        run_prefix
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + grid_hash[:8]
        + "_"
        + target_contract["target_block_signature_sha256"][:8]
        + "_"
        + parent_f3_run_id[-8:]
        + "_"
        + uuid.uuid4().hex[:6]
    )
    master = pd.read_csv(
        FEAT / "nino34_master_weekly.csv", parse_dates=["week_ending_sunday"]
    ).set_index("week_ending_sunday")
    conditions = official_source_conditions(phases, enso_type=args.enso_type)

    available = [column for column in master.columns if master[column].notna().any()]
    if args.quick:
        predictor_names, selection_contract = [KEY_PREDICTOR], "quick:key-predictor"
    elif args.selected_only:
        predictor_names, selection_contract = load_selected_predictors(
            STATS, available, allow_all_fallback=True
        )
        selection_contract = "secondary_sensitivity:" + selection_contract
    else:
        missing_predictors = [name for name in PACIFIC_VARS if name not in master]
        if missing_predictors:
            raise KeyError(
                "Canonical Phase 4C requires all 31 Pacific variables; missing "
                f"{missing_predictors}"
            )
        predictor_names = list(PACIFIC_VARS)
        selection_contract = "canonical_all_31_physical_variables"
    if KEY_PREDICTOR not in predictor_names:
        predictor_names = [KEY_PREDICTOR, *predictor_names]
    predictors = master[
        [column for column in dict.fromkeys(predictor_names) if column in master]
    ].reindex(response.index)
    deseasonalized, predictor_metadata = harmonic_deseasonalize_predictors(predictors)
    predictor_names_used = list(deseasonalized.columns)
    analysis_contract.update(
        {
            "enso_type": args.enso_type or "combined",
            "selection_contract": selection_contract,
            "predictor_count": len(predictor_names_used),
            "predictor_catalog_sha256": predictor_catalog_sha256(predictor_names_used),
        }
    )
    predictor_metadata["selection_contract"] = selection_contract
    predictor_metadata["grid_hash_sha256"] = grid_hash
    predictor_metadata["analysis_run_id"] = run_id
    predictor_metadata["parent_f3_run_id"] = parent_f3_run_id
    predictor_metadata["parent_f3_artifact_sha256"] = parent_f3_artifact_sha256
    for column, value in analysis_contract.items():
        predictor_metadata[column] = value

    units = load_units()
    membership = load_membership(
        units,
        pixels,
        centroid_quick=args.centroid_membership_quick,
        grid_hash=grid_hash,
        geometry_contract=geometry_contract,
        replace_stale=args.replace_existing,
    )
    unit_series, coverage = aggregate_area_weighted_response(response, membership)
    unit_series = unit_series.loc[:, unit_series.notna().any()]
    print(
        f"[4C] predictors={len(deseasonalized.columns)} | conditions={len(conditions)} | "
        f"native Brazil-overlap pixels={len(pixels)} | units={unit_series.shape[1]}"
    )

    if args.quick:
        output_dir = STATS / "quick"
    elif args.selected_only:
        output_dir = STATS / "sensitivity"
    else:
        output_dir = STATS
    output_dir.mkdir(parents=True, exist_ok=True)
    unit_lags = compute_unit_lags(
        deseasonalized,
        unit_series,
        phases,
        conditions,
        fdr_alpha=FDR_ALPHA,
    )
    unit_lags["nome_unidade"] = unit_lags["id_unidade"].map(
        units.set_index("id_unidade")["nome_unidade"]
    )
    unit_lags["tipo_unidade"] = unit_lags["id_unidade"].map(
        units.set_index("id_unidade")["tipo_unidade"]
    )
    unit_lags["grid_hash_sha256"] = grid_hash
    unit_lags["analysis_run_id"] = run_id
    unit_lags["parent_f3_run_id"] = parent_f3_run_id
    unit_lags["parent_f3_artifact_sha256"] = parent_f3_artifact_sha256
    for column, value in analysis_contract.items():
        unit_lags[column] = value
    unit_lags_path = scoped_artifact_path(
        output_dir / "phase4C_native_lags_por_unidade.csv", args.enso_type
    )
    unit_lags.to_csv(unit_lags_path, index=False)
    coverage_output = coverage.copy()
    coverage_output.index.name = "week_ending_sunday"
    coverage_output = coverage_output.reset_index()
    coverage_output["analysis_run_id"] = run_id
    coverage_output["parent_f3_run_id"] = parent_f3_run_id
    coverage_output["parent_f3_artifact_sha256"] = parent_f3_artifact_sha256
    for column, value in analysis_contract.items():
        coverage_output[column] = value
    coverage_path = scoped_artifact_path(
        output_dir / "phase4C_native_cobertura_unidades.parquet", args.enso_type
    )
    coverage_output.to_parquet(coverage_path, index=False)
    predictor_treatment_path = scoped_artifact_path(
        output_dir / "phase4C_native_predictor_treatment.csv", args.enso_type
    )
    predictor_metadata.to_csv(predictor_treatment_path, index=False)

    atlas_base_path = (
        ATLAS.parent / "quick" / ATLAS.name
        if args.quick
        else ATLAS.parent / "sensitivity" / ATLAS.name
        if args.selected_only
        else ATLAS
    )
    atlas_path = scoped_artifact_path(atlas_base_path, args.enso_type)
    best_pixels = compute_pixel_atlas(
        deseasonalized,
        response,
        pixels,
        phases,
        conditions,
        destination=atlas_path,
        replace_existing=args.replace_existing,
        grid_hash=grid_hash,
        run_id=run_id,
        contract_metadata={
            **analysis_contract,
            "parent_f3_run_id": parent_f3_run_id,
            "parent_f3_artifact_sha256": parent_f3_artifact_sha256,
        },
        fdr_alpha=FDR_ALPHA,
    )
    best_pixels["analysis_run_id"] = run_id
    best_pixels["grid_hash_sha256"] = grid_hash
    best_pixels["parent_f3_run_id"] = parent_f3_run_id
    best_pixels["parent_f3_artifact_sha256"] = parent_f3_artifact_sha256
    for column, value in analysis_contract.items():
        best_pixels[column] = value
    best_pixels_path = scoped_artifact_path(
        output_dir / "phase4C_native_best_lag_pixel.parquet", args.enso_type
    )
    best_pixels.to_parquet(best_pixels_path, index=False)
    key = best_pixels[best_pixels["variavel"].eq(KEY_PREDICTOR)]
    best_lag_key_path = scoped_artifact_path(
        output_dir / "phase4C_native_best_lag_pixel_key.csv", args.enso_type
    )
    key.to_csv(best_lag_key_path, index=False)
    peak_lag_regions = summarize_peak_pixel_lags_by_region(key, membership)
    peak_lag_regions["analysis_run_id"] = run_id
    peak_lag_regions["grid_hash_sha256"] = grid_hash
    peak_lag_regions["parent_f3_run_id"] = parent_f3_run_id
    peak_lag_regions["parent_f3_artifact_sha256"] = parent_f3_artifact_sha256
    for column, value in analysis_contract.items():
        peak_lag_regions[column] = value
    peak_lag_regions_path = scoped_artifact_path(
        output_dir / "phase4C_native_peak_lag_regional_summary.csv",
        args.enso_type,
    )
    peak_lag_regions.to_csv(peak_lag_regions_path, index=False)

    field = compute_regional_field_significance(
        deseasonalized,
        unit_series,
        units,
        phases,
        conditions,
        n_permutations=args.field_permutations,
        all_predictors=args.field_all_predictors,
    )
    field["grid_hash_sha256"] = grid_hash
    field["analysis_run_id"] = run_id
    field["parent_f3_run_id"] = parent_f3_run_id
    field["parent_f3_artifact_sha256"] = parent_f3_artifact_sha256
    for column, value in analysis_contract.items():
        field[column] = value
    field_significance_path = scoped_artifact_path(
        output_dir / "phase4C_native_field_significance.csv", args.enso_type
    )
    field.to_csv(field_significance_path, index=False)
    target_manifest = target_build_manifest_path(target_contract)
    manifest_inputs = [
        FEAT / "nino34_master_weekly.csv",
        phase_table_path,
        Path(f"{phase_table_path}.manifest.json"),
        target_manifest,
        TARGET_VARIABLE_CONTRACT,
        TARGET_PIXEL_INVENTORY,
        MEMBERSHIP_CENTROID_QUICK if args.centroid_membership_quick else MEMBERSHIP_EXACT,
        *ibge_geometry_input_paths(),
        Path(__file__).resolve(),
        ROOT / "src/nino_brasil/stats/lag_analysis.py",
    ]
    sidecar_stage = (
        "F4C_QUICK"
        if args.quick
        else "F4C_SENSITIVITY"
        if args.selected_only
        else "F4C"
    )
    sidecar_contract = {
        **analysis_contract,
        "parent_f3_run_id": parent_f3_run_id,
        "parent_f3_artifact_sha256": parent_f3_artifact_sha256,
        "grid_hash_sha256": grid_hash,
        "selection_contract": selection_contract,
        "enso_type": args.enso_type or "combined",
        "predictor_names": predictor_names_used,
        "field_permutations": args.field_permutations,
    }
    write_phase4_csv_manifests(
        [
            (unit_lags_path, unit_lags),
            (coverage_path, coverage_output),
            (predictor_treatment_path, predictor_metadata),
            (best_pixels_path, best_pixels),
            (best_lag_key_path, key),
            (peak_lag_regions_path, peak_lag_regions),
            (field_significance_path, field),
        ],
        run_id=run_id,
        stage=sidecar_stage,
        contract=sidecar_contract,
        inputs=manifest_inputs,
    )
    write_phase4_directory_manifest(
        atlas_path,
        run_id=run_id,
        stage=sidecar_stage,
        contract=sidecar_contract,
        inputs=manifest_inputs,
    )
    # Os modos oficiais separados publicam as figuras somente pelo notebook
    # canônico. Isso evita recriar nomes legados e mantém cada FigF4* ligada
    # ao seu par TabF4* no manifesto público.
    if args.enso_type is None:
        save_figures(
            unit_lags,
            best_pixels,
            field,
            pixels,
            units,
            quick=args.quick,
            sensitivity=args.selected_only,
            enso_type=args.enso_type,
            run_id=run_id,
            source_paths={
                "unit_lags": unit_lags_path,
                "best_pixels": best_pixels_path,
                "field_significance": field_significance_path,
            },
            fdr_alpha=FDR_ALPHA,
        )
    else:
        print("[4C] figuras públicas serão publicadas pelo notebook canônico")
    print(
        f"[4C] complete | target grid sha256={grid_hash} | "
        f"confirmatory BH alpha={FDR_ALPHA:.2f} | run_id={run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
