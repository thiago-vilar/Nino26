#!/usr/bin/env python3
"""Fase 4D: descriptive native-pixel targets and event-jackknife hypothesis gate.

Clusters summarize F4C response profiles but never establish significance.
Inference is performed on official IBGE-region series and uses: the F4C
lag/pixel BH contract, whole-field significance across the five non-overlapping
regions, a second BH family across confirmatory precipitation/extreme targets,
and leave-one-entire-ENSO-event-out sign stability.  There is no fixed calendar
split and no response pixel is interpolated.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from nino_brasil.maps.figure_registry import save_registered_figure  # noqa: E402
from nino_brasil.artifacts import sha256_file  # noqa: E402
from nino_brasil.config import confirmatory_fdr_alpha  # noqa: E402
from nino_brasil.maps.plot_pixel_maps import plot_pixel_field  # noqa: E402
from nino_brasil.maps.spatial_support import (  # noqa: E402
    aggregate_area_weighted_response,
    build_analysis_units,
    load_ibge_biomes,
    load_ibge_regions,
)
from nino_brasil.stats.lag_analysis import (  # noqa: E402
    PHASE_ORDER,
    best_from_long_table,
    build_source_conditions,
    fdr_bh_adjusted,
    harmonic_deseasonalize_predictors,
    lagged_correlation_exact,
)
from nino_brasil.targets.chirps_native import (  # noqa: E402
    target_to_frame,
    validate_native_target,
)
from nino_brasil.viz import registrar_figura  # noqa: E402
from scripts.run_fase4c_regional import (  # noqa: E402
    ENSO_TYPES,
    PACIFIC_VARS,
    conditions_for_enso_type,
    ibge_geometry_contract,
    ibge_geometry_input_paths,
    load_canonical_phase_table,
    native_target_contract,
    phase_table_path_for_enso_type,
    predictor_catalog_sha256,
    scoped_artifact_path,
    target_build_manifest_path,
    verify_phase4_output_manifest,
    write_phase4_csv_manifests,
)


FEAT = ROOT / "data/processed/parquet/features"
STATS = ROOT / "data/processed/parquet/statistics"
FIGS = ROOT / "data/processed/figures/fase4"
TARGET = ROOT / "data/processed/zarr/features/chirps_native_weekly_targets.zarr"
PHASES = STATS / "phase3_fases_semanais_en_ln.csv"
BEST_PIXELS = STATS / "phase4C_native_best_lag_pixel.parquet"
UNIT_LAGS = STATS / "phase4C_native_lags_por_unidade.csv"
FIELD = STATS / "phase4C_native_field_significance.csv"
ATLAS = ROOT / "data/processed/zarr/statistics/phase4C_native_pixel_lags.zarr"
MEMBERSHIP = STATS / "phase4C_native_pixel_membership_exact.parquet"
TARGET_VARIABLE_CONTRACT = STATS / "phase4_chirps_target_variable_contract.csv"
TARGET_PIXEL_INVENTORY = FEAT / "phase4_chirps_native_pixels.csv"
REG_SHP = ROOT / "data/interim/ibge/BR_Regioes_2024/BR_Regioes_2024.shp"
BIO_SHP = ROOT / "data/interim/ibge/Biomas_2025/lml_bioma_e250k_v20250911_A.shp"
KEY_PREDICTOR = "nino34_ssta"
FDR_ALPHA = confirmatory_fdr_alpha(fallback=0.05)
OFFICIAL_CONDITIONS = tuple(
    f"{event_type}_{phase}"
    for event_type in ("el_nino", "la_nina")
    for phase in PHASE_ORDER
)


def f4c_artifact_paths(enso_type: str | None) -> dict[str, Path]:
    return {
        "best_pixels": scoped_artifact_path(BEST_PIXELS, enso_type),
        "unit_lags": scoped_artifact_path(UNIT_LAGS, enso_type),
        "field": scoped_artifact_path(FIELD, enso_type),
        "coverage": scoped_artifact_path(
            STATS / "phase4C_native_cobertura_unidades.parquet", enso_type
        ),
        "predictor_treatment": scoped_artifact_path(
            STATS / "phase4C_native_predictor_treatment.csv", enso_type
        ),
        "best_lag_key": scoped_artifact_path(
            STATS / "phase4C_native_best_lag_pixel_key.csv", enso_type
        ),
        "peak_lag_regions": scoped_artifact_path(
            STATS / "phase4C_native_peak_lag_regional_summary.csv", enso_type
        ),
        "atlas": scoped_artifact_path(ATLAS, enso_type),
    }
CONFIRMATORY_TARGETS = {
    "precip_robust_z": {
        "label": "weekly precipitation anomaly",
        "family": "weekly_total_precipitation",
        "expected_multiplier": 1,
    },
    "spi_gamma_3m_weekly_origin": {
        "label": "SPI 3-month weekly origin",
        "family": "accumulated_meteorological_drought",
        "expected_multiplier": 1,
    },
    "r95p_weekly_robust_z": {
        "label": "R95p weekly anomaly",
        "family": "heavy_rainfall_extremes",
        "expected_multiplier": 1,
    },
    "r99p_weekly_robust_z": {
        "label": "R99p weekly anomaly",
        "family": "heavy_rainfall_extremes",
        "expected_multiplier": 1,
    },
    "cdd_within_week_robust_z": {
        "label": "dry-spell anomaly",
        "family": "dry_spell_persistence",
        "expected_multiplier": -1,
    },
}


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype("string").str.lower().isin({"true", "1", "yes", "sim"})


def load_contracts(*, enso_type: str | None = None):
    f4c_paths = f4c_artifact_paths(enso_type)
    required = (
        TARGET,
        f4c_paths["best_pixels"],
        f4c_paths["unit_lags"],
        f4c_paths["field"],
        MEMBERSHIP,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Phase 4D requires completed native Phase 4C outputs: " + ", ".join(missing)
        )
    target = xr.open_zarr(TARGET, consolidated=None)
    validation = validate_native_target(target, deep=False)
    if not validation.valid:
        raise ValueError(f"Invalid native CHIRPS target: {validation.errors}")
    if target.attrs.get("deep_validation_passed") is not True:
        raise ValueError("Native target lacks the builder's deep-validation stamp.")
    target_contract = native_target_contract(target)
    missing_targets = set(CONFIRMATORY_TARGETS).difference(target.data_vars)
    if missing_targets:
        raise ValueError(
            "Native target lacks preregistered F4D target layers: "
            f"{sorted(missing_targets)}. Rebuild with SPI/extremes enabled."
        )
    best = pd.read_parquet(f4c_paths["best_pixels"])
    unit_lags = pd.read_csv(f4c_paths["unit_lags"])
    field = pd.read_csv(f4c_paths["field"])
    membership = pd.read_parquet(MEMBERSHIP)
    expected_implementation = {
        "f4c_runner_sha256": sha256_file(ROOT / "scripts/run_fase4c_regional.py"),
        "lag_analysis_module_sha256": sha256_file(
            ROOT / "src/nino_brasil/stats/lag_analysis.py"
        ),
    }
    expected_geometry = ibge_geometry_contract()
    expected_implementation.update(expected_geometry)
    expected_implementation.update(
        {
            "enso_type": enso_type or "combined",
            "selection_contract": "canonical_all_31_physical_variables",
            "predictor_count": 31,
            "predictor_catalog_sha256": predictor_catalog_sha256(PACIFIC_VARS),
        }
    )
    for table_name, table in (
        ("best pixels", best),
        ("unit lags", unit_lags),
        ("field significance", field),
        ("membership", membership),
    ):
        if "grid_hash_sha256" in table:
            hashes = set(table["grid_hash_sha256"].dropna().astype(str))
        elif "grid_hash" in table:
            hashes = set(table["grid_hash"].dropna().astype(str))
        else:
            hashes = set()
        if hashes != {validation.grid_hash}:
            raise ValueError(f"{table_name} belongs to another CHIRPS grid: {hashes}")
        # Geometry membership is reusable only while both the native grid and
        # the complete IBGE region/biome bundles retain the same fingerprints.
        if table_name == "membership":
            for column, expected in expected_geometry.items():
                if column not in table:
                    raise ValueError(
                        f"membership lacks {column}; rebuild F4C membership."
                    )
                values = set(table[column].dropna().astype(str).str.strip())
                values.discard("")
                if values != {expected}:
                    raise ValueError(
                        "membership belongs to different IBGE geometry: "
                        f"{column}={values}, expected {expected!r}."
                    )
            continue
        for column, expected in expected_implementation.items():
            if column not in table:
                raise ValueError(
                    f"{table_name} lacks {column}; rerun F4C with code fingerprinting."
                )
            values = set(table[column].dropna().astype(str).str.strip())
            values.discard("")
            if values != {str(expected)}:
                raise ValueError(
                    f"{table_name} was produced by different F4C code: "
                    f"{column}={values}, expected {expected!r}."
                )
        for column, expected in target_contract.items():
            if column not in table:
                raise ValueError(
                    f"{table_name} lacks {column}; rerun canonical F4C against this target."
                )
            values = set(table[column].dropna().astype(str).str.strip())
            values.discard("")
            if values != {expected}:
                raise ValueError(
                    f"{table_name} belongs to another target build: "
                    f"{column}={values}, expected {expected!r}."
                )
    return (
        target,
        validation,
        best,
        unit_lags,
        field,
        membership,
        target_contract,
        f4c_paths,
    )


def analysis_units():
    return build_analysis_units(load_ibge_regions(REG_SHP), load_ibge_biomes(BIO_SHP))


def verify_canonical_f4c_output(
    path: Path,
    *,
    expected_run_id: str,
    expected_enso_type: str | None = None,
) -> dict[str, object]:
    """Accept only the all-31, >=199-permutation canonical F4C namespace."""

    manifest = verify_phase4_output_manifest(path, expected_run_id=expected_run_id)
    contract = manifest.get("contract") or {}
    canonical_catalog = list(PACIFIC_VARS)
    if str(contract.get("stage") or "") != "F4C":
        raise ValueError(f"F4D refuses non-canonical F4C stage for {path}")
    if contract.get("selection_contract") != "canonical_all_31_physical_variables":
        raise ValueError(f"F4D requires the canonical all-31 predictor contract: {path}")
    if expected_enso_type is not None and contract.get("enso_type") != expected_enso_type:
        raise ValueError(
            f"F4D expected F4C scope {expected_enso_type!r}, got "
            f"{contract.get('enso_type')!r}: {path}"
        )
    try:
        predictor_count = int(contract.get("predictor_count"))
        field_permutations = int(contract.get("field_permutations"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid F4C numeric contract in {path}") from exc
    if predictor_count != len(canonical_catalog):
        raise ValueError(f"F4D requires exactly 31 predictors: {path}")
    if list(contract.get("predictor_names") or []) != canonical_catalog:
        raise ValueError(f"F4D predictor catalog/order mismatch: {path}")
    if contract.get("predictor_catalog_sha256") != predictor_catalog_sha256(
        canonical_catalog
    ):
        raise ValueError(f"F4D predictor catalog hash mismatch: {path}")
    if field_permutations < 199:
        raise ValueError(f"F4D requires at least 199 F4C field permutations: {path}")
    return manifest


def cluster_native_pixels(
    best: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    conditions: tuple[str, ...] = OFFICIAL_CONDITIONS,
    min_significant_profiles: int = 2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "pixel_id",
        "lat",
        "lon",
        "variavel",
        "condicao_fonte",
        "r_no_best_lag_fdr",
        "best_lag_sem_fdr",
    }
    missing = required.difference(best.columns)
    if missing:
        raise KeyError(f"F4C best-pixel table is missing {sorted(missing)}")
    work = best[best["condicao_fonte"].isin(conditions)].copy()
    work["feature"] = work["variavel"].astype(str) + "|" + work["condicao_fonte"].astype(str)
    r = work.pivot_table(
        index="pixel_id", columns="feature", values="r_no_best_lag_fdr", aggfunc="first"
    )
    lag = work.pivot_table(
        index="pixel_id", columns="feature", values="best_lag_sem_fdr", aggfunc="first"
    )
    significant_count = r.notna().sum(axis=1)
    eligible = significant_count >= min_significant_profiles
    r_feature = r.fillna(0.0)
    lag_feature = (lag / 78.0).where(r.notna()).fillna(0.0)
    present_feature = r.notna().astype(float) * 0.25
    matrix = np.column_stack(
        [r_feature.loc[eligible].to_numpy(), lag_feature.loc[eligible].to_numpy(), present_feature.loc[eligible].to_numpy()]
    )
    if matrix.shape[0] < 50:
        clusters = work[["pixel_id", "lat", "lon"]].drop_duplicates("pixel_id")
        clusters["cluster"] = -1
        clusters["n_perfis_fdr"] = clusters["pixel_id"].map(significant_count).fillna(0).astype(int)
        clusters["cluster_status"] = "insufficient_FDR_profiles_for_defensible_clustering"
        region_membership = membership[
            membership["tipo_unidade"].eq("regiao")
            & membership["centro_na_unidade"].astype(bool)
        ][["pixel_id", "nome_unidade"]].drop_duplicates("pixel_id")
        clusters = clusters.merge(region_membership, on="pixel_id", how="left").rename(
            columns={"nome_unidade": "regiao_centro_pixel"}
        )
        clusters["k_escolhido"] = 0
        clusters["silhueta_amostra"] = np.nan
        profiles = pd.DataFrame(
            columns=[
                "cluster",
                "variavel",
                "condicao_fonte",
                "n_pixels",
                "fracao_pixel_lag_fdr",
                "r_mediano_fdr",
                "lag_mediano_fdr",
            ]
        )
        ranking = pd.DataFrame(
            columns=[
                "cluster",
                "n_pixels",
                "latitude_media",
                "longitude_media",
                "regiao_predominante",
                "silhueta_amostra",
                "abs_r_fdr_medio",
                "prioridade_descritiva",
            ]
        )
        return clusters, profiles, ranking
    rng = np.random.default_rng(seed)
    sample_index = rng.choice(matrix.shape[0], size=min(3000, matrix.shape[0]), replace=False)
    sample = matrix[sample_index]
    scores: dict[int, float] = {}
    models: dict[int, MiniBatchKMeans] = {}
    for n_clusters in range(3, 8):
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed + n_clusters,
            n_init=10,
            batch_size=1024,
        ).fit(sample)
        scores[n_clusters] = float(
            silhouette_score(
                sample,
                model.labels_,
                sample_size=min(1500, len(sample)),
                random_state=seed,
            )
        )
        models[n_clusters] = model
    chosen = max(scores, key=scores.get)
    model = MiniBatchKMeans(
        n_clusters=chosen,
        random_state=seed + 100,
        n_init=1,
        batch_size=2048,
        init=models[chosen].cluster_centers_,
    ).fit(matrix)

    coordinates = work[["pixel_id", "lat", "lon"]].drop_duplicates("pixel_id")
    clusters = coordinates.set_index("pixel_id")
    clusters["cluster"] = -1
    clusters.loc[r.index[eligible], "cluster"] = model.labels_.astype(int)
    clusters["n_perfis_fdr"] = significant_count.reindex(clusters.index).fillna(0).astype(int)
    clusters["cluster_status"] = np.where(
        clusters["cluster"] >= 0, "descriptive_cluster", "insufficient_FDR_profiles"
    )
    region_membership = membership[
        membership["tipo_unidade"].eq("regiao")
        & membership["centro_na_unidade"].astype(bool)
    ][["pixel_id", "nome_unidade"]].drop_duplicates("pixel_id")
    clusters = clusters.reset_index().merge(region_membership, on="pixel_id", how="left")
    clusters = clusters.rename(columns={"nome_unidade": "regiao_centro_pixel"})
    clusters["k_escolhido"] = chosen
    clusters["silhueta_amostra"] = scores[chosen]

    labelled = work.merge(clusters[["pixel_id", "cluster"]], on="pixel_id", how="left")
    profiles = (
        labelled[labelled["cluster"] >= 0]
        .groupby(["cluster", "variavel", "condicao_fonte"], as_index=False)
        .agg(
            n_pixels=("pixel_id", "nunique"),
            fracao_pixel_lag_fdr=("best_lag_sem_fdr", lambda x: float(x.notna().mean())),
            r_mediano_fdr=("r_no_best_lag_fdr", "median"),
            lag_mediano_fdr=("best_lag_sem_fdr", "median"),
        )
    )
    ranking = (
        clusters[clusters["cluster"] >= 0]
        .groupby("cluster", as_index=False)
        .agg(
            n_pixels=("pixel_id", "size"),
            latitude_media=("lat", "mean"),
            longitude_media=("lon", "mean"),
            regiao_predominante=(
                "regiao_centro_pixel",
                lambda x: x.mode().iloc[0] if not x.mode().empty else "",
            ),
            silhueta_amostra=("silhueta_amostra", "first"),
        )
    )
    effect = (
        labelled[labelled["cluster"] >= 0]
        .groupby("cluster")["r_no_best_lag_fdr"]
        .apply(lambda values: float(values.abs().mean()))
        .rename("abs_r_fdr_medio")
    )
    ranking = ranking.merge(effect, on="cluster", how="left").sort_values(
        "abs_r_fdr_medio", ascending=False
    )
    ranking["prioridade_descritiva"] = np.arange(1, len(ranking) + 1)
    return clusters, profiles, ranking


def regional_target_series(
    target: xr.Dataset,
    membership: pd.DataFrame,
    units,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    region_ids = units.loc[units["tipo_unidade"].eq("regiao"), "id_unidade"].tolist()
    output: dict[str, pd.DataFrame] = {}
    coverage_rows: list[pd.DataFrame] = []
    for variable in CONFIRMATORY_TARGETS:
        if variable not in target:
            raise KeyError(
                f"Canonical target lacks {variable}; rebuild with SPI/extreme layers enabled."
            )
        frame, _ = target_to_frame(
            target,
            variable=variable,
            brazil_only=True,
            mask_rule="overlap",
        )
        series, coverage = aggregate_area_weighted_response(frame, membership)
        available_regions = [column for column in region_ids if column in series]
        output[variable] = series[available_regions]
        regional_coverage = coverage[available_regions].copy()
        regional_coverage.index.name = "week_ending_sunday"
        long_coverage = regional_coverage.reset_index().melt(
            id_vars="week_ending_sunday",
            var_name="id_unidade",
            value_name="valid_area_fraction",
        )
        long_coverage.insert(0, "target_chirps", variable)
        long_coverage["minimum_valid_area_fraction"] = 0.80
        long_coverage["coverage_gate_pass"] = (
            long_coverage["valid_area_fraction"] >= 0.80
        )
        coverage_rows.append(long_coverage)
    return output, pd.concat(coverage_rows, ignore_index=True)


def _expected_direction(event_type: str, region_name: str, multiplier: int) -> int:
    """Expected correlation sign against the signed Niño-3.4 SSTA predictor.

    La Niña reverses both the predictor sign and the expected rainfall anomaly,
    so their correlation slope keeps the same sign as El Niño.  An event-type
    flip would be appropriate only if the predictor were unsigned event
    intensity, which it is not in this analysis.
    """

    if event_type not in {"el_nino", "la_nina"}:
        raise ValueError(f"Unsupported ENSO event type: {event_type!r}")
    if region_name not in {"Nordeste", "Sul"}:
        return 0
    base = -1 if region_name == "Nordeste" else 1
    return int(base * multiplier)


def event_jackknife(
    predictor: pd.Series,
    response: pd.DataFrame,
    phase_table: pd.DataFrame,
    condition,
    *,
    lag: int,
) -> dict[str, object]:
    active_events = phase_table.loc[
        condition.source_mask.reindex(phase_table.index).fillna(False), "event_id"
    ].astype(str)
    event_ids = [event for event in pd.unique(active_events) if event]
    values: list[float] = []
    for event_id in event_ids:
        keep = condition.source_mask & phase_table["event_id"].astype(str).ne(event_id)
        leave_one_out = replace(
            condition,
            name=f"{condition.name}|leave_out={event_id}",
            source_mask=keep,
            description=condition.description + f"; leave out entire {event_id}",
        )
        result = lagged_correlation_exact(
            predictor,
            response,
            [lag],
            leave_one_out,
            phase_table,
        )
        value = float(result["r"][0, 0])
        if np.isfinite(value):
            values.append(value)
    array = np.asarray(values, dtype=float)
    return {
        "n_eventos_total": len(event_ids),
        "n_jackknife_valido": len(array),
        "jackknife_r_mediano": float(np.median(array)) if len(array) else np.nan,
        "jackknife_r_p10": float(np.quantile(array, 0.10)) if len(array) else np.nan,
        "jackknife_r_p90": float(np.quantile(array, 0.90)) if len(array) else np.nan,
        "jackknife_values": array,
    }


def build_gate(
    target_series: dict[str, pd.DataFrame],
    unit_lags: pd.DataFrame,
    field: pd.DataFrame,
    units,
    predictor: pd.Series,
    phase_table: pd.DataFrame,
    conditions: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    regions = units[units["tipo_unidade"].eq("regiao")]
    region_names = regions.set_index("id_unidade")["nome_unidade"]
    primary = unit_lags[
        unit_lags["variavel"].eq(KEY_PREDICTOR)
        & unit_lags["condicao_fonte"].isin(tuple(conditions))
        & unit_lags["id_unidade"].isin(region_names.index)
    ].copy()
    primary_best = best_from_long_table(
        primary,
        group_columns=["id_unidade", "condicao_fonte"],
        require_fdr=True,
    )
    rows: list[dict[str, object]] = []
    selected_lookup = primary_best.set_index(["id_unidade", "condicao_fonte"])
    for unit_id in region_names.index:
        for condition_name, condition in conditions.items():
            key = (unit_id, condition_name)
            region_name = str(region_names.loc[unit_id])
            if key not in selected_lookup.index:
                for target_name, target_meta in CONFIRMATORY_TARGETS.items():
                    expected = _expected_direction(
                        condition.tipo_enso_fonte,
                        region_name,
                        int(target_meta["expected_multiplier"]),
                    )
                    rows.append(
                        {
                            "id_unidade": unit_id,
                            "regiao": region_name,
                            "condicao_fonte": condition_name,
                            "tipo_enso_fonte": condition.tipo_enso_fonte,
                            "fase_fonte_em_t_menos_lag": condition.fase_fonte_em_t_menos_lag,
                            "variavel_pacifico": KEY_PREDICTOR,
                            "target_chirps": target_name,
                            "target_label": target_meta["label"],
                            "target_family": target_meta["family"],
                            "lag_sem_selecionado_F4C": np.nan,
                            "r": np.nan,
                            "p": np.nan,
                            "n_eff_bretherton": np.nan,
                            "p_primary_precip_field_max_lag_F4C": np.nan,
                            "direcao_esperada": expected,
                            "direcao_observada": 0,
                            "direcao_coerente": False,
                            "estabilidade_sinal_jackknife": np.nan,
                            "n_eventos_total": int(
                                phase_table.loc[condition.source_mask, "event_id"]
                                .astype(str)
                                .replace("", np.nan)
                                .nunique()
                            ),
                            "n_jackknife_valido": 0,
                            "jackknife_r_mediano": np.nan,
                            "jackknife_r_p10": np.nan,
                            "jackknife_r_p90": np.nan,
                            "primary_F4C_status": "no_lag_passed_F4C_BH",
                        }
                    )
                continue
            selected = selected_lookup.loc[key]
            if isinstance(selected, pd.DataFrame):
                selected = selected.sort_values("lag_sem", kind="mergesort").iloc[0]
            lag = int(selected["lag_sem"])
            field_row = field[
                field["variavel"].eq(KEY_PREDICTOR)
                & field["condicao_fonte"].eq(condition_name)
                & field["lag_sem"].eq(lag)
            ]
            field_p = (
                float(field_row["p_field_max_lag"].iloc[0]) if len(field_row) else np.nan
            )
            for target_name, target_meta in CONFIRMATORY_TARGETS.items():
                response = target_series[target_name][[unit_id]]
                result = lagged_correlation_exact(
                    predictor,
                    response,
                    [lag],
                    condition,
                    phase_table,
                )
                r = float(result["r"][0, 0])
                p = float(result["p"][0, 0])
                n_eff = float(result["n_eff"][0, 0])
                jackknife = event_jackknife(
                    predictor,
                    response,
                    phase_table,
                    condition,
                    lag=lag,
                )
                values = jackknife.pop("jackknife_values")
                sign_stability = (
                    float(np.mean(np.sign(values) == np.sign(r)))
                    if len(values) and np.isfinite(r) and r != 0
                    else np.nan
                )
                event_type = condition.tipo_enso_fonte
                expected = _expected_direction(
                    event_type, region_name, int(target_meta["expected_multiplier"])
                )
                rows.append(
                    {
                        "id_unidade": unit_id,
                        "regiao": region_name,
                        "condicao_fonte": condition_name,
                        "tipo_enso_fonte": event_type,
                        "fase_fonte_em_t_menos_lag": condition.fase_fonte_em_t_menos_lag,
                        "variavel_pacifico": KEY_PREDICTOR,
                        "target_chirps": target_name,
                        "target_label": target_meta["label"],
                        "target_family": target_meta["family"],
                        "lag_sem_selecionado_F4C": lag,
                        "r": r,
                        "p": p,
                        "n_eff_bretherton": n_eff,
                        "p_primary_precip_field_max_lag_F4C": field_p,
                        "direcao_esperada": expected,
                        "direcao_observada": int(np.sign(r)) if np.isfinite(r) else 0,
                        "direcao_coerente": bool(expected == 0 or np.sign(r) == expected),
                        "estabilidade_sinal_jackknife": sign_stability,
                        "primary_F4C_status": "lag_passed_F4C_BH",
                        **jackknife,
                    }
                )
    gate = pd.DataFrame(rows)
    gate["expected_direction_contract"] = (
        "slope versus signed nino34_ssta; El Nino and La Nina keep the same "
        "correlation sign because predictor and response signs both reverse"
    )
    gate["q_F4D_global_targets_regioes_fases"] = fdr_bh_adjusted(
        gate["p"].to_numpy(dtype=float)
    )
    gate["fdr_alpha_confirmatory"] = FDR_ALPHA
    gate["fdr_F4D_confirmatory"] = (
        gate["q_F4D_global_targets_regioes_fases"] < FDR_ALPHA
    )
    gate["primary_precip_field_significant_F4C"] = (
        gate["p_primary_precip_field_max_lag_F4C"] < FDR_ALPHA
    )
    gate["event_stable_0_80"] = (
        (gate["n_jackknife_valido"] >= 5)
        & (gate["estabilidade_sinal_jackknife"] >= 0.80)
    )
    gate["finite_primary_test"] = np.isfinite(gate["r"]) & np.isfinite(gate["p"])
    in_hypothesis = gate["regiao"].isin(["Nordeste", "Sul"])
    gate["gate_supports_direction"] = (
        in_hypothesis
        & gate["direcao_coerente"]
        & gate["fdr_F4D_confirmatory"]
        & gate["primary_precip_field_significant_F4C"]
        & gate["event_stable_0_80"]
    )
    gate["gate_status"] = np.select(
        [
            ~in_hypothesis,
            gate["primary_F4C_status"].eq("no_lag_passed_F4C_BH"),
            ~gate["finite_primary_test"],
            ~gate["fdr_F4D_confirmatory"],
            ~gate["primary_precip_field_significant_F4C"],
            ~gate["event_stable_0_80"],
            ~gate["direcao_coerente"],
            gate["gate_supports_direction"],
        ],
        [
            "exploratory_other_region",
            "fails_primary_F4C_BH",
            "insufficient_paired_data",
            "fails_confirmatory_FDR",
            "fails_primary_precip_field_significance",
            "fails_event_jackknife_stability",
            "opposite_direction",
            "supports",
        ],
        default="not_classified",
    )
    hypothesis_rows = gate[in_hypothesis].copy()
    summary = (
        hypothesis_rows
        .groupby(
            ["regiao", "tipo_enso_fonte", "fase_fonte_em_t_menos_lag"],
            as_index=False,
        )
        .agg(
            n_targets=("target_chirps", "nunique"),
            n_targets_support=("gate_supports_direction", "sum"),
            n_target_families=("target_family", "nunique"),
            min_event_stability=("estabilidade_sinal_jackknife", "min"),
            max_q_confirmatory=("q_F4D_global_targets_regioes_fases", "max"),
            max_p_primary_precip_field=(
                "p_primary_precip_field_max_lag_F4C",
                "max",
            ),
        )
    )
    grouping = ["regiao", "tipo_enso_fonte", "fase_fonte_em_t_menos_lag"]
    family_support = (
        hypothesis_rows.groupby([*grouping, "target_family"], as_index=False)[
            "gate_supports_direction"
        ]
        .max()
        .groupby(grouping, as_index=False)["gate_supports_direction"]
        .sum()
        .rename(columns={"gate_supports_direction": "n_target_families_support"})
    )
    summary = summary.merge(family_support, on=grouping, how="left")
    primary_support = (
        hypothesis_rows.loc[
            hypothesis_rows["target_family"].eq("weekly_total_precipitation")
        ]
        .groupby(grouping, as_index=False)["gate_supports_direction"]
        .max()
        .rename(columns={"gate_supports_direction": "primary_precipitation_support"})
    )
    summary = summary.merge(primary_support, on=grouping, how="left")
    summary["primary_precipitation_support"] = (
        summary["primary_precipitation_support"].fillna(False).astype(bool)
    )
    summary["any_target_specific_support"] = summary["n_targets_support"] > 0
    summary["hypothesis_gate"] = np.select(
        [
            summary["primary_precipitation_support"],
            summary["any_target_specific_support"],
        ],
        [
            "supports_primary_weekly_precipitation",
            "supports_target_specific_extreme_or_drought_only",
        ],
        default="no_target_specific_support",
    )
    summary["aggregation_policy"] = (
        "cada alvo/familia e interpretado separadamente; nao se exige um numero "
        "arbitrario de familias complementares"
    )
    return gate, summary


def save_cluster_figures(
    clusters: pd.DataFrame,
    profiles: pd.DataFrame,
    ranking: pd.DataFrame,
    units,
    *,
    enso_type: str | None = None,
    run_id: str,
    source_paths: dict[str, Path],
) -> None:
    figure_dir = FIGS / enso_type if enso_type else FIGS
    figure_dir.mkdir(parents=True, exist_ok=True)
    slug_suffix = f"_{enso_type}" if enso_type else ""
    valid = clusters["cluster"] >= 0
    brazil = units[units["tipo_unidade"].eq("regiao")].dissolve()
    regions = units[units["tipo_unidade"].eq("regiao")]
    fig, axis = plt.subplots(figsize=(13, 11))
    values = clusters["cluster"].where(valid).to_numpy(dtype=float)
    maximum_cluster = float(np.nanmax(values)) if np.isfinite(values).any() else 0.0
    mesh = plot_pixel_field(
        axis,
        clusters[["lat", "lon"]],
        values,
        brazil_geometry=brazil,
        boundaries=regions,
        cmap="tab10",
        vmin=-0.5,
        vmax=max(maximum_cluster + 0.5, 0.5),
        title="4D - descriptive clusters of FDR-significant native-pixel response profiles",
    )
    fig.colorbar(mesh, ax=axis, fraction=0.035, pad=0.02).set_label("cluster")
    output = save_registered_figure(
        fig,
        phase=4,
        block="D",
        index=1,
        slug=f"mapa_clusters_pixels_nativos{slug_suffix}",
        interpretation=(
            "Descriptive clusters organize original CHIRPS pixels by F4C EN/LN-phase response; "
            "unclassified pixels have fewer than two FDR-significant profiles. Clusters do not "
            "establish significance; the event-jackknife gate does."
        ),
        metadata="CHIRPS native 0.25; no interpolation; profiles from F4C BH-selected lags",
        figures_dir=figure_dir,
        reserve_bottom=0.12,
    )
    registrar_figura(
        None,
        output.stem,
        fase=4,
        bloco="D",
        titulo="Descriptive clusters on native CHIRPS pixels",
        descricao=(
            "Clusters summarize F4C FDR response profiles and carry no inferential status."
        ),
        hipotese="HIP1",
        notebook="notebooks/fase4/4D_clusters_alvo.ipynb",
        run_id=run_id,
        fontes={
            "clusters_pixels": source_paths["clusters"],
            "ranking_clusters": source_paths["ranking"],
        },
    )

    profiles = profiles.copy()
    if profiles.empty:
        fig, axis = plt.subplots(figsize=(10, 4))
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "No defensible clusters: fewer than 50 pixels had two FDR-significant profiles.",
            ha="center",
            va="center",
            fontsize=12,
        )
        output = save_registered_figure(
            fig,
            phase=4,
            block="D",
            index=2,
            slug=f"perfis_clusters_fdr{slug_suffix}",
            interpretation=(
                "The inferential gate remains valid, but descriptive clustering was not "
                "performed because the preregistered minimum support was not reached."
            ),
            metadata="No cluster profile was imputed or forced",
            figures_dir=figure_dir,
            reserve_bottom=0.12,
        )
        registrar_figura(
            None,
            output.stem,
            fase=4,
            bloco="D",
            titulo="Cluster support gate",
            descricao="No cluster was forced when FDR profile support was insufficient.",
            hipotese="HIP1",
            notebook="notebooks/fase4/4D_clusters_alvo.ipynb",
            run_id=run_id,
            fontes={
                "cluster_profiles": source_paths["profiles"],
                "cluster_ranking": source_paths["ranking"],
            },
        )
        return
    profiles["feature"] = profiles["variavel"] + " | " + profiles["condicao_fonte"]
    strength = profiles.groupby("feature")["r_mediano_fdr"].apply(
        lambda values: float(values.abs().mean())
    )
    top = strength.nlargest(min(30, len(strength))).index
    matrix = profiles[profiles["feature"].isin(top)].pivot(
        index="cluster", columns="feature", values="r_mediano_fdr"
    )
    fig, axis = plt.subplots(figsize=(max(12, 0.38 * matrix.shape[1]), 4 + 0.5 * len(matrix)))
    image = axis.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    axis.set_xticks(range(matrix.shape[1]))
    axis.set_xticklabels(matrix.columns, rotation=90, fontsize=6)
    axis.set_yticks(range(matrix.shape[0]))
    axis.set_yticklabels([f"cluster {value}" for value in matrix.index])
    axis.set_title("4D - median FDR response profiles of descriptive clusters")
    fig.colorbar(image, ax=axis, label="median r at FDR-selected lag")
    output = save_registered_figure(
        fig,
        phase=4,
        block="D",
        index=2,
        slug=f"perfis_clusters_fdr{slug_suffix}",
        interpretation=(
            "Cluster profiles retain ENSO type and source phase separately. Only F4C lags "
            "passing the declared BH family contribute; missing profiles are not zero effects."
        ),
        metadata=f"Top {matrix.shape[1]} profiles by mean absolute cluster response",
        figures_dir=figure_dir,
        reserve_bottom=0.12,
    )
    registrar_figura(
        None,
        output.stem,
        fase=4,
        bloco="D",
        titulo="FDR response profiles of descriptive clusters",
        descricao=(
            "Profiles retain ENSO type and source phase and contain only declared F4C statistics."
        ),
        hipotese="HIP1",
        notebook="notebooks/fase4/4D_clusters_alvo.ipynb",
        run_id=run_id,
        fontes={
            "cluster_profiles": source_paths["profiles"],
            "cluster_ranking": source_paths["ranking"],
        },
    )


def save_gate_figure(
    gate: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    enso_type: str | None = None,
    run_id: str,
    source_paths: dict[str, Path],
) -> None:
    figure_dir = FIGS / enso_type if enso_type else FIGS
    figure_dir.mkdir(parents=True, exist_ok=True)
    slug_suffix = f"_{enso_type}" if enso_type else ""
    subset = gate[gate["regiao"].isin(["Nordeste", "Sul"])].copy()
    subset["row"] = (
        subset["regiao"]
        + " | "
        + subset["tipo_enso_fonte"]
        + " | "
        + subset["fase_fonte_em_t_menos_lag"]
    )
    matrix = subset.pivot(index="row", columns="target_label", values="r")
    fig, axis = plt.subplots(figsize=(12, max(6, 0.35 * len(matrix) + 3)))
    image = axis.imshow(matrix.to_numpy(), cmap="BrBG", vmin=-0.6, vmax=0.6, aspect="auto")
    axis.set_xticks(range(matrix.shape[1]))
    axis.set_xticklabels(matrix.columns, rotation=35, ha="right", fontsize=8)
    axis.set_yticks(range(matrix.shape[0]))
    axis.set_yticklabels(matrix.index, fontsize=7)
    lookup = subset.set_index(["row", "target_label"])
    for i, row in enumerate(matrix.index):
        for j, target_label in enumerate(matrix.columns):
            if (row, target_label) not in lookup.index or not np.isfinite(matrix.iloc[i, j]):
                continue
            record = lookup.loc[(row, target_label)]
            mark = "*" if bool(record["gate_supports_direction"]) else ""
            axis.text(j, i, f"{matrix.iloc[i, j]:+.2f}{mark}", ha="center", va="center", fontsize=6)
    axis.set_title("4D - multi-target hypothesis gate (* passes FDR + field + event jackknife)")
    fig.colorbar(image, ax=axis, label="r at F4C-selected lag")
    output = save_registered_figure(
        fig,
        phase=4,
        block="D",
        index=3,
        slug=f"gate_multialvo_eventos{slug_suffix}",
        interpretation=(
            f"A star requires the expected physical direction, global confirmatory BH q<{FDR_ALPHA:.2f}, "
            f"the common primary-precipitation whole-field screen p<{FDR_ALPHA:.2f} corrected over lags, "
            "and >=80% sign stability when one entire ENSO event is removed. R95p and R99p "
            "belong to one heavy-rain family and are never counted as independent evidence."
        ),
        metadata="Native CHIRPS precipitation, SPI, R95p, R99p and dry-spell anomalies",
        figures_dir=figure_dir,
        reserve_bottom=0.14,
    )
    registrar_figura(
        None,
        output.stem,
        fase=4,
        bloco="D",
        titulo="Multi-target event-jackknife hypothesis gate",
        descricao=(
            "Expected direction + global FDR + primary-precipitation field screen + "
            "whole-event stability; each CHIRPS target family is interpreted separately."
        ),
        hipotese="HIP1",
        notebook="notebooks/fase4/4D_clusters_alvo.ipynb",
        run_id=run_id,
        fontes={
            "gate_tests": source_paths["gate"],
            "hypothesis_summary": source_paths["summary"],
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enso-type",
        choices=ENSO_TYPES,
        help="run one independent ENSO analysis (el_nino or la_nina)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-significant-profiles", type=int, default=2)
    args = parser.parse_args(argv)
    FIGS.mkdir(parents=True, exist_ok=True)
    STATS.mkdir(parents=True, exist_ok=True)

    (
        target,
        validation,
        best,
        unit_lags,
        field,
        membership,
        target_contract,
        f4c_paths,
    ) = load_contracts(enso_type=args.enso_type)
    source_run_ids: set[str] = set()
    parent_f3_run_ids: set[str] = set()
    parent_f3_hashes: set[str] = set()
    for source_table in (best, unit_lags, field):
        if "analysis_run_id" not in source_table:
            raise ValueError("F4C source table lacks analysis_run_id; rerun canonical F4C.")
        source_run_ids.update(
            source_table["analysis_run_id"].dropna().astype(str).str.strip()
        )
        if (
            "parent_f3_run_id" not in source_table
            or "parent_f3_artifact_sha256" not in source_table
        ):
            raise ValueError(
                "F4C source table lacks parent F3 lineage; rerun canonical F4C."
            )
        parent_f3_run_ids.update(
            source_table["parent_f3_run_id"].dropna().astype(str).str.strip()
        )
        parent_f3_hashes.update(
            source_table["parent_f3_artifact_sha256"].dropna().astype(str).str.strip()
        )
    source_run_ids.discard("")
    parent_f3_run_ids.discard("")
    parent_f3_hashes.discard("")
    if len(source_run_ids) != 1:
        raise ValueError(f"F4C source tables have incompatible run IDs: {source_run_ids}")
    if len(parent_f3_run_ids) != 1 or len(parent_f3_hashes) != 1:
        raise ValueError(
            "F4C source tables have incompatible parent F3 contracts: "
            f"run_ids={parent_f3_run_ids}, hashes={parent_f3_hashes}"
        )
    parent_f4c_run_id = next(iter(source_run_ids))
    parent_f3_run_id = next(iter(parent_f3_run_ids))
    parent_f3_artifact_sha256 = next(iter(parent_f3_hashes))
    f4c_output_paths = list(f4c_paths.values())
    for path in f4c_output_paths:
        verify_canonical_f4c_output(
            path,
            expected_run_id=parent_f4c_run_id,
            expected_enso_type=args.enso_type,
        )
    target_times = pd.DatetimeIndex(target.time.values)
    phase_table_path = phase_table_path_for_enso_type(args.enso_type)
    (
        phase_table,
        current_f3_run_id,
        current_f3_artifact_sha256,
    ) = load_canonical_phase_table(
        target_times,
        phase_table_path=phase_table_path,
    )
    if (
        current_f3_run_id != parent_f3_run_id
        or current_f3_artifact_sha256 != parent_f3_artifact_sha256
    ):
        raise ValueError(
            "Canonical F3 phase table changed after F4C; rerun F4C before F4D. "
            f"F4C parent=({parent_f3_run_id}, {parent_f3_artifact_sha256}), "
            f"current=({current_f3_run_id}, {current_f3_artifact_sha256})."
        )
    scope_prefix = f"{args.enso_type.upper()}_" if args.enso_type else ""
    run_id = (
        f"F4D_{scope_prefix}"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + validation.grid_hash[:8]
        + "_"
        + target_contract["target_block_signature_sha256"][:8]
        + "_"
        + parent_f4c_run_id[-8:]
        + "_"
        + uuid.uuid4().hex[:6]
    )
    output_contract = {
        **target_contract,
        **ibge_geometry_contract(),
        "enso_type": args.enso_type or "combined",
        "f4c_runner_sha256": sha256_file(ROOT / "scripts/run_fase4c_regional.py"),
        "f4d_runner_sha256": sha256_file(Path(__file__).resolve()),
        "lag_analysis_module_sha256": sha256_file(
            ROOT / "src/nino_brasil/stats/lag_analysis.py"
        ),
    }
    units = analysis_units()
    requested_conditions = conditions_for_enso_type(args.enso_type)
    clusters, profiles, ranking = cluster_native_pixels(
        best,
        membership,
        conditions=requested_conditions,
        min_significant_profiles=args.min_significant_profiles,
        seed=args.seed,
    )
    clusters["grid_hash_sha256"] = validation.grid_hash
    profiles["grid_hash_sha256"] = validation.grid_hash
    ranking["grid_hash_sha256"] = validation.grid_hash
    for output_table in (clusters, profiles, ranking):
        output_table["analysis_run_id"] = run_id
        output_table["parent_f4c_run_id"] = parent_f4c_run_id
        output_table["parent_f3_run_id"] = parent_f3_run_id
        output_table["parent_f3_artifact_sha256"] = parent_f3_artifact_sha256
        output_table["cluster_random_seed"] = args.seed
        output_table["cluster_min_significant_profiles"] = (
            args.min_significant_profiles
        )
        for column, value in output_contract.items():
            output_table[column] = value
    clusters_path = scoped_artifact_path(
        STATS / "phase4D_native_clusters_pixels.parquet", args.enso_type
    )
    profiles_path = scoped_artifact_path(
        STATS / "phase4D_native_cluster_profiles.csv", args.enso_type
    )
    ranking_path = scoped_artifact_path(
        STATS / "phase4D_native_cluster_ranking.csv", args.enso_type
    )
    clusters.to_parquet(clusters_path, index=False)
    profiles.to_csv(profiles_path, index=False)
    ranking.to_csv(ranking_path, index=False)

    conditions_all = build_source_conditions(
        phase_table,
        include_all_weeks=False,
        include_event_aggregates=False,
    )
    conditions = {name: conditions_all[name] for name in requested_conditions}
    master = pd.read_csv(
        FEAT / "nino34_master_weekly.csv", parse_dates=["week_ending_sunday"]
    ).set_index("week_ending_sunday")
    predictor_frame, _ = harmonic_deseasonalize_predictors(
        master[[KEY_PREDICTOR]].reindex(target_times)
    )
    target_series, target_coverage = regional_target_series(target, membership, units)
    gate, summary = build_gate(
        target_series,
        unit_lags,
        field,
        units,
        predictor_frame[KEY_PREDICTOR],
        phase_table,
        conditions,
    )
    gate["grid_hash_sha256"] = validation.grid_hash
    summary["grid_hash_sha256"] = validation.grid_hash
    gate["analysis_run_id"] = run_id
    gate["parent_f4c_run_id"] = parent_f4c_run_id
    gate["parent_f3_run_id"] = parent_f3_run_id
    gate["parent_f3_artifact_sha256"] = parent_f3_artifact_sha256
    summary["analysis_run_id"] = run_id
    summary["parent_f4c_run_id"] = parent_f4c_run_id
    summary["parent_f3_run_id"] = parent_f3_run_id
    summary["parent_f3_artifact_sha256"] = parent_f3_artifact_sha256
    for output_table in (gate, summary):
        output_table["cluster_random_seed"] = args.seed
        output_table["cluster_min_significant_profiles"] = (
            args.min_significant_profiles
        )
        for column, value in output_contract.items():
            output_table[column] = value
    target_coverage["analysis_run_id"] = run_id
    target_coverage["parent_f4c_run_id"] = parent_f4c_run_id
    target_coverage["parent_f3_run_id"] = parent_f3_run_id
    target_coverage["parent_f3_artifact_sha256"] = parent_f3_artifact_sha256
    for column, value in output_contract.items():
        target_coverage[column] = value
    gate_path = scoped_artifact_path(
        STATS / "phase4D_native_gate_event_jackknife.csv", args.enso_type
    )
    summary_path = scoped_artifact_path(
        STATS / "phase4D_native_hypothesis_summary.csv", args.enso_type
    )
    target_coverage_path = scoped_artifact_path(
        STATS / "phase4D_native_target_coverage.csv", args.enso_type
    )
    gate.to_csv(gate_path, index=False)
    summary.to_csv(summary_path, index=False)
    target_coverage.to_csv(target_coverage_path, index=False)
    target_manifest = target_build_manifest_path(target_contract)
    manifest_inputs = [
        FEAT / "nino34_master_weekly.csv",
        phase_table_path,
        Path(f"{phase_table_path}.manifest.json"),
        target_manifest,
        TARGET_VARIABLE_CONTRACT,
        TARGET_PIXEL_INVENTORY,
        MEMBERSHIP,
        *ibge_geometry_input_paths(),
        *f4c_output_paths,
        *(Path(f"{path}.manifest.json") for path in f4c_output_paths),
        ROOT / "scripts/run_fase4c_regional.py",
        Path(__file__).resolve(),
        ROOT / "src/nino_brasil/stats/lag_analysis.py",
    ]
    write_phase4_csv_manifests(
        [
            (clusters_path, clusters),
            (profiles_path, profiles),
            (ranking_path, ranking),
            (gate_path, gate),
            (summary_path, summary),
            (target_coverage_path, target_coverage),
        ],
        run_id=run_id,
        stage="F4D",
        contract={
            **output_contract,
            "parent_f4c_run_id": parent_f4c_run_id,
            "parent_f3_run_id": parent_f3_run_id,
            "parent_f3_artifact_sha256": parent_f3_artifact_sha256,
            "grid_hash_sha256": validation.grid_hash,
            "enso_type": args.enso_type or "combined",
            "cluster_random_seed": args.seed,
            "cluster_min_significant_profiles": args.min_significant_profiles,
        },
        inputs=manifest_inputs,
    )

    source_paths = {
        "clusters": clusters_path,
        "profiles": profiles_path,
        "ranking": ranking_path,
        "gate": gate_path,
        "summary": summary_path,
    }
    # As execuções oficiais Nino/Nina publicam somente pelo notebook
    # canônico, garantindo o par FigF4*/TabF4* e nenhum resíduo legado.
    if args.enso_type is None:
        save_cluster_figures(
            clusters,
            profiles,
            ranking,
            units,
            enso_type=args.enso_type,
            run_id=run_id,
            source_paths=source_paths,
        )
        save_gate_figure(
            gate,
            summary,
            enso_type=args.enso_type,
            run_id=run_id,
            source_paths=source_paths,
        )
    else:
        print("[4D] figuras públicas serão publicadas pelo notebook canônico")
    print(
        f"[4D] clusters={ranking.shape[0]} | gate tests={len(gate)} | "
        f"supports={int(gate['gate_supports_direction'].sum())} | "
        f"grid_sha256={validation.grid_hash} | run_id={run_id} | "
        f"parent_f4c_run_id={parent_f4c_run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
