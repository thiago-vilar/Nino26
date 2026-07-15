#!/usr/bin/env python3
"""Run pixel-native F6 RF/XGBoost on the canonical CHIRPS target.

Every model target is an unchanged original CHIRPS Brazil pixel.  Conditions
refer to the ENSO phase at source week ``t-lag``.  Official runs use all 31 F2
variables, eight EN/LN life-cycle conditions and whole-event purged folds.
Use pixel shards for tractable parallel execution; smoke outputs are isolated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nino_brasil.artifacts import start_artifact_run  # noqa: E402
from nino_brasil.models.phase5_cycle_ml import physical_predictor_columns  # noqa: E402
from nino_brasil.models.phase6_brazil_ml import (  # noqa: E402
    CANONICAL_ACTIVE_CONDITIONS,
    fit_pixel_teleconnection,
)
from nino_brasil.targets.chirps_native import target_to_frame, validate_native_target  # noqa: E402

FEATURES = ROOT / "data" / "processed" / "parquet" / "features"
STATISTICS = ROOT / "data" / "processed" / "parquet" / "statistics"
MODEL_BRIDGE = ROOT / "data" / "processed" / "parquet" / "modeling" / "f3_bridge"
TARGET = ROOT / "data" / "processed" / "zarr" / "features" / "chirps_native_weekly_targets.zarr"
RUNNER_CONTRACT = "f6-pixel-native-v3-causal-persistence-ridge"


def _integers(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _find(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{label} ausente: {candidates}")


def _csv_time(path: Path, *, phase: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path).rename(columns={"event_type": "tipo", "phase": "fase"})
    time_column = next((name for name in ("week_ending_sunday", "time", "date") if name in frame), None)
    if time_column is None:
        raise KeyError(f"{path}: coluna temporal ausente")
    if phase and (missing := {"tipo", "fase", "event_id"}.difference(frame.columns)):
        raise KeyError(f"{path}: colunas ausentes {sorted(missing)}")
    frame[time_column] = pd.to_datetime(frame[time_column])
    return frame.set_index(time_column).sort_index()


def _f4_references_ready() -> bool:
    """Require both isolated F4 baselines, never a positive-result gate."""

    references = {
        "el_nino": STATISTICS / "TabF4NinoD5_resumo_hipoteses.csv",
        "la_nina": STATISTICS / "TabF4NinaD5_resumo_hipoteses.csv",
    }
    for enso_type, path in references.items():
        sidecar = Path(f"{path}.manifest.json")
        if not path.is_file() or not sidecar.is_file():
            return False
        try:
            manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        contract = manifest.get("contract") or {}
        if (
            not str(manifest.get("run_id", "")).strip()
            or contract.get("enso_type") != enso_type
            or contract.get("stage") != "F4D"
        ):
            return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("rf", "xgb"), default="rf")
    parser.add_argument("--mode", choices=("official", "smoke"), default="official")
    parser.add_argument("--target", type=Path, default=TARGET)
    parser.add_argument("--target-variable", default="precip_weekly_mm")
    parser.add_argument(
        "--target-transform",
        choices=("train_robust_z", "train_anomaly_mm", "none"),
        default="train_robust_z",
    )
    parser.add_argument("--master", type=Path)
    parser.add_argument("--phase-table", type=Path)
    parser.add_argument("--lags", type=_integers, default=(4, 8, 12, 16, 20, 24))
    parser.add_argument("--conditions", type=_strings, default=CANONICAL_ACTIVE_CONDITIONS)
    parser.add_argument("--pixel-start", type=int, default=0)
    parser.add_argument("--pixel-stop", type=int)
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-predictions", action="store_true")
    parser.add_argument("--research-override-gate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "official" and args.no_predictions:
        raise ValueError("Runs F6 oficiais exigem predicoes OOS para merge e gate F8.")
    artifact_mode = (
        "smoke"
        if args.mode == "official" and args.research_override_gate
        else args.mode
    )
    if artifact_mode != args.mode:
        print(
            "[F6 exploratoria] override solicitado: o artefato sera isolado em runs/smoke."
        )
    if args.mode == "official" and not args.research_override_gate and not _f4_references_ready():
        print("[F6 bloqueada] faltam as referencias F4NinoD/F4NinaD completas e auditaveis.")
        return 3
    master_path = args.master or _find(
        [FEATURES / "nino34_master_weekly.csv"], "master F2"
    )
    phase_path = args.phase_table or _find(
        [MODEL_BRIDGE / "fases_semanais_en_ln.csv"],
        "fases semanais canonicas F3",
    )
    target = xr.open_zarr(args.target, consolidated=None)
    validation = validate_native_target(target)
    if not validation.valid:
        raise ValueError(f"Alvo CHIRPS nativo invalido: {validation.errors}")
    if args.target_variable not in target:
        raise KeyError(f"Alvo CHIRPS sem variavel {args.target_variable!r}.")
    target_units = str(target[args.target_variable].attrs.get("units", "native target units"))
    response, pixels = target_to_frame(
        target,
        variable=args.target_variable,
        brazil_only=True,
        mask_rule="overlap",
    )
    stop = args.pixel_stop if args.pixel_stop is not None else len(pixels)
    if args.mode == "smoke":
        stop = min(args.pixel_start + 4, stop)
    positions = range(args.pixel_start, stop)
    pixels = pixels.iloc[list(positions)].reset_index(drop=True)
    response = response.loc[:, pixels["pixel_id"].astype(str)]
    master = _csv_time(master_path)
    phase = _csv_time(phase_path, phase=True)
    predictors = physical_predictor_columns(master)
    lags = tuple(args.lags)
    conditions = tuple(args.conditions)
    n_estimators = args.n_estimators
    n_splits = 4
    min_train_rows = 24
    if args.mode == "smoke":
        lags = lags[:1]
        conditions = conditions[:1]
        n_estimators = min(n_estimators, 10)
        n_splits = 2
        min_train_rows = 8
    parameters = {
        "runner_contract": RUNNER_CONTRACT,
        "model": args.model,
        "target_variable": args.target_variable,
        "target_transform": args.target_transform,
        "grid_sha256": target.attrs.get("grid_hash_sha256"),
        "pixel_start": args.pixel_start,
        "pixel_stop": stop,
        "n_pixels": len(pixels),
        "lags": lags,
        "conditions": conditions,
        "n_estimators": n_estimators,
        "n_splits": n_splits,
        "min_train_events": 3,
        "min_train_rows": min_train_rows,
        "min_test_rows": 3,
        "permutation_repeats": 1,
        "store_predictions": not args.no_predictions,
        "research_override_gate": bool(args.research_override_gate),
        "phase4_statistical_baseline_required": True,
        "target_units": target_units,
        "n_physical_predictors": len(predictors),
        "phase_reference": "source_t_minus_lag",
        "interpolation_target": False,
    }
    run = start_artifact_run(
        6,
        mode=artifact_mode,
        inputs=[args.target, master_path, phase_path, Path(__file__).resolve()],
        seed=args.seed,
        parameters=parameters,
        command=" ".join([sys.executable, *sys.argv]),
    )
    try:
        result = fit_pixel_teleconnection(
            master,
            response,
            pixels,
            phase,
            predictors=predictors,
            model=args.model,
            lags_weeks=lags,
            conditions=conditions,
            n_splits=n_splits,
            min_train_events=3,
            min_train_rows=min_train_rows,
            min_test_rows=3,
            n_estimators=n_estimators,
            store_predictions=not args.no_predictions,
            target_transform=args.target_transform,
            target_name=args.target_variable,
            target_units=target_units,
            permutation_repeats=1,
            random_state=args.seed,
        )
        if result.metrics.empty:
            raise ValueError(
                "F6 nao produziu nenhuma metrica valida; o shard nao pode ser finalizado."
            )
        expected = pd.MultiIndex.from_product(
            [
                pixels["pixel_id"].astype(str).unique(),
                conditions,
                lags,
            ],
            names=["pixel_id", "condition", "lag_weeks"],
        )
        observed = pd.MultiIndex.from_frame(
            result.metrics[["pixel_id", "condition", "lag_weeks"]]
            .astype({"pixel_id": str, "condition": str, "lag_weeks": int})
            .drop_duplicates()
        )
        coverage = expected.to_frame(index=False)
        coverage["has_oos_metric"] = coverage.set_index(
            ["pixel_id", "condition", "lag_weeks"]
        ).index.isin(observed)
        coverage["coverage_contract"] = "one-or-more whole-event OOS folds"
        if args.mode == "official" and not bool(coverage["has_oos_metric"].all()):
            missing_count = int((~coverage["has_oos_metric"]).sum())
            raise ValueError(
                f"Shard F6 incompleto: {missing_count} combinacoes "
                "pixel x condicao x lag sem metrica OOS."
            )
        result.grid_contract["parent_grid_sha256"] = target.attrs.get("grid_hash_sha256", "")
        products = {
            "pixel_metrics": result.metrics,
            "pixel_oos_predictions": result.predictions,
            "pixel_variable_importance": result.importances,
            "fold_contract": result.fold_contract,
            "native_grid_contract": result.grid_contract,
            "pixel_shard_inventory": pixels,
            "predictor_contract": pd.DataFrame({"variable": predictors}),
            "shard_coverage_contract": coverage,
        }
        for name, frame in products.items():
            run.write_table(
                name,
                frame,
                description={
                    "pixel_metrics": "Skill fora da amostra por pixel CHIRPS original.",
                    "pixel_oos_predictions": "Observado/predito/baseline por pixel e semana.",
                    "pixel_variable_importance": "Importancia das 31 variaveis por pixel/fase/lag.",
                    "fold_contract": "Evento inteiro, embargo e fase no tempo fonte.",
                    "native_grid_contract": "Hash e garantia sem interpolacao do alvo.",
                    "pixel_shard_inventory": "Coordenadas originais incluídas neste shard.",
                    "predictor_contract": "Candidatas fisicas primarias do master F2.",
                    "shard_coverage_contract": "Cobertura pixel x condicao x lag exigida no shard.",
                }[name],
                methods={
                    "target_interpolation": False,
                    "event_grouped": True,
                    "source_phase_conditioning": True,
                },
            )
        run.finalize(notes="Agregar regioes/biomas somente depois das previsoes pixel-a-pixel.")
        print(f"[F6] run_id={run.run_id} | pixels={len(pixels)} | outputs={run.directory}")
        return 0
    except Exception as exc:
        run.finalize(status="failed", notes=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
