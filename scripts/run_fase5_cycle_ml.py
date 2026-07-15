#!/usr/bin/env python3
"""Run F5 RF/XGBoost on the ENSO nine-state rolling-origin experiment.

Official runs use all 31 physical F2 variables, whole-event expanding folds,
fold-only preprocessing, purge/embargo, strong baselines and conservative
train-only augmentation.  ``--mode smoke`` uses real local data with a reduced
compute budget and writes to a separate tree; it never overwrites official runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nino_brasil.artifacts import start_artifact_run  # noqa: E402
from nino_brasil.models.phase5_cycle_ml import (  # noqa: E402
    build_rolling_origin_table,
    fit_event_dimension_rolling_origin,
    fit_event_aware_phase_classifier,
    physical_predictor_columns,
)

FEATURES = ROOT / "data" / "processed" / "parquet" / "features"
STATISTICS = ROOT / "data" / "processed" / "parquet" / "statistics"
MODEL_BRIDGE = ROOT / "data" / "processed" / "parquet" / "modeling" / "f3_bridge"


def _integers(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("informe ao menos um inteiro separado por virgula")
    return values


def _existing(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{label} ausente; procurei: {', '.join(map(str, candidates))}")


def _load_master(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    time_column = next(
        (name for name in ("week_ending_sunday", "time", "date") if name in frame), None
    )
    if time_column is None:
        raise KeyError("master sem coluna temporal reconhecida")
    frame[time_column] = pd.to_datetime(frame[time_column])
    return frame.set_index(time_column).sort_index()


def _load_phase_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    time_column = next(
        (name for name in ("week_ending_sunday", "time", "date") if name in frame), None
    )
    if time_column is None:
        raise KeyError("tabela de fases sem coluna temporal reconhecida")
    aliases = {"event_type": "tipo", "phase": "fase"}
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame})
    required = {"tipo", "fase", "event_id"}
    if missing := required.difference(frame.columns):
        raise KeyError(f"tabela de fases sem {sorted(missing)}")
    frame[time_column] = pd.to_datetime(frame[time_column])
    return frame.set_index(time_column).sort_index()


def _classification_gate_pass(
    mean_skill: float,
    independent_support_gate_pass: bool,
    *,
    support_required_for_gate: bool,
) -> bool:
    """Apply the official support floor without blocking smoke execution."""

    return bool(
        np.isfinite(mean_skill)
        and mean_skill > 0.0
        and (
            not support_required_for_gate
            or bool(independent_support_gate_pass)
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("rf", "xgb"), default="rf")
    parser.add_argument("--mode", choices=("official", "smoke"), default="official")
    parser.add_argument("--horizons", type=_integers, default=(0, 4, 8, 12, 24))
    parser.add_argument("--lags", type=_integers, default=tuple(range(4, 53, 4)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--noise-copies", type=int, default=1)
    parser.add_argument("--noise-scale", type=float, default=0.02)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--min-train-active-events-per-type",
        type=int,
        default=3,
        help="Piso oficial separado para eventos El Nino e La Nina no treino de cada fold.",
    )
    parser.add_argument("--master", type=Path)
    parser.add_argument("--phase-table", type=Path)
    parser.add_argument("--events-table", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    master_path = args.master or _existing(
        [FEATURES / "nino34_master_weekly.csv"], "master F2"
    )
    phase_path = args.phase_table or _existing(
        [MODEL_BRIDGE / "fases_semanais_en_ln.csv"],
        "tabela semanal canonica F3 EN/LN x fase",
    )
    events_path = args.events_table or _existing(
        [MODEL_BRIDGE / "events_en_ln.csv"], "eventos EN/LN F3"
    )
    master = _load_master(master_path)
    phase_table = _load_phase_table(phase_path)
    events = pd.read_csv(events_path)
    predictors = physical_predictor_columns(master)
    horizons = tuple(args.horizons)
    lags = tuple(args.lags)
    n_estimators = int(args.n_estimators)
    n_splits = 5
    min_train_groups = 8
    if args.mode == "smoke":
        # Real data and all 31 variables, but one horizon/fewer trees/lags.
        horizons = (horizons[0],)
        lags = tuple(lags[: min(3, len(lags))])
        n_estimators = min(n_estimators, 20)
        n_splits = 2
        min_train_groups = 6
    noise_copies = 0 if args.no_augmentation else int(args.noise_copies)
    mixup_alpha = None if args.no_augmentation else args.mixup_alpha
    parameters = {
        "model": args.model,
        "horizons": horizons,
        "lags": lags,
        "n_estimators": n_estimators,
        "n_splits": n_splits,
        "n_physical_predictors": len(predictors),
        "noise_copies": noise_copies,
        "noise_scale": args.noise_scale,
        "mixup_alpha": mixup_alpha,
        "min_train_active_events_per_type": int(
            args.min_train_active_events_per_type
        ),
        "support_balanced_fold_start": args.mode == "official",
        "validation_unit": "whole_event_and_neutral_quarter",
        "prediction_n_jobs": 1,
    }
    run = start_artifact_run(
        5,
        mode=args.mode,
        inputs=[master_path, phase_path, events_path, Path(__file__).resolve()],
        seed=args.seed,
        parameters=parameters,
        command=" ".join([sys.executable, *sys.argv]),
    )
    try:
        run.write_table(
            "predictor_contract",
            pd.DataFrame(
                {
                    "variable": predictors,
                    "role": "physical_predictor",
                    "n_variables_contract": len(predictors),
                }
            ),
            description="As 31 variaveis fisicas do master F2 usadas como candidatas primarias.",
            dimensions={"variable": "31 physical F2 predictors"},
            primary_keys=("variable",),
        )
        rolling = build_rolling_origin_table(
            master,
            phase_table,
            predictors=predictors,
            horizons=[h for h in horizons if h > 0] or (1,),
        )
        run.write_table(
            "rolling_origin_targets",
            rolling,
            description="Origens semanais e alvos futuros; datas de pico nunca entram nos preditores.",
            dimensions={"row": "weekly origin", "horizon": "weeks"},
            methods={"role": "predictive", "future_peak_alignment": False},
            primary_keys=("origin_time",),
        )

        all_metrics: list[pd.DataFrame] = []
        all_predictions: list[pd.DataFrame] = []
        all_importance: list[pd.DataFrame] = []
        all_state_importance: list[pd.DataFrame] = []
        all_provenance: list[pd.DataFrame] = []
        all_contracts: list[pd.DataFrame] = []
        all_support: list[pd.DataFrame] = []
        for horizon in horizons:
            print(
                f"[F5] model={args.model} | horizon={horizon} weeks | "
                f"folds={n_splits} | trees={n_estimators}",
                flush=True,
            )
            result = fit_event_aware_phase_classifier(
                master,
                phase_table,
                predictors=predictors,
                lags=lags,
                horizon_weeks=horizon,
                model=args.model,
                n_splits=n_splits,
                min_train_groups=min_train_groups,
                n_estimators=n_estimators,
                augmentation_noise_copies=noise_copies,
                augmentation_noise_scale=args.noise_scale,
                augmentation_mixup_alpha=mixup_alpha,
                min_train_active_events_per_type=args.min_train_active_events_per_type,
                enforce_support_at_fold_design=args.mode == "official",
                random_state=args.seed,
            )
            for frame in (
                result.fold_metrics,
                result.predictions,
                result.importances,
                result.state_importances,
                result.augmentation_provenance,
                result.fold_contract,
                result.independent_support,
            ):
                frame.insert(0, "experiment_horizon_weeks", horizon)
                frame.insert(0, "model", args.model) if "model" not in frame else None
            all_metrics.append(result.fold_metrics)
            all_predictions.append(result.predictions)
            all_importance.append(result.importances)
            all_state_importance.append(result.state_importances)
            all_provenance.append(result.augmentation_provenance)
            all_contracts.append(result.fold_contract)
            all_support.append(result.independent_support)

        products = {
            "fold_metrics": pd.concat(all_metrics, ignore_index=True),
            "oos_predictions": pd.concat(all_predictions, ignore_index=True),
            "global_importance": pd.concat(all_importance, ignore_index=True),
            "state_importance_oos": pd.concat(all_state_importance, ignore_index=True),
            "augmentation_provenance": pd.concat(all_provenance, ignore_index=True),
            "fold_contract": pd.concat(all_contracts, ignore_index=True),
            "independent_support_by_fold": pd.concat(all_support, ignore_index=True),
        }
        event_dimension = fit_event_dimension_rolling_origin(
            master,
            events,
            predictors=predictors,
            lags=lags,
            model=args.model,
            n_splits=n_splits,
            min_train_events=min_train_groups,
            n_estimators=n_estimators,
            jitter_sigma=args.noise_scale,
            jitter_copies=noise_copies,
            random_state=args.seed,
        )
        if event_dimension.metrics.empty or event_dimension.predictions.empty:
            raise ValueError(
                "F5 nao produziu previsoes rolling-origin de pico/tempo/duracao."
            )
        products.update(
            {
                "event_dimension_metrics": event_dimension.metrics,
                "event_dimension_oos_predictions": event_dimension.predictions,
                "event_dimension_importance_oos": event_dimension.importances,
                "event_dimension_augmentation_provenance": event_dimension.augmentation_provenance,
            }
        )
        classification_gate = (
            products["fold_metrics"]
            .groupby("experiment_horizon_weeks", as_index=False)
            .agg(
                n_oos_folds=("fold", "nunique"),
                mean_skill=("skill_f1_vs_best_baseline", "mean"),
            )
        )
        support_gate = (
            products["independent_support_by_fold"]
            .groupby("experiment_horizon_weeks", as_index=False)
            .agg(
                min_train_el_nino_events=("n_train_el_nino_events", "min"),
                min_train_la_nina_events=("n_train_la_nina_events", "min"),
                independent_support_gate_pass=("independent_support_gate_pass", "all"),
                min_train_active_events_per_type_required=(
                    "min_train_active_events_per_type_required",
                    "max",
                ),
            )
        )
        classification_gate = classification_gate.merge(
            support_gate,
            on="experiment_horizon_weeks",
            how="left",
            validate="one_to_one",
        )
        support_required_for_gate = args.mode == "official"
        gate_rows: list[dict[str, object]] = [
            {
                "component": f"classification_h{int(row.experiment_horizon_weeks):02d}",
                "metric": "mean_skill_f1_vs_best_persistence_or_seasonal",
                "value": float(row.mean_skill),
                "threshold_rule": (
                    ">0; official also requires the predeclared active-event floor "
                    "separately for El Nino and La Nina"
                ),
                "n_oos_units": int(row.n_oos_folds),
                "oos_unit": "whole-event fold",
                "min_train_el_nino_events": int(row.min_train_el_nino_events),
                "min_train_la_nina_events": int(row.min_train_la_nina_events),
                "min_train_active_events_per_type_required": int(
                    row.min_train_active_events_per_type_required
                ),
                "independent_support_gate_pass": bool(
                    row.independent_support_gate_pass
                ),
                "support_required_for_gate": support_required_for_gate,
                "gate_pass": _classification_gate_pass(
                    float(row.mean_skill),
                    bool(row.independent_support_gate_pass),
                    support_required_for_gate=support_required_for_gate,
                ),
            }
            for row in classification_gate.itertuples(index=False)
        ]
        for row in event_dimension.metrics.itertuples(index=False):
            skill = float(row.skill_mae_vs_type_climatology)
            gate_rows.append(
                {
                    "component": f"event_dimension_{row.target}",
                    "metric": "skill_mae_vs_type_climatology",
                    "value": skill,
                    "threshold_rule": ">0",
                    "n_oos_units": int(row.n_oos_events),
                    "oos_unit": "independent ENSO event",
                    "gate_pass": bool(np.isfinite(skill) and skill > 0.0),
                }
            )
        products["scientific_gate"] = pd.DataFrame(gate_rows)
        descriptions = {
            "fold_metrics": "Skill fora da amostra e gate versus persistencia/climatologia sazonal.",
            "oos_predictions": "Probabilidades semanais fora da amostra para os nove estados ENSO.",
            "global_importance": "Importancia global do ajuste de treino por fold.",
            "state_importance_oos": "Ablacao por permutacao no teste, separada por EN/LN e fase.",
            "augmentation_provenance": "Linhagem original_event_id/augmentation_id; sinteticos nao sao eventos independentes.",
            "fold_contract": "Datas, grupos inteiros, embargo e limite de ajuste do preprocessing.",
            "independent_support_by_fold": (
                "Suporte independente EN/LN no treino/teste; blocos neutros ficam separados."
            ),
            "event_dimension_metrics": "Skill por evento para magnitude, tempo ate pico e duracao.",
            "event_dimension_oos_predictions": "Predicoes de evento em folds expansivos sem evento futuro no treino.",
            "event_dimension_importance_oos": "Permutacao OOS das 31 variaveis agrupando todos os lags.",
            "event_dimension_augmentation_provenance": "Cada jitter aponta para o evento original e nunca conta como evento independente.",
            "scientific_gate": (
                "Gate numérico por horizonte e alvo de evento; resultados negativos são preservados."
            ),
        }
        for name, frame in products.items():
            run.write_table(
                name,
                frame,
                description=descriptions[name],
                methods={
                    "model": args.model,
                    "event_grouped": True,
                    "preprocessing_train_only": True,
                    "augmentation_train_only": True,
                },
            )
        gate = classification_gate.set_index("experiment_horizon_weeks")["mean_skill"]
        print(f"[F5] run_id={run.run_id} | mode={args.mode} | model={args.model}")
        print(gate.rename("mean_skill_f1_vs_best_baseline").to_string())
        print(f"[F5] outputs: {run.directory}")
        run.finalize(
            notes=(
                "Gate F5 de classificacao exige skill_f1_vs_best_baseline > 0 e, "
                "no modo oficial, o piso predeclarado separado de eventos El Nino/La Nina; "
                "os tres alvos de evento exigem skill > 0. Horizonte 0 e caracterizacao "
                "diagnostica; horizontes >0 sao rolling-origin."
            )
        )
        return 0
    except Exception as exc:
        run.finalize(status="failed", notes=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
