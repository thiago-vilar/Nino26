#!/usr/bin/env python3
"""Train/evaluate the real F7 PyTorch ConvLSTM with whole-event folds.

Spatial GLORYS channels are fused with the named sequence of all 31 physical F2
variables.  ``smoke`` uses the same real inputs with one epoch and separate
outputs. A execução oficial exige referências F5 completas para comparação,
mas nunca exige que o método anterior tenha resultado positivo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nino_brasil.artifacts import start_artifact_run  # noqa: E402
from nino_brasil.models.event_validation import (  # noqa: E402
    ENSO_PHASES,
    ENSO_STATES,
    ENSO_TYPES,
    canonical_event_ids,
    canonical_state_labels,
    event_phase_sample_weights,
    independent_fold_support,
    parse_state,
)
from nino_brasil.models.phase5_cycle_ml import (  # noqa: E402
    FoldHarmonicPreprocessor,
    physical_predictor_columns,
    valid_phase_target_mask,
)
from nino_brasil.models.phase7_convlstm import (  # noqa: E402
    ENSOSequenceDataset,
    EVENT_INTERVAL_NOMINAL_COVERAGE,
    EVENT_INTERVAL_Z,
    PacificCube,
    augment_sequence_batch,
    build_sequence_event_targets,
    build_convlstm_classifier,
    channel_occlusion_importance,
    event_level_target_table,
    fit_spatial_harmonic_normalizer,
    gaussian_event_equal_audit,
    make_sequence_dataset,
    multitask_loss,
    scalar_occlusion_importance,
    sequence_event_folds,
)

FEATURES = ROOT / "data" / "processed" / "parquet" / "features"
STATISTICS = ROOT / "data" / "processed" / "parquet" / "statistics"
MODEL_BRIDGE = ROOT / "data" / "processed" / "parquet" / "modeling" / "f3_bridge"
DEFAULT_CUBE = ROOT / "data" / "processed" / "zarr" / "modeling" / "phase7_pacific_weekly.zarr"
DEFAULT_EVENTS = MODEL_BRIDGE / "events_en_ln.csv"
EVENT_TARGET_COLUMNS = (
    "peak_magnitude_c",
    "event_time_to_peak_weeks",
    "event_duration_weeks",
)


def _find(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{label} ausente: {candidates}")


def _csv_time(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    time_column = next((name for name in ("week_ending_sunday", "time", "date") if name in frame), None)
    if time_column is None:
        raise KeyError(f"{path}: coluna temporal ausente")
    frame = frame.rename(columns={"event_type": "tipo", "phase": "fase"})
    if missing := required.difference(frame.columns):
        raise KeyError(f"{path}: colunas ausentes {sorted(missing)}")
    frame[time_column] = pd.to_datetime(frame[time_column])
    return frame.set_index(time_column).sort_index()


def _official_f5_references(
    horizon_weeks: int,
    *,
    root: Path | None = None,
) -> list[dict[str, object]]:
    """Return every complete official F5 run for the exact F7 horizon."""

    horizon_weeks = int(horizon_weeks)
    if horizon_weeks <= 0:
        return []
    root = root or ROOT / "data" / "processed" / "runs" / "official" / "fase5"
    runs = sorted(root.glob("F5_*"), reverse=True) if root.exists() else []
    references: list[dict[str, object]] = []
    for run in runs:
        metrics_path = run / "tables" / "fold_metrics.csv"
        predictions_path = run / "tables" / "oos_predictions.csv"
        manifest_path = run / "run_manifest.json"
        if not metrics_path.exists() or not predictions_path.exists() or not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        parameters = manifest.get("parameters", {})
        manifest_phase = pd.to_numeric(
            pd.Series([manifest.get("phase")]), errors="coerce"
        ).iloc[0]
        declared_horizons = pd.to_numeric(
            pd.Series(parameters.get("horizons", []), dtype="object"), errors="coerce"
        ).dropna()
        if (
            manifest.get("status") != "complete"
            or manifest.get("mode") != "official"
            or not np.isfinite(manifest_phase)
            or int(manifest_phase) != 5
            or str(manifest.get("run_id", "")) != run.name
            or horizon_weeks not in declared_horizons.astype(int).tolist()
        ):
            continue
        try:
            metrics = pd.read_csv(metrics_path)
        except (OSError, pd.errors.ParserError):
            continue
        required = {
            "experiment_horizon_weeks",
            "f1_macro_9state",
            "skill_f1_vs_best_baseline",
        }
        if not required.issubset(metrics.columns):
            continue
        metric_horizon = pd.to_numeric(
            metrics["experiment_horizon_weeks"], errors="coerce"
        )
        exact = metrics.loc[metric_horizon.eq(horizon_weeks)].copy()
        skill = pd.to_numeric(exact["skill_f1_vs_best_baseline"], errors="coerce").dropna()
        if skill.empty:
            continue
        references.append(
            {
                "run_id": run.name,
                "run_path": run,
                "model": str(parameters.get("model", "")),
                "metrics_path": metrics_path,
                "predictions_path": predictions_path,
                "mean_f1": float(
                    pd.to_numeric(exact["f1_macro_9state"], errors="coerce").mean()
                ),
                "mean_skill_vs_baseline": float(skill.mean()),
            }
        )
    return references


def _f5_gate_passes(
    horizon_weeks: int,
    *,
    root: Path | None = None,
) -> bool:
    return any(
        float(reference["mean_skill_vs_baseline"]) > 0.0
        for reference in _official_f5_references(horizon_weeks, root=root)
    )


def _paired_f5_comparison(
    f7_predictions: pd.DataFrame,
    references: list[dict[str, object]],
    *,
    horizon_weeks: int,
    minimum_coverage: float = 0.80,
) -> pd.DataFrame:
    """Compare F7 and each eligible F5 model on exactly matching OOS origins."""

    from sklearn.metrics import f1_score

    f7 = f7_predictions.copy()
    f7["origin_time"] = pd.to_datetime(f7["origin_time"])
    if f7["origin_time"].duplicated().any():
        raise ValueError("F7 OOS predictions contain duplicate origin_time rows.")
    columns = [
        "f5_run_id",
        "f5_model",
        "horizon_weeks",
        "n_f7_oos_origins",
        "n_paired_oos_origins",
        "paired_coverage",
        "minimum_paired_coverage",
        "f1_macro_f7_on_paired_origins",
        "f1_macro_f5_on_paired_origins",
        "skill_f1_f7_minus_f5_paired",
        "paired_gate_pass",
    ]
    rows: list[dict[str, object]] = []
    for reference in references:
        f5 = pd.read_csv(Path(reference["predictions_path"]))
        required = {
            "origin_time",
            "observed_state",
            "predicted_state",
            "experiment_horizon_weeks",
        }
        if missing := required.difference(f5.columns):
            raise KeyError(
                f"F5 reference {reference['run_id']} lacks {sorted(missing)}"
            )
        horizon = pd.to_numeric(f5["experiment_horizon_weeks"], errors="coerce")
        f5 = f5.loc[horizon.eq(int(horizon_weeks))].copy()
        if "is_original_observation" in f5:
            original = f5["is_original_observation"].astype(str).str.lower().isin(
                {"true", "1"}
            )
            f5 = f5.loc[original]
        f5["origin_time"] = pd.to_datetime(f5["origin_time"])
        if f5["origin_time"].duplicated().any():
            raise ValueError(
                f"F5 reference {reference['run_id']} has duplicate OOS origins."
            )
        paired = f7.merge(
            f5[["origin_time", "observed_state", "predicted_state"]],
            on="origin_time",
            how="inner",
            suffixes=("_f7", "_f5"),
            validate="one_to_one",
        )
        if not paired["observed_state_f7"].eq(paired["observed_state_f5"]).all():
            raise ValueError(
                f"F5/F7 observed-state mismatch for {reference['run_id']}."
            )
        coverage = len(paired) / len(f7) if len(f7) else 0.0
        if paired.empty:
            f1_f7 = f1_f5 = np.nan
        else:
            truth = paired["observed_state_f7"]
            f1_f7 = float(
                f1_score(
                    truth,
                    paired["predicted_state_f7"],
                    labels=list(ENSO_STATES),
                    average="macro",
                    zero_division=0,
                )
            )
            f1_f5 = float(
                f1_score(
                    truth,
                    paired["predicted_state_f5"],
                    labels=list(ENSO_STATES),
                    average="macro",
                    zero_division=0,
                )
            )
        rows.append(
            {
                "f5_run_id": reference["run_id"],
                "f5_model": reference["model"],
                "horizon_weeks": int(horizon_weeks),
                "n_f7_oos_origins": len(f7),
                "n_paired_oos_origins": len(paired),
                "paired_coverage": coverage,
                "minimum_paired_coverage": minimum_coverage,
                "f1_macro_f7_on_paired_origins": f1_f7,
                "f1_macro_f5_on_paired_origins": f1_f5,
                "skill_f1_f7_minus_f5_paired": f1_f7 - f1_f5,
                "paired_gate_pass": bool(
                    coverage >= minimum_coverage
                    and np.isfinite(f1_f7)
                    and np.isfinite(f1_f5)
                    and f1_f7 > f1_f5
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _scalar_sequences(values: np.ndarray, ends: np.ndarray, seq_len: int) -> np.ndarray:
    return np.stack([values[end - seq_len + 1 : end + 1] for end in ends]).astype("float32")


def _state_components(states: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_to_id = {state: i for i, state in enumerate(ENSO_STATES)}
    type_to_id = {"neutro": 0, "el_nino": 1, "la_nina": 2}
    phase_to_id = {"neutro": 0, **{phase: i + 1 for i, phase in enumerate(ENSO_PHASES)}}
    state_ids, type_ids, phase_ids = [], [], []
    for state in states:
        event_type, phase = parse_state(state)
        state_ids.append(state_to_id[state])
        type_ids.append(type_to_id[event_type])
        phase_ids.append(phase_to_id[phase])
    return np.asarray(state_ids), np.asarray(type_ids), np.asarray(phase_ids)


def _f7_event_dimension_gate(
    fold_metrics: pd.DataFrame,
    event_probabilistic_metrics: pd.DataFrame,
    *,
    minimum_skill: float,
    maximum_absolute_calibration_error: float,
) -> dict[str, object]:
    """Pool F7 event-dimension evidence with equal weight per OOS event."""

    if not 0.0 <= float(maximum_absolute_calibration_error) <= 1.0:
        raise ValueError("maximum_absolute_calibration_error deve estar em [0, 1].")
    probability_required = {
        "target",
        "n_events",
        "interval_coverage_event_equal",
    }
    if missing := probability_required.difference(event_probabilistic_metrics.columns):
        raise KeyError(
            f"event_probabilistic_metrics sem colunas do gate: {sorted(missing)}"
        )
    result: dict[str, object] = {
        "minimum_event_dimension_skill_required": float(minimum_skill),
        "max_interval90_absolute_calibration_error": float(
            maximum_absolute_calibration_error
        ),
    }
    dimension_passes: list[bool] = []
    calibration_passes: list[bool] = []
    for target in EVENT_TARGET_COLUMNS:
        n_column = f"n_test_events_{target}"
        model_column = f"mae_event_equal_{target}"
        baseline_column = f"mae_baseline_event_equal_{target}"
        required = {n_column, model_column, baseline_column}
        if missing := required.difference(fold_metrics.columns):
            raise KeyError(f"fold_metrics sem colunas do gate {target}: {sorted(missing)}")
        n_events = pd.to_numeric(fold_metrics[n_column], errors="coerce")
        model_mae = pd.to_numeric(fold_metrics[model_column], errors="coerce")
        baseline_mae = pd.to_numeric(fold_metrics[baseline_column], errors="coerce")
        valid = n_events.gt(0) & model_mae.notna() & baseline_mae.notna()
        total_events = int(n_events.loc[valid].sum())
        if total_events:
            pooled_model = float(np.average(model_mae.loc[valid], weights=n_events.loc[valid]))
            pooled_baseline = float(
                np.average(baseline_mae.loc[valid], weights=n_events.loc[valid])
            )
            skill = 1.0 - pooled_model / pooled_baseline if pooled_baseline > 0 else np.nan
        else:
            skill = np.nan
        skill_pass = bool(np.isfinite(skill) and skill > float(minimum_skill))
        result[f"n_oos_events_{target}"] = total_events
        result[f"skill_mae_event_equal_{target}"] = skill
        result[f"event_skill_gate_pass_{target}"] = skill_pass
        dimension_passes.append(skill_pass)

        probability = event_probabilistic_metrics.loc[
            event_probabilistic_metrics["target"].eq(target)
        ].copy()
        probability_n = pd.to_numeric(probability["n_events"], errors="coerce")
        probability_coverage = pd.to_numeric(
            probability["interval_coverage_event_equal"], errors="coerce"
        )
        probability_valid = probability_n.gt(0) & probability_coverage.notna()
        probability_total = int(probability_n.loc[probability_valid].sum())
        if probability_total:
            coverage = float(
                np.average(
                    probability_coverage.loc[probability_valid],
                    weights=probability_n.loc[probability_valid],
                )
            )
            calibration_error = abs(coverage - EVENT_INTERVAL_NOMINAL_COVERAGE)
        else:
            coverage = np.nan
            calibration_error = np.nan
        calibration_pass = bool(
            np.isfinite(calibration_error)
            and calibration_error <= float(maximum_absolute_calibration_error)
        )
        result[f"interval90_coverage_event_equal_{target}"] = coverage
        result[f"interval90_absolute_calibration_error_{target}"] = calibration_error
        result[f"interval_calibration_gate_pass_{target}"] = calibration_pass
        calibration_passes.append(calibration_pass)

    result["event_dimension_gate_pass"] = bool(
        dimension_passes and all(dimension_passes)
    )
    result["interval_calibration_gate_pass"] = bool(
        calibration_passes and all(calibration_passes)
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("official", "smoke"), default="official")
    parser.add_argument("--cube", type=Path, default=DEFAULT_CUBE)
    parser.add_argument("--master", type=Path)
    parser.add_argument("--phase-table", type=Path)
    parser.add_argument("--events-table", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--min-train-active-events-per-type", type=int, default=3)
    parser.add_argument("--minimum-event-dimension-skill", type=float, default=0.0)
    parser.add_argument(
        "--max-interval90-absolute-calibration-error", type=float, default=0.15
    )
    parser.add_argument("--research-override-gate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_mode = (
        "smoke"
        if args.mode == "official" and args.research_override_gate
        else args.mode
    )
    if artifact_mode != args.mode:
        print(
            "[F7 exploratoria] override solicitado: o artefato sera isolado em runs/smoke."
        )
    f5_references = (
        _official_f5_references(args.horizon) if args.mode == "official" else []
    )
    eligible_f5_references = f5_references
    if (
        args.mode == "official"
        and not args.research_override_gate
        and not eligible_f5_references
    ):
        print(
            "[F7 bloqueada] nenhum run F5 oficial/completo está disponível "
            f"para comparação no horizonte exato h={args.horizon}."
        )
        return 3
    # Predictive CV must start from physical F2 values.  The source-adjusted
    # companion is a full-sample descriptive product and is never a model input.
    master_path = args.master or _find(
        [FEATURES / "nino34_master_weekly.csv"], "master F2"
    )
    phase_path = args.phase_table or _find(
        [MODEL_BRIDGE / "fases_semanais_en_ln.csv"],
        "fases semanais canonicas F3",
    )
    if not args.cube.exists():
        raise FileNotFoundError(
            f"Cubo F7 ausente: {args.cube}. Rode scripts/build_phase7_pacific_cube.py."
        )
    master = _csv_time(master_path, set())
    phase = _csv_time(phase_path, {"tipo", "fase", "event_id"})
    if not args.events_table.exists():
        raise FileNotFoundError(f"Tabela de eventos F3 ausente: {args.events_table}")
    events = pd.read_csv(args.events_table)
    predictors = physical_predictor_columns(master)
    dataset = xr.open_zarr(args.cube, consolidated=False)
    channel_names = tuple(
        name
        for name in dataset.data_vars
        if name not in {"valid_day_count", "expected_day_count", "complete_week"}
    )
    spatial_times = pd.DatetimeIndex(dataset.time.values)
    common = spatial_times.intersection(master.index).intersection(phase.index).sort_values()
    spatial = dataset[list(channel_names)].sel(time=common).to_array("channel").transpose(
        "time", "lat", "lon", "channel"
    )
    values = np.asarray(spatial.values, dtype="float32")
    cube = PacificCube(
        values=values,
        times=common,
        lat=np.asarray(dataset.lat.values),
        lon=np.asarray(dataset.lon.values),
        channel_names=channel_names,
        finite_mask=np.isfinite(values),
        source_path=str(args.cube),
    )
    phase_common = phase.reindex(common)
    phase_valid = valid_phase_target_mask(phase_common)
    if not bool(phase_valid.all()):
        raise ValueError("A tabela de fases tem tipo/fase ausente no periodo do cubo F7.")
    labels = canonical_state_labels(phase_common).to_numpy()
    sequence_data = make_sequence_dataset(
        cube,
        labels,
        seq_len=args.seq_len,
        horizon=args.horizon,
        event_ids=phase_common["event_id"].fillna("").to_numpy(),
    )
    complete_week = dataset["complete_week"].sel(time=common).to_numpy().astype(bool)
    spatial_sequence_available = np.asarray(
        [
            complete_week[end - args.seq_len + 1 : end + 1].all()
            for end in sequence_data.end_indices
        ],
        dtype=bool,
    )
    event_targets = build_sequence_event_targets(sequence_data, phase, events)
    target_phase = phase.reindex(sequence_data.target_times)
    folds = sequence_event_folds(
        sequence_data,
        target_phase,
        n_splits=2 if args.mode == "smoke" else 5,
        min_train_groups=6 if args.mode == "smoke" else 8,
        min_train_active_events_per_type_for_start=(
            args.min_train_active_events_per_type
            if args.mode == "official"
            else 0
        ),
        seq_len=args.seq_len,
        horizon=args.horizon,
    )
    epochs = 1 if args.mode == "smoke" else args.epochs
    hidden = min(args.hidden, 8) if args.mode == "smoke" else args.hidden
    device = args.device
    import torch
    from sklearn.metrics import f1_score

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    parameters = {
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "hidden": hidden,
        "spatial_channels": channel_names,
        "scalar_channels": predictors,
        "augmentation": not args.no_augmentation,
        "min_train_active_events_per_type": int(
            args.min_train_active_events_per_type
        ),
        "support_balanced_fold_start": args.mode == "official",
        "minimum_event_dimension_skill": float(args.minimum_event_dimension_skill),
        "max_interval90_absolute_calibration_error": float(
            args.max_interval90_absolute_calibration_error
        ),
        "research_override_gate": bool(args.research_override_gate),
        "device": device,
        "f5_reference_run_ids": [
            str(reference["run_id"]) for reference in eligible_f5_references
        ],
    }
    run_inputs = [
        args.cube,
        master_path,
        phase_path,
        args.events_table,
        Path(__file__).resolve(),
    ]
    run_inputs.extend(
        Path(reference["run_path"]) for reference in eligible_f5_references
    )
    run = start_artifact_run(
        7,
        mode=artifact_mode,
        inputs=run_inputs,
        seed=args.seed,
        parameters=parameters,
        command=" ".join([sys.executable, *sys.argv]),
    )
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    scalar_importance: list[pd.DataFrame] = []
    spatial_importance: list[pd.DataFrame] = []
    provenance: list[pd.DataFrame] = []
    event_probabilistic_summary: list[dict[str, object]] = []
    event_probabilistic_by_event: list[pd.DataFrame] = []
    contracts: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    try:
        for fold_no, fold in enumerate(folds, start=1):
            raw_train_mask = cube.times <= fold.train_end - pd.Timedelta(weeks=args.horizon)
            spatial_normalizer = fit_spatial_harmonic_normalizer(
                cube.values, cube.times, raw_train_mask
            )
            normalized_spatial = spatial_normalizer.transform(cube.values, cube.times)
            spatial_sequences = np.stack(
                [
                    normalized_spatial[end - args.seq_len + 1 : end + 1]
                    for end in sequence_data.end_indices
                ]
            ).astype("float32")
            scalar_fit_end = fold.train_end - pd.Timedelta(weeks=args.horizon)
            scalar_history = master.loc[:scalar_fit_end]
            scalar_preprocessor = FoldHarmonicPreprocessor(tuple(predictors)).fit(
                scalar_history[predictors],
                source_codes=scalar_history.get("ocean_source_code"),
            )
            scalar_aligned = master.reindex(cube.times)
            scalar_values = scalar_preprocessor.transform(
                scalar_aligned[predictors],
                source_codes=scalar_aligned.get("ocean_source_code"),
            )
            scalar_mean = scalar_values.loc[:scalar_fit_end].mean()
            scalar_std = scalar_values.loc[:scalar_fit_end].std().replace(0, 1)
            scalar_values = (scalar_values - scalar_mean) / scalar_std
            scalar_sequences = _scalar_sequences(
                scalar_values.to_numpy(dtype="float32"), sequence_data.end_indices, args.seq_len
            )
            scalar_available = np.isfinite(scalar_sequences).all(axis=(1, 2))
            model_input_available = scalar_available & spatial_sequence_available
            train = fold.train_index[model_input_available[fold.train_index]]
            test = fold.test_index[model_input_available[fold.test_index]]
            if not len(train) or not len(test):
                continue
            train_states = sequence_data.targets[train].astype(str)
            test_states = sequence_data.targets[test].astype(str)
            y_state, y_type, y_phase = _state_components(train_states)
            test_state_ids, _, _ = _state_components(test_states)
            target_phase_all = phase.reindex(sequence_data.target_times)
            support = independent_fold_support(
                target_phase_all.iloc[train],
                target_phase_all.iloc[test],
                fold=fold.name,
                min_train_active_events_per_type=args.min_train_active_events_per_type,
            )
            support_rows.append(support)
            weights = event_phase_sample_weights(
                target_phase_all.iloc[train]
            ).to_numpy(dtype="float32")
            train_event_frame = event_targets.iloc[train]
            independent_train_events = event_level_target_table(
                train_event_frame,
                EVENT_TARGET_COLUMNS,
            )
            if independent_train_events.empty:
                raise ValueError(
                    f"{fold.name}: nenhum evento ativo no treino para os targets F7."
                )
            event_mean = independent_train_events.loc[:, EVENT_TARGET_COLUMNS].mean()
            event_std = (
                independent_train_events.loc[:, EVENT_TARGET_COLUMNS]
                .std(ddof=1)
                .replace(0.0, 1.0)
                .fillna(1.0)
            )
            event_values_all = event_targets.loc[:, EVENT_TARGET_COLUMNS]
            event_mask_all = np.isfinite(event_values_all.to_numpy(dtype="float32"))
            event_scaled_all = (
                (event_values_all - event_mean) / event_std
            ).to_numpy(dtype="float32")
            event_scaled_all = np.nan_to_num(event_scaled_all, nan=0.0)
            y_event = event_scaled_all[train]
            y_event_mask = event_mask_all[train].astype("float32")
            Xsp = spatial_sequences[train]
            Xsc = scalar_sequences[train]
            validation_group_ids = canonical_event_ids(
                target_phase_all.iloc[train]
            ).to_numpy(dtype=object)
            if not args.no_augmentation:
                augmented = augment_sequence_batch(
                    Xsp,
                    validation_group_ids,
                    sample_times=sequence_data.target_times[train],
                    origin_times=sequence_data.origin_times[train],
                    states=train_states,
                    random_state=args.seed + fold_no,
                )
                scalar_scale = np.std(Xsc, axis=(0, 1), keepdims=True)
                scalar_aug = Xsc + np.random.default_rng(args.seed + fold_no).normal(
                    0.0, 0.01, Xsc.shape
                ).astype("float32") * scalar_scale
                Xsp = np.concatenate([Xsp, augmented.values])
                Xsc = np.concatenate([Xsc, scalar_aug])
                y_state = np.tile(y_state, 2)
                y_type = np.tile(y_type, 2)
                y_phase = np.tile(y_phase, 2)
                y_event = np.tile(y_event, (2, 1))
                y_event_mask = np.tile(y_event_mask, (2, 1))
                weights = np.tile(weights / 2.0, 2)
                augmented.provenance.insert(0, "fold", fold.name)
                provenance.append(augmented.provenance)
            model = build_convlstm_classifier(
                (args.seq_len, cube.values.shape[1], cube.values.shape[2], len(channel_names)),
                n_classes=len(ENSO_STATES),
                filters=(hidden,),
                scalar_channels=len(predictors),
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
            order = np.arange(len(Xsp))
            model.train()
            for epoch in range(epochs):
                np.random.default_rng(args.seed + epoch + fold_no).shuffle(order)
                for start in range(0, len(order), args.batch_size):
                    batch = order[start : start + args.batch_size]
                    spatial_batch = torch.as_tensor(Xsp[batch], dtype=torch.float32, device=device)
                    scalar_batch = torch.as_tensor(Xsc[batch], dtype=torch.float32, device=device)
                    output = model(spatial_batch, scalar_sequence=scalar_batch)
                    batch_weight = torch.as_tensor(weights[batch], dtype=torch.float32, device=device)
                    loss, _ = multitask_loss(
                        output,
                        state_target=torch.as_tensor(
                            y_state[batch], dtype=torch.long, device=device
                        ),
                        type_target=torch.as_tensor(
                            y_type[batch], dtype=torch.long, device=device
                        ),
                        phase_target=torch.as_tensor(
                            y_phase[batch], dtype=torch.long, device=device
                        ),
                        event_target=torch.as_tensor(
                            y_event[batch], dtype=torch.float32, device=device
                        ),
                        event_mask=torch.as_tensor(
                            y_event_mask[batch], dtype=torch.float32, device=device
                        ),
                        sample_weight=batch_weight,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            model.eval()
            test_spatial = spatial_sequences[test]
            test_scalar = scalar_sequences[test]
            with torch.no_grad():
                test_output = model(
                    torch.as_tensor(test_spatial, dtype=torch.float32, device=device),
                    scalar_sequence=torch.as_tensor(test_scalar, dtype=torch.float32, device=device),
                )
                logits = test_output["state_logits"]
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                event_prediction = (
                    test_output["event_mean"].cpu().numpy()
                    * event_std.to_numpy(dtype="float32")[None]
                    + event_mean.to_numpy(dtype="float32")[None]
                )
                event_log_scale_standardized = (
                    test_output["event_log_scale"].cpu().numpy()
                )
                event_prediction_scale = (
                    np.exp(event_log_scale_standardized)
                    * event_std.to_numpy(dtype="float32")[None]
                )
            predicted_ids = probabilities.argmax(axis=1)
            predicted_states = np.asarray(ENSO_STATES, dtype=object)[predicted_ids]
            train_target_times = sequence_data.target_times[train]
            test_target_times = sequence_data.target_times[test]
            modal_week = pd.Series(train_states, index=train_target_times.isocalendar().week).groupby(level=0).agg(
                lambda values: values.mode().iloc[0]
            )
            fallback = pd.Series(train_states).mode().iloc[0]
            seasonal = np.asarray([modal_week.get(week, fallback) for week in test_target_times.isocalendar().week])
            origin_states = canonical_state_labels(phase.reindex(sequence_data.origin_times[test])).to_numpy()
            f1 = f1_score(test_states, predicted_states, labels=list(ENSO_STATES), average="macro", zero_division=0)
            f1_seasonal = f1_score(test_states, seasonal, labels=list(ENSO_STATES), average="macro", zero_division=0)
            f1_persistence = f1_score(test_states, origin_states, labels=list(ENSO_STATES), average="macro", zero_division=0)
            metric_row = {
                    "fold": fold.name,
                    "n_train_sequences_original": len(train),
                    "n_train_sequences_optimisation": len(Xsp),
                    "n_test_sequences": len(test),
                    "n_test_events": target_phase_all.iloc[test]["event_id"].replace("", np.nan).nunique(),
                    "f1_macro_9state": f1,
                    "f1_baseline_seasonal": f1_seasonal,
                    "f1_baseline_persistence": f1_persistence,
                    "skill_f1_vs_best_baseline": f1 - max(f1_seasonal, f1_persistence),
                    "gate_pass": bool(f1 > max(f1_seasonal, f1_persistence)),
                    "event_regression_unit": "event-equal mean absolute error",
                }
            test_event_frame = event_targets.iloc[test]
            event_row_audit: dict[str, dict[str, np.ndarray]] = {}
            for target_no, target_name in enumerate(EVENT_TARGET_COLUMNS):
                observed_event = pd.to_numeric(
                    test_event_frame[target_name], errors="coerce"
                ).to_numpy(dtype=float)
                predicted_event = event_prediction[:, target_no]
                baseline_event = np.full(
                    len(test), float(event_mean[target_name]), dtype=float
                )
                event_ids_for_metric = test_event_frame["event_id"].fillna("").astype(str)
                valid_event = (
                    np.isfinite(observed_event)
                    & np.isfinite(predicted_event)
                    & event_ids_for_metric.ne("").to_numpy()
                )
                if valid_event.any():
                    errors = pd.DataFrame(
                        {
                            "event_id": event_ids_for_metric.to_numpy()[valid_event],
                            "model": np.abs(
                                predicted_event[valid_event] - observed_event[valid_event]
                            ),
                            "baseline": np.abs(
                                baseline_event[valid_event] - observed_event[valid_event]
                            ),
                        }
                    ).groupby("event_id", sort=False).mean()
                    mae_model = float(errors["model"].mean())
                    mae_baseline = float(errors["baseline"].mean())
                    n_test_events_target = int(len(errors))
                else:
                    mae_model = mae_baseline = np.nan
                    n_test_events_target = 0
                metric_row[f"n_test_events_{target_name}"] = n_test_events_target
                metric_row[f"mae_event_equal_{target_name}"] = mae_model
                metric_row[f"mae_baseline_event_equal_{target_name}"] = mae_baseline
                metric_row[f"skill_mae_{target_name}"] = (
                    1.0 - mae_model / mae_baseline
                    if np.isfinite(mae_model)
                    and np.isfinite(mae_baseline)
                    and mae_baseline > 0
                    else np.nan
                )
                predicted_scale = event_prediction_scale[:, target_no]
                lower = predicted_event - EVENT_INTERVAL_Z * predicted_scale
                upper = predicted_event + EVENT_INTERVAL_Z * predicted_scale
                row_nll = np.full(len(test), np.nan, dtype=float)
                row_covered = np.full(len(test), np.nan, dtype=float)
                probabilistic_valid = (
                    np.isfinite(observed_event)
                    & np.isfinite(predicted_event)
                    & np.isfinite(predicted_scale)
                    & (predicted_scale > 0.0)
                    & event_ids_for_metric.ne("").to_numpy()
                )
                if probabilistic_valid.any():
                    standardised = (
                        observed_event[probabilistic_valid]
                        - predicted_event[probabilistic_valid]
                    ) / predicted_scale[probabilistic_valid]
                    row_nll[probabilistic_valid] = (
                        0.5 * standardised**2
                        + np.log(predicted_scale[probabilistic_valid])
                        + 0.5 * np.log(2.0 * np.pi)
                    )
                    row_covered[probabilistic_valid] = (
                        (observed_event[probabilistic_valid] >= lower[probabilistic_valid])
                        & (observed_event[probabilistic_valid] <= upper[probabilistic_valid])
                    ).astype(float)
                by_event, audit = gaussian_event_equal_audit(
                    observed_event,
                    predicted_event,
                    predicted_scale,
                    event_ids_for_metric.to_numpy(),
                    nominal_coverage=EVENT_INTERVAL_NOMINAL_COVERAGE,
                )
                audit_row: dict[str, object] = {
                    "fold": fold.name,
                    "target": target_name,
                    **audit,
                    "aggregation_unit": "independent_event_equal",
                    "distribution": "gaussian",
                }
                event_probabilistic_summary.append(audit_row)
                if not by_event.empty:
                    by_event.insert(0, "target", target_name)
                    by_event.insert(0, "fold", fold.name)
                    by_event["interval_nominal_coverage"] = (
                        EVENT_INTERVAL_NOMINAL_COVERAGE
                    )
                    by_event["interval_calibration_error"] = (
                        by_event["interval_coverage"]
                        - EVENT_INTERVAL_NOMINAL_COVERAGE
                    )
                    event_probabilistic_by_event.append(by_event)
                metric_row[f"gaussian_nll_event_equal_{target_name}"] = audit[
                    "gaussian_nll_event_equal"
                ]
                metric_row[f"interval90_coverage_event_equal_{target_name}"] = audit[
                    "interval_coverage_event_equal"
                ]
                metric_row[f"interval90_calibration_error_{target_name}"] = audit[
                    "interval_calibration_error"
                ]
                metric_row[f"interval90_mean_width_event_equal_{target_name}"] = audit[
                    "mean_interval_width_event_equal"
                ]
                event_row_audit[target_name] = {
                    "scale": predicted_scale,
                    "log_scale_standardized": event_log_scale_standardized[:, target_no],
                    "lower": lower,
                    "upper": upper,
                    "nll": row_nll,
                    "covered": row_covered,
                }
            metric_rows.append(metric_row)
            target_events = target_phase_all.iloc[test]["event_id"].fillna("").to_numpy()
            for row_no, target_time in enumerate(test_target_times):
                prediction_rows.append(
                    {
                        "fold": fold.name,
                        "origin_time": sequence_data.origin_times[test][row_no],
                        "target_time": target_time,
                        "event_id": target_events[row_no],
                        "observed_state": test_states[row_no],
                        "predicted_state": predicted_states[row_no],
                        **{
                            f"observed_{target_name}": float(
                                test_event_frame.iloc[row_no][target_name]
                            )
                            if pd.notna(test_event_frame.iloc[row_no][target_name])
                            else np.nan
                            for target_name in EVENT_TARGET_COLUMNS
                        },
                        **{
                            f"predicted_{target_name}": float(
                                event_prediction[row_no, target_no]
                            )
                            for target_no, target_name in enumerate(EVENT_TARGET_COLUMNS)
                        },
                        **{
                            f"predicted_scale_{target_name}": float(
                                event_row_audit[target_name]["scale"][row_no]
                            )
                            for target_name in EVENT_TARGET_COLUMNS
                        },
                        **{
                            f"predicted_log_scale_standardized_{target_name}": float(
                                event_row_audit[target_name]["log_scale_standardized"][row_no]
                            )
                            for target_name in EVENT_TARGET_COLUMNS
                        },
                        **{
                            f"predicted_lower90_{target_name}": float(
                                event_row_audit[target_name]["lower"][row_no]
                            )
                            for target_name in EVENT_TARGET_COLUMNS
                        },
                        **{
                            f"predicted_upper90_{target_name}": float(
                                event_row_audit[target_name]["upper"][row_no]
                            )
                            for target_name in EVENT_TARGET_COLUMNS
                        },
                        **{
                            f"gaussian_nll_{target_name}": float(
                                event_row_audit[target_name]["nll"][row_no]
                            )
                            if np.isfinite(event_row_audit[target_name]["nll"][row_no])
                            else np.nan
                            for target_name in EVENT_TARGET_COLUMNS
                        },
                        **{
                            f"interval90_covered_{target_name}": float(
                                event_row_audit[target_name]["covered"][row_no]
                            )
                            if np.isfinite(event_row_audit[target_name]["covered"][row_no])
                            else np.nan
                            for target_name in EVENT_TARGET_COLUMNS
                        },
                        **{
                            f"prob_{state}": float(probabilities[row_no, state_id])
                            for state_id, state in enumerate(ENSO_STATES)
                        },
                    }
                )
            max_xai = min(len(test), 32 if args.mode == "smoke" else len(test))
            scalar_imp = scalar_occlusion_importance(
                model,
                test_spatial[:max_xai],
                test_scalar[:max_xai],
                predictors,
                target_state_ids=test_state_ids[:max_xai],
                device=device,
            )
            scalar_imp.insert(0, "fold", fold.name)
            scalar_importance.append(scalar_imp)
            # Spatial channel XAI uses a zero scalar reference through a small wrapper.
            class FixedScalar(torch.nn.Module):
                def __init__(self, base, scalar):
                    super().__init__(); self.base = base; self.register_buffer("scalar", scalar)
                def forward(self, spatial_values):
                    return self.base(spatial_values, scalar_sequence=self.scalar[: len(spatial_values)])
            wrapper = FixedScalar(
                model,
                torch.as_tensor(test_scalar[:max_xai], dtype=torch.float32, device=device),
            )
            spatial_imp = channel_occlusion_importance(
                wrapper,
                test_spatial[:max_xai],
                channel_names=channel_names,
                target_state_ids=test_state_ids[:max_xai],
                device=device,
            )
            spatial_imp.insert(0, "fold", fold.name)
            spatial_importance.append(spatial_imp)
            model_path = run.directory / "models" / f"{fold.name}.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "parameters": parameters,
                    "event_target_mean": event_mean.to_dict(),
                    "event_target_std": event_std.to_dict(),
                    "event_interval_nominal_coverage": EVENT_INTERVAL_NOMINAL_COVERAGE,
                },
                model_path,
            )
            run.register_file(model_path, role="pytorch_state_dict", description=f"F7 {fold.name}")
            contracts.append(
                {
                    "fold": fold.name,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "purge_weeks": fold.purge_weeks,
                    "spatial_preprocessing_train_only": True,
                    "scalar_preprocessing_train_only": True,
                    "input_availability_filter": True,
                    "n_train_excluded_incomplete_spatial_window": int(
                        (~spatial_sequence_available[fold.train_index]).sum()
                    ),
                    "n_test_excluded_incomplete_spatial_window": int(
                        (~spatial_sequence_available[fold.test_index]).sum()
                    ),
                    "n_train_excluded_scalar_missing": int(
                        (~scalar_available[fold.train_index]).sum()
                    ),
                    "n_test_excluded_scalar_missing": int(
                        (~scalar_available[fold.test_index]).sum()
                    ),
                    "whole_event_split": True,
                    "n_train_el_nino_events": support["n_train_el_nino_events"],
                    "n_train_la_nina_events": support["n_train_la_nina_events"],
                    "n_train_neutral_blocks": support["n_train_neutral_blocks"],
                    "min_train_active_events_per_type_required": support[
                        "min_train_active_events_per_type_required"
                    ],
                    "independent_support_gate_pass": support[
                        "independent_support_gate_pass"
                    ],
                    "multitask_event_targets": ";".join(EVENT_TARGET_COLUMNS),
                    "event_target_normalization_train_only": True,
                    "event_regression_metric_unit": "event_equal",
                }
            )
        fold_metrics = pd.DataFrame(metric_rows)
        oos_predictions = pd.DataFrame(prediction_rows)
        event_probabilistic_metrics = pd.DataFrame(event_probabilistic_summary)
        independent_support_by_fold = pd.DataFrame(support_rows)
        event_dimension_gate = _f7_event_dimension_gate(
            fold_metrics,
            event_probabilistic_metrics,
            minimum_skill=args.minimum_event_dimension_skill,
            maximum_absolute_calibration_error=(
                args.max_interval90_absolute_calibration_error
            ),
        )
        independent_support_gate_pass = bool(
            not independent_support_by_fold.empty
            and independent_support_by_fold["independent_support_gate_pass"]
            .astype(bool)
            .all()
        )
        support_required_for_gate = artifact_mode == "official"
        paired_f5 = _paired_f5_comparison(
            oos_predictions,
            eligible_f5_references,
            horizon_weeks=args.horizon,
        )
        mean_skill_vs_baseline = float(
            pd.to_numeric(
                fold_metrics["skill_f1_vs_best_baseline"], errors="coerce"
            ).mean()
        )
        if paired_f5.empty:
            best_f5_run_id = ""
            paired_skill_vs_best_f5 = np.nan
            paired_coverage = np.nan
            paired_f5_gate = args.mode != "official"
        else:
            best_row = paired_f5.sort_values(
                ["f1_macro_f5_on_paired_origins", "f5_run_id"],
                ascending=[False, True],
                kind="mergesort",
            ).iloc[0]
            best_f5_run_id = str(best_row["f5_run_id"])
            paired_skill_vs_best_f5 = float(
                best_row["skill_f1_f7_minus_f5_paired"]
            )
            paired_coverage = float(best_row["paired_coverage"])
            paired_f5_gate = bool(best_row["paired_gate_pass"])
        scientific_gate = pd.DataFrame(
            [
                {
                    "horizon_weeks": int(args.horizon),
                    "mean_skill_f1_vs_best_persistence_or_seasonal": mean_skill_vs_baseline,
                    "baseline_gate_pass": bool(mean_skill_vs_baseline > 0.0),
                    "best_f5_reference_run_id": best_f5_run_id,
                    "paired_coverage_best_f5": paired_coverage,
                    "skill_f1_f7_minus_best_f5_paired": paired_skill_vs_best_f5,
                    "paired_f5_gate_pass": paired_f5_gate,
                    "f5_comparison_required": args.mode == "official",
                    "min_train_active_events_per_type_required": int(
                        args.min_train_active_events_per_type
                    ),
                    "min_train_el_nino_events": int(
                        independent_support_by_fold["n_train_el_nino_events"].min()
                    ),
                    "min_train_la_nina_events": int(
                        independent_support_by_fold["n_train_la_nina_events"].min()
                    ),
                    "independent_support_gate_pass": independent_support_gate_pass,
                    "support_required_for_gate": support_required_for_gate,
                    **event_dimension_gate,
                    "scientific_gate_pass": bool(
                        mean_skill_vs_baseline > 0.0
                        and paired_f5_gate
                        and bool(event_dimension_gate["event_dimension_gate_pass"])
                        and bool(
                            event_dimension_gate["interval_calibration_gate_pass"]
                        )
                        and (
                            not support_required_for_gate
                            or independent_support_gate_pass
                        )
                    ),
                    "comparison_contract": (
                        "same OOS origin_time and observed state; best eligible RF/XGB F5; "
                        "minimum paired coverage 0.80; all three event dimensions must beat "
                        "their event-equal baselines; 90% intervals must meet the predeclared "
                        "absolute calibration tolerance"
                    ),
                }
            ]
        )
        products = {
            "fold_metrics": fold_metrics,
            "oos_predictions": oos_predictions,
            "scalar_variable_importance_oos": pd.concat(scalar_importance, ignore_index=True),
            "spatial_channel_importance_oos": pd.concat(spatial_importance, ignore_index=True),
            "augmentation_provenance": pd.concat(provenance, ignore_index=True) if provenance else pd.DataFrame(),
            "event_probabilistic_metrics": event_probabilistic_metrics,
            "event_probabilistic_metrics_by_event": (
                pd.concat(event_probabilistic_by_event, ignore_index=True)
                if event_probabilistic_by_event
                else pd.DataFrame()
            ),
            "fold_contract": pd.DataFrame(contracts),
            "independent_support_by_fold": independent_support_by_fold,
            "paired_f5_comparison": paired_f5,
            "scientific_gate": scientific_gate,
        }
        for name, frame in products.items():
            run.write_table(
                name,
                frame,
                description={
                    "fold_metrics": "Skill F7 fora da amostra contra persistencia e climatologia.",
                    "oos_predictions": "Probabilidades dos nove estados por origem/horizonte.",
                    "scalar_variable_importance_oos": "Ablacao das 31 variaveis F2 no teste.",
                    "spatial_channel_importance_oos": "Ablacao dos campos GLORYS no teste.",
                    "augmentation_provenance": "Linhagem de masking/noise apenas no treino.",
                    "event_probabilistic_metrics": (
                        "NLL, cobertura, largura e calibracao gaussianas com peso igual por evento."
                    ),
                    "event_probabilistic_metrics_by_event": (
                        "Metricas probabilisticas OOS antes da media entre eventos independentes."
                    ),
                    "fold_contract": "Folds inteiros por evento e limites do preprocessing.",
                    "independent_support_by_fold": (
                        "Eventos EN/LN independentes no treino/teste; blocos neutros separados."
                    ),
                    "paired_f5_comparison": (
                        "Comparacao RF/XGB F5 versus F7 nas mesmas origens OOS e mesmo horizonte."
                    ),
                    "scientific_gate": (
                        "Gate conjunto: classificacao, F5 pareado, suporte EN/LN, tres dimensoes "
                        "de evento e calibracao predeclarada dos intervalos de 90%."
                    ),
                }[name],
                methods={"framework": "PyTorch", "architecture": "ConvLSTM+GRU scalar fusion"},
            )
        gate_pass = bool(scientific_gate["scientific_gate_pass"].iloc[0])
        run.finalize(
            notes=(
                "F7 scientific_gate_pass="
                f"{gate_pass}; exige media OOS positiva contra baseline e F5 pareado."
            )
        )
        print(f"[F7] run_id={run.run_id} | outputs={run.directory}")
        return 0
    except Exception as exc:
        run.finalize(status="failed", notes=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
