#!/usr/bin/env python3
"""Train/evaluate F8 ConvLSTM on the exact native CHIRPS grid.

The Pacific spatial encoder is fused with all 31 named F2 variables and decoded
convolutionally to the unchanged CHIRPS latitude/longitude shape.  Loss uses a
Brazil-overlap mask and area weights; predictions retain every original pixel.
Official execution requires a complete F6 reference on the same native grid,
regardless of whether F6 passed its own scientific gate. Its confirmatory gate
requires calibrated central 90% prediction intervals,
audited with equal weight per independent event rather than per pixel-week.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
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
    ENSO_TYPES,
    canonical_state_labels,
    event_phase_sample_weights,
)
from nino_brasil.models.phase5_cycle_ml import (  # noqa: E402
    FoldHarmonicPreprocessor,
    physical_predictor_columns,
    valid_phase_target_mask,
)
from nino_brasil.models.phase6_brazil_ml import (  # noqa: E402
    CANONICAL_ACTIVE_CONDITIONS,
    causal_persistence_target,
    fit_fold_target_transformer,
)
from nino_brasil.models.phase7_convlstm import (  # noqa: E402
    EVENT_INTERVAL_NOMINAL_COVERAGE,
    EVENT_INTERVAL_Z,
    PacificCube,
    fit_spatial_harmonic_normalizer,
    make_sequence_dataset,
    sequence_source_phase_table,
    sequence_event_folds,
)
from nino_brasil.models.phase8_convlstm_brazil import (  # noqa: E402
    brazil_cell_area_weights,
    build_convlstm_encoder_decoder,
    field_channel_occlusion_importance,
    masked_area_weighted_gaussian_nll,
    masked_area_weighted_quantile_loss,
    native_chirps_grid,
    pixel_skill_table,
)
from nino_brasil.targets.chirps_native import target_to_frame, validate_native_target  # noqa: E402

FEATURES = ROOT / "data" / "processed" / "parquet" / "features"
STATISTICS = ROOT / "data" / "processed" / "parquet" / "statistics"
MODEL_BRIDGE = ROOT / "data" / "processed" / "parquet" / "modeling" / "f3_bridge"
PACIFIC = ROOT / "data" / "processed" / "zarr" / "modeling" / "phase7_pacific_weekly.zarr"
CHIRPS = ROOT / "data" / "processed" / "zarr" / "features" / "chirps_native_weekly_targets.zarr"
F8_MAX_INTERVAL90_ABSOLUTE_CALIBRATION_ERROR = 0.15
F8_INTERVAL_AGGREGATION_UNIT = (
    "area_weighted_native_pixel_week_within_event_then_independent_event_equal"
)


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


def _find_f6_reference_run(
    *, horizon_weeks: int, grid_sha256: str, target_variable: str
) -> tuple[Path, str, str, str] | None:
    root = ROOT / "data" / "processed" / "runs" / "official" / "fase6"
    for run in sorted(root.glob("F6_*"), reverse=True) if root.exists() else []:
        field_gate = run / "tables" / "field_gate.csv"
        manifest_path = run / "run_manifest.json"
        if not field_gate.exists() or not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        parameters = manifest.get("parameters", {})
        if (
            manifest.get("status") != "complete"
            or parameters.get("role") != "merge_pixel_shards_and_field_gate"
            or str(parameters.get("parent_grid_sha256", "")) != str(grid_sha256)
        ):
            continue
        table = pd.read_csv(field_gate)
        required = {
            "lag_weeks",
            "condition",
            "model",
            "target_variable",
            "target_transform",
            "target_units",
            "pixel_coverage_fraction",
        }
        if not required.issubset(table.columns):
            continue
        table = table.loc[
            pd.to_numeric(table["lag_weeks"], errors="coerce").eq(horizon_weeks)
            & table["target_variable"].astype(str).eq(target_variable)
            & pd.to_numeric(table.get("pixel_coverage_fraction"), errors="coerce").eq(1.0)
        ]
        for (model, transform, units), group in table.groupby(
            ["model", "target_transform", "target_units"], dropna=False
        ):
            passed = set(group["condition"].astype(str))
            if set(CANONICAL_ACTIVE_CONDITIONS).issubset(passed):
                predictions = run / "tables" / "pixel_oos_predictions.csv"
                if predictions.exists():
                    return run, str(model), str(transform), str(units)
    return None


def _native_target_bounds(target_variable: str) -> tuple[float | None, float | None, str]:
    """Predeclared physical support used only after inverse transformation."""

    name = str(target_variable).lower()
    if "percentile" in name:
        return 0.0, 1.0, "bounded percentile [0,1]"
    if name.startswith(("cdd_", "cwd_")) and "anomaly" not in name and "robust_z" not in name:
        return 0.0, 7.0, "within-week spell length [0,7] days"
    if (
        name == "precip_weekly_mm"
        or name.startswith(("rx1day_", "rx5day_", "r95p_", "r99p_"))
    ) and "anomaly" not in name and "robust_z" not in name:
        return 0.0, None, "non-negative precipitation/extreme amount"
    return None, None, "unbounded anomaly/index"


def _clip_native_support(
    values: np.ndarray, lower: float | None, upper: float | None
) -> np.ndarray:
    output = np.asarray(values, dtype="float32")
    if lower is not None:
        output = np.maximum(output, lower)
    if upper is not None:
        output = np.minimum(output, upper)
    return output


def _scalar_sequences(values: np.ndarray, ends: np.ndarray, seq_len: int) -> np.ndarray:
    return np.stack([values[end - seq_len + 1 : end + 1] for end in ends]).astype("float32")


def _quantiles(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if (
        len(values) < 3
        or len(set(values)) != len(values)
        or sorted(values) != list(values)
        or not all(0 < item < 1 for item in values)
    ):
        raise argparse.ArgumentTypeError(
            "quantis devem ser crescentes em (0,1), ex. 0.05,0.5,0.95"
        )
    if 0.5 not in values:
        raise argparse.ArgumentTypeError("inclua o quantil 0.5 para a previsao central")
    return values


def _validate_interval90_contract(args: argparse.Namespace) -> None:
    """Validate the predeclared central-IC90 contract before any model work."""

    tolerance = float(args.max_interval90_absolute_calibration_error)
    if not 0.0 <= tolerance <= F8_MAX_INTERVAL90_ABSOLUTE_CALIBRATION_ERROR:
        raise ValueError(
            "--max-interval90-absolute-calibration-error deve pertencer a "
            f"[0,{F8_MAX_INTERVAL90_ABSOLUTE_CALIBRATION_ERROR}]; "
            "o protocolo confirmatorio nao aceita tolerancia mais frouxa."
        )
    if args.distribution != "quantile":
        return
    tail_probability = (1.0 - EVENT_INTERVAL_NOMINAL_COVERAGE) / 2.0
    expected = (tail_probability, 1.0 - tail_probability)
    actual = (float(args.quantiles[0]), float(args.quantiles[-1]))
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-12):
        raise ValueError(
            "O gate F8 exige IC90 central: os quantis externos devem ser "
            f"{expected[0]:.2f} e {expected[1]:.2f}; recebidos {actual}."
        )


def _within_calibration_tolerance(error: float, maximum_error: float) -> bool:
    return bool(
        np.isfinite(error)
        and float(error) <= float(maximum_error) + 8.0 * np.finfo(float).eps
    )


def _f8_interval_calibration_audit(
    components: pd.DataFrame,
    *,
    conditions: Sequence[str] = CANONICAL_ACTIVE_CONDITIONS,
    nominal_coverage: float = EVENT_INTERVAL_NOMINAL_COVERAGE,
    maximum_absolute_calibration_error: float = (
        F8_MAX_INTERVAL90_ABSOLUTE_CALIBRATION_ERROR
    ),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit F8 IC90 calibration without counting pixels or weeks as events.

    Pixel-weeks are area-weighted only *inside* each independent ENSO event.
    Event coverages are then averaged with equal event weight for each
    fold-condition and for each condition across folds.  Both aggregation
    levels must meet the predeclared absolute calibration tolerance.
    """

    nominal_coverage = float(nominal_coverage)
    maximum_error = float(maximum_absolute_calibration_error)
    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage deve pertencer a (0,1).")
    if not 0.0 <= maximum_error <= 1.0:
        raise ValueError("maximum_absolute_calibration_error deve estar em [0,1].")
    required = {
        "fold",
        "condition",
        "event_id",
        "interval90_coverage_event",
        "interval90_value_coverage_fraction",
    }
    event_audit = components.copy()
    if event_audit.empty:
        for column in required.difference(event_audit.columns):
            event_audit[column] = pd.Series(dtype="object")
    elif missing := required.difference(event_audit.columns):
        raise KeyError(f"componentes F8 sem colunas de calibracao: {sorted(missing)}")

    condition_order = tuple(str(value) for value in conditions)
    event_audit = event_audit.loc[
        event_audit["condition"].astype(str).isin(condition_order)
    ].copy()
    event_audit["condition"] = event_audit["condition"].astype(str)
    event_audit["event_id"] = event_audit["event_id"].astype(str)
    duplicate = event_audit.duplicated(["condition", "event_id"], keep=False)
    if duplicate.any():
        examples = (
            event_audit.loc[duplicate, ["condition", "event_id", "fold"]]
            .astype(str)
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "Evento repetido entre folds/linhas F8; whole-event OOS violado: "
            f"{examples}"
        )
    event_audit["interval90_coverage_event"] = pd.to_numeric(
        event_audit["interval90_coverage_event"], errors="coerce"
    )
    event_audit["interval90_value_coverage_fraction"] = pd.to_numeric(
        event_audit["interval90_value_coverage_fraction"], errors="coerce"
    )
    event_audit["interval90_nominal_coverage"] = nominal_coverage
    event_audit["max_interval90_absolute_calibration_error"] = maximum_error
    event_audit["interval90_absolute_calibration_error_event"] = (
        event_audit["interval90_coverage_event"] - nominal_coverage
    ).abs()
    event_audit["interval_aggregation_unit"] = F8_INTERVAL_AGGREGATION_UNIT

    fold_columns = [
        "fold",
        "condition",
        "n_independent_events",
        "n_events_with_interval_coverage",
        "interval90_coverage_event_equal",
        "interval90_nominal_coverage",
        "interval90_absolute_calibration_error",
        "maximum_interval90_absolute_calibration_error",
        "minimum_interval90_value_coverage_fraction",
        "complete_native_interval_values",
        "interval_calibration_gate_pass",
        "interval_aggregation_unit",
    ]
    fold_rows: list[dict[str, object]] = []
    for (fold, condition), subset in event_audit.groupby(
        ["fold", "condition"], sort=True, dropna=False
    ):
        coverage = subset["interval90_coverage_event"]
        value_coverage = subset["interval90_value_coverage_fraction"]
        all_coverage_finite = bool(coverage.notna().all() and len(coverage))
        complete_values = bool(
            value_coverage.notna().all()
            and len(value_coverage)
            and np.allclose(
                value_coverage.to_numpy(dtype=float),
                1.0,
                rtol=0.0,
                atol=1e-12,
            )
        )
        event_equal_coverage = (
            float(coverage.mean()) if all_coverage_finite else np.nan
        )
        calibration_error = (
            abs(event_equal_coverage - nominal_coverage)
            if np.isfinite(event_equal_coverage)
            else np.nan
        )
        fold_rows.append(
            {
                "fold": fold,
                "condition": condition,
                "n_independent_events": int(subset["event_id"].nunique()),
                "n_events_with_interval_coverage": int(coverage.notna().sum()),
                "interval90_coverage_event_equal": event_equal_coverage,
                "interval90_nominal_coverage": nominal_coverage,
                "interval90_absolute_calibration_error": calibration_error,
                "maximum_interval90_absolute_calibration_error": maximum_error,
                "minimum_interval90_value_coverage_fraction": (
                    float(value_coverage.min())
                    if value_coverage.notna().any()
                    else np.nan
                ),
                "complete_native_interval_values": complete_values,
                "interval_calibration_gate_pass": bool(
                    all_coverage_finite
                    and complete_values
                    and _within_calibration_tolerance(
                        calibration_error, maximum_error
                    )
                ),
                "interval_aggregation_unit": F8_INTERVAL_AGGREGATION_UNIT,
            }
        )
    fold_audit = pd.DataFrame(fold_rows, columns=fold_columns)

    condition_columns = [
        "condition",
        "n_independent_events_interval_audit",
        "n_events_with_interval_coverage",
        "n_calibration_folds",
        "interval90_coverage_event_equal",
        "interval90_nominal_coverage",
        "interval90_absolute_calibration_error",
        "maximum_fold_interval90_absolute_calibration_error",
        "max_interval90_absolute_calibration_error",
        "minimum_interval90_value_coverage_fraction",
        "complete_native_interval_values",
        "all_fold_interval_calibration_gate_pass",
        "interval_calibration_gate_pass",
        "interval_aggregation_unit",
    ]
    condition_rows: list[dict[str, object]] = []
    for condition in condition_order:
        subset = event_audit.loc[event_audit["condition"].eq(condition)]
        folds = fold_audit.loc[fold_audit["condition"].astype(str).eq(condition)]
        coverage = subset["interval90_coverage_event"]
        value_coverage = subset["interval90_value_coverage_fraction"]
        all_coverage_finite = bool(coverage.notna().all() and len(coverage))
        complete_values = bool(
            value_coverage.notna().all()
            and len(value_coverage)
            and np.allclose(
                value_coverage.to_numpy(dtype=float),
                1.0,
                rtol=0.0,
                atol=1e-12,
            )
        )
        event_equal_coverage = (
            float(coverage.mean()) if all_coverage_finite else np.nan
        )
        calibration_error = (
            abs(event_equal_coverage - nominal_coverage)
            if np.isfinite(event_equal_coverage)
            else np.nan
        )
        fold_errors = pd.to_numeric(
            folds["interval90_absolute_calibration_error"], errors="coerce"
        )
        all_folds_pass = bool(
            not folds.empty
            and folds["interval_calibration_gate_pass"].astype(bool).all()
        )
        condition_rows.append(
            {
                "condition": condition,
                "n_independent_events_interval_audit": int(
                    subset["event_id"].nunique()
                ),
                "n_events_with_interval_coverage": int(coverage.notna().sum()),
                "n_calibration_folds": int(len(folds)),
                "interval90_coverage_event_equal": event_equal_coverage,
                "interval90_nominal_coverage": nominal_coverage,
                "interval90_absolute_calibration_error": calibration_error,
                "maximum_fold_interval90_absolute_calibration_error": (
                    float(fold_errors.max()) if fold_errors.notna().any() else np.nan
                ),
                "max_interval90_absolute_calibration_error": maximum_error,
                "minimum_interval90_value_coverage_fraction": (
                    float(value_coverage.min())
                    if value_coverage.notna().any()
                    else np.nan
                ),
                "complete_native_interval_values": complete_values,
                "all_fold_interval_calibration_gate_pass": all_folds_pass,
                "interval_calibration_gate_pass": bool(
                    all_coverage_finite
                    and complete_values
                    and all_folds_pass
                    and _within_calibration_tolerance(
                        calibration_error, maximum_error
                    )
                ),
                "interval_aggregation_unit": F8_INTERVAL_AGGREGATION_UNIT,
            }
        )
    condition_audit = pd.DataFrame(condition_rows, columns=condition_columns)
    return event_audit, fold_audit, condition_audit


def _f8_confirmatory_gate_pass(
    *,
    minimum_event_support_gate_pass: bool,
    persistence_gate_pass: bool,
    f6_comparison_gate_pass: bool,
    interval_calibration_gate_pass: bool,
) -> bool:
    """Combine every mandatory F8 condition-level confirmatory criterion."""

    return bool(
        minimum_event_support_gate_pass
        and persistence_gate_pass
        and f6_comparison_gate_pass
        and interval_calibration_gate_pass
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("official", "smoke"), default="official")
    parser.add_argument("--pacific-cube", type=Path, default=PACIFIC)
    parser.add_argument("--chirps-target", type=Path, default=CHIRPS)
    parser.add_argument("--master", type=Path)
    parser.add_argument("--phase-table", type=Path)
    parser.add_argument(
        "--target-variable",
        default="precip_weekly_mm",
        help="camada CHIRPS nativa (chuva, R95p/R99p, CDD/CWD ou anomalia)",
    )
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-transform", choices=("train_robust_z", "train_anomaly_mm"), default="train_robust_z")
    parser.add_argument("--distribution", choices=("gaussian", "quantile"), default="quantile")
    parser.add_argument("--quantiles", type=_quantiles, default=(0.05, 0.5, 0.95))
    parser.add_argument(
        "--max-interval90-absolute-calibration-error",
        type=float,
        default=F8_MAX_INTERVAL90_ABSOLUTE_CALIBRATION_ERROR,
        help=(
            "tolerancia confirmatoria predeclarada para |cobertura IC90 - 0.90| "
            "por condicao e por fold, com peso igual por evento"
        ),
    )
    parser.add_argument(
        "--minimum-f6-paired-coverage",
        type=float,
        default=0.80,
        help="fracao minima predeclarada de pares OOS ativos F8 cobertos pela F6",
    )
    parser.add_argument(
        "--minimum-independent-events-per-condition",
        type=int,
        default=3,
        help="eventos OOS independentes minimos em cada EN/LN x fase",
    )
    parser.add_argument("--research-override-gate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_interval90_contract(args)
    if args.horizon < 1:
        raise ValueError("F8 preditiva exige --horizon >= 1.")
    if not 0.0 < args.minimum_f6_paired_coverage <= 1.0:
        raise ValueError("--minimum-f6-paired-coverage deve pertencer a (0,1].")
    if args.minimum_independent_events_per_condition < 2:
        raise ValueError(
            "--minimum-independent-events-per-condition deve ser >= 2."
        )
    artifact_mode = (
        "smoke"
        if args.mode == "official" and args.research_override_gate
        else args.mode
    )
    if artifact_mode != args.mode:
        print(
            "[F8 exploratoria] override solicitado: o artefato sera isolado em runs/smoke."
        )
    master_path = args.master or _find([FEATURES / "nino34_master_weekly.csv"], "master F2")
    phase_path = args.phase_table or _find(
        [MODEL_BRIDGE / "fases_semanais_en_ln.csv"],
        "fases semanais canonicas F3",
    )
    if not args.pacific_cube.exists():
        raise FileNotFoundError(
            f"{args.pacific_cube} ausente; rode scripts/build_phase7_pacific_cube.py."
        )
    if not args.chirps_target.exists():
        raise FileNotFoundError(
            f"{args.chirps_target} ausente; rode scripts/build_phase4_chirps_targets.py "
            "e valide/promova o alvo CHIRPS nativo antes da F8."
        )
    pacific_ds = xr.open_zarr(args.pacific_cube, consolidated=False)
    chirps_ds = xr.open_zarr(args.chirps_target, consolidated=None)
    target_validation = validate_native_target(chirps_ds)
    if not target_validation.valid:
        raise ValueError(f"Alvo CHIRPS invalido: {target_validation.errors}")
    if "brazil_fraction" not in chirps_ds or "brazil_center" not in chirps_ds:
        raise ValueError(
            "F8 exige mascaras brazil_fraction e brazil_center auditadas; "
            "o bounding box inteiro nunca pode substituir o Brasil."
        )
    if args.target_variable not in chirps_ds:
        raise KeyError(f"Alvo CHIRPS sem variavel {args.target_variable!r}.")
    target_units = str(
        chirps_ds[args.target_variable].attrs.get("units", "native target units")
    )
    raw_response, pixel_table = target_to_frame(
        chirps_ds, variable=args.target_variable, brazil_only=False
    )
    master = _csv_time(master_path)
    phase = _csv_time(phase_path, phase=True)
    predictors = physical_predictor_columns(master)
    spatial_names = tuple(
        name
        for name in pacific_ds.data_vars
        if name not in {"valid_day_count", "expected_day_count", "complete_week"}
    )
    common = (
        pd.DatetimeIndex(pacific_ds.time.values)
        .intersection(raw_response.index)
        .intersection(master.index)
        .intersection(phase.index)
        .sort_values()
    )
    spatial = pacific_ds[list(spatial_names)].sel(time=common).to_array("channel").transpose(
        "time", "lat", "lon", "channel"
    )
    spatial_values = np.asarray(spatial.values, dtype="float32")
    cube = PacificCube(
        values=spatial_values,
        times=common,
        lat=np.asarray(pacific_ds.lat.values),
        lon=np.asarray(pacific_ds.lon.values),
        channel_names=spatial_names,
        finite_mask=np.isfinite(spatial_values),
        source_path=str(args.pacific_cube),
    )
    phase_common = phase.reindex(common)
    if not bool(valid_phase_target_mask(phase_common).all()):
        raise ValueError("Tabela de fases F8 possui tipo/fase ausente no periodo comum.")
    sequence_data = make_sequence_dataset(
        cube,
        labels=None,
        seq_len=args.seq_len,
        horizon=args.horizon,
        event_ids=phase_common["event_id"].fillna("").to_numpy(),
    )
    complete_week = pacific_ds["complete_week"].sel(time=common).to_numpy().astype(bool)
    spatial_sequence_available = np.asarray(
        [
            complete_week[end - args.seq_len + 1 : end + 1].all()
            for end in sequence_data.end_indices
        ],
        dtype=bool,
    )
    # Teleconnection strata belong to the Pacific source at sequence origin,
    # while the response remains CHIRPS at target_time = origin+horizon.
    source_phase = sequence_source_phase_table(sequence_data, phase)
    folds = sequence_event_folds(
        sequence_data,
        source_phase,
        n_splits=2 if args.mode == "smoke" else 5,
        min_train_groups=6 if args.mode == "smoke" else 8,
        seq_len=args.seq_len,
        horizon=args.horizon,
    )
    if args.mode == "smoke":
        folds = folds[:1]
    target_lat = np.asarray(chirps_ds.latitude.values)
    target_lon = np.asarray(chirps_ds.longitude.values)
    source_pixel_ids = (
        np.asarray(chirps_ds["pixel_id"].values).reshape(-1)
        if "pixel_id" in chirps_ds
        else pixel_table["pixel_id"].to_numpy()
    )
    grid = native_chirps_grid(
        target_lat,
        target_lon,
        pixel_ids=source_pixel_ids,
        expected_grid_sha256=str(chirps_ds.attrs.get("grid_hash_sha256", "")),
    )
    f6_reference_contract = _find_f6_reference_run(
        horizon_weeks=args.horizon,
        grid_sha256=grid.grid_sha256,
        target_variable=args.target_variable,
    )
    if (
        args.mode == "official"
        and not args.research_override_gate
        and f6_reference_contract is None
    ):
        print(
            "[F8 bloqueada] falta merge F6 completo no mesmo grid/horizonte, "
            "com cobertura integral nas oito condicoes EN/LN x fase."
        )
        return 3
    f6_reference = None
    f6_reference_run = None
    f6_reference_model = None
    f6_reference_transform = None
    if f6_reference_contract is not None:
        (
            f6_reference_run,
            f6_reference_model,
            f6_reference_transform,
            f6_reference_units,
        ) = f6_reference_contract
        if str(f6_reference_units) != target_units:
            raise ValueError(
                "Unidades F6/F8 incompativeis: "
                f"{f6_reference_units!r} != {target_units!r}."
            )
        f6_reference = pd.read_csv(
            f6_reference_run / "tables" / "pixel_oos_predictions.csv",
            parse_dates=["time"],
        )
        required_prediction = {
            "fold",
            "time",
            "pixel_id",
            "condition",
            "lag_weeks",
            "model",
            "target_variable",
            "target_transform",
            "target_units",
            "observed_native_value",
            "predicted_native_value",
        }
        missing_prediction = required_prediction.difference(f6_reference.columns)
        if missing_prediction:
            raise ValueError(
                f"Predicoes F6 sem contrato nativo: {sorted(missing_prediction)}"
            )
        f6_reference = f6_reference.loc[
            pd.to_numeric(f6_reference["lag_weeks"], errors="coerce").eq(args.horizon)
            & f6_reference["condition"].astype(str).isin(CANONICAL_ACTIVE_CONDITIONS)
            & f6_reference["model"].astype(str).eq(f6_reference_model)
            & f6_reference["target_transform"].astype(str).eq(
                f6_reference_transform
            )
            & f6_reference["target_units"].astype(str).eq(target_units)
        ]
        if "target_variable" in f6_reference:
            f6_reference = f6_reference.loc[
                f6_reference["target_variable"].astype(str).eq(args.target_variable)
            ]
        if f6_reference.empty and args.mode == "official":
            raise ValueError("Merge F6 nao possui predicoes OOS no horizonte F8.")
        duplicate_key = [
            "model",
            "fold",
            "time",
            "pixel_id",
            "condition",
            "lag_weeks",
            "target_variable",
            "target_transform",
        ]
        if f6_reference.duplicated(duplicate_key).any():
            raise ValueError("Predicoes F6 duplicadas no protocolo OOS pareado.")
        pair_key = [
            "model",
            "time",
            "pixel_id",
            "condition",
            "lag_weeks",
            "target_variable",
            "target_transform",
        ]
        if f6_reference.duplicated(pair_key).any():
            raise ValueError(
                "Predicoes F6 repetem o mesmo pixel-tempo em folds distintos; "
                "pareamento F8 ambiguo recusado."
            )
    brazil_fraction = np.asarray(
        chirps_ds["brazil_fraction"].values, dtype="float32"
    )
    brazil_mask = brazil_fraction > 0
    area_weight = brazil_cell_area_weights(target_lat, target_lon).astype("float32") * brazil_fraction
    import torch

    device = args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    epochs = 1 if args.mode == "smoke" else args.epochs
    hidden = min(args.hidden, 8) if args.mode == "smoke" else args.hidden
    parameters = {
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "epochs": epochs,
        "hidden": hidden,
        "target_transform": args.target_transform,
        "target_variable": args.target_variable,
        "distribution": args.distribution,
        "quantiles": args.quantiles,
        "interval90_nominal_coverage": EVENT_INTERVAL_NOMINAL_COVERAGE,
        "max_interval90_absolute_calibration_error": (
            args.max_interval90_absolute_calibration_error
        ),
        "interval_aggregation_unit": F8_INTERVAL_AGGREGATION_UNIT,
        "target_units": target_units,
        "native_support": _native_target_bounds(args.target_variable)[2],
        "minimum_f6_paired_coverage": args.minimum_f6_paired_coverage,
        "minimum_independent_events_per_condition": (
            args.minimum_independent_events_per_condition
        ),
        "f6_reference_model": f6_reference_model,
        "f6_reference_transform": f6_reference_transform,
        "research_override_gate": bool(args.research_override_gate),
        "target_grid_sha256": grid.grid_sha256,
        "target_interpolation": False,
        "n_target_pixels": len(grid.pixel_ids),
        "spatial_channels": spatial_names,
        "scalar_channels": predictors,
    }
    run = start_artifact_run(
        8,
        mode=artifact_mode,
        inputs=[
            args.pacific_cube,
            args.chirps_target,
            master_path,
            phase_path,
            Path(__file__).resolve(),
            *(
                [f6_reference_run / "tables" / "pixel_oos_predictions.csv"]
                if f6_reference_run is not None
                else []
            ),
        ],
        seed=args.seed,
        parameters=parameters,
        command=" ".join([sys.executable, *sys.argv]),
    )
    fold_metrics: list[dict[str, object]] = []
    pixel_metrics: list[pd.DataFrame] = []
    importance_tables: list[pd.DataFrame] = []
    contracts: list[dict[str, object]] = []
    gate_components: list[dict[str, object]] = []
    probabilistic_rows: list[dict[str, object]] = []
    try:
        for fold_no, fold in enumerate(folds, start=1):
            raw_train_mask = cube.times <= fold.train_end - pd.Timedelta(weeks=args.horizon)
            spatial_normalizer = fit_spatial_harmonic_normalizer(
                cube.values, cube.times, raw_train_mask
            )
            normalized = spatial_normalizer.transform(cube.values, cube.times)
            spatial_sequences = np.stack(
                [normalized[end - args.seq_len + 1 : end + 1] for end in sequence_data.end_indices]
            ).astype("float32")
            scalar_fit_end = fold.train_end - pd.Timedelta(weeks=args.horizon)
            scalar_history = master.loc[:scalar_fit_end]
            scalar_transformer = FoldHarmonicPreprocessor(tuple(predictors)).fit(
                scalar_history[predictors],
                source_codes=scalar_history.get("ocean_source_code"),
            )
            scalar_aligned = master.reindex(cube.times)
            scalar_values = scalar_transformer.transform(
                scalar_aligned[predictors],
                source_codes=scalar_aligned.get("ocean_source_code"),
            )
            scalar_mean = scalar_values.loc[:scalar_fit_end].mean()
            scalar_std = scalar_values.loc[:scalar_fit_end].std().replace(0, 1)
            scalar_values = (scalar_values - scalar_mean) / scalar_std
            scalar_sequences = _scalar_sequences(
                scalar_values.to_numpy(dtype="float32"), sequence_data.end_indices, args.seq_len
            )
            target_transformer = fit_fold_target_transformer(
                raw_response.reindex(common),
                fit_end=fold.train_end,
                method=args.target_transform,
                target_name=args.target_variable,
            )
            transformed_response = target_transformer.transform(
                raw_response.reindex(common)
            )
            target_frame = transformed_response.reindex(sequence_data.target_times)
            target_fields = target_frame.to_numpy(dtype="float32").reshape(
                len(target_frame), len(target_lat), len(target_lon)
            )
            # Forecast origin is target_time-horizon.  Shift raw native values
            # first, then encode them with the target week's train-only
            # climatology.  This is causal and inverses exactly to y(origin).
            persistence_transformed = causal_persistence_target(
                raw_response.reindex(common),
                target_transformer,
                horizon_weeks=args.horizon,
            )
            persistence_fields = persistence_transformed.reindex(
                sequence_data.target_times
            ).to_numpy(dtype="float32").reshape(target_fields.shape)
            train = fold.train_index
            test = fold.test_index
            scalar_available = np.isfinite(scalar_sequences).all(axis=(1, 2))
            target_available = (
                np.isfinite(target_fields) & brazil_mask[None]
            ).any(axis=(1, 2))
            model_input_available = (
                scalar_available & spatial_sequence_available & target_available
            )
            train = train[model_input_available[train]]
            test = test[model_input_available[test]]
            if args.mode == "smoke":
                train = train[-min(32, len(train)) :]
                test = test[: min(8, len(test))]
            if not len(train) or not len(test):
                continue
            model = build_convlstm_encoder_decoder(
                (args.seq_len, cube.values.shape[1], cube.values.shape[2], len(spatial_names)),
                (len(target_lat), len(target_lon)),
                filters=(hidden,),
                distribution=args.distribution,
                scalar_channels=len(predictors),
                quantiles=args.quantiles,
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
            train_weights = event_phase_sample_weights(
                source_phase.iloc[train]
            ).to_numpy(dtype="float32")
            sample_weights = np.zeros(len(sequence_data.target_times), dtype="float32")
            sample_weights[train] = train_weights
            order = np.asarray(train).copy()
            for epoch in range(epochs):
                np.random.default_rng(args.seed + fold_no + epoch).shuffle(order)
                model.train()
                for start in range(0, len(order), args.batch_size):
                    batch = order[start : start + args.batch_size]
                    spatial_batch = torch.as_tensor(spatial_sequences[batch], dtype=torch.float32, device=device)
                    scalar_batch = torch.as_tensor(scalar_sequences[batch], dtype=torch.float32, device=device)
                    target_batch = torch.as_tensor(target_fields[batch], dtype=torch.float32, device=device)
                    output = model(spatial_batch, scalar_sequence=scalar_batch)
                    if args.distribution == "gaussian":
                        loss = masked_area_weighted_gaussian_nll(
                            output["mean"],
                            output["log_scale"],
                            torch.nan_to_num(target_batch),
                            brazil_mask=torch.as_tensor(brazil_mask, device=device),
                            area_weight=torch.as_tensor(area_weight, device=device),
                            valid_mask=torch.isfinite(target_batch),
                            sample_weight=torch.as_tensor(sample_weights[batch], device=device),
                        )
                    else:
                        loss = masked_area_weighted_quantile_loss(
                            output["quantiles"],
                            torch.nan_to_num(target_batch),
                            quantiles=args.quantiles,
                            brazil_mask=torch.as_tensor(brazil_mask, device=device),
                            area_weight=torch.as_tensor(area_weight, device=device),
                            valid_mask=torch.isfinite(target_batch),
                            sample_weight=torch.as_tensor(sample_weights[batch], device=device),
                        )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            model.eval()
            predicted_batches, scale_batches, lower_batches, upper_batches = [], [], [], []
            quantile_batches: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(test), args.batch_size):
                    batch = test[start : start + args.batch_size]
                    output = model(
                        torch.as_tensor(spatial_sequences[batch], dtype=torch.float32, device=device),
                        scalar_sequence=torch.as_tensor(scalar_sequences[batch], dtype=torch.float32, device=device),
                    )
                    if args.distribution == "gaussian":
                        mean_batch = output["mean"].cpu().numpy()
                        scale_batch = output["log_scale"].exp().cpu().numpy()
                        predicted_batches.append(mean_batch)
                        scale_batches.append(scale_batch)
                        lower_batches.append(
                            mean_batch - EVENT_INTERVAL_Z * scale_batch
                        )
                        upper_batches.append(
                            mean_batch + EVENT_INTERVAL_Z * scale_batch
                        )
                    else:
                        quantile_batch = output["quantiles"].cpu().numpy()
                        quantile_batches.append(quantile_batch)
                        median_no = args.quantiles.index(0.5)
                        predicted_batches.append(quantile_batch[:, median_no])
                        lower_batches.append(quantile_batch[:, 0])
                        upper_batches.append(quantile_batch[:, -1])
                        scale_batches.append(
                            (quantile_batch[:, -1] - quantile_batch[:, 0]) / 2.0
                        )
            predicted = np.concatenate(predicted_batches)
            scale = np.concatenate(scale_batches)
            lower = np.concatenate(lower_batches)
            upper = np.concatenate(upper_batches)
            quantile_fields = (
                np.concatenate(quantile_batches)
                if quantile_batches
                else None
            )
            observed = target_fields[test]
            baseline = persistence_fields[test]
            test_times = sequence_data.target_times[test]
            predicted_native = target_transformer.inverse_array(
                predicted.reshape(len(test), -1),
                index=test_times,
                columns=raw_response.columns,
            ).reshape(predicted.shape)
            baseline_native = target_transformer.inverse_array(
                baseline.reshape(len(test), -1),
                index=test_times,
                columns=raw_response.columns,
            ).reshape(baseline.shape)
            lower_native = target_transformer.inverse_array(
                lower.reshape(len(test), -1),
                index=test_times,
                columns=raw_response.columns,
            ).reshape(lower.shape)
            upper_native = target_transformer.inverse_array(
                upper.reshape(len(test), -1),
                index=test_times,
                columns=raw_response.columns,
            ).reshape(upper.shape)
            quantile_native = None
            if quantile_fields is not None:
                quantile_native = np.stack(
                    [
                        target_transformer.inverse_array(
                            quantile_fields[:, quantile_no].reshape(len(test), -1),
                            index=test_times,
                            columns=raw_response.columns,
                        ).reshape(predicted.shape)
                        for quantile_no in range(len(args.quantiles))
                    ],
                    axis=1,
                )
            support_lower, support_upper, support_rule = _native_target_bounds(
                args.target_variable
            )
            predicted_native = _clip_native_support(
                predicted_native, support_lower, support_upper
            )
            baseline_native = _clip_native_support(
                baseline_native, support_lower, support_upper
            )
            lower_native = _clip_native_support(
                lower_native, support_lower, support_upper
            )
            upper_native = _clip_native_support(
                upper_native, support_lower, support_upper
            )
            if quantile_native is not None:
                quantile_native = _clip_native_support(
                    quantile_native, support_lower, support_upper
                )
            observed_native = raw_response.reindex(test_times).to_numpy(dtype="float32").reshape(
                observed.shape
            )
            valid = (
                np.isfinite(observed_native)
                & np.isfinite(predicted_native)
                & np.isfinite(baseline_native)
                & brazil_mask[None]
            )
            weighted = area_weight[None] * valid
            rmse = float(
                np.sqrt(
                    np.nansum(weighted * (predicted_native - observed_native) ** 2)
                    / np.nansum(weighted)
                )
            )
            rmse_base = float(
                np.sqrt(
                    np.nansum(weighted * (baseline_native - observed_native) ** 2)
                    / np.nansum(weighted)
                )
            )
            rmse_f6 = skill_vs_f6 = np.nan
            n_f6_pairs = 0
            f6_pair_coverage = np.nan
            if f6_reference is not None and not f6_reference.empty:
                time_position = {
                    pd.Timestamp(value): position
                    for position, value in enumerate(sequence_data.target_times[test])
                }
                pixel_position = {
                    str(value): position for position, value in enumerate(source_pixel_ids)
                }
                source_states_array = canonical_state_labels(
                    source_phase.iloc[test]
                ).to_numpy(dtype=object)
                f6_field = np.full((len(test), len(source_pixel_ids)), np.nan, dtype="float32")
                f6_observed_field = np.full_like(f6_field, np.nan)
                reference_subset = f6_reference.loc[
                    f6_reference["time"].isin(time_position)
                    & f6_reference["pixel_id"].astype(str).isin(pixel_position)
                ]
                for row in reference_subset.itertuples():
                    time_no = time_position.get(pd.Timestamp(row.time))
                    pixel_no = pixel_position.get(str(row.pixel_id))
                    if time_no is None or pixel_no is None:
                        continue
                    if str(row.condition) != str(source_states_array[time_no]):
                        continue
                    f6_field[time_no, pixel_no] = float(row.predicted_native_value)
                    f6_observed_field[time_no, pixel_no] = float(
                        row.observed_native_value
                    )
                f6_field = f6_field.reshape(len(test), len(target_lat), len(target_lon))
                f6_observed_field = f6_observed_field.reshape(
                    len(test), len(target_lat), len(target_lon)
                )
                observation_pairs = (
                    np.isfinite(f6_observed_field)
                    & np.isfinite(observed_native)
                    & brazil_mask[None]
                )
                if observation_pairs.any() and not np.allclose(
                    f6_observed_field[observation_pairs],
                    observed_native[observation_pairs],
                    rtol=1e-6,
                    atol=1e-5,
                ):
                    maximum_difference = float(
                        np.max(
                            np.abs(
                                f6_observed_field[observation_pairs]
                                - observed_native[observation_pairs]
                            )
                        )
                    )
                    raise ValueError(
                        "Observacoes nativas F6 divergem do cubo CHIRPS F8 "
                        f"(max_abs_diff={maximum_difference:.6g} {target_units})."
                    )
                active_sequence = np.isin(
                    source_states_array, np.asarray(CANONICAL_ACTIVE_CONDITIONS)
                )
                expected_pair_mask = (
                    np.isfinite(observed_native)
                    & np.isfinite(predicted_native)
                    & brazil_mask[None]
                    & active_sequence[:, None, None]
                )
                paired_valid = (
                    np.isfinite(observed_native)
                    & np.isfinite(predicted_native)
                    & np.isfinite(f6_field)
                    & np.isfinite(f6_observed_field)
                    & brazil_mask[None]
                )
                paired_weight = area_weight[None] * paired_valid
                n_f6_pairs = int(paired_valid.sum())
                expected_f6_pairs = int(expected_pair_mask.sum())
                f6_pair_coverage = (
                    n_f6_pairs / expected_f6_pairs if expected_f6_pairs else np.nan
                )
                if np.nansum(paired_weight) > 0:
                    rmse_f8_paired = float(
                        np.sqrt(
                            np.nansum(
                                paired_weight * (predicted_native - observed_native) ** 2
                            )
                            / np.nansum(paired_weight)
                        )
                    )
                    rmse_f6 = float(
                        np.sqrt(
                            np.nansum(
                                paired_weight * (f6_field - observed_native) ** 2
                            )
                            / np.nansum(paired_weight)
                        )
                    )
                    skill_vs_f6 = (
                        1.0 - rmse_f8_paired / rmse_f6 if rmse_f6 > 0 else np.nan
                    )
            fold_metrics.append(
                {
                    "fold": fold.name,
                    "n_train_sequences": len(train),
                    "n_test_sequences": len(test),
                    "rmse_area_weighted": rmse,
                    "rmse_persistence": rmse_base,
                    "target_units": target_units,
                    "native_support_postprocessing": support_rule,
                    "skill_rmse_vs_persistence": 1.0 - rmse / rmse_base if rmse_base > 0 else np.nan,
                    "rmse_f6_paired": rmse_f6,
                    "paired_comparison_units": target_units,
                    "skill_rmse_vs_f6_paired": skill_vs_f6,
                    "n_pixel_week_pairs_f6": n_f6_pairs,
                    "paired_coverage_fraction_f6": f6_pair_coverage,
                    "minimum_paired_coverage_f6": args.minimum_f6_paired_coverage,
                    "gate_pass": bool(
                        rmse < rmse_base
                        and (
                            f6_reference is None
                            or (
                                np.isfinite(skill_vs_f6)
                                and skill_vs_f6 > 0
                                and np.isfinite(f6_pair_coverage)
                                and f6_pair_coverage
                                >= args.minimum_f6_paired_coverage
                            )
                        )
                    ),
                    "metric_scope": "all_source_conditions",
                    "source_phase_reference": "origin_time",
                }
            )
            test_source = source_phase.iloc[test]
            test_states = canonical_state_labels(test_source)
            for state in sorted(test_states.unique()):
                state_mask = test_states.eq(state).to_numpy()
                state_valid = valid[state_mask]
                state_weighted = area_weight[None] * state_valid
                denominator = np.nansum(state_weighted)
                if denominator <= 0:
                    continue
                state_rmse = float(
                    np.sqrt(
                        np.nansum(
                            state_weighted
                            * (
                                predicted_native[state_mask]
                                - observed_native[state_mask]
                            )
                            ** 2
                        )
                        / denominator
                    )
                )
                state_base = float(
                    np.sqrt(
                        np.nansum(
                            state_weighted
                            * (
                                baseline_native[state_mask]
                                - observed_native[state_mask]
                            )
                            ** 2
                        )
                        / denominator
                    )
                )
                state_pair_weight = 0.0
                state_pair_count = 0
                state_expected_pairs = 0
                state_sq_model_f6 = np.nan
                state_sq_f6 = np.nan
                state_rmse_f6 = np.nan
                state_skill_f6 = np.nan
                state_pair_coverage = np.nan
                if f6_reference is not None and not f6_reference.empty:
                    state_paired_valid = paired_valid[state_mask]
                    state_paired_weight = area_weight[None] * state_paired_valid
                    state_pair_weight = float(np.nansum(state_paired_weight))
                    state_pair_count = int(state_paired_valid.sum())
                    state_expected_pairs = int(expected_pair_mask[state_mask].sum())
                    state_pair_coverage = (
                        state_pair_count / state_expected_pairs
                        if state_expected_pairs
                        else np.nan
                    )
                    if state_pair_weight > 0:
                        state_sq_model_f6 = float(
                            np.nansum(
                                state_paired_weight
                                * (
                                    predicted_native[state_mask]
                                    - observed_native[state_mask]
                                )
                                ** 2
                            )
                        )
                        state_sq_f6 = float(
                            np.nansum(
                                state_paired_weight
                                * (
                                    f6_field[state_mask]
                                    - observed_native[state_mask]
                                )
                                ** 2
                            )
                        )
                        state_rmse_model_paired = float(
                            np.sqrt(state_sq_model_f6 / state_pair_weight)
                        )
                        state_rmse_f6 = float(
                            np.sqrt(state_sq_f6 / state_pair_weight)
                        )
                        state_skill_f6 = (
                            1.0 - state_rmse_model_paired / state_rmse_f6
                            if state_rmse_f6 > 0
                            else np.nan
                        )
                state_gate = bool(
                    state in CANONICAL_ACTIVE_CONDITIONS
                    and state_rmse < state_base
                    and (
                        f6_reference is None
                        or (
                            np.isfinite(state_skill_f6)
                            and state_skill_f6 > 0
                            and np.isfinite(state_pair_coverage)
                            and state_pair_coverage
                            >= args.minimum_f6_paired_coverage
                        )
                    )
                )
                state_event_series = test_source.loc[test_states.index[state_mask], "event_id"]
                if state in CANONICAL_ACTIVE_CONDITIONS and state_event_series.isna().any():
                    raise ValueError(f"{state}: evento ativo sem event_id no gate F8.")
                state_event_ids = state_event_series.dropna().astype(str).unique()
                for event_id in state_event_ids:
                    event_mask = state_mask & test_source["event_id"].astype(str).eq(
                        event_id
                    ).to_numpy()
                    event_valid = valid[event_mask]
                    event_weight = area_weight[None] * event_valid
                    event_denominator = float(np.nansum(event_weight))
                    if event_denominator <= 0:
                        continue
                    event_mse_model = float(
                        np.nansum(
                            event_weight
                            * (
                                predicted_native[event_mask]
                                - observed_native[event_mask]
                            )
                            ** 2
                        )
                        / event_denominator
                    )
                    event_mse_persistence = float(
                        np.nansum(
                            event_weight
                            * (
                                baseline_native[event_mask]
                                - observed_native[event_mask]
                            )
                            ** 2
                        )
                        / event_denominator
                    )
                    event_interval_valid = (
                        event_valid
                        & np.isfinite(lower_native[event_mask])
                        & np.isfinite(upper_native[event_mask])
                    )
                    event_interval_weight = (
                        area_weight[None] * event_interval_valid
                    )
                    event_interval_denominator = float(
                        np.nansum(event_interval_weight)
                    )
                    event_interval_value_coverage = (
                        event_interval_denominator / event_denominator
                        if event_denominator > 0
                        else np.nan
                    )
                    event_interval_coverage = (
                        float(
                            np.nansum(
                                event_interval_weight
                                * (
                                    (
                                        observed_native[event_mask]
                                        >= lower_native[event_mask]
                                    )
                                    & (
                                        observed_native[event_mask]
                                        <= upper_native[event_mask]
                                    )
                                )
                            )
                            / event_interval_denominator
                        )
                        if event_interval_denominator > 0
                        else np.nan
                    )
                    event_pair_count = 0
                    event_expected_count = 0
                    event_pair_coverage = np.nan
                    event_mse_model_f6 = np.nan
                    event_mse_f6 = np.nan
                    if f6_reference is not None and not f6_reference.empty:
                        event_paired_valid = paired_valid[event_mask]
                        event_paired_weight = area_weight[None] * event_paired_valid
                        event_pair_denominator = float(
                            np.nansum(event_paired_weight)
                        )
                        event_pair_count = int(event_paired_valid.sum())
                        event_expected_count = int(
                            expected_pair_mask[event_mask].sum()
                        )
                        event_pair_coverage = (
                            event_pair_count / event_expected_count
                            if event_expected_count
                            else np.nan
                        )
                        if event_pair_denominator > 0:
                            event_mse_model_f6 = float(
                                np.nansum(
                                    event_paired_weight
                                    * (
                                        predicted_native[event_mask]
                                        - observed_native[event_mask]
                                    )
                                    ** 2
                                )
                                / event_pair_denominator
                            )
                            event_mse_f6 = float(
                                np.nansum(
                                    event_paired_weight
                                    * (
                                        f6_field[event_mask]
                                        - observed_native[event_mask]
                                    )
                                    ** 2
                                )
                                / event_pair_denominator
                            )
                    gate_components.append(
                        {
                            "fold": fold.name,
                            "condition": state,
                            "event_id": event_id,
                            "event_mse_model": event_mse_model,
                            "event_mse_persistence": event_mse_persistence,
                            "event_mse_model_paired_f6": event_mse_model_f6,
                            "event_mse_f6": event_mse_f6,
                            "paired_coverage_fraction_f6": event_pair_coverage,
                            "paired_count": event_pair_count,
                            "expected_pair_count": event_expected_count,
                            "n_sequences": int(event_mask.sum()),
                            "n_native_pixel_weeks": int(event_valid.sum()),
                            "n_interval_pixel_weeks": int(
                                event_interval_valid.sum()
                            ),
                            "interval90_eligible_area_weight": event_denominator,
                            "interval90_available_area_weight": (
                                event_interval_denominator
                            ),
                            "interval90_value_coverage_fraction": (
                                event_interval_value_coverage
                            ),
                            "interval90_coverage_event": event_interval_coverage,
                        }
                    )
                fold_metrics.append(
                    {
                        "fold": fold.name,
                        "n_train_sequences": len(train),
                        "n_test_sequences": int(state_mask.sum()),
                        "n_test_events": int(len(state_event_ids)),
                        "rmse_area_weighted": state_rmse,
                        "rmse_persistence": state_base,
                        "target_units": target_units,
                        "rmse_f6_paired": state_rmse_f6,
                        "skill_rmse_vs_f6_paired": state_skill_f6,
                        "paired_coverage_fraction_f6": state_pair_coverage,
                        "skill_rmse_vs_persistence": (
                            1.0 - state_rmse / state_base if state_base > 0 else np.nan
                        ),
                        "gate_pass": state_gate,
                        "gate_role": "fold_diagnostic_only",
                        "metric_scope": state,
                        "source_phase_reference": "origin_time",
                    }
                )
            # Proper probabilistic diagnostics on native target units.  Rows are
            # emitted globally and separately for each active EN/LN phase.
            probability_scopes: list[tuple[str, np.ndarray]] = [
                ("all_source_conditions", np.ones(len(test), dtype=bool))
            ]
            probability_scopes.extend(
                (
                    state,
                    test_states.eq(state).to_numpy(),
                )
                for state in CANONICAL_ACTIVE_CONDITIONS
                if test_states.eq(state).any()
            )
            interval_alpha = 1.0 - EVENT_INTERVAL_NOMINAL_COVERAGE
            for scope, sequence_mask in probability_scopes:
                scope_valid = valid & sequence_mask[:, None, None]
                scope_weight = area_weight[None] * scope_valid
                denominator_probability = float(np.nansum(scope_weight))
                if denominator_probability <= 0:
                    continue
                interval_valid = (
                    scope_valid
                    & np.isfinite(lower_native)
                    & np.isfinite(upper_native)
                )
                interval_weight = area_weight[None] * interval_valid
                interval_denominator = float(np.nansum(interval_weight))
                coverage = (
                    float(
                        np.nansum(
                            interval_weight
                            * (
                                (observed_native >= lower_native)
                                & (observed_native <= upper_native)
                            )
                        )
                        / interval_denominator
                    )
                    if interval_denominator > 0
                    else np.nan
                )
                sharpness = (
                    float(
                        np.nansum(interval_weight * (upper_native - lower_native))
                        / interval_denominator
                    )
                    if interval_denominator > 0
                    else np.nan
                )
                interval_score = (
                    (upper_native - lower_native)
                    + (2.0 / interval_alpha)
                    * (lower_native - observed_native)
                    * (observed_native < lower_native)
                    + (2.0 / interval_alpha)
                    * (observed_native - upper_native)
                    * (observed_native > upper_native)
                )
                interval_score_mean = (
                    float(
                        np.nansum(interval_weight * interval_score)
                        / interval_denominator
                    )
                    if interval_denominator > 0
                    else np.nan
                )
                absolute_error = np.abs(observed_native - predicted_native)
                wis_values = (
                    0.5 * absolute_error
                    + (interval_alpha / 2.0) * interval_score
                ) / 1.5
                wis = (
                    float(
                        np.nansum(interval_weight * wis_values)
                        / interval_denominator
                    )
                    if interval_denominator > 0
                    else np.nan
                )
                probabilistic_rows.append(
                    {
                        "fold": fold.name,
                        "source_condition": scope,
                        "metric": "weighted_interval_score",
                        "quantile": np.nan,
                        "value": wis,
                        "interval_coverage": coverage,
                        "nominal_coverage": 1.0 - interval_alpha,
                        "coverage_error": coverage - (1.0 - interval_alpha),
                        "mean_interval_width": sharpness,
                        "mean_interval_score": interval_score_mean,
                        "target_units": target_units,
                        "n_sequences": int(sequence_mask.sum()),
                    }
                )
                if args.distribution == "gaussian":
                    gaussian_valid = (
                        np.isfinite(observed)
                        & np.isfinite(predicted)
                        & np.isfinite(scale)
                        & (scale > 0)
                        & brazil_mask[None]
                        & sequence_mask[:, None, None]
                    )
                    gaussian_weight = area_weight[None] * gaussian_valid
                    gaussian_denominator = float(np.nansum(gaussian_weight))
                    gaussian_nll = (
                        0.5 * np.log(2.0 * np.pi)
                        + np.log(scale)
                        + 0.5 * ((observed - predicted) / scale) ** 2
                    )
                    probabilistic_rows.append(
                        {
                            "fold": fold.name,
                            "source_condition": scope,
                            "metric": "gaussian_nll_transformed",
                            "quantile": np.nan,
                            "value": (
                                float(
                                    np.nansum(gaussian_weight * gaussian_nll)
                                    / gaussian_denominator
                                )
                                if gaussian_denominator > 0
                                else np.nan
                            ),
                            "interval_coverage": np.nan,
                            "nominal_coverage": np.nan,
                            "coverage_error": np.nan,
                            "mean_interval_width": np.nan,
                            "mean_interval_score": np.nan,
                            "target_units": "transformed target scale",
                            "n_sequences": int(sequence_mask.sum()),
                        }
                    )
                if quantile_native is not None:
                    for quantile_no, quantile in enumerate(args.quantiles):
                        forecast = quantile_native[:, quantile_no]
                        error = observed_native - forecast
                        pinball = np.maximum(
                            quantile * error, (quantile - 1.0) * error
                        )
                        probabilistic_rows.append(
                            {
                                "fold": fold.name,
                                "source_condition": scope,
                                "metric": "pinball_loss",
                                "quantile": quantile,
                                "value": float(
                                    np.nansum(scope_weight * pinball)
                                    / denominator_probability
                                ),
                                "interval_coverage": np.nan,
                                "nominal_coverage": np.nan,
                                "coverage_error": np.nan,
                                "mean_interval_width": np.nan,
                                "mean_interval_score": np.nan,
                                "target_units": target_units,
                                "n_sequences": int(sequence_mask.sum()),
                            }
                        )
            pixel_table_metrics = pixel_skill_table(
                observed_native,
                predicted_native,
                baseline_native,
                grid,
                brazil_mask=brazil_mask,
                lower=lower_native,
                upper=upper_native,
            )
            pixel_table_metrics.insert(0, "fold", fold.name)
            pixel_metrics.append(pixel_table_metrics)
            xai_count = min(len(test), 2 if args.mode == "smoke" else 16)
            importance = field_channel_occlusion_importance(
                model,
                spatial_sequences[test[:xai_count]],
                scalar_sequences[test[:xai_count]],
                spatial_names=spatial_names,
                scalar_names=predictors,
                brazil_mask=brazil_mask,
                observed=observed[:xai_count],
                area_weight=area_weight,
                device=device,
            )
            importance["source_condition"] = "all"
            importance.insert(0, "fold", fold.name)
            importance_tables.append(importance)
            for state in CANONICAL_ACTIVE_CONDITIONS:
                state_positions = np.flatnonzero(test_states.eq(state).to_numpy())
                if len(state_positions) < 2:
                    continue
                state_positions = state_positions[
                    : min(len(state_positions), 2 if args.mode == "smoke" else 8)
                ]
                state_importance = field_channel_occlusion_importance(
                    model,
                    spatial_sequences[test[state_positions]],
                    scalar_sequences[test[state_positions]],
                    spatial_names=spatial_names,
                    scalar_names=predictors,
                    brazil_mask=brazil_mask,
                    observed=observed[state_positions],
                    area_weight=area_weight,
                    device=device,
                )
                state_importance["source_condition"] = state
                state_importance.insert(0, "fold", fold.name)
                importance_tables.append(state_importance)
            prediction_store = run.directory / "fields" / f"{fold.name}.zarr"
            prediction_store.parent.mkdir(parents=True, exist_ok=True)
            field_variables: dict[str, object] = {
                "observed_transformed": (
                    ("time", "latitude", "longitude"),
                    observed,
                ),
                "predicted_central_transformed": (
                    ("time", "latitude", "longitude"),
                    predicted,
                ),
                "predicted_uncertainty_transformed": (
                    ("time", "latitude", "longitude"),
                    scale,
                ),
                "predicted_lower_transformed": (
                    ("time", "latitude", "longitude"),
                    lower,
                ),
                "predicted_upper_transformed": (
                    ("time", "latitude", "longitude"),
                    upper,
                ),
                "persistence_transformed": (
                    ("time", "latitude", "longitude"),
                    baseline,
                ),
                "observed_native_value": (
                    ("time", "latitude", "longitude"),
                    observed_native,
                ),
                "predicted_native_value": (
                    ("time", "latitude", "longitude"),
                    predicted_native,
                ),
                "predicted_lower_native_value": (
                    ("time", "latitude", "longitude"),
                    lower_native,
                ),
                "predicted_upper_native_value": (
                    ("time", "latitude", "longitude"),
                    upper_native,
                ),
                "persistence_native_value": (
                    ("time", "latitude", "longitude"),
                    baseline_native,
                ),
                "source_state": (("time",), test_states.to_numpy(dtype=str)),
                "pixel_id": (
                    ("latitude", "longitude"),
                    source_pixel_ids.reshape(len(target_lat), len(target_lon)),
                ),
            }
            if quantile_fields is not None and quantile_native is not None:
                field_variables["predicted_quantile_transformed"] = (
                    ("time", "quantile", "latitude", "longitude"),
                    quantile_fields,
                )
                field_variables["predicted_quantile_native_value"] = (
                    ("time", "quantile", "latitude", "longitude"),
                    quantile_native,
                )
            xr.Dataset(
                field_variables,
                coords={
                    "time": sequence_data.target_times[test],
                    **(
                        {"quantile": np.asarray(args.quantiles, dtype="float32")}
                        if quantile_fields is not None
                        else {}
                    ),
                    "latitude": target_lat,
                    "longitude": target_lon,
                },
                attrs={
                    "grid_hash_sha256": grid.grid_sha256,
                    "target_interpolation": False,
                    "target_transform_fit_end": str(fold.train_end),
                    "target_variable": args.target_variable,
                    "target_units": target_units,
                    "native_support_postprocessing": support_rule,
                    "uncertainty_semantics": (
                        "gaussian standard deviation"
                        if args.distribution == "gaussian"
                        else "half width of outer requested quantile interval"
                    ),
                    "source_phase_reference": "origin_time",
                },
            ).to_zarr(prediction_store, mode="w", consolidated=True, zarr_format=2)
            run.register_directory(
                prediction_store,
                role="native_chirps_oos_fields",
                description=f"Observed/predicted probabilistic fields for {fold.name}",
            )
            model_path = run.directory / "models" / f"{fold.name}.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "parameters": parameters}, model_path)
            run.register_file(model_path, role="pytorch_state_dict", description=f"F8 {fold.name}")
            contracts.append(
                {
                    "fold": fold.name,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "purge_weeks": fold.purge_weeks,
                    "target_preprocessing_train_only": True,
                    "target_tail_scale_contract": (
                        "train-only conditional-positive L1; N+>=20; otherwise "
                        "train positive threshold floor; no imputation"
                        if target_transformer.tail_fallback_code is not None
                        else "not_applicable"
                    ),
                    "tail_positive_scale_supported_pixels": (
                        int(target_transformer.tail_fallback_code.eq(0).sum())
                        if target_transformer.tail_fallback_code is not None
                        else 0
                    ),
                    "tail_threshold_fallback_pixels": (
                        int(target_transformer.tail_fallback_code.eq(1).sum())
                        if target_transformer.tail_fallback_code is not None
                        else 0
                    ),
                    "tail_unsupported_pixels": (
                        int(target_transformer.tail_fallback_code.eq(2).sum())
                        if target_transformer.tail_fallback_code is not None
                        else 0
                    ),
                    "target_grid_sha256": grid.grid_sha256,
                    "target_interpolation": False,
                    "whole_event_split": True,
                    "event_grouping_time": "origin_time",
                    "sample_weights_train_only": True,
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
                    "n_train_excluded_target_missing": int(
                        (~target_available[fold.train_index]).sum()
                    ),
                    "n_test_excluded_target_missing": int(
                        (~target_available[fold.test_index]).sum()
                    ),
                }
            )
        component_frame = pd.DataFrame(gate_components)
        (
            event_interval_audit,
            interval_fold_audit,
            interval_condition_audit,
        ) = _f8_interval_calibration_audit(
            component_frame,
            maximum_absolute_calibration_error=(
                args.max_interval90_absolute_calibration_error
            ),
        )
        confirmatory_rows: list[dict[str, object]] = []
        for condition in CANONICAL_ACTIVE_CONDITIONS:
            calibration_fields = (
                interval_condition_audit.loc[
                    interval_condition_audit["condition"].eq(condition)
                ]
                .iloc[0]
                .to_dict()
            )
            calibration_fields.pop("condition", None)
            subset = component_frame.loc[
                component_frame.get("condition", pd.Series(dtype=str)).astype(str).eq(
                    condition
                )
            ]
            if subset.empty:
                confirmatory_rows.append(
                    {
                        "condition": condition,
                        "n_sequences": 0,
                        "n_independent_events": 0,
                        "minimum_independent_events_required": (
                            args.minimum_independent_events_per_condition
                        ),
                        "rmse_f8_native": np.nan,
                        "rmse_persistence_native": np.nan,
                        "skill_vs_persistence": np.nan,
                        "rmse_f8_paired_f6_native": np.nan,
                        "rmse_f6_native": np.nan,
                        "skill_vs_f6": np.nan,
                        "paired_coverage_fraction_f6": 0.0,
                        "minimum_event_support_gate_pass": False,
                        "persistence_gate_pass": False,
                        "f6_comparison_gate_pass": False,
                        **calibration_fields,
                        "target_variable": args.target_variable,
                        "target_units": target_units,
                        "horizon_weeks": args.horizon,
                        "f6_reference_model": f6_reference_model,
                        "gate_pass": False,
                        "gate_reason": "active condition absent from OOS evaluation",
                    }
                )
                continue
            if subset["event_id"].duplicated().any():
                raise ValueError(
                    f"{condition}: evento repetido entre folds F8; whole-event OOS violado."
                )
            n_events = int(subset["event_id"].nunique())
            # Confirmatory score is event-equal: compute MSE inside each event
            # first, then average events. Long episodes cannot dominate short
            # ones merely by contributing more weeks.
            rmse_state = float(np.sqrt(subset["event_mse_model"].mean()))
            rmse_persistence_state = float(
                np.sqrt(subset["event_mse_persistence"].mean())
            )
            paired_expected = int(subset["expected_pair_count"].sum())
            paired_count = int(subset["paired_count"].sum())
            paired_coverage = (
                paired_count / paired_expected if paired_expected else np.nan
            )
            event_pair_coverage_min = float(
                pd.to_numeric(
                    subset["paired_coverage_fraction_f6"], errors="coerce"
                ).min()
            )
            paired_model_mse = pd.to_numeric(
                subset["event_mse_model_paired_f6"], errors="coerce"
            )
            paired_f6_mse = pd.to_numeric(
                subset["event_mse_f6"], errors="coerce"
            )
            if (
                f6_reference is not None
                and paired_model_mse.notna().all()
                and paired_f6_mse.notna().all()
            ):
                rmse_state_paired = float(np.sqrt(paired_model_mse.mean()))
                rmse_state_f6 = float(np.sqrt(paired_f6_mse.mean()))
                skill_state_f6 = (
                    1.0 - rmse_state_paired / rmse_state_f6
                    if rmse_state_f6 > 0
                    else np.nan
                )
            else:
                rmse_state_paired = rmse_state_f6 = skill_state_f6 = np.nan
            event_support_pass = bool(
                n_events >= args.minimum_independent_events_per_condition
            )
            persistence_gate_pass = bool(
                np.isfinite(rmse_state)
                and np.isfinite(rmse_persistence_state)
                and rmse_state < rmse_persistence_state
            )
            f6_comparison_gate_pass = bool(
                f6_reference is None
                or (
                    np.isfinite(skill_state_f6)
                    and skill_state_f6 > 0
                    and np.isfinite(paired_coverage)
                    and paired_coverage >= args.minimum_f6_paired_coverage
                    and np.isfinite(event_pair_coverage_min)
                    and event_pair_coverage_min
                    >= args.minimum_f6_paired_coverage
                )
            )
            interval_calibration_gate_pass = bool(
                calibration_fields["interval_calibration_gate_pass"]
            )
            state_pass = _f8_confirmatory_gate_pass(
                minimum_event_support_gate_pass=event_support_pass,
                persistence_gate_pass=persistence_gate_pass,
                f6_comparison_gate_pass=f6_comparison_gate_pass,
                interval_calibration_gate_pass=interval_calibration_gate_pass,
            )
            failed_criteria: list[str] = []
            if not event_support_pass:
                failed_criteria.append("insufficient independent events")
            if not persistence_gate_pass:
                failed_criteria.append("F8 does not beat causal persistence")
            if not f6_comparison_gate_pass:
                failed_criteria.append("paired F6 criterion failed")
            if not interval_calibration_gate_pass:
                failed_criteria.append(
                    "event-equal IC90 calibration failed by condition or fold"
                )
            confirmatory_rows.append(
                {
                    "condition": condition,
                    "n_sequences": int(subset["n_sequences"].sum()),
                    "n_independent_events": n_events,
                    "minimum_independent_events_required": (
                        args.minimum_independent_events_per_condition
                    ),
                    "rmse_f8_native": rmse_state,
                    "rmse_persistence_native": rmse_persistence_state,
                    "skill_vs_persistence": (
                        1.0 - rmse_state / rmse_persistence_state
                        if rmse_persistence_state > 0
                        else np.nan
                    ),
                    "rmse_f8_paired_f6_native": rmse_state_paired,
                    "rmse_f6_native": rmse_state_f6,
                    "skill_vs_f6": skill_state_f6,
                    "paired_coverage_fraction_f6": paired_coverage,
                    "minimum_event_paired_coverage_f6": event_pair_coverage_min,
                    "minimum_paired_coverage_f6": args.minimum_f6_paired_coverage,
                    "minimum_event_support_gate_pass": event_support_pass,
                    "persistence_gate_pass": persistence_gate_pass,
                    "f6_comparison_gate_pass": f6_comparison_gate_pass,
                    **calibration_fields,
                    "target_variable": args.target_variable,
                    "target_units": target_units,
                    "horizon_weeks": args.horizon,
                    "f6_reference_model": f6_reference_model,
                    "gate_pass": state_pass,
                    "gate_reason": (
                        "event-equal skill beats causal persistence and fixed F6; "
                        "central IC90 is calibrated by condition and fold"
                        if state_pass
                        else "; ".join(failed_criteria)
                    ),
                }
            )
        confirmatory_gate = pd.DataFrame(confirmatory_rows)
        overall_gate_pass = bool(
            len(confirmatory_gate) == len(CANONICAL_ACTIVE_CONDITIONS)
            and confirmatory_gate["gate_pass"].all()
        )
        confirmatory_gate["overall_eight_condition_gate_pass"] = overall_gate_pass

        products = {
            "fold_metrics": pd.DataFrame(fold_metrics),
            "pixel_metrics": pd.concat(pixel_metrics, ignore_index=True),
            "input_importance_oos": pd.concat(importance_tables, ignore_index=True),
            "probabilistic_metrics": pd.DataFrame(probabilistic_rows),
            "confirmatory_metrics_by_event": event_interval_audit,
            "probabilistic_gate_by_condition_fold": interval_fold_audit,
            "confirmatory_gate_by_condition": confirmatory_gate,
            "fold_contract": pd.DataFrame(contracts),
            "native_pixel_inventory": pixel_table,
            "predictor_contract": pd.DataFrame({"variable": predictors}),
        }
        for name, frame in products.items():
            run.write_table(
                name,
                frame,
                description={
                    "fold_metrics": "Skill espacial area-ponderado fora da amostra.",
                    "pixel_metrics": "RMSE/r/coverage por pixel CHIRPS original.",
                    "input_importance_oos": "Ablacao dos campos e das 31 variaveis no teste.",
                    "probabilistic_metrics": (
                        "Diagnosticos descritivos pixel-semana: pinball, WIS, cobertura e largura."
                    ),
                    "confirmatory_metrics_by_event": (
                        "Componentes OOS por evento independente; pixels-semanas nativos sao "
                        "ponderados apenas dentro do evento."
                    ),
                    "probabilistic_gate_by_condition_fold": (
                        "Calibracao IC90 event-equal por fold e condicao EN/LN x fase."
                    ),
                    "confirmatory_gate_by_condition": (
                        "Gate F8 isolado nas oito condicoes: skill, F6, suporte e IC90."
                    ),
                    "fold_contract": "Evento inteiro, embargo, hashes e limites de ajuste.",
                    "native_pixel_inventory": "Pixel_id/coordenadas/fração Brasil sem regrid.",
                    "predictor_contract": "As 31 variaveis escalares fundidas ao encoder.",
                }[name],
                methods={"framework": "PyTorch", "target_interpolation": False, "probabilistic": True},
            )
        run.finalize(
            notes=(
                "F8 avanca apenas se as oito condicoes superarem persistencia causal "
                "e F6 pareada e calibrarem IC90 por evento/fold; "
                f"overall_gate_pass={overall_gate_pass}."
            )
        )
        print(f"[F8] run_id={run.run_id} | outputs={run.directory}")
        return 0
    except Exception as exc:
        run.finalize(status="failed", notes=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
