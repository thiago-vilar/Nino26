"""Fase 6 - distribuicao no Brasil com Machine Learning (RF/XGBoost) + XAI.

Mesmo estudo espaco-temporal da Fase 4 (Pacifico -> anomalia de chuva no Brasil),
agora com ML. Para cada unidade oficial (regiao IBGE, bioma, recortes Caatinga e
Mata Atlantica do Nordeste) e cada condicao (todas / fases de El Nino e La Nina),
treina RF/XGBoost prevendo a anomalia de chuva agregada a partir da janela
deslizante de variaveis do Pacifico (lags 4-52 sem). Compara contra o baseline de
persistencia/climatologia e ranqueia variaveis por ganho e (opcional) SHAP.

Reaproveita a agregacao area-ponderada de ``maps.spatial_support`` e o construtor de
janelas de ``models.phase5_cycle_ml``. A validacao e sempre cronologica.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from nino_brasil.models.phase5_cycle_ml import build_lagged_features
from nino_brasil.models.event_validation import (
    ENSO_PHASES,
    ENSO_TYPES,
    assert_continuous_weekly_index,
    canonical_event_ids,
    condition_mask as canonical_condition_mask,
    event_phase_sample_weights,
    make_event_folds,
)
from nino_brasil.models.phase5_cycle_ml import (
    FoldHarmonicPreprocessor,
    OCEAN_RAW_VARS,
    physical_predictor_columns,
    valid_phase_target_mask,
)

PHASE_ORDER: tuple[str, ...] = ("genese", "crescimento", "pico", "decaimento")
ZERO_INFLATED_TAIL_TARGETS: frozenset[str] = frozenset(
    {"r95p_weekly_mm", "r99p_weekly_mm"}
)
MINIMUM_POSITIVE_TAIL_WEEKS = 20


def _make_regressor(model: str, random_state: int):
    if model == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, objective="reg:squarederror",
            random_state=random_state, n_jobs=-1,
        )
    return RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)


def _skill(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[finite], predicted[finite]
    if len(observed) < 3:
        return {"rmse": np.nan, "r": np.nan, "r2_oos": np.nan}
    rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    r = float(np.corrcoef(observed, predicted)[0, 1]) if observed.std() > 0 and predicted.std() > 0 else np.nan
    denom = float(np.sum((observed - observed.mean()) ** 2))
    r2 = 1.0 - float(np.sum((predicted - observed) ** 2)) / denom if denom > 0 else np.nan
    return {"rmse": rmse, "r": r, "r2_oos": r2}


@dataclass
class TeleconnectionMLResult:
    skill: pd.DataFrame
    importances: pd.DataFrame


def fit_unit_teleconnection(
    lagged: pd.DataFrame,
    unit_series: pd.DataFrame,
    phase_table: pd.DataFrame,
    *,
    model: str = "rf",
    conditions: Sequence[str] = ("todas",),
    n_splits: int = 5,
    min_obs: int = 120,
    random_state: int = 42,
) -> TeleconnectionMLResult:
    """Legacy regional pilot; official F6 uses ``fit_pixel_teleconnection``.

    Retained only to reproduce historical summaries and unit tests. It must not
    be used for the F6 gate because it aggregates pixels and uses TimeSeriesSplit.
    """

    skill_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    features = list(lagged.columns)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def condition_mask(name: str) -> pd.Series:
        return canonical_condition_mask(phase_table, name)

    for unit_id in unit_series.columns:
        target = unit_series[unit_id]
        # Persistencia na semana-CALENDARIO anterior, calculada na serie completa
        # antes de qualquer mascara (as linhas condicionadas pulam semanas).
        persistence_full = target.sort_index().shift(1)
        for cond in conditions:
            mask = condition_mask(cond).reindex(lagged.index).fillna(False)
            frame = lagged.join(target.rename("__y")).loc[mask].dropna()
            if len(frame) < min_obs:
                continue
            Xv = frame[features].to_numpy()
            yv = frame["__y"].to_numpy()
            week_of_year = frame.index.isocalendar().week.to_numpy()
            preds = np.full(len(yv), np.nan)
            seasonal = np.full(len(yv), np.nan)
            for train, test in tscv.split(Xv):
                estimator = _make_regressor(model, random_state)
                estimator.fit(Xv[train], yv[train])
                preds[test] = estimator.predict(Xv[test])
                # Baseline sazonal: media por semana-do-ano ajustada SO no treino.
                woy_mean = pd.Series(yv[train]).groupby(week_of_year[train]).mean()
                fallback = float(np.mean(yv[train]))
                seasonal[test] = np.array(
                    [woy_mean.get(w, fallback) for w in week_of_year[test]], dtype=float
                )
            expanding = pd.Series(yv).expanding().mean().shift().to_numpy()
            persistence = persistence_full.reindex(frame.index).to_numpy()
            ml = _skill(yv, preds)
            baselines = {
                "media_expansiva": _skill(yv, expanding),
                "persistencia": _skill(yv, persistence),
                "climatologia_semana_do_ano": _skill(yv, seasonal),
            }
            best_name, best = min(
                ((k, v) for k, v in baselines.items() if np.isfinite(v["rmse"])),
                key=lambda kv: kv[1]["rmse"],
                default=("media_expansiva", baselines["media_expansiva"]),
            )
            skill_rows.append({
                "id_unidade": unit_id, "condicao": cond, "modelo": model,
                "n_obs": len(frame), **{f"{k}_ml": v for k, v in ml.items()},
                "rmse_baseline": baselines["media_expansiva"]["rmse"],
                "rmse_persistencia": baselines["persistencia"]["rmse"],
                "rmse_clim_semana_do_ano": baselines["climatologia_semana_do_ano"]["rmse"],
                "melhor_baseline": best_name,
                "rmse_melhor_baseline": best["rmse"],
                "skill_rmse_vs_baseline": (
                    1.0 - ml["rmse"] / baselines["media_expansiva"]["rmse"]
                    if np.isfinite(baselines["media_expansiva"]["rmse"])
                    and baselines["media_expansiva"]["rmse"] > 0 else np.nan
                ),
                "skill_rmse_vs_melhor_baseline": (
                    1.0 - ml["rmse"] / best["rmse"]
                    if np.isfinite(best["rmse"]) and best["rmse"] > 0 else np.nan
                ),
                "validacao": (
                    "TimeSeriesSplit; baselines=media expansiva, persistencia semanal, "
                    "climatologia semana-do-ano (fit so no treino); gate G3 exige "
                    "skill_rmse_vs_melhor_baseline > 0"
                ),
            })
            estimator = _make_regressor(model, random_state)
            estimator.fit(Xv, yv)
            for feature, imp in zip(features, getattr(estimator, "feature_importances_", np.zeros(len(features)))):
                importance_rows.append({
                    "id_unidade": unit_id, "condicao": cond, "variavel": feature,
                    "importancia_ganho": float(imp),
                })
    return TeleconnectionMLResult(
        skill=pd.DataFrame(skill_rows),
        importances=pd.DataFrame(importance_rows),
    )


def top_importances_by_unit(importances: pd.DataFrame, *, top_n: int = 8) -> pd.DataFrame:
    """Return the ``top_n`` gain-ranked predictors per unit/condition."""

    if importances.empty:
        return importances
    return (
        importances.sort_values("importancia_ganho", ascending=False)
        .groupby(["id_unidade", "condicao"], sort=False)
        .head(top_n)
        .reset_index(drop=True)
    )


CANONICAL_ACTIVE_CONDITIONS: tuple[str, ...] = tuple(
    f"{event_type}_{phase}" for event_type in ENSO_TYPES for phase in ENSO_PHASES
)


@dataclass
class PixelTeleconnectionResult:
    """Pixel-native, out-of-sample F6 products."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    importances: pd.DataFrame
    fold_contract: pd.DataFrame
    grid_contract: pd.DataFrame


def validate_native_pixel_contract(
    response: pd.DataFrame,
    pixel_table: pd.DataFrame,
) -> pd.DataFrame:
    """Validate that F6 consumes the exact original CHIRPS pixel identifiers."""

    import hashlib

    required = {"pixel_id", "lat", "lon"}
    missing = required.difference(pixel_table.columns)
    if missing:
        raise KeyError(f"pixel_table sem colunas obrigatorias: {sorted(missing)}")
    pixels = pixel_table.copy()
    pixels["pixel_id"] = pixels["pixel_id"].astype(str)
    if pixels["pixel_id"].duplicated().any():
        raise ValueError("pixel_id duplicado no contrato CHIRPS.")
    response_ids = pd.Index(response.columns.astype(str))
    table_ids = pd.Index(pixels["pixel_id"])
    if response_ids.duplicated().any():
        raise ValueError("Colunas de resposta CHIRPS duplicadas.")
    if set(response_ids) != set(table_ids):
        missing_response = sorted(set(table_ids).difference(response_ids))[:5]
        missing_table = sorted(set(response_ids).difference(table_ids))[:5]
        raise ValueError(
            "Resposta e pixel_table nao descrevem o mesmo grid nativo; "
            f"ausentes_na_resposta={missing_response}, ausentes_na_tabela={missing_table}."
        )
    canonical = pixels.sort_values("pixel_id")[["pixel_id", "lat", "lon"]]
    payload = canonical.to_csv(index=False, float_format="%.8f").encode("utf-8")
    grid_hash = hashlib.sha256(payload).hexdigest()
    return pd.DataFrame(
        [
            {
                "grid_name": "chirps_native",
                "n_pixels": len(canonical),
                "grid_sha256": grid_hash,
                "interpolation_applied": False,
                "pixel_identity_preserved": True,
            }
        ]
    )


def _make_pixel_regressor(model: str, random_state: int, n_estimators: int):
    if model == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError("XGBoost nao instalado; use model='rf' ou instale xgboost.") from exc
        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    if model != "rf":
        raise ValueError("model deve ser 'rf' ou 'xgb'.")
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def _stable_seed(global_seed: int, *identifiers: object) -> int:
    """Derive a partition-invariant sklearn seed from scientific identifiers."""

    payload = "|".join([str(int(global_seed)), *(str(value) for value in identifiers)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def _circular_event_permutation(
    values: np.ndarray,
    groups: Sequence[object],
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute within whole-event blocks without creating cross-event weeks."""

    output = np.asarray(values, dtype=float).copy()
    labels = pd.Series(list(groups), dtype="string").fillna("missing_group")
    for label in labels.unique():
        positions = np.flatnonzero(labels.eq(label).to_numpy())
        if len(positions) > 1:
            offset = int(rng.integers(1, len(positions)))
            output[positions] = np.roll(output[positions], offset)
    return output


def _pixel_metrics(observed: np.ndarray, predicted: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(observed) & np.isfinite(predicted) & np.isfinite(baseline)
    observed = observed[finite]
    predicted = predicted[finite]
    baseline = baseline[finite]
    if len(observed) < 3:
        return {
            "rmse_ml": np.nan,
            "rmse_baseline": np.nan,
            "mae_ml": np.nan,
            "r_ml": np.nan,
            "skill_rmse_vs_baseline": np.nan,
        }
    rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    rmse_baseline = float(np.sqrt(np.mean((baseline - observed) ** 2)))
    return {
        "rmse_ml": rmse,
        "rmse_baseline": rmse_baseline,
        "mae_ml": float(np.mean(np.abs(predicted - observed))),
        "r_ml": (
            float(np.corrcoef(observed, predicted)[0, 1])
            if observed.std() > 0 and predicted.std() > 0
            else np.nan
        ),
        "skill_rmse_vs_baseline": (
            1.0 - rmse / rmse_baseline if rmse_baseline > 0 else np.nan
        ),
    }


@dataclass(frozen=True)
class FoldTargetTransformer:
    """Train-fitted seasonal target transform with an auditable inverse."""

    method: str
    centre: pd.DataFrame
    scale: pd.DataFrame
    fit_end: pd.Timestamp
    target_name: str = "precip_weekly_mm"
    scale_source: pd.DataFrame | None = None
    positive_week_count: pd.Series | None = None
    pooled_positive_l1_scale: pd.Series | None = None
    tail_threshold_floor: pd.Series | None = None
    tail_fallback_code: pd.Series | None = None

    def transform(self, response: pd.DataFrame) -> pd.DataFrame:
        centre = self.centre.reindex(index=response.index, columns=response.columns)
        scale = self.scale.reindex(index=response.index, columns=response.columns)
        return (response - centre) / scale

    def inverse_array(
        self,
        values: np.ndarray,
        *,
        index: pd.DatetimeIndex | Sequence[object],
        columns: Sequence[object],
    ) -> np.ndarray:
        time = pd.DatetimeIndex(index)
        names = [str(value) for value in columns]
        centre = self.centre.copy()
        scale = self.scale.copy()
        centre.columns = centre.columns.astype(str)
        scale.columns = scale.columns.astype(str)
        centre_values = centre.reindex(index=time, columns=names).to_numpy(dtype=float)
        scale_values = scale.reindex(index=time, columns=names).to_numpy(dtype=float)
        array = np.asarray(values, dtype=float)
        if array.shape != centre_values.shape:
            raise ValueError(
                f"inverse target shape {array.shape} != {(len(time), len(names))}"
            )
        return array * scale_values + centre_values


def fit_fold_target_transformer(
    response: pd.DataFrame,
    *,
    fit_end: pd.Timestamp,
    method: str,
    target_name: str = "precip_weekly_mm",
) -> FoldTargetTransformer:
    """Fit rainfall seasonality/scale on historical target weeks only.

    R95p/R99p weekly amounts use the same contract-v4 hurdle-aware magnitude
    floor as the canonical target builder.  With at least 20 positive training
    weeks, the floor is ``sqrt(pi/2) * mean(abs(residual) | Y>0)``.  With less
    support, the smallest positive weekly training amount is an explicitly
    labelled ``threshold_floor`` proxy; it is not a variance estimate and no
    value is imputed.  A pixel with no positive training amount remains
    undefined.  Every quantity is fitted from ``response.loc[:fit_end]``.
    """

    if method not in {"train_anomaly_mm", "train_robust_z"}:
        if method == "none":
            zero = pd.DataFrame(
                0.0, index=response.index, columns=response.columns, dtype="float32"
            )
            one = pd.DataFrame(
                1.0, index=response.index, columns=response.columns, dtype="float32"
            )
            return FoldTargetTransformer(
                method, zero, one, pd.Timestamp(fit_end), target_name=target_name
            )
        raise ValueError("target_transform invalido.")
    history = response.loc[:fit_end]
    if len(history) < 104:
        raise ValueError("Historico de treino insuficiente para climatologia CHIRPS.")
    history_week = history.index.isocalendar().week.to_numpy()
    all_week = response.index.isocalendar().week.to_numpy()
    centres = history.groupby(history_week).median()
    fallback_centre = history.median()
    centre_rows = np.vstack(
        [centres.loc[week].to_numpy() if week in centres.index else fallback_centre.to_numpy() for week in all_week]
    )
    centre_frame = pd.DataFrame(
        centre_rows, index=response.index, columns=response.columns, dtype="float32"
    )
    anomaly = response.to_numpy(dtype=float) - centre_rows
    transformed = pd.DataFrame(anomaly, index=response.index, columns=response.columns)
    if method == "train_anomaly_mm":
        scale_frame = pd.DataFrame(
            1.0, index=response.index, columns=response.columns, dtype="float32"
        )
        return FoldTargetTransformer(
            method,
            centre_frame,
            scale_frame,
            pd.Timestamp(fit_end),
            target_name=target_name,
        )
    history_anomaly = transformed.loc[history.index]
    seasonal_scale = history_anomaly.abs().groupby(history_week).median() * 1.4826
    pooled_scale = history_anomaly.abs().median() * 1.4826
    pooled_l1_scale = history_anomaly.abs().mean() * np.sqrt(np.pi / 2.0)
    is_zero_inflated_tail = str(target_name).lower() in ZERO_INFLATED_TAIL_TARGETS
    positive_week_count: pd.Series | None = None
    pooled_positive_l1_scale: pd.Series | None = None
    tail_threshold_floor: pd.Series | None = None
    tail_fallback_code: pd.Series | None = None
    if is_zero_inflated_tail:
        positive = history > 0
        positive_week_count = positive.sum(axis=0).astype("int64")
        pooled_positive_l1_scale = (
            history_anomaly.abs().where(positive).mean(axis=0)
            * np.sqrt(np.pi / 2.0)
        )
        # A weekly R95p/R99p amount is positive only after the underlying
        # daily threshold has been crossed.  Its smallest positive *training*
        # value is therefore a leakage-safe physical floor when too few
        # positive weeks exist to estimate a magnitude scale.
        tail_threshold_floor = history.where(positive).min(axis=0)
        positive_supported = (
            positive_week_count.ge(MINIMUM_POSITIVE_TAIL_WEEKS)
            & pooled_positive_l1_scale.ge(0.10)
            & np.isfinite(pooled_positive_l1_scale)
        )
        threshold_supported = (
            ~positive_supported
            & tail_threshold_floor.ge(0.10)
            & np.isfinite(tail_threshold_floor)
        )
        tail_fallback_code = pd.Series(
            np.select(
                [positive_supported.to_numpy(), threshold_supported.to_numpy()],
                [0, 1],
                default=2,
            ).astype("uint8"),
            index=history.columns,
            name="tail_fallback_code",
        )

    scale_by_week: dict[int, pd.Series] = {}
    source_by_week: dict[int, pd.Series] = {}
    for week in sorted({int(value) for value in all_week}):
        seasonal = seasonal_scale.loc[week] if week in seasonal_scale.index else pooled_scale
        candidate_values = [seasonal, pooled_scale, pooled_l1_scale]
        candidate_names = ["seasonal_mad", "pooled_mad", "pooled_l1"]
        if is_zero_inflated_tail:
            if any(
                value is None
                for value in (
                    positive_week_count,
                    pooled_positive_l1_scale,
                    tail_threshold_floor,
                    tail_fallback_code,
                )
            ):
                raise AssertionError("tail-scale audit state is incomplete")
            positive_candidate = pooled_positive_l1_scale.where(
                positive_week_count.ge(MINIMUM_POSITIVE_TAIL_WEEKS)
            )
            threshold_candidate = tail_threshold_floor.where(
                tail_fallback_code.eq(1)
            )
            candidate_values.extend([positive_candidate, threshold_candidate])
            candidate_names.extend(["pooled_positive_l1", "threshold_floor"])
        candidates = pd.concat(
            candidate_values, axis=1, keys=candidate_names
        )
        candidates = candidates.where(candidates >= 0.10)
        selected = candidates.max(axis=1, skipna=True)
        source = pd.Series("undefined", index=candidates.index, dtype="string")
        for candidate_name in reversed(candidate_names):
            source.loc[
                selected.notna() & selected.eq(candidates[candidate_name])
            ] = candidate_name
        scale_by_week[week] = selected
        source_by_week[week] = source
    scale = np.vstack(
        [scale_by_week[int(week)].to_numpy(dtype=float) for week in all_week]
    )
    scale_frame = pd.DataFrame(
        scale, index=response.index, columns=response.columns, dtype="float32"
    )
    scale_source = pd.DataFrame.from_dict(source_by_week, orient="index")
    scale_source.index.name = "week_of_year"
    scale_source = scale_source.reindex(columns=response.columns)
    return FoldTargetTransformer(
        method,
        centre_frame,
        scale_frame,
        pd.Timestamp(fit_end),
        target_name=target_name,
        scale_source=scale_source,
        positive_week_count=positive_week_count,
        pooled_positive_l1_scale=pooled_positive_l1_scale,
        tail_threshold_floor=tail_threshold_floor,
        tail_fallback_code=tail_fallback_code,
    )


def fold_target_transform(
    response: pd.DataFrame,
    *,
    fit_end: pd.Timestamp,
    method: str,
    target_name: str = "precip_weekly_mm",
) -> pd.DataFrame:
    """Compatibility wrapper returning the transformed target DataFrame."""

    transformer = fit_fold_target_transformer(
        response, fit_end=fit_end, method=method, target_name=target_name
    )
    return transformer.transform(response)


def causal_persistence_target(
    response: pd.DataFrame,
    transformer: FoldTargetTransformer,
    *,
    horizon_weeks: int,
) -> pd.DataFrame:
    """Encode y(t-h) on the target-time scale without using future rainfall."""

    if horizon_weeks < 1:
        raise ValueError("horizon_weeks must be >= 1 for a predictive persistence baseline.")
    return transformer.transform(response.shift(int(horizon_weeks)))


def fit_pixel_teleconnection(
    master: pd.DataFrame,
    pixel_response: pd.DataFrame,
    pixel_table: pd.DataFrame,
    phase_table: pd.DataFrame,
    *,
    predictors: Sequence[str] | None = None,
    model: str = "rf",
    lags_weeks: Sequence[int] = (4, 8, 12, 16, 20, 24),
    conditions: Sequence[str] = CANONICAL_ACTIVE_CONDITIONS,
    n_splits: int = 4,
    min_train_events: int = 3,
    min_train_rows: int = 24,
    min_test_rows: int = 3,
    n_estimators: int = 250,
    store_predictions: bool = True,
    target_transform: str = "none",
    target_name: str = "precip_weekly_mm",
    target_units: str = "native target units",
    permutation_repeats: int = 1,
    random_state: int = 42,
) -> PixelTeleconnectionResult:
    """RF/XGB per original CHIRPS pixel, lag and source ENSO phase.

    For response week ``t`` and lag ``L``, both predictors and the condition are
    evaluated at source week ``t-L``.  No spatial interpolation or regional
    aggregation occurs.  Folds are expanding, whole-event and purged.  Regional
    or biome summaries must be derived *after* these pixel predictions.
    """

    if not all(
        isinstance(frame.index, pd.DatetimeIndex)
        for frame in (master, pixel_response, phase_table)
    ):
        raise TypeError("master, pixel_response e phase_table devem usar DatetimeIndex.")
    grid_contract = validate_native_pixel_contract(pixel_response, pixel_table)
    predictors = list(predictors or physical_predictor_columns(master))
    common = master.index.intersection(pixel_response.index).intersection(phase_table.index).sort_values()
    assert_continuous_weekly_index(common, name="F6 common weekly index")
    master = master.reindex(common)
    response = pixel_response.reindex(common).copy()
    response.columns = response.columns.astype(str)
    pixels = pixel_table.copy()
    pixels["pixel_id"] = pixels["pixel_id"].astype(str)
    pixels = pixels.set_index("pixel_id")
    phase_table = phase_table.reindex(common)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    contract_rows: list[dict[str, object]] = []

    for lag in sorted({int(value) for value in lags_weeks}):
        if lag < 0:
            raise ValueError("lags_weeks nao pode conter valores negativos.")
        source_phase = phase_table.shift(lag)
        # Neutral weeks legitimately have no event_id; active weeks must have
        # one.  This keeps condition="todas" semantically complete.
        valid_source = valid_phase_target_mask(source_phase)
        for condition in conditions:
            cond_mask = canonical_condition_mask(source_phase, condition) & valid_source
            condition_times = common[cond_mask.to_numpy()]
            if len(condition_times) < min_train_rows + min_test_rows:
                continue
            phase_subset = source_phase.reindex(condition_times)
            groups = canonical_event_ids(phase_subset)
            try:
                folds = make_event_folds(
                    condition_times,
                    groups,
                    n_splits=n_splits,
                    min_train_groups=min_train_events,
                    purge_weeks=lag,
                )
            except ValueError:
                continue
            for fold_no, fold in enumerate(folds, start=1):
                train_times = condition_times[fold.train_index]
                test_times = condition_times[fold.test_index]
                if len(train_times) < min_train_rows or len(test_times) < min_test_rows:
                    continue
                feature_fit_end = fold.train_end - pd.Timedelta(weeks=lag)
                fit_history = master.loc[:feature_fit_end, predictors]
                source_codes = master.get("ocean_source_code")
                preprocessor = FoldHarmonicPreprocessor(tuple(predictors)).fit(
                    fit_history,
                    source_codes=(source_codes.loc[fit_history.index] if source_codes is not None else None),
                )
                source_features = preprocessor.transform(
                    master[predictors], source_codes=source_codes
                ).shift(lag)
                target_transformer = fit_fold_target_transformer(
                    response,
                    fit_end=fold.train_end,
                    method=target_transform,
                    target_name=target_name,
                )
                fold_response = target_transformer.transform(response)
                # A causal persistence forecast for response week t can only
                # use the target observed at source/origin week t-lag.  Shift
                # the *raw* response first and express it in the target week's
                # fold-fitted scale; shifting an already standardized series
                # would mix seasonal centres and fail to invert to y(t-lag).
                fold_persistence = causal_persistence_target(
                    response, target_transformer, horizon_weeks=lag
                )
                X_train = source_features.reindex(train_times)
                X_test = source_features.reindex(test_times)
                usable = X_train.notna().any(axis=0)
                X_train = X_train.loc[:, usable]
                X_test = X_test.loc[:, usable]
                ocean_features = [name for name in X_train.columns if name in OCEAN_RAW_VARS]
                train_source_valid = (
                    X_train[ocean_features].notna().all(axis=1)
                    if ocean_features
                    else pd.Series(True, index=X_train.index)
                )
                test_source_valid = (
                    X_test[ocean_features].notna().all(axis=1)
                    if ocean_features
                    else pd.Series(True, index=X_test.index)
                )
                medians = X_train.median(numeric_only=True)
                X_train = X_train.fillna(medians)
                X_test = X_test.fillna(medians)
                feature_names = list(X_train.columns)
                # Event/type/phase frequencies are learned inside the training
                # portion only.  Computing them on ``phase_subset`` would let
                # future test events alter the optimisation weights.
                weights = event_phase_sample_weights(
                    phase_subset.reindex(train_times)
                )
                contract_rows.append(
                    {
                        "fold": fold.name,
                        "condition": condition,
                        "lag_weeks": lag,
                        "train_end_response": fold.train_end,
                        "preprocessing_fit_end_source": feature_fit_end,
                        "test_start": fold.test_start,
                        "test_end": fold.test_end,
                        "purge_weeks": fold.purge_weeks,
                        "n_train_events": len(fold.train_groups),
                        "n_test_events": len(fold.test_groups),
                        "train_event_ids": ";".join(map(str, fold.train_groups)),
                        "test_event_ids": ";".join(map(str, fold.test_groups)),
                        "phase_evaluated_at": "source_t_minus_lag",
                        "target_variable": target_name,
                        "target_tail_scale_contract": (
                            "train-only conditional-positive L1; N+>=20; "
                            "otherwise train positive threshold floor; no imputation"
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
                    }
                )
                for pixel_no, pixel_id in enumerate(response.columns):
                    y_train = pd.to_numeric(fold_response.loc[train_times, pixel_id], errors="coerce")
                    y_test = pd.to_numeric(fold_response.loc[test_times, pixel_id], errors="coerce")
                    train_ok = (
                        y_train.notna()
                        & np.isfinite(X_train.to_numpy()).all(axis=1)
                        & train_source_valid
                    )
                    test_ok = (
                        y_test.notna()
                        & np.isfinite(X_test.to_numpy()).all(axis=1)
                        & test_source_valid
                    )
                    if int(train_ok.sum()) < min_train_rows or int(test_ok.sum()) < min_test_rows:
                        continue
                    stable_seed = _stable_seed(
                        random_state, model, condition, lag, fold.name, pixel_id
                    )
                    estimator = _make_pixel_regressor(
                        model,
                        random_state=stable_seed,
                        n_estimators=n_estimators,
                    )
                    estimator.fit(
                        X_train.loc[train_ok].to_numpy(dtype=float),
                        y_train.loc[train_ok].to_numpy(dtype=float),
                        sample_weight=weights.loc[train_ok].to_numpy(dtype=float),
                    )
                    test_index = y_test.index[test_ok]
                    observed = y_test.loc[test_index].to_numpy(dtype=float)
                    predicted = estimator.predict(X_test.loc[test_index].to_numpy(dtype=float))
                    persistence = pd.to_numeric(
                        fold_persistence[pixel_id], errors="coerce"
                    ).reindex(test_index)
                    # Confirmatory F4 comparator: a predeclared linear Ridge
                    # relation fitted on the identical train fold and evaluated
                    # on the identical OOS pixel-weeks.  This operationalizes
                    # the required F4(statistical) -> F6(ML) incremental gate.
                    statistical = Ridge(alpha=1.0)
                    statistical.fit(
                        X_train.loc[train_ok].to_numpy(dtype=float),
                        y_train.loc[train_ok].to_numpy(dtype=float),
                        sample_weight=weights.loc[train_ok].to_numpy(dtype=float),
                    )
                    statistical_prediction = statistical.predict(
                        X_test.loc[test_index].to_numpy(dtype=float)
                    )
                    train_mean = float(y_train.loc[train_ok].mean())
                    week_mean = y_train.loc[train_ok].groupby(
                        y_train.loc[train_ok].index.isocalendar().week
                    ).mean()
                    seasonal = np.asarray(
                        [week_mean.get(week, train_mean) for week in test_index.isocalendar().week],
                        dtype=float,
                    )
                    baseline_options = {
                        "climatology_train_mean": np.full(len(observed), train_mean),
                        "climatology_week_of_year_train": seasonal,
                        "persistence_week": persistence.to_numpy(dtype=float),
                        "phase4_statistical_ridge": statistical_prediction,
                    }
                    baseline_scores = {}
                    for name, values in baseline_options.items():
                        valid_baseline = np.isfinite(values) & np.isfinite(observed)
                        baseline_scores[name] = (
                            float(
                                np.sqrt(
                                    np.mean(
                                        (values[valid_baseline] - observed[valid_baseline]) ** 2
                                    )
                                )
                            )
                            if int(valid_baseline.sum()) >= min_test_rows
                            else np.nan
                        )
                    # Persistence is the predeclared operational comparator.
                    # The gate additionally requires the model to beat *every*
                    # independently reported baseline.  No baseline is selected
                    # retrospectively from test outcomes.
                    primary_name = "persistence_week"
                    primary_values = baseline_options[primary_name]
                    scores = _pixel_metrics(observed, predicted, primary_values)
                    rmse_ml = scores["rmse_ml"]
                    skills = {
                        name: (
                            1.0 - rmse_ml / rmse
                            if np.isfinite(rmse_ml) and np.isfinite(rmse) and rmse > 0
                            else np.nan
                        )
                        for name, rmse in baseline_scores.items()
                    }
                    finite_skills = [value for value in skills.values() if np.isfinite(value)]
                    gate_pass = bool(
                        len(finite_skills) == len(baseline_options)
                        and all(value > 0 for value in finite_skills)
                    )
                    pixel_meta = pixels.loc[pixel_id]
                    metric_rows.append(
                        {
                            "fold": fold.name,
                            "pixel_id": pixel_id,
                            "lat": float(pixel_meta["lat"]),
                            "lon": float(pixel_meta["lon"]),
                            "condition": condition,
                            "lag_weeks": lag,
                            "model": model,
                            "n_train": int(train_ok.sum()),
                            "n_test": int(test_ok.sum()),
                            "primary_baseline": primary_name,
                            "baseline_selection": "predeclared; no test-set selection",
                            "rmse_climatology_train_mean": baseline_scores[
                                "climatology_train_mean"
                            ],
                            "rmse_climatology_week_of_year_train": baseline_scores[
                                "climatology_week_of_year_train"
                            ],
                            "rmse_persistence_week": baseline_scores["persistence_week"],
                            "rmse_phase4_statistical_ridge": baseline_scores[
                                "phase4_statistical_ridge"
                            ],
                            "skill_vs_climatology_train_mean": skills[
                                "climatology_train_mean"
                            ],
                            "skill_vs_climatology_week_of_year_train": skills[
                                "climatology_week_of_year_train"
                            ],
                            "skill_vs_persistence_week": skills["persistence_week"],
                            "skill_vs_phase4_statistical_ridge": skills[
                                "phase4_statistical_ridge"
                            ],
                            "persistence_origin_lag_weeks": lag,
                            "phase4_comparator": "Ridge(alpha=1.0), identical fold/features",
                            **scores,
                            "gate_pass": gate_pass,
                            "target_grid": "chirps_native",
                            "target_transform": target_transform,
                            "target_variable": target_name,
                        }
                    )
                    training_gain = dict(
                        zip(
                            feature_names,
                            getattr(
                                estimator,
                                "feature_importances_",
                                np.zeros(len(feature_names)),
                            ),
                        )
                    )
                    X_test_valid = X_test.loc[test_index].to_numpy(dtype=float)
                    base_valid = np.isfinite(observed) & np.isfinite(predicted)
                    base_rmse = (
                        float(
                            np.sqrt(
                                np.mean((predicted[base_valid] - observed[base_valid]) ** 2)
                            )
                        )
                        if int(base_valid.sum()) >= min_test_rows
                        else np.nan
                    )
                    rng = np.random.default_rng(
                        _stable_seed(
                            random_state,
                            "permutation",
                            model,
                            condition,
                            lag,
                            fold.name,
                            pixel_id,
                        )
                    )
                    test_event_groups = canonical_event_ids(
                        phase_subset.reindex(test_index)
                    ).to_numpy(dtype=object)
                    for feature_no, feature in enumerate(feature_names):
                        deltas: list[float] = []
                        for _ in range(max(0, int(permutation_repeats))):
                            permuted = X_test_valid.copy()
                            permuted[:, feature_no] = _circular_event_permutation(
                                permuted[:, feature_no], test_event_groups, rng
                            )
                            changed = estimator.predict(permuted)
                            valid_changed = np.isfinite(observed) & np.isfinite(changed)
                            if int(valid_changed.sum()) < min_test_rows:
                                continue
                            changed_rmse = float(
                                np.sqrt(
                                    np.mean(
                                        (changed[valid_changed] - observed[valid_changed]) ** 2
                                    )
                                )
                            )
                            deltas.append(changed_rmse - base_rmse)
                        importance_rows.append(
                            {
                                "fold": fold.name,
                                "pixel_id": pixel_id,
                                "condition": condition,
                                "lag_weeks": lag,
                                "variable": feature,
                                "importance_gain_train": float(training_gain[feature]),
                                "delta_rmse_permutation_oos": (
                                    float(np.mean(deltas)) if deltas else np.nan
                                ),
                                "n_permutation_repeats": len(deltas),
                                "importance_scope": (
                                    "out_of_sample_circular_within_event_permutation"
                                ),
                            }
                        )
                    if store_predictions:
                        predicted_raw = target_transformer.inverse_array(
                            predicted[:, None], index=test_index, columns=[pixel_id]
                        )[:, 0]
                        persistence_native = target_transformer.inverse_array(
                            primary_values[:, None], index=test_index, columns=[pixel_id]
                        )[:, 0]
                        statistical_native = target_transformer.inverse_array(
                            statistical_prediction[:, None],
                            index=test_index,
                            columns=[pixel_id],
                        )[:, 0]
                        observed_native = pd.to_numeric(
                            response.loc[test_index, pixel_id], errors="coerce"
                        ).to_numpy(dtype=float)
                        source_rows = phase_subset.reindex(test_index)
                        for row_no, (time, obs, pred) in enumerate(
                            zip(test_index, observed, predicted)
                        ):
                            source_row = source_rows.iloc[row_no]
                            prediction_rows.append(
                                {
                                    "fold": fold.name,
                                    "time": time,
                                    "source_time": time - pd.Timedelta(weeks=lag),
                                    "pixel_id": pixel_id,
                                    "condition": condition,
                                    "lag_weeks": lag,
                                    "model": model,
                                    "target_transform": target_transform,
                                    "target_variable": target_name,
                                    "target_units": target_units,
                                    "observed": float(obs),
                                    "predicted": float(pred),
                                    "observed_native_value": float(
                                        observed_native[row_no]
                                    ),
                                    "predicted_native_value": float(predicted_raw[row_no]),
                                    "baseline_persistence_native_value": float(
                                        persistence_native[row_no]
                                    ),
                                    "baseline_phase4_statistical_native_value": float(
                                        statistical_native[row_no]
                                    ),
                                    "baseline_persistence": float(
                                        baseline_options["persistence_week"][row_no]
                                    ),
                                    "baseline_climatology_train_mean": float(
                                        baseline_options["climatology_train_mean"][row_no]
                                    ),
                                    "baseline_climatology_week_of_year_train": float(
                                        baseline_options[
                                            "climatology_week_of_year_train"
                                        ][row_no]
                                    ),
                                    "baseline_phase4_statistical_ridge": float(
                                        statistical_prediction[row_no]
                                    ),
                                    "primary_baseline": primary_name,
                                    "persistence_origin_lag_weeks": lag,
                                    "source_event_id": source_row.get("event_id"),
                                    "source_tipo": source_row.get("tipo"),
                                    "source_fase": source_row.get("fase"),
                                }
                            )
    return PixelTeleconnectionResult(
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.DataFrame(prediction_rows),
        importances=pd.DataFrame(importance_rows),
        fold_contract=pd.DataFrame(contract_rows).drop_duplicates().reset_index(drop=True),
        grid_contract=grid_contract,
    )
