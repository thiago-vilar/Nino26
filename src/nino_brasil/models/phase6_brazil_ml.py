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
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from nino_brasil.models.phase5_cycle_ml import build_lagged_features

PHASE_ORDER: tuple[str, ...] = ("genese", "crescimento", "pico", "decaimento")


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
    """RF/XGB per unit and condition; chronological skill vs mean baseline + gains."""

    skill_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    features = list(lagged.columns)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def condition_mask(name: str) -> pd.Series:
        if name == "todas":
            return pd.Series(True, index=phase_table.index)
        event_type, _, phase = name.partition("_")
        return phase_table["tipo"].eq(event_type) & phase_table["fase"].eq(phase)

    for unit_id in unit_series.columns:
        target = unit_series[unit_id]
        for cond in conditions:
            mask = condition_mask(cond).reindex(lagged.index).fillna(False)
            frame = lagged.join(target.rename("__y")).loc[mask].dropna()
            if len(frame) < min_obs:
                continue
            Xv = frame[features].to_numpy()
            yv = frame["__y"].to_numpy()
            preds = np.full(len(yv), np.nan)
            for train, test in tscv.split(Xv):
                estimator = _make_regressor(model, random_state)
                estimator.fit(Xv[train], yv[train])
                preds[test] = estimator.predict(Xv[test])
            baseline = pd.Series(yv).expanding().mean().shift().to_numpy()
            ml = _skill(yv, preds)
            base = _skill(yv, baseline)
            skill_rows.append({
                "id_unidade": unit_id, "condicao": cond, "modelo": model,
                "n_obs": len(frame), **{f"{k}_ml": v for k, v in ml.items()},
                "rmse_baseline": base["rmse"],
                "skill_rmse_vs_baseline": (
                    1.0 - ml["rmse"] / base["rmse"] if np.isfinite(base["rmse"]) and base["rmse"] > 0 else np.nan
                ),
                "validacao": "TimeSeriesSplit; baseline=media expansiva",
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
