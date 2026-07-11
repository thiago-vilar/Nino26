"""Fase 5 - projecao do ciclo ENSO com Random Forest e XGBoost + XAI.

Mesmo mecanismo da Fase 3 (genese -> crescimento -> pico -> decaimento), agora
com Machine Learning. Duas frentes preditivas:

* **Classificacao semanal de fase** do ciclo (4 classes) a partir da matriz
  multivariada transformada por janela deslizante (lags de 4 a 52 semanas). Serve
  para mapear as 4 fases e ranquear, por SHAP/impureza, as variaveis mais
  significativas de cada etapa.
* **Projecao do evento** a partir de condicoes precursoras da genese:
  ``Y_pico`` (intensidade maxima), ``Y_tempo_para_pico`` (semanas de antecedencia)
  e ``Y_duracao`` (periodo continuo com ONI/OISST >= +0.5 C por >=5 estacoes
  moveis sobrepostas).

Regras de negocio obrigatorias implementadas aqui:

- **Coerencia fisica**: :func:`convert_precip_m_per_day_to_mm_month` converte a
  precipitacao ``tp`` (m/dia do ERA5) para mm acumulados mensais antes do modelo.
- **Anomalia antes do modelo** (parecer 2026-07-10):
  :func:`prepare_pacific_predictors` remove ciclo anual + tendencia das
  variaveis oceanicas cruas; sem isso o modelo aprende calendario.
- **Eixo de recarga**: os lags de 15-20 semanas de ``OHC0-300``, ``SSH``, ``D20`` e
  ``tau_x`` sao marcados como precursores dinamicos (``recharge_precursor``).
- **Selecao**: RFECV por ganho/impureza (sem importancia a priori).
- **Validacao cronologica**: ``TimeSeriesSplit`` e leave-one-event-out; NUNCA
  split aleatorio. Baselines obrigatorios: semana-do-ano e persistencia.
- **Desbalanceamento**: jitter gaussiano e, quando disponivel, SMOGN.
- **XAI**: SHAP (summary/force/waterfall) e dependencia parcial (PDP).

Dependencias opcionais (``xgboost``, ``shap``, ``smogn``) sao importadas sob
guarda: o modulo importa mesmo sem elas, e cada rotina avisa o que falta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit

PHASE_ORDER: tuple[str, ...] = ("genese", "crescimento", "pico", "decaimento")
# Precursores dinamicos do eixo de recarga subsuperficial/atmosferico.
RECHARGE_PRECURSORS: tuple[str, ...] = ("ohc_0_300", "ssh_m", "d20_m", "tau_x_anom")
RECHARGE_LAG_WINDOW: tuple[int, int] = (15, 20)  # semanas de antecedencia monitoradas
# Semanas por mes: cada "estacao movel" de 3 meses avanca de UM mes.
WEEKS_PER_MONTH: float = 365.2425 / 12.0 / 7.0  # ~4.348
# Variaveis oceanicas do master que sao valores fisicos CRUS (nao anomalias).
# nino34_ssta e as variaveis *_anom do ERA5 ja sao anomalias e ficam de fora.
OCEAN_RAW_VARS: tuple[str, ...] = (
    "d20_m", "tilt_m", "tilt_slope",
    "ohc_0_100", "ohc_0_300", "ohc_0_700", "ohc_300_700",
    "ssh_m", "wwv",
    "t50m", "t100m", "t150m", "t200m", "t300m", "t500m", "t700m",
)


def prepare_pacific_predictors(
    master: pd.DataFrame,
    predictors: Sequence[str],
    *,
    anomalize_ocean: bool = True,
    base: tuple[str, str] = ("1991-01-01", "2020-12-31"),
    harmonics: int = 3,
    detrend: bool = True,
) -> pd.DataFrame:
    """Seleciona preditores do master e anomaliza as variaveis oceanicas cruas.

    Regra cientifica (parecer 2026-07-10): SSTA e as variaveis ERA5 ja sao
    anomalias; as variaveis oceanicas de subsuperficie sao cruas e o ciclo anual
    domina sua variancia semanal (ex.: D20 com amplitude sazonal ~25 m vs sigma
    total ~15 m). Sem esta etapa, classificadores de fase e correlacoes
    condicionadas podem aprender calendario (phase-locking) em vez de mecanismo
    interanual. Climatologia e tendencia sao ajustadas SO na base (1991-2020) e
    extrapoladas deterministicamente - nada do periodo de teste entra no fit.
    """
    from nino_brasil.stats.climatology import harmonic_anomaly_matrix

    present = [c for c in predictors if c in master.columns]
    frame = master[present].copy()
    if not anomalize_ocean:
        return frame
    targets = [c for c in present if c in OCEAN_RAW_VARS]
    if not targets:
        return frame
    return harmonic_anomaly_matrix(
        frame, targets, base=base, harmonics=harmonics, detrend=detrend
    )


def convert_precip_m_per_day_to_mm_month(
    precip_m_per_day: pd.Series,
    *,
    days_in_period: pd.Series | float = 30.0,
) -> pd.Series:
    """Convert ERA5 ``tp`` (m/day) to accumulated mm over the period.

    ``1 m/day = 1000 mm/day``; multiplying by the number of days in the period
    yields accumulated millimetres, preserving the water balance before the model.
    """

    mm_per_day = pd.to_numeric(precip_m_per_day, errors="coerce") * 1000.0
    out = mm_per_day * days_in_period
    out.name = (precip_m_per_day.name or "tp") + "_mm_acumulado"
    return out


def build_lagged_features(
    matrix: pd.DataFrame,
    predictors: Sequence[str],
    *,
    lags: Sequence[int] = tuple(range(4, 53, 2)),
) -> pd.DataFrame:
    """Sliding window: feature ``var_lagL`` = predictor value at ``t - L`` weeks.

    Only past information enters row ``t`` (physically causal). Columns whose
    predictor/lag fall in the recharge precursor window are named so they can be
    audited (``__recharge`` suffix) but keep numeric dtype.
    """

    matrix = matrix.sort_index()
    frames: dict[str, pd.Series] = {}
    for name in predictors:
        if name not in matrix.columns:
            continue
        base = pd.to_numeric(matrix[name], errors="coerce")
        for lag in lags:
            column = f"{name}_lag{int(lag)}"
            if name in RECHARGE_PRECURSORS and RECHARGE_LAG_WINDOW[0] <= lag <= RECHARGE_LAG_WINDOW[1]:
                column += "__recharge"
            frames[column] = base.shift(int(lag))
    features = pd.DataFrame(frames, index=matrix.index)
    return features


def build_event_targets(events: pd.DataFrame) -> pd.DataFrame:
    """Per-event regression/classification targets from the 4A event table.

    Expects the columns produced by ``fase4_utils.enso_events``: ``event_id``,
    ``tipo``, ``onset``, ``pico``, ``fim``, ``duracao_estacoes``, ``oni_pico_c``.
    """

    required = {"event_id", "tipo", "onset", "pico", "oni_pico_c", "duracao_estacoes"}
    missing = required.difference(events.columns)
    if missing:
        raise KeyError(f"events is missing columns: {sorted(missing)}")
    out = events.copy()
    onset = pd.to_datetime(out["onset"])
    peak = pd.to_datetime(out["pico"])
    out["Y_pico"] = pd.to_numeric(out["oni_pico_c"], errors="coerce").abs()
    out["Y_tempo_para_pico_sem"] = (peak - onset).dt.days / 7.0
    # Duracao em semanas por DIFERENCA DE DATAS (onset -> fim). As "estacoes
    # moveis" de 3 meses avancam de UM mes; multiplicar por 13 tratava cada
    # estacao como trimestre disjunto e inflava a duracao ~3x (bug corrigido no
    # parecer 2026-07-10; ex.: EN 1997/98 saia com 143 sem em vez de ~48).
    if "fim" in out.columns:
        fim = pd.to_datetime(out["fim"])
        out["Y_duracao_sem"] = (fim - onset).dt.days / 7.0
    else:  # tabela legada sem 'fim': estacoes ~ meses -> semanas por mes.
        out["Y_duracao_sem"] = (
            pd.to_numeric(out["duracao_estacoes"], errors="coerce") * WEEKS_PER_MONTH
        )
    return out[
        ["event_id", "tipo", "onset", "pico", "Y_pico", "Y_tempo_para_pico_sem", "Y_duracao_sem"]
    ]


def precursor_features_at_onset(
    lagged: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Read the sliding-window features on each event's onset week."""

    index = lagged.index
    rows: list[pd.Series] = []
    for _, event in events.iterrows():
        onset = pd.Timestamp(event["onset"])
        position = index.get_indexer([onset], method="nearest")[0]
        row = lagged.iloc[position].copy()
        row.name = event["event_id"]
        rows.append(row)
    return pd.DataFrame(rows)


def leave_one_event_out_indices(event_ids: Sequence[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Nested leave-one-event-out folds (chronology preserved within train)."""

    event_ids = np.asarray(list(event_ids))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for held in pd.unique(event_ids):
        test = np.flatnonzero(event_ids == held)
        train = np.flatnonzero(event_ids != held)
        if len(train) and len(test):
            folds.append((train, test))
    return folds


def gaussian_jitter(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    sigma: float = 0.05,
    n_copies: int = 1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Additive Gaussian noise (jittering) for tabular regularisation.

    Noise scales with each column's standard deviation, so the perturbation is
    dimensionless and never dominates a physical variable.
    """

    rng = np.random.default_rng(random_state)
    scale = X.std(ddof=0).to_numpy()[None, :]
    parts_x = [X]
    parts_y = [y]
    for _ in range(max(int(n_copies), 0)):
        noise = rng.normal(0.0, sigma, size=X.shape) * scale
        parts_x.append(pd.DataFrame(X.to_numpy() + noise, columns=X.columns))
        parts_y.append(pd.Series(y.to_numpy(), name=y.name))
    return (
        pd.concat(parts_x, ignore_index=True),
        pd.concat(parts_y, ignore_index=True),
    )


def smogn_augment(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """SMOGN synthetic resampling for regression, when the package is present."""

    try:
        import smogn  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "smogn nao instalado; use gaussian_jitter ou 'pip install smogn'."
        ) from exc
    frame = X.copy()
    frame["__target"] = y.to_numpy()
    resampled = smogn.smoter(data=frame, y="__target")
    return resampled.drop(columns="__target"), resampled["__target"].rename(y.name)


def rfecv_select(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task: str = "classification",
    n_splits: int = 5,
    min_features: int = 3,
    random_state: int = 42,
) -> tuple[list[str], "RFECV"]:
    """Recursive feature elimination with chronological CV (gain/impurity)."""

    valid = X.dropna()
    y = y.loc[valid.index]
    if task == "classification":
        estimator = RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1,
            class_weight="balanced_subsample",
        )
        scoring = "f1_macro"
    else:
        estimator = RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1
        )
        scoring = "neg_root_mean_squared_error"
    splits = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(valid) // 20)))
    selector = RFECV(
        estimator,
        step=max(1, X.shape[1] // 20),
        cv=splits,
        scoring=scoring,
        min_features_to_select=min_features,
        n_jobs=-1,
    )
    selector.fit(valid.to_numpy(), y.to_numpy())
    selected = list(valid.columns[selector.support_])
    return selected, selector


@dataclass
class PhaseClassifierResult:
    model: object
    features: list[str]
    importances: pd.DataFrame
    cv_scores: pd.DataFrame
    classes: list[str]


def fit_phase_classifier(
    X: pd.DataFrame,
    phase_labels: pd.Series,
    *,
    model: str = "rf",
    n_splits: int = 5,
    random_state: int = 42,
) -> PhaseClassifierResult:
    """Train RF/XGB to classify the four cycle phases and score chronologically.

    Alem do F1 do modelo, cada fold reporta dois baselines obrigatorios do gate
    G2 (parecer 2026-07-10): ``semana_do_ano`` (classe modal da semana do ano no
    treino - captura o phase-locking sazonal do ENSO) e ``persistencia`` (rotulo
    da semana anterior). O ML so tem merito se superar ambos. Importante: os
    rotulos de fase sao definidos post hoc (plateau do pico e genese pre-onset
    conhecidos apos o evento); os F1 medem CARACTERIZACAO diagnostica, nao
    previsao operacional.
    """

    from sklearn.metrics import f1_score

    data = X.join(phase_labels.rename("__fase")).dropna()
    data = data[data["__fase"].isin(PHASE_ORDER)]
    if data.empty:
        raise ValueError("Sem semanas rotuladas nas 4 fases para treinar o classificador.")
    features = list(X.columns)
    y = data["__fase"].to_numpy()
    Xv = data[features].to_numpy()

    # Baseline de persistencia: rotulo observado na semana-calendario anterior
    # (eixo semanal completo; inicio de evento vindo de 'neutro' conta erro).
    prev_week = phase_labels.sort_index().shift(1)
    persist_pred = prev_week.reindex(data.index).fillna("__sem_rotulo").to_numpy()
    week_of_year = data.index.isocalendar().week.to_numpy()

    if model == "xgb":
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder

        encoder = LabelEncoder().fit(y)
        estimator = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=random_state,
            n_jobs=-1, objective="multi:softprob", eval_metric="mlogloss",
        )
        y_fit = encoder.transform(y)
    else:
        estimator = RandomForestClassifier(
            n_estimators=400, random_state=random_state, n_jobs=-1,
            class_weight="balanced_subsample",
        )
        encoder = None
        y_fit = y

    splits = TimeSeriesSplit(n_splits=n_splits)
    scores: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(splits.split(Xv), start=1):
        estimator.fit(Xv[train], y_fit[train])
        pred = estimator.predict(Xv[test])
        if encoder is not None:
            pred = encoder.inverse_transform(pred)
            truth = encoder.inverse_transform(y_fit[test])
        else:
            truth = y_fit[test]
        # Baseline sazonal: classe modal por semana-do-ano ajustada SO no treino.
        train_woy = pd.Series(y[train], index=week_of_year[train])
        modal_by_woy = train_woy.groupby(level=0).agg(lambda s: s.mode().iloc[0])
        global_mode = pd.Series(y[train]).mode().iloc[0]
        seasonal_pred = np.array(
            [modal_by_woy.get(w, global_mode) for w in week_of_year[test]], dtype=object
        )
        f1_kwargs = {"average": "macro", "labels": list(PHASE_ORDER), "zero_division": 0}
        scores.append(
            {"fold": fold, "n_teste": len(test),
             "f1_macro": float(f1_score(truth, pred, **f1_kwargs)),
             "f1_baseline_semana_do_ano": float(f1_score(truth, seasonal_pred, **f1_kwargs)),
             "f1_baseline_persistencia": float(f1_score(truth, persist_pred[test], **f1_kwargs))}
        )
    estimator.fit(Xv, y_fit)
    importance = getattr(estimator, "feature_importances_", np.zeros(len(features)))
    importances = (
        pd.DataFrame({"variavel": features, "importancia_ganho": importance})
        .sort_values("importancia_ganho", ascending=False)
        .reset_index(drop=True)
    )
    return PhaseClassifierResult(
        model=estimator,
        features=features,
        importances=importances,
        cv_scores=pd.DataFrame(scores),
        classes=list(PHASE_ORDER),
    )


@dataclass
class EventRegressionResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    importances: pd.DataFrame


def fit_event_regressions(
    onset_features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    model: str = "rf",
    target_columns: Sequence[str] = ("Y_pico", "Y_tempo_para_pico_sem", "Y_duracao_sem"),
    jitter_sigma: float = 0.05,
    jitter_copies: int = 2,
    random_state: int = 42,
) -> EventRegressionResult:
    """Regressao por evento (Y_pico, Y_tempo_para_pico, Y_duracao) com LOO.

    Completa a "arquitetura preditiva dupla" da Fase 5: cada evento retido e
    previsto por um modelo treinado nos demais (leave-one-event-out), com
    augmentation por jitter APENAS no treino (sem vazamento) e baseline de
    climatologia dos eventos de treino (media), no mesmo protocolo da Fase 3.
    Colunas com NaN em qualquer onset (lags que precedem o inicio da serie)
    sao descartadas e reportadas em ``metrics['n_features']``.
    """

    frame = onset_features.copy()
    meta = targets.set_index("event_id")
    frame = frame.loc[[e for e in frame.index if e in meta.index]]
    frame["tipo_el_nino"] = (meta.loc[frame.index, "tipo"] == "el_nino").astype(float)
    feature_cols = [c for c in frame.columns if frame[c].notna().all()]
    X_all = frame[feature_cols].to_numpy(dtype=float)
    event_ids = np.asarray(frame.index)
    folds = leave_one_event_out_indices(event_ids)

    metric_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    for target in target_columns:
        if target not in meta.columns:
            continue
        y_all = pd.to_numeric(meta.loc[frame.index, target], errors="coerce").to_numpy()
        ok = np.isfinite(y_all)
        pred = np.full(len(y_all), np.nan)
        clim = np.full(len(y_all), np.nan)
        estimator = None
        for train, test in folds:
            train = train[ok[train]]
            if len(train) < 3 or not ok[test].any():
                continue
            X_tr = pd.DataFrame(X_all[train], columns=feature_cols)
            y_tr = pd.Series(y_all[train])
            if jitter_copies > 0:
                X_tr, y_tr = gaussian_jitter(
                    X_tr, y_tr, sigma=jitter_sigma, n_copies=jitter_copies,
                    random_state=random_state,
                )
            if model == "xgb":
                from xgboost import XGBRegressor

                estimator = XGBRegressor(
                    n_estimators=400, max_depth=3, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    objective="reg:squarederror", random_state=random_state, n_jobs=-1,
                )
            else:
                estimator = RandomForestRegressor(
                    n_estimators=500, max_features="sqrt", min_samples_leaf=2,
                    random_state=random_state, n_jobs=-1,
                )
            estimator.fit(X_tr.to_numpy(), y_tr.to_numpy())
            pred[test] = estimator.predict(X_all[test])
            clim[test] = float(np.mean(y_all[train]))
        valid = ok & np.isfinite(pred) & np.isfinite(clim)
        obs, prd, base = y_all[valid], pred[valid], clim[valid]
        if len(obs) >= 3:
            mae = float(np.mean(np.abs(prd - obs)))
            mae_clim = float(np.mean(np.abs(base - obs)))
            r = float(np.corrcoef(obs, prd)[0, 1]) if prd.std() > 0 else float("nan")
            metric_rows.append({
                "alvo": target, "modelo": model, "n_eventos": int(valid.sum()),
                "n_features": len(feature_cols),
                "r_loo": round(r, 3),
                "mae_loo": round(mae, 3),
                "rmse_loo": round(float(np.sqrt(np.mean((prd - obs) ** 2))), 3),
                "mae_climatologia": round(mae_clim, 3),
                "skill_vs_climatologia": round(1.0 - mae / mae_clim, 3) if mae_clim > 0 else float("nan"),
                "protocolo": (
                    "leave-one-event-out; jitter so no treino; baseline=media dos eventos de treino"
                ),
            })
        for event, o, p_, c_ in zip(event_ids, y_all, pred, clim):
            pred_rows.append({
                "event_id": event, "alvo": target, "observado": o,
                "previsto_loo": p_, "climatologia_treino": c_,
            })
        # Importancias em ajuste final com todos os eventos (descritivo).
        if estimator is not None and ok.any():
            estimator.fit(X_all[ok], y_all[ok])
            gains = getattr(estimator, "feature_importances_", np.zeros(len(feature_cols)))
            for feature, gain in zip(feature_cols, gains):
                importance_rows.append({
                    "alvo": target, "variavel": feature, "importancia_ganho": float(gain),
                })
    return EventRegressionResult(
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.DataFrame(pred_rows),
        importances=pd.DataFrame(importance_rows),
    )


def shap_summary_values(model, X: pd.DataFrame, *, max_samples: int = 2000):
    """Return SHAP values for tree models, when the shap package is installed."""

    try:
        import shap  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("shap nao instalado; 'pip install shap' para XAI.") from exc
    sample = X.dropna()
    if len(sample) > max_samples:
        sample = sample.sample(max_samples, random_state=42)
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(sample), sample


def partial_dependence_frame(
    model,
    X: pd.DataFrame,
    features: Sequence[str],
    *,
    grid_resolution: int = 30,
) -> pd.DataFrame:
    """Tabulate 1-D partial dependence to map non-linear Bjerknes thresholds."""

    from sklearn.inspection import partial_dependence

    valid = X.dropna()
    rows: list[dict[str, object]] = []
    for feature in features:
        if feature not in valid.columns:
            continue
        result = partial_dependence(
            model, valid, [list(valid.columns).index(feature)],
            grid_resolution=grid_resolution, kind="average",
        )
        grid = result["grid_values"][0]
        average = np.asarray(result["average"])
        for series in average.reshape(average.shape[0], -1):
            for value, response in zip(grid, series):
                rows.append({"variavel": feature, "valor": float(value), "resposta_pdp": float(response)})
    return pd.DataFrame(rows)
