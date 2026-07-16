"""Fase 5 - projecao auditavel do ciclo ENSO com RF e XGBoost.

Mesmo mecanismo da Fase 3 (genese -> crescimento -> pico -> decaimento), agora
com Machine Learning. Duas frentes preditivas:

* **Classificacao semanal de fase** do ciclo (4 classes) a partir da matriz
  multivariada transformada por janela deslizante (lags de 4 a 52 semanas). Serve
  para mapear as quatro fases e ranquear variaveis por importancia de
  permutacao fora da amostra, agrupada por variavel fisica.
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
- **Validacao oficial**: :func:`fit_event_aware_phase_classifier` usa
  rolling-origin por evento inteiro, embargo, nove estados e inicia os testes
  somente depois de haver suporte independente minimo de EN e LN no treino.
  Rotinas com ``TimeSeriesSplit`` e RFECV permanecem apenas para reproduzir
  pilotos legados; nao pertencem ao caminho oficial.
- **Desbalanceamento**: pesos tipo/evento/fase e augmentation conservador,
  exclusivamente no treino e com linhagem do evento original.
- **Interpretacao oficial**: permutacao OOS repetida, sem usar o conjunto de
  teste para selecionar features. SHAP/PDP sao utilitarios exploratorios
  legados e nao sustentam os gates oficiais.

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

from nino_brasil.models.event_validation import (
    ENSO_STATES,
    AugmentedTrainingData,
    assert_continuous_weekly_index,
    augment_training_rows,
    balanced_active_event_test_start,
    canonical_event_ids,
    canonical_state_labels,
    event_phase_sample_weights,
    independent_fold_support,
    make_event_folds,
    parse_state,
)

PHASE_ORDER: tuple[str, ...] = ("genese", "crescimento", "pico", "decaimento")
STATE_ORDER: tuple[str, ...] = ENSO_STATES
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

MASTER_METADATA_COLUMNS: frozenset[str] = frozenset(
    {
        "week_ending_sunday",
        "time",
        "date",
        "ocean_source_code",
        "ocean_source",
        "run_id",
        "source_seam_flag",
    }
)


def physical_predictor_columns(master: pd.DataFrame, *, expected: int | None = None) -> list[str]:
    """Return physical master variables, excluding dates and provenance columns.

    When ``expected`` is omitted, the current named Phase 2 contract is loaded
    dynamically; reduced sensitivity experiments may pass another value.
    """

    columns: list[str] = []
    for name in master.columns:
        lower = str(name).lower()
        if lower in MASTER_METADATA_COLUMNS or lower.endswith("_source_code"):
            continue
        if pd.api.types.is_numeric_dtype(master[name]):
            columns.append(str(name))
    if expected is None:
        from nino_brasil.data.phase2_master import PHYSICAL_COLUMNS

        expected = len(PHYSICAL_COLUMNS)
    if len(columns) != int(expected):
        raise ValueError(
            f"Contrato F2/F5 exige {expected} variaveis fisicas; encontrei {len(columns)}: {columns}"
        )
    return columns


@dataclass
class FoldHarmonicPreprocessor:
    """Harmonic/detrend coefficients and imputation fitted on training dates only."""

    columns: tuple[str, ...]
    harmonics: int = 3
    detrend: bool = True
    coefficients: dict[str, np.ndarray] = field(default_factory=dict)
    source_coefficients: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    fitted_source_families: tuple[str, ...] = ()
    medians: pd.Series | None = None
    origin: pd.Timestamp | None = None

    def _design(self, index: pd.DatetimeIndex) -> np.ndarray:
        angle = 2.0 * np.pi * (index.dayofyear.to_numpy(dtype=float) - 1.0) / 365.2425
        parts: list[np.ndarray] = [np.ones(len(index), dtype=float)]
        for harmonic in range(1, self.harmonics + 1):
            parts.extend([np.sin(harmonic * angle), np.cos(harmonic * angle)])
        if self.detrend:
            if self.origin is None:
                raise RuntimeError("Preprocessor ainda nao ajustado.")
            parts.append((index - self.origin).days.to_numpy(dtype=float) / 365.2425)
        return np.column_stack(parts)

    @staticmethod
    def _source_families(source_codes: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        families = pd.to_numeric(source_codes.reindex(index), errors="coerce").map(
            {1: "ufs", 2: "glorys", 3: "glorys"}
        )
        return families.astype("string")

    def fit(
        self,
        frame: pd.DataFrame,
        source_codes: pd.Series | None = None,
    ) -> "FoldHarmonicPreprocessor":
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.empty:
            raise ValueError("fit exige DataFrame temporal de treino nao vazio.")
        self.origin = pd.Timestamp(frame.index.min())
        design = self._design(frame.index)
        self.coefficients = {}
        self.source_coefficients = {}
        source_family = (
            self._source_families(source_codes, frame.index)
            if source_codes is not None
            else None
        )
        for name in self.columns:
            if name not in frame or name not in OCEAN_RAW_VARS:
                continue
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            if source_family is not None:
                for family in source_family.dropna().unique():
                    family_mask = (
                        source_family.eq(family).fillna(False).to_numpy(dtype=bool)
                    )
                    valid = family_mask & np.isfinite(values)
                    if int(valid.sum()) < design.shape[1] + 2:
                        continue
                    coef, *_ = np.linalg.lstsq(design[valid], values[valid], rcond=None)
                    self.source_coefficients[(str(family), name)] = coef
                continue
            valid = np.isfinite(values)
            if int(valid.sum()) < design.shape[1] + 2:
                raise ValueError(f"{name}: treino insuficiente para harmonicos+detrend.")
            self.coefficients[name], *_ = np.linalg.lstsq(design[valid], values[valid], rcond=None)
        self.fitted_source_families = tuple(
            sorted({family for family, _ in self.source_coefficients})
        )
        transformed = self._transform_without_impute(frame, source_codes=source_codes)
        self.medians = transformed.loc[:, list(self.columns)].median(numeric_only=True)
        return self

    def _transform_without_impute(
        self,
        frame: pd.DataFrame,
        *,
        source_codes: pd.Series | None = None,
    ) -> pd.DataFrame:
        if self.origin is None:
            raise RuntimeError("Chame fit antes de transform.")
        out = frame.loc[:, [c for c in self.columns if c in frame]].apply(
            pd.to_numeric, errors="coerce"
        )
        design = self._design(frame.index)
        if self.source_coefficients:
            if source_codes is None:
                raise ValueError("source_codes e obrigatorio para um preprocessor source-aware.")
            source_family = self._source_families(source_codes, frame.index)
            for name in (column for column in self.columns if column in OCEAN_RAW_VARS):
                adjusted = np.full(len(frame), np.nan, dtype=float)
                values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
                for family in source_family.dropna().unique():
                    coefficient = self.source_coefficients.get((str(family), name))
                    if coefficient is None:
                        continue
                    mask = source_family.eq(family).fillna(False).to_numpy(dtype=bool)
                    adjusted[mask] = values[mask] - design[mask] @ coefficient
                out[name] = adjusted
        for name, coef in self.coefficients.items():
            out[name] = out[name].to_numpy(dtype=float) - design @ coef
        return out

    def transform(
        self,
        frame: pd.DataFrame,
        source_codes: pd.Series | None = None,
    ) -> pd.DataFrame:
        out = self._transform_without_impute(frame, source_codes=source_codes)
        if self.medians is None:
            raise RuntimeError("Chame fit antes de transform.")
        filled = out.fillna(self.medians)
        if self.source_coefficients:
            assert source_codes is not None
            families = self._source_families(source_codes, frame.index)
            unknown = ~families.isin(self.fitted_source_families).fillna(False)
            ocean = [name for name in self.columns if name in OCEAN_RAW_VARS]
            filled.loc[unknown, ocean] = np.nan
        return filled

    def fit_transform(
        self,
        frame: pd.DataFrame,
        source_codes: pd.Series | None = None,
    ) -> pd.DataFrame:
        return self.fit(frame, source_codes=source_codes).transform(
            frame, source_codes=source_codes
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


def build_rolling_origin_table(
    master: pd.DataFrame,
    phase_table: pd.DataFrame,
    *,
    predictors: Sequence[str] | None = None,
    horizons: Sequence[int] = (1, 4, 8, 12, 24),
    ssta_column: str = "nino34_ssta",
) -> pd.DataFrame:
    """Build a genuinely predictive origin-time table.

    Every row contains only predictors observed at origin ``t``.  Future state
    and SSTA columns are explicit targets at ``t + h``.  This is distinct from
    retrospective peak-aligned diagnostics, whose peak date is known post hoc.
    """

    if not isinstance(master.index, pd.DatetimeIndex):
        raise TypeError("master deve usar DatetimeIndex.")
    common = master.index.intersection(phase_table.index).sort_values()
    assert_continuous_weekly_index(common, name="F5 common weekly index")
    if predictors is None:
        predictors = physical_predictor_columns(master)
    out = master.reindex(common).loc[:, list(predictors)].copy()
    out.insert(0, "origin_time", common)
    states = canonical_state_labels(phase_table.reindex(common))
    for horizon in sorted({int(h) for h in horizons}):
        if horizon < 1:
            raise ValueError("Horizontes preditivos precisam ser >= 1 semana.")
        out[f"target_time_h{horizon:02d}"] = common + pd.Timedelta(weeks=horizon)
        out[f"target_state_h{horizon:02d}"] = states.shift(-horizon).to_numpy()
        if ssta_column in master:
            out[f"target_ssta_h{horizon:02d}"] = (
                pd.to_numeric(master.reindex(common)[ssta_column], errors="coerce")
                .shift(-horizon)
                .to_numpy()
            )
        if "event_id" in phase_table:
            out[f"target_event_id_h{horizon:02d}"] = (
                phase_table.reindex(common)["event_id"].shift(-horizon).fillna("").to_numpy()
            )
    out["dataset_role"] = "rolling_origin_predictive"
    return out.reset_index(drop=True)


@dataclass
class EventAwareClassifierResult:
    """Out-of-sample products of the official nine-state F5 experiment."""

    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    importances: pd.DataFrame
    state_importances: pd.DataFrame
    augmentation_provenance: pd.DataFrame
    fold_contract: pd.DataFrame
    independent_support: pd.DataFrame
    predictors: list[str]
    feature_names: list[str]


@dataclass(frozen=True)
class FoldStateEncoder:
    """Contiguous, training-fold-only encoder for the canonical ENSO states.

    XGBoost requires labels in every fit to be contiguous from zero.  A global
    nine-state integer mapping violates that contract whenever an early rolling
    fold has not observed one or more states.  This encoder is therefore fitted
    exclusively on the training labels and expands probabilities back to the
    stable nine-column project contract after prediction.
    """

    classes: tuple[str, ...]

    @classmethod
    def fit(cls, labels: Sequence[object] | pd.Series) -> "FoldStateEncoder":
        observed = {str(value) for value in labels}
        unknown = observed.difference(STATE_ORDER)
        if unknown:
            raise ValueError(f"Estados fora do contrato canonico: {sorted(unknown)}")
        classes = tuple(state for state in STATE_ORDER if state in observed)
        if len(classes) < 2:
            raise ValueError("O fold de treino precisa conter ao menos dois estados ENSO.")
        return cls(classes=classes)

    def transform(self, labels: Sequence[object] | pd.Series) -> np.ndarray:
        mapping = {state: class_id for class_id, state in enumerate(self.classes)}
        encoded: list[int] = []
        for value in labels:
            state = str(value)
            if state not in mapping:
                raise ValueError(f"Estado {state!r} nao observado no treino do fold.")
            encoded.append(mapping[state])
        return np.asarray(encoded, dtype=int)

    def inverse_transform(self, encoded: Sequence[int] | np.ndarray) -> np.ndarray:
        values = np.asarray(encoded, dtype=int)
        if ((values < 0) | (values >= len(self.classes))).any():
            raise ValueError("ID local de classe fora do encoder do fold.")
        return np.asarray([self.classes[value] for value in values], dtype=object)

    def expand_probabilities(
        self,
        probability: np.ndarray,
        estimator_classes: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        """Map local estimator columns to the stable nine-state column order."""

        raw = np.asarray(probability, dtype=float)
        estimator_ids = np.asarray(estimator_classes, dtype=int)
        if raw.ndim != 2 or raw.shape[1] != len(estimator_ids):
            raise ValueError("Matriz de probabilidades incompativel com estimator.classes_.")
        expanded = np.zeros((raw.shape[0], len(STATE_ORDER)), dtype=float)
        global_id = {state: state_id for state_id, state in enumerate(STATE_ORDER)}
        for column, local_id in enumerate(estimator_ids):
            if local_id < 0 or local_id >= len(self.classes):
                raise ValueError("Classe retornada pelo estimador nao pertence ao encoder local.")
            expanded[:, global_id[self.classes[local_id]]] = raw[:, column]
        row_sums = expanded.sum(axis=1)
        if (
            not np.isfinite(row_sums).all()
            or (row_sums <= 0.0).any()
            or not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-6)
        ):
            raise ValueError("Probabilidades remapeadas nao somam um.")
        # XGBoost returns float32 probabilities whose row sum can differ from
        # one by a few ulps.  Normalize in float64 so sklearn.log_loss receives
        # a proper probability simplex without warnings or implicit repair.
        expanded /= row_sums[:, None]
        return expanded


def paired_state_permutation_deltas(
    estimator: object,
    fold_encoder: FoldStateEncoder,
    X_reference: np.ndarray,
    probability: np.ndarray,
    truth_ids: np.ndarray,
    predictor_positions: dict[str, np.ndarray],
    predictors: Sequence[str],
    *,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return paired OOS Brier deltas for all states with one prediction per repeat.

    A perturbed predictor matrix is shared by every state because one
    ``predict_proba`` call already returns the complete probability vector.
    Output arrays have shape ``(repeats, len(STATE_ORDER))``; an unavailable
    predictor receives an empty first dimension.
    """

    if repeats < 1:
        raise ValueError("repeats deve ser >= 1.")
    reference = np.asarray(X_reference, dtype=float)
    base_probability = np.asarray(probability, dtype=float)
    truth = np.asarray(truth_ids, dtype=int)
    if reference.ndim != 2 or base_probability.shape != (
        len(reference),
        len(STATE_ORDER),
    ):
        raise ValueError("Matrizes OOS incompativeis com o contrato de nove estados.")
    if truth.shape != (len(reference),) or (
        (truth < 0) | (truth >= len(STATE_ORDER))
    ).any():
        raise ValueError("truth_ids fora do contrato canonico de estados.")

    indicators = (
        truth[:, None] == np.arange(len(STATE_ORDER), dtype=int)[None, :]
    ).astype(float)
    state_counts = indicators.sum(axis=0).astype(int)
    base_brier = np.mean((base_probability - indicators) ** 2, axis=0)
    estimator_classes = np.asarray(getattr(estimator, "classes_"), dtype=int)
    deltas_by_predictor: dict[str, np.ndarray] = {}
    for predictor in predictors:
        positions = np.asarray(
            predictor_positions.get(str(predictor), np.asarray([], dtype=int)),
            dtype=int,
        )
        if ((positions < 0) | (positions >= reference.shape[1])).any():
            raise ValueError(f"Posicao de feature invalida para {predictor!r}.")
        repeat_deltas: list[np.ndarray] = []
        if len(positions):
            for _ in range(int(repeats)):
                permutation = rng.permutation(len(reference))
                permuted = reference.copy()
                permuted[:, positions] = reference[permutation][:, positions]
                changed_raw = estimator.predict_proba(permuted)
                changed = fold_encoder.expand_probabilities(
                    changed_raw, estimator_classes
                )
                repeat_deltas.append(
                    np.mean((changed - indicators) ** 2, axis=0) - base_brier
                )
        deltas_by_predictor[str(predictor)] = (
            np.vstack(repeat_deltas)
            if repeat_deltas
            else np.empty((0, len(STATE_ORDER)), dtype=float)
        )
    return state_counts, deltas_by_predictor


def configure_serial_prediction(estimator: object) -> object:
    """Keep parallel fitting but make small OOS prediction batches serial.

    The project environment's sklearn/joblib combination emits one warning per
    tree when nested parallel prediction is used on tiny event folds. Changing
    ``n_jobs`` after fitting does not alter the fitted ensemble; it only avoids
    process/thread overhead and keeps official logs compact and deterministic.
    """

    get_params = getattr(estimator, "get_params", None)
    set_params = getattr(estimator, "set_params", None)
    if callable(get_params) and callable(set_params):
        parameters = get_params(deep=False)
        if "n_jobs" in parameters:
            set_params(n_jobs=1)
    return estimator


def valid_phase_target_mask(phase_table: pd.DataFrame) -> pd.Series:
    """Keep valid neutral weeks while requiring ``event_id`` only when active.

    Missing ``tipo``/``fase`` marks unavailable future rows created by shifting
    a forecast horizon.  In contrast, a neutral observation legitimately has no
    event id and remains a member of the nine-state experiment.
    """

    required = {"tipo", "fase", "event_id"}
    if missing := required.difference(phase_table.columns):
        raise KeyError(f"phase_table sem colunas obrigatorias: {sorted(missing)}")
    valid = phase_table["tipo"].notna() & phase_table["fase"].notna()
    if valid.any():
        available = phase_table.loc[valid]
        labels = canonical_state_labels(available)
        event_ids = available["event_id"].fillna("").astype(str).str.strip()
        missing_active_id = labels.ne("neutro") & event_ids.eq("")
        if missing_active_id.any():
            examples = available.index[missing_active_id][:5].astype(str).tolist()
            raise ValueError(
                "event_id e obrigatorio somente nos estados ENSO ativos; "
                f"faltou em {examples}"
            )
    return valid.rename("valid_phase_target")


def _phase_estimator(model: str, *, random_state: int, n_estimators: int):
    if model == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise ImportError("XGBoost nao instalado; use model='rf' ou instale xgboost.") from exc
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
    if model != "rf":
        raise ValueError("model deve ser 'rf' ou 'xgb'.")
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight=None,
        random_state=random_state,
        n_jobs=-1,
    )


def fit_event_aware_phase_classifier(
    master: pd.DataFrame,
    phase_table: pd.DataFrame,
    *,
    predictors: Sequence[str] | None = None,
    lags: Sequence[int] = tuple(range(4, 53, 4)),
    horizon_weeks: int = 0,
    model: str = "rf",
    n_splits: int = 5,
    min_train_groups: int = 8,
    purge_weeks: int | None = None,
    n_estimators: int = 300,
    augmentation_noise_copies: int = 0,
    augmentation_noise_scale: float = 0.02,
    augmentation_mixup_alpha: float | None = None,
    min_train_active_events_per_type: int = 3,
    enforce_support_at_fold_design: bool = True,
    state_importance_repeats: int = 3,
    random_state: int = 42,
) -> EventAwareClassifierResult:
    """Fit RF/XGB using all 31 variables and event-independent evaluation.

    Seasonal-cycle/trend coefficients and imputers are fitted separately inside
    each fold using historical data only.  The validation target has nine states
    (neutral + EN/LN x four phases).  Optional augmentation is applied only to
    training rows and never changes ``n_test_events``.
    """

    from sklearn.metrics import f1_score, log_loss

    if not isinstance(master.index, pd.DatetimeIndex) or not isinstance(
        phase_table.index, pd.DatetimeIndex
    ):
        raise TypeError("master e phase_table devem usar DatetimeIndex.")
    if horizon_weeks < 0:
        raise ValueError("horizon_weeks deve ser >= 0.")
    if state_importance_repeats < 1:
        raise ValueError("state_importance_repeats deve ser >= 1.")
    predictors = list(predictors or physical_predictor_columns(master))
    missing = sorted(set(predictors).difference(master.columns))
    if missing:
        raise KeyError(f"Preditores ausentes do master: {missing}")
    common = master.index.intersection(phase_table.index).sort_values()
    master = master.reindex(common)
    phase_table = phase_table.reindex(common)
    target_phase = phase_table.shift(-int(horizon_weeks))
    valid_target = valid_phase_target_mask(target_phase)
    common = common[valid_target]
    master = master.reindex(common)
    origin_phase = phase_table.reindex(common)
    target_phase = target_phase.reindex(common)
    y = canonical_state_labels(target_phase)
    groups = canonical_event_ids(target_phase)
    purge = int(purge_weeks if purge_weeks is not None else max(lags, default=0) + horizon_weeks)
    earliest_test_start = balanced_active_event_test_start(
        target_phase,
        min_train_active_events_per_type=(
            min_train_active_events_per_type
            if enforce_support_at_fold_design
            else 0
        ),
        purge_weeks=purge,
    )
    folds = make_event_folds(
        common,
        groups,
        n_splits=n_splits,
        min_train_groups=min_train_groups,
        purge_weeks=purge,
        earliest_test_start=earliest_test_start,
    )
    state_to_id = {state: state_id for state_id, state in enumerate(STATE_ORDER)}
    origin_states = canonical_state_labels(origin_phase)

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    state_importance_rows: list[dict[str, object]] = []
    provenance_rows: list[pd.DataFrame] = []
    contract_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    feature_names: list[str] = []

    for fold_no, fold in enumerate(folds, start=1):
        fit_history = master.loc[: fold.train_end, predictors]
        source_codes = master.get("ocean_source_code")
        preprocessor = FoldHarmonicPreprocessor(tuple(predictors)).fit(
            fit_history,
            source_codes=(source_codes.loc[fit_history.index] if source_codes is not None else None),
        )
        prepared = preprocessor.transform(master[predictors], source_codes=source_codes)
        lagged = build_lagged_features(prepared, predictors, lags=lags)
        train_times = common[fold.train_index]
        test_times = common[fold.test_index]
        X_train = lagged.reindex(train_times)
        X_test = lagged.reindex(test_times)
        usable = X_train.notna().any(axis=0)
        X_train = X_train.loc[:, usable]
        X_test = X_test.loc[:, usable]
        ocean_features = [
            column
            for column in X_train.columns
            if any(column.startswith(f"{name}_lag") for name in OCEAN_RAW_VARS)
        ]
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
        finite_train = np.isfinite(X_train.to_numpy()).all(axis=1) & train_source_valid.to_numpy()
        finite_test = np.isfinite(X_test.to_numpy()).all(axis=1) & test_source_valid.to_numpy()
        X_train = X_train.iloc[np.flatnonzero(finite_train)]
        X_test = X_test.iloc[np.flatnonzero(finite_test)]
        if X_train.empty or X_test.empty:
            continue
        feature_names = list(X_train.columns)
        y_train_text = y.reindex(X_train.index)
        y_test_text = y.reindex(X_test.index)
        train_phase = target_phase.reindex(X_train.index)
        test_phase = target_phase.reindex(X_test.index)
        support = independent_fold_support(
            train_phase,
            test_phase,
            fold=fold.name,
            min_train_active_events_per_type=min_train_active_events_per_type,
        )
        support_rows.append(support)
        augmented: AugmentedTrainingData = augment_training_rows(
            X_train,
            y_train_text,
            train_phase,
            n_noise_copies=augmentation_noise_copies,
            noise_scale=augmentation_noise_scale,
            mixup_alpha=augmentation_mixup_alpha,
            random_state=random_state + fold_no,
        )
        augmented.provenance.insert(0, "fold", fold.name)
        provenance_rows.append(augmented.provenance)
        estimator = _phase_estimator(
            model, random_state=random_state + fold_no, n_estimators=n_estimators
        )
        fold_encoder = FoldStateEncoder.fit(augmented.y.astype(str))
        y_train_encoded = fold_encoder.transform(augmented.y.astype(str))
        estimator.fit(
            augmented.X.to_numpy(dtype=float),
            y_train_encoded,
            sample_weight=augmented.sample_weight.to_numpy(dtype=float),
        )
        configure_serial_prediction(estimator)
        pred_encoded = estimator.predict(X_test.to_numpy(dtype=float)).astype(int)
        pred_text = fold_encoder.inverse_transform(pred_encoded)
        truth_text = y_test_text.to_numpy(dtype=object)
        probability_raw = estimator.predict_proba(X_test.to_numpy(dtype=float))
        probability = fold_encoder.expand_probabilities(
            probability_raw, np.asarray(estimator.classes_, dtype=int)
        )

        # Strong baselines fitted without looking at test outcomes.
        target_week = (X_train.index + pd.Timedelta(weeks=horizon_weeks)).isocalendar().week
        modal_by_week = pd.Series(y_train_text.to_numpy(), index=target_week).groupby(level=0).agg(
            lambda values: values.mode().iloc[0]
        )
        global_mode = y_train_text.mode().iloc[0]
        test_week = (X_test.index + pd.Timedelta(weeks=horizon_weeks)).isocalendar().week
        seasonal_pred = np.array([modal_by_week.get(w, global_mode) for w in test_week], dtype=object)
        if horizon_weeks:
            persistence_pred = origin_states.reindex(X_test.index).to_numpy(dtype=object)
        else:
            persistence_pred = y.shift(1).reindex(X_test.index).fillna("neutro").to_numpy(dtype=object)
        labels_for_score = list(STATE_ORDER)
        f1 = float(
            f1_score(truth_text, pred_text, labels=labels_for_score, average="macro", zero_division=0)
        )
        f1_seasonal = float(
            f1_score(
                truth_text, seasonal_pred, labels=labels_for_score, average="macro", zero_division=0
            )
        )
        f1_persistence = float(
            f1_score(
                truth_text,
                persistence_pred,
                labels=labels_for_score,
                average="macro",
                zero_division=0,
            )
        )
        present_truth = np.asarray([state_to_id[str(value)] for value in truth_text], dtype=int)
        metric_rows.append(
            {
                "fold": fold.name,
                "model": model,
                "horizon_weeks": horizon_weeks,
                "n_train_rows_original": len(X_train),
                "n_train_rows_optimisation": len(augmented.X),
                "n_test_rows": len(X_test),
                "n_train_events": target_phase.reindex(X_train.index)["event_id"].replace("", np.nan).nunique(),
                "n_test_events": target_phase.reindex(X_test.index)["event_id"].replace("", np.nan).nunique(),
                "f1_macro_9state": f1,
                "f1_baseline_seasonal": f1_seasonal,
                "f1_baseline_persistence": f1_persistence,
                "skill_f1_vs_best_baseline": f1 - max(f1_seasonal, f1_persistence),
                "log_loss": float(log_loss(present_truth, probability, labels=np.arange(len(STATE_ORDER)))),
                "gate_pass": bool(f1 > max(f1_seasonal, f1_persistence)),
            }
        )
        target_events = target_phase.reindex(X_test.index)["event_id"].fillna("")
        for row_no, origin in enumerate(X_test.index):
            prediction_rows.append(
                {
                    "fold": fold.name,
                    "origin_time": origin,
                    "target_time": origin + pd.Timedelta(weeks=horizon_weeks),
                    "event_id": target_events.loc[origin],
                    "observed_state": truth_text[row_no],
                    "predicted_state": pred_text[row_no],
                    "is_original_observation": True,
                    **{
                        f"prob_{state}": float(probability[row_no, state_no])
                        for state_no, state in enumerate(STATE_ORDER)
                    },
                }
            )

        global_importance = getattr(estimator, "feature_importances_", np.zeros(len(feature_names)))
        for feature, value in zip(feature_names, global_importance):
            importance_rows.append(
                {
                    "fold": fold.name,
                    "feature": feature,
                    "importance_gain": float(value),
                    "importance_scope": "global_training_fit",
                }
            )
        rng = np.random.default_rng(random_state + 10_000 + fold_no)
        X_reference = X_test.to_numpy(dtype=float)
        truth_ids = np.asarray([state_to_id[str(value)] for value in truth_text], dtype=int)
        predictor_positions = {
            predictor: np.asarray(
                [
                    position
                    for position, feature in enumerate(feature_names)
                    if feature.startswith(f"{predictor}_lag")
                ],
                dtype=int,
            )
            for predictor in predictors
        }
        # One predict_proba call returns probabilities for every canonical
        # state.  Reuse each predictor-level permutation across all nine states
        # instead of recomputing the same perturbed matrix once per state.  The
        # shared permutation is also a common-random-number design: state
        # importances within a fold/repeat are directly paired and have less
        # Monte-Carlo noise, while all 31 physical variables and all requested
        # repetitions remain in the official experiment.
        state_counts, predictor_deltas = paired_state_permutation_deltas(
            estimator,
            fold_encoder,
            X_reference,
            probability,
            truth_ids,
            predictor_positions,
            predictors,
            repeats=int(state_importance_repeats),
            rng=rng,
        )

        # Preserve the stable state-major output order used by prior runs.
        for state_id, state in enumerate(STATE_ORDER):
            n_test_state = int(state_counts[state_id])
            for predictor in predictors:
                positions = predictor_positions[predictor]
                deltas = (
                    predictor_deltas[predictor][:, state_id]
                    if n_test_state and len(positions)
                    else np.asarray([], dtype=float)
                )
                status = (
                    "estimated"
                    if len(deltas)
                    else ("state_absent_in_test" if not n_test_state else "predictor_unavailable")
                )
                state_importance_rows.append(
                    {
                        "fold": fold.name,
                        "state": state,
                        "event_type": parse_state(state)[0],
                        "phase": parse_state(state)[1],
                        "predictor": predictor,
                        "delta_brier_permutation_oos": (
                            float(np.mean(deltas)) if len(deltas) else np.nan
                        ),
                        "delta_brier_permutation_oos_sd": (
                            float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
                            if len(deltas)
                            else np.nan
                        ),
                        "n_permutation_repeats": len(deltas),
                        "n_lag_features_permuted": len(positions),
                        "n_test_state": n_test_state,
                        "importance_scope": "oos_grouped_all_lags_by_physical_predictor",
                        "permutation_pairing": "shared_predictor_repeat_across_states",
                        "estimation_status": status,
                    }
                )
        contract_rows.append(
            {
                "fold": fold.name,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "purge_weeks": fold.purge_weeks,
                "n_train_groups": len(fold.train_groups),
                "n_test_groups": len(fold.test_groups),
                "test_groups": ";".join(fold.test_groups),
                "preprocessing_fit_end": fold.train_end,
                "augmentation_train_only": True,
                "support_balanced_fold_start": bool(enforce_support_at_fold_design),
                "earliest_admissible_test_start": earliest_test_start,
                "n_train_el_nino_events": support["n_train_el_nino_events"],
                "n_train_la_nina_events": support["n_train_la_nina_events"],
                "n_train_neutral_blocks": support["n_train_neutral_blocks"],
                "min_train_active_events_per_type_required": support[
                    "min_train_active_events_per_type_required"
                ],
                "independent_support_gate_pass": support[
                    "independent_support_gate_pass"
                ],
                "fold_encoder_classes": ";".join(fold_encoder.classes),
                "probability_contract_classes": ";".join(STATE_ORDER),
                "state_importance_all_predictors": len(predictors),
                "state_importance_repeats": int(state_importance_repeats),
                "state_importance_permutation_pairing": (
                    "shared_predictor_repeat_across_states"
                ),
                "prediction_n_jobs": 1,
            }
        )

    if not metric_rows:
        raise ValueError("Nenhum fold F5 produziu treino/teste validos.")
    return EventAwareClassifierResult(
        fold_metrics=pd.DataFrame(metric_rows),
        predictions=pd.DataFrame(prediction_rows),
        importances=pd.DataFrame(importance_rows),
        state_importances=pd.DataFrame(state_importance_rows),
        augmentation_provenance=(
            pd.concat(provenance_rows, ignore_index=True) if provenance_rows else pd.DataFrame()
        ),
        fold_contract=pd.DataFrame(contract_rows),
        independent_support=pd.DataFrame(support_rows),
        predictors=predictors,
        feature_names=feature_names,
    )


def build_event_targets(events: pd.DataFrame) -> pd.DataFrame:
    """Per-event targets from the isolated F3Nino/F3Nina modeling bridge."""

    required = {"event_id", "tipo", "onset", "pico", "oni_pico_c"}
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
    elif "duracao_meses" in out:
        out["Y_duracao_sem"] = (
            pd.to_numeric(out["duracao_meses"], errors="coerce") * WEEKS_PER_MONTH
        )
    elif "duracao_estacoes" in out:  # legado: estacoes sobrepostas avancam mensalmente
        out["Y_duracao_sem"] = (
            pd.to_numeric(out["duracao_estacoes"], errors="coerce") * WEEKS_PER_MONTH
        )
    else:
        raise KeyError("events precisa de fim, duracao_meses ou duracao_estacoes.")
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
        # Use the last completed W-SUN observation at or before onset.  Nearest
        # could select a Sunday after onset and leak future conditions.
        position = index.get_indexer([onset], method="pad")[0]
        if position < 0:
            continue
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
    """Legacy four-phase diagnostic retained for historical reproducibility.

    Official F5 uses :func:`fit_event_aware_phase_classifier`. Além do F1 do
    modelo legado, cada fold reporta dois baselines obrigatorios do gate
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
    augmentation_provenance: pd.DataFrame = field(default_factory=pd.DataFrame)


def fit_event_dimension_rolling_origin(
    master: pd.DataFrame,
    events: pd.DataFrame,
    *,
    predictors: Sequence[str] | None = None,
    lags: Sequence[int] = tuple(range(4, 53, 4)),
    model: str = "rf",
    n_splits: int = 5,
    min_train_events: int = 6,
    n_estimators: int = 300,
    jitter_sigma: float = 0.02,
    jitter_copies: int = 1,
    random_state: int = 42,
) -> EventRegressionResult:
    """Predict event peak, time-to-peak and duration with expanding event folds.

    Each event contributes one onset-origin row.  Harmonic/source preprocessing,
    imputation, event balancing and jitter are fitted only on earlier training
    events.  The test event is never used to set weights or preprocessing.
    """

    if not isinstance(master.index, pd.DatetimeIndex):
        raise TypeError("master deve usar DatetimeIndex.")
    assert_continuous_weekly_index(master.index, name="F5 event-dimension master")
    predictors = list(predictors or physical_predictor_columns(master))
    target_table = build_event_targets(events).sort_values("onset").reset_index(drop=True)
    onset_dates = pd.DatetimeIndex(pd.to_datetime(target_table["onset"]))
    positions = master.index.get_indexer(onset_dates, method="pad")
    valid_origin = positions >= 0
    target_table = target_table.loc[valid_origin].reset_index(drop=True)
    origin_times = master.index[positions[valid_origin]]
    if pd.Index(origin_times).duplicated().any():
        raise ValueError("Dois eventos mapearam para a mesma origem semanal.")
    event_ids = target_table["event_id"].astype(str).to_numpy()
    folds = make_event_folds(
        pd.DatetimeIndex(origin_times),
        event_ids,
        n_splits=n_splits,
        min_train_groups=min_train_events,
        purge_weeks=max((int(value) for value in lags), default=0),
    )
    target_columns = ("Y_pico", "Y_tempo_para_pico_sem", "Y_duracao_sem")
    prediction_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    augmentation_rows: list[dict[str, object]] = []
    source_codes = master.get("ocean_source_code")

    for fold_no, fold in enumerate(folds, start=1):
        fit_history = master.loc[: fold.train_end, predictors]
        preprocessor = FoldHarmonicPreprocessor(tuple(predictors)).fit(
            fit_history,
            source_codes=(
                source_codes.loc[fit_history.index] if source_codes is not None else None
            ),
        )
        prepared = preprocessor.transform(master[predictors], source_codes=source_codes)
        lagged = build_lagged_features(prepared, predictors, lags=lags)
        event_features = lagged.reindex(origin_times)
        train = np.asarray(fold.train_index, dtype=int)
        test = np.asarray(fold.test_index, dtype=int)
        usable = event_features.iloc[train].notna().any(axis=0)
        X_train = event_features.iloc[train].loc[:, usable]
        X_test = event_features.iloc[test].loc[:, usable]
        ocean_features = [
            column
            for column in X_train.columns
            if any(column.startswith(f"{name}_lag") for name in OCEAN_RAW_VARS)
        ]
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
        train_finite = np.isfinite(X_train.to_numpy()).all(axis=1) & train_source_valid.to_numpy()
        test_finite = np.isfinite(X_test.to_numpy()).all(axis=1) & test_source_valid.to_numpy()
        train = train[train_finite]
        test = test[test_finite]
        X_train = X_train.iloc[np.flatnonzero(train_finite)]
        X_test = X_test.iloc[np.flatnonzero(test_finite)]
        if len(train) < 3 or not len(test):
            continue
        train_types = target_table.iloc[train]["tipo"].astype(str)
        type_counts = train_types.value_counts()
        sample_weight = train_types.map(lambda value: 1.0 / type_counts.loc[value]).to_numpy()
        sample_weight = sample_weight / sample_weight.mean()

        for target_name in target_columns:
            y_all = pd.to_numeric(target_table[target_name], errors="coerce").to_numpy(dtype=float)
            valid_train_target = np.isfinite(y_all[train])
            valid_test_target = np.isfinite(y_all[test])
            if int(valid_train_target.sum()) < 3 or not valid_test_target.any():
                continue
            train_selected = train[valid_train_target]
            test_selected = test[valid_test_target]
            X_fit = X_train.iloc[np.flatnonzero(valid_train_target)].copy()
            y_fit = pd.Series(y_all[train_selected], name=target_name)
            weight_fit = sample_weight[valid_train_target]
            for event_position in train_selected:
                augmentation_rows.append(
                    {
                        "fold": fold.name,
                        "target": target_name,
                        "original_event_id": event_ids[event_position],
                        "augmentation_id": "original",
                        "augmentation_method": "none",
                        "independent_event": True,
                        "jitter_sigma": 0.0,
                    }
                )
            if jitter_copies > 0:
                original_count = len(X_fit)
                X_fit, y_fit = gaussian_jitter(
                    X_fit,
                    y_fit,
                    sigma=jitter_sigma,
                    n_copies=jitter_copies,
                    random_state=random_state + fold_no,
                )
                weight_fit = np.tile(weight_fit / (jitter_copies + 1), jitter_copies + 1)
                if len(X_fit) != original_count * (jitter_copies + 1):
                    raise AssertionError("Jitter F5 produziu tamanho inesperado.")
                for copy_no in range(1, jitter_copies + 1):
                    for event_position in train_selected:
                        augmentation_rows.append(
                            {
                                "fold": fold.name,
                                "target": target_name,
                                "original_event_id": event_ids[event_position],
                                "augmentation_id": f"jitter_{copy_no:02d}",
                                "augmentation_method": "train_only_covariate_jitter",
                                "independent_event": False,
                                "jitter_sigma": float(jitter_sigma),
                            }
                        )
            if model == "xgb":
                try:
                    from xgboost import XGBRegressor
                except ImportError as exc:  # pragma: no cover
                    raise ImportError("XGBoost nao instalado.") from exc
                estimator = XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=3,
                    learning_rate=0.04,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=random_state + fold_no,
                    n_jobs=-1,
                )
            elif model == "rf":
                estimator = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=random_state + fold_no,
                    n_jobs=-1,
                )
            else:
                raise ValueError("model deve ser 'rf' ou 'xgb'.")
            estimator.fit(X_fit.to_numpy(dtype=float), y_fit.to_numpy(), sample_weight=weight_fit)
            configure_serial_prediction(estimator)
            X_eval = X_test.iloc[np.flatnonzero(valid_test_target)]
            prediction = estimator.predict(X_eval.to_numpy(dtype=float))
            global_baseline = float(np.average(y_all[train_selected], weights=sample_weight[valid_train_target]))
            type_baseline_map = (
                pd.DataFrame(
                    {
                        "tipo": target_table.iloc[train_selected]["tipo"].astype(str).to_numpy(),
                        "target": y_all[train_selected],
                        "weight": sample_weight[valid_train_target],
                    }
                )
                .groupby("tipo")
                .apply(
                    lambda group: float(np.average(group["target"], weights=group["weight"])),
                    include_groups=False,
                )
            )
            test_types = target_table.iloc[test_selected]["tipo"].astype(str)
            type_baseline = np.asarray(
                [type_baseline_map.get(value, global_baseline) for value in test_types],
                dtype=float,
            )
            for local_no, event_position in enumerate(test_selected):
                prediction_rows.append(
                    {
                        "fold": fold.name,
                        "event_id": event_ids[event_position],
                        "tipo": target_table.iloc[event_position]["tipo"],
                        "origin_time": origin_times[event_position],
                        "target": target_name,
                        "observed": y_all[event_position],
                        "predicted": float(prediction[local_no]),
                        "baseline_global_train": global_baseline,
                        "baseline_type_train": float(type_baseline[local_no]),
                        "preprocessing_fit_end": fold.train_end,
                        "augmentation_train_only": True,
                    }
                )
            base_error = np.abs(prediction - y_all[test_selected])
            rng = np.random.default_rng(random_state + 50_000 + fold_no)
            predictor_positions = {
                predictor: [
                    column_no
                    for column_no, column in enumerate(X_eval.columns)
                    if column.startswith(f"{predictor}_lag")
                ]
                for predictor in predictors
            }
            for predictor, columns in predictor_positions.items():
                if not columns:
                    continue
                permuted = X_eval.to_numpy(dtype=float).copy()
                order = rng.permutation(len(permuted))
                permuted[:, columns] = permuted[order][:, columns]
                changed = estimator.predict(permuted)
                importance_rows.append(
                    {
                        "fold": fold.name,
                        "target": target_name,
                        "predictor": predictor,
                        "delta_mae_permutation_oos": float(
                            np.mean(np.abs(changed - y_all[test_selected]) - base_error)
                        ),
                        "n_lag_features_permuted": len(columns),
                        "n_test_events": len(test_selected),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    metric_rows: list[dict[str, object]] = []
    if not predictions.empty:
        for target_name, group in predictions.groupby("target", sort=False):
            observed = pd.to_numeric(group["observed"], errors="coerce").to_numpy(dtype=float)
            predicted = pd.to_numeric(group["predicted"], errors="coerce").to_numpy(dtype=float)
            baseline = pd.to_numeric(group["baseline_type_train"], errors="coerce").to_numpy(dtype=float)
            mae = float(np.mean(np.abs(predicted - observed)))
            mae_baseline = float(np.mean(np.abs(baseline - observed)))
            metric_rows.append(
                {
                    "target": target_name,
                    "model": model,
                    "n_oos_events": int(group["event_id"].nunique()),
                    "mae_oos_event": mae,
                    "rmse_oos_event": float(np.sqrt(np.mean((predicted - observed) ** 2))),
                    "mae_type_climatology_train": mae_baseline,
                    "skill_mae_vs_type_climatology": (
                        1.0 - mae / mae_baseline if mae_baseline > 0 else np.nan
                    ),
                    "validation": "expanding whole-event folds; event is independent unit",
                }
            )
    return EventRegressionResult(
        metrics=pd.DataFrame(metric_rows),
        predictions=predictions,
        importances=pd.DataFrame(importance_rows),
        augmentation_provenance=pd.DataFrame(augmentation_rows),
    )


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
