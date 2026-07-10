"""Leakage-safe variable reduction and interpretable Phase-4 MLR helpers.

The routines in this module serve two distinct scientific purposes:

* reduce a wide Pacific state vector without counting several proxies of the
  same physical dimension as independent evidence; and
* quantify *contemporaneous association* between the remaining Pacific
  anomalies and the continuous local ONI within each ENSO phase.

Every transform used for validation (seasonal adjustment, redundancy filter,
imputation, scaling, PCA and supervised ranking) is fitted on the training
block only.  Full-period fits are therefore descriptive reference fits, not
estimates of out-of-sample performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


def _require_datetime_frame(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if not isinstance(X.index, pd.DatetimeIndex):
        raise TypeError("X must have a DatetimeIndex.")
    if X.empty or X.shape[1] == 0:
        raise ValueError("X must contain observations and variables.")
    if X.columns.duplicated().any():
        raise ValueError("X columns must be unique.")
    return X.sort_index().astype(float)


def _harmonic_design(index: pd.DatetimeIndex, n_harmonics: int) -> np.ndarray:
    day = index.dayofyear.to_numpy(dtype=float)
    angle = 2.0 * np.pi * (day - 1.0) / 365.2425
    columns = [np.ones(len(index), dtype=float)]
    for harmonic in range(1, n_harmonics + 1):
        columns.extend(
            [np.sin(harmonic * angle), np.cos(harmonic * angle)]
        )
    return np.column_stack(columns)


class HarmonicDeseasonalizer(BaseEstimator, TransformerMixin):
    """Remove a smooth calendar cycle estimated from the fitted block only."""

    def __init__(self, n_harmonics: int = 3, min_observations: int = 30):
        self.n_harmonics = n_harmonics
        self.min_observations = min_observations

    def fit(self, X: pd.DataFrame, y=None):
        X = _require_datetime_frame(X)
        if self.n_harmonics < 0:
            raise ValueError("n_harmonics must be non-negative.")
        if self.min_observations < 2 * self.n_harmonics + 2:
            raise ValueError("min_observations is too small for the harmonic design.")
        design = _harmonic_design(X.index, self.n_harmonics)
        coefficients = np.full((design.shape[1], X.shape[1]), np.nan, dtype=float)
        for column_index, name in enumerate(X.columns):
            values = X[name].to_numpy(dtype=float)
            finite = np.isfinite(values)
            if int(finite.sum()) < self.min_observations:
                raise ValueError(
                    f"Variable {name!r} has fewer than {self.min_observations} "
                    "finite training observations."
                )
            coefficients[:, column_index], *_ = np.linalg.lstsq(
                design[finite], values[finite], rcond=None
            )
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.coefficients_ = coefficients
        self.train_start_ = pd.Timestamp(X.index.min())
        self.train_end_ = pd.Timestamp(X.index.max())
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("HarmonicDeseasonalizer must be fitted first.")
        X = _require_datetime_frame(X)
        missing = [name for name in self.feature_names_in_ if name not in X.columns]
        if missing:
            raise ValueError(f"Missing fitted variables: {missing}")
        X = X.loc[:, self.feature_names_in_]
        seasonal_cycle = _harmonic_design(X.index, self.n_harmonics) @ self.coefficients_
        residual = X.to_numpy(dtype=float) - seasonal_cycle
        residual[~np.isfinite(X.to_numpy(dtype=float))] = np.nan
        return pd.DataFrame(residual, index=X.index, columns=X.columns)


class CorrelationRedundancyFilter(BaseEstimator, TransformerMixin):
    """Keep one deterministic medoid from every high-correlation component."""

    def __init__(
        self,
        *,
        correlation_threshold: float = 0.90,
        max_missing_fraction: float = 0.20,
        min_variance: float = 1e-12,
    ):
        self.correlation_threshold = correlation_threshold
        self.max_missing_fraction = max_missing_fraction
        self.min_variance = min_variance

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("X must be a non-empty DataFrame.")
        if not 0.0 < self.correlation_threshold < 1.0:
            raise ValueError("correlation_threshold must be between zero and one.")
        if not 0.0 <= self.max_missing_fraction < 1.0:
            raise ValueError("max_missing_fraction must be in [0, 1).")

        X = X.astype(float)
        columns = list(X.columns)
        missing_fraction = X.isna().mean()
        variance = X.var(ddof=0, skipna=True)
        eligible = [
            name
            for name in columns
            if missing_fraction[name] <= self.max_missing_fraction
            and np.isfinite(variance[name])
            and variance[name] > self.min_variance
        ]
        if not eligible:
            raise ValueError("No variable passed coverage and variance filters.")

        medians = X[eligible].median()
        imputed = X[eligible].fillna(medians)
        correlation = imputed.corr(method="pearson").abs().fillna(0.0)

        # Connected components make transitive redundancy explicit.  The
        # representative is the most central, best-covered observed variable;
        # original order is the deterministic tie-breaker.
        unseen = set(eligible)
        components: list[list[str]] = []
        while unseen:
            seed = next(name for name in eligible if name in unseen)
            unseen.remove(seed)
            stack = [seed]
            component: list[str] = []
            while stack:
                current = stack.pop()
                component.append(current)
                neighbours = [
                    name
                    for name in eligible
                    if name in unseen
                    and correlation.loc[current, name] >= self.correlation_threshold
                ]
                for name in neighbours:
                    unseen.remove(name)
                    stack.append(name)
            component.sort(key=columns.index)
            components.append(component)

        retained: list[str] = []
        representative_for: dict[str, str] = {}
        group_for: dict[str, str] = {}
        for component_index, component in enumerate(components, start=1):
            representative = max(
                component,
                key=lambda name: (
                    float(correlation.loc[name, component].mean()),
                    float(1.0 - missing_fraction[name]),
                    -columns.index(name),
                ),
            )
            retained.append(representative)
            for name in component:
                representative_for[name] = representative
                group_for[name] = f"R{component_index:02d}"

        rows: list[dict[str, object]] = []
        for order, name in enumerate(columns):
            if missing_fraction[name] > self.max_missing_fraction:
                representative = ""
                group = "excluida_cobertura"
                kept = False
                reason = (
                    f"excluida: fracao ausente {missing_fraction[name]:.3f} > "
                    f"{self.max_missing_fraction:.3f} no treino"
                )
                corr_to_rep = np.nan
            elif not np.isfinite(variance[name]) or variance[name] <= self.min_variance:
                representative = ""
                group = "excluida_variancia"
                kept = False
                reason = "excluida: variancia nula/quase nula no treino"
                corr_to_rep = np.nan
            else:
                representative = representative_for[name]
                group = group_for[name]
                kept = name == representative
                corr_to_rep = float(correlation.loc[name, representative])
                if kept:
                    reason = "retida: representante medoid do grupo de redundancia"
                else:
                    reason = (
                        f"excluida: redundante com {representative} "
                        f"(|r|={corr_to_rep:.3f})"
                    )
            rows.append(
                {
                    "ordem_original": order,
                    "variavel": name,
                    "retida_redundancia": bool(kept),
                    "grupo_redundancia": group,
                    "representante": representative,
                    "motivo_redundancia": reason,
                    "fracao_ausente_treino": float(missing_fraction[name]),
                    "variancia_anomalia_treino": float(variance[name]),
                    "abs_r_representante": corr_to_rep,
                    "limiar_abs_r": float(self.correlation_threshold),
                }
            )

        self.feature_names_in_ = np.asarray(columns, dtype=object)
        self.selected_features_ = tuple(retained)
        self.audit_ = pd.DataFrame(rows)
        self.correlation_ = correlation
        self.n_features_in_ = len(columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("CorrelationRedundancyFilter must be fitted first.")
        missing = [name for name in self.selected_features_ if name not in X.columns]
        if missing:
            raise ValueError(f"Missing retained variables: {missing}")
        return X.loc[:, self.selected_features_].copy()


@dataclass
class FittedReduction:
    seasonalizer: HarmonicDeseasonalizer
    redundancy_filter: CorrelationRedundancyFilter
    imputer: SimpleImputer
    scaler: StandardScaler
    pca: PCA
    selected_features: tuple[str, ...]
    component_representatives: tuple[str, ...]
    audit: pd.DataFrame
    pca_variance: pd.DataFrame
    pca_loadings: pd.DataFrame

    def standardized_matrix(self, X: pd.DataFrame) -> np.ndarray:
        anomalies = self.seasonalizer.transform(X)
        reduced = self.redundancy_filter.transform(anomalies)
        return self.scaler.transform(self.imputer.transform(reduced))

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.pca.transform(self.standardized_matrix(X))

    def reconstruct_standardized(self, X: pd.DataFrame) -> np.ndarray:
        return self.pca.inverse_transform(self.transform(X))


def fit_reduction_pipeline(
    X_train: pd.DataFrame,
    *,
    n_harmonics: int = 3,
    correlation_threshold: float = 0.90,
    max_missing_fraction: float = 0.20,
    min_variance: float = 1e-12,
    variance_target: float = 0.90,
) -> FittedReduction:
    """Fit harmonic anomalies -> redundancy filter -> scale -> PCA."""

    X_train = _require_datetime_frame(X_train)
    if not 0.0 < variance_target <= 1.0:
        raise ValueError("variance_target must be in (0, 1].")

    seasonalizer = HarmonicDeseasonalizer(n_harmonics=n_harmonics).fit(X_train)
    anomalies = seasonalizer.transform(X_train)
    redundancy_filter = CorrelationRedundancyFilter(
        correlation_threshold=correlation_threshold,
        max_missing_fraction=max_missing_fraction,
        min_variance=min_variance,
    ).fit(anomalies)
    reduced = redundancy_filter.transform(anomalies)
    imputer = SimpleImputer(strategy="median").fit(reduced)
    imputed = imputer.transform(reduced)
    scaler = StandardScaler().fit(imputed)
    standardized = scaler.transform(imputed)

    if standardized.shape[1] == 1:
        pca = PCA(n_components=1, svd_solver="full").fit(standardized)
    else:
        pca = PCA(n_components=variance_target, svd_solver="full").fit(standardized)

    names = list(redundancy_filter.selected_features_)
    component_names = [f"PC{index + 1}" for index in range(pca.n_components_)]
    # Correlation loadings for standardized variables.
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)[None, :]
    representatives: list[str] = []
    used: set[str] = set()
    for component_index in range(pca.n_components_):
        candidates = sorted(
            range(len(names)),
            key=lambda row: (-abs(loadings[row, component_index]), row),
        )
        selected_index = next((row for row in candidates if names[row] not in used), candidates[0])
        representative = names[selected_index]
        representatives.append(representative)
        used.add(representative)

    variance_rows = []
    for index, component in enumerate(component_names):
        variance_rows.append(
            {
                "componente": component,
                "variancia_explicada": float(pca.explained_variance_ratio_[index]),
                "variancia_acumulada": float(
                    pca.explained_variance_ratio_[: index + 1].sum()
                ),
                "representante": representatives[index],
                "n_variaveis_apos_redundancia": len(names),
                "meta_variancia": float(variance_target),
            }
        )
    pca_variance = pd.DataFrame(variance_rows)

    loading_rows = []
    for row, name in enumerate(names):
        for column, component in enumerate(component_names):
            loading_rows.append(
                {
                    "componente": component,
                    "variavel": name,
                    "loading": float(loadings[row, column]),
                    "abs_loading": float(abs(loadings[row, column])),
                    "representante_componente": name == representatives[column],
                }
            )
    pca_loadings = pd.DataFrame(loading_rows)

    audit = redundancy_filter.audit_.copy()
    maximum_loading = (
        pca_loadings.groupby("variavel")["abs_loading"].max()
        if not pca_loadings.empty
        else pd.Series(dtype=float)
    )
    audit["representante_pca_referencia"] = audit["variavel"].isin(representatives)
    audit["max_abs_loading_pca"] = audit["variavel"].map(maximum_loading)
    audit["inicio_treino"] = X_train.index.min()
    audit["fim_treino"] = X_train.index.max()
    audit["n_semanas_treino"] = len(X_train)

    return FittedReduction(
        seasonalizer=seasonalizer,
        redundancy_filter=redundancy_filter,
        imputer=imputer,
        scaler=scaler,
        pca=pca,
        selected_features=tuple(names),
        component_representatives=tuple(representatives),
        audit=audit,
        pca_variance=pca_variance,
        pca_loadings=pca_loadings,
    )


def expanding_time_folds(
    index: pd.DatetimeIndex,
    *,
    n_folds: int = 5,
    min_train_years: int = 8,
) -> list[dict[str, object]]:
    """Create expanding folds from data-supported calendar-year blocks.

    No climate breakpoint is declared.  Boundaries are determined from the
    observed year span and are used only for out-of-sample validation.
    """

    index = pd.DatetimeIndex(index).sort_values().unique()
    years = np.asarray(sorted(pd.unique(index.year)), dtype=int)
    if n_folds < 1:
        raise ValueError("n_folds must be positive.")
    if len(years) < min_train_years + n_folds:
        raise ValueError("Not enough calendar years for the requested folds.")
    candidate_test_years = years[min_train_years:]
    test_blocks = [block for block in np.array_split(candidate_test_years, n_folds) if len(block)]
    folds: list[dict[str, object]] = []
    for fold_index, test_years in enumerate(test_blocks, start=1):
        train_years = years[years < test_years.min()]
        train_mask = np.isin(index.year, train_years)
        test_mask = np.isin(index.year, test_years)
        folds.append(
            {
                "fold": f"temporal_{fold_index:02d}",
                "train_index": pd.DatetimeIndex(index[train_mask]),
                "test_index": pd.DatetimeIndex(index[test_mask]),
                "train_start": pd.Timestamp(index[train_mask].min()),
                "train_end": pd.Timestamp(index[train_mask].max()),
                "test_start": pd.Timestamp(index[test_mask].min()),
                "test_end": pd.Timestamp(index[test_mask].max()),
            }
        )
    return folds


@dataclass
class TemporalReductionResult:
    reference_fit: FittedReduction
    selected_variables: pd.DataFrame
    fold_metrics: pd.DataFrame
    fold_selection: pd.DataFrame


def temporal_pca_stability(
    X: pd.DataFrame,
    *,
    n_folds: int = 5,
    min_train_years: int = 8,
    stability_threshold: float = 0.60,
    **reduction_kwargs,
) -> TemporalReductionResult:
    """Assess PCA representatives and reconstruction on future time blocks."""

    X = _require_datetime_frame(X)
    if not 0.0 <= stability_threshold <= 1.0:
        raise ValueError("stability_threshold must be in [0, 1].")
    reference = fit_reduction_pipeline(X, **reduction_kwargs)
    folds = expanding_time_folds(
        X.index, n_folds=n_folds, min_train_years=min_train_years
    )
    fold_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for fold in folds:
        train = X.loc[fold["train_index"]]
        test = X.loc[fold["test_index"]]
        fitted = fit_reduction_pipeline(train, **reduction_kwargs)
        observed = fitted.standardized_matrix(test)
        reconstructed = fitted.reconstruct_standardized(test)
        finite = np.isfinite(observed) & np.isfinite(reconstructed)
        denominator = float(np.sum(observed[finite] ** 2))
        preservation = (
            1.0 - float(np.sum((observed[finite] - reconstructed[finite]) ** 2)) / denominator
            if denominator > 0
            else np.nan
        )
        fold_rows.append(
            {
                **{key: value for key, value in fold.items() if not key.endswith("_index")},
                "n_treino": len(train),
                "n_teste": len(test),
                "n_variaveis_entrada": X.shape[1],
                "n_apos_redundancia": len(fitted.selected_features),
                "n_componentes_pca": fitted.pca.n_components_,
                "variancia_acumulada_treino": float(
                    fitted.pca.explained_variance_ratio_.sum()
                ),
                "variancia_preservada_teste": preservation,
            }
        )
        retained = set(fitted.selected_features)
        representatives = set(fitted.component_representatives)
        audit = fitted.audit.set_index("variavel")
        for name in X.columns:
            row = audit.loc[name]
            selection_rows.append(
                {
                    "fold": fold["fold"],
                    "variavel": name,
                    "retida_redundancia": name in retained,
                    "representante_pca": name in representatives,
                    "grupo_redundancia": row["grupo_redundancia"],
                    "representante": row["representante"],
                    "train_end": fold["train_end"],
                    "test_start": fold["test_start"],
                }
            )

    fold_selection = pd.DataFrame(selection_rows)
    stability = (
        fold_selection.groupby("variavel")["representante_pca"]
        .mean()
        .rename("estabilidade")
    )
    redundancy_stability = (
        fold_selection.groupby("variavel")["retida_redundancia"]
        .mean()
        .rename("estabilidade_redundancia")
    )
    selected = reference.audit.copy()
    selected = selected.merge(stability, on="variavel", how="left")
    selected = selected.merge(redundancy_stability, on="variavel", how="left")
    selected["selecionada"] = (
        selected["representante_pca_referencia"]
        & (selected["estabilidade"] >= stability_threshold)
    )
    selected["metodo"] = (
        "anomalia_harmonica_treino+filtro_abs_r+PCA90+estabilidade_temporal"
    )
    selected["motivo"] = np.where(
        selected["selecionada"],
        "selecionada: representa PC retido e reaparece de forma estavel nos treinos temporais",
        np.where(
            ~selected["retida_redundancia"],
            selected["motivo_redundancia"],
            "nao selecionada: redundancia removida, loading secundario ou representante temporalmente instavel",
        ),
    )
    selected["limiar_estabilidade"] = float(stability_threshold)
    selected["n_folds_estabilidade"] = len(folds)
    selected = selected.sort_values("ordem_original").reset_index(drop=True)
    return TemporalReductionResult(
        reference_fit=reference,
        selected_variables=selected,
        fold_metrics=pd.DataFrame(fold_rows),
        fold_selection=fold_selection,
    )


def epsilon_squared_kruskal(groups: Iterable[pd.Series]) -> tuple[float, float, int]:
    valid = [pd.Series(group).dropna() for group in groups]
    valid = [group for group in valid if len(group) >= 10]
    if len(valid) < 3:
        return np.nan, np.nan, int(sum(len(group) for group in valid))
    statistic, p_value = stats.kruskal(*valid)
    n = int(sum(len(group) for group in valid))
    k = len(valid)
    effect = max((float(statistic) - k + 1.0) / (n - k), 0.0)
    return effect, float(p_value), n


def stable_phase_influence(
    X: pd.DataFrame,
    phase_table: pd.DataFrame,
    *,
    phase_order: Sequence[str],
    event_types: Sequence[str] = ("el_nino", "la_nina"),
    min_effect: float = 0.06,
    stability_threshold: float = 0.60,
    n_folds: int = 5,
    min_train_years: int = 8,
    **reduction_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank phase effects repeatedly on expanding training histories.

    A variable is selected inside a fold only when it is a retained PCA
    representative *and* reaches a pre-specified medium epsilon-squared effect
    (>=0.06) for at least one ENSO type.  The final flag requires recurrence in
    the requested fraction of expanding histories.
    """

    X = _require_datetime_frame(X)
    phase_table = phase_table.reindex(X.index)
    folds = expanding_time_folds(
        X.index, n_folds=n_folds, min_train_years=min_train_years
    )
    rows: list[dict[str, object]] = []
    for fold in folds:
        train_index = fold["train_index"]
        train = X.loc[train_index]
        labels = phase_table.loc[train_index]
        reduction = fit_reduction_pipeline(train, **reduction_kwargs)
        anomalies = reduction.seasonalizer.transform(train)
        representatives = set(reduction.component_representatives)
        for name in X.columns:
            effects: dict[str, float] = {}
            for event_type in event_types:
                groups = [
                    anomalies.loc[
                        labels["tipo"].eq(event_type) & labels["fase"].eq(phase),
                        name,
                    ]
                    for phase in phase_order
                ]
                effect, _, _ = epsilon_squared_kruskal(groups)
                effects[event_type] = effect
            finite_effects = [value for value in effects.values() if np.isfinite(value)]
            maximum = max(finite_effects) if finite_effects else np.nan
            rows.append(
                {
                    "fold": fold["fold"],
                    "train_end": fold["train_end"],
                    "variavel": name,
                    "representante_pca": name in representatives,
                    "epsilon2_max": maximum,
                    **{f"epsilon2_{event_type}": value for event_type, value in effects.items()},
                    "selecionada_fold": bool(
                        name in representatives
                        and np.isfinite(maximum)
                        and maximum >= min_effect
                    ),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("variavel", as_index=False)
        .agg(
            estabilidade=("selecionada_fold", "mean"),
            importancia_media=("epsilon2_max", "mean"),
            importancia_mediana=("epsilon2_max", "median"),
            n_folds=("fold", "nunique"),
        )
    )
    summary["selecionada_estavel"] = summary["estabilidade"] >= stability_threshold
    summary["limiar_epsilon2"] = float(min_effect)
    summary["limiar_estabilidade"] = float(stability_threshold)
    summary["metodo"] = "representante_PCA+epsilon2_fase_em_treinos_temporais"
    summary = summary.sort_values(
        ["selecionada_estavel", "estabilidade", "importancia_media"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return summary, detail


def _event_balanced_abs_spearman(
    X: pd.DataFrame,
    y: pd.Series,
    events: pd.Series,
) -> pd.Series:
    scores: dict[str, float] = {}
    for name in X.columns:
        correlations: list[float] = []
        for event in pd.unique(events.dropna()):
            mask = events.eq(event)
            pair = pd.concat([X.loc[mask, name], y.loc[mask]], axis=1).dropna()
            if len(pair) < 4 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
                continue
            correlation = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic
            if np.isfinite(correlation):
                correlations.append(abs(float(correlation)))
        if correlations:
            scores[name] = float(np.median(correlations))
        else:
            pair = pd.concat([X[name], y], axis=1).dropna()
            correlation = (
                stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic
                if len(pair) >= 4
                and pair.iloc[:, 0].nunique() >= 2
                and pair.iloc[:, 1].nunique() >= 2
                else np.nan
            )
            scores[name] = abs(float(correlation)) if np.isfinite(correlation) else 0.0
    return pd.Series(scores, dtype=float).sort_values(ascending=False)


@dataclass
class StandardizedMLRFit:
    feature_names: tuple[str, ...]
    imputer: SimpleImputer
    x_scaler: StandardScaler
    y_mean: float
    y_scale: float
    beta: np.ndarray
    covariance_hac: np.ndarray
    coefficients: pd.DataFrame
    diagnostics: dict[str, object]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        values = self.x_scaler.transform(self.imputer.transform(X.loc[:, self.feature_names]))
        design = np.column_stack([np.ones(len(values)), values])
        return (design @ self.beta) * self.y_scale + self.y_mean


def _event_hac_covariance(
    design: np.ndarray,
    residual: np.ndarray,
    events: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    bread = np.linalg.pinv(design.T @ design)
    meat = np.zeros((design.shape[1], design.shape[1]), dtype=float)
    score = design * residual[:, None]
    for event in pd.unique(events):
        positions = np.flatnonzero(events == event)
        event_score = score[positions]
        meat += event_score.T @ event_score
        event_lag = min(max_lag, max(len(event_score) - 1, 0))
        for lag in range(1, event_lag + 1):
            weight = 1.0 - lag / (max_lag + 1.0)
            gamma = event_score[lag:].T @ event_score[:-lag]
            meat += weight * (gamma + gamma.T)
    correction = len(design) / max(len(design) - design.shape[1], 1)
    covariance = correction * (bread @ meat @ bread)
    return (covariance + covariance.T) / 2.0


def _vif_values(standardized: np.ndarray) -> np.ndarray:
    if standardized.shape[1] == 1:
        return np.ones(1, dtype=float)
    values = np.full(standardized.shape[1], np.nan, dtype=float)
    for column in range(standardized.shape[1]):
        target = standardized[:, column]
        others = np.delete(standardized, column, axis=1)
        design = np.column_stack([np.ones(len(others)), others])
        fitted = design @ np.linalg.lstsq(design, target, rcond=None)[0]
        denominator = float(np.sum((target - target.mean()) ** 2))
        r_squared = 1.0 - float(np.sum((target - fitted) ** 2)) / denominator if denominator else 1.0
        values[column] = 1.0 / max(1.0 - r_squared, 1e-12)
    return values


def fit_standardized_mlr(
    X: pd.DataFrame,
    y: pd.Series,
    events: pd.Series,
    *,
    hac_lags: int = 13,
) -> StandardizedMLRFit:
    """Fit OLS on standardized variables with event-bounded Newey-West SEs."""

    frame = pd.concat([X, y.rename("__target"), events.rename("__event")], axis=1).sort_index()
    frame = frame.loc[frame["__target"].notna() & frame["__event"].notna()]
    if frame.empty:
        raise ValueError("No aligned observations for MLR.")
    feature_names = tuple(X.columns)
    imputer = SimpleImputer(strategy="median").fit(frame.loc[:, feature_names])
    imputed = imputer.transform(frame.loc[:, feature_names])
    x_scaler = StandardScaler().fit(imputed)
    standardized = x_scaler.transform(imputed)
    target = frame["__target"].to_numpy(dtype=float)
    y_mean = float(target.mean())
    y_scale = float(target.std(ddof=0))
    if not np.isfinite(y_scale) or y_scale <= 0:
        raise ValueError("MLR target must have non-zero variance.")
    target_standardized = (target - y_mean) / y_scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    if len(design) <= design.shape[1] + 1:
        raise ValueError("Insufficient residual degrees of freedom for MLR.")
    beta = np.linalg.lstsq(design, target_standardized, rcond=None)[0]
    fitted_standardized = design @ beta
    residual = target_standardized - fitted_standardized
    event_values = frame["__event"].astype(str).to_numpy()
    covariance = _event_hac_covariance(
        design, residual, event_values, max_lag=hac_lags
    )
    standard_error = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    n_events = int(pd.Series(event_values).nunique())
    degrees_freedom = max(n_events - 1, 1)
    critical = float(stats.t.ppf(0.975, degrees_freedom))
    t_value = beta / np.where(standard_error > 0, standard_error, np.nan)
    p_value = 2.0 * stats.t.sf(np.abs(t_value), degrees_freedom)
    vif = np.r_[np.nan, _vif_values(standardized)]
    names = ("intercepto",) + feature_names
    coefficients = pd.DataFrame(
        {
            "termo": names,
            "beta_padronizado": beta,
            "erro_padrao_hac": standard_error,
            "ic95_hac_inferior": beta - critical * standard_error,
            "ic95_hac_superior": beta + critical * standard_error,
            "t_hac": t_value,
            "p_hac": p_value,
            "vif": vif,
        }
    )

    fitted_raw = fitted_standardized * y_scale + y_mean
    residual_raw = target - fitted_raw
    sse = float(np.sum(residual**2))
    sst = float(np.sum((target_standardized - target_standardized.mean()) ** 2))
    r_squared = 1.0 - sse / sst if sst > 0 else np.nan
    adjusted = (
        1.0 - (1.0 - r_squared) * (len(design) - 1) / (len(design) - design.shape[1])
        if np.isfinite(r_squared)
        else np.nan
    )
    dw_numerator = 0.0
    for event in pd.unique(event_values):
        event_residual = residual[event_values == event]
        if len(event_residual) > 1:
            dw_numerator += float(np.sum(np.diff(event_residual) ** 2))
    dw_denominator = float(np.sum(residual**2))
    condition_number = float(np.linalg.cond(design))
    diagnostics = {
        "n_observacoes": len(design),
        "n_eventos": n_events,
        "n_preditores": len(feature_names),
        "r2_ajuste": r_squared,
        "r2_ajustado": adjusted,
        "rmse_oni": float(np.sqrt(np.mean(residual_raw**2))),
        "mae_oni": float(np.mean(np.abs(residual_raw))),
        "durbin_watson_dentro_evento": (
            dw_numerator / dw_denominator if dw_denominator > 0 else np.nan
        ),
        "condicao_design": condition_number,
        "vif_max": float(np.nanmax(vif[1:])),
        "hac_lags_semanas": int(hac_lags),
        "covariancia": "Newey-West/Bartlett limitada ao mesmo evento ENSO",
    }
    return StandardizedMLRFit(
        feature_names=feature_names,
        imputer=imputer,
        x_scaler=x_scaler,
        y_mean=y_mean,
        y_scale=y_scale,
        beta=beta,
        covariance_hac=covariance,
        coefficients=coefficients,
        diagnostics=diagnostics,
    )


@dataclass
class PhaseMLRFit:
    reduction: FittedReduction
    selected_features: tuple[str, ...]
    ranking: pd.DataFrame
    model: StandardizedMLRFit

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        anomalies = self.reduction.seasonalizer.transform(X)
        return self.model.predict(anomalies.loc[:, self.selected_features])


def fit_phase_mlr(
    X: pd.DataFrame,
    y: pd.Series,
    events: pd.Series,
    *,
    excluded_features: Sequence[str] = ("nino34_ssta",),
    hac_lags: int = 13,
    **reduction_kwargs,
) -> PhaseMLRFit:
    """Select within training data and fit an interpretable phase-specific MLR."""

    aligned = pd.concat([X, y.rename("__target"), events.rename("__event")], axis=1).sort_index()
    aligned = aligned.loc[aligned["__target"].notna() & aligned["__event"].notna()]
    X_train = aligned.loc[:, X.columns]
    y_train = aligned["__target"]
    event_train = aligned["__event"].astype(str)
    reduction = fit_reduction_pipeline(X_train, **reduction_kwargs)
    anomalies = reduction.seasonalizer.transform(X_train)
    candidates = [
        name
        for name in reduction.component_representatives
        if name not in set(excluded_features)
    ]
    if not candidates:
        raise ValueError("No non-tautological PCA representative remained for MLR.")
    ranking_score = _event_balanced_abs_spearman(
        anomalies.loc[:, candidates], y_train, event_train
    )
    n_events = int(event_train.nunique())
    # Events, not autocorrelated weeks, determine the conservative dimension.
    max_predictors = max(1, min(len(candidates), n_events // 2))
    selected = tuple(ranking_score.index[:max_predictors])
    ranking = ranking_score.rename("abs_spearman_mediana_eventos").reset_index()
    ranking = ranking.rename(columns={"index": "variavel"})
    ranking["selecionada_mlr"] = ranking["variavel"].isin(selected)
    ranking["n_eventos_treino"] = n_events
    ranking["max_preditores_regra"] = max_predictors
    ranking["regra_dimensao"] = "p <= floor(n_eventos_treino/2)"
    model = fit_standardized_mlr(
        anomalies.loc[:, selected], y_train, event_train, hac_lags=hac_lags
    )
    return PhaseMLRFit(
        reduction=reduction,
        selected_features=selected,
        ranking=ranking,
        model=model,
    )


@dataclass
class WalkForwardMLRResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    selection: pd.DataFrame
    coefficients: pd.DataFrame
    diagnostics: pd.DataFrame


def _safe_regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[finite], predicted[finite]
    if not len(observed):
        return {"rmse": np.nan, "mae": np.nan, "r": np.nan, "r2_oos": np.nan}
    rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    mae = float(np.mean(np.abs(predicted - observed)))
    correlation = (
        float(np.corrcoef(observed, predicted)[0, 1])
        if len(observed) >= 3 and np.std(observed) > 0 and np.std(predicted) > 0
        else np.nan
    )
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    r2 = 1.0 - float(np.sum((predicted - observed) ** 2)) / denominator if denominator > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "r": correlation, "r2_oos": r2}


def walk_forward_phase_mlr(
    X: pd.DataFrame,
    oni_weekly: pd.Series,
    phase_table: pd.DataFrame,
    *,
    phase_order: Sequence[str],
    event_types: Sequence[str] = ("el_nino", "la_nina"),
    min_train_events: int = 5,
    excluded_features: Sequence[str] = ("nino34_ssta",),
    hac_lags: int = 13,
    **reduction_kwargs,
) -> WalkForwardMLRResult:
    """Walk forward by whole ENSO event, then fit full reference models."""

    X = _require_datetime_frame(X)
    phase_table = phase_table.reindex(X.index)
    oni_weekly = oni_weekly.reindex(X.index)
    prediction_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []

    for event_type in event_types:
        for phase in phase_order:
            stratum = phase_table["tipo"].eq(event_type) & phase_table["fase"].eq(phase)
            events = phase_table.loc[stratum, "event_id"].replace("", np.nan).dropna()
            event_order = (
                events.reset_index()
                .groupby("event_id")[events.index.name or "index"]
                .min()
                .sort_values()
                .index.tolist()
            )
            if len(event_order) <= min_train_events:
                continue

            for test_position in range(min_train_events, len(event_order)):
                train_events = event_order[:test_position]
                test_event = event_order[test_position]
                train_mask = stratum & phase_table["event_id"].isin(train_events) & oni_weekly.notna()
                test_mask = stratum & phase_table["event_id"].eq(test_event) & oni_weekly.notna()
                if int(train_mask.sum()) < 30 or int(test_mask.sum()) < 2:
                    continue
                fit = fit_phase_mlr(
                    X.loc[train_mask],
                    oni_weekly.loc[train_mask],
                    phase_table.loc[train_mask, "event_id"],
                    excluded_features=excluded_features,
                    hac_lags=hac_lags,
                    **reduction_kwargs,
                )
                predicted = fit.predict(X.loc[test_mask])
                baseline = float(oni_weekly.loc[train_mask].mean())
                fold_name = f"{event_type}|{phase}|teste_{test_event}"
                for timestamp, observed, estimate in zip(
                    X.index[test_mask], oni_weekly.loc[test_mask], predicted
                ):
                    prediction_rows.append(
                        {
                            "fold": fold_name,
                            "tipo": event_type,
                            "fase": phase,
                            "evento_teste": test_event,
                            "semana": timestamp,
                            "oni_observado_c": float(observed),
                            "oni_previsto_mlr_c": float(estimate),
                            "oni_baseline_treino_c": baseline,
                            "n_eventos_treino": len(train_events),
                            "variaveis": "+".join(fit.selected_features),
                        }
                    )
                for _, row in fit.ranking.iterrows():
                    selection_rows.append(
                        {
                            "fold": fold_name,
                            "tipo": event_type,
                            "fase": phase,
                            "evento_teste": test_event,
                            **row.to_dict(),
                        }
                    )

            # Full-period reference coefficients; never used as validation.
            full_mask = stratum & oni_weekly.notna()
            if int(full_mask.sum()) < 30 or phase_table.loc[full_mask, "event_id"].nunique() < 3:
                continue
            reference = fit_phase_mlr(
                X.loc[full_mask],
                oni_weekly.loc[full_mask],
                phase_table.loc[full_mask, "event_id"],
                excluded_features=excluded_features,
                hac_lags=hac_lags,
                **reduction_kwargs,
            )
            for _, row in reference.model.coefficients.iterrows():
                coefficient_rows.append(
                    {
                        "tipo": event_type,
                        "fase": phase,
                        "ajuste": "referencia_periodo_completo_descritivo",
                        **row.to_dict(),
                    }
                )
            diagnostic_rows.append(
                {
                    "tipo": event_type,
                    "fase": phase,
                    "ajuste": "referencia_periodo_completo_descritivo",
                    "variaveis": "+".join(reference.selected_features),
                    **reference.model.diagnostics,
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    metric_rows: list[dict[str, object]] = []
    if not predictions.empty:
        for (event_type, phase), group in predictions.groupby(["tipo", "fase"]):
            mlr = _safe_regression_metrics(
                group["oni_observado_c"], group["oni_previsto_mlr_c"]
            )
            baseline = _safe_regression_metrics(
                group["oni_observado_c"], group["oni_baseline_treino_c"]
            )
            metric_rows.append(
                {
                    "tipo": event_type,
                    "fase": phase,
                    "n_semanas_teste": len(group),
                    "n_eventos_teste": group["evento_teste"].nunique(),
                    "rmse_mlr": mlr["rmse"],
                    "mae_mlr": mlr["mae"],
                    "r_mlr": mlr["r"],
                    "r2_oos_mlr": mlr["r2_oos"],
                    "rmse_baseline": baseline["rmse"],
                    "mae_baseline": baseline["mae"],
                    "skill_rmse_vs_media_treino": (
                        1.0 - mlr["rmse"] / baseline["rmse"]
                        if np.isfinite(baseline["rmse"]) and baseline["rmse"] > 0
                        else np.nan
                    ),
                    "validacao": "walk-forward por evento inteiro; todos os transforms fitados no treino",
                }
            )
    return WalkForwardMLRResult(
        predictions=predictions,
        metrics=pd.DataFrame(metric_rows),
        selection=pd.DataFrame(selection_rows),
        coefficients=pd.DataFrame(coefficient_rows),
        diagnostics=pd.DataFrame(diagnostic_rows),
    )

