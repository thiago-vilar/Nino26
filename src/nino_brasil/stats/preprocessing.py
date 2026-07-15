"""Leakage-safe preprocessing for weekly physical predictors."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeasonalTrendConfig:
    """Configuration for a harmonic seasonal-cycle and linear-trend fit."""

    harmonics: int = 3
    remove_trend: bool = True
    detrend_pre_anomalized: bool = True
    standardize: bool = True
    min_observations: int = 104

    def __post_init__(self) -> None:
        if self.harmonics < 1:
            raise ValueError("harmonics must be at least 1")
        if self.min_observations < 8:
            raise ValueError("min_observations must be at least 8")


def _validate_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.DatetimeIndex(out.index)
    if out.index.has_duplicates or not out.index.is_monotonic_increasing:
        raise ValueError(f"{name} must have a unique, increasing DatetimeIndex")
    if out.columns.has_duplicates:
        raise ValueError(f"{name} must have unique columns")
    return out.apply(pd.to_numeric, errors="coerce")


class SeasonalTrendTransformer:
    """Fit seasonal/trend transformations using training dates only.

    The class has a deliberately small API: call :meth:`fit` with the training
    rows and :meth:`transform` on any chronologically compatible rows.  It does
    not accept a full data frame plus a mask, reducing the chance that a caller
    accidentally estimates climatology or scaling statistics on validation or
    test data.
    """

    def __init__(
        self,
        *,
        config: SeasonalTrendConfig = SeasonalTrendConfig(),
        already_anomalous: Iterable[str] = (),
    ) -> None:
        self.config = config
        self.already_anomalous = frozenset(str(value) for value in already_anomalous)
        self.columns_: tuple[str, ...] | None = None
        self.coefficients_: dict[str, np.ndarray] = {}
        self.design_names_: dict[str, tuple[str, ...]] = {}
        self.center_: dict[str, float] = {}
        self.scale_: dict[str, float] = {}
        self.n_fit_: dict[str, int] = {}
        self.reference_day_: float | None = None
        self.fit_start_: pd.Timestamp | None = None
        self.fit_end_: pd.Timestamp | None = None

    def _design(self, index: pd.DatetimeIndex, *, anomalous: bool) -> tuple[np.ndarray, tuple[str, ...]]:
        if self.reference_day_ is None:
            raise RuntimeError("transformer is not fitted")
        absolute_day = (
            index.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float)
            / pd.Timedelta(days=1).value
        )
        trend_years = (absolute_day - self.reference_day_) / 365.2425
        day = index.dayofyear.to_numpy(dtype=float) - 1.0
        angle = 2.0 * np.pi * day / 365.2425
        parts: list[np.ndarray] = [np.ones(len(index), dtype=float)]
        names = ["intercept"]
        include_trend = self.config.remove_trend and (
            not anomalous or self.config.detrend_pre_anomalized
        )
        if include_trend:
            parts.append(trend_years)
            names.append("linear_trend_years")
        if not anomalous:
            for harmonic in range(1, self.config.harmonics + 1):
                parts.extend((np.sin(harmonic * angle), np.cos(harmonic * angle)))
                names.extend((f"sin_{harmonic}", f"cos_{harmonic}"))
        return np.column_stack(parts), tuple(names)

    def fit(self, train: pd.DataFrame) -> "SeasonalTrendTransformer":
        """Estimate every coefficient and scale from ``train`` only."""

        frame = _validate_frame(train, name="train")
        if frame.empty:
            raise ValueError("train cannot be empty")
        self.columns_ = tuple(str(column) for column in frame.columns)
        self.fit_start_ = pd.Timestamp(frame.index.min())
        self.fit_end_ = pd.Timestamp(frame.index.max())
        absolute_day = (
            frame.index.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float)
            / pd.Timedelta(days=1).value
        )
        self.reference_day_ = float(np.mean(absolute_day))
        self.coefficients_.clear()
        self.design_names_.clear()
        self.center_.clear()
        self.scale_.clear()
        self.n_fit_.clear()

        for column in self.columns_:
            anomalous = column in self.already_anomalous or column.endswith("_anom")
            design, names = self._design(frame.index, anomalous=anomalous)
            values = frame[column].to_numpy(dtype=float)
            valid = np.isfinite(values)
            required = max(self.config.min_observations, design.shape[1] + 2)
            if int(valid.sum()) < required:
                raise ValueError(
                    f"{column!r} has {int(valid.sum())} finite training rows; "
                    f"at least {required} are required"
                )
            beta, *_ = np.linalg.lstsq(design[valid], values[valid], rcond=None)
            residual = values[valid] - design[valid] @ beta
            center = float(np.mean(residual))
            scale = float(np.std(residual, ddof=1)) if self.config.standardize else 1.0
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"{column!r} has zero residual variance in training data")
            self.coefficients_[column] = beta
            self.design_names_[column] = names
            self.center_[column] = center
            self.scale_[column] = scale
            self.n_fit_[column] = int(valid.sum())
        return self

    def _require_fitted(self) -> tuple[str, ...]:
        if self.columns_ is None:
            raise RuntimeError("fit must be called before transform")
        return self.columns_

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply frozen training coefficients without refitting."""

        columns = self._require_fitted()
        values = _validate_frame(frame, name="frame")
        if tuple(str(column) for column in values.columns) != columns:
            raise ValueError("frame columns and order must match the fitted training frame")
        out = pd.DataFrame(index=values.index)
        for column in columns:
            anomalous = column in self.already_anomalous or column.endswith("_anom")
            design, names = self._design(values.index, anomalous=anomalous)
            if names != self.design_names_[column]:
                raise RuntimeError("internal design contract changed after fitting")
            residual = values[column].to_numpy(dtype=float) - design @ self.coefficients_[column]
            if self.config.standardize:
                residual = (residual - self.center_[column]) / self.scale_[column]
            out[column] = residual
        return out

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)

    def metadata(self, *, evaluation_mode: str = "rolling_origin_operacional") -> pd.DataFrame:
        """Return one auditable row per transformed predictor."""

        columns = self._require_fitted()
        rows = []
        for column in columns:
            anomalous = column in self.already_anomalous or column.endswith("_anom")
            rows.append(
                {
                    "variavel": column,
                    "pre_anomalia": anomalous,
                    "transformacao": "residuo_harmonico_tendencia",
                    "harmonicos": 0 if anomalous else self.config.harmonics,
                    "tendencia_removida": "linear" if "linear_trend_years" in self.design_names_[column] else "nao",
                    "padronizado_com_treino": bool(self.config.standardize),
                    "ajuste_inicio": self.fit_start_,
                    "ajuste_fim": self.fit_end_,
                    "n_ajuste": self.n_fit_[column],
                    "escala_treino": self.scale_[column],
                    "termos_modelo": json.dumps(self.design_names_[column]),
                    "coeficientes": json.dumps(self.coefficients_[column].tolist()),
                    "evaluation_mode": evaluation_mode,
                    "fit_uses_evaluation_rows": False,
                }
            )
        return pd.DataFrame(rows)


def full_sample_diagnostic_transform(
    frame: pd.DataFrame,
    *,
    config: SeasonalTrendConfig = SeasonalTrendConfig(),
    already_anomalous: Iterable[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transform a full period for composites, explicitly marked diagnostic.

    This helper is intentionally named so its output cannot be mistaken for a
    leakage-safe forecast experiment.  Prediction code must fit a transformer
    separately inside every training fold.
    """

    transformer = SeasonalTrendTransformer(config=config, already_anomalous=already_anomalous)
    transformed = transformer.fit_transform(frame)
    metadata = transformer.metadata(evaluation_mode="diagnostico_retrospectivo_amostra_completa")
    metadata["fit_uses_evaluation_rows"] = True
    return transformed, metadata
