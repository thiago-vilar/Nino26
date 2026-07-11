from __future__ import annotations

import numpy as np
import pandas as pd


def _fourier_design(day_of_year: np.ndarray, harmonics: int) -> np.ndarray:
    angle = 2.0 * np.pi * (day_of_year.astype(float) - 1.0) / 365.2425
    columns = [np.ones_like(angle)]
    for harmonic in range(1, harmonics + 1):
        columns.append(np.sin(harmonic * angle))
        columns.append(np.cos(harmonic * angle))
    return np.column_stack(columns)


def harmonic_anomaly_matrix(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    base: tuple[str, str] = ("1991-01-01", "2020-12-31"),
    harmonics: int = 3,
    detrend: bool = True,
) -> pd.DataFrame:
    """Anomalia harmonica (e detrend linear) por coluna, com ajuste na base.

    Remove das colunas indicadas o ciclo anual (Fourier ``harmonics`` termos) e,
    opcionalmente, a tendencia linear secular. Ambos sao ajustados APENAS na
    janela ``base`` (default 1991-2020, a climatologia canonica do projeto) e
    extrapolados deterministicamente para fora dela - nenhuma informacao do
    periodo de teste pos-base entra no ajuste (parecer 2026-05-26, item 1.2).

    Motivacao (parecer 2026-07-10): as variaveis oceanicas do master semanal sao
    valores fisicos crus cujo ciclo anual domina a variancia semanal (ex.: D20
    com amplitude sazonal ~25 m vs sigma total ~15 m). Correlacoes condicionadas
    a semanas de evento e classificadores de fase podem entao aprender
    calendario em vez de mecanismo interanual.

    Colunas fora de ``columns`` retornam inalteradas; os nomes sao preservados
    para nao quebrar consumidores a jusante (a proveniencia deve ser declarada
    nas tabelas de inventario).
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("frame index must be a DatetimeIndex.")
    if columns is None:
        columns = list(frame.columns)
    out = frame.copy()
    index = frame.index
    design = _fourier_design(index.dayofyear.to_numpy(), harmonics)
    if detrend:
        t0 = pd.Timestamp(base[0])
        years = ((index - t0).days.to_numpy(dtype=float)) / 365.2425
        design = np.column_stack([design, years])
    in_base = (index >= pd.Timestamp(base[0])) & (index <= pd.Timestamp(base[1]))
    for name in columns:
        if name not in frame.columns:
            continue
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        fit_mask = in_base & np.isfinite(values)
        minimum = design.shape[1] + 2
        if int(fit_mask.sum()) < minimum:
            raise ValueError(
                f"coluna '{name}': {int(fit_mask.sum())} amostras validas na base; "
                f"minimo {minimum} para {harmonics} harmonicos"
                + (" + tendencia" if detrend else "")
            )
        coef, *_ = np.linalg.lstsq(design[fit_mask], values[fit_mask], rcond=None)
        out[name] = values - design @ coef
    return out


def harmonic_weekly_climatology(
    series: pd.Series,
    *,
    harmonics: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Fit a smooth annual Fourier climatology for weekly analysis.

    The caller is responsible for passing only the training block when the
    result is used in inference or skill estimates. This avoids the noisy
    52-bin week-of-year climatology while preserving the annual cycle.
    """
    if harmonics < 1:
        raise ValueError("harmonics must be at least 1.")
    s = series.sort_index()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("series index must be a DatetimeIndex.")
    observed = s.dropna()
    minimum = 1 + 2 * harmonics
    if observed.size < minimum:
        raise ValueError(f"at least {minimum} finite samples are required.")

    x_fit = _fourier_design(observed.index.dayofyear.to_numpy(), harmonics)
    coef, *_ = np.linalg.lstsq(x_fit, observed.to_numpy(dtype=float), rcond=None)
    x_all = _fourier_design(s.index.dayofyear.to_numpy(), harmonics)
    clim = pd.Series(x_all @ coef, index=s.index, name=f"{s.name}_harmonic_clim")
    anom = (s - clim).rename(f"{s.name}_harmonic_anom")
    return anom, clim
