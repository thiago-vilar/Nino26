"""Utilitários da fase3_nino — Dinâmica e Evolução do El Niño (Pacífico).

Funções compartilhadas pelos notebooks F3N1..F3N7: carregamento dos produtos
F1/F2, normalização semanal, catálogo de eventos, ciclo de vida, correlação
defasada com N efetivo e exportação padronizada (figuras 600 dpi png+pdf e
tabelas numéricas em ``numeric-tables/fase3``).

Quando um insumo real não está disponível localmente, os loaders devolvem
tensores simulados com as dimensões corretas e a coluna/atributo
``dado_simulado`` marcado — nenhum notebook quebra por ausência de dado.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib

if "ipykernel" not in sys.modules:
    # Fora do Jupyter (scripts/testes) não há display; dentro do notebook o
    # backend inline permanece ativo para renderizar as figuras no corpo.
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


CLIM_START = "1991-01-01"
CLIM_END = "2020-12-31"
WEEK_FREQ = "W-SUN"
EARTH_DEG_M = 111_320.0  # metros por grau de longitude no equador


# ---------------------------------------------------------------------------
# Caminhos e exportação
# ---------------------------------------------------------------------------

def project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def fig_dir() -> Path:
    return project_root() / "data/processed/figures/fase3"


def table_dir() -> Path:
    return project_root() / "data/processed/numeric-tables/fase3"


def cache_dir() -> Path:
    return project_root() / "data/processed/parquet/features"


def ensure_dirs() -> None:
    for path in (fig_dir(), table_dir(), cache_dir()):
        path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Exporta a figura oficial em .png a 600 dpi (formato único do projeto)."""
    ensure_dirs()
    path = fig_dir() / f"{name}.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"[figura] {path.name} -> {fig_dir()}")
    return path


def save_table(frame: pd.DataFrame, name: str, *, index: bool = True) -> Path:
    """Exporta a tabela numérica associada a uma figura (regra de ouro)."""
    ensure_dirs()
    path = table_dir() / f"{name}.csv"
    frame.to_csv(path, index=index)
    print(f"[tabela] {path.name} ({frame.shape[0]}x{frame.shape[1]}) -> {table_dir()}")
    return path


# ---------------------------------------------------------------------------
# Carregamento dos produtos F1/F2 (com fallback simulado)
# ---------------------------------------------------------------------------

def _open_zarr(path: Path) -> xr.Dataset | None:
    for kwargs in ({"consolidated": None}, {"consolidated": False}, {}):
        try:
            return xr.open_zarr(path, **kwargs)
        except Exception:
            continue
    return None


def dummy_master_weekly(n_weeks: int = 2376, n_vars: int = 44, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("1981-01-04", periods=n_weeks, freq=WEEK_FREQ, name="week_ending_sunday")
    base = np.cumsum(rng.normal(0, 0.15, n_weeks))
    data = {"nino34_ssta": base - base.mean()}
    for k in range(1, n_vars):
        lag = rng.integers(0, 30)
        noise = rng.normal(0, 1.0, n_weeks)
        data[f"var_simulada_{k:02d}"] = np.roll(data["nino34_ssta"], lag) * rng.uniform(-1, 1) + noise
    frame = pd.DataFrame(data, index=index)
    frame.attrs["dado_simulado"] = True
    return frame


def load_master_weekly() -> pd.DataFrame:
    """Matriz semanal W-SUN da F2 (44 variáveis físicas + ocean_source_code)."""
    zarr_path = project_root() / "data/processed/zarr/features/nino34_master_weekly.zarr"
    csv_path = project_root() / "data/processed/parquet/features/nino34_master_weekly.csv"
    if zarr_path.exists():
        ds = _open_zarr(zarr_path)
        if ds is not None:
            frame = ds.to_dataframe()
            ds.close()
            frame.index = pd.DatetimeIndex(frame.index)
            frame.attrs["dado_simulado"] = False
            return frame.sort_index()
    if csv_path.exists():
        frame = pd.read_csv(csv_path, parse_dates=["week_ending_sunday"]).set_index("week_ending_sunday")
        frame.attrs["dado_simulado"] = False
        return frame.sort_index()
    print("AVISO: master semanal F2 ausente; usando dados simulados com as dimensões esperadas.")
    return dummy_master_weekly()


def _weekly_lon_matrix_from_stores(
    stores: list[Path],
    *,
    variable_candidates: tuple[str, ...],
    lat_band: tuple[float, float],
) -> pd.DataFrame | None:
    parts: list[pd.DataFrame] = []
    for store in stores:
        ds = _open_zarr(store)
        if ds is None:
            continue
        try:
            name = next((v for v in variable_candidates if v in ds), None)
            if name is None or "time" not in ds.coords:
                continue
            array = ds[name]
            lat_name = "lat" if "lat" in array.dims else "latitude"
            lon_name = "lon" if "lon" in array.dims else "longitude"
            lat = ds[lat_name]
            band = array.sel({lat_name: slice(*lat_band)} if float(lat[0]) < float(lat[-1]) else {lat_name: slice(lat_band[1], lat_band[0])})
            equatorial = band.mean(lat_name, skipna=True)
            extra = [d for d in equatorial.dims if d not in ("time", lon_name)]
            if extra:
                equatorial = equatorial.isel({d: 0 for d in extra})
            weekly = equatorial.resample(time=WEEK_FREQ).mean()
            frame = weekly.to_pandas()
            if isinstance(frame, pd.Series):
                frame = frame.to_frame()
            frame.columns = [round(float(c), 3) for c in frame.columns]
            parts.append(frame)
        finally:
            ds.close()
    if not parts:
        return None
    combined = pd.concat(parts).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def load_oisst_equatorial_weekly(*, lat_band: tuple[float, float] = (-2.0, 2.0), refresh: bool = False) -> pd.DataFrame:
    """SST semanal média equatorial por longitude (OISST regridado 0,25°).

    Domínio disponível localmente: 170°W–30°W. Cache em parquet para evitar
    varrer 45 anos a cada execução.
    """
    cache = cache_dir() / "fase3_oisst_sst_equatorial_weekly_by_lon.parquet"
    if cache.exists() and not refresh:
        frame = pd.read_parquet(cache)
        frame.index = pd.DatetimeIndex(frame.index)
        frame.attrs["dado_simulado"] = False
        return frame
    stores = sorted((project_root() / "data/processed/zarr/regridded").glob("noaa_oisst_*.zarr"))
    frame = _weekly_lon_matrix_from_stores(stores, variable_candidates=("sst",), lat_band=lat_band)
    if frame is None:
        print("AVISO: OISST regridado ausente; matriz lon x tempo simulada.")
        return _dummy_lon_matrix(lon_start=-170.0, lon_end=-30.0)
    ensure_dirs()
    frame.columns = [str(c) for c in frame.columns]
    frame.to_parquet(cache)
    frame.columns = [float(c) for c in frame.columns]
    frame.attrs["dado_simulado"] = False
    return frame


def load_ssh_equatorial_weekly(*, lat_band: tuple[float, float] = (-2.0, 2.0), refresh: bool = False) -> pd.DataFrame:
    """SSH semanal médio equatorial por longitude (UFS+GLORYS, 120°E–280°E)."""
    cache = cache_dir() / "fase3_ssh_equatorial_weekly_by_lon.parquet"
    if cache.exists() and not refresh:
        frame = pd.read_parquet(cache)
        frame.index = pd.DatetimeIndex(frame.index)
        frame.attrs["dado_simulado"] = False
        return frame
    root = project_root() / "data/processed/zarr/ocean_daily"
    stores: list[Path] = []
    for source in ("noaa_ufs", "glorys12", "glorys12_operational"):
        stores.extend(sorted((root / source).rglob("*.zarr")))
    frame = _weekly_lon_matrix_from_stores(
        stores, variable_candidates=("sea_surface_height", "zos", "ssh"), lat_band=lat_band
    )
    if frame is None:
        print("AVISO: cubos UFS+GLORYS ausentes; matriz lon x tempo simulada.")
        return _dummy_lon_matrix(lon_start=120.0, lon_end=280.0)
    ensure_dirs()
    frame.columns = [str(c) for c in frame.columns]
    frame.to_parquet(cache)
    frame.columns = [float(c) for c in frame.columns]
    frame.attrs["dado_simulado"] = False
    return frame


def _dummy_lon_matrix(*, lon_start: float, lon_end: float, n_weeks: int = 2300, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lons = np.arange(lon_start, lon_end + 0.25, 1.0)
    index = pd.date_range("1981-09-06", periods=n_weeks, freq=WEEK_FREQ)
    time = np.arange(n_weeks)[:, None]
    phase = (lons[None, :] - lons[0]) / 30.0
    values = np.sin(2 * np.pi * (time / 52.0) - phase) + rng.normal(0, 0.3, (n_weeks, lons.size))
    frame = pd.DataFrame(values, index=index, columns=lons)
    frame.attrs["dado_simulado"] = True
    return frame


# ---------------------------------------------------------------------------
# Normalização (Etapa 4.1)
# ---------------------------------------------------------------------------

def weekly_anomaly(series: pd.Series, *, clim_start: str = CLIM_START, clim_end: str = CLIM_END, smooth: int = 5) -> pd.Series:
    """Anomalia semanal: remove a climatologia por semana-do-ano (1..53).

    A climatologia é ajustada apenas no período de referência (1991–2020,
    OMM) e suavizada circularmente para reduzir o ruído de 53 caixas.
    """
    values = pd.to_numeric(series, errors="coerce")
    if not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError("weekly_anomaly requer DatetimeIndex")
    base = values.loc[clim_start:clim_end]
    if base.notna().sum() < 104:
        base = values
    week = base.index.isocalendar().week.astype(int).clip(upper=53)
    climatology = base.groupby(week.values).mean().reindex(range(1, 54)).interpolate(limit_direction="both")
    pad = smooth // 2
    circular = pd.concat([climatology.iloc[-pad:], climatology, climatology.iloc[:pad]])
    climatology = circular.rolling(smooth, center=True, min_periods=1).mean().iloc[pad:-pad]
    all_week = values.index.isocalendar().week.astype(int).clip(upper=53)
    mapped = pd.Series(all_week.map(climatology).to_numpy(dtype=float), index=values.index)
    return values - mapped


def zscore(series: pd.Series, *, clim_start: str = CLIM_START, clim_end: str = CLIM_END) -> pd.Series:
    """Z-score com média/desvio do período climatológico de referência."""
    values = pd.to_numeric(series, errors="coerce")
    base = values.loc[clim_start:clim_end]
    if base.notna().sum() < 52:
        base = values
    sigma = float(base.std(ddof=1))
    if not np.isfinite(sigma) or sigma == 0:
        return values * np.nan
    return (values - float(base.mean())) / sigma


def deseason_and_zscore(frame: pd.DataFrame, *, already_anomaly_suffix: str = "_anom") -> pd.DataFrame:
    """Aplica anomalização semanal + z-score a todas as colunas físicas.

    Colunas ``*_anom`` e ``nino34_ssta`` já são anomalias (F2); recebem apenas
    o z-score. As demais (oceânicas absolutas) passam antes pela remoção da
    sazonalidade semanal.
    """
    out = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        if column == "ocean_source_code":
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if column.endswith(already_anomaly_suffix) or column == "nino34_ssta":
            out[column] = zscore(series)
        else:
            out[column] = zscore(weekly_anomaly(series))
    return out


# ---------------------------------------------------------------------------
# Eventos e ciclo de vida (Etapas 4.2 e 4.3)
# ---------------------------------------------------------------------------

def smooth_ssta(ssta: pd.Series, window_weeks: int = 13) -> pd.Series:
    """Média móvel centrada de ~3 meses (adaptação semanal do ONI)."""
    return pd.to_numeric(ssta, errors="coerce").rolling(window_weeks, center=True, min_periods=window_weeks // 2).mean()


def detect_el_nino_events(
    ssta: pd.Series,
    *,
    smooth_weeks: int = 13,
    threshold_c: float = 0.5,
    min_duration_weeks: int = 22,
) -> pd.DataFrame:
    """Catálogo de eventos El Niño semanais (Trenberth 1997 adaptado).

    Evento: SSTA suavizada (média móvel centrada de ``smooth_weeks``)
    ≥ ``threshold_c`` por pelo menos ``min_duration_weeks`` semanas
    consecutivas (~5 meses, equivalente às 5 estações sobrepostas do ONI).
    """
    smooth = smooth_ssta(ssta, smooth_weeks)
    above = smooth >= threshold_c
    group = (above != above.shift(fill_value=False)).cumsum()
    rows = []
    for _, block in smooth[above].groupby(group[above]):
        if len(block) < min_duration_weeks:
            continue
        peak_week = block.idxmax()
        peak = float(block.max())
        rows.append(
            {
                "inicio": block.index[0].date(),
                "fim": block.index[-1].date(),
                "duracao_semanas": int(len(block)),
                "semana_pico": peak_week.date(),
                "ssta_pico_c": round(peak, 3),
                "classe": _event_class(peak),
            }
        )
    catalog = pd.DataFrame(rows)
    if not catalog.empty:
        catalog.insert(0, "evento", [f"EN_{pd.Timestamp(r).year}" for r in catalog["semana_pico"]])
    return catalog


def _event_class(peak_c: float) -> str:
    if peak_c >= 2.0:
        return "muito_forte"
    if peak_c >= 1.5:
        return "forte"
    if peak_c >= 1.0:
        return "moderado"
    return "fraco"


def classify_life_cycle(
    ssta: pd.Series,
    catalog: pd.DataFrame,
    *,
    smooth_weeks: int = 13,
    peak_fraction: float = 0.90,
    genesis_max_weeks: int = 26,
) -> pd.Series:
    """Fase do ciclo de vida por semana: genese, crescimento, faixa_pico, decaimento.

    Faixa de pico: patamar contínuo com SSTA suavizada ≥ ``peak_fraction`` do
    máximo do evento (documentação do projeto). Gênese: até 26 semanas antes
    do cruzamento de +0,5 °C, desde o último mínimo local (recarga, Jin 1997).
    """
    smooth = smooth_ssta(ssta, smooth_weeks)
    phase = pd.Series("neutro", index=smooth.index, dtype="object")
    for _, event in catalog.iterrows():
        start = pd.Timestamp(event["inicio"])
        end = pd.Timestamp(event["fim"])
        segment = smooth.loc[start:end]
        if segment.empty:
            continue
        peak_value = float(segment.max())
        plateau = segment[segment >= peak_fraction * peak_value]
        peak_start, peak_end = plateau.index[0], plateau.index[-1]
        phase.loc[start:peak_start - pd.Timedelta(weeks=1)] = "crescimento"
        phase.loc[peak_start:peak_end] = "faixa_pico"
        phase.loc[peak_end + pd.Timedelta(weeks=1):end] = "decaimento"
        pre = smooth.loc[:start - pd.Timedelta(weeks=1)].tail(genesis_max_weeks)
        if len(pre):
            minimum = pre.idxmin()
            phase.loc[minimum:start - pd.Timedelta(weeks=1)] = "genese"
    return phase


# ---------------------------------------------------------------------------
# Estatística (Etapa 4.4)
# ---------------------------------------------------------------------------

def effective_sample_size(x: pd.Series, y: pd.Series) -> float:
    """N efetivo de Bretherton et al. (1999): N (1-r1 r2)/(1+r1 r2)."""
    paired = pd.concat([x, y], axis=1).dropna()
    n = len(paired)
    if n < 4:
        return float(n)
    r1 = paired.iloc[:, 0].autocorr(1)
    r2 = paired.iloc[:, 1].autocorr(1)
    if not (np.isfinite(r1) and np.isfinite(r2)):
        return float(n)
    factor = (1 - r1 * r2) / (1 + r1 * r2)
    return float(max(4.0, n * factor))


def correlation_p_value(r: float, n_eff: float) -> float:
    from scipy import stats

    if not np.isfinite(r) or n_eff <= 3 or abs(r) >= 1:
        return np.nan
    t = r * np.sqrt((n_eff - 2) / (1 - r**2))
    return float(2 * stats.t.sf(abs(t), df=n_eff - 2))


def lagged_correlations(
    predictors: pd.DataFrame,
    target: pd.Series,
    *,
    lags_weeks: list[int],
    method: str = "pearson",
) -> pd.DataFrame:
    """Correlação preditor(t-lag) x alvo(t) com N efetivo e p-valor."""
    from scipy import stats

    rows = []
    for column in predictors.columns:
        series = pd.to_numeric(predictors[column], errors="coerce")
        for lag in lags_weeks:
            shifted = series.shift(lag)
            paired = pd.concat([shifted, target], axis=1).dropna()
            if len(paired) < 30:
                rows.append({"variavel": column, "lag_semanas": lag, "r": np.nan, "n": len(paired), "n_eff": np.nan, "p_valor": np.nan})
                continue
            if method == "spearman":
                r = float(stats.spearmanr(paired.iloc[:, 0], paired.iloc[:, 1]).statistic)
            else:
                r = float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
            n_eff = effective_sample_size(paired.iloc[:, 0], paired.iloc[:, 1])
            rows.append(
                {
                    "variavel": column,
                    "lag_semanas": lag,
                    "r": r,
                    "n": int(len(paired)),
                    "n_eff": round(n_eff, 1),
                    "p_valor": correlation_p_value(r, n_eff),
                }
            )
    return pd.DataFrame(rows)


def pca_from_zscores(frame: pd.DataFrame, *, variance_target: float = 0.80):
    """PCA por SVD nos z-scores (linhas completas); tenta o pacote eofs antes.

    Devolve (scores, loadings, variancia_explicada, n_componentes).
    """
    filled = frame.dropna()
    matrix = filled.to_numpy(dtype=float)
    matrix = matrix - matrix.mean(axis=0)
    try:  # pacote recomendado no prompt; fallback numpy é matematicamente idêntico
        from eofs.standard import Eof

        solver = Eof(matrix)
        variance = solver.varianceFraction()
        n_keep = int(np.searchsorted(np.cumsum(variance), variance_target) + 1)
        scores = solver.pcs(npcs=n_keep)
        loadings = solver.eofs(neofs=n_keep).T
    except Exception:
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        variance = (s**2) / float((s**2).sum())
        n_keep = int(np.searchsorted(np.cumsum(variance), variance_target) + 1)
        scores = u[:, :n_keep] * s[:n_keep]
        loadings = vt[:n_keep].T
    score_frame = pd.DataFrame(scores, index=filled.index, columns=[f"PC{i+1}" for i in range(n_keep)])
    loading_frame = pd.DataFrame(loadings, index=filled.columns, columns=score_frame.columns)
    return score_frame, loading_frame, pd.Series(variance[:n_keep], index=score_frame.columns, name="fracao_variancia"), n_keep


# ---------------------------------------------------------------------------
# Kelvin / Hovmöller (Etapas 4.6 e 4.7)
# ---------------------------------------------------------------------------

def lon_anomaly_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Anomalia semanal (clim. por semana-do-ano) coluna a coluna (longitude)."""
    return pd.DataFrame({column: weekly_anomaly(matrix[column]) for column in matrix.columns}, index=matrix.index)


def kelvin_bandpass(anomaly: pd.DataFrame, *, smooth_deg: float = 10.0) -> pd.DataFrame:
    """Isola a assinatura de Kelvin: remove a média zonal instantânea e
    suaviza em longitude (~10°) para reter escalas de milhares de km."""
    demeaned = anomaly.sub(anomaly.mean(axis=1), axis=0)
    lons = np.array([float(c) for c in anomaly.columns])
    step = float(np.median(np.diff(np.sort(lons)))) or 1.0
    window = max(3, int(round(smooth_deg / step)))
    smoothed = demeaned.T.rolling(window, center=True, min_periods=1).mean().T
    return smoothed


def phase_speed_from_lags(
    signal: pd.DataFrame,
    *,
    reference_lon: float,
    max_lag_weeks: int = 10,
) -> pd.DataFrame:
    """Velocidade de fase por correlação defasada entre longitudes.

    Para cada longitude, encontra o lag (semanas) que maximiza a correlação
    com a longitude de referência; a regressão lag x distância dá a
    velocidade média de propagação (m/s).
    """
    lons = np.array([float(c) for c in signal.columns])
    reference = signal[signal.columns[int(np.argmin(np.abs(lons - reference_lon)))]]
    rows = []
    for column in signal.columns:
        lon = float(column)
        best_lag, best_r = 0, -np.inf
        for lag in range(0, max_lag_weeks + 1):
            r = reference.corr(signal[column].shift(-lag))
            if np.isfinite(r) and r > best_r:
                best_lag, best_r = lag, float(r)
        rows.append({"lon": lon, "distancia_km": (lon - reference_lon) * EARTH_DEG_M / 1000.0, "lag_otimo_semanas": best_lag, "r_max": best_r})
    return pd.DataFrame(rows)


def fit_phase_speed(lags: pd.DataFrame, *, min_r: float = 0.3) -> dict[str, float]:
    valid = lags[(lags["r_max"] >= min_r) & (lags["distancia_km"] > 0) & (lags["lag_otimo_semanas"] > 0)]
    if len(valid) < 3:
        return {"velocidade_m_s": np.nan, "n_pontos": int(len(valid)), "r2": np.nan}
    x = valid["lag_otimo_semanas"].to_numpy(dtype=float) * 7 * 86400.0
    y = valid["distancia_km"].to_numpy(dtype=float) * 1000.0
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(((y - predicted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "velocidade_m_s": float(slope),
        "n_pontos": int(len(valid)),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
    }
