"""fase4_features.py
================================================================================
Fase 4 - Montagem da matriz ampla de features e utilidades estatisticas.

Objetivo deste modulo (usado pelos notebooks B e C da Fase 4):
  * Carregar a espinha dorsal diaria ja calculada (SST Nino 3.4 + features
    oceanicas) de ``data/processed/parquet/physics_precalc_timeseries.csv``.
  * Construir indices atmosfericos ERA5 (single e pressure levels) na caixa
    Nino 3.4, em media diaria, preservando valores brutos para que anomalias
    sejam fitadas dentro de cada fold.
  * Montar UMA matriz diaria ampla com TODAS as variaveis processadas; a Fase 4
    estatistica deriva dela o eixo semanal canonico de 7 dias.
  * Definir blocos fisicos e reduzir colinearidade por bloco.
  * Fornecer split temporal com embargo e padronizacao z-score intra-fold e
    "source-aware" (respeitando a transicao de fonte oceanica do projeto).

Decisoes alinhadas a configs/project.yaml:
  - climatology_window_days: 15
  - eixo de analise: semanal de 7 dias; diario fica como insumo bruto
  - lag_days: 7..168 (passo 7) ; janelas de rolagem 7/30/90/180
  - climatologia semanal: 2-3 harmonicos anuais fitados apenas no treino
  - diario e mensal nunca sao misturados por interpolacao
  - lacunas nao sao preenchidas em silencio (ficam NaN auditavel)
  - oceano: media/desvio por fonte (ocean_source_code 1=UFS,2=GLORYS,3=GLO12)
  - significancia: N efetivo por autocorrelacao e FDR de Benjamini-Hochberg

Requer: numpy, pandas, xarray, zarr, scipy, scikit-learn.
"""
from __future__ import annotations

import glob
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from nino_brasil.stats.significance import (
    benjamini_hochberg_fdr,
    correlation_p_value,
    effective_sample_size,
)

# --------------------------------------------------------------------------- #
# Localizacao do projeto                                                       #
# --------------------------------------------------------------------------- #
def find_root(start: Path | str | None = None) -> Path:
    """Sobe diretorios ate achar pyproject.toml (raiz do projeto)."""
    p = Path(start or Path.cwd()).resolve()
    while p != p.parent and not (p / "pyproject.toml").exists():
        p = p.parent
    return p


ROOT = find_root()
DATA = ROOT / "data"
PROC = DATA / "processed"
ZARR = PROC / "zarr"
STATS = PROC / "parquet" / "statistics"
PHYSICS_CSV = PROC / "parquet" / "physics_precalc_timeseries.csv"
PEAK_CSV = PROC / "parquet" / "features" / "nino34_oisst_event_reference.csv"
PROGRESSION_CSV = PROC / "parquet" / "modeling" / "enso_peak_progression.csv"

# --------------------------------------------------------------------------- #
# Constantes do projeto                                                        #
# --------------------------------------------------------------------------- #
CLIM_BASE = ("1991-01-01", "2020-12-31")  # referencia descritiva; analises futuras devem ajustar no treino
CLIM_WINDOW = 15                          # janela day-of-year (+- dias)
LAGS = list(range(7, 169, 7))             # 7..168 (config lag_days)
ROLL_WINDOWS = (7, 30, 90, 180)           # janelas de media/delta

NINO34_BOX = {"lat": (-5.0, 5.0), "lon": (-170.0, -120.0)}
BRAZIL_BOX = {"lat": (-35.0, 7.0), "lon": (-75.0, -30.0)}
ATL4_BOX = {"lat": (-3.0, 3.0), "lon": (-50.0, -25.0)}
ATL3_BOX = {"lat": (-3.0, 3.0), "lon": (-20.0, 0.0)}
TNA_BOX = {"lat": (5.5, 23.5), "lon": (-57.5, -15.0)}
TSA_BOX = {"lat": (-20.0, 0.0), "lon": (-30.0, 10.0)}
IOD_WEST_BOX = {"lat": (-10.0, 10.0), "lon": (50.0, 70.0)}
IOD_EAST_BOX = {"lat": (-10.0, 0.0), "lon": (90.0, 110.0)}

ERA5_SINGLE_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_latent_heat_flux",
    "surface_sensible_heat_flux",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "total_column_water_vapour",
]
ERA5_PRESSURE_VARS = [
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
    "vertical_velocity",
    "divergence",
]
PRESSURE_LEVELS = (850, 500, 200)

# colunas oceanicas-base ja presentes no physics backbone (sem derivadas)
OCEAN_BASE_VARS = [
    "d20_nino34_mean_m",
    "ohc_0_100_nino34_j_m2", "ohc_0_300_nino34_j_m2",
    "ohc_0_700_nino34_j_m2", "ohc_300_700_nino34_j_m2",
    "ssh_nino34_mean_m", "sss_nino34_mean",
    "temperature_50m_nino34_c", "temperature_100m_nino34_c",
    "temperature_150m_nino34_c", "temperature_200m_nino34_c",
    "temperature_300m_nino34_c", "temperature_500m_nino34_c",
    "temperature_700m_nino34_c",
    "thermocline_tilt_m", "thermocline_tilt_slope_m_per_degree",
    "wwv_equatorial_pacific_m3",
]


# --------------------------------------------------------------------------- #
# Leitura ERA5 -> indice diario na caixa Nino 3.4                             #
# --------------------------------------------------------------------------- #
def _era5_glob(kind: str, region: str, var: str, year: int) -> list[str]:
    sub = "single_levels" if kind == "single" else "pressure_levels"
    tag = "single" if kind == "single" else "pressure"
    pat = (
        ZARR / "era5" / sub / str(year) / var
        / f"era5_{tag}_{region}_{var}_{year}_daily.zarr"
    )
    return sorted(glob.glob(str(pat)))


def era5_box_index(
    var: str,
    kind: str = "single",
    region: str = "nino34",
    level: int | None = None,
    years: Iterable[int] = range(1981, 2026),
    reducer: str = "mean",
) -> pd.Series:
    """Media (ou desvio) espacial diaria de uma variavel ERA5 na caixa.

    Para pressure levels e obrigatorio informar ``level`` (850/500/200 hPa).
    Retorna uma Series diaria indexada por tempo. Anos ausentes sao ignorados
    (a lacuna aparece como ausencia de datas, sem preenchimento).
    """
    import xarray as xr

    pieces: list[pd.Series] = []
    for y in years:
        stores = _era5_glob(kind, region, var, int(y))
        if not stores:
            continue
        ds = xr.open_zarr(stores[0])
        da = ds[var]
        if "pressure_level" in da.dims:
            if level is None:
                raise ValueError(f"{var}: informe level para pressure_levels")
            da = da.sel(pressure_level=level)
        if "number" in da.dims:
            da = da.isel(number=0, drop=True)
        spatial = [d for d in da.dims if d in ("latitude", "longitude")]
        red = da.std(dim=spatial) if reducer == "std" else da.mean(dim=spatial)
        s = red.to_series()
        pieces.append(s)
        ds.close()
    if not pieces:
        return pd.Series(dtype="float64", name=var)
    out = pd.concat(pieces).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out.index = pd.to_datetime(out.index).normalize()
    out.name = f"{kind}_{var}" + (f"_{level}" if level else "")
    return out


def build_era5_nino34_indices(
    years: Iterable[int] = range(1981, 2026),
    levels: Sequence[int] = (850, 200),
) -> pd.DataFrame:
    """Constroi todos os indices atmosfericos Nino 3.4 (single + pressure).

    Single levels: um indice por variavel.
    Pressure levels: um indice por variavel x nivel (default 850 e 200 hPa,
    que capturam baixa e alta troposfera - circulacao de Walker).
    """
    cols: dict[str, pd.Series] = {}
    for v in ERA5_SINGLE_VARS:
        s = era5_box_index(v, kind="single", region="nino34", years=years)
        if len(s):
            cols[f"atm_{v}"] = s
    for v in ERA5_PRESSURE_VARS:
        for lev in levels:
            s = era5_box_index(v, kind="pressure", region="nino34",
                               level=lev, years=years)
            if len(s):
                cols[f"atm_{v}_{lev}hpa"] = s
    if not cols:
        return pd.DataFrame()
    df = pd.concat(cols, axis=1)
    df.index.name = "time"
    return df


# --------------------------------------------------------------------------- #
# Climatologia / anomalia (janela de 15 dias no day-of-year)                  #
# --------------------------------------------------------------------------- #
def doy_climatology(
    s: pd.Series, base: tuple[str, str] = CLIM_BASE, window: int = CLIM_WINDOW
) -> tuple[pd.Series, pd.Series]:
    """Anomalia e climatologia suave por dia-do-ano (janela circular +-window).

    A climatologia deve ser estimada no treino de cada fold. O ``base`` default
    e apenas uma referencia descritiva 1991-2020 para graficos sem claim de
    skill preditivo. Retorna (anomalia, climatologia).
    """
    s = s.sort_index()
    mask = (s.index >= pd.Timestamp(base[0])) & (s.index <= pd.Timestamp(base[1]))
    base_s = s[mask].dropna()
    if base_s.empty:
        base_s = s.dropna()
    dvals = base_s.index.dayofyear.values
    vals = base_s.values.astype("float64")
    clim_arr = np.full(367, np.nan)
    for d in range(1, 367):
        diff = np.abs(dvals - d)
        diff = np.minimum(diff, 366 - diff)
        m = diff <= window
        if m.any():
            clim_arr[d] = np.nanmean(vals[m])
    full_doy = s.index.dayofyear.values
    clim = pd.Series(clim_arr[full_doy], index=s.index, name=f"{s.name}_clim")
    anom = (s - clim).rename(f"{s.name}_anom")
    return anom, clim


def standardized_anomaly(
    s: pd.Series, base: tuple[str, str] = CLIM_BASE, window: int = CLIM_WINDOW
) -> pd.Series:
    """Anomalia padronizada (dividida pelo desvio sazonal do dia-do-ano).

    Util para variaveis com variancia fortemente sazonal (fluxos, precip).
    """
    anom, _ = doy_climatology(s, base, window)
    s2 = s.sort_index()
    mask = (s2.index >= pd.Timestamp(base[0])) & (s2.index <= pd.Timestamp(base[1]))
    base_s = s2[mask].dropna()
    dvals = base_s.index.dayofyear.values
    vals = base_s.values.astype("float64")
    std_arr = np.full(367, np.nan)
    for d in range(1, 367):
        diff = np.abs(dvals - d)
        diff = np.minimum(diff, 366 - diff)
        m = diff <= window
        if m.sum() > 3:
            std_arr[d] = np.nanstd(vals[m])
    full_doy = s2.index.dayofyear.values
    std = pd.Series(std_arr[full_doy], index=s2.index)
    std = std.replace(0, np.nan)
    return (anom / std).rename(f"{s.name}_zanom")


# --------------------------------------------------------------------------- #
# Derivadas temporais (media/delta) consistentes com o backbone existente     #
# --------------------------------------------------------------------------- #
def add_roll_delta(
    df: pd.DataFrame, cols: Sequence[str], windows: Sequence[int] = ROLL_WINDOWS
) -> pd.DataFrame:
    """Adiciona ``<col>_mean_<w>d`` e ``<col>_delta_<w>d`` para cada janela."""
    out: dict[str, pd.Series] = {}
    for c in cols:
        x = df[c]
        for w in windows:
            out[f"{c}_mean_{w}d"] = x.rolling(w, min_periods=max(2, w // 2)).mean()
            out[f"{c}_delta_{w}d"] = x - x.shift(w)
    extra = pd.DataFrame(out, index=df.index)
    return pd.concat([df, extra], axis=1)


# --------------------------------------------------------------------------- #
# Backbone + montagem da matriz ampla                                         #
# --------------------------------------------------------------------------- #
def load_physics_backbone(base: str = "stored") -> pd.DataFrame:
    """Carrega o backbone diario (SST Nino 3.4 + features oceanicas + lags).

    ``base="stored"`` (padrao) mantem a anomalia pre-calculada. Use
    ``base="reference"`` apenas para produtos descritivos 1991-2020. O modo
    legado ``base="full"`` e aceito como alias descritivo e emite aviso; nao
    deve ser usado para ranking, CV ou skill.
    """
    df = pd.read_csv(PHYSICS_CSV, parse_dates=["time"]).set_index("time").sort_index()
    df.index = df.index.normalize()
    if base == "full":
        warnings.warn(
            "base='full' e descritivo/legado; use climatologia fitada no treino "
            "do fold para qualquer avaliacao inferencial ou preditiva.",
            RuntimeWarning,
        )
        base = "reference"
    if base == "reference" and "nino34_sst" in df.columns:
        anom, _ = doy_climatology(df["nino34_sst"], base=CLIM_BASE, window=CLIM_WINDOW)
        df["nino34_ssta"] = anom.values
        for w in ROLL_WINDOWS:
            df[f"nino34_ssta_mean_{w}d"] = anom.rolling(
                w, min_periods=max(2, w // 2)).mean().values
            df[f"nino34_ssta_delta_{w}d"] = (anom - anom.shift(w)).values
        df.attrs["climatology_base"] = f"{CLIM_BASE[0]}:{CLIM_BASE[1]} (referencia descritiva)"
    return df


def assemble_feature_matrix(
    years: Iterable[int] = range(1981, 2026),
    atmosphere: bool = True,
    save: bool = True,
    out_name: str = "phase4_feature_matrix_daily.parquet",
    anomaly_mode: str = "raw",
) -> pd.DataFrame:
    """Une backbone (SST+oceano) + atmosfera ERA5 (anomalia + lags) numa matriz.

    Por padrao, a atmosfera entra como indice bruto diario com medias/deltas
    7/30/90/180 d. Use ``anomaly_mode="reference"`` apenas para produto
    descritivo; ranking/skill devem recalcular anomalia dentro de cada fold.
    Lacunas ficam NaN.
    Salva em data/processed/parquet/modeling/<out_name>.
    """
    if anomaly_mode not in {"raw", "reference"}:
        raise ValueError("anomaly_mode must be 'raw' or 'reference'.")
    base = load_physics_backbone()
    parts = [base]
    if atmosphere:
        atm = build_era5_nino34_indices(years=years)
        if not atm.empty:
            if anomaly_mode == "reference":
                warnings.warn(
                    "Atmospheric anomalies with anomaly_mode='reference' are "
                    "descriptive only; fold-safe analyses must fit climatology "
                    "inside each training split.",
                    RuntimeWarning,
                )
                cols = {}
                for c in atm.columns:
                    a, _ = doy_climatology(atm[c])
                    cols[f"{c}_anom"] = a
                atm_features = pd.DataFrame(cols, index=atm.index)
            else:
                atm_features = atm.add_prefix("raw_")
            atm_features = add_roll_delta(atm_features, list(atm_features.columns))
            atm_features = atm_features.reindex(base.index)
            parts.append(atm_features)
    matrix = pd.concat(parts, axis=1)
    matrix = matrix.loc[:, ~matrix.columns.duplicated()]
    if save:
        out = PROC / "parquet" / "modeling" / out_name
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            matrix.to_parquet(out)
        except Exception as exc:  # pragma: no cover - fallback p/ ambiente sem pyarrow
            warnings.warn(f"parquet falhou ({exc}); salvando CSV")
            matrix.to_csv(out.with_suffix(".csv"))
    return matrix


# --------------------------------------------------------------------------- #
# Blocos fisicos                                                              #
# --------------------------------------------------------------------------- #
def assign_block(col: str) -> str:
    """Classifica uma coluna num bloco fisico (para reduzir colinearidade)."""
    c = col.lower()
    if c.startswith("nino34_sst") or "ssta" in c or c == "nino34_anom_c":
        return "sst"
    if any(k in c for k in ("ohc_", "temperature_", "d20", "wwv")):
        return "ocean_heat"
    if "thermocline_tilt" in c:
        return "thermocline_tilt"
    if "ssh_" in c:
        return "sea_level"
    if "sss_" in c or "salinity" in c:
        return "salinity"
    if "u_component" in c or "v_component" in c or "u10" in c or "v10" in c \
            or "wind" in c:
        return "wind"
    if "mean_sea_level_pressure" in c or "geopotential" in c:
        return "pressure"
    if "vertical_velocity" in c or "divergence" in c \
            or "total_column_water_vapour" in c or "specific_humidity" in c:
        return "convection"
    if "heat_flux" in c or "radiation" in c:
        return "heat_flux"
    return "other"


def block_table(feature_cols: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"feature": list(feature_cols),
         "block": [assign_block(c) for c in feature_cols]}
    )


# --------------------------------------------------------------------------- #
# Reducao de colinearidade por bloco (cluster por correlacao)                 #
# --------------------------------------------------------------------------- #
def reduce_collinearity_by_block(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    threshold: float = 0.9,
    target: pd.Series | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Dentro de cada bloco, agrupa colunas com |corr| alta e mantem 1 medoide.

    Se ``target`` for dado, o representante do cluster e a coluna de maior
    |corr| com o alvo; senao, a de menor redundancia media. Retorna a lista de
    colunas mantidas e um mapa feature->cluster->mantida.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    kept: list[str] = []
    rows = []
    blocks = block_table(feature_cols)
    for blk, grp in blocks.groupby("block"):
        cols = [c for c in grp["feature"] if c in df.columns]
        sub = df[cols].dropna(how="all", axis=1)
        cols = list(sub.columns)
        if len(cols) <= 1:
            kept.extend(cols)
            for c in cols:
                rows.append((c, blk, 1, c))
            continue
        corr = sub.corr().abs().fillna(0.0)
        dist = 1.0 - corr.values
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0
        try:
            Z = linkage(squareform(dist, checks=False), method="average")
            labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
        except Exception:
            labels = np.arange(len(cols)) + 1
        for cl in np.unique(labels):
            members = [cols[i] for i in range(len(cols)) if labels[i] == cl]
            if target is not None:
                tcorr = {m: abs(df[m].corr(target)) for m in members}
                rep = max(members, key=lambda m: (tcorr[m] if pd.notna(tcorr[m]) else -1))
            else:
                mean_red = {m: corr.loc[m, members].mean() for m in members}
                rep = min(members, key=lambda m: mean_red[m])
            kept.append(rep)
            for m in members:
                rows.append((m, blk, int(cl), rep))
    mapping = pd.DataFrame(rows, columns=["feature", "block", "cluster", "kept"])
    return kept, mapping


# --------------------------------------------------------------------------- #
# Validacao temporal com embargo + padronizacao intra-fold                    #
# --------------------------------------------------------------------------- #
class TemporalBlockSplit:
    """Split temporal expansivo com embargo (gap) entre treino e teste.

    Evita vazamento por autocorrelacao: o embargo (em dias) deve ser >= ao
    maior lag usado (default 168 d).
    """

    def __init__(self, n_splits: int = 5, embargo_days: int = 168):
        self.n_splits = n_splits
        self.embargo_days = embargo_days

    def split(self, times: pd.DatetimeIndex):
        times = pd.DatetimeIndex(times)
        order = np.argsort(times.values)
        n = len(times)
        fold = n // (self.n_splits + 1)
        emb = pd.Timedelta(days=self.embargo_days)
        for k in range(1, self.n_splits + 1):
            tr_end = fold * k
            train_idx = order[:tr_end]
            cut = times[order[tr_end - 1]] + emb
            test_mask = times[order] > cut
            test_idx = order[test_mask][: fold]
            if len(test_idx) == 0:
                continue
            yield train_idx, test_idx


def fold_zscore(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: Sequence[str],
    source_col: str | None = "ocean_source_code",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Z-score ajustado SO no treino. Se ``source_col`` existir, ajusta media
    e desvio por fonte (source-aware) para colunas oceanicas; o restante usa
    estatistica global do treino."""
    tr, te = train.copy(), test.copy()
    use_source = source_col is not None and source_col in train.columns
    for c in cols:
        if c not in tr.columns:
            continue
        is_ocean = any(k in c for k in ("ohc_", "temperature_", "d20", "wwv",
                                        "ssh_", "sss_", "thermocline"))
        if use_source and is_ocean:
            for src, g in tr.groupby(source_col):
                mu, sd = g[c].mean(), g[c].std()
                sd = sd if sd and not np.isnan(sd) else 1.0
                m_tr = tr[source_col] == src
                m_te = te[source_col] == src
                tr.loc[m_tr, c] = (tr.loc[m_tr, c] - mu) / sd
                te.loc[m_te, c] = (te.loc[m_te, c] - mu) / sd
        else:
            mu, sd = tr[c].mean(), tr[c].std()
            sd = sd if sd and not np.isnan(sd) else 1.0
            tr[c] = (tr[c] - mu) / sd
            te[c] = (te[c] - mu) / sd
    return tr, te


# --------------------------------------------------------------------------- #
# Cross-correlation defasada (precursores)                                    #
# --------------------------------------------------------------------------- #
def daily_percentile_thresholds(
    s: pd.Series, qs: Sequence[float] = (0.90, 0.95)
) -> dict[float, float]:
    """Limiares globais de percentil da SSTA diaria (ex.: p90, p95).

    Mesma logica do limiar p90 ja gravado no backbone (percentil global da
    distribuicao diaria de SSTA), agora generalizado para varios niveis.
    """
    x = s.dropna()
    return {q: float(x.quantile(q)) for q in qs}


def severity_classify(
    s: pd.Series, thresholds: dict[float, float]
) -> pd.DataFrame:
    """Classifica cada dia por severidade segundo limiares crescentes.

    Retorna colunas: above_p90/above_p95 (0/1), duration_ge_p90/p95_days
    (dias consecutivos acima do limiar) e severity_level (0=neutro, 1>=p90,
    2>=p95). Permite ler a *forca* e a *evolucao* do sinal.
    """
    th = dict(sorted(thresholds.items()))
    out = pd.DataFrame(index=s.index)
    sev = pd.Series(0, index=s.index, dtype="int64")
    for q, t in th.items():
        tag = f"p{int(round(q * 100))}"
        above = (s >= t).astype("int64")
        out[f"above_{tag}"] = above
        sev = sev + above
        grp = (above != above.shift()).cumsum()
        out[f"duration_ge_{tag}_days"] = above.groupby(grp).cumsum() * above
    out["severity_level"] = sev
    return out


def detrend_series(anom: pd.Series) -> pd.Series:
    """Remove tendencia linear (minimos quadrados) de uma serie de anomalia.

    Alternativa *trend-aware* a base-fixa: usar quando comparar a FORCA de
    picos entre decadas diferentes, removendo o aquecimento de fundo que a base
    1991-2020 fixa deixa na serie.
    """
    idx = anom.index
    x = (idx - idx[0]).days.to_numpy(dtype="float64")
    y = anom.to_numpy(dtype="float64")
    m = np.isfinite(y)
    if m.sum() < 10:
        return anom
    b = np.polyfit(x[m], y[m], 1)
    return (anom - (b[0] * x + b[1])).rename(f"{anom.name}_detrended")


def lagged_correlation(
    feature: pd.Series, target: pd.Series, lags: Sequence[int] = LAGS,
    method: str = "spearman",
) -> pd.DataFrame:
    """Correlacao da feature em t-lag contra o alvo em t, para cada lag.

    lag>0 significa feature ANTECEDE o alvo (precursor). Retorna DataFrame
    com colunas: lag, corr, n, n_eff e p_effective.
    """
    df = pd.concat({"f": feature, "y": target}, axis=1).dropna()
    rows = []
    for L in lags:
        shifted = df["f"].shift(L)
        pair = pd.concat([shifted, df["y"]], axis=1).dropna()
        if len(pair) > 30:
            r = pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)
            n_eff = effective_sample_size(pair.iloc[:, 0], pair.iloc[:, 1])
            p_eff = correlation_p_value(float(r), n_eff)
        else:
            r = np.nan
            n_eff = np.nan
            p_eff = np.nan
        rows.append((L, r, len(pair), n_eff, p_eff))
    return pd.DataFrame(rows, columns=["lag", "corr", "n", "n_eff", "p_effective"])


__all__ = [
    "ROOT", "DATA", "PROC", "ZARR", "STATS", "PHYSICS_CSV", "PEAK_CSV",
    "PROGRESSION_CSV", "CLIM_BASE", "CLIM_WINDOW", "LAGS", "ROLL_WINDOWS",
    "NINO34_BOX", "BRAZIL_BOX", "ATL4_BOX", "ATL3_BOX", "TNA_BOX", "TSA_BOX",
    "IOD_WEST_BOX", "IOD_EAST_BOX", "ERA5_SINGLE_VARS", "ERA5_PRESSURE_VARS",
    "OCEAN_BASE_VARS", "find_root", "era5_box_index",
    "build_era5_nino34_indices", "doy_climatology", "standardized_anomaly",
    "add_roll_delta", "load_physics_backbone", "assemble_feature_matrix",
    "assign_block", "block_table", "reduce_collinearity_by_block",
    "TemporalBlockSplit", "fold_zscore", "lagged_correlation",
    "daily_percentile_thresholds", "severity_classify", "detrend_series",
    "effective_sample_size", "correlation_p_value", "benjamini_hochberg_fdr",
]
