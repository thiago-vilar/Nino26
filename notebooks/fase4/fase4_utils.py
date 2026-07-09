"""Utilidades compartilhadas da Fase 4 (4A-4D) - teleconexao ENSO -> chuva Brasil.

Convencoes (herdadas da Fase 3):
- Eixo canonico semanal: semanas terminando no domingo (W-SUN).
- Tabelas numericas em data/processed/parquet/statistics/phase4*_.csv
- Figuras em data/processed/figures/fase4/
- Nenhuma figura sem saida numerica correspondente.
- Toda significancia usa N_eff (Bretherton) + FDR Benjamini-Hochberg.

Desenho cientifico da Fase 4 (parecer 2026-07-08, expandido):
1. 4A separa o ciclo ENSO em 4 fases (genese, crescimento/acoplamento, pico,
   decaimento) para El Nino E La Nina, com criterio estatistico explicito.
2. 4B estudo puramente estatistico de quais variaveis do Pacifico mais
   determinam cada fase (I. Genese, II. Crescimento, III. Pico, IV. Decaimento).
3. 4C distribui o sinal PIXEL-A-PIXEL: conjunto Pacifico x precipitacao CHIRPS
   0.25, lags semanais, Brasil inteiro e depois recortes NEB e Sul.
4. 4D SO DEPOIS clusteriza os alvos mais afetados, com lag de atuacao por tipo
   de sinal, estabilidade temporal e gate para as fases de modelagem.
Escopo estritamente Pacifico -> Brasil.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

FEAT = ROOT / "data/processed/parquet/features"
STATS = ROOT / "data/processed/parquet/statistics"
FIGS = ROOT / "data/processed/figures/fase4"
ZSTATS = ROOT / "data/processed/zarr/statistics"
RAIN_ZARR = ROOT / "data/processed/zarr/brazil_precipitation"
for _p in (STATS, FIGS, ZSTATS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- constantes
BRAZIL_BOX = {"lat_min": -35.0, "lat_max": 7.0, "lon_min": -75.0, "lon_max": -32.0}
# Caixas regionais aproximadas (leitura executiva; refinamento IBGE e opcional)
REGIOES = {
    "NEB": {"lat_min": -18.0, "lat_max": -1.0, "lon_min": -48.0, "lon_max": -34.5},
    "SUL": {"lat_min": -34.0, "lat_max": -22.0, "lon_min": -58.0, "lon_max": -48.0},
}
CLIM_BASE = ("1991-01-01", "2020-12-31")   # base descritiva do projeto
GENESE_SEMANAS = 26   # janela pre-onset onde os precursores se organizam (3C/3H)
PICO_FRACAO = 0.90    # pico = |ONI| >= 90% do |ONI| maximo do evento
LIMIAR_ONI = 0.5      # NOAA/ONI local
MIN_ESTACOES = 5      # estacoes moveis sobrepostas

PACIFIC_VARS = ["nino34_ssta", "d20_m", "ohc_0_300", "ohc_0_700",
                "ssh_m", "tilt_m", "wwv", "tau_x_anom_nino34_pa"]

VAR_SHORT = {
    "nino34_ssta": "SSTA", "d20_m": "D20", "ohc_0_300": "OHC0-300",
    "ohc_0_700": "OHC0-700", "ssh_m": "SSH", "tilt_m": "Tilt",
    "wwv": "WWV", "tau_x_anom_nino34_pa": "tau_x anom.",
}

EN_CLASSES = [(2.0, "muito_forte"), (1.5, "forte"), (1.0, "moderado"), (0.5, "fraco")]
LN_CLASSES = [(2.0, "muito_forte"), (1.5, "forte"), (1.0, "moderada"), (0.5, "fraca")]

FASE_ORDER = ["genese", "crescimento", "pico", "decaimento"]
FASE_CORES = {"genese": "#93c5fd", "crescimento": "#fca5a5", "pico": "#111827", "decaimento": "#d1d5db"}


def var_label(name: str, *, short: bool = True) -> str:
    return VAR_SHORT.get(name, name)


# ---------------------------------------------------------------- IO basico
def load_pacific_weekly() -> pd.DataFrame:
    """Matriz semanal canonica da Fase 3 (somente Pacifico; sem metricas auxiliares)."""
    w = pd.read_csv(FEAT / "phase3_indices_semanais.csv",
                    parse_dates=["week_ending_sunday"]).set_index("week_ending_sunday")
    return w[[c for c in PACIFIC_VARS if c in w.columns]]


def load_oni_monthly() -> pd.Series:
    m = pd.read_csv(FEAT / "nino34_monthly_oisst.csv", parse_dates=["time"])
    m = m[m.get("month_complete", True).astype(bool)]
    return m.set_index("time")["oni_local_c"].astype(float)


def neff_corr(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    """Pearson r com graus de liberdade efetivos de Bretherton; retorna (r, p, n_eff)."""
    from scipy import stats as st
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    if len(x) < 30:
        return float("nan"), float("nan"), 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    a, b = float(x.autocorr(1)), float(y.autocorr(1))
    ne = max(len(x) * (1 - a * b) / (1 + a * b), 4.0)
    t = r * np.sqrt((ne - 2) / max(1 - r ** 2, 1e-9))
    return r, float(2 * st.t.sf(abs(t), ne - 2)), ne


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> Path:
    path = STATS / name
    df.to_csv(path, index=index)
    print(f"[tabela] {path.relative_to(ROOT)}")
    return path


def save_fig(fig, name: str) -> Path:
    import tempfile, shutil, time
    path = FIGS / name
    tmp = Path(tempfile.gettempdir()) / name
    fig.savefig(tmp, dpi=150, bbox_inches="tight")
    last = None
    for _ in range(5):
        try:
            shutil.copyfile(tmp, path)
            if path.stat().st_size > 0:
                break
        except OSError as exc:
            last = exc; time.sleep(0.4)
    else:
        if last:
            raise last
    print(f"[figura] {path.relative_to(ROOT)}")
    return path


def stamp_caption(fig, *, variavel, area, periodo, fonte, n=None, extra=None):
    partes = [f"Variavel: {variavel}", f"Area: {area}", f"Periodo: {periodo}", f"Fonte: {fonte}"]
    if n:
        partes.append(f"n={n}")
    if extra:
        partes.append(extra)
    fig.text(0.5, -0.04, " | ".join(partes), ha="center", va="top", fontsize=6.5, color="#444", wrap=True)


def add_note(ax, text: str, *, loc: str = "lower right") -> None:
    anchors = {"lower right": (0.985, 0.02, "right", "bottom"),
               "upper right": (0.985, 0.98, "right", "top"),
               "lower left": (0.015, 0.02, "left", "bottom"),
               "upper left": (0.015, 0.98, "left", "top")}
    x, y, ha, va = anchors.get(loc, anchors["lower right"])
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#888", "alpha": 0.86})


# ------------------------------------------------- 4A: eventos e fases ENSO
def enso_events(oni: pd.Series) -> pd.DataFrame:
    """Eventos El Nino (ONI>=+0.5) e La Nina (ONI<=-0.5) por >=5 estacoes moveis.

    Retorna onset, pico (mes de |ONI| maximo), fim, classe de intensidade e tipo.
    O criterio e simetrico e local (OISST NINO-BRASIL), sem rotulo externo.
    """
    rows = []
    for tipo, sign in (("el_nino", 1.0), ("la_nina", -1.0)):
        cond = (sign * oni) >= LIMIAR_ONI
        grupo = (cond != cond.shift()).cumsum()
        for _, bloco in oni.groupby(grupo):
            if not cond.loc[bloco.index[0]] or len(bloco) < MIN_ESTACOES:
                continue
            mag = (sign * bloco)
            pico_t = mag.idxmax()
            pico_v = float(oni.loc[pico_t])
            classes = EN_CLASSES if sign > 0 else LN_CLASSES
            classe = next(lbl for thr, lbl in classes if abs(pico_v) >= thr)
            rows.append({
                "event_id": f"{tipo}_{bloco.index[0].year}_{bloco.index[-1].year}",
                "tipo": tipo, "classe": classe,
                "onset": bloco.index[0], "pico": pico_t,
                "fim": bloco.index[-1] + pd.offsets.MonthEnd(0),
                "duracao_estacoes": int(len(bloco)),
                "oni_pico_c": round(pico_v, 3),
            })
    ev = pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)
    return ev


def enso_phase_weekly(oni: pd.Series, ev: pd.DataFrame,
                      weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Rotulo semanal do ciclo ENSO em 4 fases, para EN e LN.

    Criterio estatistico das fases, por evento:
      - pico        : meses com |ONI| >= PICO_FRACAO x |ONI_pico| (plateau do pico);
      - crescimento : do onset ate o inicio do plateau;
      - decaimento  : do fim do plateau ate o fim do evento;
      - genese      : GENESE_SEMANAS antes do onset, apenas em semanas neutras
                      (nao sobrescreve outro evento; conflito = mantem o evento).
    Semanas restantes = 'neutro'.
    """
    fase = pd.Series("neutro", index=weeks, name="fase")
    tipo = pd.Series("neutro", index=weeks, name="tipo")
    eid = pd.Series("", index=weeks, name="event_id")

    def _mark(t0, t1, f, tp, e, overwrite=True):
        m = (weeks >= pd.to_datetime(t0)) & (weeks <= pd.to_datetime(t1))
        if not overwrite:
            m = m & (fase.values == "neutro")
        fase.iloc[np.where(m)[0]] = f
        tipo.iloc[np.where(m)[0]] = tp
        eid.iloc[np.where(m)[0]] = e

    for _, e in ev.iterrows():
        sign = 1.0 if e.tipo == "el_nino" else -1.0
        bloco = oni.loc[e.onset:e.fim]
        mag = sign * bloco
        plateau = mag[mag >= PICO_FRACAO * mag.max()]
        p0, p1 = plateau.index.min(), plateau.index.max() + pd.offsets.MonthEnd(0)
        _mark(e.onset, p0 - pd.Timedelta(days=1), "crescimento", e.tipo, e.event_id)
        _mark(p0, p1, "pico", e.tipo, e.event_id)
        _mark(p1 + pd.Timedelta(days=1), e.fim, "decaimento", e.tipo, e.event_id)
    # genese por ultimo, sem sobrescrever eventos
    for _, e in ev.iterrows():
        g0 = pd.to_datetime(e.onset) - pd.Timedelta(weeks=GENESE_SEMANAS)
        _mark(g0, pd.to_datetime(e.onset) - pd.Timedelta(days=1),
              "genese", e.tipo, e.event_id, overwrite=False)
    out = pd.concat([fase, tipo, eid], axis=1)
    out.index.name = "week_ending_sunday"
    return out


# ------------------------------------------------- 4B: chuva pixel-a-pixel
def _harmonic_design(week_of_year: np.ndarray, n_harm: int = 3) -> np.ndarray:
    t = 2.0 * np.pi * (week_of_year - 1) / 52.1775
    cols = [np.ones_like(t)]
    for k in range(1, n_harm + 1):
        cols += [np.sin(k * t), np.cos(k * t)]
    return np.column_stack(cols)


def harmonic_standardized_anomaly(weekly: pd.DataFrame, *, n_harm: int = 3,
                                  base: tuple[str, str] = CLIM_BASE) -> pd.DataFrame:
    """Anomalia padronizada por pixel com climatologia harmonica (fit na base).

    media(t) = harmonicos ajustados na base; sigma(t) = harmonicos ajustados ao
    residuo absoluto (suaviza a sazonalidade da variancia). Evita 52 medias cruas.
    """
    woy = weekly.index.isocalendar().week.to_numpy().astype(float)
    X = _harmonic_design(woy, n_harm)
    inbase = (weekly.index >= pd.to_datetime(base[0])) & (weekly.index <= pd.to_datetime(base[1]))
    Y = weekly.to_numpy(dtype="float64")
    Xb = X[inbase]
    beta, *_ = np.linalg.lstsq(Xb, np.nan_to_num(Y[inbase]), rcond=None)
    clim = X @ beta
    resid = Y - clim
    beta_s, *_ = np.linalg.lstsq(Xb, np.abs(np.nan_to_num(resid[inbase])), rcond=None)
    sigma = np.clip(X @ beta_s, 1e-6, None) * np.sqrt(np.pi / 2.0)  # E|x| -> sigma gaussiana
    z = resid / sigma
    return pd.DataFrame(z, index=weekly.index, columns=weekly.columns).astype("float32")


def build_chirps_weekly_zanom(*, force: bool = False, min_cov: float = 0.95) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CHIRPS diario -> soma semanal W-SUN no Brasil -> anomalia padronizada por pixel.

    Cacheia em FEAT/phase4_chirps_weekly_zanom.parquet (semanas x pixels) e
    FEAT/phase4_chirps_pixels.csv (pixel_id, lat, lon). Primeira execucao e pesada.
    """
    cache_z = FEAT / "phase4_chirps_weekly_zanom.parquet"
    cache_px = FEAT / "phase4_chirps_pixels.csv"
    if cache_z.exists() and cache_px.exists() and not force:
        z = pd.read_parquet(cache_z)
        z.index = pd.to_datetime(z.index)
        px = pd.read_csv(cache_px)
        return z, px
    import xarray as xr
    stores = sorted(RAIN_ZARR.glob("chirps_p25_*.zarr"))
    if not stores:
        raise FileNotFoundError(f"nenhum store CHIRPS em {RAIN_ZARR}")
    weekly_parts = []
    for s in stores:
        ds = xr.open_zarr(s, consolidated=False)
        da = ds["precip"].sel(
            latitude=slice(BRAZIL_BOX["lat_min"], BRAZIL_BOX["lat_max"]),
            longitude=slice(BRAZIL_BOX["lon_min"], BRAZIL_BOX["lon_max"]))
        wk = da.resample(time="W-SUN").sum(min_count=4)  # mm/semana
        weekly_parts.append(wk.load())
        print(f"  [chirps] {s.name} ok")
    wk_all = xr.concat(weekly_parts, dim="time")
    wk_all = wk_all.groupby("time").mean("time")  # semanas duplicadas na virada de ano
    df = wk_all.to_dataframe(name="p").reset_index()
    mat = df.pivot_table(index="time", columns=["latitude", "longitude"], values="p")
    cov = mat.notna().mean()
    mat = mat.loc[:, cov >= min_cov]
    z = harmonic_standardized_anomaly(mat)
    px = pd.DataFrame({"pixel_id": range(mat.shape[1]),
                       "lat": [c[0] for c in mat.columns],
                       "lon": [c[1] for c in mat.columns]})
    z.columns = px["pixel_id"].astype(str)
    z.to_parquet(cache_z)
    px.to_csv(cache_px, index=False)
    print(f"[cache] {cache_z.relative_to(ROOT)} {z.shape}")
    return z, px


def _lag1(a: np.ndarray) -> np.ndarray:
    a0, a1 = a[:-1], a[1:]
    a0 = a0 - np.nanmean(a0, axis=0)
    a1 = a1 - np.nanmean(a1, axis=0)
    num = np.nansum(a0 * a1, axis=0)
    den = np.sqrt(np.nansum(a0**2, axis=0) * np.nansum(a1**2, axis=0))
    return np.clip(num / np.where(den == 0, np.nan, den), -0.99, 0.99)


def lagged_corr_pixel(x: pd.Series, R: pd.DataFrame, lags: list[int],
                      mask_weeks: pd.Series | None = None) -> dict:
    """Correlacao defasada pixel-a-pixel: x(t-L) vs chuva(t), com N_eff e p.

    mask_weeks: booleano por semana-alvo (ex.: apenas semanas EN ou LN) para
    condicionar o sinal ao tipo de evento. Retorna dict com r, p, n_eff (lags x px).
    """
    from scipy import stats as st
    Rv = R.to_numpy(dtype="float64")
    b = _lag1(Rv)                                   # autocorr da chuva por pixel
    r_out = np.full((len(lags), R.shape[1]), np.nan, dtype="float32")
    p_out = np.full_like(r_out, np.nan)
    n_out = np.full_like(r_out, np.nan)
    base_mask = np.ones(len(R), dtype=bool) if mask_weeks is None else mask_weeks.reindex(R.index).fillna(False).to_numpy()
    for i, L in enumerate(lags):
        xl = x.shift(L).reindex(R.index)
        m = base_mask & xl.notna().to_numpy() & ~np.isnan(Rv).any(axis=1)
        n = int(m.sum())
        if n < 30:
            continue
        xv = xl.to_numpy()[m]
        a = float(_lag1(xv[:, None])[0])
        zx = (xv - xv.mean()) / xv.std()
        Rm = Rv[m]
        Zr = (Rm - Rm.mean(axis=0)) / Rm.std(axis=0)
        r = (Zr.T @ zx) / n
        neff = np.clip(n * (1 - a * b) / (1 + a * b), 4, None)
        t = r * np.sqrt((neff - 2) / np.clip(1 - r**2, 1e-9, None))
        p = 2.0 * st.t.sf(np.abs(t), neff - 2)
        r_out[i], p_out[i], n_out[i] = r, p, neff
    return {"r": r_out, "p": p_out, "n_eff": n_out, "lags": np.array(lags)}


def fdr_bh(p: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Mascara FDR Benjamini-Hochberg sobre TODOS os testes fornecidos."""
    flat = p.ravel()
    ok = np.isfinite(flat)
    mask = np.zeros_like(flat, dtype=bool)
    if ok.sum() == 0:
        return mask.reshape(p.shape)
    ps = flat[ok]
    order = np.argsort(ps)
    ranked = ps[order]
    mtests = len(ranked)
    thresh = alpha * (np.arange(1, mtests + 1) / mtests)
    below = ranked <= thresh
    if below.any():
        cut = ranked[np.where(below)[0].max()]
        mask[ok] = flat[ok] <= cut
    return mask.reshape(p.shape)


def best_lag_maps(res: dict, sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Por pixel: lag do |r| maximo significativo e o r correspondente."""
    r = np.where(sig, res["r"], np.nan)
    all_nan = np.isnan(r).all(axis=0)
    idx = np.nanargmax(np.abs(np.where(all_nan[None, :], -np.inf, r)), axis=0)
    best_r = r[idx, np.arange(r.shape[1])]
    best_lag = res["lags"][idx].astype(float)
    best_lag[all_nan] = np.nan
    best_r = np.where(all_nan, np.nan, best_r)
    return best_lag, best_r


def region_mask(px: pd.DataFrame, box: dict) -> np.ndarray:
    return ((px["lat"] >= box["lat_min"]) & (px["lat"] <= box["lat_max"]) &
            (px["lon"] >= box["lon_min"]) & (px["lon"] <= box["lon_max"])).to_numpy()


def pixel_map(ax, px: pd.DataFrame, values: np.ndarray, *, cmap="RdBu_r",
              vmin=None, vmax=None, title="", extent: dict | None = None,
              point_size: float = 4, draw_regions: bool = True):
    """Scatter-grid rapido dos pixels 0.25 (sem dependencia de cartopy).

    extent: caixa {'lat_min','lat_max','lon_min','lon_max'} para recorte
    regional (zoom); default = Brasil inteiro.
    """
    box_ext = extent or BRAZIL_BOX
    sc = ax.scatter(px["lon"], px["lat"], c=values, s=point_size, marker="s",
                    cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_xlim(box_ext["lon_min"], box_ext["lon_max"])
    ax.set_ylim(box_ext["lat_min"], box_ext["lat_max"])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    if draw_regions:
        for nome, box in REGIOES.items():
            ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                (box["lon_min"], box["lat_min"]), box["lon_max"] - box["lon_min"],
                box["lat_max"] - box["lat_min"], fill=False, edgecolor="k", lw=1.0, ls="--"))
            ax.text(box["lon_min"] + 0.4, box["lat_max"] - 1.2, nome, fontsize=8, weight="bold")
    return sc
