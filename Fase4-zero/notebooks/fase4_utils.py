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
4. 4D SO DEPOIS agrupa descritivamente os alvos mais afetados, com lag de
   atuacao por tipo de sinal, estabilidade temporal e gate estatistico da
   hipotese NEB seco / Sul umido em El Nino.
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
MASTER = FEAT / "nino34_master_weekly.csv"
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

PACIFIC_VARS = [
    "nino34_ssta",
    "d20_m", "tilt_m", "tilt_slope",
    "ohc_0_100", "ohc_0_300", "ohc_0_700", "ohc_300_700",
    "ssh_m", "wwv",
    "t50m", "t100m", "t150m", "t200m", "t300m", "t500m", "t700m",
    "tau_x_anom", "u10_anom", "v10_anom",
    "mslp_anom", "tcwv_anom",
    "slhf_anom", "sshf_anom", "ssr_anom", "str_anom",
    "u850_anom", "u200_anom",
    "omega850_anom", "omega500_anom", "div850_anom",
]

VAR_META = {
    "nino34_ssta": ("SSTA Nino 3.4", "C", "oceano_superficie"),
    "d20_m": ("Profundidade da isoterma 20 C (D20)", "m", "oceano_recarga"),
    "tilt_m": ("Inclinacao zonal da termoclina", "m", "oceano_recarga"),
    "tilt_slope": ("Gradiente zonal da termoclina", "m/grau", "oceano_recarga"),
    "ohc_0_100": ("Conteudo de calor 0-100 m", "J m-2", "oceano_recarga"),
    "ohc_0_300": ("Conteudo de calor 0-300 m", "J m-2", "oceano_recarga"),
    "ohc_0_700": ("Conteudo de calor 0-700 m", "J m-2", "oceano_recarga"),
    "ohc_300_700": ("Conteudo de calor 300-700 m", "J m-2", "oceano_recarga"),
    "ssh_m": ("Altura dinamica/superficie do mar", "m", "oceano_recarga"),
    "wwv": ("Volume de agua quente", "m3", "oceano_recarga"),
    "t50m": ("Temperatura 50 m", "C", "oceano_subsuperficie"),
    "t100m": ("Temperatura 100 m", "C", "oceano_subsuperficie"),
    "t150m": ("Temperatura 150 m", "C", "oceano_subsuperficie"),
    "t200m": ("Temperatura 200 m", "C", "oceano_subsuperficie"),
    "t300m": ("Temperatura 300 m", "C", "oceano_subsuperficie"),
    "t500m": ("Temperatura 500 m", "C", "oceano_subsuperficie"),
    "t700m": ("Temperatura 700 m", "C", "oceano_subsuperficie"),
    "tau_x_anom": ("Tensao zonal do vento", "Pa", "atmosfera_bjerknes"),
    "u10_anom": ("Vento zonal 10 m", "m s-1", "atmosfera_bjerknes"),
    "v10_anom": ("Vento meridional 10 m", "m s-1", "atmosfera_bjerknes"),
    "mslp_anom": ("Pressao ao nivel medio do mar", "Pa", "atmosfera_bjerknes"),
    "tcwv_anom": ("Vapor d'agua integrado na coluna", "kg m-2", "atmosfera_bjerknes"),
    "slhf_anom": ("Fluxo turbulento de calor latente", "W m-2", "atmosfera_bjerknes"),
    "sshf_anom": ("Fluxo turbulento de calor sensivel", "W m-2", "atmosfera_bjerknes"),
    "ssr_anom": ("Radiacao solar liquida na superficie", "W m-2", "atmosfera_bjerknes"),
    "str_anom": ("Radiacao termica liquida na superficie", "W m-2", "atmosfera_bjerknes"),
    "u850_anom": ("Vento zonal 850 hPa", "m s-1", "atmosfera_bjerknes"),
    "u200_anom": ("Vento zonal 200 hPa", "m s-1", "atmosfera_bjerknes"),
    "omega850_anom": ("Velocidade vertical omega 850 hPa", "Pa s-1", "atmosfera_bjerknes"),
    "omega500_anom": ("Velocidade vertical omega 500 hPa", "Pa s-1", "atmosfera_bjerknes"),
    "div850_anom": ("Divergencia 850 hPa", "s-1", "atmosfera_bjerknes"),
}

VAR_SHORT = {
    "nino34_ssta": "SSTA", "d20_m": "D20", "tilt_m": "Tilt",
    "tilt_slope": "Tilt slope", "ohc_0_100": "OHC0-100",
    "ohc_0_300": "OHC0-300", "ohc_0_700": "OHC0-700",
    "ohc_300_700": "OHC300-700", "ssh_m": "SSH", "wwv": "WWV",
    "t50m": "T50", "t100m": "T100", "t150m": "T150", "t200m": "T200",
    "t300m": "T300", "t500m": "T500", "t700m": "T700",
    "tau_x_anom": "tau_x", "u10_anom": "U10", "v10_anom": "V10",
    "mslp_anom": "MSLP", "tcwv_anom": "TCWV", "slhf_anom": "SLHF",
    "sshf_anom": "SSHF", "ssr_anom": "SSR", "str_anom": "STR",
    "u850_anom": "U850", "u200_anom": "U200",
    "omega850_anom": "Omega850", "omega500_anom": "Omega500",
    "div850_anom": "Div850",
}

EN_CLASSES = [(2.0, "muito_forte"), (1.5, "forte"), (1.0, "moderado"), (0.5, "fraco")]
LN_CLASSES = [(2.0, "muito_forte"), (1.5, "forte"), (1.0, "moderada"), (0.5, "fraca")]

FASE_ORDER = ["genese", "crescimento", "pico", "decaimento"]
FASE_CORES = {"genese": "#93c5fd", "crescimento": "#fca5a5", "pico": "#111827", "decaimento": "#d1d5db"}


def var_label(name: str, *, short: bool = True) -> str:
    if short:
        return VAR_SHORT.get(name, name)
    return VAR_META.get(name, (name, "", ""))[0]


def var_unit(name: str) -> str:
    return VAR_META.get(name, ("", "", ""))[1]


def var_group(name: str) -> str:
    return VAR_META.get(name, ("", "", "outros"))[2]


# ---------------------------------------------------------------- IO basico
def ocean_raw_vars(columns) -> list[str]:
    """Variaveis oceanicas CRUAS do master (grupos recarga/subsuperficie).

    `nino34_ssta` (oceano_superficie) e as `*_anom` do ERA5 ja sao anomalias.
    """
    return [c for c in columns if var_group(c) in ("oceano_recarga", "oceano_subsuperficie")]


def load_pacific_weekly(*, anomalize_ocean: bool = True, detrend: bool = True) -> pd.DataFrame:
    """Matriz semanal canonica completa da Fase 2/3 usada na Fase 4.

    Inclui todas as variaveis numericas do master semanal: oceano unificado
    (1981-09-06 a 2026-06-14 quando valido) e ERA5 atmosferico
    (1981-01-04 a 2026-07-05). Cada teste usa a intersecao valida da variavel.

    `anomalize_ocean=True` (default apos o parecer 2026-07-10) remove das
    variaveis oceanicas cruas o ciclo anual (climatologia harmonica ajustada em
    CLIM_BASE) e a tendencia linear secular. Sem isso, correlacoes condicionadas
    a semanas EN/LN misturam calendario (phase-locking sazonal) e aquecimento de
    fundo com teleconexao interanual - o ciclo anual de D20 (~25 m) supera o
    proprio sigma total (~15 m). SSTA e ERA5 permanecem como estao (ja sao
    anomalias). Use `anomalize_ocean=False` para reproduzir a leitura antiga.
    """
    w = pd.read_csv(MASTER, parse_dates=["week_ending_sunday"]).set_index("week_ending_sunday")
    w = w[[c for c in PACIFIC_VARS if c in w.columns]]
    if anomalize_ocean:
        from nino_brasil.stats.climatology import harmonic_anomaly_matrix

        raw = ocean_raw_vars(w.columns)
        if raw:
            w = harmonic_anomaly_matrix(w, raw, base=CLIM_BASE, harmonics=3, detrend=detrend)
    return w


def pacific_variable_inventory(w: pd.DataFrame | None = None, *,
                               anomalize_ocean: bool = True) -> pd.DataFrame:
    """Tabela auditavel de variaveis usadas pela Fase 4 e sua cobertura real."""
    if w is None:
        w = load_pacific_weekly(anomalize_ocean=anomalize_ocean)
    raw = set(ocean_raw_vars(w.columns))
    rows = []
    for c in w.columns:
        s = pd.to_numeric(w[c], errors="coerce")
        ok = s.notna()
        first = w.index[ok].min() if ok.any() else pd.NaT
        last = w.index[ok].max() if ok.any() else pd.NaT
        if c in raw:
            tratamento = (
                "anomalia harmonica 1991-2020 + detrend linear (fit na base)"
                if anomalize_ocean else "valor fisico cru (ciclo anual presente)"
            )
        else:
            tratamento = "anomalia da fonte (Fase 2)"
        rows.append({
            "fonte": "master semanal NINO26",
            "variavel": c,
            "nome": var_label(c, short=False),
            "abreviacao": var_label(c),
            "grupo": var_group(c),
            "unidade": var_unit(c),
            "tratamento": tratamento,
            "serie_temporal_valida": (
                f"{first.date()} a {last.date()}" if pd.notna(first) else ""
            ),
            "n_semanas_validas": int(ok.sum()),
            "cobertura_%": round(100 * float(ok.mean()), 2),
            "intervalo_coleta_analise": "semanal W-SUN",
        })
    return pd.DataFrame(rows)


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
    """Salva a figura sob o código canônico (Fig_<F><B><NN>) e co-gera a
    numeric-table correspondente (nino_brasil.viz), por sobreposição. Fallback
    robusto se a co-geração falhar (mantém a figura sob o nome original)."""
    try:
        from nino_brasil.viz import cogerar_de_figura
        return cogerar_de_figura(fig, name, fase=4, notebook="notebooks/fase4")
    except Exception as exc:
        import tempfile, shutil, time
        print(f"[aviso] co-geracao falhou ({exc}); salvando figura simples.")
        path = FIGS / name
        tmp = Path(tempfile.gettempdir()) / name
        fig.savefig(tmp, dpi=150, bbox_inches="tight")
        for _ in range(5):
            try:
                shutil.copyfile(tmp, path)
                if path.stat().st_size > 0:
                    break
            except OSError:
                time.sleep(0.4)
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
        qualified = mag >= PICO_FRACAO * mag.max()
        run_id = qualified.ne(qualified.shift(fill_value=False)).cumsum()
        peak_run = run_id.loc[mag.idxmax()]
        # Only the contiguous qualified component containing the absolute
        # extreme is peak.  Disconnected shoulders cannot be joined by min/max.
        plateau = mag[qualified & run_id.eq(peak_run)]
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
    """Compatibilidade: anomalia robusta por semana ISO, nunca sigma ~ 0.

    O ajuste harmonico da escala ao residuo absoluto foi removido porque podia
    produzir escala negativa; o ``clip(1e-6)`` subsequente gerava |z| de milhoes.
    O argumento ``n_harm`` permanece apenas para nao quebrar notebooks antigos.
    A implementacao canonica usa mediana/MAD por semana ISO, fallback no MAD dos
    residuos e retorna NaN quando a variancia de referencia nao e identificavel.
    """
    import xarray as xr
    from nino_brasil.targets.chirps_native import robust_weekly_anomalies

    if not isinstance(weekly.index, pd.DatetimeIndex):
        raise TypeError("weekly precisa de DatetimeIndex semanal")
    da = xr.DataArray(
        weekly.to_numpy(dtype="float64"),
        coords={"time": weekly.index, "pixel": np.arange(weekly.shape[1])},
        dims=("time", "pixel"),
        name="precip_weekly_mm",
    )
    result = robust_weekly_anomalies(da, baseline=base)["precip_robust_z"]
    return pd.DataFrame(
        result.values,
        index=weekly.index,
        columns=weekly.columns,
        dtype="float32",
    )


def build_chirps_weekly_zanom(*, force: bool = False, min_cov: float = 0.95) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega o alvo canonico CHIRPS nativo usado por F4/F6/F8.

    A construcao pesada saiu do notebook e esta em
    ``scripts/build_phase4_chirps_targets.py``.  O cubo mantem o retangulo
    original CHIRPS e as mascaras ``brazil_fraction``/``brazil_center``; esta
    funcao devolve somente os pixels cujo centro esta no Brasil, sem alterar
    coordenadas ou valores. ``min_cov`` e mantido apenas por compatibilidade.
    """
    import xarray as xr
    from nino_brasil.targets.chirps_native import target_to_frame, validate_native_target

    target_path = ROOT / "data/processed/zarr/features/chirps_native_weekly_targets.zarr"
    if force:
        raise ValueError(
            "force nao e permitido dentro do notebook: execute "
            "python scripts/build_phase4_chirps_targets.py --replace-existing; "
            "a versao anterior sera arquivada, nunca apagada silenciosamente."
        )
    if not target_path.exists():
        raise FileNotFoundError(
            f"alvo CHIRPS nativo ausente: {target_path}. Execute antes: "
            "python scripts/build_phase4_chirps_targets.py"
        )
    ds = xr.open_zarr(target_path, consolidated=None)
    validation = validate_native_target(ds, deep=False)
    if not validation.valid:
        raise ValueError(f"alvo CHIRPS nativo invalido: {validation.errors}")
    if ds.attrs.get("deep_validation_passed") is not True:
        raise ValueError("alvo CHIRPS sem carimbo de validacao profunda do builder")
    z, px = target_to_frame(
        ds,
        variable="precip_robust_z",
        brazil_only=True,
        mask_rule="center",
    )
    print(
        f"[target] {target_path.relative_to(ROOT)} {z.shape} | "
        f"grid_sha256={validation.grid_hash} | pixels CHIRPS nativos"
    )
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


def lagged_corr_pixel_matrix(X: pd.DataFrame, R: pd.DataFrame, lags: list[int],
                             mask_weeks: pd.Series | None = None) -> dict:
    """Correlacao defasada para muitas variaveis ao mesmo tempo.

    Retorna arrays com dimensoes (variavel, lag, pixel). A estatistica e a mesma
    de `lagged_corr_pixel`, mas usa produto matricial por grupo de variaveis com
    a mesma mascara valida em cada lag, acelerando a Fase 4C completa.
    """
    from scipy import stats as st

    X = X.reindex(R.index)
    vars_ = list(X.columns)
    Rv = R.to_numpy(dtype="float64")
    b = _lag1(Rv)
    base_mask = np.ones(len(R), dtype=bool) if mask_weeks is None else mask_weeks.reindex(R.index).fillna(False).to_numpy()

    r_out = np.full((len(vars_), len(lags), R.shape[1]), np.nan, dtype="float32")
    p_out = np.full_like(r_out, np.nan)
    n_out = np.full_like(r_out, np.nan)

    for li, L in enumerate(lags):
        Xl = X.shift(L)
        Xv = Xl.to_numpy(dtype="float64")
        groups: dict[bytes, tuple[np.ndarray, list[int]]] = {}
        for vi in range(len(vars_)):
            m = base_mask & np.isfinite(Xv[:, vi])
            if int(m.sum()) < 30:
                continue
            key = m.tobytes()
            if key not in groups:
                groups[key] = (m, [])
            groups[key][1].append(vi)

        for m, vis in groups.values():
            n = int(m.sum())
            Xg = Xv[m][:, vis]
            Rm = Rv[m]
            x_mean = np.nanmean(Xg, axis=0)
            x_std = np.nanstd(Xg, axis=0)
            r_mean = np.nanmean(Rm, axis=0)
            r_std = np.nanstd(Rm, axis=0)
            x_std[x_std == 0] = np.nan
            r_std[r_std == 0] = np.nan
            Xz = (Xg - x_mean) / x_std
            Rz = (Rm - r_mean) / r_std
            corr = (Xz.T @ Rz) / n
            corr = np.clip(corr, -0.999999, 0.999999)

            a = _lag1(Xg)
            neff = np.clip(n * (1 - a[:, None] * b[None, :]) / (1 + a[:, None] * b[None, :]), 4, None)
            t = corr * np.sqrt((neff - 2) / np.clip(1 - corr**2, 1e-9, None))
            p = 2.0 * st.t.sf(np.abs(t), neff - 2)

            r_out[vis, li, :] = corr.astype("float32")
            p_out[vis, li, :] = p.astype("float32")
            n_out[vis, li, :] = neff.astype("float32")

    return {"r": r_out, "p": p_out, "n_eff": n_out, "lags": np.array(lags), "variavel": vars_}


def lag_window_inventory(x: pd.Series, target_index: pd.DatetimeIndex, lags: list[int],
                         conditions: dict[str, pd.Series | None]) -> pd.DataFrame:
    """Descreve as janelas de pareamento usadas para identificar cada lag.

    Para um lag L, a janela estatistica e sempre:

        variavel_pacifico(t-L) pareada com anomalia_chuva_pixel(t)

    Portanto, para cada semana-alvo t no CHIRPS, o preditor vem L semanas antes.
    A tabela registra inicio/fim da janela-alvo, inicio/fim da janela do
    preditor e numero de pares validos por condicao.
    """
    rows = []
    target_index = pd.DatetimeIndex(target_index)
    for cond, mask in conditions.items():
        base_mask = np.ones(len(target_index), dtype=bool)
        if mask is not None:
            base_mask = mask.reindex(target_index).fillna(False).to_numpy()
        for L in lags:
            xl = x.shift(L).reindex(target_index)
            m = base_mask & xl.notna().to_numpy()
            alvo = target_index[m]
            if len(alvo):
                pred = alvo - pd.to_timedelta(L, unit="W")
                rows.append({
                    "condicao": cond,
                    "lag_sem": int(L),
                    "janela_alvo_inicio": alvo.min().date(),
                    "janela_alvo_fim": alvo.max().date(),
                    "janela_pacifico_inicio": pred.min().date(),
                    "janela_pacifico_fim": pred.max().date(),
                    "n_pares_semanais": int(len(alvo)),
                    "regra": "Pacifico(t-lag) pareado com anomalia_chuva_pixel(t)",
                })
            else:
                rows.append({
                    "condicao": cond,
                    "lag_sem": int(L),
                    "janela_alvo_inicio": "",
                    "janela_alvo_fim": "",
                    "janela_pacifico_inicio": "",
                    "janela_pacifico_fim": "",
                    "n_pares_semanais": 0,
                    "regra": "Pacifico(t-lag) pareado com anomalia_chuva_pixel(t)",
                })
    return pd.DataFrame(rows)


def fdr_bh(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
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
