"""Utilidades compartilhadas dos notebooks da Fase 3 (3A-3I/3K).

Convencoes:
- Eixo canonico semanal: semanas terminando no domingo (W-SUN).
- Tabelas numericas em data/processed/parquet/statistics/phase3*_.csv
- Figuras em data/processed/figures/fase3/
- Nenhuma figura sem saida numerica correspondente.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

FEAT = ROOT / "data/processed/parquet/features"
STATS = ROOT / "data/processed/parquet/statistics"
FIGS = ROOT / "data/processed/figures/fase3"
STATS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

RHO_AIR = 1.2       # kg m-3
CD_NEUTRAL = 1.3e-3  # arrasto neutro aproximado


def load_daily_nino34() -> pd.DataFrame:
    df = pd.read_csv(FEAT / "nino34_daily_oisst.csv", parse_dates=["time"])
    return df.set_index("time").sort_index()


def load_physical_signal() -> pd.DataFrame:
    df = pd.read_csv(FEAT / "nino34_physical_signal.csv", parse_dates=["time"])
    return df.set_index("time").sort_index()


def load_atlantic() -> pd.DataFrame:
    df = pd.read_csv(FEAT / "tropical_atlantic_sst_daily.csv", parse_dates=["time"])
    return df.set_index("time").sort_index()


def load_dhw() -> pd.DataFrame:
    df = pd.read_csv(FEAT / "nino34_dhw_daily.csv", parse_dates=["time"])
    return df.set_index("time").sort_index()


def load_dhw_variants() -> pd.DataFrame:
    """Arquivo bruto de DHW; a Fase 3 publica apenas `dhw_cweek_0p5_12w`."""
    df = pd.read_csv(FEAT / "nino34_dhw_variants.csv", parse_dates=["time"])
    return df.set_index("time").sort_index()


def load_atmo() -> pd.DataFrame:
    df = pd.read_csv(FEAT / "era5_nino34_atmo_cache.csv", parse_dates=["time"])
    return df.set_index("time").sort_index()


def load_events() -> pd.DataFrame:
    ev = pd.read_csv(FEAT / "nino34_oisst_event_reference.csv",
                     parse_dates=["event_start", "event_end", "peak_time"])
    return ev


NOAA_INTENSITY_GROUPS = (
    {
        "grupo": "fraco",
        "rotulo_curto": "Fraco",
        "rotulo": "El Nino fraco (0.5 <= ONI < 1.0 C)",
        "definicao": "pico da media movel de 3 meses na Nino 3.4 entre +0.5 e +0.9 C",
        "color": "#f9a825",
        "linestyle": "--",
        "linewidth": 1.5,
    },
    {
        "grupo": "moderado",
        "rotulo_curto": "Moderado",
        "rotulo": "El Nino moderado (1.0 <= ONI < 1.5 C)",
        "definicao": "pico da media movel de 3 meses na Nino 3.4 entre +1.0 e +1.4 C",
        "color": "#e65100",
        "linestyle": "--",
        "linewidth": 1.7,
    },
    {
        "grupo": "forte",
        "rotulo_curto": "Forte",
        "rotulo": "El Nino forte (1.5 <= ONI < 2.0 C)",
        "definicao": "pico da media movel de 3 meses na Nino 3.4 entre +1.5 e +1.9 C",
        "color": "#b71c1c",
        "linestyle": "--",
        "linewidth": 1.9,
    },
    {
        "grupo": "muito_forte",
        "rotulo_curto": "Muito forte",
        "rotulo": "El Nino muito forte / super (ONI >= 2.0 C)",
        "definicao": "pico da media movel de 3 meses na Nino 3.4 igual ou acima de +2.0 C",
        "color": "#111827",
        "linestyle": "-",
        "linewidth": 2.4,
    },
)
NOAA_CLASS_ORDER = tuple(item["grupo"] for item in NOAA_INTENSITY_GROUPS)
ELNINO_MEAN_GROUP_ORDER = NOAA_CLASS_ORDER


def normalize_noaa_class(value: object) -> str:
    mapping = {
        "weak_el_nino": "fraco",
        "moderate_el_nino": "moderado",
        "strong_el_nino": "forte",
        "super_el_nino": "muito_forte",
        "very_strong_el_nino": "muito_forte",
    }
    text = str(value)
    return mapping.get(text, text)


def add_noaa_classification(events: pd.DataFrame) -> pd.DataFrame:
    """Garante classe NOAA/ONI local e metadados de corte em eventos Nino 3.4."""
    out = events.copy()
    if "peak_class" in out:
        out["classe_noaa"] = out["peak_class"].map(normalize_noaa_class)
    else:
        value = pd.to_numeric(out.get("peak_oni_local_c", out.get("peak_ssta_c")), errors="coerce")
        out["classe_noaa"] = pd.cut(
            value,
            bins=[-float("inf"), 0.5, 1.0, 1.5, 2.0, float("inf")],
            labels=["neutral", "fraco", "moderado", "forte", "muito_forte"],
            right=False,
        ).astype(str)
    out["elegivel_noaa_oni"] = out["classe_noaa"].isin(NOAA_CLASS_ORDER)
    out["limiar_oni_evento_c"] = 0.5
    out["min_estacoes_sobrepostas"] = 5
    return out


def events_noaa() -> pd.DataFrame:
    """Eventos El Nino locais pela regra NOAA/ONI compativel da Fase 3."""
    return add_noaa_classification(load_events()).query("elegivel_noaa_oni").copy()


def elnino_mean_group_table() -> pd.DataFrame:
    """Tabela de referencia das classes oficiais NOAA usadas nos compostos."""
    return pd.DataFrame(NOAA_INTENSITY_GROUPS).copy()


def elnino_mean_groups(events: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Retorna medias por classe NOAA sempre na ordem fraco->muito_forte."""
    ev = events_noaa() if events is None else add_noaa_classification(events)
    return {group: ev.query("classe_noaa == @group").copy() for group in NOAA_CLASS_ORDER}


def elnino_group_style(group: str) -> dict:
    row = elnino_mean_group_table().set_index("grupo").loc[group]
    return row.to_dict()


def elnino_group_label(group: str, *, short: bool = False) -> str:
    row = elnino_group_style(group)
    return str(row["rotulo_curto" if short else "rotulo"])


def load_eqband_weekly() -> pd.DataFrame:
    df = pd.read_parquet(FEAT / "equatorial_pacific_ssta_weekly_by_lon.parquet")
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(float)
    return df.loc[:, (df.columns >= 120) & (df.columns <= 280)]


def load_ssh_events() -> pd.DataFrame:
    df = pd.read_parquet(FEAT / "ssh_equatorial_daily_by_lon_events.parquet")
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(float)
    return df


def zonal_wind_stress_proxy(u10: pd.Series) -> pd.Series:
    """Proxy bruto de estresse zonal do vento: tau_x = rho_a * Cd * |u10| * u10.

    Caveat metodologico: usa u10 medio da caixa Nino 3.4 (cache ERA5 local);
    o protocolo pede tau_x na caixa Nino 4 - substituir quando o recorte
    Nino 4 for materializado. O sinal (leste/oeste) e a fase temporal sao
    preservados; a magnitude absoluta nao deve ser interpretada.
    """
    return (RHO_AIR * CD_NEUTRAL * u10.abs() * u10).rename("tau_x_raw_proxy_nino34_pa")


def daily_doy_anomaly(series: pd.Series, *, base_start: str = "1991-01-01", base_end: str = "2020-12-31") -> pd.Series:
    """Anomalia diaria simples por dia-do-ano, estimada no periodo base."""
    s = series.sort_index().astype(float)
    base = s.loc[base_start:base_end]
    clim = base.groupby(base.index.dayofyear).mean()
    out = s - pd.Series(s.index.dayofyear, index=s.index).map(clim).astype(float)
    return out.rename(f"{series.name}_anom")


def to_weekly(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    res = df.resample("W-SUN")
    return getattr(res, how)()


def weekly_matrix() -> pd.DataFrame:
    """Matriz semanal canonica da Fase 3 (indices fisicos + controles)."""
    phys = load_physical_signal()
    cols = {
        "nino34_ssta": "nino34_ssta",
        "d20_nino34_mean_m": "d20_m",
        "ohc_0_300_nino34_j_m2": "ohc_0_300",
        "ohc_0_700_nino34_j_m2": "ohc_0_700",
        "wwv_equatorial_pacific_m3": "wwv",
        "thermocline_tilt_m": "tilt_m",
        "ssh_nino34_mean_m": "ssh_m",
    }
    base = phys[list(cols)].rename(columns=cols)
    # Escopo Fase 3 = diagnostico fisico do Pacifico (Nino 3.4). Indices
    # atlanticos foram removidos: controles inter-bacia sao materia da Fase 4
    # (teleconexao Brasil), nao do diagnostico fisico do Pacifico.
    dhwv = load_dhw_variants()
    # DHW principal = HotSpot diario da anomalia SSTA >=0.5 C, acumulado em
    # janela movel de 12 semanas. A flag de persistencia exige 20 semanas
    # consecutivas de media movel 12 semanas >=0.5 C.
    dhw = dhwv[["dhw_cweek_0p5_12w", "oni_12w_mean_c", "elnino_thermal_persistent_20w"]].copy()
    atmo = load_atmo()
    tau_raw = zonal_wind_stress_proxy(atmo["atm_10m_u_component_of_wind"])
    tau = daily_doy_anomaly(tau_raw).rename("tau_x_anom_nino34_pa").to_frame()
    daily = base.join([dhw, tau], how="outer")
    weekly = to_weekly(daily)
    weekly.index.name = "week_ending_sunday"
    return weekly


def sources_note() -> pd.DataFrame:
    return pd.DataFrame([
        (
            "nino34_ssta",
            "OISST v2.1 local",
            "anomalia diaria 1991-2020, agregada para W-SUN",
            "1981-09+",
            "C",
        ),
        (
            "ssta_equatorial_lon_hovmoller",
            "OISST v2.1 local",
            "anomalia diaria por longitude 1991-2020, faixa 2S-2N",
            "1981-09+",
            "C",
        ),
        (
            "d20_m / ohc_* / wwv / tilt_m / ssh_m",
            "UFS 1981-92 (ponte) -> GLORYS12 1993+ -> GLO12 cauda",
            "valor fisico/indice original; nao e anomalia climatologica na matriz semanal",
            "sensibilidade 1993+",
            "m / J m-2 / m3 / m / m",
        ),
        (
            "dhw_cweek_0p5_12w",
            "derivado da SSTA OISST (HotSpot diario >=0.5C, acumulo 12 sem)",
            "anomalias <0.5C descartadas; soma positiva em C-weeks",
            "valido 1981-11+",
            "C-weeks",
        ),
        (
            "oni_12w_mean_c / elnino_thermal_persistent_20w",
            "derivado da SSTA OISST",
            "media movel 12 semanas e flag >=0.5C por 20 semanas consecutivas",
            "valido 1981-11+",
            "C / 0-1",
        ),
        (
            "tau_x_anom_nino34_pa",
            "ERA5 u10 caixa Nino 3.4 (proxy; referencia desejada: Nino 4)",
            "anomalia diaria 1991-2020 do proxy de estresse zonal do vento",
            "1981+",
            "Pa",
        ),
    ], columns=["variavel", "fonte", "forma", "janela_real", "unidade"])


def add_event_shading(ax, events: pd.DataFrame, color="#f4c7c3", alpha=0.5):
    for _, ev in events.iterrows():
        ax.axvspan(ev["event_start"], ev["event_end"], color=color, alpha=alpha, lw=0)


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> Path:
    path = STATS / name
    df.to_csv(path, index=index)
    print(f"[tabela] {path.relative_to(ROOT)}")

    return path


def save_fig(fig, name):
    """Salva figura de forma robusta a inconsistencias do FS montado:
    escreve num arquivo local temporario e copia para FIGS com ate 5 tentativas."""
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
        if last: raise last
    print(f"[figura] {path.relative_to(ROOT)}")
    return path


CAIXAS = {
    "nino34": "Nino 3.4 NOAA/CPC (5N-5S, 170W-120W)",
    "nino4": "Nino 4 NOAA/CPC (5N-5S, 160E-150W)",
    "equatorial": "Banda diagnostica equatorial (2S-2N, 120E-80W)",
}


VAR_LABELS = {
    "nino34_ssta": "SSTA Nino 3.4 (C)",
    "d20_m": "D20 / profundidade da termoclina (m)",
    "ohc_0_300": "OHC 0-300 m (J m-2)",
    "ohc_0_700": "OHC 0-700 m (J m-2)",
    "wwv": "WWV Pacifico equatorial (m3)",
    "tilt_m": "Tilt da termoclina (m)",
    "ssh_m": "SSH Nino 3.4 (m)",
    "dhw_cweek_0p5_12w": "DHW C-week >=0.5C (12 sem)",
    "oni_12w_mean_c": "Media SSTA 12 sem (C)",
    "elnino_thermal_persistent_20w": "Persistencia termica 20 sem",
    "tau_x_anom_nino34_pa": "tau_x anom. Nino 3.4 (Pa)",
}

VAR_SHORT = {
    "nino34_ssta": "SSTA",
    "d20_m": "D20",
    "ohc_0_300": "OHC0-300",
    "ohc_0_700": "OHC0-700",
    "wwv": "WWV",
    "tilt_m": "Tilt",
    "ssh_m": "SSH",
    "dhw_cweek_0p5_12w": "DHW >=0.5C",
    "oni_12w_mean_c": "SSTA 12w",
    "elnino_thermal_persistent_20w": "Persist. 20w",
    "tau_x_anom_nino34_pa": "tau_x anom.",
}


def var_label(name: str, *, short: bool = False) -> str:
    labels = VAR_SHORT if short else VAR_LABELS
    return labels.get(name, name)


def lon_label(lon: float) -> str:
    lon = float(lon)
    if lon == 180:
        return "180"
    if lon < 180:
        return f"{int(round(lon))}E"
    return f"{int(round(360 - lon))}W"


def format_lon_axis(ax, *, xlabel: str = "Longitude oficial (W/E; oeste -> leste)") -> None:
    ticks = [120, 160, 200, 240, 280]
    ax.set_xlim(120, 280)
    ax.set_xticks(ticks)
    ax.set_xticklabels([lon_label(t) for t in ticks], fontsize=8)
    ax.set_xlabel(xlabel)


def add_nino34_lon_band(ax, *, label: bool = True) -> None:
    ax.axvspan(190, 240, color="#000000", alpha=0.06, lw=0)
    ax.axvline(190, color="k", ls="--", lw=0.7)
    ax.axvline(240, color="k", ls="--", lw=0.7)
    if label:
        ax.text(
            215,
            0.985,
            "Nino 3.4\n170W-120W",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )


def add_note(ax, text: str, *, loc: str = "lower right") -> None:
    anchors = {
        "lower right": (0.985, 0.02, "right", "bottom"),
        "upper right": (0.985, 0.98, "right", "top"),
        "lower left": (0.015, 0.02, "left", "bottom"),
        "upper left": (0.015, 0.98, "left", "top"),
    }
    x, y, ha, va = anchors.get(loc, anchors["lower right"])
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#888", "alpha": 0.86},
    )


def stamp_caption(fig, *, variavel, area, periodo, fonte, n=None, extra=None):
    partes = [f"Variavel: {variavel}", f"Area: {area}", f"Periodo: {periodo}", f"Fonte: {fonte}"]
    if n:
        partes.append(f"n={n}")
    if extra:
        partes.append(extra)
    fig.text(0.5, -0.04, " | ".join(partes), ha="center", va="top", fontsize=6.5, color="#444", wrap=True)
