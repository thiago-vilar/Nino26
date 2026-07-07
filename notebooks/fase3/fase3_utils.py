"""Utilidades compartilhadas dos notebooks da Fase 3 (3A-3G).

Convencoes:
- Eixo canonico semanal: semanas terminando no domingo (W-SUN).
- Tabelas numericas em data/processed/parquet/statistics/phase3*_.csv
- Figuras em data/processed/figures/fase3/
- Nenhuma figura sem saida numerica correspondente.
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


def load_atmo() -> pd.DataFrame:
    df = pd.read_csv(FEAT / "era5_nino34_atmo_cache.csv", parse_dates=["time"])
    return df.set_index("time").sort_index()


def load_events() -> pd.DataFrame:
    ev = pd.read_csv(FEAT / "nino34_oisst_event_reference.csv",
                     parse_dates=["event_start", "event_end", "peak_time"])
    return ev


def load_eqband_weekly() -> pd.DataFrame:
    df = pd.read_parquet(FEAT / "equatorial_pacific_ssta_weekly_by_lon.parquet")
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(float)
    return df


def load_ssh_events() -> pd.DataFrame:
    df = pd.read_parquet(FEAT / "ssh_equatorial_daily_by_lon_events.parquet")
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(float)
    return df


def tau_x_proxy(u10: pd.Series) -> pd.Series:
    """Proxy de estresse zonal do vento: tau_x = rho_a * Cd * |u10| * u10.

    Caveat metodologico: usa u10 medio da caixa Nino 3.4 (cache ERA5 local);
    o protocolo pede tau_x na caixa Nino 4 - substituir quando o recorte
    Nino 4 for materializado. O sinal (leste/oeste) e a fase temporal sao
    preservados; a magnitude absoluta nao deve ser interpretada.
    """
    return (RHO_AIR * CD_NEUTRAL * u10.abs() * u10).rename("tau_x_proxy_nino34_pa")


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
        "sss_nino34_mean": "sss",
    }
    base = phys[list(cols)].rename(columns=cols)
    atl = load_atlantic()[["atl3_ssta", "atl4_ssta", "tna_ssta", "tsa_ssta"]]
    dhw = load_dhw()[["nino34_dhw_12w_cweeks"]].rename(columns={"nino34_dhw_12w_cweeks": "dhw_12w"})
    atmo = load_atmo()
    tau = tau_x_proxy(atmo["atm_10m_u_component_of_wind"]).to_frame()
    daily = base.join([atl, dhw, tau], how="outer")
    weekly = to_weekly(daily)
    weekly.index.name = "week_ending_sunday"
    return weekly


def sources_note() -> pd.DataFrame:
    return pd.DataFrame([
        ("nino34_ssta", "OISST v2.1 local", "1981-09+", "C"),
        ("d20_m / ohc_* / wwv / tilt_m / ssh_m / sss", "UFS 1981-92 (ponte) -> GLORYS12 1993+ -> GLO12 cauda", "sensibilidade 1993+", "m / J m-2 / m3 / m / m / psu"),
        ("atl3/atl4/tna/tsa_ssta", "OISST v2.1 global local (fix nino.py)", "1981-09+", "C"),
        ("dhw_12w", "derivado da SSTA OISST (limiar 1C, 12 semanas)", "valido 1981-11-23+", "C-weeks"),
        ("tau_x_proxy_nino34_pa", "ERA5 u10 caixa Nino 3.4 (proxy; protocolo pede Nino 4)", "1981+", "Pa"),
    ], columns=["variavel", "fonte", "janela_real", "unidade"])


def add_event_shading(ax, events: pd.DataFrame, color="#f4c7c3", alpha=0.5):
    for _, ev in events.iterrows():
        ax.axvspan(ev["event_start"], ev["event_end"], color=color, alpha=alpha, lw=0)


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> Path:
    path = STATS / name
    df.to_csv(path, index=index)
    print(f"[tabela] {path.relative_to(ROOT)}")
    return path


def save_fig(fig, name: str) -> Path:
    path = FIGS / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[figura] {path.relative_to(ROOT)}")
    return path
