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


MASTER_WEEKLY = FEAT / "nino34_master_weekly.csv"
PACIFIC_CORE = ["nino34_ssta", "d20_m", "ohc_0_300", "ohc_0_700", "wwv",
                "tilt_m", "ssh_m", "tau_x_anom_nino34_pa"]


def weekly_matrix() -> pd.DataFrame:
    """Matriz semanal canonica da Fase 3, lida da MATRIZ-MESTRE unificada da Fase 2
    (`nino34_master_weekly.csv`): oceanicas UFS/GLORYS/GLO12 + atmosfericas ERA5
    (Bjerknes), eixo W-SUN 1981-2026. `ocean_source_code` (metadado) fica de fora;
    mantem o nome `tau_x_anom_nino34_pa` que os notebooks esperam. Se o master nao
    existir, cai no caminho legado (8 variaveis)."""
    if MASTER_WEEKLY.exists():
        m = pd.read_csv(MASTER_WEEKLY, parse_dates=["week_ending_sunday"]).set_index("week_ending_sunday")
        m = m.drop(columns=[c for c in ["ocean_source_code"] if c in m.columns])
        if "tau_x_anom" in m.columns:
            m = m.rename(columns={"tau_x_anom": "tau_x_anom_nino34_pa"})
        m.index.name = "week_ending_sunday"
        return m
    phys = load_physical_signal()
    cols = {
        "nino34_ssta": "nino34_ssta", "d20_nino34_mean_m": "d20_m",
        "ohc_0_300_nino34_j_m2": "ohc_0_300", "ohc_0_700_nino34_j_m2": "ohc_0_700",
        "wwv_equatorial_pacific_m3": "wwv", "thermocline_tilt_m": "tilt_m",
        "ssh_nino34_mean_m": "ssh_m",
    }
    base = phys[list(cols)].rename(columns=cols)
    atmo = load_atmo()
    tau_raw = zonal_wind_stress_proxy(atmo["atm_10m_u_component_of_wind"])
    tau = daily_doy_anomaly(tau_raw).rename("tau_x_anom_nino34_pa").to_frame()
    weekly = to_weekly(base.join([tau], how="outer"))
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


def format_lon_axis(ax, *, xlabel: str = "Longitude oficial (120E -> 80W; oeste para leste)") -> None:
    ticks = [120, 140, 160, 180, 200, 220, 240, 260, 280]
    ax.set_xlim(120, 280)
    ax.set_xticks(ticks)
    ax.set_xticklabels([lon_label(t) for t in ticks], fontsize=7.5)
    ax.set_xlabel(xlabel)


def format_lag_axis(ax, *, max_lag: int | None = None) -> None:
    """Padroniza eixo vertical de mapas longitude-lag."""
    if max_lag is None:
        lo, hi = ax.get_ylim()
        max_lag = int(round(max(lo, hi)))
    ticks = [t for t in range(0, max_lag + 1, 13)]
    if ticks[-1] != max_lag:
        ticks.append(max_lag)
    ax.set_yticks(ticks)
    ax.set_ylabel("Lag do precursor (semanas antes da SSTA alvo)")
    ax.set_ylim(max_lag, 0)


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


def peak_precursor_value(weekly: pd.DataFrame, var: str, peak_time, lead_weeks: int, halfwin: int = 2) -> float:
    """Valor do precursor `var` centrado `lead_weeks` semanas antes do pico (media +-halfwin semanas)."""
    idx = weekly.index
    target = pd.to_datetime(peak_time) - pd.Timedelta(weeks=int(lead_weeks))
    i = idx.get_indexer([target], method="nearest")[0]
    lo, hi = max(0, i - halfwin), min(len(idx), i + halfwin + 1)
    return float(weekly[var].iloc[lo:hi].mean())


def loo_peak_hindcast(weekly: pd.DataFrame, events: pd.DataFrame, spec: dict[str, int],
                      target_col: str = "peak_oni_local_c"):
    """Hindcast leave-one-event-out (LOO) do pico ONI local.

    spec = {variavel: lead_semanas}. Para cada evento, ajusta OLS nos demais
    eventos e preve o pico do evento deixado de fora. Baseline honesto =
    climatologia LOO (media dos picos dos eventos de treino). Retorna
    (DataFrame por evento, dict de metricas). Sem ML pesado: regressao linear.
    """
    import numpy as np
    ev = events.reset_index(drop=True)
    X = np.column_stack([
        [peak_precursor_value(weekly, v, p, L) for p in ev["peak_time"]]
        for v, L in spec.items()
    ])
    y = ev[target_col].astype(float).values
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    pred = np.full(len(y), np.nan)
    for i in range(len(y)):
        if not ok[i]:
            continue
        tr = ok.copy(); tr[i] = False
        A = np.column_stack([np.ones(int(tr.sum())), X[tr]])
        beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred[i] = float(np.concatenate(([1.0], X[i])) @ beta)
    res = ev[["event_id", target_col]].copy().rename(columns={target_col: "oni_pico_obs_c"})
    res["oni_pico_prev_loo_c"] = np.round(pred, 3)
    m = np.isfinite(pred)
    obs, prd = y[m], pred[m]
    clim = np.array([float(np.mean(np.delete(obs, i))) for i in range(len(obs))])
    r_loo = float(np.corrcoef(obs, prd)[0, 1]) if int(m.sum()) > 2 else float("nan")
    mae = float(np.mean(np.abs(prd - obs)))
    mae_clim = float(np.mean(np.abs(clim - obs)))
    metrics = {
        "n_eventos": int(m.sum()),
        "r_loo": round(r_loo, 3),
        "mae_loo_c": round(mae, 3),
        "rmse_loo_c": round(float(np.sqrt(np.mean((prd - obs) ** 2))), 3),
        "mae_climatologia_c": round(mae_clim, 3),
        "skill_vs_climatologia": round(1.0 - mae / mae_clim, 3),
        "residuo_std_c": round(float(np.std(prd - obs, ddof=1)), 3) if int(m.sum()) > 2 else float("nan"),
    }
    return res, metrics


def _event_design_matrix(weekly: pd.DataFrame, events: pd.DataFrame, spec: dict[str, int],
                         target_col: str = "peak_oni_local_c"):
    import numpy as np

    ev = events.reset_index(drop=True)
    X = np.column_stack([
        [peak_precursor_value(weekly, v, p, L) for p in ev["peak_time"]]
        for v, L in spec.items()
    ])
    y = ev[target_col].astype(float).values
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return ev, X, y, ok


def candidate_peak_specs(*, horizons=(8, 12, 15, 20, 26)) -> list[dict]:
    """Grade fisicamente pre-especificada para predizer pico ENSO.

    Mantem poucos graus de liberdade por modelo (<=3 preditores) para n pequeno.
    A selecao do melhor candidato deve ser avaliada por nested LOO, nao pelo
    mesmo LOO usado para escolher o candidato.
    """
    blocks = [
        ("ohc300", ["ohc_0_300"], "calor subsuperficial Nino 3.4"),
        ("ssh", ["ssh_m"], "altura dinamica/proxy de recarga"),
        ("tau_x", ["tau_x_anom_nino34_pa"], "acoplamento vento-superficie"),
        ("d20", ["d20_m"], "profundidade da termoclina"),
        ("tilt", ["tilt_m"], "inclinacao da termoclina"),
        ("wwv", ["wwv"], "volume de agua quente Pacifico equatorial"),
        ("recharge_core", ["ohc_0_300", "ssh_m", "d20_m"], "recarga subsuperficial compacta"),
        ("recharge_tilt", ["ohc_0_300", "tilt_m", "d20_m"], "recarga + inclinacao"),
        ("wind_recharge", ["ohc_0_300", "ssh_m", "tau_x_anom_nino34_pa"], "recarga + vento"),
        ("wwv_recharge", ["ohc_0_300", "d20_m", "wwv"], "recarga + memoria basinwide"),
    ]
    rows = []
    for H in horizons:
        for name, variables, rationale in blocks:
            rows.append({
                "modelo": f"{name}_{int(H)}w",
                "familia": name,
                "horizonte_sem": int(H),
                "variaveis": "+".join(variables),
                "n_preditores": len(variables),
                "racional": rationale,
                "spec": {v: int(H) for v in variables},
            })
    return rows


def select_best_spec_by_loo(weekly: pd.DataFrame, events: pd.DataFrame, candidates: list[dict],
                            target_col: str = "peak_oni_local_c") -> tuple[dict, pd.DataFrame]:
    """Seleciona candidato por LOO dentro do conjunto fornecido.

    Uso correto: chamar esta funcao apenas dentro de um conjunto de treino, ou
    depois de estimar o desempenho do procedimento por `nested_loo_peak_hindcast`.
    """
    import numpy as np

    rows = []
    for cand in candidates:
        _, met = loo_peak_hindcast(weekly, events, cand["spec"], target_col=target_col)
        row = {k: v for k, v in cand.items() if k != "spec"}
        row.update(met)
        row["spec"] = cand["spec"]
        rows.append(row)
    table = pd.DataFrame(rows)
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.sort_values(
        ["skill_vs_climatologia", "mae_loo_c", "n_preditores", "modelo"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    return table.iloc[0].to_dict(), table


def nested_loo_peak_hindcast(weekly: pd.DataFrame, events: pd.DataFrame, candidates: list[dict],
                             target_col: str = "peak_oni_local_c"):
    """Nested leave-one-event-out para avaliar selecao de modelo + regressao.

    Loop externo: deixa um evento fora para teste. Loop interno: nos eventos de
    treino, escolhe o melhor candidato por LOO. O skill final avalia o protocolo
    completo de selecao + ajuste, reduzindo o vies otimista do "flat LOO".
    """
    import numpy as np

    ev = events.reset_index(drop=True)
    y_all = ev[target_col].astype(float).values
    pred = np.full(len(ev), np.nan)
    clim = np.full(len(ev), np.nan)
    selected = []

    for i in range(len(ev)):
        train = ev.drop(index=i).reset_index(drop=True)
        test = ev.iloc[[i]].reset_index(drop=True)
        best, inner_table = select_best_spec_by_loo(weekly, train, candidates, target_col=target_col)
        spec = best["spec"]
        _, Xtr, ytr, oktr = _event_design_matrix(weekly, train, spec, target_col=target_col)
        _, Xte, _, okte = _event_design_matrix(weekly, test, spec, target_col=target_col)

        p = len(spec)
        clim[i] = float(np.nanmean(ytr[oktr])) if oktr.any() else np.nan
        if int(oktr.sum()) <= p + 1 or not bool(okte[0]):
            selected.append({**{k: best[k] for k in best if k != "spec"}, "outer_event_id": ev.loc[i, "event_id"]})
            continue

        A = np.column_stack([np.ones(int(oktr.sum())), Xtr[oktr]])
        beta, *_ = np.linalg.lstsq(A, ytr[oktr], rcond=None)
        pred[i] = float(np.concatenate(([1.0], Xte[0])) @ beta)
        sel = {k: best[k] for k in best if k != "spec"}
        sel.update({
            "outer_event_id": ev.loc[i, "event_id"],
            "inner_best_skill": best.get("skill_vs_climatologia"),
            "inner_rank_rows": len(inner_table),
        })
        selected.append(sel)

    res = ev[["event_id", target_col]].copy().rename(columns={target_col: "oni_pico_obs_c"})
    res["oni_pico_prev_nested_loo_c"] = np.round(pred, 3)
    res["oni_pico_climatologia_treino_c"] = np.round(clim, 3)
    selected_df = pd.DataFrame(selected)
    if not selected_df.empty:
        keep = ["outer_event_id", "modelo", "familia", "horizonte_sem", "variaveis", "n_preditores", "inner_best_skill"]
        res = res.merge(selected_df[[c for c in keep if c in selected_df.columns]],
                        left_on="event_id", right_on="outer_event_id", how="left").drop(columns=["outer_event_id"])

    m = np.isfinite(pred) & np.isfinite(clim) & np.isfinite(y_all)
    obs, prd, base = y_all[m], pred[m], clim[m]
    r_loo = float(np.corrcoef(obs, prd)[0, 1]) if int(m.sum()) > 2 else float("nan")
    mae = float(np.mean(np.abs(prd - obs))) if int(m.sum()) else float("nan")
    mae_clim = float(np.mean(np.abs(base - obs))) if int(m.sum()) else float("nan")
    metrics = {
        "n_eventos": int(m.sum()),
        "r_nested_loo": round(r_loo, 3),
        "mae_nested_loo_c": round(mae, 3),
        "rmse_nested_loo_c": round(float(np.sqrt(np.mean((prd - obs) ** 2))), 3) if int(m.sum()) else float("nan"),
        "mae_climatologia_c": round(mae_clim, 3),
        "skill_vs_climatologia": round(1.0 - mae / mae_clim, 3) if mae_clim and np.isfinite(mae_clim) else float("nan"),
        "residuo_std_c": round(float(np.std(prd - obs, ddof=1)), 3) if int(m.sum()) > 2 else float("nan"),
        "protocolo": "nested leave-one-event-out: inner LOO seleciona candidato; outer LOO avalia evento retido",
    }
    return res, metrics, selected_df


def fit_and_project_peak(weekly: pd.DataFrame, events: pd.DataFrame, spec: dict[str, int],
                         target_col: str = "peak_oni_local_c") -> dict:
    """Ajusta OLS em TODOS os eventos e projeta o pico usando o estado mais recente.

    A leitura e condicional: assume que os valores atuais das variaveis sao os
    precursores de um pico ~lead semanas a frente. A incerteza deve vir do
    residuo LOO correspondente (loo_peak_hindcast com o mesmo spec).
    """
    import numpy as np
    ev = events.reset_index(drop=True)
    X = np.column_stack([
        [peak_precursor_value(weekly, v, p, L) for p in ev["peak_time"]]
        for v, L in spec.items()
    ])
    y = ev[target_col].astype(float).values
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    A = np.column_stack([np.ones(int(ok.sum())), X[ok]])
    beta, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    latest = {}
    for v in spec:
        s = weekly[v].dropna()
        latest[v] = float(s.iloc[-3:].mean())
        last_week = s.index.max()
    x_now = np.concatenate(([1.0], [latest[v] for v in spec]))
    leads = list(spec.values())
    return {
        "pico_projetado_c": round(float(x_now @ beta), 3),
        "ultima_semana_dado": str(pd.to_datetime(last_week).date()),
        "antecedencia_min_sem": int(min(leads)),
        "antecedencia_max_sem": int(max(leads)),
        "janela_pico_projetada_ini": str((pd.to_datetime(last_week) + pd.Timedelta(weeks=min(leads))).date()),
        "janela_pico_projetada_fim": str((pd.to_datetime(last_week) + pd.Timedelta(weeks=max(leads))).date()),
        "valores_atuais": {v: round(latest[v], 4) for v in spec},
    }


def stamp_caption(fig, *, variavel, area, periodo, fonte, n=None, extra=None):
    partes = [f"Variavel: {variavel}", f"Area: {area}", f"Periodo: {periodo}", f"Fonte: {fonte}"]
    if n:
        partes.append(f"n={n}")
    if extra:
        partes.append(extra)
    fig.text(0.5, -0.04, " | ".join(partes), ha="center", va="top", fontsize=6.5, color="#444", wrap=True)
