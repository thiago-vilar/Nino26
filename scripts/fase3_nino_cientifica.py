"""FASE3-NINO reconstruída: ciclo de vida do El Niño com critério ONI.

Esta implementação separa rigorosamente três objetos que antes foram
indevidamente confundidos: (1) ocorrência de El Niño, definida pelo critério
ONI compatível; (2) semanas de alta intensidade (P90), usadas apenas como
estratificação; e (3) quatro fases diagnósticas retrospectivas.  O módulo é
consumido pelos notebooks em ``notebooks/fase3_nino/cientifica``.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import friedmanchisquare, spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nino_brasil.data.phase2_master import PHYSICAL_COLUMNS, VARIABLE_SPECS  # noqa: E402


PHASES = ("genese", "crescimento", "faixa_de_pico", "decaimento")
PHASE_LABELS = {
    "genese": "Gênese / pré-condicionamento",
    "crescimento": "Crescimento acoplado",
    "faixa_de_pico": "Faixa de pico",
    "decaimento": "Decaimento / descarga",
}
PHASE_COLORS = {
    "genese": "#4C78A8",
    "crescimento": "#F58518",
    "faixa_de_pico": "#E45756",
    "decaimento": "#54A24B",
}
FIGURE_DPI = int(os.environ.get("FASE3_NINO_FIGURE_DPI", "700"))
NUMERIC_ROOT = ROOT / "data" / "processed" / "numeric-tables" / "fase3_nino_cientifica"
FIGURE_ROOT = ROOT / "data" / "processed" / "figures" / "fase3_nino_cientifica"
FEATURE_ROOT = ROOT / "data" / "processed" / "parquet" / "features"
ZARR_ROOT = ROOT / "data" / "processed" / "zarr"
SPATIAL_CACHE = ROOT / "data" / "interim" / "fase3_nino_ssta_semanal_1deg.nc"
WIND_CACHE = ROOT / "data" / "interim" / "fase3_nino_vento_semanal_nino34.nc"

SPECS = {item.name: item for item in VARIABLE_SPECS}
OCEAN_ABSOLUTE = {
    name for name, spec in SPECS.items() if spec.representation_source_adjusted == "source_seasonal_anomaly_detrended"
}
FAMILY = {}
for name in PHYSICAL_COLUMNS:
    if name == "nino34_ssta":
        FAMILY[name] = "SST alvo"
    elif name in {"d20_m", "tilt_m", "tilt_slope", "ssh_m", "wwv"}:
        FAMILY[name] = "termoclina / recarga"
    elif name.startswith("ohc") or name.startswith("t"):
        FAMILY[name] = "calor subsuperficial"
    elif name in {"tau_x_anom", "u10_anom", "v10_anom", "u850_anom", "u200_anom"}:
        FAMILY[name] = "vento / circulação"
    else:
        FAMILY[name] = "atmosfera / fluxo"


@dataclass(frozen=True)
class Core:
    raw: pd.DataFrame
    anomaly: pd.DataFrame
    zscore: pd.DataFrame
    contract: pd.DataFrame
    monthly_oni: pd.DataFrame
    events: pd.DataFrame
    lifecycle: pd.DataFrame
    p90: float
    p90_windows: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialisable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def write_table(frame: pd.DataFrame, code: str, name: str, *, description: str) -> Path:
    NUMERIC_ROOT.mkdir(parents=True, exist_ok=True)
    path = NUMERIC_ROOT / f"{code}_{name}.csv"
    frame.to_csv(path, index=False)
    metadata = {
        "artifact_type": "numeric_table",
        "phase": "FASE3-NINO",
        "notebook": code,
        "description": description,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "sha256": _sha256(path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=_serialisable), encoding="utf-8")
    return path


def write_figure(fig: plt.Figure, code: str, name: str, *, tables: Iterable[Path], description: str) -> Path:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    path = FIGURE_ROOT / f"{code}_{name}.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    metadata = {
        "artifact_type": "figure",
        "phase": "FASE3-NINO",
        "notebook": code,
        "description": description,
        "dpi": FIGURE_DPI,
        "source_tables": [str(item) for item in tables],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _weekly_climatology_anomaly(series: pd.Series) -> pd.Series:
    """Remove ciclo sazonal usando a climatologia 1991–2020, suavizada circularmente."""
    valid = series.dropna()
    reference = valid.loc["1991-01-01":"2020-12-31"]
    if len(reference) < 104:
        reference = valid
    week = reference.index.isocalendar().week.astype(int)
    clim = reference.groupby(week).mean().reindex(range(1, 54))
    expanded = pd.concat([clim.tail(3), clim, clim.head(3)], ignore_index=True)
    smooth = expanded.rolling(5, center=True, min_periods=1).mean().iloc[3:-3].to_numpy()
    lookup = pd.Series(smooth, index=range(1, 54))
    return series - series.index.isocalendar().week.astype(int).map(lookup).to_numpy()


def _load_weekly_master() -> pd.DataFrame:
    path = FEATURE_ROOT / "nino34_master_weekly.csv"
    master = pd.read_csv(path, parse_dates=["week_ending_sunday"])
    master = master.set_index("week_ending_sunday").sort_index()
    master.index.name = "time"
    missing = set(PHYSICAL_COLUMNS).difference(master.columns)
    if missing:
        raise KeyError(f"Matriz semanal sem as 31 variáveis: {sorted(missing)}")
    return master.loc[:, list(PHYSICAL_COLUMNS)].apply(pd.to_numeric, errors="coerce")


def _transform_master(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anomaly = pd.DataFrame(index=raw.index)
    rows: list[dict[str, object]] = []
    for variable in PHYSICAL_COLUMNS:
        spec = SPECS[variable]
        if variable in OCEAN_ABSOLUTE:
            anomaly[variable] = _weekly_climatology_anomaly(raw[variable])
            operation = "anomalia_semanal_1991_2020_suavizada"
        else:
            anomaly[variable] = raw[variable]
            operation = "anomalia_de_origem_preservada"
        reference = anomaly.loc["1991-01-01":"2020-12-31", variable].dropna()
        if len(reference) < 104:
            reference = anomaly[variable].dropna()
        mean = float(reference.mean())
        sd = float(reference.std(ddof=0))
        rows.append({
            "variavel": variable, "fonte": spec.source, "unidade": spec.units,
            "sentido_positivo": spec.positive, "representacao_entrada": spec.representation_source_adjusted,
            "operacao": operation, "periodo_referencia": "1991-2020 quando disponível; histórico completo como fallback",
            "n_validos": int(anomaly[variable].notna().sum()), "media_referencia": mean, "dp_referencia": sd,
        })
    zscore = anomaly.copy()
    for row in rows:
        variable = str(row["variavel"])
        sd = float(row["dp_referencia"])
        zscore[variable] = (anomaly[variable] - float(row["media_referencia"])) / sd if sd > 0 else np.nan
    return anomaly, zscore, pd.DataFrame(rows)


def _contiguous_windows(mask: pd.Series, label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    run = (mask.ne(mask.shift(fill_value=False))).cumsum()
    number = 0
    for _, group in mask.groupby(run):
        if not bool(group.iloc[0]):
            continue
        number += 1
        rows.append({"janela_id": f"{label}_{number:03d}", "inicio": group.index.min(), "fim": group.index.max(), "duracao_semanas": int(len(group))})
    return pd.DataFrame(rows)


def _detect_oni_events(monthly: pd.DataFrame, *, peak_fraction: float = 0.90) -> pd.DataFrame:
    signal = monthly.set_index("time")["oni_local_c"].dropna().sort_index()
    warm = signal.ge(0.5)
    runs = (warm.ne(warm.shift(fill_value=False))).cumsum()
    rows: list[dict[str, object]] = []
    for _, part in warm.groupby(runs):
        if not bool(part.iloc[0]) or len(part) < 5:
            continue
        dates = part.index
        values = signal.loc[dates]
        peak_time = values.idxmax()
        peak = float(values.max())
        band_mask = values.ge(peak * peak_fraction)
        band_runs = (band_mask.ne(band_mask.shift(fill_value=False))).cumsum()
        containing = next(group for _, group in band_mask.groupby(band_runs) if bool(group.iloc[0]) and peak_time in group.index)
        start, end = dates.min(), dates.max()
        suffix = f"{start.year}_{end.year}" if start.year != end.year else str(start.year)
        rows.append({
            "event_id": f"ELNINO_ONI_{suffix}", "inicio_oni_mes": start, "fim_oni_mes": end,
            "pico_oni_mes": peak_time, "oni_pico_c": peak, "duracao_meses": int(len(values)),
            "inicio_faixa_pico_mes": containing.index.min(), "fim_faixa_pico_mes": containing.index.max(),
            "fracao_faixa_pico": peak_fraction,
            "criterio_evento": "ONI local OISST: média móvel centrada de 3 meses >= +0,5 °C por >=5 estações sobrepostas",
            "rotulo_disponivel_na_origem": False,
        })
    return pd.DataFrame(rows)


def _week_on_or_after(value: pd.Timestamp) -> pd.Timestamp:
    return pd.date_range(pd.Timestamp(value), pd.Timestamp(value) + pd.Timedelta(days=6), freq="W-SUN")[0]


def _week_on_or_before(value: pd.Timestamp) -> pd.Timestamp:
    return pd.date_range(pd.Timestamp(value) - pd.Timedelta(days=6), pd.Timestamp(value), freq="W-SUN")[-1]


def build_lifecycle(events: pd.DataFrame, index: pd.DatetimeIndex, *, genesis_weeks: int = 26) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        onset = _week_on_or_after(pd.Timestamp(event.inicio_oni_mes))
        end = _week_on_or_before(pd.Timestamp(event.fim_oni_mes) + pd.offsets.MonthEnd(1))
        peak_start = _week_on_or_after(pd.Timestamp(event.inicio_faixa_pico_mes))
        peak_end = _week_on_or_before(pd.Timestamp(event.fim_faixa_pico_mes) + pd.offsets.MonthEnd(1))
        ranges = {
            "genese": (onset - pd.Timedelta(weeks=genesis_weeks), onset - pd.Timedelta(weeks=1)),
            "crescimento": (onset, peak_start - pd.Timedelta(weeks=1)),
            "faixa_de_pico": (peak_start, peak_end),
            "decaimento": (peak_end + pd.Timedelta(weeks=1), end),
        }
        phase_counts = {}
        for phase, (start, finish) in ranges.items():
            dates = index[(index >= start) & (index <= finish)]
            phase_counts[phase] = int(len(dates))
            for date in dates:
                records.append({
                    "time": date, "event_id": event.event_id, "fase": phase, "fase_pt": PHASE_LABELS[phase],
                    "semana_relativa_onset": int((date - onset).days // 7),
                    "rotulo_disponivel_na_origem": False, "modo_rotulo": "diagnostico_retrospectivo",
                })
        event_rows.append({**event._asdict(), "onset_semanal": onset, "fim_semanal": end,
                           "inicio_pico_semanal": peak_start, "fim_pico_semanal": peak_end,
                           **{f"n_{name}_semanas": count for name, count in phase_counts.items()},
                           "quatro_fases_completas": all(phase_counts[name] > 0 for name in PHASES)})
    return pd.DataFrame(event_rows), pd.DataFrame(records).sort_values(["event_id", "time"])


def build_core() -> Core:
    raw = _load_weekly_master()
    anomaly, zscore, contract = _transform_master(raw)
    monthly = pd.read_csv(FEATURE_ROOT / "nino34_monthly_oisst.csv", parse_dates=["time"])
    events = _detect_oni_events(monthly)
    events, lifecycle = build_lifecycle(events, raw.index)
    p90 = float(anomaly["nino34_ssta"].quantile(0.90))
    windows = _contiguous_windows(anomaly["nino34_ssta"].ge(p90), "P90_SSTA")
    return Core(raw, anomaly, zscore, contract, monthly, events, lifecycle, p90, windows)


def _event_phase_means(core: Core) -> pd.DataFrame:
    joined = core.zscore.join(core.anomaly.add_suffix("__anom")).join(core.lifecycle.set_index("time"), how="inner")
    rows: list[dict[str, object]] = []
    for (event_id, phase), group in joined.groupby(["event_id", "fase"], sort=False):
        row: dict[str, object] = {"event_id": event_id, "fase": phase, "n_semanas": int(len(group)),
                                  "n_semanas_p90": int((group["nino34_ssta__anom"] >= core.p90).sum())}
        for variable in PHYSICAL_COLUMNS:
            row[variable] = float(group[variable].mean())
            row[f"{variable}__anom"] = float(group[f"{variable}__anom"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_ci(values: pd.Series, seed: int) -> tuple[float, float]:
    values = values.dropna().to_numpy(dtype=float)
    if len(values) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(1500, len(values)))].mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))  # type: ignore[return-value]


def _phase_statistics(phase_means: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats: list[dict[str, object]] = []
    discrimination: list[dict[str, object]] = []
    for position, variable in enumerate(PHYSICAL_COLUMNS):
        pivot = phase_means.pivot(index="event_id", columns="fase", values=variable).reindex(columns=PHASES)
        complete = pivot.dropna()
        pvalue = np.nan
        w = np.nan
        if len(complete) >= 5:
            test = friedmanchisquare(*(complete[phase] for phase in PHASES))
            pvalue = float(test.pvalue)
            w = float(test.statistic / (len(complete) * (len(PHASES) - 1)))
        discrimination.append({"variavel": variable, "familia": FAMILY[variable], "n_eventos_completos": int(len(complete)),
                               "p_friedman": pvalue, "kendall_w": w})
        for phase in PHASES:
            values = pivot[phase]
            lo, hi = _bootstrap_ci(values, 1000 + position * 10 + PHASES.index(phase))
            stats.append({"fase": phase, "fase_pt": PHASE_LABELS[phase], "variavel": variable,
                          "familia": FAMILY[variable], "media_z_entre_eventos": float(values.mean()),
                          "mediana_z_entre_eventos": float(values.median()), "ic95_inf": lo, "ic95_sup": hi,
                          "n_eventos_independentes": int(values.notna().sum())})
    disc = pd.DataFrame(discrimination)
    valid = disc["p_friedman"].notna()
    order = disc.loc[valid, "p_friedman"].sort_values().index
    n = len(order)
    q = pd.Series(np.nan, index=disc.index)
    running = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        actual_rank = n - rank + 1
        running = min(running, float(disc.loc[idx, "p_friedman"]) * n / actual_rank)
        q.loc[idx] = running
    disc["q_friedman_bh"] = q
    disc["significativo_fdr_05"] = disc["q_friedman_bh"].le(0.05)
    return pd.DataFrame(stats), disc


def _family_groups() -> list[list[str]]:
    return [
        ["d20_m", "tilt_m", "tilt_slope", "ssh_m", "wwv"],
        ["ohc_0_100", "ohc_0_300", "ohc_0_700", "ohc_300_700"],
        ["t50m", "t100m", "t150m", "t200m", "t300m", "t500m", "t700m"],
        ["tau_x_anom", "u10_anom", "v10_anom", "u850_anom", "u200_anom"],
        ["mslp_anom", "tcwv_anom", "slhf_anom", "sshf_anom", "ssr_anom", "str_anom", "omega850_anom", "omega500_anom", "div850_anom"],
    ]


def _plot_history_rows(core: Core, code: str, table: Path) -> list[Path]:
    paths: list[Path] = []
    for number, group in enumerate(_family_groups(), start=1):
        fig, axes = plt.subplots(len(group), 1, figsize=(18, 3.0 * len(group)), sharex=True)
        axes = np.atleast_1d(axes)
        for axis, variable in zip(axes, group):
            axis.plot(core.zscore.index, core.zscore["nino34_ssta"], color="#111111", lw=1.05, alpha=.8, label="SSTA Niño-3.4 (z)")
            axis.plot(core.zscore.index, core.zscore[variable], color="#1f77b4", lw=.95, label=f"{variable} (z)")
            axis.axhline(0, color="0.65", lw=.6)
            axis.set_ylabel(f"{variable}\n(z)")
            axis.grid(axis="y", alpha=.22)
            axis.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
        axes[0].set_title("Série histórica semanal: SSTA Niño-3.4 e variável comparada (uma variável por linha)", loc="left", fontsize=14, weight="bold")
        axes[-1].set_xlabel("Ano")
        fig.tight_layout()
        paths.append(write_figure(fig, code, f"serie_historica_grupo_{number:02d}", tables=[table], description="Comparação temporal padronizada entre a SSTA Niño-3.4 e cada variável; uma série comparada por linha."))
    return paths


def _plot_oni_classification(core: Core, code: str, table: Path) -> Path:
    monthly = core.monthly_oni.set_index("time")
    fig, ax = plt.subplots(figsize=(19, 7.2))
    ax.plot(monthly.index, monthly["oni_local_c"], color="#155E75", lw=1.45, label="ONI local (OISST; média móvel centrada de 3 meses)")
    ax.axhline(.5, color="#C2410C", lw=1.2, ls="--", label="limiar ONI +0,5 °C")
    for event in core.events.itertuples(index=False):
        ax.axvspan(event.inicio_oni_mes, event.fim_oni_mes + pd.offsets.MonthEnd(1), color="#F59E0B", alpha=.20)
        ax.text(event.pico_oni_mes, event.oni_pico_c + .09, event.event_id.replace("ELNINO_ONI_", ""), ha="center", va="bottom", rotation=90, fontsize=8)
    ax.set(title="El Niño por critério ONI: P90 é mostrado separadamente como intensidade, não como definição de evento", ylabel="°C", xlabel="Ano")
    ax.grid(axis="y", alpha=.25); ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return write_figure(fig, code, "serie_oni_eventos", tables=[table], description="Série ONI local, limiar científico e intervalos classificados como eventos El Niño.")


def _plot_p90_stratification(core: Core, code: str, table: Path) -> Path:
    fig, ax = plt.subplots(figsize=(19, 7.2))
    ax.plot(core.anomaly.index, core.anomaly["nino34_ssta"], color="#334155", lw=.95, label="SSTA semanal Niño-3.4")
    ax.axhline(core.p90, color="#BE123C", lw=1.3, ls="--", label=f"P90 histórico = {core.p90:.2f} °C")
    high = core.anomaly["nino34_ssta"].where(core.anomaly["nino34_ssta"].ge(core.p90))
    ax.fill_between(high.index, core.p90, high.to_numpy(), where=high.notna(), color="#E11D48", alpha=.45, label="semanas de intensidade P90")
    for event in core.events.itertuples(index=False):
        ax.axvspan(event.onset_semanal, event.fim_semanal, color="#F59E0B", alpha=.10)
    ax.set(title="Estratificação P90 dentro e fora dos eventos ONI", ylabel="anomalia de SST (°C)", xlabel="Ano")
    ax.grid(axis="y", alpha=.25); ax.legend(ncol=3, loc="upper left", frameon=False)
    fig.tight_layout()
    return write_figure(fig, code, "p90_intensidade", tables=[table], description="P90 como estratificação de intensidade sobreposta aos eventos ONI; não é regra de ocorrência.")


def _plot_composite(phase_means: pd.DataFrame, core: Core, code: str, table: Path) -> list[Path]:
    paths: list[Path] = []
    selected_groups = _family_groups()
    for number, group in enumerate(selected_groups, start=1):
        fig, axes = plt.subplots(len(group), 1, figsize=(16, 2.65 * len(group)), sharex=True)
        axes = np.atleast_1d(axes)
        for axis, variable in zip(axes, group):
            for phase in PHASES:
                values = phase_means[phase_means["fase"].eq(phase)]
                if values.empty:
                    continue
                axis.scatter(np.full(len(values), PHASES.index(phase)), values[variable], s=22, alpha=.55, color=PHASE_COLORS[phase])
                axis.errorbar(PHASES.index(phase), values[variable].mean(), yerr=values[variable].sem(), fmt="o", color=PHASE_COLORS[phase], capsize=3, lw=1.2)
            axis.axhline(0, color="0.65", lw=.65); axis.grid(axis="y", alpha=.22); axis.set_ylabel(f"{variable}\n(z)")
        axes[0].set_title("Composto por evento ONI e fase: média ± erro-padrão; pontos = eventos independentes", loc="left", fontsize=14, weight="bold")
        axes[-1].set_xticks(range(4), [PHASE_LABELS[p] for p in PHASES], rotation=12, ha="right")
        fig.tight_layout()
        paths.append(write_figure(fig, code, f"composto_fases_grupo_{number:02d}", tables=[table], description="Composto de médias evento-fase; cada linha é uma variável para preservar legibilidade."))
    return paths


def _p90_intensity_composite(core: Core) -> pd.DataFrame:
    """Compara, com peso igual por evento, o ciclo ONI inteiro e seu núcleo P90."""
    rows: list[dict[str, object]] = []
    for event in core.events.itertuples(index=False):
        period = core.zscore.loc[(core.zscore.index >= event.onset_semanal) & (core.zscore.index <= event.fim_semanal)]
        warm = period.loc[core.anomaly.loc[period.index, "nino34_ssta"].ge(core.p90)]
        for condition, frame in (("evento_ONI_completo", period), ("nucleo_P90_dentro_ONI", warm)):
            if frame.empty:
                continue
            for variable in PHYSICAL_COLUMNS:
                rows.append({"event_id": event.event_id, "condicao": condition, "variavel": variable,
                             "familia": FAMILY[variable], "media_z": float(frame[variable].mean()), "n_semanas": int(len(frame)),
                             "p90_ssta_c": core.p90})
    return pd.DataFrame(rows)


def _plot_p90_composite(frame: pd.DataFrame, code: str, table: Path) -> list[Path]:
    paths: list[Path] = []
    colors = {"evento_ONI_completo": "#64748B", "nucleo_P90_dentro_ONI": "#BE123C"}
    labels = {"evento_ONI_completo": "evento ONI completo", "nucleo_P90_dentro_ONI": "núcleo P90 dentro do ONI"}
    for number, group in enumerate(_family_groups(), start=1):
        fig, axes = plt.subplots(len(group), 1, figsize=(16, 2.5 * len(group)), sharex=True)
        axes = np.atleast_1d(axes)
        for axis, variable in zip(axes, group):
            sample = frame[frame["variavel"].eq(variable)]
            for position, condition in enumerate(("evento_ONI_completo", "nucleo_P90_dentro_ONI")):
                values = sample[sample["condicao"].eq(condition)]["media_z"]
                lo, hi = _bootstrap_ci(values, 1600 + number * 10 + position)
                axis.errorbar(position, values.mean(), yerr=[[values.mean() - lo], [hi - values.mean()]], fmt="o", capsize=3, color=colors[condition], label=labels[condition])
                axis.scatter(np.full(len(values), position), values, color=colors[condition], alpha=.35, s=18)
            axis.axhline(0, color="0.65", lw=.65); axis.grid(axis="y", alpha=.22); axis.set_ylabel(f"{variable}\n(z)")
        axes[0].set_title("Intensidade P90 dentro de eventos ONI: comparação com o evento completo (uma variável por linha)", loc="left", fontsize=14, weight="bold")
        axes[0].legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
        axes[-1].set_xticks([0, 1], [labels["evento_ONI_completo"], labels["nucleo_P90_dentro_ONI"]])
        fig.tight_layout()
        paths.append(write_figure(fig, code, f"composto_p90_grupo_{number:02d}", tables=[table], description="Comparação de cada variável entre o evento ONI completo e as semanas P90 internas, ponderada igualmente por evento."))
    return paths


def _plot_phase_diagnosis(stats: pd.DataFrame, core: Core, code: str, table: Path) -> list[Path]:
    focus = ["nino34_ssta", "wwv", "d20_m", "ohc_0_300", "ssh_m", "tau_x_anom", "u10_anom", "mslp_anom", "tcwv_anom", "omega500_anom"]
    paths: list[Path] = []
    for phase in PHASES:
        subset = stats[(stats["fase"] == phase) & (stats["variavel"].isin(focus))].copy().sort_values("media_z_entre_eventos")
        fig, ax = plt.subplots(figsize=(14, 8.5))
        ax.errorbar(subset["media_z_entre_eventos"], np.arange(len(subset)),
                    xerr=[subset["media_z_entre_eventos"] - subset["ic95_inf"], subset["ic95_sup"] - subset["media_z_entre_eventos"]],
                    fmt="o", color=PHASE_COLORS[phase], ecolor="#64748B", capsize=3)
        ax.axvline(0, color="0.35", lw=.8); ax.set_yticks(np.arange(len(subset)), subset["variavel"])
        ax.set(xlabel="média entre eventos (z; IC bootstrap 95%)", title=f"{PHASE_LABELS[phase]} — assinatura observada dos marcadores físicos")
        ax.grid(axis="x", alpha=.25); fig.tight_layout()
        paths.append(write_figure(fig, code, f"diagnostico_{phase}", tables=[table], description=f"Assinatura observada da fase {phase}; médias por evento e intervalo bootstrap."))
    return paths


def _phase_justifications(stats: pd.DataFrame) -> pd.DataFrame:
    specification = {
        "genese": ("Pré-condicionamento / recarga", ["wwv", "d20_m", "ohc_0_300", "ssh_m"],
                    "Recarga de volume/energia e aprofundamento relativo da termoclina antecedem a fase quente em parte importante dos eventos.",
                    "Jin (1997); Meinen & McPhaden (2000)"),
        "crescimento": ("Amplificação acoplada", ["nino34_ssta", "tau_x_anom", "u10_anom", "tilt_m", "ohc_0_300"],
                         "Aquecimento de SSTA, relaxamento/alteração zonal do vento e ajuste termoclínico são consistentes com a realimentação de Bjerknes.",
                         "Jin (1997); Timmermann et al. (2018)"),
        "faixa_de_pico": ("Máxima anomalia térmica", ["nino34_ssta", "t50m", "tcwv_anom", "mslp_anom"],
                           "A SSTA e as respostas oceano-atmosfera associadas atingem seu maior sinal médio do ciclo.",
                           "Timmermann et al. (2018)"),
        "decaimento": ("Descarga / enfraquecimento", ["nino34_ssta", "wwv", "d20_m", "tau_x_anom"],
                        "Redução da anomalia térmica e reorganização da recarga/termoclina caracterizam a saída da fase quente.",
                        "Jin (1997); Meinen & McPhaden (2000)"),
    }
    rows = []
    for phase, (name, markers, mechanism, reference) in specification.items():
        observed = stats[(stats["fase"] == phase) & (stats["variavel"].isin(markers))].set_index("variavel")
        values = "; ".join(f"{variable}={observed.loc[variable, 'media_z_entre_eventos']:+.2f} z" for variable in markers if variable in observed.index)
        rows.append({"fase": phase, "classificacao": name, "marcadores_primarios": " | ".join(markers),
                     "evidencia_observada_media_z": values, "interpretacao_fisica_hipotetica": mechanism,
                     "referencia_mecanismo": reference, "limite_inferencial": "Descrição observacional por evento; não estabelece causalidade."})
    return pd.DataFrame(rows)


def _plot_phase_cycle(stats: pd.DataFrame, code: str, table: Path) -> Path:
    variables = ["nino34_ssta", "wwv", "d20_m", "ohc_0_300", "tau_x_anom", "u10_anom", "mslp_anom", "tcwv_anom"]
    fig, axes = plt.subplots(len(variables), 1, figsize=(16, 2.55 * len(variables)), sharex=True)
    for axis, variable in zip(axes, variables):
        sample = stats[stats["variavel"].eq(variable)].set_index("fase").reindex(PHASES)
        mean = sample["media_z_entre_eventos"]
        axis.plot(range(4), mean, color="#0F766E", marker="o", lw=1.8)
        axis.fill_between(range(4), sample["ic95_inf"].to_numpy(), sample["ic95_sup"].to_numpy(), color="#0F766E", alpha=.14)
        axis.axhline(0, color="0.55", lw=.65); axis.grid(axis="y", alpha=.22); axis.set_ylabel(f"{variable}\n(z)")
    axes[0].set_title("Ciclo de vida observado: SSTA e marcadores físicos por fase (média entre eventos; faixa = IC 95%)", loc="left", fontsize=14, weight="bold")
    axes[-1].set_xticks(range(4), [PHASE_LABELS[item] for item in PHASES], rotation=12, ha="right")
    fig.tight_layout()
    return write_figure(fig, code, "ciclo_quatro_fases_marcadores", tables=[table], description="Evolução de SSTA e marcadores físicos nas quatro fases, uma variável por linha.")


def _reduce_variables(phase_means: pd.DataFrame, discrimination: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictors = [item for item in PHYSICAL_COLUMNS if item != "nino34_ssta"]
    matrix = phase_means[predictors].copy()
    corr = matrix.corr(method="spearman", min_periods=8).fillna(0)
    distance = (1 - corr.abs()).copy()
    values = distance.to_numpy(copy=True)
    np.fill_diagonal(values, 0)
    distance = pd.DataFrame(values, index=distance.index, columns=distance.columns)
    clusters = fcluster(linkage(squareform(distance.values, checks=False), method="average"), t=.15, criterion="distance")
    phase_signal = phase_means.melt(id_vars=["event_id", "fase"], value_vars=predictors, var_name="variavel", value_name="z").groupby(["variavel", "fase"])["z"].mean().unstack("fase").reindex(columns=PHASES)
    disc = discrimination.set_index("variavel")
    rows: list[dict[str, object]] = []
    for variable, cluster in zip(predictors, clusters):
        row = {"variavel": variable, "familia": FAMILY[variable], "cluster_correlacao": int(cluster),
               "kendall_w": float(disc.loc[variable, "kendall_w"]), "q_friedman_bh": float(disc.loc[variable, "q_friedman_bh"]),
               "sinal_maximo_abs": float(phase_signal.loc[variable].abs().max()),
               "fase_de_maior_sinal": str(phase_signal.loc[variable].abs().idxmax())}
        rows.append(row)
    ranking = pd.DataFrame(rows)
    ranking["score_descritivo"] = ranking["kendall_w"].fillna(0) * ranking["sinal_maximo_abs"].fillna(0)
    ranking["representante_cluster"] = False
    for cluster, group in ranking.groupby("cluster_correlacao"):
        chosen = group.sort_values(["score_descritivo", "variavel"], ascending=[False, True]).index[0]
        ranking.loc[chosen, "representante_cluster"] = True
    candidates = ranking[ranking["representante_cluster"]].copy()
    # O conjunto do ciclo é deliberadamente pequeno: até dois marcadores por
    # família física, evitando que variáveis pouco correlacionadas ocupem uma
    # figura sem ganhar interpretação adicional.
    selected = (candidates.sort_values("score_descritivo", ascending=False)
                .groupby("familia", as_index=False, group_keys=False).head(2)
                .sort_values("score_descritivo", ascending=False).copy())
    selected["motivo"] = "Representante do cluster |rho Spearman| >= 0,85 e entre os dois maiores scores da família física; SSTA alvo excluída."
    return ranking.sort_values(["cluster_correlacao", "score_descritivo"], ascending=[True, False]), selected.sort_values("score_descritivo", ascending=False)


def _plot_reduction(selected: pd.DataFrame, phase_means: pd.DataFrame, code: str, table: Path) -> Path:
    variables = selected["variavel"].tolist()
    long = phase_means.melt(id_vars=["event_id", "fase"], value_vars=variables, var_name="variavel", value_name="z")
    fig, axes = plt.subplots(len(variables), 1, figsize=(16, max(6, 2.25 * len(variables))), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, variable in zip(axes, variables):
        sample = long[long["variavel"].eq(variable)]
        means = sample.groupby("fase")["z"].mean().reindex(PHASES)
        axis.plot(range(4), means, marker="o", color="#0F766E", lw=1.8)
        axis.axhline(0, color="0.65", lw=.65); axis.grid(axis="y", alpha=.22); axis.set_ylabel(f"{variable}\n(z)")
    axes[0].set_title("Ciclo de vida reduzido: representantes não redundantes (uma variável por linha)", loc="left", fontsize=14, weight="bold")
    axes[-1].set_xticks(range(4), [PHASE_LABELS[p] for p in PHASES], rotation=12, ha="right")
    fig.tight_layout()
    return write_figure(fig, code, "ciclo_vida_reduzido", tables=[table], description="Trajetória por fase das variáveis representantes após redução de redundância.")


def _plot_reduction_audit(ranking: pd.DataFrame, selected: pd.DataFrame, code: str, table: Path) -> Path:
    ordered = ranking.sort_values("score_descritivo", ascending=True).copy()
    chosen = set(selected["variavel"])
    palette = {"termoclina / recarga": "#4C78A8", "calor subsuperficial": "#F58518", "vento / circulação": "#54A24B", "atmosfera / fluxo": "#B279A2"}
    colors = [palette.get(item, "#64748B") for item in ordered["familia"]]
    fig, ax = plt.subplots(figsize=(15, 13))
    ax.hlines(ordered["variavel"], 0, ordered["score_descritivo"], color="#CBD5E1", lw=1.2)
    ax.scatter(ordered["score_descritivo"], ordered["variavel"], s=[75 if item in chosen else 28 for item in ordered["variavel"]], c=colors,
               marker="D", edgecolor=["#111827" if item in chosen else "none" for item in ordered["variavel"]], linewidth=.8)
    ax.set(title="Auditoria da redução: score de transição, redundância e conjunto final", xlabel="Kendall W × maior |média por fase| (z)", ylabel="variável")
    ax.grid(axis="x", alpha=.25)
    ax.text(.99, .02, "Losango com borda = selecionada para o ciclo\nCor = família física", transform=ax.transAxes, ha="right", va="bottom", fontsize=10)
    fig.tight_layout()
    return write_figure(fig, code, "auditoria_reducao_variaveis", tables=[table], description="Auditoria visual do ranking de redução; marcadores escolhidos para o ciclo são destacados.")


def _pca_by_phase(phase_means: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictors = [item for item in PHYSICAL_COLUMNS if item != "nino34_ssta"]
    variances: list[dict[str, object]] = []; loadings: list[dict[str, object]] = []; scores: list[dict[str, object]] = []
    for phase in PHASES:
        sample = phase_means[phase_means["fase"].eq(phase)].set_index("event_id")[predictors]
        clean = SimpleImputer(strategy="median").fit_transform(sample)
        scaled = StandardScaler().fit_transform(clean)
        n_components = min(4, scaled.shape[0], scaled.shape[1])
        model = PCA(n_components=n_components, random_state=42).fit(scaled)
        for number, ratio in enumerate(model.explained_variance_ratio_, start=1):
            variances.append({"fase": phase, "componente": f"PC{number}", "variancia_explicada": float(ratio), "variancia_acumulada": float(model.explained_variance_ratio_[:number].sum()), "n_eventos": int(len(sample))})
            for variable, loading in zip(predictors, model.components_[number - 1]):
                loadings.append({"fase": phase, "componente": f"PC{number}", "variavel": variable, "familia": FAMILY[variable], "loading": float(loading)})
        transformed = model.transform(scaled)
        for event_id, row in zip(sample.index, transformed):
            scores.append({"fase": phase, "event_id": event_id, **{f"PC{i+1}": float(value) for i, value in enumerate(row)}})
    return pd.DataFrame(variances), pd.DataFrame(loadings), pd.DataFrame(scores)


def _plot_pca(variance: pd.DataFrame, loadings: pd.DataFrame, scores: pd.DataFrame, code: str, tables: list[Path]) -> list[Path]:
    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(14, 7))
    for phase in PHASES:
        data = variance[variance["fase"].eq(phase)]
        ax.plot(data["componente"], data["variancia_acumulada"] * 100, marker="o", lw=1.7, label=PHASE_LABELS[phase], color=PHASE_COLORS[phase])
    ax.set(title="PCA multivariada por fase: variância acumulada", ylabel="variância explicada acumulada (%)", xlabel="componente")
    ax.set_ylim(0, 100); ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout()
    paths.append(write_figure(fig, code, "pca_scree_por_fase", tables=tables, description="Comparação da variância acumulada da PCA multivariada em cada fase."))
    for phase in PHASES:
        subset = loadings[(loadings["fase"] == phase) & (loadings["componente"] == "PC1")].copy()
        subset["abs"] = subset["loading"].abs(); subset = subset.nlargest(12, "abs").sort_values("loading")
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = np.where(subset["loading"] >= 0, "#B91C1C", "#1D4ED8")
        ax.barh(subset["variavel"], subset["loading"], color=colors)
        ax.axvline(0, color="0.2", lw=.8); ax.set(title=f"PCA — {PHASE_LABELS[phase]}: 12 maiores loadings da PC1", xlabel="loading da PC1")
        ax.grid(axis="x", alpha=.25); fig.tight_layout()
        paths.append(write_figure(fig, code, f"pca_loadings_pc1_{phase}", tables=tables, description=f"Loadings da PC1 da PCA multivariada para a fase {phase}."))
    return paths


def _open_oisst_year(year: int):
    import xarray as xr
    path = ZARR_ROOT / "cpc_noaa" / "oisst" / f"sst.day.mean.{year}.zarr"
    return xr.open_zarr(path, consolidated=False)


def _spatial_dates(core: Core) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(core.lifecycle["time"].unique()))


def ensure_ssta_cache(core: Core):
    import xarray as xr
    if SPATIAL_CACHE.exists():
        return xr.open_dataset(SPATIAL_CACHE)
    dates = _spatial_dates(core)
    climatology_sum: dict[int, Any] = {}; climatology_count: dict[int, int] = {}
    for year in range(1991, 2021):
        ds = _open_oisst_year(year)
        field = ds["sst"].sel(lat=slice(-20, 20), lon=slice(120, 280)).coarsen(lat=4, lon=4, boundary="trim").mean()
        monthly = field.groupby("time.month").mean("time").load()
        for month in range(1, 13):
            value = monthly.sel(month=month)
            climatology_sum[month] = value if month not in climatology_sum else climatology_sum[month] + value
            climatology_count[month] = climatology_count.get(month, 0) + 1
        ds.close()
    climatology = xr.concat([climatology_sum[m] / climatology_count[m] for m in range(1, 13)], dim=pd.Index(range(1, 13), name="month"))
    fields = []
    for year, group in pd.Series(dates, index=dates).groupby(dates.year):
        ds = _open_oisst_year(int(year))
        wanted = pd.DatetimeIndex(group.to_numpy())
        available = wanted[wanted.isin(pd.DatetimeIndex(ds.time.values))]
        if len(available):
            field = ds["sst"].sel(time=available, lat=slice(-20, 20), lon=slice(120, 280)).coarsen(lat=4, lon=4, boundary="trim").mean().load()
            baseline = xr.concat(
                [climatology.sel(month=int(pd.Timestamp(value).month)) for value in pd.DatetimeIndex(field.time.values)],
                dim=field.time,
            )
            baseline = baseline.assign_coords(time=field.time)
            anomaly = field - baseline
            fields.append(anomaly.rename("ssta"))
        ds.close()
    result = xr.concat(fields, dim="time").sortby("time").to_dataset(name="ssta")
    result.attrs.update({"baseline": "climatologia mensal OISST 1991-2020", "grid": "1 grau por coarsening de OISST 0,25°", "domain": "20S-20N, 120E-280E"})
    SPATIAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    result.to_netcdf(SPATIAL_CACHE)
    return result


def _phase_map_table(core: Core, spatial) -> pd.DataFrame:
    phase_lookup = core.lifecycle.set_index("time")[["event_id", "fase"]]
    rows: list[dict[str, object]] = []
    for phase in PHASES:
        dates = phase_lookup[phase_lookup["fase"].eq(phase)].index.intersection(pd.DatetimeIndex(spatial.time.values))
        if not len(dates):
            continue
        mean = spatial["ssta"].sel(time=dates).mean("time")
        frame = mean.to_dataframe(name="ssta_composta_c").reset_index()
        frame["fase"] = phase; frame["n_semanas"] = len(dates); frame["n_eventos"] = phase_lookup.loc[dates, "event_id"].nunique()
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _plot_ssta_maps(map_table: pd.DataFrame, code: str, table: Path) -> list[Path]:
    paths: list[Path] = []
    lim = float(np.nanquantile(np.abs(map_table["ssta_composta_c"]), .985))
    for phase in PHASES:
        sample = map_table[map_table["fase"].eq(phase)]
        pivot = sample.pivot(index="lat", columns="lon", values="ssta_composta_c").sort_index()
        fig, ax = plt.subplots(figsize=(16, 7.5))
        image = ax.pcolormesh(pivot.columns, pivot.index, pivot.values, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), shading="auto")
        ax.axvspan(190, 240, color="none", ec="black", lw=1.0); ax.axhline(0, color="0.2", lw=.7)
        ax.set(title=f"Composto espacial de SSTA — {PHASE_LABELS[phase]}", xlabel="longitude (°E)", ylabel="latitude (°)")
        colorbar = fig.colorbar(image, ax=ax, pad=.02); colorbar.set_label("SSTA composta (°C; climatologia mensal 1991–2020)")
        fig.tight_layout()
        paths.append(write_figure(fig, code, f"mapa_ssta_{phase}", tables=[table], description=f"Mapa composto de SSTA OISST na fase {phase}; caixa Niño-3.4 demarcada."))
    return paths


def _spatial_eof(core: Core, spatial) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = core.lifecycle.set_index("time")[["event_id", "fase"]]
    rows: list[dict[str, object]] = []; maps: list[pd.DataFrame] = []
    for phase in PHASES:
        samples = []
        event_ids = []
        for event_id, dates in lookup[lookup["fase"].eq(phase)].groupby("event_id"):
            common = dates.index.intersection(pd.DatetimeIndex(spatial.time.values))
            if len(common):
                samples.append(spatial["ssta"].sel(time=common).mean("time").values.reshape(-1)); event_ids.append(event_id)
        data = np.asarray(samples)
        if len(data) < 4:
            continue
        # A PCA espacial usa somente a covariação observada: células pontualmente
        # ausentes recebem a média entre eventos da própria célula, sem criar
        # sinal anômalo onde não houve observação.
        finite = np.isfinite(data)
        col_mean = np.divide(np.nansum(data, axis=0), finite.sum(axis=0), out=np.zeros(data.shape[1]), where=finite.sum(axis=0) > 0)
        data = np.where(np.isfinite(data), data, col_mean[None, :])
        model = PCA(n_components=min(2, data.shape[0], data.shape[1]), random_state=42).fit(data)
        pattern = model.components_[0].reshape(spatial.sizes["lat"], spatial.sizes["lon"])
        # Convenção de sinal: padrão positivo na média da caixa Niño-3.4.
        lon_mask = (spatial.lon.values >= 190) & (spatial.lon.values <= 240)
        if np.nanmean(pattern[:, lon_mask]) < 0:
            pattern *= -1
        grid = pd.DataFrame(pattern, index=spatial.lat.values, columns=spatial.lon.values).rename_axis("lat").reset_index().melt(id_vars="lat", var_name="lon", value_name="eof1_loading")
        grid["fase"] = phase; grid["variancia_pc1"] = float(model.explained_variance_ratio_[0]); maps.append(grid)
        for event_id, score in zip(event_ids, model.transform(data)[:, 0]):
            rows.append({"fase": phase, "event_id": event_id, "pc1_score": float(score), "variancia_pc1": float(model.explained_variance_ratio_[0])})
    return pd.concat(maps, ignore_index=True), pd.DataFrame(rows)


def _plot_eof_maps(eofs: pd.DataFrame, code: str, tables: list[Path]) -> list[Path]:
    paths: list[Path] = []
    lim = float(np.nanquantile(np.abs(eofs["eof1_loading"]), .99))
    for phase in PHASES:
        sample = eofs[eofs["fase"].eq(phase)]
        if sample.empty:
            continue
        pivot = sample.pivot(index="lat", columns="lon", values="eof1_loading").sort_index()
        fig, ax = plt.subplots(figsize=(16, 7.5))
        image = ax.pcolormesh(pivot.columns, pivot.index, pivot.values, cmap="PuOr_r", norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), shading="auto")
        explained = sample["variancia_pc1"].iloc[0] * 100
        ax.set(title=f"EOF1 de SSTA por evento — {PHASE_LABELS[phase]} ({explained:.1f}% da variância entre eventos)", xlabel="longitude (°E)", ylabel="latitude (°)")
        ax.axvspan(190, 240, color="none", ec="black", lw=1.0); ax.axhline(0, color="0.2", lw=.7)
        fig.colorbar(image, ax=ax, pad=.02, label="loading EOF1 (sinal orientado positivo em Niño-3.4)")
        fig.tight_layout()
        paths.append(write_figure(fig, code, f"mapa_eof1_ssta_{phase}", tables=tables, description=f"Mapa EOF1 de SSTA por evento na fase {phase}."))
    return paths


def _wind_phase_table(core: Core) -> pd.DataFrame:
    # A fonte ERA5 disponível possui médias espaciais no master; esta tabela não inventa cobertura longitudinal.
    joined = core.anomaly[["tau_x_anom", "u10_anom", "v10_anom"]].join(core.lifecycle.set_index("time"), how="inner")
    rows = []
    for (phase, variable), group in joined.melt(id_vars=["event_id", "fase"], value_vars=["tau_x_anom", "u10_anom", "v10_anom"], var_name="variavel", value_name="anomalia").groupby(["fase", "variavel"]):
        per_event = group.groupby("event_id")["anomalia"].mean()
        lo, hi = _bootstrap_ci(per_event, 700 + len(rows))
        rows.append({"fase": phase, "variavel": variable, "media_entre_eventos": float(per_event.mean()), "ic95_inf": lo, "ic95_sup": hi, "n_eventos": int(len(per_event)), "dominio": "caixa Niño-3.4"})
    return pd.DataFrame(rows)


def _open_wind_year(year: int):
    import xarray as xr
    root = ZARR_ROOT / "era5" / "single_levels" / str(year)
    fields = {}
    for variable, token in (("u10", "10m_u_component_of_wind"), ("v10", "10m_v_component_of_wind")):
        paths = sorted(root.rglob(f"*nino34*{token}*{year}*daily.zarr"))
        if not paths:
            paths = sorted(root.rglob(f"*nino34*{year}*daily.zarr"))
        for path in paths:
            dataset = xr.open_zarr(path, consolidated=False)
            if token in dataset:
                fields[variable] = dataset[token]
                break
            dataset.close()
    if set(fields) != {"u10", "v10"}:
        raise FileNotFoundError(f"Campos ERA5 u10/v10 ausentes para {year}")
    return fields


def _monthly_baseline_for_times(climatology, times: pd.DatetimeIndex):
    import xarray as xr
    parts = [climatology.sel(month=int(pd.Timestamp(value).month)).reset_coords("month", drop=True) for value in times]
    return xr.concat(parts, dim=pd.Index(times, name="time"), coords="minimal", compat="override")


def ensure_wind_cache(core: Core):
    """Compõe anomalias u10/v10 no domínio espacial disponível de Niño-3.4."""
    import xarray as xr
    if WIND_CACHE.exists():
        return xr.open_dataset(WIND_CACHE)
    clim_sum: dict[tuple[str, int], Any] = {}; clim_count: dict[tuple[str, int], int] = {}
    for year in range(1991, 2021):
        fields = _open_wind_year(year)
        for key, data in fields.items():
            monthly = data.groupby("time.month").mean("time").load()
            for month in range(1, 13):
                value = monthly.sel(month=month)
                identity = (key, month)
                clim_sum[identity] = value if identity not in clim_sum else clim_sum[identity] + value
                clim_count[identity] = clim_count.get(identity, 0) + 1
    climatology = {
        key: xr.concat([clim_sum[(key, month)] / clim_count[(key, month)] for month in range(1, 13)], dim=pd.Index(range(1, 13), name="month"))
        for key in ("u10", "v10")
    }
    lookup = core.lifecycle.set_index("time")[["event_id", "fase"]]
    samples: list[Any] = []
    for (event_id, phase), frame in lookup.groupby(["event_id", "fase"]):
        dates = pd.DatetimeIndex(frame.index)
        components = {"u10": [], "v10": []}
        for year, date_group in pd.Series(dates, index=dates).groupby(dates.year):
            fields = _open_wind_year(int(year))
            requested = pd.DatetimeIndex(date_group.to_numpy())
            for key, data in fields.items():
                available = requested[requested.isin(pd.DatetimeIndex(data.time.values))]
                if len(available):
                    values = data.sel(time=available).load()
                    baseline = _monthly_baseline_for_times(climatology[key], pd.DatetimeIndex(values.time.values))
                    components[key].append(values - baseline)
        if not components["u10"] or not components["v10"]:
            continue
        u = xr.concat(components["u10"], dim="time", coords="minimal", compat="override").mean("time")
        v = xr.concat(components["v10"], dim="time", coords="minimal", compat="override").mean("time")
        sample = xr.Dataset({"u10_anom_ms": u, "v10_anom_ms": v}).expand_dims(amostra=[len(samples)])
        sample = sample.assign_coords(event_id=("amostra", [event_id]), fase=("amostra", [phase]))
        samples.append(sample)
    if not samples:
        raise RuntimeError("Não houve cobertura ERA5 espacial para os eventos ONI")
    stacked = xr.concat(samples, dim="amostra", coords="minimal", compat="override")
    result = stacked.groupby("fase").mean("amostra")
    result.attrs.update({"baseline": "climatologia mensal ERA5 1991-2020", "domain": "5S-5N, 170W-120W (caixa Niño-3.4)", "unit": "m s-1"})
    WIND_CACHE.parent.mkdir(parents=True, exist_ok=True)
    result.to_netcdf(WIND_CACHE)
    return result


def _wind_map_table(wind_map) -> pd.DataFrame:
    table = wind_map.to_dataframe().reset_index()
    table["speed_anom_ms"] = np.hypot(table["u10_anom_ms"], table["v10_anom_ms"])
    table["dominio"] = "5S-5N, 170W-120W (caixa Niño-3.4; não cobre toda a bacia)"
    return table


def _plot_wind_maps(wind_map, code: str, table: Path) -> list[Path]:
    paths: list[Path] = []
    limit = float(np.nanquantile(np.abs(wind_map["u10_anom_ms"].values), .985))
    for phase in PHASES:
        if phase not in wind_map.fase.values:
            continue
        sample = wind_map.sel(fase=phase)
        fig, ax = plt.subplots(figsize=(16, 7.4))
        image = ax.pcolormesh(sample.longitude, sample.latitude, sample["u10_anom_ms"], cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit), shading="auto")
        stride = 8
        ax.quiver(sample.longitude.values[::stride], sample.latitude.values[::stride], sample["u10_anom_ms"].values[::stride, ::stride], sample["v10_anom_ms"].values[::stride, ::stride], color="#111827", scale=25, width=.0023)
        ax.axhline(0, color="0.2", lw=.7)
        ax.set(title=f"Mapa composto de vento a 10 m — {PHASE_LABELS[phase]} (domínio disponível: Niño-3.4)", xlabel="longitude (°)", ylabel="latitude (°)")
        fig.colorbar(image, ax=ax, pad=.02, label="anomalia u10 (m s⁻¹; climatologia mensal 1991–2020)")
        fig.tight_layout()
        paths.append(write_figure(fig, code, f"mapa_vento_u10_v10_{phase}", tables=[table], description=f"Mapa ERA5 composto u10/v10 na fase {phase}; domínio restrito à caixa Niño-3.4."))
    return paths


def _kelvin_table(core: Core) -> pd.DataFrame:
    path = FEATURE_ROOT / "ssh_equatorial_daily_by_lon_events.parquet"
    ssh = pd.read_parquet(path); ssh.index = pd.to_datetime(ssh.index); ssh = ssh.sort_index()
    records: list[dict[str, object]] = []
    for event in core.events.itertuples(index=False):
        start = event.onset_semanal - pd.Timedelta(weeks=12); end = event.fim_semanal
        section = ssh.loc[(ssh.index >= start) & (ssh.index <= end)]
        if section.empty:
            continue
        reference = ssh.loc[(ssh.index >= start) & (ssh.index < event.onset_semanal)]
        if len(reference) < 30:
            continue
        anomaly = section - reference.mean(axis=0)
        longitudes = anomaly.columns.astype(float).to_numpy()
        west = anomaly.loc[:, (longitudes >= 120) & (longitudes <= 160)].mean(axis=1)
        east = anomaly.loc[:, (longitudes >= 220) & (longitudes <= 260)].mean(axis=1)
        lags = range(0, 91)
        correlations = [(lag, west.corr(east.shift(-lag))) for lag in lags]
        lag, corr = max(correlations, key=lambda item: -np.inf if pd.isna(item[1]) else item[1])
        records.append({"event_id": event.event_id, "inicio": start, "fim": end, "n_dias": len(section), "lag_oeste_para_leste_dias": lag,
                        "correlacao_maxima": corr, "criterio": "SSH relativo à média dos 84 dias pré-onset; máximo de correlação oeste(120-160E)→leste(220-260E)",
                        "interpretacao": "diagnóstico de coerência temporal compatível com propagação; não é detecção individual definitiva de onda de Kelvin"})
    return pd.DataFrame(records)


def _plot_wind(wind: pd.DataFrame, code: str, table: Path) -> Path:
    variables = ["tau_x_anom", "u10_anom", "v10_anom"]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for axis, variable in zip(axes, variables):
        sample = wind[wind["variavel"].eq(variable)].set_index("fase").reindex(PHASES)
        axis.errorbar(range(4), sample["media_entre_eventos"], yerr=[sample["media_entre_eventos"]-sample["ic95_inf"], sample["ic95_sup"]-sample["media_entre_eventos"]], fmt="o-", color="#7C3AED", capsize=4)
        axis.axhline(0, color="0.4", lw=.7); axis.grid(axis="y", alpha=.25); axis.set_ylabel(variable)
    axes[0].set_title("Vento e tensão zonal na caixa Niño-3.4: médias por evento e fase", loc="left", fontsize=14, weight="bold")
    axes[-1].set_xticks(range(4), [PHASE_LABELS[p] for p in PHASES], rotation=12, ha="right")
    fig.tight_layout()
    return write_figure(fig, code, "vento_por_fase", tables=[table], description="Evolução do vento e tensão zonal sobre a caixa Niño-3.4; não é um mapa da bacia inteira.")


def _hovmoller(core: Core) -> tuple[pd.DataFrame, pd.DataFrame]:
    ssta = pd.read_parquet(FEATURE_ROOT / "equatorial_pacific_ssta_weekly_by_lon.parquet")
    ssta.index = pd.to_datetime(ssta.index); ssta = ssta.sort_index()
    frames = []
    phase_rows = []
    for event in core.events.itertuples(index=False):
        onset = event.onset_semanal
        wanted = pd.date_range(onset - pd.Timedelta(weeks=30), onset + pd.Timedelta(weeks=45), freq="W-SUN")
        common = wanted.intersection(ssta.index)
        for date in common:
            frames.append(pd.DataFrame({"semana_relativa": int((date - onset).days // 7), "lon": ssta.columns.astype(float), "ssta_c": ssta.loc[date].to_numpy(), "event_id": event.event_id}))
        lifecycle = core.lifecycle[core.lifecycle["event_id"].eq(event.event_id)]
        for phase in PHASES:
            values = lifecycle[lifecycle["fase"].eq(phase)]["semana_relativa_onset"]
            if len(values): phase_rows.append({"event_id": event.event_id, "fase": phase, "mediana_semana_relativa": float(values.median())})
    composite = pd.concat(frames).groupby(["semana_relativa", "lon"], as_index=False)["ssta_c"].mean()
    return composite, pd.DataFrame(phase_rows)


def _plot_hovmoller(hov: pd.DataFrame, guides: pd.DataFrame, code: str, tables: list[Path]) -> Path:
    pivot = hov.pivot(index="semana_relativa", columns="lon", values="ssta_c").sort_index()
    lim = float(np.nanquantile(np.abs(pivot.to_numpy()), .985))
    fig, ax = plt.subplots(figsize=(18, 11))
    image = ax.pcolormesh(pivot.columns, pivot.index, pivot.values, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), shading="auto")
    for phase in PHASES:
        sample = guides[guides["fase"].eq(phase)]
        if len(sample): ax.axhline(sample["mediana_semana_relativa"].median(), color=PHASE_COLORS[phase], lw=1.5, label=PHASE_LABELS[phase])
    ax.axvspan(190, 240, color="none", ec="black", lw=1.0)
    ax.set(title="Hovmöller composto de SSTA equatorial (2°S–2°N), alinhado ao onset ONI", xlabel="longitude (°E)", ylabel="semanas relativas ao onset ONI")
    ax.legend(loc="upper left", frameon=True); fig.colorbar(image, ax=ax, pad=.02, label="SSTA (°C; climatologia 1991–2020)")
    fig.tight_layout()
    return write_figure(fig, code, "hovmoller_ssta_onset", tables=tables, description="Hovmöller SSTA equatorial alinhado ao onset ONI, com posições medianas das fases.")


def _sensitivity(core: Core) -> pd.DataFrame:
    rows=[]
    for genesis in (13, 26, 39):
        for fraction in (.80, .90, .95):
            events = _detect_oni_events(core.monthly_oni, peak_fraction=fraction)
            events, life = build_lifecycle(events, core.raw.index, genesis_weeks=genesis)
            for phase in PHASES:
                rows.append({"janela_genese_semanas": genesis, "fracao_faixa_pico": fraction, "fase": phase,
                            "mediana_duracao_semanas": float(events[f"n_{phase}_semanas"].median()), "n_eventos": int(len(events)),
                            "configuracao_canonica": genesis == 26 and fraction == .90})
    return pd.DataFrame(rows)


def _plot_sensitivity(frame: pd.DataFrame, code: str, table: Path) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True)
    for axis, phase in zip(axes, PHASES):
        sample = frame[frame["fase"].eq(phase)]
        for fraction in (.80, .90, .95):
            part=sample[sample["fracao_faixa_pico"].eq(fraction)]
            axis.plot(part["janela_genese_semanas"], part["mediana_duracao_semanas"], marker="o", label=f"faixa de pico {fraction:.0%}")
        axis.set_ylabel(f"{phase}\nsemanas"); axis.grid(alpha=.25); axis.legend(frameon=False, ncol=3)
    axes[0].set_title("Sensibilidade das fronteiras: duração mediana por fase", loc="left", fontsize=14, weight="bold")
    axes[-1].set_xlabel("janela de gênese (semanas)"); fig.tight_layout()
    return write_figure(fig, code, "sensibilidade_fronteiras", tables=[table], description="Sensibilidade de duração das fases às escolhas de janela de gênese e banda de pico.")


REFERENCES = [
    {"autor_ano": "NOAA CPC (2026)", "referencia": "Oceanic Niño Index (ONI), ERSSTv5: definição de anomalia Niño-3.4 em média de 3 meses e persistência de 5 estações sobrepostas.", "url": "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php"},
    {"autor_ano": "Huang et al. (2021)", "referencia": "Improvements of the Daily Optimum Interpolation Sea Surface Temperature (DOISST) Version 2.1. Journal of Climate, 34, 2923–2939. doi:10.1175/JCLI-D-20-0166.1.", "url": "https://doi.org/10.1175/JCLI-D-20-0166.1"},
    {"autor_ano": "Meinen & McPhaden (2000)", "referencia": "Observations of Warm Water Volume Changes in the Equatorial Pacific and Their Relationship to El Niño and La Niña. Journal of Climate, 13, 3551–3559.", "url": "https://doi.org/10.1175/1520-0442(2000)013%3C3551:OOWWVC%3E2.0.CO;2"},
    {"autor_ano": "Jin (1997)", "referencia": "An Equatorial Ocean Recharge Paradigm for ENSO. Journal of the Atmospheric Sciences, 54, 811–829.", "url": "https://doi.org/10.1175/1520-0469(1997)054%3C0811:AEORPF%3E2.0.CO;2"},
    {"autor_ano": "Kessler, McPhaden & Weickmann (1995)", "referencia": "Forcing of intraseasonal Kelvin waves in the equatorial Pacific. JGR Oceans, 100, 10613–10631.", "url": "https://doi.org/10.1029/95JC00382"},
    {"autor_ano": "Timmermann et al. (2018)", "referencia": "El Niño–Southern Oscillation complexity. Nature, 559, 535–545.", "url": "https://doi.org/10.1038/s41586-018-0252-6"},
]


def run_notebook(code: str) -> dict[str, Any]:
    core = build_core()
    tables: list[Path] = []; figures: list[Path] = []
    if code == "F3NINO_01":
        coverage = core.contract.copy()
        coverage["inicio_dado"] = [core.raw[item].first_valid_index() for item in coverage["variavel"]]
        coverage["fim_dado"] = [core.raw[item].last_valid_index() for item in coverage["variavel"]]
        t1=write_table(coverage, code, "contrato_e_cobertura_31_variaveis", description="Transformação, unidades e cobertura das 31 variáveis."); tables.append(t1)
        long=core.zscore.reset_index().melt(id_vars="time", value_vars=PHYSICAL_COLUMNS, var_name="variavel", value_name="zscore")
        t2=write_table(long, code, "serie_historica_zscore", description="Série semanal padronizada das 31 variáveis."); tables.append(t2)
        figures += _plot_history_rows(core, code, t2)
    elif code == "F3NINO_02":
        t1=write_table(core.events, code, "catalogo_eventos_oni", description="Catálogo de eventos El Niño pelo critério ONI local compatível."); tables.append(t1)
        t2=write_table(core.p90_windows, code, "janelas_p90_intensidade", description="Janelas semanais P90, usadas somente como intensidade."); tables.append(t2)
        t3=write_table(core.lifecycle, code, "rotulos_retrospectivos_quatro_fases", description="Rótulos semanais retrospectivos das quatro fases nos eventos ONI."); tables.append(t3)
        figures += [_plot_oni_classification(core, code, t1), _plot_p90_stratification(core, code, t2)]
    elif code == "F3NINO_03":
        means=_event_phase_means(core); t1=write_table(means, code, "medias_evento_fase_31_variaveis", description="Média de cada variável por evento ONI e fase."); tables.append(t1)
        stats, disc=_phase_statistics(means); t2=write_table(stats, code, "composto_por_fase_ic95", description="Composto por fase, com evento como unidade independente."); tables.append(t2)
        p90_composite = _p90_intensity_composite(core); t3=write_table(p90_composite, code, "composto_intensidade_p90_dentro_eventos_oni", description="Comparação evento ONI completo versus núcleo P90, com peso igual por evento."); tables.append(t3)
        figures += _plot_composite(means, core, code, t1)
        figures += _plot_p90_composite(p90_composite, code, t3)
    elif code == "F3NINO_04":
        means=_event_phase_means(core); stats, disc=_phase_statistics(means)
        t1=write_table(stats, code, "assinaturas_fisicas_quatro_fases", description="Médias, dispersão e IC bootstrap por fase e variável."); tables.append(t1)
        t2=write_table(disc, code, "discriminacao_friedman_fdr", description="Teste de Friedman entre fases, tamanho de efeito Kendall W e FDR BH."); tables.append(t2)
        t3=write_table(_phase_justifications(stats), code, "classificacao_e_justificativas_fisicas", description="Classificação das quatro fases, marcadores, evidência observada e base física."); tables.append(t3)
        figures += _plot_phase_diagnosis(stats, core, code, t1)
        figures.append(_plot_phase_cycle(stats, code, t1))
    elif code == "F3NINO_05":
        means=_event_phase_means(core); _, disc=_phase_statistics(means); ranking, selected=_reduce_variables(means, disc)
        t1=write_table(ranking, code, "redundancia_e_ranking_variaveis", description="Clusters de redundância e ranking descritivo; SSTA alvo excluída."); tables.append(t1)
        t2=write_table(selected, code, "representantes_nao_redundantes", description="Variáveis representantes não redundantes para leitura do ciclo."); tables.append(t2)
        figures += [_plot_reduction_audit(ranking, selected, code, t1), _plot_reduction(selected, means, code, t2)]
    elif code == "F3NINO_06":
        means=_event_phase_means(core); variance, loadings, scores=_pca_by_phase(means)
        t1=write_table(variance, code, "pca_multivariada_variancia_por_fase", description="Variância explicada da PCA aplicada separadamente por fase."); tables.append(t1)
        t2=write_table(loadings, code, "pca_multivariada_loadings_por_fase", description="Loadings PCA por fase; SSTA alvo excluída."); tables.append(t2)
        t3=write_table(scores, code, "pca_multivariada_scores_por_fase", description="Scores PCA por evento e fase."); tables.append(t3)
        figures += _plot_pca(variance, loadings, scores, code, tables)
        spatial=ensure_ssta_cache(core); eofs, eof_scores=_spatial_eof(core, spatial)
        t4=write_table(eofs, code, "eof1_ssta_mapas_por_fase", description="Padrões espaciais EOF1 de SSTA entre eventos por fase."); tables.append(t4)
        t5=write_table(eof_scores, code, "eof1_ssta_scores_por_evento", description="Scores espaciais EOF1 por evento e fase."); tables.append(t5)
        figures += _plot_eof_maps(eofs, code, [t4,t5])
    elif code == "F3NINO_07":
        spatial=ensure_ssta_cache(core); maps=_phase_map_table(core, spatial)
        t1=write_table(maps, code, "compostos_espaciais_ssta_por_fase", description="Campos compostos de SSTA OISST por fase."); tables.append(t1)
        figures += _plot_ssta_maps(maps, code, t1)
    elif code == "F3NINO_08":
        wind=_wind_phase_table(core); kelvin=_kelvin_table(core)
        t1=write_table(wind, code, "vento_e_tensao_por_fase", description="Vento/tensão na caixa Niño-3.4 por fase."); tables.append(t1)
        t2=write_table(kelvin, code, "coerencia_ssh_oeste_leste_kelvin", description="Diagnóstico de coerência temporal SSH oeste-leste em anos cobertos."); tables.append(t2)
        wind_map = ensure_wind_cache(core); wind_map_table = _wind_map_table(wind_map)
        t3=write_table(wind_map_table, code, "mapas_vento_u10_v10_por_fase", description="Campos ERA5 compostos de vento a 10 m por fase no domínio espacial Niño-3.4."); tables.append(t3)
        figures.append(_plot_wind(wind, code, t1))
        figures += _plot_wind_maps(wind_map, code, t3)
    elif code == "F3NINO_09":
        hov,guides=_hovmoller(core); t1=write_table(hov, code, "hovmoller_ssta_composto", description="Composto longitude-tempo SSTA alinhado ao onset ONI."); tables.append(t1)
        t2=write_table(guides, code, "guias_fases_hovmoller", description="Posições temporais das fases sobre o Hovmöller."); tables.append(t2)
        figures.append(_plot_hovmoller(hov,guides,code,[t1,t2]))
    elif code == "F3NINO_10":
        sensitivity=_sensitivity(core); t1=write_table(sensitivity, code, "sensibilidade_fronteiras_fases", description="Sensibilidade às janelas de gênese e frações de pico."); tables.append(t1)
        figures.append(_plot_sensitivity(sensitivity,code,t1))
    elif code == "F3NINO_11":
        means=_event_phase_means(core); stats,disc=_phase_statistics(means); ranking,selected=_reduce_variables(means,disc)
        summary=pd.DataFrame([{"n_eventos_oni":len(core.events),"p90_ssta_c":core.p90,"n_semanas_historicas":len(core.raw),"primeira_sst_oisst":str(core.monthly_oni.time.min().date()),"criterio_primario":"ONI local compatível; +0,5°C por >=5 estações móveis","uso_p90":"estratificação de intensidade, não definição"}])
        t1=write_table(summary, code, "sintese_metodologica", description="Síntese de critérios e cobertura."); tables.append(t1)
        t2=write_table(selected, code, "variaveis_representantes_sintese", description="Representantes não redundantes para o ciclo de vida."); tables.append(t2)
        t3=write_table(pd.DataFrame(REFERENCES), code, "referencias_bibliograficas", description="Referências que sustentam definições e interpretação."); tables.append(t3)
        figures.append(_plot_reduction(selected,means,code,t2))
    else:
        raise ValueError(f"Notebook não reconhecido: {code}")
    return {"tables":[str(path) for path in tables],"figures":[str(path) for path in figures],"n_eventos_oni":int(len(core.events)),"limiar_p90_c":core.p90,"dpi":FIGURE_DPI}
