"""Reader-facing workflows used by the clean canonical notebooks.

Heavy numerical work stays in the phase runners.  A notebook validates those
outputs, derives a bounded audit table, publishes the matching Fig/Tab pair and
shows conclusions supported by that table.  This keeps notebooks reproducible
without silently repeating long training or pixel calculations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from nino_brasil.stats.phase3_inference import confirmed_friedman_discriminants

from nino_brasil.artifact_codes import (
    notebook_code_for,
    parse_notebook_code,
    table_code,
)
from nino_brasil.notebook_catalog import NOTEBOOK_BY_CODE


PHASE_ORDER = ("genese", "crescimento", "pico", "decaimento")

FAMILY_COLORS = {
    "alvo_termico_nino34": "#7c2d12",
    "oceano_subsuperficie": "#1d4ed8",
    "oceano_superficie": "#0891b2",
    "acoplamento_vento_oceano": "#15803d",
    "atmosfera": "#a16207",
    "outra": "#6b7280",
}


def _lon_label(lon: float) -> str:
    lon = float(lon)
    if lon == 180:
        return "180"
    if lon < 180:
        return f"{int(round(lon))}E"
    return f"{int(round(360 - lon))}W"


def _format_lon_axis(axis, *, xlabel: str = "") -> None:
    ticks = [120, 140, 160, 180, 200, 220, 240, 260, 280]
    axis.set_xlim(120, 280)
    axis.set_xticks(ticks)
    axis.set_xticklabels([_lon_label(t) for t in ticks], fontsize=7)
    if xlabel:
        axis.set_xlabel(xlabel, fontsize=8)


def _add_nino34_lon_band(axis) -> None:
    axis.axvspan(190, 240, color="#000000", alpha=0.07, lw=0)
    axis.axvline(190, color="k", ls="--", lw=0.6)
    axis.axvline(240, color="k", ls="--", lw=0.6)


@dataclass
class WorkflowResult:
    artifacts: pd.DataFrame
    summary: pd.DataFrame
    takeaways: list[str]
    limitations: list[str]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"formato tabular não suportado: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_augmentation_ablation(
    root: Path,
    *,
    phase: int,
    selected_run_ids: set[str],
) -> pd.DataFrame:
    audit_root = root / "data/audit/augmentation_ablation"
    candidates: list[tuple[pd.Timestamp, str, pd.DataFrame]] = []
    for manifest_path in audit_root.glob("*/augmentation_ablation_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runs = manifest.get("runs") or {}
        compared = {
            str((runs.get(name) or {}).get("run_id", ""))
            for name in ("with_augmentation", "without_augmentation")
        }
        output = manifest_path.parent / "augmentation_ablation.csv"
        output_record = (manifest.get("outputs") or {}).get(output.name) or {}
        if (
            manifest.get("status") != "complete"
            or int(manifest.get("phase", -1)) != int(phase)
            or manifest.get("mode") != "official"
            or "" in compared
            or not compared.issubset(selected_run_ids)
            or not output.is_file()
            or _sha256_file(output) != str(output_record.get("sha256", ""))
        ):
            continue
        frame = pd.read_csv(output)
        frame["ablation_manifest"] = str(manifest_path.relative_to(root))
        frame["record_type"] = "augmentation_ablation_metric"
        timestamp = pd.to_datetime(manifest.get("created_at"), utc=True, errors="coerce")
        if pd.isna(timestamp):
            timestamp = pd.Timestamp.min.tz_localize("UTC")
        candidates.append((timestamp, str(manifest.get("model_backend", "default")), frame))
    selected: list[pd.DataFrame] = []
    seen: set[str] = set()
    for _timestamp_value, backend, frame in sorted(candidates, reverse=True, key=lambda item: item[0]):
        if backend in seen:
            continue
        selected.append(frame)
        seen.add(backend)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def _message_figure(title: str, message: str):
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.axis("off")
    axis.text(0.5, 0.58, title, ha="center", va="center", fontsize=14, weight="bold")
    axis.text(0.5, 0.38, message, ha="center", va="center", fontsize=10, wrap=True)
    return fig


def _bar_figure(
    table: pd.DataFrame,
    *,
    label: str,
    value: str,
    title: str,
    top: int = 20,
):
    if table.empty or label not in table or value not in table:
        return _message_figure(title, "A tabela auditada não contém linhas suficientes para o gráfico.")
    data = table[[label, value]].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.dropna().sort_values(value, key=lambda values: values.abs()).tail(top)
    if data.empty:
        return _message_figure(title, "Nenhum valor numérico disponível após a validação.")
    fig, axis = plt.subplots(figsize=(10, max(4.5, 0.32 * len(data) + 1.5)))
    colors = np.where(data[value] >= 0, "#b91c1c", "#2563eb")
    axis.barh(data[label].astype(str), data[value], color=colors)
    axis.axvline(0, color="#111827", lw=0.8)
    axis.set_title(title)
    axis.set_xlabel(value)
    fig.tight_layout()
    return fig


def _phase_profile_figure(table: pd.DataFrame, value: str, title: str):
    if table.empty or "fase" not in table or value not in table:
        return _message_figure(title, "Perfil por fase indisponível.")
    data = table.copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    if "variavel" not in data:
        data["variavel"] = "valor"
    data["fase"] = pd.Categorical(data["fase"], PHASE_ORDER, ordered=True)
    data = data.dropna(subset=["fase", value]).sort_values("fase")
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for variable, group in data.groupby("variavel", observed=True):
        profile = group.groupby("fase", observed=False)[value].mean().reindex(PHASE_ORDER)
        axis.plot(PHASE_ORDER, profile, marker="o", label=str(variable))
    axis.axhline(0, color="#111827", lw=0.8)
    axis.set_title(title)
    axis.set_ylabel(value)
    if data["variavel"].nunique() <= 12:
        axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def _phase4_pixel_figure(table: pd.DataFrame, title: str):
    required = {
        "lat",
        "lon",
        "condicao_fonte",
        "r_no_best_lag_fdr",
        "best_lag_sem_fdr",
    }
    if table.empty or not required.issubset(table.columns):
        return _message_figure(title, "Tabela pixel-a-pixel incompleta para o mapa.")
    figure, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    correlation_artist = lag_artist = None
    for column, phase in enumerate(PHASE_ORDER):
        subset = table.loc[
            table["condicao_fonte"].astype(str).str.endswith(f"_{phase}")
        ].copy()
        lon = pd.to_numeric(subset["lon"], errors="coerce")
        lat = pd.to_numeric(subset["lat"], errors="coerce")
        correlation = pd.to_numeric(subset["r_no_best_lag_fdr"], errors="coerce")
        lag = pd.to_numeric(subset["best_lag_sem_fdr"], errors="coerce")
        for row in (0, 1):
            axes[row, column].scatter(
                lon,
                lat,
                c="#e5e7eb",
                s=2,
                linewidths=0,
                rasterized=True,
            )
        supported = correlation.notna() & lag.notna()
        correlation_artist = axes[0, column].scatter(
            lon[supported],
            lat[supported],
            c=correlation[supported],
            cmap="BrBG",
            vmin=-0.7,
            vmax=0.7,
            s=3,
            linewidths=0,
            rasterized=True,
        )
        lag_artist = axes[1, column].scatter(
            lon[supported],
            lat[supported],
            c=lag[supported],
            cmap="viridis",
            vmin=0,
            vmax=78,
            s=3,
            linewidths=0,
            rasterized=True,
        )
        axes[0, column].set_title(phase)
        axes[1, column].set_xlabel("longitude")
        axes[0, column].set_aspect("equal", adjustable="box")
        axes[1, column].set_aspect("equal", adjustable="box")
    axes[0, 0].set_ylabel("latitude\ncorrelação r (FDR)")
    axes[1, 0].set_ylabel("latitude\nlag semanal (FDR)")
    if correlation_artist is not None:
        figure.colorbar(
            correlation_artist,
            ax=axes[0, :].tolist(),
            shrink=0.78,
            label="r no melhor lag por pixel",
        )
    if lag_artist is not None:
        figure.colorbar(
            lag_artist,
            ax=axes[1, :].tolist(),
            shrink=0.78,
            label="lag (semanas)",
        )
    figure.suptitle(title)
    figure.subplots_adjust(top=0.90, wspace=0.08, hspace=0.12, right=0.92)
    return figure


def _hovmoller_events_figure(
    equatorial_ssta: pd.DataFrame,
    peaks: pd.DataFrame,
    *,
    signal_label: str,
):
    """Painéis Hovmöller SSTA por evento com epicentro e pico de vento marcados."""
    rows = peaks.loc[peaks["status"].eq("ok")].copy()
    if rows.empty:
        return _message_figure(
            f"{signal_label}: Hovmöller por evento",
            "Nenhum evento elegível com dados de SSTA equatorial.",
        )
    for column in ("onset", "pico", "fim", "epicentro_data", "pico_vento_data"):
        rows[column] = pd.to_datetime(rows[column], errors="coerce")
    n_panels = len(rows)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(15, 3.6 * n_rows), sharex=True
    )
    axes_flat = np.atleast_1d(axes).ravel()
    lon = equatorial_ssta.columns.to_numpy(dtype=float)
    mesh = None
    for axis, (_, event) in zip(axes_flat, rows.iterrows()):
        start = event["onset"] - pd.Timedelta(weeks=26)
        segment = equatorial_ssta.loc[start: event["fim"]].dropna(how="all")
        if segment.empty:
            axis.text(0.5, 0.5, "sem dados", transform=axis.transAxes, ha="center")
            continue
        relative = (segment.index - event["pico"]).days / 7.0
        mesh = axis.pcolormesh(
            lon, relative, segment.to_numpy(dtype=float),
            cmap="RdBu_r", vmin=-3, vmax=3, shading="auto", rasterized=True,
        )
        _add_nino34_lon_band(axis)
        _format_lon_axis(axis)
        axis.axhline(0, color="k", lw=0.7, ls="--")
        epic_week = (event["epicentro_data"] - event["pico"]).days / 7.0
        axis.plot(
            event["epicentro_lon"], epic_week,
            marker="*", ms=14, mfc="#ffd700", mec="k", mew=0.9, zorder=6,
        )
        axis.annotate(
            f"epicentro {event['epicentro_lon_rotulo']}\n{event['ssta_extremo_c']:+.1f} °C",
            xy=(event["epicentro_lon"], epic_week), xytext=(8, 8),
            textcoords="offset points", fontsize=6.8, weight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.9},
            zorder=6,
        )
        if pd.notna(event["pico_vento_data"]):
            wind_week = (event["pico_vento_data"] - event["pico"]).days / 7.0
            axis.axhline(wind_week, color="#15803d", lw=1.2, ls=":")
            axis.annotate(
                f"pico de vento (τx {event['tau_x_extremo_pa']:+.3f} Pa)",
                xy=(123, wind_week), xytext=(2, 3), textcoords="offset points",
                fontsize=6.6, color="#14532d", weight="bold",
            )
        axis.set_title(
            f"{event['event_id']} | {event['classe']} | pico ONI {event['oni_pico_c']:+.1f} °C",
            fontsize=8.5,
        )
        axis.set_ylabel("semanas rel. ao pico", fontsize=7.5)
        axis.tick_params(labelsize=7)
    for axis in axes_flat[n_panels:]:
        axis.axis("off")
    thermal_extreme = "aquecimento" if signal_label == "El Niño" else "resfriamento"
    fig.suptitle(
        f"{signal_label}: Hovmöller SSTA equatorial (2S–2N) por evento — "
        f"estrela = epicentro do {thermal_extreme}; linha verde = pico da anomalia de vento zonal",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.93, hspace=0.28, wspace=0.14)
    if mesh is not None:
        fig.colorbar(
            mesh, ax=axes_flat[:n_panels].tolist(), shrink=0.8,
            label="SSTA semanal 2S–2N (°C)",
        )
    return fig


def _kelvin_sla_figure(
    ssh_by_lon: pd.DataFrame,
    pulses: pd.DataFrame,
    *,
    signal_label: str,
):
    """Hovmöller de SLA nas janelas com cobertura, com setas de propagação Kelvin."""
    import matplotlib.dates as mdates
    import matplotlib.patheffects as pe

    valid = pulses.loc[pulses["status"].eq("ok")].copy()
    if valid.empty:
        return _message_figure(
            f"{signal_label}: propagação equatorial (SLA)",
            "Nenhuma janela de evento com cobertura de SSH para o diagnóstico de Kelvin.",
        )
    for column in ("janela_inicio", "janela_fim", "pulso_data", "chegada_estimada_leste"):
        valid[column] = pd.to_datetime(valid[column], errors="coerce")
    events = list(valid.groupby("event_id", sort=False))
    fig, axes = plt.subplots(len(events), 1, figsize=(12.5, 3.6 * len(events)))
    axes_flat = np.atleast_1d(axes).ravel()
    lon = ssh_by_lon.columns.to_numpy(dtype=float)
    mesh = None
    for axis, (event_id, group) in zip(axes_flat, events):
        segment = ssh_by_lon.loc[
            group["janela_inicio"].iloc[0]: group["janela_fim"].iloc[0]
        ].dropna(how="all")
        if segment.empty:
            axis.text(0.5, 0.5, "sem dados SSH", transform=axis.transAxes, ha="center")
            continue
        sla = segment - segment.mean(axis=0)
        mesh = axis.pcolormesh(
            lon, segment.index, sla.to_numpy(dtype=float),
            cmap="RdYlBu_r", vmin=-0.2, vmax=0.2, shading="auto", rasterized=True,
        )
        _add_nino34_lon_band(axis)
        _format_lon_axis(axis)
        axis.set_title(f"{event_id} | {group['classe'].iloc[0]}", fontsize=9)
        axis.tick_params(labelsize=7)
        axis.yaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axis.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for _, pulse in group.iterrows():
            if pd.isna(pulse["pulso_data"]) or pd.isna(pulse["chegada_estimada_leste"]):
                continue
            axis.annotate(
                "",
                xy=(255, mdates.date2num(pulse["chegada_estimada_leste"])),
                xytext=(163, mdates.date2num(pulse["pulso_data"])),
                arrowprops={
                    "arrowstyle": "-|>,head_width=0.5,head_length=1.0",
                    "lw": 2.6, "color": "#111827", "shrinkA": 0, "shrinkB": 0,
                    "path_effects": [pe.withStroke(linewidth=4.5, foreground="white")],
                },
            )
    fig.suptitle(
        f"{signal_label}: SLA equatorial — setas = propagação tipo Kelvin "
        "(oeste → leste, ~2,4 m/s a partir dos pulsos 150E–180)",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.90, hspace=0.35)
    if mesh is not None:
        fig.colorbar(
            mesh, ax=axes_flat[: len(events)].tolist(), shrink=0.8,
            label="SLA local da janela (m)",
        )
    return fig


def _class_composite_figure(table: pd.DataFrame, *, signal_label: str, sign: float):
    styles = {
        "fraco": ("#f9a825", 1.4),
        "moderado": ("#e65100", 1.7),
        "forte": ("#b91c1c", 2.0),
        "muito_forte": ("#111827", 2.4),
        "todas_classes_analisadas": ("#6b7280", 1.2),
    }
    fig, axis = plt.subplots(figsize=(11, 5.6))
    for class_name, group in table.groupby("classe", sort=False):
        color, width = styles.get(str(class_name), ("#2563eb", 1.4))
        group = group.sort_values("semana_rel_pico")
        n_events = int(group["n_eventos"].max())
        axis.plot(
            group["semana_rel_pico"], group["ssta_media_c"],
            color=color, lw=width,
            ls="--" if class_name == "todas_classes_analisadas" else "-",
            label=f"{class_name} (n={n_events})",
        )
    axis.axvline(0, color="k", lw=0.8, ls="--")
    axis.axhline(sign * 0.5, color="#b91c1c", lw=0.8, ls=":", label=f"limiar {sign * 0.5:+.1f} °C")
    axis.axhline(sign * 1.0, color="#7f1d1d", lw=0.8, ls=":", label=f"corte de elegibilidade {sign * 1.0:+.1f} °C")
    axis.axhline(0, color="#6b7280", lw=0.6)
    axis.set_xlabel("semanas relativas ao pico (0 = pico ONI do evento)")
    axis.set_ylabel("SSTA Niño 3.4 (°C)")
    axis.set_title(f"{signal_label}: composto de SSTA por classe de intensidade")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _influence_figure(table: pd.DataFrame, *, signal_label: str):
    descriptive = table.loc[table["metrica"].eq("peso_descritivo_do_estado_pct")]
    discriminant = table.loc[table["metrica"].eq("peso_discriminante_entre_fases_pct")]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    axes_flat = axes.ravel()
    for axis, phase in zip(axes_flat[:4], PHASE_ORDER):
        subset = descriptive.loc[descriptive["fase"].eq(phase)].nlargest(10, "valor_pct")
        subset = subset.iloc[::-1]
        colors = [FAMILY_COLORS.get(str(f), "#6b7280") for f in subset["familia"]]
        axis.barh(subset["variavel"].astype(str), subset["valor_pct"], color=colors)
        axis.set_title(f"{phase} — peso descritivo do estado (%)", fontsize=9)
        axis.tick_params(labelsize=7.5)
        axis.grid(axis="x", alpha=0.25)
    subset = discriminant.nlargest(10, "valor_pct").iloc[::-1]
    colors = [FAMILY_COLORS.get(str(f), "#6b7280") for f in subset["familia"]]
    bars = axes_flat[4].barh(subset["variavel"].astype(str), subset["valor_pct"], color=colors)
    for bar, confirmed in zip(bars, subset["confirmado_fdr"]):
        if confirmed is not True and str(confirmed).lower() != "true":
            bar.set_alpha(0.35)
    axes_flat[4].set_title("peso discriminante entre fases (%) — Kendall W", fontsize=9)
    axes_flat[4].tick_params(labelsize=7.5)
    axes_flat[4].grid(axis="x", alpha=0.25)
    legend_axis = axes_flat[5]
    legend_axis.axis("off")
    families_present = [
        family
        for family in FAMILY_COLORS
        if family in set(table["familia"].dropna().astype(str))
    ]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[family])
        for family in families_present
    ]
    legend_axis.legend(
        handles, families_present, loc="upper left", fontsize=8.5,
        title="família física", title_fontsize=9,
    )
    legend_axis.text(
        0.02, 0.42,
        "Percentual = participação relativa da variável física; alvo SSTA excluído.\n"
        "Painéis por fase: |média z entre eventos| normalizada na fase.\n"
        "Painel Kendall W: quanto a variável distingue as quatro fases\n"
        "no mesmo evento; barras translúcidas não passaram no BH-FDR.",
        transform=legend_axis.transAxes, fontsize=8.5, va="top", linespacing=1.5,
    )
    fig.suptitle(
        f"{signal_label}: participação descritiva das variáveis físicas (alvo excluído)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _event_lifecycle_figure(table: pd.DataFrame, *, signal_label: str):
    required = {"event_id", "fase", "inicio", "fim"}
    if table.empty or not required.issubset(table.columns):
        return _message_figure(
            f"{signal_label}: ciclos de vida delimitados",
            "Tabela de ciclo de vida incompleta.",
        )
    data = table.copy()
    data["inicio"] = pd.to_datetime(data["inicio"], errors="coerce")
    data["fim"] = pd.to_datetime(data["fim"], errors="coerce")
    data = data.dropna(subset=["inicio", "fim"])
    events = list(data.sort_values("inicio")["event_id"].drop_duplicates())
    phase_colors = {
        "genese": "#60a5fa",
        "crescimento": "#f59e0b",
        "pico": "#b91c1c",
        "decaimento": "#7c3aed",
    }
    fig, axis = plt.subplots(figsize=(14, max(5.5, 0.58 * len(events) + 2)))
    y_by_event = {event_id: position for position, event_id in enumerate(events)}
    for _, row in data.iterrows():
        start = mdates.date2num(row["inicio"])
        width = max(1.0, mdates.date2num(row["fim"] + pd.Timedelta(days=1)) - start)
        axis.barh(
            y_by_event[row["event_id"]],
            width,
            left=start,
            height=0.72,
            color=phase_colors.get(str(row["fase"]), "#6b7280"),
            edgecolor="white",
            linewidth=0.5,
        )
    axis.set_yticks(range(len(events)), events)
    axis.invert_yaxis()
    axis.xaxis_date()
    axis.xaxis.set_major_locator(mdates.YearLocator(base=5))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.set_xlabel("tempo civil (a faixa de gênese antecede o onset ONI)")
    axis.set_title(f"{signal_label}: ciclos de vida dos eventos elegíveis no escopo")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=phase_colors[phase]) for phase in PHASE_ORDER
    ]
    axis.legend(handles, PHASE_ORDER, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def _phase_boundary_sensitivity_figure(table: pd.DataFrame, *, signal_label: str):
    required = {
        "fase",
        "duracao_semanas",
        "janela_genese_semanas",
        "fracao_faixa_pico",
        "configuracao_canonica",
    }
    if table.empty or not required.issubset(table.columns):
        return _message_figure(
            f"{signal_label}: sensibilidade das fronteiras de fase",
            "Tabela de sensibilidade incompleta.",
        )
    data = table.copy()
    data["duracao_semanas"] = pd.to_numeric(data["duracao_semanas"], errors="coerce")
    data["fracao_faixa_pico"] = pd.to_numeric(data["fracao_faixa_pico"], errors="coerce")
    data["janela_genese_semanas"] = pd.to_numeric(data["janela_genese_semanas"], errors="coerce")
    summary = data.groupby(
        ["fase", "janela_genese_semanas", "fracao_faixa_pico"], as_index=False
    ).agg(
        duracao_mediana_semanas=("duracao_semanas", "median"),
        duracao_q25_semanas=("duracao_semanas", lambda values: values.quantile(0.25)),
        duracao_q75_semanas=("duracao_semanas", lambda values: values.quantile(0.75)),
        n_eventos=("event_id", "nunique"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5), sharex=True)
    for axis, phase in zip(axes.ravel(), PHASE_ORDER):
        phase_data = summary.loc[summary["fase"].eq(phase)]
        for genesis_weeks, group in phase_data.groupby("janela_genese_semanas"):
            group = group.sort_values("fracao_faixa_pico")
            axis.plot(
                100 * group["fracao_faixa_pico"],
                group["duracao_mediana_semanas"],
                marker="o",
                label=f"gênese {int(genesis_weeks)} sem",
            )
        canonical = phase_data.loc[
            phase_data["janela_genese_semanas"].eq(26)
            & np.isclose(phase_data["fracao_faixa_pico"], 0.90)
        ]
        if not canonical.empty:
            axis.scatter(
                [90],
                canonical["duracao_mediana_semanas"],
                marker="*",
                s=180,
                c="#111827",
                zorder=5,
                label="canônica 26 sem / 90%",
            )
        axis.set_title(phase)
        axis.set_ylabel("duração mediana (semanas)")
        axis.grid(alpha=0.25)
    for axis in axes[-1, :]:
        axis.set_xlabel("limiar relativo da faixa de pico (% do extremo do evento)")
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle(
        f"{signal_label}: sensibilidade das quatro fases — gênese 13/26/39 semanas × pico 80/90/95%",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _transition_guides_figure(table: pd.DataFrame, *, signal_label: str):
    if table.empty:
        return _message_figure(
            f"{signal_label}: variáveis-guia de transição",
            "Nenhum marcador de transição pôde ser calculado.",
        )
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5))
    axes_flat = axes.ravel()
    transitions = list(table["transicao_monitorada"].drop_duplicates())
    for axis, transition in zip(axes_flat, transitions):
        subset = table.loc[
            table["transicao_monitorada"].eq(transition)
            & table["guia_principal"].astype(str).str.lower().isin({"true", "1"})
        ].sort_values("score_guia").copy()
        if subset.empty:
            axis.text(0.5, 0.5, "sem marcador selecionado", transform=axis.transAxes, ha="center", fontsize=9)
            axis.set_title(str(transition).replace("_", " "), fontsize=9.5)
            axis.axis("off")
            continue
        colors = [FAMILY_COLORS.get(str(f), "#6b7280") for f in subset["familia"]]
        bars = axis.barh(
            subset["variavel"].astype(str),
            subset["efeito_pareado_dz"],
            color=colors,
        )
        for bar, (_, row) in zip(bars, subset.iterrows()):
            confirmed = bool(row.get("significativo_transicao_fdr", False))
            if not confirmed:
                bar.set_alpha(0.35)
            axis.annotate(
                f"{row['consistencia_direcao_pct']:.0f}% dos eventos | q={row['q_transicao_bh']:.3f}",
                xy=(max(bar.get_width(), 0), bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0),
                textcoords="offset points", fontsize=7,
                ha="left", va="center",
            )
        axis.axvline(0, color="#111827", lw=0.8)
        axis.set_title(str(transition).replace("_", " "), fontsize=9.5)
        axis.set_xlabel("efeito pareado da mudança (dz)", fontsize=8)
        axis.tick_params(labelsize=7.5)
        axis.grid(axis="x", alpha=0.25)
    for axis in axes_flat[len(transitions):]:
        axis.axis("off")
    fig.suptitle(
        f"{signal_label}: marcadores empíricos das transições adjacentes (evento pareado)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _variable_sets_figure(table: pd.DataFrame, *, signal_label: str):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5))
    axes_flat = axes.ravel()
    for axis, phase in zip(axes_flat, PHASE_ORDER):
        subset = table.loc[
            table["fase"].eq(phase) & table["integra_conjunto_descritor"]
        ].sort_values("rank_na_fase").iloc[::-1]
        if subset.empty:
            axis.text(0.5, 0.5, "sem conjunto para esta fase", transform=axis.transAxes, ha="center")
            axis.axis("off")
            continue
        colors = [FAMILY_COLORS.get(str(f), "#6b7280") for f in subset["familia"]]
        value = "score_descritor" if "score_descritor" in subset else "sensibilidade_fase"
        bars = axis.barh(subset["variavel"].astype(str), subset[value], color=colors)
        for bar, (_, row) in zip(bars, subset.iterrows()):
            axis.annotate(
                f"sens={row['sensibilidade_fase']:.2f} | W={row['kendall_w_entre_fases']:.2f}",
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0), textcoords="offset points", fontsize=7, va="center",
            )
        axis.set_title(f"{phase} — conjunto físico diversificado", fontsize=9.5)
        axis.set_xlabel("score = sensibilidade × √Kendall W", fontsize=8)
        axis.tick_params(labelsize=7.5)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle(
        f"{signal_label}: conjuntos físicos por fase (alvo excluído; redundância limitada)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


class NotebookWorkflow:
    def __init__(self, root: Path, notebook_code: str, *, mode: str = "official") -> None:
        self.root = Path(root).resolve()
        self.code = parse_notebook_code(notebook_code).value
        self.parsed = parse_notebook_code(self.code)
        self.mode = str(mode)
        self.spec = NOTEBOOK_BY_CODE[self.code]

    def describe(self) -> dict[str, object]:
        description = {
            "notebook_code": self.code,
            "phase": self.parsed.phase,
            "enso_type": self.parsed.enso_type,
            "mode": self.mode,
            "execution_policy": "numeric-core-first-viewer-publisher",
        }
        print(json.dumps(description, indent=2, ensure_ascii=False))
        return description

    @property
    def phase3_stats(self) -> Path:
        return (
            self.root
            / "data/processed/parquet/statistics"
            / self.parsed.namespace
        )

    def _f3_table(self, block: str, ordinal: int, slug: str) -> Path:
        notebook = notebook_code_for(3, block, self.parsed.enso_type)
        return self.phase3_stats / f"{table_code(notebook, ordinal, slug=slug)}.csv"

    def _f4_table(self, block: str, ordinal: int, slug: str, suffix: str) -> Path:
        notebook = notebook_code_for(4, block, self.parsed.enso_type)
        return (
            self.root
            / "data/processed/parquet/statistics"
            / f"{table_code(notebook, ordinal, slug=slug)}{suffix}"
        )

    def input_paths(self) -> list[tuple[Path, bool, str]]:
        phase = self.parsed.phase
        block = self.parsed.block
        if phase == 2:
            return [
                (self.root / "data/processed/parquet/features/nino34_master_weekly.csv", True, "master F2"),
                (self.root / "data/processed/parquet/statistics/phase2_variable_contract.csv", True, "contrato das 31 variáveis"),
                (self.root / "data/processed/parquet/statistics/phase2_master_source_adjusted_v1_audit.csv", True, "auditoria de cobertura F2"),
                (self.root / "data/processed/parquet/statistics/phase2_master_validation.csv", True, "validação estrutural F2"),
                (self.root / "data/processed/parquet/statistics/phase2_master_audit.csv", True, "frescor e cobertura por variável"),
                (self.root / "data/processed/parquet/statistics/phase2_ctd_validation.csv", True, "validação CTD/WOD"),
                (self.root / "data/processed/parquet/statistics/phase1_ibge_boundaries_audit.csv", False, "auditoria IBGE da F1"),
            ]
        if phase == 3:
            mapping: dict[str, list[tuple[Path, bool, str]]] = {
                "A": [
                    (self._f3_table("A", 2, "estatisticas_por_fase"), True, "estatísticas evento-fase"),
                    (self.root / "data/processed/parquet/features/nino34_master_weekly.csv", True, "índices semanais F2"),
                    (self.root / "data/processed/parquet/statistics/phase2_variable_contract.csv", True, "contrato das 31 variáveis"),
                ],
                "B": [
                    (self._f3_table("B", 1, "eventos"), True, "catálogo completo de eventos"),
                    (self._f3_table("B", 3, "ciclo_eventos"), True, "ciclo de vida canônico"),
                    (self._f3_table("B", 4, "sensibilidade_faixa_pico"), True, "sensibilidade da faixa de pico"),
                    (self._f3_table("B", 5, "sensibilidade_fronteiras_fases"), True, "sensibilidade das quatro fases"),
                ],
                "C": [(self._f3_table("C", 2, "melhores_lags_fdr"), True, "melhores lags FDR")],
                "D": [(self._f3_table("D", 1, "friedman_fdr"), True, "Friedman/FDR")],
                "E": [(self._f3_table("E", 1, "bootstrap_resumo"), True, "bootstrap de eventos")],
                "F": [
                    (self._f3_table("F", 1, "hovmoller_picos"), True, "picos de aquecimento e vento por evento"),
                    (self._f3_table("F", 2, "kelvin_pulsos_sla"), True, "pulsos de SLA tipo Kelvin"),
                    (self._f3_table("A", 2, "estatisticas_por_fase"), True, "estatísticas físicas"),
                    (self.root / "data/processed/parquet/features/equatorial_pacific_ssta_weekly_by_lon.parquet", True, "SSTA equatorial 2S–2N por longitude"),
                    (self.root / "data/processed/parquet/features/ssh_equatorial_daily_by_lon_events.parquet", True, "SSH equatorial diária por longitude"),
                ],
                "G": [(self._f3_table("G", 1, "composto_ssta_classe"), True, "composto SSTA por classe")],
                "H": [(self._f3_table("A", 2, "estatisticas_por_fase"), True, "gênese por variável")],
                "I": [
                    (self._f3_table("C", 2, "melhores_lags_fdr"), True, "lags FDR"),
                    (self._f3_table("D", 1, "friedman_fdr"), True, "efeitos entre fases"),
                    (self._f3_table("E", 1, "bootstrap_resumo"), True, "estabilidade por evento"),
                    (self._f3_table("I", 3, "influencia_percentual"), True, "influência percentual por variável"),
                    (self._f3_table("I", 4, "guias_transicao_fase"), True, "variáveis-guia de transição"),
                ],
                "K": [(self._f3_table("K", 1, "pca_variancia"), True, "variância PCA")],
                "L": [
                    (self._f3_table("L", 1, "duracao_fases"), True, "duração das fases"),
                    (self._f3_table("L", 2, "conjuntos_variaveis_fase"), True, "conjuntos de variáveis por fase"),
                ],
            }
            return mapping[block]
        if phase == 4 and block == "C":
            return [
                (self._f4_table("C", 5, "chave_pixel", ".csv"), True, "lags por pixel CHIRPS original"),
                (self._f4_table("C", 8, "lag_pico_regioes", ".csv"), True, "lag de pico resumido após pixels"),
                (self._f4_table("C", 6, "significancia_campo", ".csv"), True, "significância de campo"),
                (self.root / "data/processed/zarr/features/chirps_native_weekly_targets.zarr", True, "CHIRPS nativo"),
            ]
        if phase == 4 and block == "D":
            return [
                (self._f4_table("D", 3, "ranking_clusters", ".csv"), True, "ranking de clusters"),
                (self._f4_table("D", 5, "resumo_hipoteses", ".csv"), True, "síntese de hipóteses"),
                (self.root / "data/processed/zarr/features/chirps_native_weekly_targets.zarr", True, "CHIRPS nativo"),
            ]
        if phase in {5, 6, 7, 8}:
            paths = [
                (
                    self.root / f"data/processed/runs/{self.mode}/fase{phase}",
                    True,
                    f"ArtifactRuns F{phase}",
                )
            ]
            if phase in {5, 7}:
                paths.append(
                    (
                        self.root / "data/audit/augmentation_ablation",
                        True,
                        "ablação augmentation ON/OFF",
                    )
                )
            return paths
        raise KeyError(self.code)

    def input_inventory(self) -> pd.DataFrame:
        rows = []
        for path, required, role in self.input_paths():
            rows.append(
                {
                    "role": role,
                    "path": str(path.relative_to(self.root)),
                    "required": required,
                    "exists": path.exists(),
                    "bytes": path.stat().st_size if path.is_file() else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def require_inputs(self) -> None:
        missing = [
            path for path, required, _role in self.input_paths() if required and not path.exists()
        ]
        if missing:
            listed = "\n".join(f"- {path.relative_to(self.root)}" for path in missing)
            raise FileNotFoundError(
                f"Entradas ausentes para {self.code}. Execute primeiro o runner da fase:\n{listed}"
            )

    def _phase2_contract(self, master: pd.DataFrame, contract: pd.DataFrame) -> pd.DataFrame:
        """Cria uma linha auditável por variável física do master semanal."""
        dates = pd.to_datetime(master["week_ending_sunday"], errors="coerce")
        variables = [column for column in master.columns if column not in {"week_ending_sunday", "ocean_source_code"}]
        specs = contract.loc[contract["name"].isin(variables)].copy().set_index("name")
        rows: list[dict[str, object]] = []
        for variable in variables:
            values = pd.to_numeric(master[variable], errors="coerce")
            valid = values.notna()
            valid_dates = dates.loc[valid]
            gaps = valid_dates.sort_values().diff().dt.days.div(7).sub(1)
            largest_gap = int(gaps.clip(lower=0).max()) if gaps.notna().any() else 0
            source = str(specs.at[variable, "source"])
            representation = str(specs.at[variable, "representation_source_adjusted"])
            coverage = float(valid.mean())
            ready = bool(valid.any() and coverage >= 0.95 and str(specs.at[variable, "units"]).strip())
            if variable == "nino34_ssta":
                family = "anomalia oceânica (Niño-3.4)"
            elif source == "ERA5":
                family = "atmosféricas (anomalias ERA5)"
            else:
                family = "oceanográficas"
            rows.append({
                "variavel": variable,
                "familia": family,
                "fonte": source,
                "unidade": specs.at[variable, "units"],
                "intervalo_entrada": "diário",
                "intervalo_analise": "semanal (W-SUN)",
                "tratamento_ate_master": (
                    "ERA5: registros horários agregados a médias diárias; depois média semanal W-SUN."
                    if source == "ERA5" else "Séries físicas diárias; depois média semanal W-SUN."
                ),
                "representacao_no_master": representation,
                "inicio_disponivel": valid_dates.min().date().isoformat() if not valid_dates.empty else "",
                "fim_disponivel": valid_dates.max().date().isoformat() if not valid_dates.empty else "",
                "semanas_validas": int(valid.sum()),
                "semanas_grade": int(len(master)),
                "cobertura_percentual": round(coverage * 100, 3),
                "maior_lacuna_semanas": largest_gap,
                "apta_sem_tratamento_extra": ready,
                "escopo_da_aprovacao": "estatística descritiva/associativa semanal; usar apenas semanas observadas" if ready else "requer inspeção adicional antes de análise",
                "alerta_modelagem": "em CV preditiva, refazer climatologia/ajuste no fold de treino" if "source_seasonal_anomaly_detrended" in representation else "em CV preditiva, preservar a separação temporal treino/teste",
                "mensal_disponivel_na_f2": False,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _phase2_availability_figure(contract: pd.DataFrame):
        data = contract.sort_values(["familia", "variavel"]).copy()
        fig, axis = plt.subplots(figsize=(11, max(7, 0.34 * len(data) + 2)))
        colors = np.where(data["apta_sem_tratamento_extra"], "#15803d", "#b91c1c")
        axis.barh(data["variavel"], data["cobertura_percentual"], color=colors)
        axis.axvline(95, color="#92400e", lw=1, ls="--", label="limiar operacional: 95%")
        axis.set_xlim(0, 101)
        axis.set_xlabel("cobertura na grade semanal (%)")
        starts = pd.to_datetime(data["inicio_disponivel"], errors="coerce")
        ends = pd.to_datetime(data["fim_disponivel"], errors="coerce")
        period = f"{starts.min():%d/%m/%Y}–{ends.max():%d/%m/%Y}"
        axis.set_title(f"F2 — disponibilidade semanal de {len(data)} variáveis ({period})")
        axis.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        return fig

    @staticmethod
    def _phase2_series_figure(long: pd.DataFrame, *, title: str, max_columns: int = 3):
        variables = list(long["variavel"].drop_duplicates())
        rows = int(np.ceil(len(variables) / max_columns))
        fig, axes = plt.subplots(rows, max_columns, figsize=(17, max(3.2 * rows, 5)), sharex=True)
        axes_flat = np.atleast_1d(axes).ravel()
        for axis, variable in zip(axes_flat, variables):
            data = long.loc[long["variavel"].eq(variable)]
            axis.plot(data["semana"], data["valor"], color="#1d4ed8", lw=0.55)
            # Zero é referência física para as anomalias; para os campos
            # absolutos oceânicos ele apenas comprimiria a variabilidade.
            if variable.endswith("_anom") or variable == "nino34_ssta":
                axis.axhline(0, color="#6b7280", lw=0.5, zorder=0)
            axis.set_title(f"{variable} [{data['unidade'].iloc[0]}]", fontsize=8)
            axis.grid(axis="y", alpha=0.2, lw=0.4)
        for axis in axes_flat[len(variables):]:
            axis.set_visible(False)
        fig.suptitle(title, y=0.995, fontsize=13, weight="bold")
        fig.supxlabel("semana de término (W-SUN)")
        fig.supylabel("valor na unidade original do master")
        fig.tight_layout(rect=(0.02, 0.02, 1, 0.98))
        return fig

    @staticmethod
    def _phase2_freshness_figure(audit: pd.DataFrame):
        table = audit.copy()
        table["fim"] = pd.to_datetime(table["fim"], errors="coerce")
        latest = table.groupby("variavel", as_index=False)["fim"].max().sort_values("fim")
        reference = pd.Timestamp.now().normalize()
        latest["defasagem_dias"] = (reference - latest["fim"]).dt.days
        fig, axis = plt.subplots(figsize=(12, max(5, 0.28 * len(latest))))
        axis.barh(latest["variavel"], latest["defasagem_dias"], color="#2563eb")
        axis.set_xlabel("dias desde o último valor válido até a data da execução")
        axis.set_title("F2 — frescor real por variável")
        axis.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        return latest, fig

    @staticmethod
    def _phase2_ctd_figure(ctd: pd.DataFrame):
        table = ctd.copy()
        fig, axis = plt.subplots(figsize=(12, 5))
        axis.plot(table["ano"], table["termoclina_ctd_media_m"], label="CTD/WOD", lw=1.2)
        axis.plot(table["ano"], table["d20_reanalise_media_m"], label="UFS+GLORYS", lw=1.2)
        axis.set_title("F2 — validação independente CTD/WOD × UFS+GLORYS")
        axis.set_xlabel("ano")
        axis.set_ylabel("profundidade média (m)")
        axis.legend()
        axis.grid(alpha=0.2)
        fig.tight_layout()
        return fig

    def _phase2_diagnostics(self) -> tuple[list[tuple[int, str, pd.DataFrame, object, str]], list[str]]:
        master = pd.read_csv(self.input_paths()[0][0], parse_dates=["week_ending_sunday"])
        contract = self._phase2_contract(master, pd.read_csv(self.input_paths()[1][0]))
        variables = [column for column in master.columns if column not in {"week_ending_sunday", "ocean_source_code"}]
        by_variable = contract.set_index("variavel")

        def long_table(columns: list[str]) -> pd.DataFrame:
            table = master[["week_ending_sunday", *columns]].melt(id_vars="week_ending_sunday", var_name="variavel", value_name="valor").rename(columns={"week_ending_sunday": "semana"})
            return table.merge(contract[["variavel", "familia", "fonte", "unidade", "intervalo_analise"]], on="variavel", how="left", validate="many_to_one")

        ocean = [v for v in variables if by_variable.at[v, "familia"] == "oceanográficas"]
        atmosphere = [v for v in variables if by_variable.at[v, "familia"] == "atmosféricas (anomalias ERA5)"]
        nino = [v for v in variables if by_variable.at[v, "familia"] == "anomalia oceânica (Niño-3.4)"]
        ocean_table, atmosphere_table, anomaly_table = long_table(ocean), long_table(atmosphere), long_table(nino)

        def last_complete_week(table: pd.DataFrame) -> pd.Timestamp:
            wide = table.pivot(index="semana", columns="variavel", values="valor")
            complete = wide.apply(pd.to_numeric, errors="coerce").dropna(how="any")
            return pd.to_datetime(complete.index, errors="coerce").max()

        def week_label(value: pd.Timestamp) -> str:
            return value.strftime("%d/%m/%Y") if pd.notna(value) else "não disponível"

        group_weeks = {
            "oceanográfico": last_complete_week(ocean_table),
            "atmosférico ERA5": last_complete_week(atmosphere_table),
            "SST/SSTA Niño 3.4": last_complete_week(anomaly_table),
        }
        group_status = pd.DataFrame(
            {
                "grupo": list(group_weeks),
                "ultima_semana_valida_w_sun": list(group_weeks.values()),
                "frequencia": "semanal W-SUN",
                "criterio": "semana com valor válido em todas as variáveis do grupo",
            }
        )
        fig_status, axis_status = plt.subplots(figsize=(10, 3.8))
        status_plot = group_status.dropna(subset=["ultima_semana_valida_w_sun"])
        axis_status.scatter(status_plot["ultima_semana_valida_w_sun"], status_plot["grupo"], s=70, color="#15803d")
        axis_status.set_title("F2 — última semana válida por grupo (fechamento W-SUN)")
        axis_status.set_xlabel("semana de término")
        axis_status.grid(axis="x", alpha=0.2)
        fig_status.tight_layout()
        freshness_source = pd.read_csv(self.input_paths()[4][0])
        freshness_table, freshness_figure = self._phase2_freshness_figure(freshness_source)
        ctd_table = pd.read_csv(self.input_paths()[5][0])
        diagnostics = [
            (1, "contrato_disponibilidade", contract, self._phase2_availability_figure(contract), "Painel de disponibilidade e contrato F2"),
            (2, "ultima_semana_valida_grupos", group_status, fig_status, "Última semana válida por grupo — grade semanal W-SUN"),
            (3, "series_oceanograficas", ocean_table, self._phase2_series_figure(ocean_table, title=f"F2 — grupo oceanográfico · última semana completa {week_label(group_weeks['oceanográfico'])}"), "Sanidade temporal — grupo oceanográfico"),
            (4, "series_atmosfericas", atmosphere_table, self._phase2_series_figure(atmosphere_table, title=f"F2 — grupo atmosférico ERA5 · última semana completa {week_label(group_weeks['atmosférico ERA5'])}"), "Sanidade temporal — grupo atmosférico"),
            (5, "serie_anomalia_nino34", anomaly_table, self._phase2_series_figure(anomaly_table, title=f"F2 — grupo SST/SSTA Niño 3.4 · última semana completa {week_label(group_weeks['SST/SSTA Niño 3.4'])}", max_columns=1), "Sanidade temporal — grupo SST/SSTA Niño 3.4"),
            (6, "frescor_fontes", freshness_table, freshness_figure, "Frescor real das fontes de entrada"),
            (7, "validacao_ctd_wod", ctd_table, self._phase2_ctd_figure(ctd_table), "Validação CTD/WOD × UFS+GLORYS"),
        ]
        ibge_path = self.input_paths()[6][0]
        if ibge_path.is_file():
            ibge = pd.read_csv(ibge_path)
            fig, axis = plt.subplots(figsize=(9, 3.5))
            axis.bar(ibge["produto"], ibge["geometrias"], color="#2563eb")
            axis.set_title("F1 — malhas IBGE disponibilizadas para as fases espaciais")
            axis.set_ylabel("geometrias")
            axis.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            diagnostics.append((8, "auditoria_ibge", ibge, fig, "Auditoria das malhas IBGE"))
        takeaways = [
            f"O master semanal cobre {master['week_ending_sunday'].min():%d/%m/%Y} a {master['week_ending_sunday'].max():%d/%m/%Y}.",
            f"{int(contract['apta_sem_tratamento_extra'].sum())}/{len(contract)} variáveis atendem ao contrato para estatística semanal sem limpeza adicional.",
            "As entradas nativas são diárias (ERA5 parte de registros horários) e o produto da F2 é semanal; não há variável mensal independente nesta fase.",
            "A reconstituição da F2 usa exclusivamente períodos semanais com fechamento no domingo (W-SUN); datas diárias são apenas insumos da agregação.",
            "Últimas semanas completas: " + "; ".join(f"{group}: {week_label(value)}" for group, value in group_weeks.items()) + ".",
            f"A maior data válida observada no audit é {pd.to_datetime(freshness_source['fim']).max():%d/%m/%Y}; o eixo semanal sozinho não comprova atualização das fontes.",
            "O notebook publica a validação CTD/WOD e, quando disponível, a auditoria IBGE produzida pela F1.",
        ]
        return diagnostics, takeaways

    @staticmethod
    def _f3_weekly_trend_figure(long: pd.DataFrame, *, title: str, columns: int = 3):
        """Séries no valor original e tendência OLS apenas descritiva por índice."""
        variables = list(long["variavel"].drop_duplicates())
        rows = int(np.ceil(len(variables) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(17, max(3.15 * rows, 5)), sharex=True)
        axes_flat = np.atleast_1d(axes).ravel()
        for axis, variable in zip(axes_flat, variables):
            data = long.loc[long["variavel"].eq(variable)].sort_values("semana")
            axis.plot(data["semana"], data["valor"], color="#2563eb", lw=0.5, label="semanal")
            axis.plot(data["semana"], data["tendencia_ols"], color="#d97706", lw=1.1, label="tendência OLS")
            if variable.endswith("_anom") or variable == "nino34_ssta":
                axis.axhline(0, color="#6b7280", lw=0.45, zorder=0)
            slope = pd.to_numeric(data["tendencia_unidade_por_ano"], errors="coerce").dropna()
            slope_label = f"{slope.iloc[0]:+.3g}/ano" if len(slope) else "indisponível"
            axis.set_title(f"{variable} [{data['unidade'].iloc[0]}] | {slope_label}", fontsize=7.8)
            axis.grid(axis="y", alpha=0.2, lw=0.4)
        for axis in axes_flat[len(variables):]:
            axis.set_visible(False)
        axes_flat[0].legend(loc="best", fontsize=6.8)
        fig.suptitle(title, y=0.995, fontsize=13, weight="bold")
        fig.supxlabel("semana de término (W-SUN)")
        fig.supylabel("valor na unidade original do master")
        fig.tight_layout(rect=(0.02, 0.02, 1, 0.98))
        return fig

    def _f3nino_a_weekly_indices(self) -> tuple[list[tuple[int, str, pd.DataFrame, object, str]], pd.DataFrame, list[str]]:
        """Publica todas as séries semanais sem confundir tendência com inferência."""
        paths = self.input_paths()
        master = pd.read_csv(paths[1][0], parse_dates=["week_ending_sunday"])
        contract = pd.read_csv(paths[2][0])
        variables = [name for name in master.columns if name not in {"week_ending_sunday", "ocean_source_code"}]
        specs = contract.loc[contract["name"].isin(variables)].set_index("name")

        def make_long(columns: list[str], family: str) -> pd.DataFrame:
            long = master[["week_ending_sunday", *columns]].melt(
                id_vars="week_ending_sunday", var_name="variavel", value_name="valor"
            ).rename(columns={"week_ending_sunday": "semana"})
            long["valor"] = pd.to_numeric(long["valor"], errors="coerce")
            long["familia"] = family
            long["fonte"] = long["variavel"].map(specs["source"])
            long["unidade"] = long["variavel"].map(specs["units"])
            long["representacao_master"] = long["variavel"].map(specs["representation_raw"])
            long["metodo_tendencia"] = "OLS linear por série semanal completa; diagnóstico descritivo, sem teste de hipótese"
            long["tendencia_ols"] = np.nan
            long["tendencia_unidade_por_ano"] = np.nan
            for variable, index in long.groupby("variavel", sort=False).groups.items():
                subset = long.loc[index]
                valid = subset["valor"].notna() & subset["semana"].notna()
                if int(valid.sum()) < 2:
                    continue
                x = (subset.loc[valid, "semana"] - subset.loc[valid, "semana"].min()).dt.days.to_numpy() / 365.25
                slope, intercept = np.polyfit(x, subset.loc[valid, "valor"].to_numpy(), 1)
                x_all = (subset["semana"] - subset.loc[valid, "semana"].min()).dt.days.to_numpy() / 365.25
                long.loc[index, "tendencia_ols"] = intercept + slope * x_all
                long.loc[index, "tendencia_unidade_por_ano"] = slope
            return long

        ocean = [variable for variable in variables if str(specs.at[variable, "source"]) != "ERA5"]
        atmosphere = [variable for variable in variables if str(specs.at[variable, "source"]) == "ERA5"]
        ocean_table = make_long(ocean, "oceanográficas")
        atmosphere_table = make_long(atmosphere, "atmosféricas")
        trend_summary = pd.concat([ocean_table, atmosphere_table], ignore_index=True).groupby(
            ["variavel", "familia", "fonte", "unidade", "tendencia_unidade_por_ano"], as_index=False
        ).agg(inicio=("semana", "min"), fim=("semana", "max"), semanas_validas=("valor", "count"))
        diagnostics = [
            (1, "series_oceanicas_semanais", ocean_table, self._f3_weekly_trend_figure(ocean_table, title="F3NinoA — índices oceânicos semanais e tendência histórica"), "Índices oceânicos semanais e tendência histórica"),
            (2, "series_atmosfericas_semanais", atmosphere_table, self._f3_weekly_trend_figure(atmosphere_table, title="F3NinoA — índices atmosféricos semanais e tendência histórica"), "Índices atmosféricos semanais e tendência histórica"),
        ]
        takeaways = [
            f"Foram publicados {len(ocean)} índices oceânicos e {len(atmosphere)} índices atmosféricos, separados em painéis completos.",
            f"A série semanal cobre {master['week_ending_sunday'].min():%d/%m/%Y} a {master['week_ending_sunday'].max():%d/%m/%Y}.",
            "A linha laranja é tendência OLS descritiva da série inteira; não testa fases, causalidade ou previsão e não substitui F3NinoC–E.",
        ]
        return diagnostics, trend_summary, takeaways

    def _phase3_summary(self) -> tuple[pd.DataFrame, object, list[str]]:
        block = self.parsed.block
        sources = [_read_table(path) for path, _required, _role in self.input_paths()]
        source = sources[0]
        signal_label = "El Niño" if self.parsed.enso_type == "el_nino" else "La Niña"
        if block == "A":
            table = source.copy()
            value = "media_z_entre_eventos" if "media_z_entre_eventos" in table else "nivel_z_medio"
            variability = table.groupby("variavel")[value].agg(lambda values: values.max() - values.min())
            keep = variability.nlargest(min(12, len(variability))).index
            table = table[table["variavel"].isin(keep)].copy()
            fig = _phase_profile_figure(table, value, f"{signal_label}: assinatura das variáveis por fase")
        elif block == "B":
            table = source.groupby(["fase"], as_index=False).agg(
                duracao_media_semanas=("duracao_semanas", "mean"),
                n_eventos=("event_id", "nunique"),
            )
            fig = _bar_figure(table, label="fase", value="duracao_media_semanas", title=f"{signal_label}: duração das fases")
        elif block == "C":
            table = source.copy()
            value = "r_pearson" if "r_pearson" in table else next(
                (column for column in table if column.startswith("r_")), "lag_semanas"
            )
            if value in table:
                table[value] = pd.to_numeric(table[value], errors="coerce")
                table = table.sort_values(value, key=lambda values: values.abs()).tail(30)
            fig = _bar_figure(table, label="variavel", value=value, title=f"{signal_label}: precursores e lags FDR")
        elif block == "D":
            table = confirmed_friedman_discriminants(source)
            value = "kendall_w_entre_fases"
            table = table.sort_values(value, ascending=False).head(30) if value in table else table
            fig = _bar_figure(table, label="variavel", value=value, title=f"{signal_label}: efeito entre fases após FDR")
        elif block == "E":
            table = source.copy()
            numeric = [column for column in table.select_dtypes(include="number") if "lag" not in column.lower()]
            value = (
                "frequencia_selecao"
                if "frequencia_selecao" in table
                else (numeric[0] if numeric else "lag_semanas")
            )
            label = "variavel" if "variavel" in table else table.columns[0]
            fig = _bar_figure(table, label=label, value=value, title=f"{signal_label}: estabilidade entre eventos")
        elif block == "H":
            value = "media_z_entre_eventos" if "media_z_entre_eventos" in source else "nivel_z_medio"
            table = source[source["fase"].eq("genese")].copy()
            table[value] = pd.to_numeric(table[value], errors="coerce")
            table = table.sort_values(value, key=lambda values: values.abs()).tail(30)
            fig = _bar_figure(table, label="variavel", value=value, title=f"{signal_label}: precursores na gênese")
        elif block == "K":
            table = source.copy()
            value = "var_explicada"
            table["rotulo"] = table["fase"].astype(str) + " | " + table["componente"].astype(str)
            fig = _bar_figure(table, label="rotulo", value=value, title=f"{signal_label}: variância PCA por fase")
        else:
            raise KeyError(block)
        takeaways = [
            f"A tabela auditada contém {len(table)} linhas para {signal_label}.",
            "A leitura mantém eventos independentes separados e não soma semanas como novas réplicas.",
        ]
        return table, fig, takeaways

    def _load_lon_parquet(self, path: Path, *, lon_min: float | None = None, lon_max: float | None = None) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index)
        frame.columns = frame.columns.astype(float)
        if lon_min is not None:
            frame = frame.loc[:, frame.columns >= lon_min]
        if lon_max is not None:
            frame = frame.loc[:, frame.columns <= lon_max]
        return frame.sort_index()

    def _phase3_multi(self) -> tuple[list[tuple[int, str, pd.DataFrame, object, str]], pd.DataFrame, list[str]]:
        """Blocos F3 com múltiplos pares Fig/Tab (B, F, G, I, L)."""
        block = self.parsed.block
        signal_label = "El Niño" if self.parsed.enso_type == "el_nino" else "La Niña"
        sign = 1.0 if self.parsed.enso_type == "el_nino" else -1.0
        paths = [path for path, _required, _role in self.input_paths()]
        if block == "B":
            events = pd.read_csv(paths[0])
            lifecycle = pd.read_csv(paths[1])
            peak_sensitivity = pd.read_csv(paths[2])
            boundary_sensitivity = pd.read_csv(paths[3])
            eligible = events.loc[
                events["elegivel_analise"].astype(str).str.lower().isin({"true", "1"})
            ]
            diagnostics = [
                (
                    1,
                    "ciclos_vida_eventos",
                    lifecycle,
                    _event_lifecycle_figure(lifecycle, signal_label=signal_label),
                    "Ciclos de vida delimitados por evento elegível",
                ),
                (
                    2,
                    "sensibilidade_fronteiras_fases",
                    boundary_sensitivity,
                    _phase_boundary_sensitivity_figure(
                        boundary_sensitivity, signal_label=signal_label
                    ),
                    "Sensibilidade das fronteiras de gênese e faixa de pico",
                ),
            ]
            canonical_peak = peak_sensitivity.loc[
                peak_sensitivity["configuracao_canonica"].astype(str).str.lower().isin({"true", "1"})
            ]
            eligibility = (
                str(events["criterio_elegibilidade"].iloc[0])
                if "criterio_elegibilidade" in events and len(events)
                else "critério registrado na tabela de eventos"
            )
            takeaways = [
                f"O catálogo conserva {len(events)} eventos detectados; {len(eligible)} atendem ao corte de análise ({eligibility}).",
                "A gênese canônica é uma janela diagnóstica de 26 semanas antes do onset ONI; 13 e 39 semanas medem a dependência dessa escolha.",
                f"A faixa de pico canônica usa 90% do extremo e tem duração mediana de {canonical_peak['duracao_faixa_pico_meses'].median():.1f} meses entre os eventos elegíveis.",
            ]
            return diagnostics, lifecycle, takeaways
        if block == "F":
            peaks = pd.read_csv(paths[0])
            pulses = pd.read_csv(paths[1])
            stats = pd.read_csv(paths[2])
            equatorial = self._load_lon_parquet(paths[3], lon_min=120, lon_max=280)
            ssh = self._load_lon_parquet(paths[4])
            coupling = stats[
                stats["variavel"].astype(str).str.contains(
                    r"ssh|sla|d20|tilt|tau_x|wwv", case=False, regex=True
                )
            ].copy()
            value = "media_z_entre_eventos" if "media_z_entre_eventos" in coupling else "nivel_z_medio"
            diagnostics = [
                (
                    1,
                    "hovmoller_picos",
                    peaks,
                    _hovmoller_events_figure(equatorial, peaks, signal_label=signal_label),
                    "Hovmöller SSTA por evento com epicentro e pico de vento",
                ),
                (
                    2,
                    "kelvin_pulsos_sla",
                    pulses,
                    _kelvin_sla_figure(ssh, pulses, signal_label=signal_label),
                    "Propagação equatorial tipo Kelvin nas janelas com SSH",
                ),
                (
                    3,
                    "acoplamento_por_fase",
                    coupling,
                    _phase_profile_figure(
                        coupling, value, f"{signal_label}: vento, SSH/SLA e termoclina por fase"
                    ),
                    "Acoplamento vento–oceano e termoclina nas quatro fases",
                ),
            ]
            with_wind = peaks.loc[peaks["status"].eq("ok"), "antecedencia_vento_semanas"]
            leads = pd.to_numeric(with_wind, errors="coerce").dropna()
            covered = pulses.loc[pulses["status"].eq("ok"), "event_id"].nunique()
            thermal_extreme = "aquecimento" if sign > 0 else "resfriamento"
            takeaways = [
                f"{int(peaks['status'].eq('ok').sum())} eventos elegíveis possuem epicentro de {thermal_extreme} identificado na faixa 2S–2N.",
                f"O pico de vento antecede o epicentro térmico em {int((leads > 0).sum())}/{len(leads)} eventos (mediana {leads.median():+.0f} semanas)." if len(leads) else "Sem estatística de antecedência do vento.",
                f"O diagnóstico de Kelvin por SLA cobre {covered} janelas de evento; as demais não têm SSH diário disponível.",
            ]
            return diagnostics, peaks, takeaways
        if block == "G":
            composites = pd.read_csv(paths[0])
            diagnostics = [
                (
                    1,
                    "composto_ssta_classe",
                    composites,
                    _class_composite_figure(composites, signal_label=signal_label, sign=sign),
                    "Composto de SSTA por classe de intensidade",
                )
            ]
            classes = composites.loc[~composites["classe"].eq("todas_classes_analisadas")]
            n_by_class = classes.groupby("classe")["n_eventos"].max()
            takeaways = [
                "Compostos alinhados ao pico de cada evento; classes com pico ≥ 1 °C: "
                + ", ".join(f"{name} (n={int(count)})" for name, count in n_by_class.items())
                + ".",
                "A curva média por classe descreve a trajetória típica; a dispersão entre eventos permanece na tabela.",
            ]
            return diagnostics, composites, takeaways
        if block == "I":
            lags = pd.read_csv(paths[0])
            effects = pd.read_csv(paths[1])
            stability = pd.read_csv(paths[2])
            influence = pd.read_csv(paths[3])
            guides = pd.read_csv(paths[4])
            lag_columns = [column for column in ("variavel", "fase", "lag_semanas", "r_pearson") if column in lags]
            effect_columns = [column for column in ("variavel", "kendall_w_entre_fases", "q_friedman_bh") if column in effects]
            integrated = lags[lag_columns].merge(effects[effect_columns], on="variavel", how="outer")
            if {"variavel", "frequencia_selecao"}.issubset(stability.columns):
                stable = (
                    stability.groupby("variavel", as_index=False)["frequencia_selecao"]
                    .max()
                    .rename(columns={"frequencia_selecao": "estabilidade_bootstrap"})
                )
                integrated = integrated.merge(stable, on="variavel", how="outer")
            else:
                integrated["estabilidade_bootstrap"] = np.nan
            integrated["score_integrado"] = (
                pd.to_numeric(integrated.get("r_pearson"), errors="coerce").abs().fillna(0)
                + pd.to_numeric(integrated.get("kendall_w_entre_fases"), errors="coerce").fillna(0)
                + pd.to_numeric(integrated.get("estabilidade_bootstrap"), errors="coerce").fillna(0)
            )
            integrated = integrated.sort_values("score_integrado", ascending=False).head(30)
            diagnostics = [
                (
                    1,
                    "sintese_integrada",
                    integrated,
                    _bar_figure(integrated, label="variavel", value="score_integrado", title=f"{signal_label}: síntese auditável (efeito + lag + estabilidade)"),
                    "Síntese integrada de efeito, antecedência e estabilidade",
                ),
                (
                    2,
                    "influencia_percentual",
                    influence,
                    _influence_figure(influence, signal_label=signal_label),
                    "Influência percentual de cada variável por fase",
                ),
                (
                    3,
                    "guias_transicao_fase",
                    guides,
                    _transition_guides_figure(guides, signal_label=signal_label),
                    "Variáveis-guia para a mudança de fase",
                ),
            ]
            main_guides = guides.loc[guides.get("guia_principal", pd.Series(dtype=bool)) == True]  # noqa: E712
            takeaways = [
                f"A síntese integra {len(integrated)} variáveis; o score soma |r| no melhor lag, Kendall W e estabilidade bootstrap.",
                "A influência percentual é participação relativa (peso descritivo por fase e peso discriminante Kendall W), não variância explicada causal.",
                f"{len(main_guides)} marcadores físicos compõem os top-3 das transições adjacentes; eles são retrospectivos e não constituem detector online." if len(main_guides) else "Nenhum marcador de transição pôde ser selecionado.",
            ]
            return diagnostics, influence, takeaways
        if block == "L":
            duration = pd.read_csv(paths[0])
            sets_table = pd.read_csv(paths[1])
            duration = duration.copy()
            duration["rotulo"] = duration["fase"].astype(str) + " | " + duration["classe"].astype(str)
            diagnostics = [
                (
                    1,
                    "duracao_fases",
                    duration,
                    _bar_figure(duration, label="rotulo", value="duracao_media_semanas", title=f"{signal_label}: duração média das fases por classe"),
                    "Duração das quatro fases por classe de intensidade",
                ),
                (
                    2,
                    "conjuntos_variaveis_fase",
                    sets_table,
                    _variable_sets_figure(sets_table, signal_label=signal_label),
                    "Conjuntos de variáveis que melhor descrevem cada fase",
                ),
            ]
            descriptor = sets_table.loc[sets_table["integra_conjunto_descritor"] == True]  # noqa: E712
            families = descriptor.groupby("fase")["familia"].agg(lambda values: ", ".join(sorted(set(values))))
            takeaways = [
                "Conjunto físico diversificado por fase (alvo térmico excluído da seleção): "
                + "; ".join(f"{phase}: {family}" for phase, family in families.items())
                + ".",
                "A duração média por fase e classe usa somente os eventos elegíveis do sinal isolado.",
            ]
            return diagnostics, sets_table, takeaways
        raise KeyError(block)

    def _resolve_run_id(self) -> str:
        run_id = ""
        for path, _required, _role in self.input_paths():
            sidecar = Path(f"{path}.manifest.json")
            if sidecar.is_file():
                try:
                    run_id = str(json.loads(sidecar.read_text(encoding="utf-8")).get("run_id") or "").strip()
                except (OSError, json.JSONDecodeError):
                    run_id = ""
            if run_id:
                break
        return run_id

    def _publish_pairs(
        self,
        diagnostics: list[tuple[int, str, pd.DataFrame, object, str]],
        summary: pd.DataFrame,
        takeaways: list[str],
        limitations: list[str],
        *,
        run_id: str,
    ) -> WorkflowResult:
        from nino_brasil.viz import registrar_par_notebook

        artifacts = []
        for ordinal, slug, table, figure, title in diagnostics:
            artifact = registrar_par_notebook(
                figure,
                self.code,
                ordinal,
                table,
                slug=slug,
                titulo=title,
                descricao=self.spec.question,
                hipotese=self.spec.hypothesis,
                notebook=self.spec.relative_path,
                run_id=run_id,
            )
            plt.close(figure)
            artifacts.append(asdict(artifact))
        return WorkflowResult(
            artifacts=pd.DataFrame(artifacts),
            summary=summary.head(200),
            takeaways=takeaways,
            limitations=limitations,
        )

    def _phase4_summary(self) -> tuple[pd.DataFrame, object, list[str]]:
        tables = [_read_table(path) for path, _required, _role in self.input_paths() if path.is_file()]
        block = self.parsed.block
        label = "El Niño" if self.parsed.enso_type == "el_nino" else "La Niña"
        if block == "C":
            pixel_lags, regional_peak, field = tables
            columns = [
                column
                for column in (
                    "pixel_id",
                    "lat",
                    "lon",
                    "variavel",
                    "condicao_fonte",
                    "best_lag_sem_fdr",
                    "r_no_best_lag_fdr",
                    "grid_hash_sha256",
                    "analysis_run_id",
                )
                if column in pixel_lags
            ]
            table = pixel_lags[columns].copy()
            fig = _phase4_pixel_figure(
                table,
                f"F4 {label}: resposta e lag por pixel CHIRPS nas quatro fases",
            )
        else:
            ranking, hypotheses = tables
            table = hypotheses.copy()
            value = "n_targets_support" if "n_targets_support" in table else next(
                (column for column in table.select_dtypes(include="number")), table.columns[-1]
            )
            if {"regiao", "fase_fonte_em_t_menos_lag"}.issubset(table.columns):
                table["rotulo"] = (
                    table["regiao"].astype(str)
                    + " | "
                    + table["fase_fonte_em_t_menos_lag"].astype(str)
                )
                label_column = "rotulo"
            else:
                label_column = next(
                    (column for column in ("hypothesis", "condicao_fonte", "fase_fonte_em_t_menos_lag") if column in table),
                    table.columns[0],
                )
            fig = _bar_figure(table, label=label_column, value=value, title=f"F4 {label}: suporte espacial/extremos")
        takeaways = [
            f"A síntese contém {len(table)} linhas auditadas para {label}.",
            "Latitude, longitude e pixel_id pertencem à grade CHIRPS nativa.",
        ]
        if block == "C" and not regional_peak.empty:
            supported = regional_peak.loc[
                pd.to_numeric(
                    regional_peak.get("n_pixels_com_lag_fdr"), errors="coerce"
                ).fillna(0).gt(0)
            ]
            takeaways.append(
                f"O lag regional de pico foi resumido somente depois dos pixels; "
                f"{len(supported)}/{len(regional_peak)} regiões possuem ao menos um pixel FDR."
            )
        return table, fig, takeaways

    def _model_summary(self) -> tuple[pd.DataFrame, object, list[str]]:
        from scripts.notebook_run_viewer import audit_artifact_runs

        audit, selected = audit_artifact_runs(
            self.root, phase=self.parsed.phase, mode=self.mode
        )
        if not selected:
            raise FileNotFoundError(
                f"Nenhum ArtifactRun F{self.parsed.phase} {self.mode} completo e íntegro."
            )
        selected_ids = {str(item["run_id"]) for item in selected}
        runs = audit[audit["valid"] & audit["run_id"].astype(str).isin(selected_ids)].copy()
        runs["record_type"] = "artifact_run_gate"
        runs["gate_rate"] = np.where(
            runs["gate_rows"] > 0,
            runs["gate_passes"] / runs["gate_rows"],
            np.nan,
        )
        backend = runs["model"].fillna("").replace("", "modelo")
        if self.parsed.phase in {5, 7} and "augmentation" in runs:
            arm = np.where(runs["augmentation"].astype(bool), "aug=ON", "aug=OFF")
            backend = backend.astype(str) + " | " + arm
        runs["rotulo"] = backend.astype(str) + " | " + runs["run_id"]
        fig = _bar_figure(runs, label="rotulo", value="gate_rate", title=f"F{self.parsed.phase}: taxa de condições que passam o gate")
        ablation = pd.DataFrame()
        if self.parsed.phase in {5, 7}:
            ablation = _latest_augmentation_ablation(
                self.root,
                phase=self.parsed.phase,
                selected_run_ids=selected_ids,
            )
            if ablation.empty:
                raise FileNotFoundError(
                    f"Ablação augmentation ON/OFF auditável ausente para F{self.parsed.phase}."
                )
            direction = ablation["direction"].astype(str)
            delta = pd.to_numeric(ablation["delta_with_minus_without"], errors="coerce")
            ablation["effect_in_favour_of_augmentation"] = np.where(
                direction.eq("lower_is_better"), -delta, delta
            )
        table = pd.concat([runs, ablation], ignore_index=True, sort=False)
        takeaways = [
            f"Foram selecionados {len(selected)} ArtifactRuns íntegros pelo término validado.",
            f"O gate possui {int(runs['gate_passes'].sum())} aprovações em {int(runs['gate_rows'].sum())} linhas avaliadas.",
        ]
        if not ablation.empty:
            effects = pd.to_numeric(
                ablation["effect_in_favour_of_augmentation"], errors="coerce"
            ).dropna()
            takeaways.append(
                f"A ablação pareada encontrou melhora em {int((effects > 0).sum())}/"
                f"{len(effects)} comparações, sem aumentar o N de eventos independentes."
            )
        return table, fig, takeaways

    def run(self) -> WorkflowResult:
        self.require_inputs()
        if self.parsed.phase == 2:
            phase2_manifest = self.root / "data/processed/parquet/statistics/phase2_master_run_manifest.json"
            manifest = json.loads(phase2_manifest.read_text(encoding="utf-8"))
            run_id = str(manifest.get("run_id") or "").strip()
            if not run_id:
                raise RuntimeError("run_id auditável ausente para F2Z")
            diagnostics, takeaways = self._phase2_diagnostics()
            from nino_brasil.viz import registrar_par_notebook

            artifacts = []
            for ordinal, slug, table, figure, title in diagnostics:
                artifact = registrar_par_notebook(
                    figure,
                    self.code,
                    ordinal,
                    table,
                    slug=slug,
                    titulo=title,
                    descricao=self.spec.question,
                    hipotese=self.spec.hypothesis,
                    notebook=self.spec.relative_path,
                    run_id=run_id,
                )
                plt.close(figure)
                artifacts.append(asdict(artifact))
            return WorkflowResult(
                artifacts=pd.DataFrame(artifacts),
                summary=diagnostics[0][2],
                takeaways=takeaways,
                limitations=[
                    "A aprovação vale para análise estatística semanal sem limpeza adicional; lacunas continuam explícitas na tabela de contrato.",
                    "Em modelagem preditiva, qualquer climatologia ou ajuste deve ser estimado somente no período de treino de cada fold.",
                ],
            )
        if self.code == "F3NinoA":
            diagnostics, summary, takeaways = self._f3nino_a_weekly_indices()
            run_id = self._resolve_run_id()
            if not run_id:
                raise RuntimeError("run_id auditável ausente para F3NinoA")
            return self._publish_pairs(
                diagnostics,
                summary,
                takeaways,
                limitations=[
                    "Tendências lineares da série inteira são diagnósticas; autocorrelação, mudanças de fonte e não estacionariedade impedem tratá-las como teste inferencial.",
                    "Os testes entre fases, lags e estabilidade permanecem respectivamente em F3NinoD, F3NinoC e F3NinoE.",
                ],
                run_id=run_id,
            )
        if self.parsed.phase == 3 and self.parsed.block in {"B", "F", "G", "I", "L"}:
            diagnostics, summary, takeaways = self._phase3_multi()
            run_id = self._resolve_run_id()
            if not run_id:
                raise RuntimeError(f"run_id auditável ausente para {self.code}")
            return self._publish_pairs(
                diagnostics,
                summary,
                takeaways,
                limitations=[
                    "As figuras resumem tabelas auditadas; consulte tabela, manifesto e unidade antes de interpretar valores individuais.",
                    "Leituras de propagação e influência são diagnósticas por evento; não há detector formal de onda nem atribuição causal.",
                    "Somente eventos elegíveis do sinal isolado entram nos painéis; o número de eventos independentes acompanha cada tabela.",
                ],
                run_id=run_id,
            )
        elif self.parsed.phase == 3:
            table, figure, takeaways = self._phase3_summary()
        elif self.parsed.phase == 4:
            table, figure, takeaways = self._phase4_summary()
        else:
            table, figure, takeaways = self._model_summary()

        from nino_brasil.viz import registrar_par_notebook

        run_id = ""
        for path, _required, _role in self.input_paths():
            sidecar = Path(f"{path}.manifest.json")
            if sidecar.is_file():
                try:
                    run_id = str(json.loads(sidecar.read_text(encoding="utf-8")).get("run_id") or "").strip()
                except (OSError, json.JSONDecodeError):
                    run_id = ""
            if run_id:
                break
        if not run_id and "analysis_run_id" in table and len(table):
            run_id = str(table["analysis_run_id"].dropna().astype(str).iloc[0]).strip()
        if not run_id and "run_id" in table and len(table):
            run_id = str(table["run_id"].dropna().astype(str).iloc[0]).strip()
        lineage_columns = (
            "analysis_run_id",
            "run_id",
            "with_augmentation_run_id",
            "without_augmentation_run_id",
            "parent_f3_run_id",
            "parent_f4c_run_id",
        )
        lineage_ids: set[str] = set()
        for column in lineage_columns:
            if column not in table:
                continue
            lineage_ids.update(
                value
                for value in table[column].dropna().astype(str).str.strip()
                if value and value.lower() != "nan"
            )
        if len(lineage_ids) > 1:
            digest = hashlib.sha256(
                "\n".join(sorted(lineage_ids)).encode("utf-8")
            ).hexdigest()[:16]
            run_id = f"aggregate_{digest}"
        elif not run_id and len(lineage_ids) == 1:
            run_id = next(iter(lineage_ids))
        if not run_id:
            phase2_manifest = self.root / "data/processed/parquet/statistics/phase2_master_run_manifest.json"
            if phase2_manifest.is_file():
                run_id = str(json.loads(phase2_manifest.read_text(encoding="utf-8")).get("run_id") or "").strip()
        if not run_id:
            raise RuntimeError(f"run_id auditável ausente para {self.code}")

        artifact = registrar_par_notebook(
            figure,
            self.code,
            1,
            table,
            titulo=self.spec.title,
            descricao=self.spec.question,
            hipotese=self.spec.hypothesis,
            notebook=self.spec.relative_path,
            run_id=run_id,
        )
        artifacts = pd.DataFrame([asdict(artifact)])
        limitations = [
            "A figura resume a tabela publicada; consulte a tabela completa e seu manifesto antes de interpretar valores individuais.",
            "Resultados negativos ou tabelas vazias são preservados e não são convertidos em falha técnica.",
        ]
        return WorkflowResult(
            artifacts=artifacts,
            summary=table.head(100),
            takeaways=takeaways,
            limitations=limitations,
        )
