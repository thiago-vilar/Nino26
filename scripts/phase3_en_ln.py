#!/usr/bin/env python3
"""Build the canonical, auditable Phase 3 scientific products.

The monthly local ONI is used only to delimit completed ENSO events.  Every
physical diagnostic runs on the Phase 2 weekly master matrix.  Retrospective
event diagnostics and rolling-origin prediction targets are written as
separate products so future-defined peak phases never become model features.

Examples
--------
Full registered analysis::

    python scripts/phase3_en_ln.py

Fast contract smoke (does not replace an official full run)::

    python scripts/phase3_en_ln.py --quick --output-suffix _quick
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Sequence
import uuid

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nino_brasil.events.enso import (  # noqa: E402
    ENSO_ACTIVE_PHASES,
    ENSO_STATE_ORDER,
    EnsoLifecycleConfig,
    build_enso_lifecycle,
    build_rolling_origin_targets,
    detect_enso_events,
    peak_band_sensitivity,
)
from nino_brasil.artifact_codes import (  # noqa: E402
    notebook_code_for,
    table_code,
)
from nino_brasil.stats.phase3_inference import (  # noqa: E402
    PRECURSOR_TARGET_EXCLUSION_POLICY,
    bootstrap_lag_selection_by_event,
    phase3_precursor_columns,
    scan_lagged_correlations,
    select_best_lags,
)
from nino_brasil.stats.preprocessing import (  # noqa: E402
    SeasonalTrendConfig,
    full_sample_diagnostic_transform,
)
from nino_brasil.stats.semantic_tables import (  # noqa: E402
    SemanticTableContract,
    write_semantic_csv,
)
from nino_brasil.stats.validation import (  # noqa: E402
    event_purged_rolling_origin_folds,
    folds_audit_table,
    required_purge_weeks,
)
from nino_brasil.stats.significance import benjamini_hochberg_fdr  # noqa: E402


FEAT = ROOT / "data/processed/parquet/features"
STATS = ROOT / "data/processed/parquet/statistics"
MASTER_PATH = FEAT / "nino34_master_weekly.csv"
ONI_PATH = FEAT / "nino34_monthly_oisst.csv"
HOVMOLLER_SSTA_PATH = FEAT / "equatorial_pacific_ssta_weekly_by_lon.parquet"
SSH_EVENTS_PATH = FEAT / "ssh_equatorial_daily_by_lon_events.parquet"
CODE_INPUTS = (
    Path(__file__).resolve(),
    ROOT / "src/nino_brasil/events/enso.py",
    ROOT / "src/nino_brasil/stats/phase3_inference.py",
    ROOT / "src/nino_brasil/stats/preprocessing.py",
    ROOT / "src/nino_brasil/stats/semantic_tables.py",
    ROOT / "src/nino_brasil/stats/validation.py",
    ROOT / "src/nino_brasil/artifact_codes.py",
)

# Public F3 tables inherit the code of the notebook responsible for their
# scientific interpretation.  The descriptive suffix remains human-readable;
# the immutable audit key is the TabF3Nino*/TabF3Nina* pre-code.
PHASE3_TABLE_LAYOUT: dict[str, tuple[str, int, str]] = {
    "phase3_preprocessing_contract": ("A", 1, "preprocessamento"),
    "phase3_fase_stats_variaveis": ("A", 2, "estatisticas_por_fase"),
    "phase3_events_en_ln": ("B", 1, "eventos"),
    "phase3_fases_semanais_en_ln": ("B", 2, "fases_semanais"),
    "phase3_event_lifecycle_en_ln": ("B", 3, "ciclo_eventos"),
    "phase3_peak_band_sensitivity": ("B", 4, "sensibilidade_faixa_pico"),
    "phase3_phase_boundary_sensitivity": ("B", 5, "sensibilidade_fronteiras_fases"),
    "phase3_lag_scan_en_ln_fases": ("C", 1, "varredura_lags"),
    "phase3_best_lags_fdr": ("C", 2, "melhores_lags_fdr"),
    "phase3_discriminantes_por_periodo": ("D", 1, "friedman_fdr"),
    "phase3_lag_event_bootstrap_summary": ("E", 1, "bootstrap_resumo"),
    "phase3_lag_event_bootstrap_replicates": ("E", 2, "bootstrap_replicas"),
    "phase3_rolling_origin_targets": ("I", 1, "alvos_rolling_origin"),
    "phase3_rolling_origin_folds": ("I", 2, "folds_eventos"),
    "phase3_pca_por_fase": ("K", 1, "pca_variancia"),
    "phase3_pca_loadings_por_fase": ("K", 2, "pca_loadings"),
    "phase3_duracao_por_tipo_classe": ("L", 1, "duracao_fases"),
    "phase3_hovmoller_picos_eventos": ("F", 1, "hovmoller_picos"),
    "phase3_kelvin_pulsos_sla": ("F", 2, "kelvin_pulsos_sla"),
    "phase3_composto_ssta_classe": ("G", 1, "composto_ssta_classe"),
    "phase3_influencia_percentual": ("I", 3, "influencia_percentual"),
    "phase3_guias_transicao_fase": ("I", 4, "guias_transicao_fase"),
    "phase3_conjuntos_variaveis_fase": ("L", 2, "conjuntos_variaveis_fase"),
}

VARIABLE_FAMILIES: dict[str, str] = {
    "nino34_ssta": "alvo_termico_nino34",
    "d20_m": "oceano_subsuperficie",
    "tilt_m": "oceano_subsuperficie",
    "tilt_slope": "oceano_subsuperficie",
    "ohc_0_100": "oceano_subsuperficie",
    "ohc_0_300": "oceano_subsuperficie",
    "ohc_0_700": "oceano_subsuperficie",
    "ohc_300_700": "oceano_subsuperficie",
    "wwv": "oceano_subsuperficie",
    "t50m": "oceano_subsuperficie",
    "t100m": "oceano_subsuperficie",
    "t150m": "oceano_subsuperficie",
    "t200m": "oceano_subsuperficie",
    "t300m": "oceano_subsuperficie",
    "t500m": "oceano_subsuperficie",
    "t700m": "oceano_subsuperficie",
    "ssh_m": "oceano_superficie",
    "tau_x_anom": "acoplamento_vento_oceano",
    "u10_anom": "acoplamento_vento_oceano",
    "v10_anom": "acoplamento_vento_oceano",
    "mslp_anom": "atmosfera",
    "tcwv_anom": "atmosfera",
    "slhf_anom": "atmosfera",
    "sshf_anom": "atmosfera",
    "ssr_anom": "atmosfera",
    "str_anom": "atmosfera",
    "u850_anom": "atmosfera",
    "u200_anom": "atmosfera",
    "omega850_anom": "atmosfera",
    "omega500_anom": "atmosfera",
    "div850_anom": "atmosfera",
}


def variable_family(name: str) -> str:
    return VARIABLE_FAMILIES.get(str(name), "outra")


def lon_label(lon: float) -> str:
    lon = float(lon)
    if lon == 180:
        return "180"
    if lon < 180:
        return f"{int(round(lon))}E"
    return f"{int(round(360 - lon))}W"


def phase3_public_stem(stem: str, enso_type: str | None) -> str:
    """Return the public Tab pre-code for an isolated F3 analysis."""

    if enso_type is None:
        # Kept only for backwards-compatible diagnostics; official execution
        # is always isolated by the Nino/Nina wrappers.
        return stem
    try:
        block, ordinal, slug = PHASE3_TABLE_LAYOUT[stem]
    except KeyError as exc:
        raise KeyError(f"F3 table without public artifact code: {stem}") from exc
    notebook = notebook_code_for(3, block, enso_type)
    return table_code(notebook, ordinal, slug=slug)
ALREADY_ANOMALOUS = {
    "nino34_ssta",
    "tau_x_anom",
    "tau_x_anom_nino34_pa",
    "u10_anom",
    "v10_anom",
    "mslp_anom",
    "tcwv_anom",
    "slhf_anom",
    "sshf_anom",
    "ssr_anom",
    "str_anom",
    "u850_anom",
    "u200_anom",
    "omega850_anom",
    "omega500_anom",
    "div850_anom",
}


def load_master() -> pd.DataFrame:
    frame = pd.read_csv(MASTER_PATH, parse_dates=["week_ending_sunday"])
    frame = frame.set_index("week_ending_sunday").sort_index()
    frame = frame.drop(columns=[column for column in ("ocean_source_code",) if column in frame])
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if frame.index.has_duplicates:
        raise ValueError("Phase 2 master matrix contains duplicate weeks")
    return frame


def load_oni() -> pd.Series:
    frame = pd.read_csv(ONI_PATH, parse_dates=["time"])
    if "month_complete" in frame:
        complete = frame["month_complete"]
        if complete.dtype == object:
            complete = complete.astype(str).str.lower().isin({"true", "1", "yes", "sim"})
        frame = frame.loc[complete.astype(bool)]
    return frame.set_index("time")["oni_local_c"].astype(float).sort_index()


def lifecycle_event_table(lifecycle: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    classes = events.set_index("event_id")["classe"].to_dict()
    rows = []
    active = lifecycle[lifecycle["fase"].isin(ENSO_ACTIVE_PHASES)]
    for (event_id, event_type, phase), group in active.groupby(["event_id", "tipo", "fase"]):
        rows.append(
            {
                "event_id": event_id,
                "tipo": event_type,
                "classe": classes.get(event_id, ""),
                "fase": phase,
                "inicio": group.index.min(),
                "fim": group.index.max(),
                "duracao_semanas": int(len(group)),
                "modo_rotulo": "diagnostico_retrospectivo",
            }
        )
    return pd.DataFrame(rows).sort_values(["inicio", "fase"]).reset_index(drop=True)


def phase_boundary_sensitivity(
    oni: pd.Series,
    events: pd.DataFrame,
    weekly_index: pd.DatetimeIndex,
    *,
    genesis_windows_weeks: Sequence[int] = (13, 26, 39),
    peak_fractions: Sequence[float] = (0.80, 0.90, 0.95),
    canonical_genesis_weeks: int = 26,
    canonical_peak_fraction: float = 0.90,
) -> pd.DataFrame:
    """Quantify how all four phase boundaries react to declared choices.

    The literature supports a pre-onset recharge stage and a mature/peak band,
    but it does not prescribe a universal weekly genesis boundary.  Therefore
    the canonical 26-week pre-onset window is accompanied by 13/39-week
    alternatives, while the peak band is recomputed at 80/90/95% of each
    event's own maximum.  The event onset/end definition itself stays fixed.
    """

    genesis_windows = tuple(dict.fromkeys(int(value) for value in genesis_windows_weeks))
    fractions = tuple(dict.fromkeys(float(value) for value in peak_fractions))
    if not genesis_windows or any(value < 1 for value in genesis_windows):
        raise ValueError("genesis_windows_weeks must contain positive integers")
    if not fractions or any(not 0 < value <= 1 for value in fractions):
        raise ValueError("peak_fractions must be in (0, 1]")

    tables: list[pd.DataFrame] = []
    for genesis_weeks in genesis_windows:
        for peak_fraction in fractions:
            bands = peak_band_sensitivity(
                oni,
                events,
                fractions=(peak_fraction,),
                canonical_fraction=canonical_peak_fraction,
            ).set_index("event_id")
            varied_events = events.copy()
            varied_events["faixa_pico_inicio"] = varied_events["event_id"].map(
                bands["faixa_pico_inicio"]
            )
            varied_events["faixa_pico_fim"] = varied_events["event_id"].map(
                bands["faixa_pico_fim"]
            )
            config = EnsoLifecycleConfig(
                genesis_weeks=genesis_weeks,
                peak_fraction=peak_fraction,
                peak_sensitivity_fractions=fractions,
            )
            varied_lifecycle = build_enso_lifecycle(
                varied_events,
                weekly_index,
                config=config,
            )
            table = lifecycle_event_table(varied_lifecycle, varied_events)
            table["janela_genese_semanas"] = genesis_weeks
            table["fracao_faixa_pico"] = peak_fraction
            table["configuracao_canonica"] = bool(
                genesis_weeks == canonical_genesis_weeks
                and np.isclose(peak_fraction, canonical_peak_fraction)
            )
            table["interpretacao_sensibilidade"] = (
                "janela de genese e faixa relativa de pico variam; onset e fim do evento permanecem fixos"
            )
            tables.append(table)

    result = pd.concat(tables, ignore_index=True)
    canonical = result.loc[result["configuracao_canonica"], [
        "event_id",
        "fase",
        "duracao_semanas",
    ]].rename(columns={"duracao_semanas": "duracao_canonica_semanas"})
    result = result.merge(canonical, on=["event_id", "fase"], how="left", validate="many_to_one")
    result["delta_duracao_vs_canonica_semanas"] = (
        result["duracao_semanas"] - result["duracao_canonica_semanas"]
    )
    return result.sort_values(
        ["event_id", "janela_genese_semanas", "fracao_faixa_pico", "fase"]
    ).reset_index(drop=True)


def phase_statistics(
    transformed: pd.DataFrame,
    lifecycle: pd.DataFrame,
    *,
    fdr_alpha: float = 0.05,
    event_types: Sequence[str] = ("el_nino", "la_nina"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < fdr_alpha < 1:
        raise ValueError("fdr_alpha must be in (0, 1)")
    event_types = tuple(dict.fromkeys(event_types))
    if not event_types or not set(event_types).issubset({"el_nino", "la_nina"}):
        raise ValueError("event_types must contain el_nino and/or la_nina")
    joined = transformed.join(lifecycle[["tipo", "fase", "event_id"]], how="inner")
    active = joined[joined["fase"].isin(ENSO_ACTIVE_PHASES) & joined["event_id"].ne("")]
    active = active[active["tipo"].isin(event_types)]
    variables = list(transformed.columns)
    event_means = (
        active.groupby(["tipo", "fase", "event_id"])[variables]
        .mean()
        .reset_index()
    )
    rows = []
    for (event_type, phase), group in event_means.groupby(["tipo", "fase"]):
        weekly_group = active[
            active["tipo"].eq(event_type) & active["fase"].eq(phase)
        ]
        for variable in variables:
            values = group[variable].dropna()
            event_volatility = weekly_group.groupby("event_id")[variable].apply(
                lambda series: series.diff().abs().mean()
            ).dropna()
            mean_level = float(values.mean()) if len(values) else np.nan
            mean_volatility = float(event_volatility.mean()) if len(event_volatility) else np.nan
            rows.append(
                {
                    "tipo": event_type,
                    "fase": phase,
                    "variavel": variable,
                    "media_z_entre_eventos": mean_level,
                    "desvio_z_entre_eventos": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "volatilidade_semanal_media_entre_eventos": mean_volatility,
                    "n_eventos_independentes": int(len(values)),
                    "n_semanas_descritivas": int(weekly_group[variable].notna().sum()),
                    # Aliases retained only so older report builders can read
                    # the table; inference still uses event-level quantities.
                    "nivel_z_medio": mean_level,
                    "volatilidade_sem": mean_volatility,
                    "n_semanas": int(weekly_group[variable].notna().sum()),
                    "unidade_inferencial": "media_evento_fase",
                    "evaluation_mode": "diagnostico_retrospectivo",
                }
            )
    phase_stats = pd.DataFrame(rows)

    discrimination_rows = []
    for event_type in event_types:
        subset = event_means[event_means["tipo"].eq(event_type)]
        for variable in variables:
            pivot = subset.pivot(index="event_id", columns="fase", values=variable)
            pivot = pivot.reindex(columns=ENSO_ACTIVE_PHASES).dropna()
            n_events = len(pivot)
            statistic = p_value = kendall_w = np.nan
            if n_events >= 3:
                result = friedmanchisquare(*(pivot[phase].to_numpy() for phase in ENSO_ACTIVE_PHASES))
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
                kendall_w = float(statistic / (n_events * (len(ENSO_ACTIVE_PHASES) - 1)))
            discrimination_rows.append(
                {
                    "tipo": event_type,
                    "variavel": variable,
                    "friedman_chi2": statistic,
                    "p_friedman": p_value,
                    "kendall_w_entre_fases": kendall_w,
                    "n_eventos_completos": n_events,
                    "unidade_inferencial": "evento_pareado_nas_quatro_fases",
                    "interpretacao": (
                        "Kendall W e p de Friedman brutos; confirmacao somente apos "
                        "BH-FDR separado por tipo ENSO"
                    ),
                }
            )
    discrimination = pd.DataFrame(discrimination_rows)
    corrected: list[pd.DataFrame] = []
    for event_type, group in discrimination.groupby("tipo", sort=False):
        group = group.copy()
        rejected, q_values = benjamini_hochberg_fdr(
            group["p_friedman"].to_numpy(),
            alpha=fdr_alpha,
        )
        family_size = int(len(group))
        valid_p_count = int(np.isfinite(group["p_friedman"]).sum())
        group["q_friedman_bh"] = q_values
        group["significativo_friedman_fdr"] = rejected
        group["friedman_family_id"] = f"F3_friedman_{event_type}_all_variables"
        group["friedman_family_size"] = family_size
        group["friedman_valid_p_count"] = valid_p_count
        group["friedman_fdr_alpha"] = fdr_alpha
        group["resultado_confirmatorio"] = np.where(
            rejected,
            "confirmado_q_bh_le_alpha",
            "nao_confirmado_fdr",
        )
        corrected.append(group)
    return phase_stats, pd.concat(corrected, ignore_index=True)


def phase_pca(transformed: pd.DataFrame, lifecycle: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = transformed.join(lifecycle[["tipo", "fase", "event_id"]], how="inner")
    active = joined[joined["fase"].isin(ENSO_ACTIVE_PHASES) & joined["event_id"].ne("")]
    variables = list(transformed.columns)
    event_means = active.groupby(["tipo", "fase", "event_id"])[variables].mean()
    variance_rows: list[dict[str, object]] = []
    loading_rows: list[dict[str, object]] = []
    for (event_type, phase), group in event_means.groupby(level=[0, 1]):
        matrix = group.droplevel([0, 1]).dropna(axis=1, how="any").dropna(axis=0, how="any")
        if len(matrix) < 3 or matrix.shape[1] < 2:
            continue
        scaled = StandardScaler().fit_transform(matrix)
        pca = PCA(n_components=min(len(matrix) - 1, matrix.shape[1])).fit(scaled)
        for component, explained in enumerate(pca.explained_variance_ratio_[:4], start=1):
            variance_rows.append(
                {
                    "tipo": event_type,
                    "fase": phase,
                    "componente": f"PC{component}",
                    "var_explicada": float(explained),
                    "var_acumulada": float(pca.explained_variance_ratio_[:component].sum()),
                    "n_eventos_independentes": int(len(matrix)),
                    "n_variaveis": int(matrix.shape[1]),
                    "evaluation_mode": "pca_descritiva_evento_fase",
                }
            )
        for variable_index, variable in enumerate(matrix.columns):
            row = {
                "tipo": event_type,
                "fase": phase,
                "variavel": variable,
                "n_eventos_independentes": int(len(matrix)),
                "evaluation_mode": "pca_descritiva_evento_fase",
            }
            for component in range(min(4, len(pca.components_))):
                row[f"PC{component + 1}"] = float(pca.components_[component, variable_index])
            loading_rows.append(row)
    return pd.DataFrame(variance_rows), pd.DataFrame(loading_rows)


def load_equatorial_ssta_by_lon() -> pd.DataFrame:
    frame = pd.read_parquet(HOVMOLLER_SSTA_PATH)
    frame.index = pd.to_datetime(frame.index)
    frame.columns = frame.columns.astype(float)
    return frame.loc[:, (frame.columns >= 120) & (frame.columns <= 280)].sort_index()


def load_ssh_daily_by_lon() -> pd.DataFrame:
    frame = pd.read_parquet(SSH_EVENTS_PATH)
    frame.index = pd.to_datetime(frame.index)
    frame.columns = frame.columns.astype(float)
    return frame.sort_index()


def hovmoller_event_peaks(
    equatorial_ssta: pd.DataFrame,
    tau_x_weekly: pd.Series,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Extremos de aquecimento (epicentro) e de vento zonal por evento.

    O epicentro e o extremo espaco-temporal da SSTA equatorial 2S-2N dentro do
    evento; o pico de vento e o extremo da anomalia de estresse zonal na janela
    que vai de 26 semanas antes do onset ate o fim da faixa de pico.  Ambos sao
    orientados pelo sinal (El Nino: quente/oeste positivo; La Nina: frio/leste).
    """

    rows: list[dict[str, object]] = []
    lon = equatorial_ssta.columns.to_numpy(dtype=float)
    for _, event in events.iterrows():
        sign = 1.0 if event["tipo"] == "el_nino" else -1.0
        onset = pd.Timestamp(event["onset"])
        end = pd.Timestamp(event["fim"])
        peak_band_end = pd.Timestamp(event["faixa_pico_fim"])
        segment = equatorial_ssta.loc[onset:end].dropna(how="all")
        row: dict[str, object] = {
            "event_id": event["event_id"],
            "tipo": event["tipo"],
            "classe": event["classe"],
            "onset": onset,
            "pico": pd.Timestamp(event["pico"]),
            "fim": end,
            "oni_pico_c": float(event["oni_pico_c"]),
        }
        if segment.empty:
            row.update({"status": "sem_dados_ssta_equatorial"})
            rows.append(row)
            continue
        signed = sign * segment.to_numpy(dtype=float)
        flat = np.nanargmax(signed)
        time_pos, lon_pos = np.unravel_index(flat, signed.shape)
        epic_time = pd.Timestamp(segment.index[time_pos])
        epic_lon = float(lon[lon_pos])
        wind = tau_x_weekly.loc[onset - pd.Timedelta(weeks=26): peak_band_end].dropna()
        if wind.empty:
            wind_time, wind_value = pd.NaT, np.nan
        else:
            wind_time = pd.Timestamp((sign * wind).idxmax())
            wind_value = float(wind.loc[wind_time])
        lead_weeks = (
            round(float((epic_time - wind_time).days) / 7.0, 1)
            if pd.notna(wind_time)
            else np.nan
        )
        row.update(
            {
                "epicentro_data": epic_time,
                "epicentro_lon": epic_lon,
                "epicentro_lon_rotulo": lon_label(epic_lon),
                "ssta_extremo_c": float(segment.iloc[time_pos, lon_pos]),
                "pico_vento_data": wind_time,
                "tau_x_extremo_pa": wind_value,
                "antecedencia_vento_semanas": lead_weeks,
                "status": "ok",
                "leitura": (
                    "pico de vento antecede o epicentro termico"
                    if pd.notna(wind_time) and np.isfinite(lead_weeks) and lead_weeks > 0
                    else "verificar ordenamento vento-oceano no Hovmoller"
                ),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out["janela_vento"] = "onset-26 semanas ate o fim da faixa de pico"
    out["faixa_latitudinal"] = "2S-2N"
    return out


KELVIN_SPEED_M_S = 2.4
KELVIN_CROSSING_DAYS = 65  # ~100 graus de longitude a 2.4 m/s


def kelvin_sla_pulses(
    ssh_by_lon: pd.DataFrame,
    events: pd.DataFrame,
    *,
    min_coverage_days: int = 60,
    max_pulses: int = 3,
) -> pd.DataFrame:
    """Pulsos de SLA no Pacifico oeste (150E-180) e chegada estimada no leste.

    A SLA e local a janela do evento (SSH menos a media temporal por
    longitude).  Pulsos sao maximos da media 150E-180 suavizada em 15 dias,
    separados por pelo menos 75 dias, orientados pelo sinal do evento.
    """

    rows: list[dict[str, object]] = []
    lon = ssh_by_lon.columns.to_numpy(dtype=float)
    west = (lon >= 150) & (lon <= 180)
    for _, event in events.iterrows():
        sign = 1.0 if event["tipo"] == "el_nino" else -1.0
        start = pd.Timestamp(event["onset"]) - pd.Timedelta(weeks=52)
        end = pd.Timestamp(event["fim"])
        segment = ssh_by_lon.loc[start:end].dropna(how="all")
        base = {
            "event_id": event["event_id"],
            "tipo": event["tipo"],
            "classe": event["classe"],
            "janela_inicio": start,
            "janela_fim": end,
            "cobertura_ssh_dias": int(len(segment)),
        }
        if len(segment) < min_coverage_days:
            rows.append(
                {
                    **base,
                    "pulso_ordinal": 0,
                    "pulso_data": pd.NaT,
                    "sla_oeste_m": np.nan,
                    "chegada_estimada_leste": pd.NaT,
                    "velocidade_assumida_m_s": KELVIN_SPEED_M_S,
                    "status": "sem_cobertura_ssh",
                }
            )
            continue
        sla = segment - segment.mean(axis=0)
        west_series = (
            (sign * sla.loc[:, west].mean(axis=1))
            .rolling(15, min_periods=5)
            .mean()
            .dropna()
        )
        remaining = west_series.copy()
        ordinal = 0
        while ordinal < max_pulses and not remaining.empty:
            amplitude = float(remaining.max())
            if amplitude < 0.02:
                break
            pulse_time = pd.Timestamp(remaining.idxmax())
            ordinal += 1
            arrival = pulse_time + pd.Timedelta(days=KELVIN_CROSSING_DAYS)
            rows.append(
                {
                    **base,
                    "pulso_ordinal": ordinal,
                    "pulso_data": pulse_time,
                    "sla_oeste_m": round(float(sign * amplitude), 3),
                    "chegada_estimada_leste": min(arrival, segment.index.max()),
                    "velocidade_assumida_m_s": KELVIN_SPEED_M_S,
                    "status": "ok",
                }
            )
            remaining = remaining[
                (remaining.index < pulse_time - pd.Timedelta(days=75))
                | (remaining.index > pulse_time + pd.Timedelta(days=75))
            ]
        if ordinal == 0 and len(segment) >= min_coverage_days:
            rows.append(
                {
                    **base,
                    "pulso_ordinal": 0,
                    "pulso_data": pd.NaT,
                    "sla_oeste_m": np.nan,
                    "chegada_estimada_leste": pd.NaT,
                    "velocidade_assumida_m_s": KELVIN_SPEED_M_S,
                    "status": "sem_pulso_detectado",
                }
            )
    out = pd.DataFrame(rows)
    out["definicao_pulso"] = (
        "extremo da media SLA 150E-180 suavizada 15 dias, minimo 0.02 m, "
        "separacao 75 dias, orientado pelo sinal do evento"
    )
    return out


def ssta_class_composites(
    ssta_weekly: pd.Series,
    events: pd.DataFrame,
    *,
    max_lag_weeks: int = 52,
) -> pd.DataFrame:
    """Composto de SSTA Nino 3.4 por classe, alinhado ao pico de cada evento."""

    series = ssta_weekly.dropna()
    index = series.index
    lags = list(range(-max_lag_weeks, max_lag_weeks + 1))
    per_event: dict[str, pd.Series] = {}
    for _, event in events.iterrows():
        pos = index.get_indexer([pd.Timestamp(event["pico"])], method="nearest")[0]
        values = {
            lag: float(series.iloc[pos + lag])
            for lag in lags
            if 0 <= pos + lag < len(index)
        }
        per_event[str(event["event_id"])] = pd.Series(values).reindex(lags)
    matrix = pd.DataFrame(per_event).T
    classes = events.set_index("event_id")["classe"]
    groups: dict[str, pd.DataFrame] = {
        str(class_name): matrix.loc[members.index]
        for class_name, members in classes.groupby(classes)
    }
    groups["todas_classes_analisadas"] = matrix
    event_type = str(events["tipo"].iloc[0]) if len(events) else ""
    rows: list[dict[str, object]] = []
    for class_name, group in groups.items():
        for lag in lags:
            values = group[lag].dropna()
            rows.append(
                {
                    "tipo": event_type,
                    "classe": class_name,
                    "semana_rel_pico": int(lag),
                    "ssta_media_c": float(values.mean()) if len(values) else np.nan,
                    "ssta_dp_c": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "n_eventos": int(len(values)),
                }
            )
    out = pd.DataFrame(rows)
    out["alinhamento"] = "semana 0 = semana mais proxima do pico ONI do evento"
    return out


def percent_influence(
    phase_stats: pd.DataFrame,
    discrimination: pd.DataFrame,
) -> pd.DataFrame:
    """Percentual de influencia de cada variavel na condicao do sinal.

    Duas leituras complementares e transparentes, sempre excluindo a propria
    SSTA Nino 3.4 usada para definir o estado do evento:
    - peso descritivo do estado por fase: participacao de |media z entre
      eventos| da variavel no total da fase;
    - peso discriminante entre fases: participacao do Kendall W da variavel no
      total do sinal (Friedman pareado por evento), com o status BH-FDR.
    """

    rows: list[dict[str, object]] = []
    for (event_type, phase), group in phase_stats.groupby(["tipo", "fase"]):
        group = group.loc[
            group["variavel"].map(variable_family).ne("alvo_termico_nino34")
        ].copy()
        magnitudes = group.set_index("variavel")["media_z_entre_eventos"].abs()
        total = float(magnitudes.sum())
        for variable, magnitude in magnitudes.items():
            rows.append(
                {
                    "tipo": event_type,
                    "fase": str(phase),
                    "variavel": variable,
                    "familia": variable_family(variable),
                    "metrica": "peso_descritivo_do_estado_pct",
                    "valor_pct": float(100.0 * magnitude / total) if total > 0 else np.nan,
                    "base_metrica": (
                        "|media_z_entre_eventos| normalizada dentro da fase; "
                        "SSTA alvo excluida"
                    ),
                    "n_eventos": int(group.loc[group["variavel"].eq(variable), "n_eventos_independentes"].iloc[0]),
                    "confirmado_fdr": pd.NA,
                }
            )
    for event_type, group in discrimination.groupby("tipo"):
        group = group.loc[
            group["variavel"].map(variable_family).ne("alvo_termico_nino34")
        ].copy()
        weights = group.set_index("variavel")["kendall_w_entre_fases"].clip(lower=0)
        total = float(weights.sum())
        confirmed = group.set_index("variavel")["significativo_friedman_fdr"]
        n_events = group.set_index("variavel")["n_eventos_completos"]
        for variable, weight in weights.items():
            rows.append(
                {
                    "tipo": event_type,
                    "fase": "todas",
                    "variavel": variable,
                    "familia": variable_family(variable),
                    "metrica": "peso_discriminante_entre_fases_pct",
                    "valor_pct": float(100.0 * weight / total) if total > 0 else np.nan,
                    "base_metrica": (
                        "Kendall W do Friedman pareado normalizado entre variaveis; "
                        "SSTA alvo excluida"
                    ),
                    "n_eventos": int(n_events.loc[variable]),
                    "confirmado_fdr": bool(confirmed.loc[variable]),
                }
            )
    out = pd.DataFrame(rows)
    out["interpretacao"] = (
        "participacao relativa das variaveis fisicas com SSTA alvo excluida, "
        "nao variancia explicada causal; ler junto com o rigor estatistico do bloco D"
    )
    return out


def transition_guides(
    transformed: pd.DataFrame,
    lifecycle: pd.DataFrame,
    discrimination: pd.DataFrame,
    *,
    selected_signal: str = "nino34_ssta",
    top_guides: int = 3,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Select empirical markers of each adjacent phase transition.

    This is deliberately different from the lag scan.  A correlation between
    X(t-lag) and SSTA(t) inside a phase describes association with that state;
    it is not evidence that X detects the boundary.  Here each event supplies
    a paired before/after phase mean.  Wilcoxon + BH-FDR tests the transition,
    and a diversity-aware top-3 avoids presenting collinear depth levels as
    three independent guides.  These are retrospective markers, not an online
    change-point detector and not causal attribution.
    """

    if top_guides < 1:
        raise ValueError("top_guides must be positive")
    joined = transformed.join(lifecycle[["tipo", "fase", "event_id"]], how="inner")
    active = joined[
        joined["fase"].isin(ENSO_ACTIVE_PHASES) & joined["event_id"].ne("")
    ]
    variables = [name for name in transformed.columns if name != selected_signal]
    event_means = active.groupby(["tipo", "event_id", "fase"])[variables].mean()
    transitions = tuple(zip(ENSO_ACTIVE_PHASES[:-1], ENSO_ACTIVE_PHASES[1:]))
    kendall = discrimination.set_index(["tipo", "variavel"])
    rows: list[dict[str, object]] = []
    for event_type in tuple(active["tipo"].dropna().unique()):
        scoped = event_means.loc[event_type]
        for phase_from, phase_to in transitions:
            for variable in variables:
                pivot = scoped[variable].unstack("fase").reindex(
                    columns=[phase_from, phase_to]
                ).dropna()
                deltas = pivot[phase_to] - pivot[phase_from]
                n_events = int(len(deltas))
                mean_delta = float(deltas.mean()) if n_events else np.nan
                std_delta = float(deltas.std(ddof=1)) if n_events > 1 else np.nan
                effect = (
                    float(mean_delta / std_delta)
                    if np.isfinite(std_delta) and std_delta > 0
                    else np.nan
                )
                if n_events >= 5 and not np.allclose(deltas.to_numpy(), 0):
                    p_value = float(
                        wilcoxon(
                            deltas.to_numpy(),
                            zero_method="wilcox",
                            alternative="two-sided",
                            method="auto",
                        ).pvalue
                    )
                else:
                    p_value = np.nan
                positive_fraction = float((deltas > 0).mean()) if n_events else np.nan
                negative_fraction = float((deltas < 0).mean()) if n_events else np.nan
                consistency = float(max(positive_fraction, negative_fraction) * 100) if n_events else np.nan
                discr = kendall.loc[(event_type, variable)] if (event_type, variable) in kendall.index else None
                rows.append(
                    {
                        "tipo": event_type,
                        "fase_origem": phase_from,
                        "fase_destino": phase_to,
                        "transicao_monitorada": f"{phase_from}_para_{phase_to}",
                        "variavel": variable,
                        "familia": variable_family(variable),
                        "delta_medio_z": mean_delta,
                        "delta_mediano_z": float(deltas.median()) if n_events else np.nan,
                        "efeito_pareado_dz": effect,
                        "consistencia_direcao_pct": consistency,
                        "p_wilcoxon": p_value,
                        "n_eventos_pareados": n_events,
                        "kendall_w_entre_fases": (
                            float(discr["kendall_w_entre_fases"])
                            if discr is not None
                            else np.nan
                        ),
                        "significativo_friedman_fdr": (
                            bool(discr["significativo_friedman_fdr"])
                            if discr is not None
                            else False
                        ),
                    }
                )

    table = pd.DataFrame(rows)
    corrected: list[pd.DataFrame] = []
    for (_event_type, _transition), group in table.groupby(
        ["tipo", "transicao_monitorada"], sort=False
    ):
        group = group.copy()
        rejected, q_values = benjamini_hochberg_fdr(
            group["p_wilcoxon"].to_numpy(), alpha=fdr_alpha
        )
        group["q_transicao_bh"] = q_values
        group["significativo_transicao_fdr"] = rejected
        group["transition_family_size"] = len(group)
        group["transition_fdr_alpha"] = fdr_alpha
        corrected.append(group)
    table = pd.concat(corrected, ignore_index=True)
    table["score_guia"] = (
        pd.to_numeric(table["efeito_pareado_dz"], errors="coerce").abs().fillna(0)
        * pd.to_numeric(table["consistencia_direcao_pct"], errors="coerce").fillna(0)
        / 100.0
        * np.sqrt(
            pd.to_numeric(table["kendall_w_entre_fases"], errors="coerce")
            .clip(lower=0)
            .fillna(0)
        )
    )
    table["rank_transicao"] = np.nan
    table["guia_principal"] = False
    family_caps = {
        "oceano_subsuperficie": 2,
        "atmosfera": 2,
        "oceano_superficie": 1,
        "acoplamento_vento_oceano": 1,
        "outra": 1,
    }
    for (_event_type, _transition), index in table.groupby(
        ["tipo", "transicao_monitorada"], sort=False
    ).groups.items():
        candidates = table.loc[index].sort_values(
            ["significativo_transicao_fdr", "significativo_friedman_fdr", "score_guia"],
            ascending=[False, False, False],
        )
        family_counts: dict[str, int] = {}
        selected: list[int] = []
        for row_index, row in candidates.iterrows():
            family = str(row["familia"])
            if family_counts.get(family, 0) >= family_caps.get(family, 1):
                continue
            selected.append(row_index)
            family_counts[family] = family_counts.get(family, 0) + 1
            if len(selected) == top_guides:
                break
        table.loc[candidates.index, "rank_transicao"] = np.arange(1, len(candidates) + 1)
        table.loc[selected, "guia_principal"] = True
    table["direcao_da_mudanca"] = np.where(
        table["delta_medio_z"] >= 0,
        "aumenta_da_fase_origem_para_destino",
        "diminui_da_fase_origem_para_destino",
    )
    table["tipo_guia"] = "marcador_empirico_retrospectivo_da_transicao"
    table["justificativa"] = (
        "mudanca pareada entre eventos; Wilcoxon com BH-FDR por transicao; "
        "score combina efeito, consistencia e Kendall W; selecao limita redundancia por familia"
    )
    return table.sort_values(
        ["tipo", "fase_destino", "guia_principal", "rank_transicao"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def phase_variable_sets(
    phase_stats: pd.DataFrame,
    discrimination: pd.DataFrame,
    *,
    top_k: int = 5,
    selected_signal: str = "nino34_ssta",
) -> pd.DataFrame:
    """Conjunto de variaveis que melhor descreve cada fase, com justificativa.

    Sensibilidade da fase = |media z entre eventos| / desvio z entre eventos
    (sinal-ruido entre eventos independentes).  O alvo termico permanece como
    referencia na tabela, mas nao pode selecionar a si mesmo como explicador.
    A escolha limita representantes por familia para reduzir redundancia entre
    D20/OHC/WWV/temperaturas em profundidade.
    """

    kendall = discrimination.set_index(["tipo", "variavel"])[
        ["kendall_w_entre_fases", "significativo_friedman_fdr"]
    ]
    table = phase_stats[
        [
            "tipo",
            "fase",
            "variavel",
            "media_z_entre_eventos",
            "desvio_z_entre_eventos",
            "n_eventos_independentes",
        ]
    ].copy()
    table["familia"] = table["variavel"].map(variable_family)
    table["e_alvo_termico"] = table["variavel"].eq(selected_signal)
    deviation = pd.to_numeric(table["desvio_z_entre_eventos"], errors="coerce")
    magnitude = pd.to_numeric(table["media_z_entre_eventos"], errors="coerce").abs()
    table["sensibilidade_fase"] = np.where(
        deviation > 0, magnitude / deviation, np.nan
    )
    table = table.join(
        kendall, on=["tipo", "variavel"]
    )
    table["score_descritor"] = (
        table["sensibilidade_fase"]
        * np.sqrt(
            pd.to_numeric(table["kendall_w_entre_fases"], errors="coerce")
            .clip(lower=0)
        )
    )
    parts: list[pd.DataFrame] = []
    for (_event_type, _phase), group in table.groupby(["tipo", "fase"], sort=False):
        group = group.copy()
        group["rank_na_fase"] = (
            group["score_descritor"].rank(method="first", ascending=False)
        )
        group["integra_conjunto_descritor"] = False
        candidates = group.loc[
            ~group["e_alvo_termico"]
            & group["significativo_friedman_fdr"].fillna(False)
        ].sort_values("score_descritor", ascending=False)
        family_caps = {
            "oceano_subsuperficie": 2,
            "atmosfera": 2,
            "oceano_superficie": 1,
            "acoplamento_vento_oceano": 1,
            "outra": 1,
        }
        selected: list[int] = []
        counts: dict[str, int] = {}
        for row_index, row in candidates.iterrows():
            family = str(row["familia"])
            if counts.get(family, 0) >= family_caps.get(family, 1):
                continue
            selected.append(row_index)
            counts[family] = counts.get(family, 0) + 1
            if len(selected) == top_k:
                break
        group.loc[selected, "integra_conjunto_descritor"] = True
        parts.append(group)
    out = pd.concat(parts, ignore_index=True)
    out["justificativa"] = (
        "score = sensibilidade entre eventos vezes raiz de Kendall W; alvo termico excluido "
        "da selecao; ate 2 representantes subsuperficiais/atmosfericos e 1 das demais familias"
    )
    return out


class Writer:
    def __init__(
        self,
        *,
        output_dir: Path,
        suffix: str,
        run_id: str,
        parameters: dict[str, object],
        enso_type: str | None,
    ) -> None:
        self.output_dir = output_dir
        self.suffix = suffix
        self.run_id = run_id
        self.parameters = parameters
        self.enso_type = enso_type

    def path(self, stem: str) -> Path:
        public_stem = phase3_public_stem(stem, self.enso_type)
        return self.output_dir / f"{public_stem}{self.suffix}.csv"

    def write(
        self,
        frame: pd.DataFrame,
        stem: str,
        *,
        method: str,
        description: str,
        evaluation_mode: str,
        primary_keys: tuple[str, ...] = (),
        allowed_values: dict[str, tuple[str, ...]] | None = None,
        units: dict[str, str] | None = None,
        fdr_family: str | None = None,
        lag_convention: str | None = None,
        extra_inputs: tuple[Path, ...] = (),
    ) -> None:
        public_stem = phase3_public_stem(stem, self.enso_type)
        contract = SemanticTableContract(
            table_id=public_stem,
            phase="F3",
            method=method,
            description=description,
            evaluation_mode=evaluation_mode,
            primary_keys=primary_keys,
            allowed_values=allowed_values or {},
            units=units or {},
            fdr_family=fdr_family,
            lag_convention=lag_convention,
            random_seed=int(self.parameters["random_seed"]),
        )
        output = write_semantic_csv(
            frame,
            self.path(stem),
            contract=contract,
            inputs=(MASTER_PATH, ONI_PATH, *extra_inputs, *CODE_INPUTS),
            parameters={**self.parameters, "legacy_descriptive_stem": stem},
            run_id=self.run_id,
            project_root=ROOT,
        )
        print(f"[tabela] {output.csv_path.relative_to(ROOT)} sha256={output.sha256[:12]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-lag", type=int, default=52)
    parser.add_argument("--min-pairs", type=int, default=12)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-n", type=int, default=1_000)
    parser.add_argument(
        "--bootstrap-top",
        type=int,
        default=5,
        help=(
            "cinco precursores priorizados por tipo/fase apos BH-FDR; "
            "o bootstrap mede estabilidade e nao refaz o screening de variaveis"
        ),
    )
    parser.add_argument("--random-state", type=int, default=20260712)
    parser.add_argument(
        "--min-peak-c",
        type=float,
        default=None,
        help=(
            "magnitude minima do pico |ONI| para o evento entrar na analise; "
            "padrao do protocolo F3Nino: 1.0 (somente El Ninos com anomalia de "
            "pico acima de 1 grau); demais escopos: 0.0"
        ),
    )
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--enso-type",
        choices=("el_nino", "la_nina"),
        default=None,
        help=(
            "restringe toda a caracterizacao retrospectiva a um unico sinal; "
            "sem esta opcao preserva a analise conjunta historica"
        ),
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.quick:
        args.max_lag = min(args.max_lag, 12)
        args.bootstrap_n = min(args.bootstrap_n, 50)
    if args.max_lag < 0 or args.bootstrap_n < 0 or args.bootstrap_top < 1:
        parser.error("lags/bootstrap count cannot be negative and bootstrap-top must be positive")

    event_types = (args.enso_type,) if args.enso_type else ("el_nino", "la_nina")
    scope_directory = {
        "el_nino": "F3Nino",
        "la_nina": "F3Nina",
    }.get(args.enso_type)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
    elif args.quick:
        pilot_name = args.output_suffix.strip("_") or "quick"
        output_dir = STATS / "pilots"
        if scope_directory:
            output_dir = output_dir / scope_directory
        output_dir = output_dir / pilot_name
        args.output_suffix = ""
    elif scope_directory:
        output_dir = STATS / scope_directory
    else:
        output_dir = STATS
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    config = EnsoLifecycleConfig()
    master = load_master()
    signal_column = "nino34_ssta" if "nino34_ssta" in master else master.columns[0]
    precursor_columns, excluded_target_aliases = phase3_precursor_columns(
        master.columns,
        selected_signal=signal_column,
    )
    if not precursor_columns:
        raise ValueError("F3 has no precursor candidate after target-alias exclusion")
    min_peak_c = (
        float(args.min_peak_c)
        if args.min_peak_c is not None
        else (1.0 if args.enso_type == "el_nino" else 0.0)
    )
    if min_peak_c < 0:
        parser.error("--min-peak-c cannot be negative")
    parameters = {
        "max_lag": args.max_lag,
        "min_pairs": args.min_pairs,
        "fdr_alpha": args.fdr_alpha,
        "bootstrap_n": args.bootstrap_n,
        "bootstrap_top": args.bootstrap_top,
        "bootstrap_screening_rule": "top_k_por_tipo_fase_apos_bh_fdr_no_scan_original",
        "bootstrap_interpretation": "estabilidade_dos_precursores_priorizados_nao_novo_screening",
        "random_seed": args.random_state,
        "peak_fraction": 0.90,
        "peak_sensitivity_fractions": [0.80, 0.90, 0.95],
        "quick": args.quick,
        "selected_signal": signal_column,
        "precursor_target_exclusion_policy": PRECURSOR_TARGET_EXCLUSION_POLICY,
        "excluded_target_aliases": list(excluded_target_aliases),
        "n_precursor_candidates": len(precursor_columns),
        "precursor_candidate_columns": list(precursor_columns),
        "enso_type_scope": args.enso_type or "el_nino_and_la_nina",
        "event_types": list(event_types),
        "min_peak_magnitude_c": min_peak_c,
    }
    writer = Writer(
        output_dir=output_dir,
        suffix=args.output_suffix,
        run_id=run_id,
        parameters=parameters,
        enso_type=args.enso_type,
    )
    oni = load_oni()
    events_catalog = detect_enso_events(oni, config=config)
    events_catalog = (
        events_catalog.loc[events_catalog["tipo"].isin(event_types)]
        .copy()
        .reset_index(drop=True)
    )
    if events_catalog.empty:
        raise ValueError(f"F3 has no completed event for ENSO scope {event_types!r}")
    events_catalog["magnitude_minima_analise_c"] = min_peak_c
    events_catalog["elegivel_analise"] = (
        events_catalog["magnitude_pico_c"] >= min_peak_c
    )
    events_catalog["criterio_elegibilidade"] = (
        f"|ONI_pico| >= {min_peak_c:.2f} C"
        if min_peak_c > 0
        else "todos os eventos detectados"
    )
    events = events_catalog.loc[events_catalog["elegivel_analise"]].copy().reset_index(drop=True)
    if events.empty:
        raise ValueError(
            f"F3 has no event with peak magnitude >= {min_peak_c} C for scope {event_types!r}"
        )
    parameters["n_eventos_detectados"] = int(len(events_catalog))
    parameters["n_eventos_elegiveis"] = int(len(events))
    lifecycle = build_enso_lifecycle(events, master.index, config=config)
    lifecycle_table = lifecycle.reset_index()
    event_lifecycle = lifecycle_event_table(lifecycle, events)
    duration = (
        event_lifecycle.groupby(["tipo", "classe", "fase"], as_index=False)
        .agg(
            duracao_media_semanas=("duracao_semanas", "mean"),
            duracao_mediana_semanas=("duracao_semanas", "median"),
            n_eventos=("event_id", "nunique"),
        )
    )
    sensitivity = peak_band_sensitivity(
        oni,
        events,
        fractions=config.peak_sensitivity_fractions,
        canonical_fraction=config.peak_fraction,
    )
    boundary_sensitivity = phase_boundary_sensitivity(
        oni,
        events,
        master.index,
        genesis_windows_weeks=(13, 26, 39),
        peak_fractions=config.peak_sensitivity_fractions,
        canonical_genesis_weeks=config.genesis_weeks,
        canonical_peak_fraction=config.peak_fraction,
    )

    writer.write(
        events_catalog,
        "phase3_events_en_ln",
        method="ONI local simetrico com 5 estacoes consecutivas",
        description=(
            "Catalogo completo de eventos com faixa de pico canonica e flag de "
            "elegibilidade pela magnitude minima do pico"
        ),
        evaluation_mode="diagnostico_retrospectivo",
        primary_keys=("event_id",),
        allowed_values={"tipo": event_types},
        units={
            "oni_pico_c": "degC",
            "magnitude_pico_c": "degC",
            "magnitude_minima_analise_c": "degC",
        },
    )
    writer.write(
        lifecycle_table,
        "phase3_fases_semanais_en_ln",
        method="segmentacao canonica nove estados",
        description="Estado ENSO semanal com origem retrospectiva declarada",
        evaluation_mode="diagnostico_retrospectivo",
        primary_keys=("week_ending_sunday",),
        allowed_values={"estado_enso": ENSO_STATE_ORDER},
    )
    writer.write(
        event_lifecycle,
        "phase3_event_lifecycle_en_ln",
        method="duracao semanal por evento e fase",
        description="Inicio, fim e duracao das quatro fases por evento",
        evaluation_mode="diagnostico_retrospectivo",
        primary_keys=("event_id", "fase"),
        allowed_values={"fase": ENSO_ACTIVE_PHASES, "tipo": event_types},
        units={"duracao_semanas": "week"},
    )
    writer.write(
        duration,
        "phase3_duracao_por_tipo_classe",
        method="resumo entre eventos",
        description="Duracao media e mediana por tipo, classe e fase",
        evaluation_mode="diagnostico_retrospectivo",
        primary_keys=("tipo", "classe", "fase"),
        units={"duracao_media_semanas": "week", "duracao_mediana_semanas": "week"},
    )
    writer.write(
        sensitivity,
        "phase3_peak_band_sensitivity",
        method="faixa relativa ao extremo por evento",
        description="Sensibilidade da faixa de pico a 80/90/95 por cento",
        evaluation_mode="diagnostico_retrospectivo_sensibilidade",
        primary_keys=("event_id", "fracao_faixa_pico"),
        units={"duracao_faixa_pico_meses": "month"},
    )
    writer.write(
        boundary_sensitivity,
        "phase3_phase_boundary_sensitivity",
        method="grade declarada de 13/26/39 semanas de genese x faixa de pico 80/90/95 por cento",
        description=(
            "Sensibilidade das quatro fases a janela pre-onset e a largura relativa "
            "da faixa de pico; onset e fim do evento permanecem fixos"
        ),
        evaluation_mode="diagnostico_retrospectivo_sensibilidade",
        primary_keys=(
            "event_id",
            "fase",
            "janela_genese_semanas",
            "fracao_faixa_pico",
        ),
        allowed_values={"fase": ENSO_ACTIVE_PHASES, "tipo": event_types},
        units={
            "duracao_semanas": "week",
            "duracao_canonica_semanas": "week",
            "delta_duracao_vs_canonica_semanas": "week",
            "janela_genese_semanas": "week",
        },
    )

    equatorial_ssta = load_equatorial_ssta_by_lon()
    tau_column = "tau_x_anom" if "tau_x_anom" in master else "tau_x_anom_nino34_pa"
    hovmoller_peaks = hovmoller_event_peaks(equatorial_ssta, master[tau_column], events)
    writer.write(
        hovmoller_peaks,
        "phase3_hovmoller_picos_eventos",
        method="extremos espaco-temporais da SSTA equatorial e do vento zonal",
        description=(
            "Epicentro do aquecimento (SSTA 2S-2N por longitude) e pico da "
            "anomalia de estresse zonal por evento elegivel"
        ),
        evaluation_mode="diagnostico_retrospectivo",
        primary_keys=("event_id",),
        units={
            "ssta_extremo_c": "degC",
            "tau_x_extremo_pa": "Pa",
            "antecedencia_vento_semanas": "week",
            "epicentro_lon": "degrees_east",
        },
        extra_inputs=(HOVMOLLER_SSTA_PATH,),
    )
    kelvin_pulses = kelvin_sla_pulses(load_ssh_daily_by_lon(), events)
    writer.write(
        kelvin_pulses,
        "phase3_kelvin_pulsos_sla",
        method="pulsos de SLA no Pacifico oeste e chegada estimada a 2.4 m/s",
        description=(
            "Diagnostico de propagacao equatorial tipo Kelvin nas janelas de "
            "evento com cobertura de SSH; leitura qualitativa, sem detector formal"
        ),
        evaluation_mode="diagnostico_retrospectivo",
        primary_keys=("event_id", "pulso_ordinal"),
        units={"sla_oeste_m": "m", "velocidade_assumida_m_s": "m s-1"},
        extra_inputs=(SSH_EVENTS_PATH,),
    )
    composites = ssta_class_composites(master[signal_column], events)
    writer.write(
        composites,
        "phase3_composto_ssta_classe",
        method="composto por evento alinhado ao pico, resumido por classe",
        description=(
            "Trajetoria media e dispersao da SSTA Nino 3.4 por classe de "
            "intensidade dos eventos elegiveis"
        ),
        evaluation_mode="diagnostico_retrospectivo",
        primary_keys=("classe", "semana_rel_pico"),
        units={"ssta_media_c": "degC", "ssta_dp_c": "degC", "semana_rel_pico": "week"},
    )

    transformed, preprocessing = full_sample_diagnostic_transform(
        master,
        config=SeasonalTrendConfig(harmonics=3, remove_trend=True, standardize=True),
        already_anomalous=ALREADY_ANOMALOUS,
    )
    transformed_precursors, transformed_excluded = phase3_precursor_columns(
        transformed.columns,
        selected_signal=signal_column,
    )
    if transformed_precursors != precursor_columns or transformed_excluded != excluded_target_aliases:
        raise ValueError("preprocessing changed the registered precursor/target-alias catalogue")
    writer.write(
        preprocessing,
        "phase3_preprocessing_contract",
        method="harmonicos+tendencia+escala",
        description="Coeficientes e periodo de ajuste por variavel; somente diagnostico full-sample",
        evaluation_mode="diagnostico_retrospectivo_amostra_completa",
        primary_keys=("variavel",),
    )
    phase_stats, discrimination = phase_statistics(
        transformed,
        lifecycle,
        fdr_alpha=args.fdr_alpha,
        event_types=event_types,
    )
    writer.write(
        phase_stats,
        "phase3_fase_stats_variaveis",
        method="media por evento-fase",
        description="Comportamento das 31 variaveis sem pseudorreplicar semanas",
        evaluation_mode="diagnostico_retrospectivo",
        primary_keys=("tipo", "fase", "variavel"),
    )
    writer.write(
        discrimination,
        "phase3_discriminantes_por_periodo",
        method="Friedman pareado por evento e Kendall W",
        description=(
            "Kendall W do Friedman pareado; evidencia confirmatoria somente com "
            "q de Benjamini-Hochberg menor ou igual ao alfa, em familia separada por tipo"
        ),
        evaluation_mode="diagnostico_retrospectivo_inferencial",
        primary_keys=("tipo", "variavel"),
        fdr_family="uma familia por tipo ENSO sobre todas as variaveis",
    )
    influence = percent_influence(phase_stats, discrimination)
    writer.write(
        influence,
        "phase3_influencia_percentual",
        method="participacao relativa de |media z| por fase e de Kendall W por sinal",
        description=(
            "Percentual de influencia de cada variavel na condicao do sinal: "
            "peso descritivo do estado por fase e peso discriminante entre fases"
        ),
        evaluation_mode="diagnostico_retrospectivo",
        primary_keys=("tipo", "fase", "variavel", "metrica"),
        units={"valor_pct": "percent"},
    )
    variable_sets = phase_variable_sets(
        phase_stats,
        discrimination,
        selected_signal=signal_column,
    )
    writer.write(
        variable_sets,
        "phase3_conjuntos_variaveis_fase",
        method="ranking de sensibilidade entre eventos por fase",
        description=(
            "Conjunto descritor de cada fase: top variaveis por sensibilidade "
            "|media z|/desvio z, com familia fisica e Kendall W de apoio"
        ),
        evaluation_mode="diagnostico_retrospectivo",
        primary_keys=("tipo", "fase", "variavel"),
    )
    pca_variance, pca_loadings = phase_pca(transformed, lifecycle)
    writer.write(
        pca_variance,
        "phase3_pca_por_fase",
        method="PCA de medias evento-fase",
        description="Variancia PCA descritiva com evento como unidade",
        evaluation_mode="pca_descritiva_evento_fase",
        primary_keys=("tipo", "fase", "componente"),
    )
    writer.write(
        pca_loadings,
        "phase3_pca_loadings_por_fase",
        method="PCA de medias evento-fase",
        description="Cargas PCA descritivas por tipo e fase",
        evaluation_mode="pca_descritiva_evento_fase",
        primary_keys=("tipo", "fase", "variavel"),
    )

    targets = build_rolling_origin_targets(
        master[signal_column],
        horizons_weeks=(1, 4, 8, 12, 26),
        threshold_c=config.threshold_c,
        lifecycle=lifecycle,
    )
    writer.write(
        targets,
        "phase3_rolling_origin_targets",
        method="origem semanal e horizonte futuro exato",
        description="Alvos operacionais sem fase retrospectiva nas features",
        evaluation_mode="rolling_origin_operacional",
        primary_keys=("origin_time", "horizon_weeks"),
        units={"signal_at_origin_c": "degC", "target_signal_c": "degC", "horizon_weeks": "week"},
    )
    fold_samples = targets[targets["horizon_weeks"].eq(26)].rename(columns={"target_event_id": "event_id"})
    fold_samples = fold_samples[fold_samples["target_signal_c"].notna()].reset_index(drop=True)
    purge = required_purge_weeks(max_feature_lag_weeks=args.max_lag, target_horizon_weeks=26)
    try:
        folds = event_purged_rolling_origin_folds(
            fold_samples,
            purge_weeks=purge,
            min_train_events=3,
        )
        fold_table = folds_audit_table(folds)
    except ValueError as exc:
        fold_table = pd.DataFrame(
            [{
                "fold_id": "nenhum",
                "test_event_id": "",
                "train_rows": 0,
                "test_rows": 0,
                "n_train_events": 0,
                "train_event_ids": "",
                "train_end": pd.NaT,
                "test_start": pd.NaT,
                "test_end": pd.NaT,
                "gap_days": np.nan,
                "purge_weeks": purge,
                "evaluation_mode": "rolling_origin_evento_agrupado_purgado",
                "future_events_in_training": False,
                "test_event_in_training": False,
                "status": f"insuficiente: {exc}",
            }]
        )
    writer.write(
        fold_table,
        "phase3_rolling_origin_folds",
        method="expanding window por evento com embargo",
        description="Fronteiras auditaveis compartilhadas com F5/F7",
        evaluation_mode="rolling_origin_evento_agrupado_purgado",
        primary_keys=("fold_id",),
        units={"purge_weeks": "week", "gap_days": "day"},
    )

    lags = tuple(range(args.max_lag + 1))
    scan = scan_lagged_correlations(
        transformed,
        transformed[signal_column],
        lifecycle,
        target_name=signal_column,
        lags_weeks=lags,
        condition_time="source",
        min_pairs=args.min_pairs,
        fdr_alpha=args.fdr_alpha,
        event_types=event_types,
    )
    writer.write(
        scan,
        "phase3_lag_scan_en_ln_fases",
        method="Pearson, N efetivo AR1 segmentado, BH-FDR e Simes",
        description=(
            "Importancia temporal somente dos precursores fisicos; o alvo e todos "
            "os seus aliases sao excluidos antes do scan"
        ),
        evaluation_mode="diagnostico_retrospectivo_inferencial",
        primary_keys=("tipo", "fase", "variavel", "lag_semanas"),
        fdr_family="tipo x fase x todos os precursores candidatos e lags",
        lag_convention="X(t-lag) versus alvo(t); fase avaliada em t-lag",
        units={"lag_semanas": "week"},
    )
    best = select_best_lags(scan, require_fdr=True)
    writer.write(
        best,
        "phase3_best_lags_fdr",
        method="max abs(r) depois de BH-FDR",
        description=(
            "Um lag auditavel por tipo/fase/precursor quando significativo; alvo e "
            "aliases permanecem fora do ranking"
        ),
        evaluation_mode="diagnostico_retrospectivo_inferencial",
        primary_keys=("tipo", "fase", "variavel"),
        units={"lag_semanas": "week"},
    )

    guides = transition_guides(
        transformed,
        lifecycle,
        discrimination,
        selected_signal=signal_column,
        fdr_alpha=args.fdr_alpha,
    )
    writer.write(
        guides,
        "phase3_guias_transicao_fase",
        method="mudanca pareada entre fases adjacentes, Wilcoxon, BH-FDR e selecao por familia",
        description=(
            "Top-3 marcadores empiricos de cada transicao adjacente; alvo termico "
            "excluido e redundancia por familia limitada"
        ),
        evaluation_mode="diagnostico_retrospectivo_inferencial",
        primary_keys=("tipo", "transicao_monitorada", "variavel"),
        fdr_family="uma familia por tipo x transicao sobre preditores fisicos, excluindo o alvo",
    )

    if args.bootstrap_n:
        summaries = []
        replicates = []
        ranked = best.copy()
        ranked["bootstrap_screening_rank"] = ranked.groupby(
            ["tipo", "fase"],
            sort=False,
        )["r_pearson"].transform(lambda values: values.abs().rank(method="first", ascending=False))
        ranked["bootstrap_screening_top_k"] = args.bootstrap_top
        ranked["bootstrap_screening_rule"] = (
            "top_k_por_tipo_fase_apos_bh_fdr_no_scan_original"
        )
        ranked = ranked.sort_values(["tipo", "fase", "bootstrap_screening_rank"])
        for (event_type, phase), group in ranked.groupby(["tipo", "fase"]):
            for _, selected in group.head(args.bootstrap_top).iterrows():
                variable = str(selected["variavel"])
                result = bootstrap_lag_selection_by_event(
                    transformed[variable],
                    transformed[signal_column],
                    lifecycle,
                    predictor_name=variable,
                    target_name=signal_column,
                    n_precursor_candidates=len(precursor_columns),
                    excluded_target_aliases=excluded_target_aliases,
                    screening_rank=int(selected["bootstrap_screening_rank"]),
                    screening_top_k=args.bootstrap_top,
                    lags_weeks=lags,
                    event_type=event_type,
                    phase=phase,
                    n_boot=args.bootstrap_n,
                    min_pairs=max(8, args.min_pairs // 2),
                    random_state=args.random_state,
                )
                summary = result.summary.copy()
                summary["variavel"] = variable
                summary["n_precursores_fdr_na_familia"] = len(group)
                replica = result.replicates.copy()
                replica["tipo"] = event_type
                replica["fase"] = phase
                replica["variavel"] = variable
                replica["n_precursores_fdr_na_familia"] = len(group)
                summaries.append(summary)
                replicates.append(replica)
        if summaries:
            summary_table = pd.concat(summaries, ignore_index=True)
            replicate_table = pd.concat(replicates, ignore_index=True)
            writer.write(
                summary_table,
                "phase3_lag_event_bootstrap_summary",
                method="bootstrap de eventos repetindo selecao do lag",
                description=(
                    "Estabilidade de lag somente dos cinco precursores priorizados por "
                    "tipo/fase apos BH-FDR no scan original"
                ),
                evaluation_mode="diagnostico_retrospectivo_inferencial",
                primary_keys=("tipo", "fase", "variavel", "lag_semanas"),
                units={"lag_semanas": "week"},
            )
            writer.write(
                replicate_table,
                "phase3_lag_event_bootstrap_replicates",
                method="bootstrap de eventos repetindo selecao do lag",
                description=(
                    "Replicas completas do bootstrap de estabilidade para o top-k "
                    "predeclarado, sem reabrir o screening de variaveis"
                ),
                evaluation_mode="diagnostico_retrospectivo_inferencial",
                primary_keys=("tipo", "fase", "variavel", "bootstrap_replicate"),
                units={"lag_selecionado_semanas": "week"},
            )
        else:
            # Ausencia de candidato apos FDR e um resultado auditavel, nao uma
            # razao para omitir o contrato consumido pelos notebooks scoped.
            writer.write(
                pd.DataFrame(
                    columns=[
                        "tipo",
                        "fase",
                        "variavel",
                        "lag_semanas",
                        "frequencia_selecao",
                    ]
                ),
                "phase3_lag_event_bootstrap_summary",
                method="bootstrap de eventos repetindo selecao do lag",
                description="Nenhum precursor passou o screening FDR para bootstrap",
                evaluation_mode="diagnostico_retrospectivo_inferencial",
                primary_keys=("tipo", "fase", "variavel", "lag_semanas"),
                units={"lag_semanas": "week"},
            )
            writer.write(
                pd.DataFrame(
                    columns=[
                        "tipo",
                        "fase",
                        "variavel",
                        "bootstrap_replicate",
                        "lag_selecionado_semanas",
                    ]
                ),
                "phase3_lag_event_bootstrap_replicates",
                method="bootstrap de eventos repetindo selecao do lag",
                description="Nenhuma replica: nenhum precursor passou o screening FDR",
                evaluation_mode="diagnostico_retrospectivo_inferencial",
                primary_keys=("tipo", "fase", "variavel", "bootstrap_replicate"),
                units={"lag_selecionado_semanas": "week"},
            )

    try:
        output_label = output_dir.relative_to(ROOT)
    except ValueError:
        output_label = output_dir
    print(
        f"F3 pronta: escopo={args.enso_type or 'el_nino+la_nina'}, "
        f"eventos_elegiveis={len(events)}/{len(events_catalog)} "
        f"(|ONI_pico| >= {min_peak_c:.2f} C), variaveis={master.shape[1]}, "
        f"saida={output_label}, run_id={run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
