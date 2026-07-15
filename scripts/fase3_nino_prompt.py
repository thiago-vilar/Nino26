"""Núcleo reprodutível da FASE3-NINO especificada no prompt do projeto.

O módulo é intencionalmente separado da Fase 3 histórica.  A unidade básica é
a semana, a variável-alvo é a anomalia semanal de Niño 3.4 e o recorte dos
episódios é definido pelo percentil 90 dessa própria série.  Os notebooks são
publicadores finos: todos os cálculos e os contratos de saída vivem aqui para
que a execução seja repetível fora do Jupyter.

Nenhuma saída gerada por este módulo deve ser interpretada como causalidade.
"indispensável" significa representante descritivo não redundante no conjunto
observado; a palavra não significa que a variável seja condição necessária da
dinâmica do ENSO.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.data.phase2_master import PHYSICAL_COLUMNS, VARIABLE_SPECS  # noqa: E402


TARGET = "nino34_ssta"
START_DATE = pd.Timestamp("1981-01-01")
END_DATE = pd.Timestamp.today().normalize()
PERCENTILE = 90.0
PEAK_FRACTION = 0.90
GENESIS_WEEKS = 26
CORRELATION_THRESHOLD = 0.85
FIGURE_DPI = int(os.environ.get("FASE3_NINO_FIGURE_DPI", "1400"))
FIGURE_NAMESPACE = "FASE3_NINO"


def _outputs_root() -> Path:
    """Use the requested WSL path when mounted, otherwise the Windows root."""

    requested = Path("/mnt/c/DEV/NINO26")
    if requested.exists():
        return requested
    return ROOT


OUTPUT_ROOT = _outputs_root() / "data" / "processed"
NUMERIC_ROOT = OUTPUT_ROOT / "numeric-tables" / FIGURE_NAMESPACE
FIGURE_ROOT = OUTPUT_ROOT / "figures" / FIGURE_NAMESPACE


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, pd.DataFrame, pd.Series)) else False:
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_time_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    candidates = ["week_ending_sunday", "time", "date", "datetime", "valid_time"]
    time_column = next((column for column in candidates if column in out.columns), None)
    if time_column is None:
        if isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index)
        else:
            raise ValueError("A entrada precisa conter uma coluna temporal reconhecível.")
    else:
        out[time_column] = pd.to_datetime(out[time_column], errors="coerce")
        out = out.dropna(subset=[time_column]).set_index(time_column)
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _build_weekly_from_available_inputs() -> pd.DataFrame | None:
    """Bridge the checked-in daily physical inputs when the F2 master is partial.

    The bridge is used only when the canonical master has an entirely empty
    physical block.  Its provenance is recorded in the validation table so a
    mixed or partial source cannot pass unnoticed.
    """

    physical_path = ROOT / "data/processed/parquet/features/nino34_physical_signal.csv"
    atmosphere_path = ROOT / "data/processed/parquet/features/era5_nino34_daily_cache.parquet"
    if not physical_path.is_file() or not atmosphere_path.is_file():
        return None
    ocean = pd.read_csv(physical_path, parse_dates=["time"]).set_index("time").sort_index()
    ocean_map = {
        "nino34_ssta": "nino34_ssta",
        "d20_m": "d20_nino34_mean_m",
        "tilt_m": "thermocline_tilt_m",
        "tilt_slope": "thermocline_tilt_slope_m_per_degree",
        "ohc_0_100": "ohc_0_100_nino34_j_m2",
        "ohc_0_300": "ohc_0_300_nino34_j_m2",
        "ohc_0_700": "ohc_0_700_nino34_j_m2",
        "ohc_300_700": "ohc_300_700_nino34_j_m2",
        "ssh_m": "ssh_nino34_mean_m",
        "wwv": "wwv_equatorial_pacific_m3",
        "t50m": "temperature_50m_nino34_c",
        "t100m": "temperature_100m_nino34_c",
        "t150m": "temperature_150m_nino34_c",
        "t200m": "temperature_200m_nino34_c",
        "t300m": "temperature_300m_nino34_c",
        "t500m": "temperature_500m_nino34_c",
        "t700m": "temperature_700m_nino34_c",
    }
    ocean_weekly = ocean.rename(columns={source: target for target, source in ocean_map.items() if source in ocean.columns})
    ocean_weekly = ocean_weekly.loc[:, [column for column in PHYSICAL_COLUMNS[:17] if column in ocean_weekly]].resample("W-SUN").mean()
    atmo = pd.read_parquet(atmosphere_path)
    atmo = _normalise_time_index(atmo)
    if "u10" not in atmo or "v10" not in atmo:
        return None
    speed = np.hypot(pd.to_numeric(atmo["u10"], errors="coerce"), pd.to_numeric(atmo["v10"], errors="coerce"))
    atmo["tau_x"] = 1.2 * 1.3e-3 * speed * pd.to_numeric(atmo["u10"], errors="coerce")
    atmo_map = {
        "u10_anom": "u10",
        "v10_anom": "v10",
        "mslp_anom": "mslp",
        "tcwv_anom": "tcwv",
        "slhf_anom": "slhf",
        "sshf_anom": "sshf",
        "ssr_anom": "ssr",
        "str_anom": "str",
        "u850_anom": "u850",
        "u200_anom": "u200",
        "omega850_anom": "omega850",
        "omega500_anom": "omega500",
        "div850_anom": "div850",
        "tau_x_anom": "tau_x",
    }
    atmo_weekly_raw = atmo.rename(columns={source: target for target, source in atmo_map.items() if source in atmo.columns})
    atmo_weekly = atmo_weekly_raw.resample("W-SUN").mean()
    for column in [name for name in PHYSICAL_COLUMNS[17:] if name in atmo_weekly]:
        atmo_weekly[column] = _weekly_climatology_anomaly(atmo_weekly[column])
    bridge = pd.concat([ocean_weekly, atmo_weekly], axis=1).reindex(columns=list(PHYSICAL_COLUMNS))
    return bridge


def load_weekly_master(path: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the 31-variable weekly history from 1981 to present."""

    input_path = Path(path) if path else ROOT / "data/processed/parquet/features/nino34_master_weekly.csv"
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Matriz semanal das 31 variáveis não encontrada: {input_path}. "
            "Execute a ingestão/Fase 2 antes da FASE3-NINO."
        )
    frame = _normalise_time_index(pd.read_csv(input_path))
    missing = [column for column in PHYSICAL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"A matriz semanal não contém as 31 variáveis. Ausentes: {missing}")
    extra = [column for column in frame.columns if column not in PHYSICAL_COLUMNS and column != "ocean_source_code"]
    frame = frame.loc[:, [*PHYSICAL_COLUMNS, *(["ocean_source_code"] if "ocean_source_code" in frame else [])]]
    frame = frame.loc[(frame.index >= START_DATE) & (frame.index <= END_DATE)]
    if frame.empty:
        raise ValueError("A matriz semanal não tem observações dentro de 1981–presente.")
    numeric = frame.loc[:, PHYSICAL_COLUMNS].apply(pd.to_numeric, errors="coerce")
    source_label = "master_f2"
    ocean_empty = numeric.loc[:, list(PHYSICAL_COLUMNS[:17])].notna().sum().eq(0).all()
    if ocean_empty:
        bridge = _build_weekly_from_available_inputs()
        if bridge is not None:
            numeric = bridge.loc[(bridge.index >= START_DATE) & (bridge.index <= END_DATE)].reindex(columns=list(PHYSICAL_COLUMNS))
            source_label = "ponte_explicitada_physical_signal_plus_era5_daily"
    if numeric.notna().sum().eq(0).any():
        empty = numeric.columns[numeric.notna().sum().eq(0)].tolist()
        raise ValueError(f"Variáveis sem observações numéricas: {empty}")
    # A FASE3-NINO trabalha numa grade semanal W-SUN.  Para dados já semanais
    # a operação é idempotente; para dados diários ela faz a organização pedida.
    numeric = numeric.resample("W-SUN").mean()
    numeric = numeric.loc[:, list(PHYSICAL_COLUMNS)]
    validation = pd.DataFrame(
        {
            "variavel": list(PHYSICAL_COLUMNS),
            "unidade": [spec.units for spec in VARIABLE_SPECS],
            "fonte": [spec.source for spec in VARIABLE_SPECS],
            "representacao_entrada": [spec.representation_raw for spec in VARIABLE_SPECS],
            "inicio": [numeric[column].first_valid_index() for column in PHYSICAL_COLUMNS],
            "fim": [numeric[column].last_valid_index() for column in PHYSICAL_COLUMNS],
            "n_semanas_validas": [int(numeric[column].notna().sum()) for column in PHYSICAL_COLUMNS],
            "cobertura_pct": [float(100 * numeric[column].notna().mean()) for column in PHYSICAL_COLUMNS],
            "n_lacunas": [int(numeric[column].isna().sum()) for column in PHYSICAL_COLUMNS],
            "colunas_extras_ignoradas": ["|".join(extra)] * len(PHYSICAL_COLUMNS),
            "origem_serie": [source_label] * len(PHYSICAL_COLUMNS),
        }
    )
    return numeric, validation


def _weekly_climatology_anomaly(series: pd.Series) -> pd.Series:
    iso_week = series.index.isocalendar().week.astype(int)
    baseline_mask = (series.index >= pd.Timestamp("1991-01-01")) & (series.index <= pd.Timestamp("2020-12-31"))
    baseline = series.loc[baseline_mask]
    if baseline.notna().sum() < 104:
        baseline = series
    climatology = baseline.groupby(iso_week[baseline_mask]).mean()
    mapped = pd.Series(iso_week, index=series.index).map(climatology)
    return series - mapped.to_numpy()


def transform_weekly(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create explicit anomaly and standardized representations for all variables."""

    anomaly = pd.DataFrame(index=master.index)
    zscore = pd.DataFrame(index=master.index)
    rows: list[dict[str, Any]] = []
    anomaly_names = {TARGET, *[column for column in PHYSICAL_COLUMNS if column.endswith("_anom")]}
    for spec in VARIABLE_SPECS:
        series = pd.to_numeric(master[spec.name], errors="coerce").astype(float)
        if spec.name in anomaly_names:
            transformed = series.copy()
            method = "anomalia_de_entrada_preservada"
        else:
            transformed = _weekly_climatology_anomaly(series)
            method = "anomalia_sazonal_semanal_1991_2020_ou_toda_a_serie"
        finite = transformed.dropna()
        mean = float(finite.mean()) if len(finite) else np.nan
        std = float(finite.std(ddof=0)) if len(finite) else np.nan
        z = (transformed - mean) / std if np.isfinite(std) and std > 0 else transformed * np.nan
        anomaly[spec.name] = transformed
        zscore[spec.name] = z
        rows.append(
            {
                "variavel": spec.name,
                "fonte": spec.source,
                "unidade_entrada": spec.units,
                "metodo_transformacao": method,
                "media_anomalia": mean,
                "desvio_anomalia": std,
                "n_valido": int(finite.size),
                "alvo_p90_aplicado": bool(spec.name == TARGET),
                "uso_multivariado": "zscore_da_anomalia_semanal",
            }
        )
    return anomaly, zscore, pd.DataFrame(rows)


def _run_id() -> str:
    return os.environ.get("FASE3_NINO_RUN_ID", f"FASE3_NINO_{pd.Timestamp.utcnow():%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}")


def write_table(table: pd.DataFrame, notebook: str, name: str, *, run_id: str | None = None) -> Path:
    NUMERIC_ROOT.mkdir(parents=True, exist_ok=True)
    table_path = NUMERIC_ROOT / f"{notebook}_{name}.csv"
    out = table.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.strftime("%Y-%m-%d")
    out.to_csv(table_path, index=False, encoding="utf-8")
    metadata = {
        "run_id": run_id or _run_id(),
        "notebook": notebook,
        "table": name,
        "rows": int(len(out)),
        "columns": list(out.columns),
        "path": str(table_path.resolve()),
        "sha256": _sha256(table_path),
        "python": sys.version,
        "platform": platform.platform(),
        "method": "FASE3-NINO prompt specification",
    }
    table_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_value), encoding="utf-8")
    return table_path


def write_figure(fig: plt.Figure, notebook: str, name: str, *, table_path: Path | None = None, run_id: str | None = None) -> Path:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURE_ROOT / f"{notebook}_{name}.png"
    fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    metadata = {
        "run_id": run_id or _run_id(),
        "notebook": notebook,
        "figure": name,
        "dpi": FIGURE_DPI,
        "path": str(figure_path.resolve()),
        "sha256": _sha256(figure_path),
        "table_source": str(table_path.resolve()) if table_path else None,
    }
    figure_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_value), encoding="utf-8")
    return figure_path


def _phase_name(value: str) -> str:
    return {"genese": "gênese", "crescimento": "crescimento", "pico": "faixa de pico", "decaimento": "decaimento"}.get(value, value)


def classify_p90(master: pd.DataFrame, *, percentile: float = PERCENTILE, peak_fraction: float = PEAK_FRACTION, genesis_weeks: int = GENESIS_WEEKS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Classify contiguous weekly P90 episodes and label their four phases."""

    target = pd.to_numeric(master[TARGET], errors="coerce")
    threshold = float(np.nanpercentile(target.dropna().to_numpy(), percentile))
    exceedance = target.ge(threshold)
    groups = exceedance.ne(exceedance.shift(fill_value=False)).cumsum()
    events: list[dict[str, Any]] = []
    lifecycle_rows: list[dict[str, Any]] = []
    event_number = 0
    for group_id, grouped_target in target.groupby(groups):
        mask = exceedance.loc[grouped_target.index]
        if not bool(mask.iloc[0]):
            continue
        dates = grouped_target.index[mask.to_numpy()]
        values = target.loc[dates]
        if len(dates) == 0:
            continue
        event_number += 1
        event_id = f"ELNINO_P90_{event_number:03d}"
        peak_time = values.idxmax()
        peak_value = float(values.max())
        band = values.ge(peak_fraction * peak_value)
        peak_dates = values.index[band.to_numpy()]
        peak_start, peak_end = peak_dates.min(), peak_dates.max()
        onset, end = dates.min(), dates.max()
        genesis_start = onset - pd.Timedelta(weeks=genesis_weeks)
        event = {
            "event_id": event_id,
            "tipo": "el_nino",
            "percentil_classificador": percentile,
            "limiar_p90_c": threshold,
            "onset_p90": onset,
            "fim_p90": end,
            "duracao_p90_semanas": int(len(dates)),
            "pico_time": peak_time,
            "pico_anomalia_c": peak_value,
            "inicio_faixa_pico": peak_start,
            "fim_faixa_pico": peak_end,
            "duracao_faixa_pico_semanas": int(len(peak_dates)),
            "genese_inicio": genesis_start,
            "genese_fim": onset - pd.Timedelta(weeks=1),
            "crescimento_inicio": onset,
            "crescimento_fim": peak_start - pd.Timedelta(weeks=1),
            "decaimento_inicio": peak_end + pd.Timedelta(weeks=1),
            "decaimento_fim": end,
            "quatro_fases_com_observacoes": bool(len(dates) >= 4),
            "regra_evento": "runs_contiguos_semanais_com_nino34_ssta_maior_ou_igual_a_P90",
            "regra_pico": f"nino34_ssta >= {peak_fraction:.2f} * pico_do_evento",
            "regra_genese": f"{genesis_weeks} semanas anteriores_ao_onset_P90",
            "rotulo": "retrospectivo; não disponível no tempo de origem",
        }
        events.append(event)
        phase_ranges = {
            "genese": (genesis_start, onset - pd.Timedelta(weeks=1)),
            "crescimento": (onset, peak_start - pd.Timedelta(weeks=1)),
            "pico": (peak_start, peak_end),
            "decaimento": (peak_end + pd.Timedelta(weeks=1), end),
        }
        for phase, (start, finish) in phase_ranges.items():
            if finish < start:
                continue
            phase_dates = pd.date_range(start, finish, freq="W-SUN")
            for relative_week, when in enumerate(phase_dates, start=-genesis_weeks):
                lifecycle_rows.append(
                    {
                        "event_id": event_id,
                        "tipo": "el_nino",
                        "time": when,
                        "fase": phase,
                        "fase_pt": _phase_name(phase),
                        "semana_relativa_ao_onset": int((when - onset).days // 7),
                        "semana_relativa_ao_inicio_evento": int((when - genesis_start).days // 7),
                        "rotulo_disponivel_na_origem": False,
                        "modo_rotulo": "retrospectivo",
                    }
                )
    events_frame = pd.DataFrame(events)
    lifecycle = pd.DataFrame(lifecycle_rows)
    if not events_frame.empty:
        events_frame = events_frame.sort_values("onset_p90").reset_index(drop=True)
    if not lifecycle.empty:
        lifecycle = lifecycle.sort_values(["event_id", "time"]).reset_index(drop=True)
    threshold_table = pd.DataFrame(
        [{
            "alvo": TARGET,
            "percentil": percentile,
            "limiar_p90_c": threshold,
            "n_semanas_validas": int(target.notna().sum()),
            "n_semanas_acima_p90": int(exceedance.sum()),
            "n_eventos_p90": int(len(events_frame)),
            "periodo_inicio": target.first_valid_index(),
            "periodo_fim": target.last_valid_index(),
        }]
    )
    return events_frame, lifecycle, threshold_table


def cycle_composite(anomaly: pd.DataFrame, zscore: pd.DataFrame, lifecycle: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if lifecycle.empty:
        return pd.DataFrame(), pd.DataFrame()
    data = lifecycle.merge(anomaly.reset_index().rename(columns={anomaly.index.name or "index": "time"}), on="time", how="left")
    zdata = lifecycle.merge(zscore.reset_index().rename(columns={zscore.index.name or "index": "time"}), on="time", how="left", suffixes=("", "_z"))
    rows: list[dict[str, Any]] = []
    raw_long = data.melt(id_vars=["event_id", "time", "semana_relativa_ao_onset", "fase"], value_vars=list(PHYSICAL_COLUMNS), var_name="variavel", value_name="valor")
    z_long = zdata.melt(id_vars=["event_id", "time", "semana_relativa_ao_onset", "fase"], value_vars=list(PHYSICAL_COLUMNS), var_name="variavel", value_name="z")
    for (relative_week, variable), group in raw_long.groupby(["semana_relativa_ao_onset", "variavel"]):
        event_means = group.groupby("event_id", as_index=False)["valor"].mean()
        z_group = z_long.loc[(z_long["semana_relativa_ao_onset"].eq(relative_week)) & (z_long["variavel"].eq(variable))]
        z_event_means = z_group.groupby("event_id", as_index=False)["z"].mean()
        rows.append({
            "semana_relativa_ao_onset": int(relative_week),
            "variavel": variable,
            "media_entre_eventos": float(event_means["valor"].mean()) if len(event_means) else np.nan,
            "mediana_entre_eventos": float(event_means["valor"].median()) if len(event_means) else np.nan,
            "q25_entre_eventos": float(event_means["valor"].quantile(.25)) if len(event_means) else np.nan,
            "q75_entre_eventos": float(event_means["valor"].quantile(.75)) if len(event_means) else np.nan,
            "media_z_entre_eventos": float(z_event_means["z"].mean()) if len(z_event_means) else np.nan,
            "q25_z_entre_eventos": float(z_event_means["z"].quantile(.25)) if len(z_event_means) else np.nan,
            "q75_z_entre_eventos": float(z_event_means["z"].quantile(.75)) if len(z_event_means) else np.nan,
            "n_eventos_independentes": int(event_means["event_id"].nunique()),
            "unidade": next(spec.units for spec in VARIABLE_SPECS if spec.name == variable),
        })
    composite = pd.DataFrame(rows).sort_values(["variavel", "semana_relativa_ao_onset"]).reset_index(drop=True)
    phase_rows: list[dict[str, Any]] = []
    long = zdata.melt(id_vars=["event_id", "time", "fase", "fase_pt"], value_vars=list(PHYSICAL_COLUMNS), var_name="variavel", value_name="z")
    for (phase, variable), group in long.groupby(["fase", "variavel"]):
        event_means = group.groupby("event_id")["z"].mean().dropna()
        phase_rows.append({
            "fase": phase,
            "fase_pt": _phase_name(phase),
            "variavel": variable,
            "media_z_entre_eventos": float(event_means.mean()) if len(event_means) else np.nan,
            "mediana_z_entre_eventos": float(event_means.median()) if len(event_means) else np.nan,
            "desvio_z_entre_eventos": float(event_means.std(ddof=1)) if len(event_means) > 1 else np.nan,
            "n_eventos_independentes": int(event_means.size),
        })
    return composite, pd.DataFrame(phase_rows)


def _safe_slope(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) < 3:
        return np.nan
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values.to_numpy(dtype=float), 1)[0])


def phase_statistics(anomaly: pd.DataFrame, zscore: pd.DataFrame, lifecycle: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if lifecycle.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = lifecycle.merge(anomaly.reset_index().rename(columns={anomaly.index.name or "index": "time"}), on="time", how="left")
    zmerged = lifecycle.merge(zscore.reset_index().rename(columns={zscore.index.name or "index": "time"}), on="time", how="left", suffixes=("", "_z"))
    rows: list[dict[str, Any]] = []
    for (event_id, phase, variable), group in merged.melt(id_vars=["event_id", "fase", "time"], value_vars=list(PHYSICAL_COLUMNS), var_name="variavel", value_name="valor").groupby(["event_id", "fase", "variavel"]):
        values = group["valor"].dropna()
        rows.append({
            "event_id": event_id,
            "fase": phase,
            "fase_pt": _phase_name(phase),
            "variavel": variable,
            "media": float(values.mean()) if len(values) else np.nan,
            "mediana": float(values.median()) if len(values) else np.nan,
            "desvio": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
            "p25": float(values.quantile(.25)) if len(values) else np.nan,
            "p75": float(values.quantile(.75)) if len(values) else np.nan,
            "inclinacao_por_semana": _safe_slope(values),
            "n_semanas_validas": int(values.size),
            "unidade": next(spec.units for spec in VARIABLE_SPECS if spec.name == variable),
        })
    stats = pd.DataFrame(rows)
    z_rows: list[dict[str, Any]] = []
    for (event_id, phase, variable), group in zmerged.melt(id_vars=["event_id", "fase", "time"], value_vars=list(PHYSICAL_COLUMNS), var_name="variavel", value_name="z").groupby(["event_id", "fase", "variavel"]):
        values = group["z"].dropna()
        z_rows.append({"event_id": event_id, "fase": phase, "variavel": variable, "media_z": float(values.mean()) if len(values) else np.nan})
    stats = stats.merge(pd.DataFrame(z_rows), on=["event_id", "fase", "variavel"], how="left", validate="one_to_one")
    means = stats.pivot_table(index=["event_id", "variavel"], columns="fase", values="media_z")
    rationale: list[dict[str, Any]] = []
    for variable in PHYSICAL_COLUMNS:
        for phase in ("genese", "crescimento", "pico", "decaimento"):
            values = means.loc[means.index.get_level_values("variavel") == variable] if variable in means.index.get_level_values("variavel") else pd.DataFrame()
            phase_values = values[phase].dropna() if not values.empty and phase in values else pd.Series(dtype=float)
            baseline = values["genese"].dropna() if not values.empty and "genese" in values else pd.Series(dtype=float)
            delta = float((phase_values - baseline.reindex(phase_values.index)).mean()) if len(phase_values) and len(baseline) else np.nan
            sign = "aumenta" if delta > 0.15 else "diminui" if delta < -0.15 else "permanece_sem_mudanca_clara"
            spec = next(spec for spec in VARIABLE_SPECS if spec.name == variable)
            rationale.append({
                "fase": phase,
                "fase_pt": _phase_name(phase),
                "variavel": variable,
                "familia_fisica": "oceano" if variable in PHYSICAL_COLUMNS[:17] else "atmosfera_vento",
                "media_z_vs_genese": delta,
                "direcao_observada": sign,
                "sinal_fisico_esperado": spec.positive,
                "justificativa_operacional": f"Na {_phase_name(phase)}, {variable} {sign} em relação à gênese; isto é evidência descritiva, não prova causal.",
            })
    return stats, pd.DataFrame(rationale)


def _union_find_clusters(correlation: pd.DataFrame, threshold: float) -> dict[str, int]:
    variables = list(correlation.columns)
    parent = {variable: variable for variable in variables}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for i, left in enumerate(variables):
        for right in variables[i + 1 :]:
            value = correlation.loc[left, right]
            if np.isfinite(value) and abs(float(value)) >= threshold:
                union(left, right)
    roots: dict[str, int] = {}
    labels: dict[str, int] = {}
    for variable in variables:
        root = find(variable)
        if root not in roots:
            roots[root] = len(roots) + 1
        labels[variable] = roots[root]
    return labels


def reduce_variables(zscore: pd.DataFrame, phase_stats: pd.DataFrame, *, threshold: float = CORRELATION_THRESHOLD) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce collinear predictors, retaining one data-supported representative per cluster/phase."""

    predictors = [column for column in PHYSICAL_COLUMNS if column != TARGET]
    value_column = "media_z" if "media_z" in phase_stats.columns else "media"
    phase_event = phase_stats.loc[phase_stats["variavel"].isin(predictors)].pivot_table(index=["event_id", "fase"], columns="variavel", values=value_column)
    correlation = phase_event.corr(min_periods=3).reindex(index=predictors, columns=predictors)
    clusters = _union_find_clusters(correlation, threshold) if not correlation.empty else {}
    score_rows: list[dict[str, Any]] = []
    for variable in predictors:
        subset = phase_stats.loc[phase_stats["variavel"].eq(variable)]
        phase_score = subset.groupby("fase")[value_column].apply(lambda values: float(np.nanmean(np.abs(values))))
        score = float(phase_score.mean()) if len(phase_score) else np.nan
        family = "oceano_subsuperficie" if variable in PHYSICAL_COLUMNS[1:17] else "vento_atmosfera"
        score_rows.append({
            "variavel": variable,
            "familia_fisica": family,
            "cluster_correlacao": int(clusters.get(variable, -1)),
            "score_descritivo": score,
            "correlacao_threshold": threshold,
            "representacao_entrada": next(spec.representation_raw for spec in VARIABLE_SPECS if spec.name == variable),
        })
    selection = pd.DataFrame(score_rows)
    selected: list[dict[str, Any]] = []
    for phase in ("genese", "crescimento", "pico", "decaimento"):
        phase_subset = phase_stats.loc[(phase_stats["fase"].eq(phase)) & (phase_stats["variavel"].isin(predictors))]
        for cluster_id, cluster_group in selection.groupby("cluster_correlacao"):
            names = cluster_group["variavel"].tolist()
            candidates = phase_subset.loc[phase_subset["variavel"].isin(names)].copy()
            if candidates.empty:
                continue
            candidates["abs_media"] = candidates[value_column].abs()
            winner = candidates.sort_values(["abs_media", "variavel"], ascending=[False, True]).iloc[0]
            selected.append({"fase": phase, "variavel": winner["variavel"], "cluster_correlacao": int(cluster_id), "score_fase": float(winner["abs_media"])})
    winners = pd.DataFrame(selected)
    winner_names = set(winners["variavel"]) if not winners.empty else set()
    selection["representante_na_reducao"] = selection["variavel"].isin(winner_names)
    selection["indispensavel_descritivo"] = selection["representante_na_reducao"]
    selection["decisao"] = np.where(selection["representante_na_reducao"], "manter_representante", "reduzir_por_redundancia_ou_menor_sinal")
    selection["justificativa"] = np.where(selection["representante_na_reducao"], "maior |média| dentro do cluster em pelo menos uma fase", "substituída pelo representante do cluster")
    return selection.sort_values(["representante_na_reducao", "score_descritivo"], ascending=[False, False]).reset_index(drop=True), winners


def pca_by_phase(phase_stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictors = [column for column in PHYSICAL_COLUMNS if column != TARGET]
    variance_rows: list[dict[str, Any]] = []
    loading_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for phase in ("genese", "crescimento", "pico", "decaimento"):
        matrix = phase_stats.loc[phase_stats["fase"].eq(phase)].pivot(index="event_id", columns="variavel", values="media").reindex(columns=predictors)
        matrix = matrix.dropna(axis=1, how="all").dropna(axis=0, thresh=max(2, int(len(predictors) * .5)))
        if len(matrix) < 2 or matrix.shape[1] < 2:
            continue
        matrix = matrix.fillna(matrix.median())
        scaled = StandardScaler().fit_transform(matrix)
        n_components = min(matrix.shape[0], matrix.shape[1])
        model = PCA(n_components=n_components, random_state=0).fit(scaled)
        scores = model.transform(scaled)
        for component, (ratio, cumulative) in enumerate(zip(model.explained_variance_ratio_, np.cumsum(model.explained_variance_ratio_)), start=1):
            variance_rows.append({"fase": phase, "fase_pt": _phase_name(phase), "componente": component, "variancia_explicada_pct": float(100 * ratio), "variancia_acumulada_pct": float(100 * cumulative), "n_eventos": int(len(matrix)), "n_variaveis": int(matrix.shape[1])})
        for component in range(n_components):
            for variable, loading in zip(matrix.columns, model.components_[component]):
                loading_rows.append({"fase": phase, "componente": component + 1, "variavel": variable, "loading": float(loading), "abs_loading": float(abs(loading))})
        for event_id, values in zip(matrix.index, scores):
            for component, score in enumerate(values, start=1):
                score_rows.append({"fase": phase, "event_id": event_id, "componente": component, "score": float(score)})
    return pd.DataFrame(variance_rows), pd.DataFrame(loading_rows), pd.DataFrame(score_rows)


def _read_optional_spatial(path: Path | str | None, kind: str) -> pd.DataFrame | None:
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.extend([
        ROOT / f"data/processed/parquet/features/equatorial_pacific_{kind}_weekly_by_lon.parquet",
        ROOT / f"data/processed/parquet/features/equatorial_pacific_{kind}_weekly.parquet",
        ROOT / f"data/processed/parquet/features/{kind}_weekly_grid.parquet",
    ])
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            frame = pd.read_parquet(candidate)
            return _normalise_time_index(frame)
        except Exception:
            continue
    return None


def _wide_long_spatial(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    data = frame.copy()
    data.index.name = "time"
    time_index = data.index
    if {"longitude", "lon"}.intersection(data.columns) and {"latitude", "lat"}.intersection(data.columns):
        lon_col = "longitude" if "longitude" in data.columns else "lon"
        lat_col = "latitude" if "latitude" in data.columns else "lat"
        value_col = next((column for column in data.columns if column not in {lon_col, lat_col}), None)
        if value_col:
            long = data.reset_index().rename(columns={lon_col: "longitude", lat_col: "latitude", value_col: value_name})
            return long[["time", "longitude", "latitude", value_name]]
    rows: list[dict[str, Any]] = []
    for column in data.columns:
        if column in {"ocean_source_code"}:
            continue
        try:
            numeric_lon = float(str(column).replace("E", "").replace("W", ""))
        except ValueError:
            continue
        longitude = numeric_lon if "W" not in str(column).upper() else -numeric_lon
        for when, value in data[column].items():
            rows.append({"time": when, "longitude": longitude, "latitude": 0.0, value_name: value})
    return pd.DataFrame(rows)


def spatial_ssta_composite(lifecycle: pd.DataFrame, *, path: Path | str | None = None) -> tuple[pd.DataFrame, str]:
    frame = _read_optional_spatial(path, "ssta")
    if frame is None:
        return pd.DataFrame([{ "status": "entrada espacial SSTA ausente; composto não calculado" }]), "ausente"
    long = _wide_long_spatial(frame, "ssta")
    if long.empty:
        return pd.DataFrame([{ "status": "entrada SSTA sem colunas de longitude reconhecíveis" }]), "invalida"
    base = lifecycle[["event_id", "time", "semana_relativa_ao_onset"]].merge(long, on="time", how="inner")
    composite = base.groupby(["semana_relativa_ao_onset", "longitude", "latitude"], as_index=False).agg(
        ssta_media=("ssta", "mean"), ssta_mediana=("ssta", "median"), n_eventos=("event_id", "nunique")
    )
    return composite, "disponivel"


def kelvin_wind_diagnostics(anomaly: pd.DataFrame, lifecycle: pd.DataFrame, events: pd.DataFrame, *, ssta_path: Path | str | None = None, wind_path: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    ssta = _read_optional_spatial(ssta_path, "ssta")
    wind = _read_optional_spatial(wind_path, "tau_x")
    rows: list[dict[str, Any]] = []
    if ssta is not None:
        long_ssta = _wide_long_spatial(ssta, "ssta")
        if not long_ssta.empty:
            ssta_event = lifecycle[["event_id", "time", "semana_relativa_ao_onset"]].merge(long_ssta, on="time", how="inner")
            peaks = ssta_event.sort_values("ssta", ascending=False).groupby("event_id").first().reset_index()
            for row in peaks.itertuples(index=False):
                rows.append({"event_id": row.event_id, "tipo_pico": "SSTA_equatorial", "time": row.time, "semana_relativa_ao_onset": row.semana_relativa_ao_onset, "longitude": row.longitude, "valor": row.ssta})
    if wind is not None:
        long_wind = _wide_long_spatial(wind, "tau_x")
        if not long_wind.empty:
            wind_event = lifecycle[["event_id", "time", "semana_relativa_ao_onset"]].merge(long_wind, on="time", how="inner")
            peaks = wind_event.sort_values("tau_x", ascending=False).groupby("event_id").first().reset_index()
            for row in peaks.itertuples(index=False):
                rows.append({"event_id": row.event_id, "tipo_pico": "vento_zonal_oeste_tau_x", "time": row.time, "semana_relativa_ao_onset": row.semana_relativa_ao_onset, "longitude": row.longitude, "valor": row.tau_x})
    if not rows:
        # Fallback is deliberately labelled as temporal association, not a
        # Kelvin-wave detection, when the longitude-resolved inputs are absent.
        merged = lifecycle.merge(anomaly[[TARGET, "tau_x_anom"]].reset_index().rename(columns={anomaly.index.name or "index": "time"}), on="time", how="left")
        for event_id, group in merged.groupby("event_id"):
            row = group.loc[group[TARGET].idxmax()]
            rows.append({"event_id": event_id, "tipo_pico": "SSTA_Nino34_temporal_fallback", "time": row["time"], "semana_relativa_ao_onset": row["semana_relativa_ao_onset"], "longitude": np.nan, "valor": row[TARGET]})
        status = "fallback_temporal_sem_propagacao_espacial"
    elif ssta is not None and wind is not None:
        status = "espacial_ssta_e_vento_longitudinal_disponiveis"
    elif ssta is not None:
        status = "espacial_ssta_disponivel_vento_longitudinal_ausente"
    elif wind is not None:
        status = "vento_longitudinal_disponivel_ssta_espacial_ausente"
    else:
        status = "fallback_temporal_sem_propagacao_espacial"
    coupling_rows: list[dict[str, Any]] = []
    base = anomaly[[TARGET, "tau_x_anom"]].copy()
    for phase in ("genese", "crescimento", "pico", "decaimento"):
        phase_times = lifecycle.loc[lifecycle["fase"].eq(phase), "time"]
        subset = base.reindex(phase_times).dropna()
        coupling_rows.append({"fase": phase, "fase_pt": _phase_name(phase), "correlacao_spearman_ssta_tau_x": float(subset[TARGET].corr(subset["tau_x_anom"], method="spearman")) if len(subset) >= 3 else np.nan, "n_semanas": int(len(subset)), "interpretacao": "associação temporal; não prova de acoplamento causal"})
    return pd.DataFrame(rows), pd.DataFrame(coupling_rows), status


def phase_boundary_sensitivity(master: pd.DataFrame, *, genesis_windows: Sequence[int] = (13, 26, 39), peak_fractions: Sequence[float] = (.80, .90, .95)) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for genesis in genesis_windows:
        for peak in peak_fractions:
            events, lifecycle, threshold = classify_p90(master, peak_fraction=peak, genesis_weeks=genesis)
            for row in events.itertuples(index=False):
                rows.append({"genesis_weeks": genesis, "peak_fraction": peak, "event_id": row.event_id, "duracao_genese": genesis, "duracao_p90": row.duracao_p90_semanas, "duracao_pico": row.duracao_faixa_pico_semanas, "pico_c": row.pico_anomalia_c, "configuracao_canonica": bool(genesis == GENESIS_WEEKS and np.isclose(peak, PEAK_FRACTION))})
    return pd.DataFrame(rows)


def _plot_series(master: pd.DataFrame, anomaly: pd.DataFrame, notebook: str, *, run_id: str, table_path: Path | None = None) -> list[Path]:
    figures: list[Path] = []
    fig, axes = plt.subplots(8, 4, figsize=(6.5, 9.5), sharex=True)
    axes = axes.ravel()
    for axis, variable in zip(axes, PHYSICAL_COLUMNS):
        axis.plot(anomaly.index, anomaly[variable], color="#1f4e79", linewidth=.75)
        axis.axhline(0, color="black", linewidth=.35)
        axis.set_title(variable, fontsize=9)
        axis.grid(alpha=.2)
    for axis in axes[len(PHYSICAL_COLUMNS):]:
        axis.axis("off")
    fig.suptitle("FASE3-NINO — séries semanais transformadas das 31 variáveis (1981–presente)", fontsize=18)
    fig.tight_layout()
    figures.append(write_figure(fig, notebook, "series_31_variaveis", table_path=table_path, run_id=run_id))
    plt.close(fig)
    return figures


def _plot_p90(master: pd.DataFrame, events: pd.DataFrame, threshold: pd.DataFrame, notebook: str, *, run_id: str, table_path: Path | None = None) -> list[Path]:
    fig, axis = plt.subplots(figsize=(10, 4.5))
    target = master[TARGET]
    axis.plot(target.index, target, color="#8c2d04", linewidth=1.0, label="Niño 3.4 SSTA semanal")
    cut = float(threshold.iloc[0]["limiar_p90_c"])
    axis.axhline(cut, color="#2166ac", linestyle="--", linewidth=1.2, label=f"P90 = {cut:.3f} °C")
    for row in events.itertuples(index=False):
        axis.axvspan(row.onset_p90, row.fim_p90, color="#fdae61", alpha=.22)
        axis.axvline(row.pico_time, color="#d73027", alpha=.35, linewidth=.7)
    axis.set_title("Classificação de episódios por P90 da anomalia semanal de Niño 3.4")
    axis.set_ylabel("anomalia (°C)")
    axis.legend(loc="upper left")
    axis.grid(alpha=.2)
    fig.tight_layout()
    path = write_figure(fig, notebook, "classificacao_p90", table_path=table_path, run_id=run_id)
    plt.close(fig)
    return [path]


def _plot_cycle(composite: pd.DataFrame, notebook: str, *, run_id: str, table_path: Path | None = None) -> list[Path]:
    value_column = "media_z_entre_eventos" if "media_z_entre_eventos" in composite.columns else "media_entre_eventos"
    pivot = composite.pivot(index="variavel", columns="semana_relativa_ao_onset", values=value_column)
    fig, axis = plt.subplots(figsize=(10, 7))
    if not pivot.empty:
        values = pivot.to_numpy(dtype=float)
        vmax = np.nanpercentile(np.abs(values), 98) if np.isfinite(values).any() else 1
        image = axis.imshow(values, aspect="auto", interpolation="nearest", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
        axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
        axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=90)
        fig.colorbar(image, ax=axis, label="média padronizada entre eventos (z-score)")
    axis.set_title("Ciclo de vida médio dos episódios Niño 3.4 acima do P90")
    axis.set_xlabel("semana relativa ao onset P90")
    fig.tight_layout()
    path = write_figure(fig, notebook, "ciclo_medio_31_variaveis", table_path=table_path, run_id=run_id)
    plt.close(fig)
    return [path]


def _plot_phases(rationale: pd.DataFrame, notebook: str, *, run_id: str, table_path: Path | None = None) -> list[Path]:
    pivot = rationale.loc[rationale["variavel"].ne(TARGET)].pivot_table(index="variavel", columns="fase", values="media_z_vs_genese")
    fig, axis = plt.subplots(figsize=(8, 8))
    if not pivot.empty:
        values = pivot.reindex(columns=["genese", "crescimento", "pico", "decaimento"]).to_numpy(dtype=float)
        vmax = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1
        image = axis.imshow(values, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
        axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
        axis.set_xticks(range(4), ["gênese", "crescimento", "pico", "decaimento"])
        fig.colorbar(image, ax=axis, label="diferença média em z vs. gênese")
    axis.set_title("Base quantitativa para justificar as quatro fases")
    fig.tight_layout()
    path = write_figure(fig, notebook, "justificativa_quatro_fases", table_path=table_path, run_id=run_id)
    plt.close(fig)
    return [path]


def _plot_reduction(selection: pd.DataFrame, winners: pd.DataFrame, phase_stats: pd.DataFrame, notebook: str, *, run_id: str, table_path: Path | None = None) -> list[Path]:
    names = selection.loc[selection["indispensavel_descritivo"], "variavel"].tolist()
    pivot = phase_stats.loc[phase_stats["variavel"].isin(names)].pivot_table(index="variavel", columns="fase", values="media")
    fig, axis = plt.subplots(figsize=(8, max(5, 0.25 * max(1, len(names)))))
    if not pivot.empty:
        image = axis.imshow(pivot.reindex(columns=["genese", "crescimento", "pico", "decaimento"]).to_numpy(), aspect="auto", cmap="RdBu_r")
        axis.set_yticks(range(len(pivot.index)), pivot.index)
        axis.set_xticks(range(4), ["gênese", "crescimento", "pico", "decaimento"])
        fig.colorbar(image, ax=axis, label="média por fase na unidade original")
    axis.set_title("Ciclo de vida do conjunto reduzido de variáveis")
    fig.tight_layout()
    path = write_figure(fig, notebook, "ciclo_vida_variaveis_reduzidas", table_path=table_path, run_id=run_id)
    plt.close(fig)
    return [path]


def _plot_pca(variance: pd.DataFrame, loadings: pd.DataFrame, notebook: str, *, run_id: str, table_path: Path | None = None) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for phase, group in variance.groupby("fase") if not variance.empty else []:
        axes[0].plot(group["componente"], group["variancia_acumulada_pct"], marker="o", label=_phase_name(phase))
    axes[0].axhline(80, color="black", linestyle="--", linewidth=.6)
    axes[0].set_xlabel("componente")
    axes[0].set_ylabel("variância acumulada (%)")
    axes[0].set_title("PCA — variância acumulada")
    axes[0].legend()
    if not loadings.empty:
        top = loadings.loc[loadings["componente"].eq(1)].sort_values("abs_loading", ascending=False).head(15)
        axes[1].barh(top["variavel"].iloc[::-1], top["loading"].iloc[::-1], color="#762a83")
        axes[1].axvline(0, color="black", linewidth=.5)
    axes[1].set_title("PCA — maiores loadings do PC1")
    fig.tight_layout()
    path = write_figure(fig, notebook, "pca_variancia_loadings", table_path=table_path, run_id=run_id)
    plt.close(fig)
    return [path]


def _plot_spatial(composite: pd.DataFrame, notebook: str, *, run_id: str, table_path: Path | None = None) -> list[Path]:
    fig, axis = plt.subplots(figsize=(10, 5))
    if {"longitude", "semana_relativa_ao_onset", "ssta_media"}.issubset(composite.columns):
        pivot = composite.groupby(["semana_relativa_ao_onset", "longitude"])["ssta_media"].mean().unstack()
        values = pivot.to_numpy(dtype=float)
        vmax = np.nanpercentile(np.abs(values), 98) if np.isfinite(values).any() else 1
        image = axis.imshow(values, aspect="auto", origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
        axis.set_yticks(range(len(pivot.index)), pivot.index)
        axis.set_xticks(range(len(pivot.columns)), [f"{value:g}" for value in pivot.columns], rotation=90)
        fig.colorbar(image, ax=axis, label="SSTA média")
        axis.set_xlabel("longitude")
        axis.set_ylabel("semana relativa ao onset")
    else:
        axis.text(.5, .5, str(composite.iloc[0].get("status", "sem entrada espacial")), ha="center", va="center", wrap=True)
        axis.axis("off")
    axis.set_title("Composto espacial de SSTA alinhado ao onset P90")
    fig.tight_layout()
    path = write_figure(fig, notebook, "composto_ssta", table_path=table_path, run_id=run_id)
    plt.close(fig)
    return [path]


def _plot_kelvin_wind(peaks: pd.DataFrame, coupling: pd.DataFrame, notebook: str, *, run_id: str, table_path: Path | None = None) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    if not peaks.empty and "longitude" in peaks and peaks["longitude"].notna().any():
        for kind, group in peaks.groupby("tipo_pico"):
            axes[0].scatter(group["semana_relativa_ao_onset"], group["longitude"], label=kind)
        axes[0].legend()
        axes[0].set_ylabel("longitude")
        axes[0].set_xlabel("semana relativa")
    else:
        axes[0].text(.5, .5, "Sem longitude-resolução: fallback temporal", ha="center", va="center")
        axes[0].set_axis_off()
    if not coupling.empty:
        axes[1].bar(coupling["fase_pt"], coupling["correlacao_spearman_ssta_tau_x"], color="#238b45")
        axes[1].axhline(0, color="black", linewidth=.5)
        axes[1].set_ylabel("Spearman SSTA–tau_x")
    axes[1].set_title("Associação SSTA–vento por fase")
    fig.suptitle("Kelvin, vento zonal e acoplamento: evidência disponível")
    fig.tight_layout()
    path = write_figure(fig, notebook, "kelvin_ventos", table_path=table_path, run_id=run_id)
    plt.close(fig)
    return [path]


def _plot_hovmoller(peaks: pd.DataFrame, notebook: str, *, run_id: str, table_path: Path | None = None) -> list[Path]:
    fig, axis = plt.subplots(figsize=(10, 5))
    if not peaks.empty and peaks["longitude"].notna().any():
        pivot = peaks.pivot_table(index="semana_relativa_ao_onset", columns="longitude", values="valor", aggfunc="mean")
        axis.plot(pivot.index, pivot.to_numpy(), alpha=.45)
        axis.set_xlabel("semana relativa ao onset")
        axis.set_ylabel("longitude / trajetória de pico")
    else:
        axis.text(.5, .5, "Hovmöller requer entrada equatorial por longitude; nenhum mapa inferido sem esses dados.", ha="center", va="center", wrap=True)
        axis.set_axis_off()
    axis.set_title("Diagnóstico Hovmöller — trajetória longitudinal dos máximos")
    fig.tight_layout()
    path = write_figure(fig, notebook, "hovmoller", table_path=table_path, run_id=run_id)
    plt.close(fig)
    return [path]


def run_notebook(notebook: str, *, master_path: Path | str | None = None, ssta_path: Path | str | None = None, wind_path: Path | str | None = None, run_id: str | None = None) -> dict[str, Any]:
    """Execute one notebook contract and return tables/figures for inline display."""

    run_id = run_id or _run_id()
    master, validation = load_weekly_master(master_path)
    anomaly, zscore, transform_contract = transform_weekly(master)
    events, lifecycle, threshold = classify_p90(master)
    outputs: list[Path] = []
    figures: list[Path] = []
    if notebook == "F3NINO_01":
        for name, table in (("01_validacao_31_variaveis", validation), ("01_contrato_transformacao", transform_contract), ("01_serie_semanal_anomalia", anomaly.reset_index().rename(columns={anomaly.index.name or "index": "time"})), ("01_serie_semanal_zscore", zscore.reset_index().rename(columns={zscore.index.name or "index": "time"}))):
            outputs.append(write_table(table, notebook, name, run_id=run_id))
        figures.extend(_plot_series(master, anomaly, notebook, table_path=outputs[2], run_id=run_id))
    elif notebook == "F3NINO_02":
        for name, table in (("02_limiar_p90", threshold), ("02_eventos_p90", events), ("02_rotulos_fases_semanais", lifecycle)):
            outputs.append(write_table(table, notebook, name, run_id=run_id))
        figures.extend(_plot_p90(master, events, threshold, notebook, table_path=outputs[1], run_id=run_id))
    elif notebook == "F3NINO_03":
        composite, phase_means = cycle_composite(anomaly, zscore, lifecycle, events)
        for name, table in (("03_composito_ciclo_31_variaveis", composite), ("03_media_por_fase_31_variaveis", phase_means)):
            outputs.append(write_table(table, notebook, name, run_id=run_id))
        figures.extend(_plot_cycle(composite, notebook, table_path=outputs[0], run_id=run_id))
    elif notebook == "F3NINO_04":
        stats, rationale = phase_statistics(anomaly, zscore, lifecycle, events)
        for name, table in (("04_estatisticas_por_evento_fase", stats), ("04_justificativas_fisicas_por_fase", rationale)):
            outputs.append(write_table(table, notebook, name, run_id=run_id))
        figures.extend(_plot_phases(rationale, notebook, table_path=outputs[1], run_id=run_id))
    elif notebook == "F3NINO_05":
        stats, rationale = phase_statistics(anomaly, zscore, lifecycle, events)
        selection, winners = reduce_variables(zscore, stats)
        for name, table in (("05_reducao_variaveis", selection), ("05_representantes_por_fase", winners), ("05_justificativa_fases", rationale)):
            outputs.append(write_table(table, notebook, name, run_id=run_id))
        figures.extend(_plot_reduction(selection, winners, stats, notebook, table_path=outputs[0], run_id=run_id))
    elif notebook == "F3NINO_06":
        stats, _ = phase_statistics(anomaly, zscore, lifecycle, events)
        variance, loadings, scores = pca_by_phase(stats)
        for name, table in (("06_pca_variancia", variance), ("06_pca_loadings", loadings), ("06_pca_scores", scores)):
            outputs.append(write_table(table, notebook, name, run_id=run_id))
        figures.extend(_plot_pca(variance, loadings, notebook, table_path=outputs[0], run_id=run_id))
    elif notebook == "F3NINO_07":
        composite, status = spatial_ssta_composite(lifecycle, path=ssta_path)
        status_table = pd.DataFrame([{ "status_entrada": status, "arquivo_ssta": str(ssta_path or "busca automática"), "observacao": "Composto não disponível quando o cubo espacial não está presente." if status != "disponivel" else "Composto espacial disponível." }])
        for name, table in (("07_composto_ssta", composite), ("07_status_composto_ssta", status_table)):
            outputs.append(write_table(table, notebook, name, run_id=run_id))
        figures.extend(_plot_spatial(composite, notebook, table_path=outputs[0], run_id=run_id))
    elif notebook == "F3NINO_08":
        peaks, coupling, status = kelvin_wind_diagnostics(anomaly, lifecycle, events, ssta_path=ssta_path, wind_path=wind_path)
        status_table = pd.DataFrame([{ "status_entrada": status, "arquivo_ssta": str(ssta_path or "busca automática"), "arquivo_vento": str(wind_path or "busca automática") }])
        for name, table in (("08_picos_kelvin_ssta_vento", peaks), ("08_associacao_ssta_vento_por_fase", coupling), ("08_status_kelvin_vento", status_table)):
            outputs.append(write_table(table, notebook, name, run_id=run_id))
        figures.extend(_plot_kelvin_wind(peaks, coupling, notebook, table_path=outputs[0], run_id=run_id))
    elif notebook == "F3NINO_09":
        peaks, coupling, status = kelvin_wind_diagnostics(anomaly, lifecycle, events, ssta_path=ssta_path, wind_path=wind_path)
        status_table = pd.DataFrame([{ "status_hovmoller": status, "criterio": "trajetória longitudinal dos máximos; sem cubo espacial, resultado permanece inconclusivo" }])
        for name, table in (("09_hovmoller_picos", peaks), ("09_status_hovmoller", status_table)):
            outputs.append(write_table(table, notebook, name, run_id=run_id))
        figures.extend(_plot_hovmoller(peaks, notebook, table_path=outputs[0], run_id=run_id))
    elif notebook == "F3NINO_10":
        sensitivity = phase_boundary_sensitivity(master)
        for name, table in (("10_sensibilidade_fronteiras", sensitivity), ("10_eventos_repeticao_p90", events)):
            outputs.append(write_table(table, notebook, name, run_id=run_id))
        fig, axis = plt.subplots(figsize=(10, 5))
        if not sensitivity.empty:
            sensitivity.groupby(["genesis_weeks", "peak_fraction"])["duracao_pico"].median().plot(kind="bar", ax=axis, color="#4d9221")
            axis.set_ylabel("mediana da duração da faixa de pico (semanas)")
        axis.set_title("Sensibilidade das quatro fases às escolhas operacionais")
        axis.grid(axis="y", alpha=.2)
        fig.tight_layout()
        figures.append(write_figure(fig, notebook, "sensibilidade_fronteiras", table_path=outputs[0], run_id=run_id))
        plt.close(fig)
    elif notebook == "F3NINO_11":
        stats, rationale = phase_statistics(anomaly, zscore, lifecycle, events)
        selection, winners = reduce_variables(zscore, stats)
        variance, loadings, scores = pca_by_phase(stats)
        summary_rows = []
        for phase in ("genese", "crescimento", "pico", "decaimento"):
            chosen = selection.loc[selection["indispensavel_descritivo"], "variavel"].tolist()
            summary_rows.append({"fase": phase, "fase_pt": _phase_name(phase), "n_eventos_p90": int(events.shape[0]), "n_variaveis_31": len(PHYSICAL_COLUMNS), "n_variaveis_representantes": len(chosen), "variaveis_representantes": "|".join(chosen), "interpretacao": "síntese descritiva; confirmar estabilidade e disponibilidade antes de qualquer uso preditivo"})
        references = pd.DataFrame({"referencia": [
            "Bjerknes (1969), Atmospheric teleconnection from the equatorial Pacific.",
            "Jin (1997), An equatorial ocean recharge paradigm for ENSO.",
            "Kessler, McPhaden & Weickmann (1995), Forcing of intraseasonal Kelvin waves in the equatorial Pacific.",
            "Meinen & McPhaden (2000), Warm water volume changes and ENSO.",
            "Timmermann et al. (2018), El Niño–Southern Oscillation complexity.",
            "NOAA/NCEI, Optimum Interpolation SST v2.1.",
        ]})
        for name, table in (("11_sintese_cientifica", pd.DataFrame(summary_rows)), ("11_referencias_bibliograficas", references), ("11_variaveis_indispensaveis_descritivas", selection)):
            outputs.append(write_table(table, notebook, name, run_id=run_id))
        figures.extend(_plot_reduction(selection, winners, stats, notebook, table_path=outputs[2], run_id=run_id))
    else:
        raise ValueError(f"Notebook não reconhecido: {notebook}")
    return {"notebook": notebook, "run_id": run_id, "tables": outputs, "figures": figures, "n_eventos_p90": int(len(events)), "limiar_p90_c": float(threshold.iloc[0]["limiar_p90_c"])}
