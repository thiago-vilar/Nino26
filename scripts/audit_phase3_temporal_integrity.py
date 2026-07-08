#!/usr/bin/env python3
"""Audita a integridade temporal dos artefatos usados na Fase 3."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "processed" / "parquet" / "features"
STATS = ROOT / "data" / "processed" / "parquet" / "statistics"


@dataclass(frozen=True)
class Artifact:
    name: str
    path: Path
    role: str
    time_col: str | None
    expected_freq: str
    max_gap_days: int | None
    expected_start: str | None = None
    stale_after_days: int | None = None
    key_cols: tuple[str, ...] = ()
    regular: bool = True
    scope: str = "fase3"


ARTIFACTS = [
    Artifact(
        "oisst_diario_nino34",
        FEAT / "nino34_daily_oisst.csv",
        "SSTA/SST diaria Nino 3.4; base da fase",
        "time",
        "D",
        1,
        expected_start="1981-09-01",
        stale_after_days=21,
        key_cols=("nino34_sst", "nino34_ssta"),
    ),
    Artifact(
        "oisst_mensal_nino34",
        FEAT / "nino34_monthly_oisst.csv",
        "SSTA mensal e media 3 meses; eventos e picos",
        "time",
        "MS",
        31,
        expected_start="1981-09-01",
        stale_after_days=45,
        key_cols=("nino34_ssta_c", "nino34_ssta_3mo_mean_c"),
    ),
    Artifact(
        "sinal_fisico_nino34",
        FEAT / "nino34_physical_signal.csv",
        "SSTA + D20/OHC/WWV/SSH diarios; salinidade fora do escopo da Fase 3",
        "time",
        "D",
        1,
        expected_start="1981-09-01",
        stale_after_days=21,
        key_cols=("nino34_ssta", "d20_nino34_mean_m", "ohc_0_300_nino34_j_m2", "ssh_nino34_mean_m"),
    ),
    Artifact(
        "era5_atmo_nino34",
        FEAT / "era5_nino34_atmo_cache.csv",
        "Vento/pressao/vapor ERA5 para proxy atmosferico",
        "time",
        "D",
        1,
        expected_start="1981-01-01",
        stale_after_days=45,
        key_cols=("atm_10m_u_component_of_wind", "atm_mean_sea_level_pressure"),
    ),
    Artifact(
        "matriz_semanal_fase3",
        FEAT / "phase3_indices_semanais.csv",
        "Matriz canonica semanal W-SUN dos notebooks",
        "week_ending_sunday",
        "W-SUN",
        7,
        expected_start="1981-01-04",
        stale_after_days=28,
        key_cols=("nino34_ssta", "d20_m", "ohc_0_300", "wwv", "ssh_m", "tau_x_anom_nino34_pa"),
    ),
    Artifact(
        "pacifico_equatorial_lon_weekly",
        FEAT / "equatorial_pacific_ssta_weekly_by_lon.parquet",
        "Hovmoller semanal SSTA por longitude",
        None,
        "W-SUN",
        7,
        expected_start="1981-09-06",
        stale_after_days=28,
    ),
    Artifact(
        "ssh_kelvin_eventos",
        FEAT / "ssh_equatorial_daily_by_lon_events.parquet",
        "SSH equatorial em janelas de eventos/Kelvin",
        None,
        "event_windows",
        None,
        expected_start=None,
        stale_after_days=None,
        regular=False,
    ),
    Artifact(
        "atlantico_tropical_legacy",
        FEAT / "tropical_atlantic_sst_daily.csv",
        "SST Atlantico tropical; legado, fora do parecer Fase 3 atual",
        "time",
        "D",
        1,
        expected_start="1981-09-01",
        stale_after_days=45,
        scope="legacy",
    ),
    Artifact(
        "eventos_elnino_referencia",
        FEAT / "nino34_oisst_event_reference.csv",
        "Tabela de eventos OISST; regra NOAA/ONI local: 3 meses >=0.5C por 5 estacoes sobrepostas",
        "peak_time",
        "events",
        None,
        regular=False,
        key_cols=("peak_oni_local_c", "peak_class"),
    ),
]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _time_values(frame: pd.DataFrame, artifact: Artifact) -> pd.Series:
    if artifact.time_col and artifact.time_col in frame.columns:
        return pd.to_datetime(frame[artifact.time_col], errors="coerce")
    if not isinstance(frame.index, pd.RangeIndex):
        return pd.Series(pd.to_datetime(frame.index, errors="coerce"), index=frame.index)
    if "time" in frame.columns:
        return pd.to_datetime(frame["time"], errors="coerce")
    raise ValueError("sem coluna temporal ou indice datetime")


def _expected_missing(times: pd.Series, artifact: Artifact) -> int:
    if not artifact.regular or artifact.expected_freq in {"events", "peaks", "event_windows"}:
        return 0
    valid = pd.to_datetime(times.dropna()).sort_values()
    if valid.empty:
        return 0
    if artifact.expected_freq == "MS":
        actual = set(valid.dt.to_period("M"))
        expected = set(pd.period_range(valid.min().to_period("M"), valid.max().to_period("M"), freq="M"))
        return max(len(expected - actual), 0)
    expected = pd.date_range(valid.min().normalize(), valid.max().normalize(), freq=artifact.expected_freq)
    actual = set(valid.dt.normalize())
    return max(len(set(expected) - actual), 0)


def _null_summary(frame: pd.DataFrame, key_cols: tuple[str, ...]) -> tuple[float | None, str]:
    cols = [c for c in key_cols if c in frame.columns]
    if not cols:
        numeric_cols = list(frame.select_dtypes("number").columns[:12])
        cols = numeric_cols
    if not cols or len(frame) == 0:
        return None, ""
    pct = frame[cols].isna().mean().sort_values(ascending=False) * 100.0
    return float(pct.iloc[0]), str(pct.index[0])


def audit_one(artifact: Artifact, audit_date: pd.Timestamp) -> dict[str, object]:
    row: dict[str, object] = {
        "artifact": artifact.name,
        "scope": artifact.scope,
        "role": artifact.role,
        "path": str(artifact.path.relative_to(ROOT)),
        "expected_freq": artifact.expected_freq,
        "regular": artifact.regular,
        "status": "ok",
        "notes": "",
    }
    notes: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []

    if not artifact.path.exists():
        row.update({"status": "error", "notes": "arquivo ausente"})
        return row

    try:
        frame = _read_table(artifact.path)
    except Exception as exc:  # pragma: no cover - report path handles the issue.
        row.update({"status": "error", "notes": f"falha leitura: {type(exc).__name__}: {exc}"})
        return row

    row["rows"] = int(len(frame))
    row["columns"] = int(len(frame.columns))
    row["bytes"] = int(artifact.path.stat().st_size)
    if frame.empty:
        row.update({"status": "error", "notes": "tabela vazia"})
        return row

    try:
        times = _time_values(frame, artifact)
        valid = pd.to_datetime(times.dropna()).sort_values()
    except Exception as exc:
        row.update({"status": "error", "notes": f"tempo invalido: {type(exc).__name__}: {exc}"})
        return row

    if valid.empty:
        row.update({"status": "error", "notes": "sem datas validas"})
        return row

    row["start"] = valid.min().date().isoformat()
    row["end"] = valid.max().date().isoformat()
    row["duplicate_timestamps"] = int(valid.duplicated().sum())
    row["missing_expected_steps"] = int(_expected_missing(times, artifact))
    if len(valid) > 1:
        diffs = valid.diff().dropna().dt.total_seconds() / 86400.0
        row["max_gap_days"] = float(diffs.max())
        row["median_gap_days"] = float(diffs.median())
    else:
        row["max_gap_days"] = None
        row["median_gap_days"] = None

    null_pct, null_col = _null_summary(frame, artifact.key_cols)
    row["max_key_null_pct"] = None if null_pct is None else round(null_pct, 2)
    row["max_key_null_col"] = null_col

    freshness_days = int((audit_date.normalize() - pd.Timestamp(valid.max()).normalize()).days)
    row["freshness_days"] = freshness_days

    if artifact.expected_start:
        start = pd.Timestamp(row["start"])
        expected_start = pd.Timestamp(artifact.expected_start)
        if start > expected_start:
            warnings.append(f"inicio posterior ao esperado ({artifact.expected_start})")
    if artifact.regular:
        if row["duplicate_timestamps"]:
            errors.append("datas duplicadas")
        if artifact.max_gap_days is not None and row["max_gap_days"] is not None:
            if float(row["max_gap_days"]) > artifact.max_gap_days:
                errors.append(f"gap max {row['max_gap_days']:.1f}d > {artifact.max_gap_days}d")
        if row["missing_expected_steps"]:
            errors.append(f"passos temporais ausentes={row['missing_expected_steps']}")
    if artifact.stale_after_days is not None and freshness_days > artifact.stale_after_days:
        warnings.append(f"defasagem {freshness_days}d > {artifact.stale_after_days}d")
    if null_pct is not None and null_pct > 20.0:
        warnings.append(f"nulos altos em {null_col}: {null_pct:.1f}%")

    if errors:
        row["status"] = "error"
    elif warnings:
        row["status"] = "warning"
    else:
        row["status"] = "ok"
    notes.extend(errors)
    notes.extend(warnings)
    row["notes"] = "; ".join(notes)
    return row


def _markdown_table(frame: pd.DataFrame) -> str:
    cols = [
        "artifact",
        "scope",
        "expected_freq",
        "start",
        "end",
        "rows",
        "max_gap_days",
        "freshness_days",
        "max_key_null_pct",
        "status",
        "notes",
    ]
    out = frame[cols].copy()
    out = out.fillna("")
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in out.iterrows():
        values = [str(row[c]).replace("|", "/") for c in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-date", default=date.today().isoformat(), help="Data de referencia YYYY-MM-DD.")
    parser.add_argument("--csv-output", default=str(STATS / "phase3_temporal_integrity_audit.csv"))
    parser.add_argument("--md-output", default=str(STATS / "phase3_temporal_integrity_audit.md"))
    args = parser.parse_args(argv)

    audit_date = pd.Timestamp(args.audit_date)
    rows = [audit_one(artifact, audit_date) for artifact in ARTIFACTS]
    report = pd.DataFrame(rows)

    csv_path = Path(args.csv_output)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(csv_path, index=False)

    md_path = Path(args.md_output)
    if not md_path.is_absolute():
        md_path = ROOT / md_path
    status_counts = report["status"].value_counts().to_dict()
    md = [
        "# Auditoria temporal da Fase 3",
        "",
        f"Data de referencia: {audit_date.date().isoformat()}",
        "",
        "Status `warning` indica ressalva de frescor/cobertura/nulos; `error` indica quebra de integridade temporal regular.",
        "",
        _markdown_table(report),
        "",
        f"Resumo: {status_counts}",
        "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")

    errors = int((report["status"] == "error").sum())
    warnings = int((report["status"] == "warning").sum())
    print(f"Temporal integrity audit: errors={errors}; warnings={warnings}; csv={csv_path}; md={md_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
