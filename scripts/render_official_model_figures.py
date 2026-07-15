"""Render auditable semantic figures from completed official F5--F8 runs.

This is deliberately a post-processing layer.  It never trains a model and it
never treats a smoke run as scientific evidence.  Every source table must be
declared by, and hash-valid inside, a completed official ``ArtifactRun``.
Before figure registration, a deterministic semantic sidecar is written next
to each source table so the global figure lineage can trace the frozen numeric
table back to its run.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nino_brasil.artifacts import sha256_file, validate_artifact_run  # noqa: E402
from nino_brasil.viz import registrar_figura  # noqa: E402


MODEL_PHASES = (5, 6, 7, 8)
MODELS = ("rf", "xgb")
NOTEBOOKS = {
    5: "notebooks/fase5/5_ciclo_ml.ipynb",
    6: "notebooks/fase6/6_brasil_ml.ipynb",
    7: "notebooks/fase7/7_ciclo_convlstm.ipynb",
    8: "notebooks/fase8/8_brasil_convlstm.ipynb",
}


class EvidenceError(RuntimeError):
    """Raised when a requested figure lacks valid declared numeric evidence."""


@dataclass(frozen=True)
class OfficialRun:
    directory: Path
    manifest: Mapping[str, Any]
    finished_at: datetime
    model: str = ""

    @property
    def phase(self) -> int:
        return int(self.manifest["phase"])

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])


@dataclass(frozen=True)
class TableSource:
    name: str
    path: Path
    frame: pd.DataFrame
    declared_sha256: str


@dataclass
class RenderReport:
    rendered: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def skip(self, code: str, reason: str) -> None:
        message = f"{code}: {reason}"
        self.skipped.append(message)
        print(f"[skip] {message}")


Validator = Callable[[str | Path], pd.DataFrame]
Registrar = Callable[..., Path]


def _parse_finished_at(value: object) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("finished_at ausente")
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("finished_at precisa incluir fuso horario")
    return parsed.astimezone(timezone.utc)


def discover_valid_official_runs(
    root: Path,
    phase: int,
    *,
    validator: Validator = validate_artifact_run,
) -> list[OfficialRun]:
    """Return only complete, hash-valid official runs for one model phase."""

    if phase not in MODEL_PHASES:
        raise ValueError(f"Fase fora do escopo F5--F8: {phase}")
    run_root = Path(root) / "data" / "processed" / "runs" / "official" / f"fase{phase}"
    candidates: list[OfficialRun] = []
    for directory in sorted(run_root.glob(f"F{phase}_*")) if run_root.exists() else []:
        if not directory.is_dir():
            continue
        manifest_path = directory / "run_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                continue
            if manifest.get("phase") != phase:
                continue
            if manifest.get("mode") != "official" or manifest.get("status") != "complete":
                continue
            if str(manifest.get("run_id") or "") != directory.name:
                continue
            if phase == 6 and (manifest.get("parameters") or {}).get("role") != (
                "merge_pixel_shards_and_field_gate"
            ):
                continue
            finished_at = _parse_finished_at(manifest.get("finished_at"))
            problems = validator(directory)
            if not isinstance(problems, pd.DataFrame) or not problems.empty:
                continue
            model = str((manifest.get("parameters") or {}).get("model") or "").lower()
            if phase in (5, 6) and model not in MODELS:
                continue
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        candidates.append(
            OfficialRun(
                directory=directory.resolve(),
                manifest=manifest,
                finished_at=finished_at,
                model=model,
            )
        )
    return candidates


def select_latest_official_runs(runs: Iterable[OfficialRun], phase: int) -> dict[str, OfficialRun]:
    """Select by validated ``finished_at`` (never by lexicographic directory order)."""

    selected: dict[str, OfficialRun] = {}
    for run in runs:
        key = run.model if phase in (5, 6) else "default"
        previous = selected.get(key)
        if previous is None or (run.finished_at, run.run_id) > (
            previous.finished_at,
            previous.run_id,
        ):
            selected[key] = run
    return selected


def _declared_table(
    run: OfficialRun,
    name: str,
    *,
    required_columns: Sequence[str] = (),
) -> TableSource:
    table_name = f"{name}.csv"
    manifest_path = run.directory / "tables_manifest.csv"
    try:
        table_manifest = pd.read_csv(manifest_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise EvidenceError(f"manifesto de tabelas ilegivel: {manifest_path}") from exc
    required_manifest = {"table", "path", "rows", "columns", "sha256"}
    if missing := required_manifest.difference(table_manifest.columns):
        raise EvidenceError(f"tables_manifest sem colunas {sorted(missing)}")
    declared = table_manifest.loc[table_manifest["table"].astype(str).eq(table_name)]
    if len(declared) != 1:
        raise EvidenceError(f"{table_name} precisa estar declarado exatamente uma vez")
    row = declared.iloc[0]
    path = (run.directory / str(row["path"])).resolve()
    canonical = (run.directory / "tables" / table_name).resolve()
    if path != canonical:
        raise EvidenceError(f"caminho nao canonico para {table_name}: {path}")
    if not path.is_file() or sha256_file(path) != str(row["sha256"]):
        raise EvidenceError(f"hash divergente ou arquivo ausente: {table_name}")
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise EvidenceError(f"tabela ilegivel: {table_name}") from exc
    if int(row["rows"]) != len(frame) or int(row["columns"]) != frame.shape[1]:
        raise EvidenceError(f"dimensoes divergem do manifesto: {table_name}")
    if missing := set(required_columns).difference(frame.columns):
        raise EvidenceError(f"{table_name} sem colunas requeridas {sorted(missing)}")
    return TableSource(
        name=name,
        path=path,
        frame=frame,
        declared_sha256=str(row["sha256"]),
    )


def write_table_sidecar(run: OfficialRun, source: TableSource) -> Path:
    """Write the deterministic adjacent sidecar consumed by figure-lineage QA."""

    frame = source.frame
    current_sha256 = sha256_file(source.path)
    if current_sha256 != source.declared_sha256:
        raise EvidenceError(
            f"{source.name}.csv mudou depois da validação do ArtifactRun"
        )
    artifact = {
        "path": str(source.path.resolve()),
        "sha256": current_sha256,
        "hash_algorithm": "SHA-256",
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "column_order": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "null_count": {str(column): int(frame[column].isna().sum()) for column in frame.columns},
    }
    run_manifest = run.directory / "run_manifest.json"
    tables_manifest = run.directory / "tables_manifest.csv"
    payload = {
        "schema_version": "nino-brasil.semantic-table.v1",
        "run_id": run.run_id,
        "created_utc": run.finished_at.isoformat(),
        "contract": {
            "table_id": source.name,
            "phase": f"F{run.phase}",
            "method": "validated_official_artifact_run_table",
            "description": "Tabela oficial validada usada como fonte de figura semantica.",
            "evaluation_mode": "official_out_of_sample",
        },
        "artifact": artifact,
        "producer_run": {
            "run_manifest_path": str(run_manifest.resolve()),
            "run_manifest_sha256": sha256_file(run_manifest),
            "tables_manifest_path": str(tables_manifest.resolve()),
            "tables_manifest_sha256": sha256_file(tables_manifest),
        },
    }
    sidecar = source.path.with_suffix(source.path.suffix + ".manifest.json")
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if sidecar.exists() and sidecar.read_text(encoding="utf-8") == serialized:
        return sidecar
    temporary = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, sidecar)
    finally:
        temporary.unlink(missing_ok=True)
    return sidecar


def _register(
    fig: plt.Figure,
    code: str,
    run: OfficialRun,
    sources: Sequence[TableSource],
    *,
    title: str,
    description: str,
    hypothesis: str,
    registrar: Registrar,
) -> Path:
    for source in sources:
        write_table_sidecar(run, source)
    return registrar(
        fig,
        code,
        fase=run.phase,
        bloco="A",
        titulo=title,
        descricao=description,
        hipotese=hypothesis,
        fontes={source.name: source.path for source in sources},
        notebook=NOTEBOOKS[run.phase],
        run_id=run.run_id,
    )


def _finite_mean(frame: pd.DataFrame, group: str, value: str) -> pd.Series:
    working = frame[[group, value]].copy()
    working[value] = pd.to_numeric(working[value], errors="coerce")
    working = working.loc[np.isfinite(working[value])]
    if working.empty:
        return pd.Series(dtype=float)
    return working.groupby(group, sort=False)[value].mean().sort_values()


def _gate_colors(values: pd.Series) -> list[str]:
    passed = values.astype(str).str.lower().isin({"true", "1"})
    return ["#2a9d8f" if value else "#d1495b" for value in passed]


def _render_f5_importance(run: OfficialRun, registrar: Registrar) -> str:
    source = _declared_table(
        run,
        "state_importance_oos",
        required_columns=("state", "predictor", "delta_brier_permutation_oos"),
    )
    working = source.frame[["state", "predictor", "delta_brier_permutation_oos"]].copy()
    working["delta_brier_permutation_oos"] = pd.to_numeric(
        working["delta_brier_permutation_oos"], errors="coerce"
    )
    working = working.loc[np.isfinite(working["delta_brier_permutation_oos"])]
    top_predictors = (
        working.assign(_absolute=working["delta_brier_permutation_oos"].abs())
        .groupby("predictor")["_absolute"]
        .mean()
        .nlargest(15)
        .index
    )
    matrix = working.loc[working["predictor"].isin(top_predictors)].pivot_table(
        index="predictor",
        columns="state",
        values="delta_brier_permutation_oos",
        aggfunc="mean",
    )
    matrix = matrix.reindex(index=list(reversed(top_predictors.tolist())))
    if matrix.empty or not np.isfinite(matrix.to_numpy(dtype=float)).any():
        raise EvidenceError("state_importance_oos sem importância OOS finita")
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(13, 7))
    image = ax.imshow(values, aspect="auto", cmap="RdBu_r", norm=_skill_norm(values))
    ax.set_xticks(range(len(matrix.columns)), matrix.columns.astype(str), rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index.astype(str))
    ax.set_xlabel("Estado ENSO")
    ax.set_ylabel("Variável F2")
    ax.set_title(f"F5 {run.model.upper()} — Δ Brier por estado/fase (top 15)")
    plt.colorbar(image, ax=ax, shrink=0.8, label="Δ Brier por permutação OOS")
    fig.tight_layout()
    code = f"Fig_5A01_{run.model}"
    try:
        _register(
            fig,
            code,
            run,
            [source],
            title=f"Importância OOS das variáveis — {run.model.upper()}",
            description=(
                "Média da variação do Brier após permutação OOS, preservando valores "
                "negativos e agregando folds, horizontes, tipo ENSO e fase."
            ),
            hypothesis="Variáveis físicas têm utilidade preditiva diferente por estado ENSO.",
            registrar=registrar,
        )
    finally:
        plt.close(fig)
    return code


def _render_f5_gate(run: OfficialRun, registrar: Registrar) -> str:
    source = _declared_table(
        run,
        "scientific_gate",
        required_columns=("component", "value", "gate_pass"),
    )
    table = source.frame[["component", "value", "gate_pass"]].copy()
    table["value"] = pd.to_numeric(table["value"], errors="coerce")
    table = table.loc[np.isfinite(table["value"])]
    if table.empty:
        raise EvidenceError("scientific_gate sem valores finitos")
    fig, ax = plt.subplots(figsize=(11, max(5, 0.45 * len(table))))
    ax.barh(
        table["component"].astype(str),
        table["value"],
        color=_gate_colors(table["gate_pass"]),
    )
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_xlabel("Skill OOS (gate passa somente acima de zero)")
    ax.set_title(f"F5 {run.model.upper()} — horizontes e dimensões do evento")
    fig.tight_layout()
    code = f"Fig_5A02_{run.model}"
    try:
        _register(
            fig,
            code,
            run,
            [source],
            title=f"Skill e gates OOS — {run.model.upper()}",
            description=(
                "Skill por horizonte de classificação e por dimensão do evento "
                "(magnitude, tempo até o pico e duração); não é PDP."
            ),
            hypothesis="O modelo supera o comparador causal predefinido fora da amostra.",
            registrar=registrar,
        )
    finally:
        plt.close(fig)
    return code


def _heatmap(
    ax: plt.Axes,
    frame: pd.DataFrame,
    value: str,
    *,
    title: str,
    cmap: str,
    norm: Normalize | None = None,
) -> None:
    working = frame[["condition", "lag_weeks", value]].copy()
    working[value] = pd.to_numeric(working[value], errors="coerce")
    working["lag_weeks"] = pd.to_numeric(working["lag_weeks"], errors="coerce")
    pivot = working.pivot_table(
        index="condition", columns="lag_weeks", values=value, aggfunc="mean"
    )
    if pivot.empty or not np.isfinite(pivot.to_numpy(dtype=float)).any():
        raise EvidenceError(f"field_gate sem valores finitos para {value}")
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm or _skill_norm(values))
    ax.set_xticks(range(len(pivot.columns)), [str(int(x)) for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), pivot.index.astype(str))
    ax.set_xlabel("Lag (semanas)")
    ax.set_title(title)
    plt.colorbar(image, ax=ax, shrink=0.8)


def _render_f6(run: OfficialRun, registrar: Registrar) -> str:
    source = _declared_table(
        run,
        "field_gate",
        required_columns=(
            "condition",
            "lag_weeks",
            "area_weighted_mean_minimum_pixel_skill",
            "fraction_pixels_positive_skill",
            "gate_pass",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    _heatmap(
        axes[0],
        source.frame,
        "area_weighted_mean_minimum_pixel_skill",
        title="Skill mínimo dos pixels, ponderado por área",
        cmap="RdBu_r",
    )
    _heatmap(
        axes[1],
        source.frame,
        "fraction_pixels_positive_skill",
        title="Fração de pixels com skill positivo",
        cmap="viridis",
        norm=Normalize(vmin=0.0, vmax=1.0),
    )
    fig.suptitle(f"F6 {run.model.upper()} — avaliação pixel/campo no CHIRPS nativo")
    code = f"Fig_6A01_{run.model}"
    try:
        _register(
            fig,
            code,
            run,
            [source],
            title=f"Skill pixel/campo CHIRPS nativo — {run.model.upper()}",
            description=(
                "Resumo do gate de campo por condição e lag: skill mínimo ponderado "
                "dos pixels e fração de pixels positivos, sem regrid do alvo."
            ),
            hypothesis="O ML supera todos os comparadores predefinidos no campo e na maioria dos pixels.",
            registrar=registrar,
        )
    finally:
        plt.close(fig)
    return code


def _render_f7_gate(run: OfficialRun, registrar: Registrar) -> str:
    source = _declared_table(
        run,
        "scientific_gate",
        required_columns=(
            "mean_skill_f1_vs_best_persistence_or_seasonal",
            "skill_f1_f7_minus_best_f5_paired",
            "scientific_gate_pass",
        ),
    )
    row = source.frame.iloc[-1]
    labels = ["vs persistência/sazonal", "vs melhor F5 pareado"]
    values = np.asarray(
        [
            pd.to_numeric(row["mean_skill_f1_vs_best_persistence_or_seasonal"], errors="coerce"),
            pd.to_numeric(row["skill_f1_f7_minus_best_f5_paired"], errors="coerce"),
        ],
        dtype=float,
    )
    if not np.isfinite(values).any():
        raise EvidenceError("scientific_gate sem skills finitos")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2a9d8f" if np.isfinite(value) and value > 0 else "#d1495b" for value in values]
    ax.barh(labels, values, color=colors)
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_xlabel("Diferença/skill F1 macro OOS")
    ax.set_title("F7 ConvLSTM — gate conjunto contra baseline e F5 pareado")
    fig.tight_layout()
    code = "Fig_7A01"
    try:
        _register(
            fig,
            code,
            run,
            [source],
            title="Gate científico OOS da F7",
            description="Skills OOS preservados, incluindo reprovações e ausência de pareamento finito.",
            hypothesis="A rede supera tanto o baseline causal quanto o melhor F5 nas mesmas origens.",
            registrar=registrar,
        )
    finally:
        plt.close(fig)
    return code


def _render_f7_importance(run: OfficialRun, registrar: Registrar) -> str:
    scalar = _declared_table(
        run,
        "scalar_variable_importance_oos",
        required_columns=("variable", "delta_brier_occlusion_oos"),
    )
    spatial = _declared_table(
        run,
        "spatial_channel_importance_oos",
        required_columns=("channel", "delta_brier_occlusion_oos"),
    )
    scalar_rank = _finite_mean(scalar.frame, "variable", "delta_brier_occlusion_oos").tail(15)
    spatial_rank = _finite_mean(spatial.frame, "channel", "delta_brier_occlusion_oos").tail(15)
    if scalar_rank.empty or spatial_rank.empty:
        raise EvidenceError("importância F7 escalar ou espacial sem valores OOS finitos")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    axes[0].barh(scalar_rank.index.astype(str), scalar_rank.values, color="#2878b5")
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("31 variáveis escalares F2 (top 15)")
    axes[0].set_xlabel("Δ Brier por oclusão OOS")
    axes[1].barh(spatial_rank.index.astype(str), spatial_rank.values, color="#e07a5f")
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Canais espaciais GLORYS")
    axes[1].set_xlabel("Δ Brier por oclusão OOS")
    fig.suptitle("F7 ConvLSTM — importância OOS por tipo de entrada")
    code = "Fig_7A02"
    try:
        _register(
            fig,
            code,
            run,
            [scalar, spatial],
            title="Importância OOS escalar e espacial da F7",
            description="Ablação OOS separada para as 31 variáveis F2 e os campos GLORYS.",
            hypothesis="Entradas escalares e espaciais oferecem utilidade incremental distinta.",
            registrar=registrar,
        )
    finally:
        plt.close(fig)
    return code


def _render_f8_gate(run: OfficialRun, registrar: Registrar) -> str:
    source = _declared_table(
        run,
        "confirmatory_gate_by_condition",
        required_columns=("condition", "skill_vs_persistence", "skill_vs_f6", "gate_pass"),
    )
    table = source.frame.copy()
    table["skill_vs_persistence"] = pd.to_numeric(table["skill_vs_persistence"], errors="coerce")
    table["skill_vs_f6"] = pd.to_numeric(table["skill_vs_f6"], errors="coerce")
    if not np.isfinite(table[["skill_vs_persistence", "skill_vs_f6"]].to_numpy()).any():
        raise EvidenceError("gate F8 sem skills finitos")
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(11, max(5, 0.55 * len(table))))
    ax.barh(y - 0.18, table["skill_vs_persistence"], height=0.34, label="vs persistência")
    ax.barh(y + 0.18, table["skill_vs_f6"], height=0.34, label="vs F6 pareada")
    ax.set_yticks(y, table["condition"].astype(str))
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_xlabel("Skill RMSE OOS")
    ax.set_title("F8 ConvLSTM — gate por condição ENSO")
    ax.legend()
    fig.tight_layout()
    code = "Fig_8A01"
    try:
        _register(
            fig,
            code,
            run,
            [source],
            title="Gate confirmatório por condição da F8",
            description="Skill OOS contra persistência e F6 pareada nas oito condições ENSO ativas.",
            hypothesis="A rede supera ambos os comparadores em cada fase de El Niño e La Niña.",
            registrar=registrar,
        )
    finally:
        plt.close(fig)
    return code


def _skill_norm(values: np.ndarray) -> Normalize:
    finite = values[np.isfinite(values)]
    if finite.size and finite.min() < 0 < finite.max():
        bound = max(abs(float(finite.min())), abs(float(finite.max())))
        return TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    if finite.size:
        low, high = float(finite.min()), float(finite.max())
        if low == high:
            high = low + 1e-12
        return Normalize(vmin=low, vmax=high)
    return Normalize(vmin=-1.0, vmax=1.0)


def _render_f8_spatial(run: OfficialRun, registrar: Registrar) -> str:
    pixels = _declared_table(
        run,
        "pixel_metrics",
        required_columns=("lat", "lon", "is_brazil", "skill_rmse_vs_baseline"),
    )
    importance = _declared_table(
        run,
        "input_importance_oos",
        required_columns=("variable", "delta_rmse_occlusion_oos"),
    )
    pixel_frame = pixels.frame.copy()
    brazil = pixel_frame["is_brazil"].astype(str).str.lower().isin({"true", "1"})
    pixel_frame = pixel_frame.loc[brazil]
    for column in ("lat", "lon", "skill_rmse_vs_baseline"):
        pixel_frame[column] = pd.to_numeric(pixel_frame[column], errors="coerce")
    pixel_map = (
        pixel_frame.groupby(["lat", "lon"], as_index=False)["skill_rmse_vs_baseline"].mean()
    )
    pixel_map = pixel_map.loc[
        np.isfinite(pixel_map[["lat", "lon", "skill_rmse_vs_baseline"]]).all(axis=1)
    ]
    ranking = _finite_mean(importance.frame, "variable", "delta_rmse_occlusion_oos").tail(15)
    if pixel_map.empty or ranking.empty:
        raise EvidenceError("mapa de skill ou importância F8 sem valores OOS finitos")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    values = pixel_map["skill_rmse_vs_baseline"].to_numpy(dtype=float)
    scatter = axes[0].scatter(
        pixel_map["lon"],
        pixel_map["lat"],
        c=values,
        s=7,
        cmap="RdBu_r",
        norm=_skill_norm(values),
        linewidths=0,
    )
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_title("Skill RMSE médio por pixel CHIRPS original")
    axes[0].set_aspect("equal", adjustable="box")
    plt.colorbar(scatter, ax=axes[0], shrink=0.8)
    axes[1].barh(ranking.index.astype(str), ranking.values, color="#6a4c93")
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Δ RMSE por oclusão OOS")
    axes[1].set_title("Entradas mais importantes (top 15)")
    fig.suptitle("F8 ConvLSTM — distribuição espacial e importância OOS")
    code = "Fig_8A02"
    try:
        _register(
            fig,
            code,
            run,
            [pixels, importance],
            title="Skill no grid CHIRPS nativo e importância OOS da F8",
            description=(
                "Mapa no pixel CHIRPS original, agregado somente entre folds OOS, e ablação "
                "das entradas por aumento de RMSE."
            ),
            hypothesis="A utilidade espacial e das entradas varia no Brasil sem interpolar o alvo.",
            registrar=registrar,
        )
    finally:
        plt.close(fig)
    return code


def render_phase(
    root: Path,
    phase: int,
    *,
    strict: bool = False,
    validator: Validator = validate_artifact_run,
    registrar: Registrar = registrar_figura,
) -> RenderReport:
    """Render all promised figures supported by valid official evidence."""

    report = RenderReport()
    selected = select_latest_official_runs(
        discover_valid_official_runs(root, phase, validator=validator), phase
    )
    jobs: list[tuple[str, OfficialRun | None, Callable[[OfficialRun, Registrar], str]]] = []
    if phase == 5:
        for model in MODELS:
            run = selected.get(model)
            jobs.extend(
                [
                    (f"Fig_5A01_{model}", run, _render_f5_importance),
                    (f"Fig_5A02_{model}", run, _render_f5_gate),
                ]
            )
    elif phase == 6:
        for model in MODELS:
            jobs.append((f"Fig_6A01_{model}", selected.get(model), _render_f6))
    elif phase == 7:
        run = selected.get("default")
        jobs.extend([("Fig_7A01", run, _render_f7_gate), ("Fig_7A02", run, _render_f7_importance)])
    elif phase == 8:
        run = selected.get("default")
        jobs.extend([("Fig_8A01", run, _render_f8_gate), ("Fig_8A02", run, _render_f8_spatial)])
    else:
        raise ValueError(f"Fase fora do escopo F5--F8: {phase}")

    for code, run, renderer in jobs:
        if run is None:
            report.skip(code, "nenhum ArtifactRun oficial completo e válido")
            continue
        try:
            rendered_code = renderer(run, registrar)
        except EvidenceError as exc:
            report.skip(code, str(exc))
        else:
            report.rendered.append(rendered_code)
            print(f"[ok] {rendered_code} <- {run.run_id}")
    if strict and report.skipped:
        raise EvidenceError("Figuras obrigatórias não renderizadas:\n- " + "\n- ".join(report.skipped))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        type=int,
        choices=MODEL_PHASES,
        action="append",
        help="Fase a renderizar; repita para várias. O padrão é F5, F6, F7 e F8.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Falha se qualquer figura prometida não tiver evidência oficial válida.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    phases = tuple(dict.fromkeys(args.phase or MODEL_PHASES))
    reports = [render_phase(ROOT, phase, strict=args.strict) for phase in phases]
    rendered = sum(len(report.rendered) for report in reports)
    skipped = sum(len(report.skipped) for report in reports)
    print(f"[resumo] figuras={rendered} | skips={skipped} | strict={args.strict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
