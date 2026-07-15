"""Fail-closed readers shared by canonical F4-F8 notebooks.

These helpers never train models or produce scientific outputs.  They select
runs by validated timestamps (not directory names), verify every ArtifactRun
hash before reading a table, and expose scientific gates without converting a
failed gate into an execution error or a positive claim.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.artifacts import validate_artifact_run  # noqa: E402
from scripts.run_fase4c_regional import verify_phase4_output_manifest  # noqa: E402
from scripts.run_fase4d_targets import verify_canonical_f4c_output  # noqa: E402


GATE_TABLE_BY_PHASE = {
    5: "scientific_gate.csv",
    6: "field_gate.csv",
    7: "scientific_gate.csv",
    8: "confirmatory_gate_by_condition.csv",
}


def _timestamp(value: object) -> pd.Timestamp:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return pd.Timestamp(datetime.min, tz=timezone.utc)
    return pd.Timestamp(parsed)


def audit_artifact_runs(
    root: Path,
    *,
    phase: int,
    mode: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Return an audit inventory plus the newest valid run per experiment arm."""

    root = Path(root).resolve()
    run_root = root / "data/processed/runs" / mode / f"fase{int(phase)}"
    rows: list[dict[str, object]] = []
    valid: list[dict[str, object]] = []
    for directory in sorted(run_root.glob(f"F{int(phase)}_*")) if run_root.is_dir() else []:
        problems: list[str] = []
        manifest_path = directory / "run_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            manifest = {}
            problems.append(f"manifest:{type(exc).__name__}")
        expected = {
            "phase": int(phase),
            "mode": mode,
            "run_id": directory.name,
            "status": "complete",
        }
        for field, value in expected.items():
            if manifest.get(field) != value:
                problems.append(f"{field}_mismatch")
        try:
            integrity = validate_artifact_run(directory)
            if not integrity.empty:
                problems.extend(
                    f"integrity:{problem}"
                    for problem in integrity.astype(str).agg("|".join, axis=1).tolist()
                )
        except Exception as exc:  # fail closed; viewer reports instead of hiding the run
            problems.append(f"integrity_exception:{type(exc).__name__}:{exc}")
        gate_table = GATE_TABLE_BY_PHASE.get(int(phase), "scientific_gate.csv")
        gate_path = directory / "tables" / gate_table
        gate_rows = gate_passes = 0
        gate_column = ""
        if gate_path.is_file() and not problems:
            gate = pd.read_csv(gate_path)
            gate_rows = len(gate)
            gate_columns = [
                column
                for column in ("gate_pass", "scientific_gate_pass", "final_gate_pass")
                if column in gate
            ]
            if gate_columns:
                gate_column = gate_columns[0]
                values = gate[gate_column].astype(str).str.lower().isin({"true", "1"})
                gate_passes = int(values.sum())
        parameters = manifest.get("parameters") or {}
        if int(phase) == 5:
            augmentation = bool(
                int(parameters.get("noise_copies", 0) or 0) > 0
                or float(parameters.get("mixup_alpha", 0.0) or 0.0) > 0
            )
        elif int(phase) == 7:
            augmentation = bool(parameters.get("augmentation", False))
        else:
            augmentation = False
        row = {
            "run_id": directory.name,
            "phase": manifest.get("phase"),
            "mode": manifest.get("mode"),
            "model": parameters.get("model", ""),
            "augmentation": augmentation,
            "status": manifest.get("status", "missing"),
            "started_at": manifest.get("started_at", ""),
            "finished_at": manifest.get("finished_at", ""),
            "gate_table": gate_table,
            "gate_column": gate_column,
            "gate_rows": gate_rows,
            "gate_passes": gate_passes,
            "valid": not problems,
            "problems": "; ".join(problems),
            "directory": str(directory),
        }
        rows.append(row)
        if not problems:
            valid.append({"directory": directory, "manifest": manifest, **row})

    valid.sort(key=lambda item: _timestamp(item.get("finished_at")), reverse=True)
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in valid:
        if phase == 5:
            experiment_arm = (
                f"{item.get('model') or 'default'}|augmentation={item['augmentation']}"
            )
        elif phase == 6:
            experiment_arm = str(item.get("model") or "default")
        elif phase == 7:
            experiment_arm = f"augmentation={item['augmentation']}"
        else:
            experiment_arm = "default"
        if experiment_arm not in seen:
            selected.append(item)
            seen.add(experiment_arm)
    audit = pd.DataFrame(rows)
    if not audit.empty:
        audit = audit.sort_values("finished_at", ascending=False, na_position="last")
    return audit, selected


def load_declared_tables(
    run: dict[str, object],
    requested: Iterable[str],
) -> dict[str, pd.DataFrame]:
    """Read only tables declared by a previously validated ArtifactRun."""

    directory = Path(run["directory"])
    integrity = validate_artifact_run(directory)
    if not integrity.empty:
        raise ValueError(f"ArtifactRun drifted after selection: {directory}")
    table_manifest = pd.read_csv(directory / "tables_manifest.csv")
    declared = set(table_manifest["table"].astype(str))
    tables: dict[str, pd.DataFrame] = {}
    for name in requested:
        if name not in declared:
            continue
        path = directory / "tables" / name
        if path.suffix.lower() == ".csv":
            tables[name] = pd.read_csv(path)
        elif path.suffix.lower() == ".parquet":
            tables[name] = pd.read_parquet(path)
    return tables


def audit_phase4_outputs(
    paths: Iterable[Path],
    *,
    canonical_f4c: bool = False,
    expected_stage: str | None = None,
    expected_enso_type: str | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """Verify one coherent Phase 4 run and every requested semantic sidecar."""

    paths = [Path(path) for path in paths]
    run_ids: set[str] = set()
    preliminary: dict[Path, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for path in paths:
        sidecar = Path(f"{path}.manifest.json")
        try:
            manifest = json.loads(sidecar.read_text(encoding="utf-8"))
            preliminary[path] = manifest
            run_id = str(manifest.get("run_id") or "").strip()
            if run_id:
                run_ids.add(run_id)
        except (OSError, json.JSONDecodeError):
            preliminary[path] = {}
    coherent_run_id = next(iter(run_ids)) if len(run_ids) == 1 else None
    for path in paths:
        problems: list[str] = []
        if not path.exists():
            problems.append("artifact_missing")
        if not Path(f"{path}.manifest.json").is_file():
            problems.append("sidecar_missing")
        if coherent_run_id is None:
            problems.append("run_id_missing_or_incoherent")
        if not problems:
            try:
                if canonical_f4c:
                    verify_canonical_f4c_output(
                        path,
                        expected_run_id=coherent_run_id,
                        expected_enso_type=expected_enso_type,
                    )
                else:
                    verify_phase4_output_manifest(path, expected_run_id=coherent_run_id)
            except Exception as exc:
                problems.append(f"integrity:{type(exc).__name__}:{exc}")
        contract = preliminary.get(path, {}).get("contract") or {}
        if expected_stage and str(contract.get("stage") or "") != expected_stage:
            problems.append("stage_mismatch")
        if expected_enso_type and contract.get("enso_type") != expected_enso_type:
            problems.append("enso_type_mismatch")
        rows.append(
            {
                "artifact": str(path),
                "run_id": preliminary.get(path, {}).get("run_id", ""),
                "stage": contract.get("stage", ""),
                "selection_contract": contract.get("selection_contract", ""),
                "valid": not problems,
                "problems": "; ".join(problems),
            }
        )
    audit = pd.DataFrame(rows)
    if audit.empty or not audit["valid"].all():
        return audit, None
    return audit, coherent_run_id


def load_phase4_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported Phase 4 table: {path}")
