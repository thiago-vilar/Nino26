"""Authoritative semantic numeric-table writer for Phase 3 outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import uuid
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SemanticTableContract:
    """Scientific and structural metadata required beside every table."""

    table_id: str
    phase: str
    method: str
    description: str
    evaluation_mode: str
    primary_keys: tuple[str, ...] = ()
    units: Mapping[str, str] = field(default_factory=dict)
    allowed_values: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    dimensions: tuple[str, ...] = ()
    random_seed: int | None = None
    fdr_family: str | None = None
    lag_convention: str | None = None


@dataclass(frozen=True)
class SemanticTableOutput:
    csv_path: Path
    metadata_path: Path
    sha256: str
    run_id: str
    rows: int


def verify_semantic_csv(
    csv_path: str | Path,
    *,
    verify_inputs: bool = True,
) -> dict[str, object]:
    """Recompute artifact/input hashes and report whether lineage still matches."""

    artifact = Path(csv_path)
    metadata_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    if not artifact.exists() or not metadata_path.exists():
        return {
            "valid": False,
            "artifact_exists": artifact.exists(),
            "manifest_exists": metadata_path.exists(),
            "artifact_hash_ok": False,
            "inputs_hash_ok": False,
        }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = metadata.get("artifact", {}).get("sha256")
    actual = sha256_file(artifact)
    input_results = []
    if verify_inputs:
        for item in metadata.get("inputs", []):
            path = Path(item["path"])
            expected_input = item.get("sha256")
            current = sha256_file(path) if path.exists() and path.is_file() else None
            input_results.append(
                {
                    "path": str(path),
                    "exists": path.exists(),
                    "expected_sha256": expected_input,
                    "current_sha256": current,
                    "hash_ok": current == expected_input,
                }
            )
    inputs_ok = all(item["hash_ok"] for item in input_results) if verify_inputs else True
    artifact_ok = bool(expected and actual == expected)
    return {
        "valid": artifact_ok and inputs_ok,
        "artifact_exists": True,
        "manifest_exists": True,
        "artifact_hash_ok": artifact_ok,
        "expected_sha256": expected,
        "current_sha256": actual,
        "inputs_hash_ok": inputs_ok,
        "inputs": input_results,
        "run_id": metadata.get("run_id"),
    }


def _git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_status(root: Path) -> dict[str, object]:
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root.as_posix()}",
                "status",
                "--porcelain=v1",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        status = result.stdout
        return {
            "dirty": bool(status.strip()),
            "porcelain_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
            "changed_paths": [line[3:] for line in status.splitlines() if len(line) > 3],
        }
    except (OSError, subprocess.CalledProcessError):
        return {"dirty": None, "porcelain_sha256": None, "changed_paths": []}


def _package_version(distribution: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(distribution)
    except Exception:
        return None


def _validate(frame: pd.DataFrame, contract: SemanticTableContract) -> None:
    missing_keys = set(contract.primary_keys).difference(frame.columns)
    if missing_keys:
        raise KeyError(f"primary keys missing from table: {sorted(missing_keys)}")
    if contract.primary_keys and frame.duplicated(list(contract.primary_keys)).any():
        examples = frame.loc[frame.duplicated(list(contract.primary_keys), keep=False), list(contract.primary_keys)].head()
        raise ValueError(f"duplicate primary keys in semantic table:\n{examples}")
    for column, allowed in contract.allowed_values.items():
        if column not in frame:
            raise KeyError(f"allowed-values column is missing: {column}")
        invalid = set(frame[column].dropna().unique()).difference(set(allowed))
        if invalid:
            raise ValueError(f"{column} contains values outside its contract: {sorted(invalid, key=str)}")
    numeric = frame.select_dtypes(include=[np.number])
    if not numeric.empty and np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("semantic tables cannot contain positive or negative infinity")
    unknown_units = set(contract.units).difference(frame.columns)
    if unknown_units:
        raise KeyError(f"units refer to absent columns: {sorted(unknown_units)}")


def write_semantic_csv(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    contract: SemanticTableContract,
    inputs: Sequence[str | Path] = (),
    parameters: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    project_root: str | Path | None = None,
) -> SemanticTableOutput:
    """Atomically write a CSV and a full-SHA256 scientific lineage sidecar."""

    table = frame.copy()
    _validate(table, contract)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = run_id or os.environ.get("NINO_RUN_ID") or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    )
    temp_csv = output.with_suffix(output.suffix + f".{uuid.uuid4().hex}.tmp")
    table.to_csv(temp_csv, index=False, lineterminator="\n")
    temp_csv.replace(output)
    digest = sha256_file(output)

    input_rows = []
    for item in inputs:
        path = Path(item)
        input_rows.append(
            {
                "path": str(path.resolve()),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() and path.is_file() else None,
                "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
            }
        )
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    metadata = {
        "schema_version": "nino-brasil.semantic-table.v1",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": asdict(contract),
        "artifact": {
            "path": str(output.resolve()),
            "sha256": digest,
            "hash_algorithm": "SHA-256",
            "rows": int(len(table)),
            "columns": int(len(table.columns)),
            "column_order": list(table.columns),
            "dtypes": {column: str(dtype) for column, dtype in table.dtypes.items()},
            "null_count": {column: int(value) for column, value in table.isna().sum().items()},
        },
        "inputs": input_rows,
        "parameters": dict(parameters or {}),
        "git_revision": _git_revision(root),
        "git_worktree": _git_status(root),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": _package_version("scipy"),
            "scikit_learn": _package_version("scikit-learn"),
        },
    }
    metadata_path = output.with_suffix(output.suffix + ".manifest.json")
    temp_metadata = metadata_path.with_suffix(metadata_path.suffix + f".{uuid.uuid4().hex}.tmp")
    temp_metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temp_metadata.replace(metadata_path)
    return SemanticTableOutput(
        csv_path=output,
        metadata_path=metadata_path,
        sha256=digest,
        run_id=run_id,
        rows=int(len(table)),
    )
