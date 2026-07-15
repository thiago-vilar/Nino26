"""Auditable semantic output runs for NINO-BRASIL.

Figures are presentation products.  Scientific authority lives in tables/Zarr
written by an :class:`ArtifactRun`, with full input/output hashes, environment,
configuration, methods, units, dimensions, seeds and fold/event lineage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import pandas as pd

from nino_brasil.io_utils import write_csv_atomic

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "data" / "processed" / "runs"
TABLE_MANIFEST_COLUMNS = (
    "table", "path", "rows", "columns", "sha256", "schema_sha256",
    "description", "units_json", "dimensions_json", "methods_json", "primary_keys",
)


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_state() -> dict[str, Any]:
    def command(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    status = command("status", "--porcelain")
    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(status and status != "unknown"),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def scientific_git_state() -> dict[str, Any]:
    """Public immutable snapshot used to refuse incompatible resumed shards."""

    return _git_state()


def _package_versions(names: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _input_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    record: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.exists(),
        "is_directory": resolved.is_dir() if resolved.exists() else False,
    }
    if not resolved.exists():
        return record
    record["size_bytes"] = resolved.stat().st_size if resolved.is_file() else None
    record["modified_ns"] = resolved.stat().st_mtime_ns
    if resolved.is_file():
        record["sha256"] = sha256_file(resolved)
    else:
        files = sorted(
            item
            for item in resolved.rglob("*")
            if item.is_file()
            and "__pycache__" not in item.parts
            and item.suffix.lower() not in {".pyc", ".pyo"}
        )
        listing = [
            {
                "relative": str(item.relative_to(resolved)).replace("\\", "/"),
                "size": item.stat().st_size,
                "sha256": sha256_file(item),
            }
            for item in files
        ]
        record["n_files"] = len(listing)
        record["tree_sha256"] = _json_hash(listing)
    return record


def scientific_input_record(path: str | Path) -> dict[str, Any]:
    """Public deterministic identity used by runners and resume orchestrators."""

    return _input_record(Path(path))


@dataclass
class ArtifactRun:
    phase: int
    mode: str
    run_id: str
    directory: Path
    manifest: dict[str, Any]
    table_rows: list[dict[str, Any]] = field(default_factory=list)
    file_rows: list[dict[str, Any]] = field(default_factory=list)

    def write_table(
        self,
        name: str,
        frame: pd.DataFrame,
        *,
        description: str,
        units: Mapping[str, str] | None = None,
        dimensions: Mapping[str, str] | None = None,
        methods: Mapping[str, Any] | None = None,
        primary_keys: Sequence[str] = (),
    ) -> Path:
        """Write one semantic CSV and add it to the run table manifest."""

        if not name or any(char in name for char in "\\/:"):
            raise ValueError(f"Nome de tabela invalido: {name!r}")
        if frame.columns.duplicated().any():
            raise ValueError(f"{name}: colunas duplicadas.")
        for key in primary_keys:
            if key not in frame:
                raise KeyError(f"{name}: chave primaria ausente {key!r}.")
        if primary_keys and frame.duplicated(list(primary_keys)).any():
            raise ValueError(f"{name}: chave primaria nao e unica: {list(primary_keys)}")
        table_dir = self.directory / "tables"
        path = write_csv_atomic(frame, table_dir / f"{name}.csv")
        schema = {column: str(dtype) for column, dtype in frame.dtypes.items()}
        self.table_rows = [row for row in self.table_rows if row["table"] != path.name]
        self.table_rows.append(
            {
                "table": path.name,
                "path": str(path.relative_to(self.directory)).replace("\\", "/"),
                "rows": int(len(frame)),
                "columns": int(frame.shape[1]),
                "sha256": sha256_file(path),
                "schema_sha256": _json_hash(schema),
                "description": description,
                "units_json": json.dumps(units or {}, sort_keys=True, ensure_ascii=False),
                "dimensions_json": json.dumps(dimensions or {}, sort_keys=True, ensure_ascii=False),
                "methods_json": json.dumps(methods or {}, sort_keys=True, ensure_ascii=False),
                "primary_keys": ";".join(primary_keys),
            }
        )
        return path

    def register_file(self, path: str | Path, *, role: str, description: str) -> Path:
        """Register a model/Zarr/figure already written inside this run directory."""

        path = Path(path)
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self.directory.resolve())
        except ValueError as exc:
            raise ValueError("Arquivos registrados precisam ficar dentro do diretorio do run.") from exc
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(resolved)
        self.file_rows = [row for row in self.file_rows if row["path"] != str(relative)]
        self.file_rows.append(
            {
                "path": str(relative).replace("\\", "/"),
                "role": role,
                "description": description,
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
        return resolved

    def register_directory(self, path: str | Path, *, role: str, description: str) -> Path:
        """Register a Zarr or other directory artifact through a deterministic tree hash."""

        path = Path(path).resolve()
        try:
            relative = path.relative_to(self.directory.resolve())
        except ValueError as exc:
            raise ValueError("Diretorios registrados precisam ficar dentro do run.") from exc
        if not path.is_dir():
            raise FileNotFoundError(path)
        record = _input_record(path)
        self.file_rows = [row for row in self.file_rows if row["path"] != str(relative)]
        self.file_rows.append(
            {
                "path": str(relative).replace("\\", "/"),
                "role": role,
                "description": description,
                "n_files": record.get("n_files"),
                "tree_sha256": record.get("tree_sha256"),
            }
        )
        return path

    def finalize(self, *, status: str = "complete", notes: str = "") -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        tables = (
            pd.DataFrame(self.table_rows).sort_values("table")
            if self.table_rows
            else pd.DataFrame(columns=TABLE_MANIFEST_COLUMNS)
        )
        tables_path = write_csv_atomic(tables, self.directory / "tables_manifest.csv")
        self.manifest.update(
            {
                "status": status,
                "notes": notes,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "n_tables": len(self.table_rows),
                "tables_manifest_sha256": sha256_file(tables_path),
                "files": self.file_rows,
            }
        )
        path = self.directory / "run_manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path


def start_artifact_run(
    phase: int,
    *,
    mode: str,
    inputs: Sequence[str | Path] = (),
    config_paths: Sequence[str | Path] = (ROOT / "configs" / "project.yaml",),
    seed: int = 42,
    parameters: Mapping[str, Any] | None = None,
    command: str = "",
) -> ArtifactRun:
    """Create an isolated official or smoke run directory and provenance record."""

    if mode not in {"official", "smoke"}:
        raise ValueError("mode deve ser 'official' ou 'smoke'.")
    now = datetime.now(timezone.utc)
    git = _git_state()
    identity = {
        "phase": int(phase),
        "mode": mode,
        "started_at": now.isoformat(),
        "git_commit": git["commit"],
        "seed": int(seed),
        "parameters": dict(parameters or {}),
    }
    run_id = f"F{phase}_{now.strftime('%Y%m%dT%H%M%SZ')}_{_json_hash(identity)[:10]}"
    directory = RUN_ROOT / mode / f"fase{phase}" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    config_records = [_input_record(Path(path)) for path in config_paths]
    implementation_root = ROOT / "src" / "nino_brasil"
    input_paths = [Path(path) for path in inputs]
    if implementation_root.exists():
        input_paths.append(implementation_root)
    unique_inputs: list[Path] = []
    seen_inputs: set[Path] = set()
    for path in input_paths:
        resolved = path.resolve()
        if resolved not in seen_inputs:
            seen_inputs.add(resolved)
            unique_inputs.append(resolved)
    input_records = [_input_record(path) for path in unique_inputs]
    manifest = {
        "schema_version": "nino26-run-v1",
        "run_id": run_id,
        "phase": int(phase),
        "mode": mode,
        "started_at": now.isoformat(),
        "command": command,
        "seed": int(seed),
        "parameters": dict(parameters or {}),
        "parameters_sha256": _json_hash(dict(parameters or {})),
        "git": git,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(
                ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "torch", "xarray", "zarr"]
            ),
        },
        "configs": config_records,
        "inputs": input_records,
    }
    return ArtifactRun(phase=int(phase), mode=mode, run_id=run_id, directory=directory, manifest=manifest)


def validate_artifact_run(directory: str | Path) -> pd.DataFrame:
    """Recompute hashes and report any missing or modified run artifact."""

    directory = Path(directory)
    problems: list[dict[str, str]] = []
    manifest_path = directory / "run_manifest.json"
    table_manifest_path = directory / "tables_manifest.csv"
    for required in (manifest_path, table_manifest_path):
        if not required.exists():
            problems.append({"type": "missing", "item": str(required)})
    if not table_manifest_path.exists():
        return pd.DataFrame(problems)
    tables = pd.read_csv(table_manifest_path)
    for row in tables.itertuples():
        path = directory / row.path
        if not path.exists():
            problems.append({"type": "missing_table", "item": str(path)})
        elif sha256_file(path) != row.sha256:
            problems.append({"type": "hash_mismatch", "item": str(path)})
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_manifest_hash = manifest.get("tables_manifest_sha256")
        if expected_manifest_hash and sha256_file(table_manifest_path) != expected_manifest_hash:
            problems.append({"type": "tables_manifest_hash_mismatch", "item": str(table_manifest_path)})
        for record in manifest.get("files", []):
            path = directory / record["path"]
            if not path.exists():
                problems.append({"type": "missing_registered_file", "item": str(path)})
            elif path.is_file() and record.get("sha256") and sha256_file(path) != record["sha256"]:
                problems.append({"type": "registered_file_hash_mismatch", "item": str(path)})
        for record in [*manifest.get("configs", []), *manifest.get("inputs", [])]:
            path = Path(str(record.get("path", "")))
            if not path.exists():
                problems.append({"type": "missing_input", "item": str(path)})
                continue
            current = _input_record(path)
            is_source = path == ROOT / "src" / "nino_brasil"
            if record.get("sha256") and current.get("sha256") != record.get("sha256"):
                kind = "warning_source_fingerprint_modified" if is_source else "input_hash_mismatch"
                problems.append({"type": kind, "item": str(path)})
            if record.get("tree_sha256") and current.get("tree_sha256") != record.get(
                "tree_sha256"
            ):
                kind = "warning_source_fingerprint_modified" if is_source else "input_tree_hash_mismatch"
                problems.append({"type": kind, "item": str(path)})
    return pd.DataFrame(problems)
