#!/usr/bin/env python3
"""Inventory or quarantine incomplete F5--F8 ``ArtifactRun`` directories.

The default operation is a read-only dry-run.  Discovery is deliberately
narrow: only direct child directories of
``data/processed/runs/{official,smoke}/fase{5,6,7,8}`` are inspected.

``--apply`` never deletes content and never acts on discovery alone.  Every
run to be moved must be named explicitly with a repeatable ``--run-id``.  The
destination is a unique directory below
``data/quarantine/incomplete_runs``.  A per-file inventory and deterministic
tree hash are verified immediately before and after each move, while the
quarantine manifest is replaced atomically after every state transition.

Completion here is intentionally an *internal ArtifactRun integrity* check:
the manifest must say ``complete`` and its declared tables/files must still
match their hashes.  Scientific gate values are never interpreted, so a
complete, integral run remains protected even when its scientific gate is
false.  Current external input drift is also outside this cleanup utility's
authority and does not turn a historical complete run into a cleanup target.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence
import uuid


ROOT = Path(__file__).resolve().parents[1]
for _import_root in (ROOT, ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

RUN_ROOT = ROOT / "data/processed/runs"
QUARANTINE = ROOT / "data/quarantine/incomplete_runs"
AUDIT_ROOT = ROOT / "data/audit"
MODES = ("official", "smoke")
PHASES = (5, 6, 7, 8)
TABLE_MANIFEST_COLUMNS = frozenset(
    {
        "table",
        "path",
        "rows",
        "columns",
        "sha256",
        "schema_sha256",
        "description",
        "units_json",
        "dimensions_json",
        "methods_json",
        "primary_keys",
    }
)
TREE_IDENTITY_KEYS = (
    "kind",
    "directory_count",
    "file_count",
    "size_bytes",
    "tree_sha256",
)


@dataclass(frozen=True)
class Candidate:
    path: Path
    run_id: str
    phase: int
    mode: str
    completion_problems: tuple[str, ...]
    tree_state: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _json_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", lambda: False)
    return path.is_symlink() or bool(is_junction())


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _inside_resolved(path: Path, parent: Path) -> bool:
    resolved = path.resolve(strict=False)
    resolved_parent = parent.resolve(strict=False)
    return resolved == resolved_parent or resolved_parent in resolved.parents


def _assert_no_link_components(path: Path, *, anchor: Path | None = None) -> None:
    """Reject symlinks/junctions in all existing components below ``anchor``."""

    anchor = ROOT if anchor is None else anchor
    absolute_anchor = _absolute_lexical(anchor)
    absolute_path = _absolute_lexical(path)
    try:
        relative = absolute_path.relative_to(absolute_anchor)
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace lexically: {path}") from exc
    current = absolute_anchor
    if _lexists(current) and _is_link_or_junction(current):
        raise ValueError(f"Workspace root cannot be a link/junction: {current}")
    for component in relative.parts:
        current = current / component
        if _lexists(current) and _is_link_or_junction(current):
            raise ValueError(f"Refusing link/junction path component: {current}")


def _assert_path(path: Path, *, allowed_root: Path) -> None:
    """Require lexical and resolved containment in an explicit workspace root."""

    absolute_path = _absolute_lexical(path)
    absolute_allowed = _absolute_lexical(allowed_root)
    try:
        absolute_path.relative_to(absolute_allowed)
    except ValueError as exc:
        raise ValueError(f"Path outside allowed root {allowed_root}: {path}") from exc
    if not _inside_resolved(path, allowed_root) or not _inside_resolved(path, ROOT):
        raise ValueError(f"Resolved path escapes the allowed workspace root: {path}")
    _assert_no_link_components(path)


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    if _is_link_or_junction(path) or not path.is_file():
        raise ValueError(f"Refusing non-regular or linked file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def _safe_relative_member(directory: Path, raw_path: object) -> Path:
    """Resolve one POSIX manifest member without permitting absolute/``..`` paths."""

    text = str(raw_path or "").replace("\\", "/").strip()
    pure = PurePosixPath(text)
    if not text or pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ValueError(f"Unsafe relative manifest path: {text or '<missing>'}")
    if pure.drive:
        raise ValueError(f"Drive-qualified manifest path is forbidden: {text}")
    path = directory.joinpath(*pure.parts)
    _assert_path(path, allowed_root=directory)
    return path


def _parse_timestamp(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _read_json_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _artifact_directory_hash(path: Path) -> tuple[int, str]:
    """Reproduce ``ArtifactRun.register_directory``'s deterministic hash."""

    files: list[Path] = []
    for directory_name, directory_names, file_names in os.walk(path, followlinks=False):
        directory = Path(directory_name)
        if _is_link_or_junction(directory):
            raise ValueError(f"Refusing linked directory: {directory}")
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            child = directory / name
            if _is_link_or_junction(child):
                raise ValueError(f"Refusing linked directory: {child}")
        for name in file_names:
            child = directory / name
            if _is_link_or_junction(child) or not child.is_file():
                raise ValueError(f"Refusing non-regular file: {child}")
            if "__pycache__" in child.parts or child.suffix.lower() in {".pyc", ".pyo"}:
                continue
            files.append(child)
    files.sort(key=lambda item: item.relative_to(path).as_posix())
    listing = [
        {
            "relative": item.relative_to(path).as_posix(),
            "size": item.stat().st_size,
            "sha256": _sha256_file(item),
        }
        for item in files
    ]
    return len(listing), _json_hash(listing)


def _audit_provenance_catalog(
    manifest: Mapping[str, Any], section: str
) -> list[str]:
    raw_records = manifest.get(section)
    if not isinstance(raw_records, list):
        return [f"missing_or_invalid_{section}_catalog"]
    problems: list[str] = []
    for index, record in enumerate(raw_records):
        label = f"{section}:{index}"
        if not isinstance(record, Mapping):
            problems.append(f"invalid_provenance_record:{label}")
            continue
        if not str(record.get("path") or "").strip():
            problems.append(f"missing_provenance_path:{label}")
        if record.get("exists") is not True:
            problems.append(f"provenance_not_present_at_run_start:{label}")
            continue
        if bool(record.get("is_directory")):
            if not _is_sha256(record.get("tree_sha256")):
                problems.append(f"invalid_provenance_tree_sha256:{label}")
        elif not _is_sha256(record.get("sha256")):
            problems.append(f"invalid_provenance_sha256:{label}")
    return problems


def audit_completion(directory: Path, *, phase: int, mode: str) -> tuple[str, ...]:
    """Audit only immutable, internal completion evidence for one ArtifactRun."""

    phase_root = RUN_ROOT / mode / f"fase{phase}"
    _assert_path(directory, allowed_root=phase_root)
    if _absolute_lexical(directory.parent) != _absolute_lexical(phase_root):
        raise ValueError(f"ArtifactRun is not a direct phase-root child: {directory}")
    if _is_link_or_junction(directory) or not directory.is_dir():
        raise ValueError(f"ArtifactRun candidate is not a regular directory: {directory}")

    problems: list[str] = []
    manifest_path = directory / "run_manifest.json"
    tables_manifest_path = directory / "tables_manifest.csv"
    manifest: Mapping[str, Any] = {}
    if not manifest_path.is_file() or _is_link_or_junction(manifest_path):
        problems.append("missing_run_manifest")
    else:
        loaded = _read_json_mapping(manifest_path)
        if loaded is None:
            problems.append("invalid_run_manifest_json")
        else:
            manifest = loaded

    if manifest:
        if manifest.get("schema_version") != "nino26-run-v1":
            problems.append("invalid_or_missing_schema_version")
        if str(manifest.get("run_id") or "") != directory.name:
            problems.append("run_id_directory_mismatch")
        try:
            recorded_phase = int(manifest.get("phase"))
        except (TypeError, ValueError):
            recorded_phase = -1
        if recorded_phase != phase:
            problems.append("manifest_phase_mismatch")
        if str(manifest.get("mode") or "") != mode:
            problems.append("manifest_mode_mismatch")
        if str(manifest.get("status") or "") != "complete":
            problems.append("run_status_not_complete")
        started = _parse_timestamp(manifest.get("started_at"))
        finished = _parse_timestamp(manifest.get("finished_at"))
        if started is None or finished is None or finished < started:
            problems.append("invalid_or_missing_completion_timestamps")
        parameters = manifest.get("parameters")
        if not isinstance(parameters, Mapping):
            problems.append("missing_or_invalid_parameters")
        elif not _is_sha256(manifest.get("parameters_sha256")):
            problems.append("missing_or_invalid_parameters_sha256")
        elif str(manifest.get("parameters_sha256")) != _json_hash(dict(parameters)):
            problems.append("parameters_sha256_mismatch")
        if not isinstance(manifest.get("environment"), Mapping):
            problems.append("missing_or_invalid_environment")
        try:
            int(manifest.get("seed"))
        except (TypeError, ValueError):
            problems.append("missing_or_invalid_seed")
        problems.extend(_audit_provenance_catalog(manifest, "inputs"))
        problems.extend(_audit_provenance_catalog(manifest, "configs"))

    table_rows: list[dict[str, str]] = []
    if not tables_manifest_path.is_file() or _is_link_or_junction(tables_manifest_path):
        problems.append("missing_tables_manifest")
    else:
        try:
            with tables_manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = set(reader.fieldnames or ())
                missing_fields = sorted(TABLE_MANIFEST_COLUMNS - fields)
                if missing_fields:
                    problems.append(
                        "tables_manifest_schema_missing:" + ",".join(missing_fields)
                    )
                table_rows = [dict(row) for row in reader]
        except (OSError, UnicodeDecodeError, csv.Error):
            problems.append("invalid_tables_manifest_csv")
            table_rows = []
        expected_hash = str(manifest.get("tables_manifest_sha256") or "")
        if not _is_sha256(expected_hash):
            problems.append("missing_or_invalid_tables_manifest_sha256")
        elif _sha256_file(tables_manifest_path) != expected_hash:
            problems.append("tables_manifest_hash_mismatch")

    seen_tables: set[str] = set()
    for index, row in enumerate(table_rows):
        name = str(row.get("table") or "").strip()
        if not name or name in seen_tables:
            problems.append(f"invalid_or_duplicate_table_name:{index}")
            continue
        seen_tables.add(name)
        if not _is_sha256(row.get("sha256")):
            problems.append(f"invalid_table_sha256:{name}")
        if not _is_sha256(row.get("schema_sha256")):
            problems.append(f"invalid_table_schema_sha256:{name}")
        try:
            table_path = _safe_relative_member(directory, row.get("path"))
        except ValueError:
            problems.append(f"unsafe_table_path:{name}")
            continue
        if table_path.name != name:
            problems.append(f"table_name_path_mismatch:{name}")
        if not table_path.is_file() or _is_link_or_junction(table_path):
            problems.append(f"missing_or_linked_table:{name}")
        elif _is_sha256(row.get("sha256")) and _sha256_file(table_path) != row["sha256"]:
            problems.append(f"table_hash_mismatch:{name}")

    if manifest:
        try:
            declared_count = int(manifest.get("n_tables"))
        except (TypeError, ValueError):
            declared_count = -1
        if declared_count != len(table_rows):
            problems.append("manifest_n_tables_mismatch")

        raw_files = manifest.get("files")
        if not isinstance(raw_files, list):
            problems.append("missing_or_invalid_registered_files_catalog")
            raw_files = []
        seen_paths: set[str] = set()
        for index, record in enumerate(raw_files):
            if not isinstance(record, Mapping):
                problems.append(f"invalid_registered_file_record:{index}")
                continue
            raw_path = str(record.get("path") or "").replace("\\", "/")
            if raw_path in seen_paths:
                problems.append(f"duplicate_registered_path:{raw_path}")
            seen_paths.add(raw_path)
            try:
                registered = _safe_relative_member(directory, raw_path)
            except ValueError:
                problems.append(f"unsafe_registered_path:{index}")
                continue
            if not registered.exists() or _is_link_or_junction(registered):
                problems.append(f"missing_or_linked_registered_path:{raw_path}")
            elif registered.is_file():
                expected = record.get("sha256")
                if not _is_sha256(expected):
                    problems.append(f"invalid_registered_file_sha256:{raw_path}")
                elif _sha256_file(registered) != expected:
                    problems.append(f"registered_file_hash_mismatch:{raw_path}")
            elif registered.is_dir():
                expected = record.get("tree_sha256")
                if not _is_sha256(expected):
                    problems.append(f"invalid_registered_tree_sha256:{raw_path}")
                else:
                    n_files, current = _artifact_directory_hash(registered)
                    if current != expected:
                        problems.append(f"registered_tree_hash_mismatch:{raw_path}")
                    if record.get("n_files") is not None:
                        try:
                            if int(record["n_files"]) != n_files:
                                problems.append(f"registered_tree_file_count_mismatch:{raw_path}")
                        except (TypeError, ValueError):
                            problems.append(f"invalid_registered_tree_file_count:{raw_path}")
            else:
                problems.append(f"registered_path_not_regular:{raw_path}")

    return tuple(dict.fromkeys(problems))


def _tree_state(directory: Path) -> dict[str, Any]:
    """Return a full per-entry inventory plus a deterministic aggregate hash."""

    if _is_link_or_junction(directory) or not directory.is_dir():
        raise ValueError(f"Refusing non-directory or linked run: {directory}")
    entries: list[dict[str, Any]] = []
    total_size = 0
    for directory_name, directory_names, file_names in os.walk(
        directory, followlinks=False
    ):
        current = Path(directory_name)
        if _is_link_or_junction(current):
            raise ValueError(f"Refusing linked directory inside run: {current}")
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            child = current / name
            if _is_link_or_junction(child):
                raise ValueError(f"Refusing linked directory inside run: {child}")
            entries.append(
                {"kind": "directory", "path": child.relative_to(directory).as_posix()}
            )
        for name in file_names:
            child = current / name
            if _is_link_or_junction(child) or not child.is_file():
                raise ValueError(f"Refusing non-regular file inside run: {child}")
            size = child.stat().st_size
            digest = _sha256_file(child)
            total_size += size
            entries.append(
                {
                    "kind": "file",
                    "path": child.relative_to(directory).as_posix(),
                    "size_bytes": size,
                    "sha256": digest,
                }
            )
    entries.sort(key=lambda item: (str(item["path"]), str(item["kind"])))
    return {
        "kind": "directory",
        "directory_count": sum(item["kind"] == "directory" for item in entries),
        "file_count": sum(item["kind"] == "file" for item in entries),
        "size_bytes": total_size,
        "tree_sha256": _json_hash(entries),
        "root_last_modified_ns": directory.stat().st_mtime_ns,
        "entries": entries,
    }


def _same_tree_state(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in TREE_IDENTITY_KEYS)


def discover_candidates() -> tuple[Candidate, ...]:
    """Find only direct F5--F8 run directories lacking valid completion evidence."""

    candidates: list[Candidate] = []
    for mode in MODES:
        for phase in PHASES:
            phase_root = RUN_ROOT / mode / f"fase{phase}"
            if not _lexists(phase_root):
                continue
            _assert_path(phase_root, allowed_root=RUN_ROOT)
            if _is_link_or_junction(phase_root) or not phase_root.is_dir():
                raise ValueError(f"Phase run root is not a regular directory: {phase_root}")
            for path in sorted(phase_root.iterdir(), key=lambda item: item.name):
                if _is_link_or_junction(path):
                    raise ValueError(f"Refusing link/junction in run discovery root: {path}")
                if not path.is_dir():
                    continue
                problems = audit_completion(path, phase=phase, mode=mode)
                if problems:
                    candidates.append(
                        Candidate(
                            path=path,
                            run_id=path.name,
                            phase=phase,
                            mode=mode,
                            completion_problems=problems,
                            tree_state=_tree_state(path),
                        )
                    )
    return tuple(sorted(candidates, key=lambda item: item.path.as_posix()))


def _write_json_atomic(path: Path, payload: object, *, allowed_root: Path) -> None:
    _assert_path(path, allowed_root=allowed_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_link_components(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=f".tmp-{uuid.uuid4().hex[:8]}",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_run_id_token(run_id: str) -> None:
    if (
        not run_id
        or run_id in {".", ".."}
        or Path(run_id).name != run_id
        or "/" in run_id
        or "\\" in run_id
    ):
        raise ValueError(f"Unsafe --run-id value: {run_id!r}")


def _candidate_lookup(candidates: Sequence[Candidate]) -> dict[str, Candidate]:
    lookup: dict[str, Candidate] = {}
    duplicates: set[str] = set()
    for candidate in candidates:
        if candidate.run_id in lookup:
            duplicates.add(candidate.run_id)
        lookup[candidate.run_id] = candidate
    if duplicates:
        raise RuntimeError(
            "Ambiguous duplicate candidate run_id values across run roots: "
            + ", ".join(sorted(duplicates))
        )
    return lookup


def _record_for_candidate(
    candidate: Candidate, *, destination_root: Path, selected: bool, apply: bool
) -> dict[str, Any]:
    destination = destination_root / candidate.path.relative_to(ROOT)
    _assert_path(destination, allowed_root=QUARANTINE)
    return {
        "run_id": candidate.run_id,
        "phase": candidate.phase,
        "mode": candidate.mode,
        "source": candidate.path.relative_to(ROOT).as_posix(),
        "destination": destination.relative_to(ROOT).as_posix(),
        "selected_by_explicit_allowlist": selected,
        "operation": "move; no deletion" if selected and apply else "no mutation",
        "status": (
            "planned"
            if selected and apply
            else "refused_not_allowlisted"
            if apply
            else "observed_dry_run"
        ),
        "completion_problems": list(candidate.completion_problems),
        "source_tree_state": dict(candidate.tree_state),
    }


def _current_matching_candidate(record: Mapping[str, Any]) -> Candidate:
    """Re-audit only the explicitly selected source immediately before moving."""

    expected_source = ROOT / str(record["source"])
    phase = int(record["phase"])
    mode = str(record["mode"])
    phase_root = RUN_ROOT / mode / f"fase{phase}"
    _assert_path(expected_source, allowed_root=phase_root)
    if _absolute_lexical(expected_source.parent) != _absolute_lexical(phase_root):
        raise RuntimeError(f"Allowlisted source is not a direct run child: {expected_source}")
    if not expected_source.is_dir() or _is_link_or_junction(expected_source):
        raise RuntimeError(
            f"Allowlisted run is no longer a regular run directory: {record['run_id']}"
        )
    problems = audit_completion(expected_source, phase=phase, mode=mode)
    if not problems:
        raise RuntimeError(
            f"Allowlisted run became complete; refusing quarantine: {record['run_id']}"
        )
    return Candidate(
        path=expected_source,
        run_id=str(record["run_id"]),
        phase=phase,
        mode=mode,
        completion_problems=problems,
        tree_state=_tree_state(expected_source),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="move only explicitly allowlisted candidates"
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="exact incomplete run ID authorized for --apply; repeat as needed",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="optional dry-run inventory path below data/audit",
    )
    args = parser.parse_args(argv)

    requested = list(args.run_id)
    for run_id in requested:
        _validate_run_id_token(run_id)
    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate --run-id values are not allowed.")
    if args.apply and not requested:
        raise ValueError("--apply requires at least one explicit --run-id allowlist entry.")

    candidates = discover_candidates()
    lookup = _candidate_lookup(candidates)
    if args.apply:
        missing = sorted(set(requested) - set(lookup))
        if missing:
            raise ValueError(
                "--apply refuses IDs that are not current incomplete candidates: "
                + ", ".join(missing)
            )

    stamp_uuid = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex}"
    )
    destination_root = QUARANTINE / stamp_uuid
    _assert_path(destination_root, allowed_root=QUARANTINE)
    records = [
        _record_for_candidate(
            candidate,
            destination_root=destination_root,
            selected=candidate.run_id in requested,
            apply=bool(args.apply),
        )
        for candidate in candidates
    ]
    payload: dict[str, Any] = {
        "schema": "nino26-incomplete-artifact-run-quarantine-v1",
        "created_at_utc": _utc_now(),
        "apply": bool(args.apply),
        "status": "in_progress" if args.apply else "dry_run_complete",
        "scope": {
            "modes": list(MODES),
            "phases": list(PHASES),
            "depth": "direct_children_only",
            "completion_policy": "internal_manifest_and_declared_output_integrity",
            "scientific_gate_values_considered": False,
            "current_external_input_drift_considered": False,
        },
        "explicit_run_id_allowlist": requested,
        "candidate_count": len(candidates),
        "selected_count": sum(bool(row["selected_by_explicit_allowlist"]) for row in records),
        "records": records,
    }

    if args.report_json:
        report_path = args.report_json if args.report_json.is_absolute() else ROOT / args.report_json
        _write_json_atomic(report_path, payload, allowed_root=AUDIT_ROOT)

    if not args.apply:
        for record in records:
            print(
                f"WOULD QUARANTINE {record['source']} "
                f"tree_sha256={record['source_tree_state']['tree_sha256'][:12]} "
                f"problems={','.join(record['completion_problems'])}"
            )
        print(f"candidates={len(records)} apply=False")
        return 0

    manifest_path = destination_root / "manifest.json"
    _write_json_atomic(manifest_path, payload, allowed_root=QUARANTINE)
    try:
        for record in records:
            if not record["selected_by_explicit_allowlist"]:
                continue
            current = _current_matching_candidate(record)
            if not _same_tree_state(record["source_tree_state"], current.tree_state):
                raise RuntimeError(
                    f"Candidate changed after inventory; refusing move: {record['run_id']}"
                )
            source = current.path
            destination = ROOT / str(record["destination"])
            phase_root = RUN_ROOT / current.mode / f"fase{current.phase}"
            _assert_path(source, allowed_root=phase_root)
            _assert_path(destination, allowed_root=QUARANTINE)
            if _lexists(destination):
                raise FileExistsError(f"Quarantine destination already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            _assert_no_link_components(destination.parent)
            shutil.move(str(source), str(destination))
            if _lexists(source):
                raise RuntimeError(f"Source still exists after move: {source}")
            destination_state = _tree_state(destination)
            if not _same_tree_state(record["source_tree_state"], destination_state):
                raise RuntimeError(f"Moved tree state does not match inventory: {destination}")
            record["destination_tree_state"] = destination_state
            record["post_move_completion_problems"] = list(
                current.completion_problems
            )
            record["status"] = "moved_and_verified"
            record["moved_at_utc"] = _utc_now()
            _write_json_atomic(manifest_path, payload, allowed_root=QUARANTINE)
    except Exception as exc:
        payload["status"] = "move_failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        _write_json_atomic(manifest_path, payload, allowed_root=QUARANTINE)
        raise

    # A final pass is intentionally limited to the explicit allowlist.  Runs
    # outside it may still be active and are neither locked nor made part of
    # this operation's success condition.
    for record in records:
        if record["selected_by_explicit_allowlist"]:
            source = ROOT / str(record["source"])
            destination = ROOT / str(record["destination"])
            if _lexists(source):
                payload["status"] = "post_move_verification_failed"
                _write_json_atomic(manifest_path, payload, allowed_root=QUARANTINE)
                raise RuntimeError(f"Selected source remains after move: {source}")
            final_destination_state = _tree_state(destination)
            if not _same_tree_state(
                record["source_tree_state"], final_destination_state
            ):
                payload["status"] = "post_move_verification_failed"
                _write_json_atomic(manifest_path, payload, allowed_root=QUARANTINE)
                raise RuntimeError(
                    f"Selected destination changed after move: {destination}"
                )
            record["final_destination_tree_state"] = final_destination_state

    payload["status"] = "complete"
    payload["completed_at_utc"] = _utc_now()
    payload["unselected_candidates_untouched"] = len(records) - int(
        payload["selected_count"]
    )
    _write_json_atomic(manifest_path, payload, allowed_root=QUARANTINE)
    print(
        f"moved={payload['selected_count']} "
        f"manifest={manifest_path.relative_to(ROOT).as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
