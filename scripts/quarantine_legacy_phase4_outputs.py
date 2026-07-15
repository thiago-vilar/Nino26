#!/usr/bin/env python3
"""Inventory or quarantine narrowly defined pre-native Phase 4 outputs.

The default is a read-only dry-run.  ``--apply`` moves, and never deletes,
only top-level legacy Phase 4 tables whose names match the allowlist below and
the exact historical pixel-atlas Zarr.  A complete CHIRPS v4 promotion must
validate before any inventory is trusted and again after every requested move.
Native-pixel products, memberships and contracts are outside this utility's
authority.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable
import uuid


ROOT = Path(__file__).resolve().parents[1]
for _import_root in (ROOT, ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))
STATISTICS = ROOT / "data/processed/parquet/statistics"
ZARR_STATISTICS = ROOT / "data/processed/zarr/statistics"
LEGACY_ATLAS = ZARR_STATISTICS / "phase4C_atlas_pixel.zarr"
ACTIVE_TARGET = ROOT / "data/processed/zarr/features/chirps_native_weekly_targets.zarr"
QUARANTINE = ROOT / "data/quarantine/stale_derived"

LEGACY_PREFIXES = ("phase40_", "phase4A_", "phase4B_", "phase4C_", "phase4D_")
PROTECTED_EXACT_NAMES = frozenset(
    {
        "phase4_chirps_native_brazil_membership.parquet",
        "phase4_chirps_target_variable_contract.csv",
    }
)
STATE_KEYS = ("kind", "file_count", "size_bytes", "content_sha256")


@dataclass(frozen=True)
class Candidate:
    path: Path
    category: str
    reason: str
    expected_kind: str


def _is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", lambda: False)
    return path.is_symlink() or bool(is_junction())


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _inside(path: Path, parents: Iterable[Path]) -> bool:
    resolved = path.resolve(strict=False)
    return any(
        resolved == parent.resolve(strict=False)
        or parent.resolve(strict=False) in resolved.parents
        for parent in parents
    )


def _assert_no_link_components(path: Path, *, anchor: Path | None = None) -> None:
    """Reject links/junctions in every existing component below ``anchor``."""

    anchor = ROOT if anchor is None else anchor
    absolute_anchor = anchor.absolute()
    absolute_path = path.absolute()
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


def _assert_workspace_path(path: Path, *, allowed_root: Path) -> None:
    """Require lexical and resolved containment in both allowlist and workspace."""

    absolute_path = path.absolute()
    absolute_allowed = allowed_root.absolute()
    try:
        absolute_path.relative_to(absolute_allowed)
    except ValueError as exc:
        raise ValueError(f"Path outside allowed root {allowed_root}: {path}") from exc
    if not _inside(path, (allowed_root,)) or not _inside(path, (ROOT,)):
        raise ValueError(f"Resolved path escapes allowed workspace roots: {path}")
    _assert_no_link_components(path)


def _is_protected_statistic(path: Path) -> bool:
    name = path.name
    lowered = name.casefold()
    return bool(
        name in PROTECTED_EXACT_NAMES
        or "_native_" in lowered
        or "contract" in lowered
        or "contrato" in lowered
    )


def discover_candidates() -> tuple[Candidate, ...]:
    """Discover direct-child files only, plus one exact historical Zarr path."""

    candidates: list[Candidate] = []
    if STATISTICS.is_dir():
        _assert_workspace_path(STATISTICS, allowed_root=ROOT)
        for path in sorted(STATISTICS.iterdir(), key=lambda item: item.name):
            if not path.name.startswith(LEGACY_PREFIXES):
                continue
            if _is_protected_statistic(path):
                continue
            # A matching link is retained so the safety gate rejects it loudly;
            # ordinary directories are ignored because this allowlist is files-only.
            if _is_link_or_junction(path) or path.is_file():
                candidates.append(
                    Candidate(
                        path=path,
                        category="legacy_phase4_table",
                        reason="pre-native Phase 4 table superseded by canonical _native_ outputs",
                        expected_kind="file",
                    )
                )
    if _lexists(LEGACY_ATLAS):
        candidates.append(
            Candidate(
                path=LEGACY_ATLAS,
                category="legacy_phase4_zarr",
                reason="exact historical phase4C pixel atlas superseded by the native atlas",
                expected_kind="directory",
            )
        )
    return tuple(sorted(candidates, key=lambda item: item.path.as_posix()))


def _tree_state(path: Path, *, expected_kind: str) -> dict[str, object]:
    if _is_link_or_junction(path):
        raise ValueError(f"Refusing link/junction candidate: {path}")
    if expected_kind == "file" and not path.is_file():
        raise ValueError(f"Legacy table candidate is not a regular file: {path}")
    if expected_kind == "directory" and not path.is_dir():
        raise ValueError(f"Legacy Zarr candidate is not a directory: {path}")

    files: list[Path] = []
    if path.is_file():
        files = [path]
    else:
        for directory_name, directory_names, file_names in os.walk(
            path, followlinks=False
        ):
            directory = Path(directory_name)
            if _is_link_or_junction(directory):
                raise ValueError(f"Refusing linked directory in candidate: {directory}")
            for name in directory_names:
                child = directory / name
                if _is_link_or_junction(child):
                    raise ValueError(f"Refusing linked directory in candidate: {child}")
            for name in file_names:
                child = directory / name
                if _is_link_or_junction(child) or not child.is_file():
                    raise ValueError(f"Refusing non-regular file in candidate: {child}")
                files.append(child)
        files.sort(key=lambda item: item.relative_to(path).as_posix())

    digest = hashlib.sha256()
    size = 0
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        with item.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
    return {
        "kind": expected_kind,
        "file_count": len(files),
        "size_bytes": size,
        "content_sha256": digest.hexdigest(),
        "last_modified_utc": datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).isoformat(),
    }


def _write_json_atomic(path: Path, payload: object) -> None:
    _assert_workspace_path(path, allowed_root=ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_link_components(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=f".tmp-{uuid.uuid4().hex[:8]}", dir=path.parent
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


def _validated_active_target() -> dict[str, object]:
    from nino_brasil.targets.chirps_native import TARGET_CONTRACT_VERSION
    from scripts.build_phase4_chirps_targets import (
        _zarr_state_fingerprint,
        validate_promoted_target,
    )

    if TARGET_CONTRACT_VERSION != "chirps-native-weekly-v4":
        raise RuntimeError(
            "Legacy F4 quarantine requires the chirps-native-weekly-v4 contract."
        )
    report = validate_promoted_target(ACTIVE_TARGET)
    if not bool(report.get("valid")):
        raise RuntimeError(
            "Canonical CHIRPS v4 target is not fully validated; refusing quarantine: "
            f"{report.get('problems')}"
        )
    for key in ("block_signature_sha256", "target_data_content_sha256"):
        if len(str(report.get(key) or "")) != 64:
            raise RuntimeError(f"Validated CHIRPS target lacks {key}.")
    if not str(report.get("build_id") or "").strip():
        raise RuntimeError("Validated CHIRPS target lacks build_id.")
    report["target_data_state_sha256"] = _zarr_state_fingerprint(ACTIVE_TARGET)
    return report


def _validated_active_target_from_receipt(receipt_path: Path) -> dict[str, object]:
    from scripts.chirps_validation_receipt import validate_deep_validation_receipt

    return validate_deep_validation_receipt(
        receipt_path,
        root=ROOT,
        active_target=ACTIVE_TARGET,
        builder_script=ROOT / "scripts/build_phase4_chirps_targets.py",
        target_module=ROOT / "src/nino_brasil/targets/chirps_native.py",
    )


def _active_target_validation(receipt_path: Path | None) -> dict[str, object]:
    if receipt_path is None:
        return _validated_active_target()
    return _validated_active_target_from_receipt(receipt_path)


def _validation_identity(report: dict[str, object]) -> dict[str, str]:
    return {
        key: str(report.get(key) or "")
        for key in (
            "build_id",
            "block_signature_sha256",
            "target_data_content_sha256",
            "target_data_state_sha256",
            "pixel_inventory",
            "manifest",
        )
    }


def _same_state(record: dict[str, object], state: dict[str, object]) -> bool:
    return all(record.get(key) == state.get(key) for key in STATE_KEYS)


def _candidate_allowed_root(candidate: Candidate) -> Path:
    if candidate.category == "legacy_phase4_table":
        if candidate.path.absolute().parent != STATISTICS.absolute():
            raise ValueError(f"Legacy table is not a direct statistics child: {candidate.path}")
        return STATISTICS
    if candidate.path.absolute() != LEGACY_ATLAS.absolute():
        raise ValueError(f"Unexpected legacy Zarr candidate: {candidate.path}")
    return ZARR_STATISTICS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="move candidates; default is dry-run"
    )
    parser.add_argument(
        "--report-json", type=Path, help="optional inventory path inside workspace"
    )
    parser.add_argument(
        "--validation-receipt",
        type=Path,
        help=(
            "reuse a fresh data/audit/chirps_deep_validation.json receipt "
            "after cheap state/code/manifest checks"
        ),
    )
    args = parser.parse_args(argv)

    if args.apply:
        active_validation = _active_target_validation(args.validation_receipt)
    else:
        # Inventory is intentionally usable before promotion.  Mutation is not:
        # ``--apply`` above remains fail-closed until the complete native-v4
        # target validates.  Recording the failed precondition in a dry-run is
        # useful evidence for the eventual cleanup decision.
        try:
            active_validation = _active_target_validation(args.validation_receipt)
        except Exception as exc:
            active_validation = {
                "valid": False,
                "eligible_for_apply": False,
                "inspection_error": f"{type(exc).__name__}: {exc}",
            }
    active_identity = _validation_identity(active_validation)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination_root = QUARANTINE / f"{stamp}_legacy_phase4_{uuid.uuid4().hex[:8]}"
    _assert_workspace_path(destination_root, allowed_root=QUARANTINE)

    records: list[dict[str, object]] = []
    for candidate in discover_candidates():
        allowed_root = _candidate_allowed_root(candidate)
        _assert_workspace_path(candidate.path, allowed_root=allowed_root)
        state = _tree_state(candidate.path, expected_kind=candidate.expected_kind)
        destination = destination_root / candidate.path.relative_to(ROOT)
        _assert_workspace_path(destination, allowed_root=QUARANTINE)
        if _lexists(destination):
            raise FileExistsError(f"Quarantine destination already exists: {destination}")
        records.append(
            {
                "source": candidate.path.relative_to(ROOT).as_posix(),
                "destination": destination.relative_to(ROOT).as_posix(),
                "category": candidate.category,
                "reason": candidate.reason,
                "operation": "move; no deletion" if args.apply else "dry-run; no mutation",
                "status": "planned" if args.apply else "observed",
                **state,
            }
        )

    payload: dict[str, object] = {
        "schema": "nino26-legacy-phase4-quarantine-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "apply": bool(args.apply),
        "status": "in_progress" if args.apply else "dry_run_complete",
        "active_target_validation": active_validation,
        "allowed_source_roots": [
            STATISTICS.relative_to(ROOT).as_posix(),
            ZARR_STATISTICS.relative_to(ROOT).as_posix(),
        ],
        "records": records,
    }

    if args.report_json:
        report_path = (
            args.report_json if args.report_json.is_absolute() else ROOT / args.report_json
        )
        _assert_workspace_path(report_path, allowed_root=ROOT / "data/audit")
        _write_json_atomic(report_path, payload)

    if not args.apply:
        for record in records:
            print(
                f"WOULD MOVE {record['source']} -> {record['destination']} "
                f"sha256={str(record['content_sha256'])[:12]}"
            )
        print(f"items={len(records)} apply=False")
        return 0

    manifest_path = destination_root / "manifest.json"
    _write_json_atomic(manifest_path, payload)
    try:
        for index, record in enumerate(records):
            source = ROOT / str(record["source"])
            destination = ROOT / str(record["destination"])
            candidate = next(
                item for item in discover_candidates() if item.path.absolute() == source.absolute()
            )
            allowed_root = _candidate_allowed_root(candidate)
            _assert_workspace_path(source, allowed_root=allowed_root)
            current = _tree_state(source, expected_kind=candidate.expected_kind)
            if not _same_state(record, current):
                raise RuntimeError(f"Candidate changed after inventory; refusing move: {source}")
            _assert_workspace_path(destination, allowed_root=QUARANTINE)
            if _lexists(destination):
                raise FileExistsError(f"Quarantine destination already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            _assert_no_link_components(destination.parent)
            shutil.move(str(source), str(destination))
            if _lexists(source):
                raise RuntimeError(f"Source still exists after move: {source}")
            moved_state = _tree_state(destination, expected_kind=candidate.expected_kind)
            if not _same_state(record, moved_state):
                raise RuntimeError(f"Moved content hash mismatch: {destination}")
            records[index]["status"] = "moved"
            records[index]["moved_at_utc"] = datetime.now(timezone.utc).isoformat()
            records[index]["destination_state"] = moved_state
            _write_json_atomic(manifest_path, payload)
    except Exception as exc:
        payload["status"] = "move_failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        try:
            payload["post_move_active_target_validation"] = _active_target_validation(
                args.validation_receipt
            )
        except Exception as validation_exc:
            payload["post_move_validation_error"] = (
                f"{type(validation_exc).__name__}: {validation_exc}"
            )
        _write_json_atomic(manifest_path, payload)
        raise

    try:
        post_validation = _active_target_validation(args.validation_receipt)
        if _validation_identity(post_validation) != active_identity:
            raise RuntimeError("Active CHIRPS v4 identity changed during quarantine.")
    except Exception as exc:
        payload["status"] = "post_move_validation_failed"
        payload["post_move_validation_error"] = f"{type(exc).__name__}: {exc}"
        _write_json_atomic(manifest_path, payload)
        raise

    payload["post_move_active_target_validation"] = post_validation
    payload["status"] = "complete"
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(manifest_path, payload)
    print(f"moved={len(records)} manifest={manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
