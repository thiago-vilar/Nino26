#!/usr/bin/env python3
"""Quarantine superseded CHIRPS builds after a validated v4 promotion.

The default is a read-only inventory. ``--apply`` is accepted only when the
canonical target passes the full builder validation, including Zarr content,
pixel inventory and variable-contract hashes. Nothing is deleted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
for _import_root in (ROOT, ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))
QUARANTINE = ROOT / "data/quarantine/phase4_chirps_superseded"
ACTIVE_TARGET = ROOT / "data/processed/zarr/features/chirps_native_weekly_targets.zarr"
ALLOWED_SOURCE_ROOTS = (
    ROOT / "data/processed/zarr/features",
    ROOT / "data/interim/chirps_weekly_native_blocks",
)
CANDIDATES = (
    ROOT
    / "data/processed/zarr/features/chirps_native_weekly_targets.zarr.staging-07987a93",
    ROOT
    / "data/processed/zarr/features/chirps_native_weekly_targets.zarr.staging-69a1bd07",
    ROOT
    / "data/processed/zarr/features/chirps_native_weekly_targets.zarr.staging-65c2bf3c",
    ROOT / "data/interim/chirps_weekly_native_blocks/9fb1146624eb4b96",
    ROOT / "data/interim/chirps_weekly_native_blocks/b6e8e5f1860464cb",
)
INVENTORY_SCHEMA = "nino26-chirps-superseded-quarantine-v2"
TREE_STATE_SCHEMA = "nino26-tree-stat-fingerprint/v1"
METADATA_ONLY_MODE = "metadata_tree_state_only"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _candidate_paths(active_block_root: Path) -> tuple[Path, ...]:
    """Discover only rejected target stagings and inactive block roots.

    The previous fixed inventory became stale every time a new atomic staging
    failed. Discovery is intentionally limited to two derived-data roots and
    exact naming/manifest contracts; raw CHIRPS stores are never candidates.
    """

    feature_root, block_root = ALLOWED_SOURCE_ROOTS
    discovered = set(CANDIDATES)
    if feature_root.is_dir():
        discovered.update(
            path
            for path in feature_root.glob(
                "chirps_native_weekly_targets.zarr.staging-*"
            )
            if path.is_dir()
        )
    if block_root.is_dir():
        discovered.update(
            path
            for path in block_root.iterdir()
            if path.is_dir() and (path / "manifest.json").is_file()
        )
    active = active_block_root.resolve()
    return tuple(
        sorted(
            (
                path
                for path in discovered
                if path.exists() and path.resolve() != active
            ),
            key=lambda path: path.resolve().as_posix(),
        )
    )


def _inside(path: Path, parents: tuple[Path, ...]) -> bool:
    resolved = path.resolve()
    return any(
        resolved == parent.resolve() or parent.resolve() in resolved.parents
        for parent in parents
    )


def _is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", lambda: False)
    return path.is_symlink() or bool(is_junction())


def _tree_inventory(path: Path) -> tuple[dict[str, object], list[Path]]:
    """Fingerprint paths, sizes and mtimes without reading file contents."""

    if _is_link_or_junction(path):
        raise ValueError(f"Refusing link/junction candidate: {path}")
    if not path.is_file() and not path.is_dir():
        raise ValueError(f"Candidate is not a regular file or directory: {path}")

    entries: list[tuple[str, str, int, int]] = []
    files: list[Path] = []
    if path.is_file():
        stat = path.stat()
        entries.append((path.name, "file", stat.st_size, stat.st_mtime_ns))
        files.append(path)
        kind = "file"
    else:
        root_stat = path.stat()
        entries.append((".", "directory", root_stat.st_size, root_stat.st_mtime_ns))
        kind = "directory"
        for directory_name, directory_names, file_names in os.walk(
            path, followlinks=False
        ):
            directory = Path(directory_name)
            if _is_link_or_junction(directory):
                raise ValueError(f"Refusing linked directory in candidate: {directory}")
            directory_names.sort()
            file_names.sort()
            for name in directory_names:
                child = directory / name
                if _is_link_or_junction(child):
                    raise ValueError(f"Refusing linked directory in candidate: {child}")
                stat = child.stat()
                entries.append(
                    (
                        child.relative_to(path).as_posix(),
                        "directory",
                        stat.st_size,
                        stat.st_mtime_ns,
                    )
                )
            for name in file_names:
                child = directory / name
                if _is_link_or_junction(child) or not child.is_file():
                    raise ValueError(f"Refusing non-regular file in candidate: {child}")
                stat = child.stat()
                entries.append(
                    (
                        child.relative_to(path).as_posix(),
                        "file",
                        stat.st_size,
                        stat.st_mtime_ns,
                    )
                )
                files.append(child)
    entries.sort(key=lambda entry: (entry[0], entry[1]))
    files.sort(key=lambda item: item.name if path.is_file() else item.relative_to(path).as_posix())
    total_size = sum(entry[2] for entry in entries if entry[1] == "file")
    state_sha256 = hashlib.sha256(
        json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return (
        {
            "kind": kind,
            "file_count": len(files),
            "size_bytes": total_size,
            "tree_state_schema": TREE_STATE_SCHEMA,
            "tree_state_entry_count": len(entries),
            "tree_state_sha256": state_sha256,
            "last_modified_utc": datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc
            ).isoformat(),
        },
        files,
    )


def _cheap_tree_state(path: Path) -> dict[str, object]:
    state, _ = _tree_inventory(path)
    return state


def _same_cheap_tree_state(
    left: dict[str, object], right: dict[str, object]
) -> bool:
    keys = (
        "kind",
        "file_count",
        "size_bytes",
        "tree_state_schema",
        "tree_state_entry_count",
        "tree_state_sha256",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def _tree_state(path: Path) -> dict[str, object]:
    cheap_before, files = _tree_inventory(path)
    digest = hashlib.sha256()
    size = 0
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(len(relative.encode("utf-8")).to_bytes(4, "big"))
        digest.update(relative.encode("utf-8"))
        with item.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
    cheap_after = _cheap_tree_state(path)
    if not _same_cheap_tree_state(cheap_before, cheap_after):
        raise RuntimeError(f"Candidate changed while hashing; refusing inventory: {path}")
    if size != int(cheap_before["size_bytes"]):
        raise RuntimeError(f"Candidate size changed while hashing: {path}")
    return {**cheap_before, "content_sha256": digest.hexdigest()}


def _write_json_atomic(path: Path, payload: object) -> None:
    if not _inside(path, (ROOT,)):
        raise ValueError(f"Refusing report outside workspace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _validated_active_target() -> dict[str, object]:
    from scripts.build_phase4_chirps_targets import (
        _zarr_state_fingerprint,
        validate_promoted_target,
    )

    report = validate_promoted_target(ACTIVE_TARGET)
    if not bool(report.get("valid")):
        raise RuntimeError(
            "Canonical CHIRPS v4 target is not fully validated; refusing quarantine: "
            f"{report.get('problems')}"
        )
    signature = str(report.get("block_signature_sha256") or "")
    if len(signature) != 64:
        raise RuntimeError("Validated target has no immutable block signature.")
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


def _read_inventory_receipt(
    receipt_path: Path,
) -> tuple[Path, dict[str, object], str]:
    receipt_path = receipt_path if receipt_path.is_absolute() else ROOT / receipt_path
    receipt_path = receipt_path.resolve()
    if not _inside(receipt_path, (ROOT / "data/audit",)) or not receipt_path.is_file():
        raise RuntimeError(
            "CHIRPS inventory receipt must be an existing JSON inside data/audit."
        )
    try:
        receipt_bytes = receipt_path.read_bytes()
        payload = json.loads(receipt_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read CHIRPS inventory receipt: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("CHIRPS inventory receipt must contain one JSON object.")
    return receipt_path, payload, hashlib.sha256(receipt_bytes).hexdigest()


def _validate_inventory_header(
    payload: dict[str, object],
    *,
    schema: str,
    active_identity: dict[str, str],
) -> list[dict[str, object]]:
    if payload.get("schema") != schema:
        raise RuntimeError("Unexpected CHIRPS inventory receipt schema.")
    if payload.get("apply") is not False:
        raise RuntimeError("Inventory receipt must come from a dry-run (apply=false).")
    receipt_active = payload.get("active_target_validation")
    if not isinstance(receipt_active, dict) or receipt_active.get("valid") is not True:
        raise RuntimeError("Inventory receipt lacks a valid CHIRPS target identity.")
    if _validation_identity(receipt_active) != active_identity:
        raise RuntimeError("CHIRPS identity/state changed since inventory dry-run.")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not all(
        isinstance(record, dict) for record in raw_records
    ):
        raise RuntimeError("Inventory receipt records are malformed.")
    return [dict(record) for record in raw_records]


def _require_exact_candidate_list(
    records: list[dict[str, object]], candidate_paths: tuple[Path, ...]
) -> None:
    expected_sources = [path.relative_to(ROOT).as_posix() for path in candidate_paths]
    receipt_sources = [str(record.get("source") or "") for record in records]
    if receipt_sources != expected_sources or len(set(receipt_sources)) != len(
        receipt_sources
    ):
        raise RuntimeError("CHIRPS candidate list changed since inventory dry-run.")


def _load_inventory_receipt(
    receipt_path: Path,
    *,
    active_identity: dict[str, str],
    candidate_paths: tuple[Path, ...],
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """Validate a dry-run inventory and return records keyed by source path."""

    receipt_path, payload, receipt_sha256 = _read_inventory_receipt(receipt_path)
    raw_records = _validate_inventory_header(
        payload, schema=INVENTORY_SCHEMA, active_identity=active_identity
    )
    _require_exact_candidate_list(raw_records, candidate_paths)

    metadata_only = payload.get("inventory_mode") == METADATA_ONLY_MODE
    by_source: dict[str, dict[str, object]] = {}
    for source, raw_record in zip(candidate_paths, raw_records, strict=True):
        record = dict(raw_record)
        current = _cheap_tree_state(source)
        if not _same_cheap_tree_state(record, current):
            raise RuntimeError(
                f"CHIRPS candidate state changed since inventory dry-run: {source}"
            )
        content_hash = str(record.get("content_sha256") or "")
        if not metadata_only and not _SHA256.fullmatch(content_hash):
            raise RuntimeError(f"Inventory receipt lacks content hash for: {source}")
        if metadata_only and record.get("content_sha256") not in (None, ""):
            raise RuntimeError(
                f"Metadata-only receipt unexpectedly claims a content hash: {source}"
            )
        by_source[source.relative_to(ROOT).as_posix()] = {
            **current,
            # Equality above proves these receipt values still describe the
            # current tree; retain them as the auditable reused inventory.
            "file_count": record["file_count"],
            "size_bytes": record["size_bytes"],
            "content_sha256": None if metadata_only else content_hash,
            "verification_scope": (
                METADATA_ONLY_MODE if metadata_only else "byte_content_plus_tree_state"
            ),
        }

    receipt_meta = {
        "path": receipt_path.relative_to(ROOT).as_posix(),
        "sha256": receipt_sha256,
        "inventory_mode": (
            METADATA_ONLY_MODE if metadata_only else "byte_content_plus_tree_state"
        ),
    }
    return by_source, receipt_meta


def _upgrade_v1_inventory_receipt(
    receipt_path: Path,
    *,
    active_identity: dict[str, str],
    candidate_paths: tuple[Path, ...],
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """Upgrade a completed v1 byte inventory using only current stat metadata."""

    receipt_path, payload, receipt_sha256 = _read_inventory_receipt(receipt_path)
    raw_records = _validate_inventory_header(
        payload,
        schema="nino26-chirps-superseded-quarantine-v1",
        active_identity=active_identity,
    )
    _require_exact_candidate_list(raw_records, candidate_paths)

    upgraded: dict[str, dict[str, object]] = {}
    for source, record in zip(candidate_paths, raw_records, strict=True):
        current = _cheap_tree_state(source)  # also rejects links/junctions recursively
        legacy_fields = (
            "kind",
            "file_count",
            "size_bytes",
            "last_modified_utc",
        )
        if any(record.get(key) != current.get(key) for key in legacy_fields):
            raise RuntimeError(
                f"CHIRPS candidate changed since v1 inventory dry-run: {source}"
            )
        content_hash = str(record.get("content_sha256") or "")
        if not _SHA256.fullmatch(content_hash):
            raise RuntimeError(f"V1 inventory lacks content hash for: {source}")
        upgraded[source.relative_to(ROOT).as_posix()] = {
            **current,
            "file_count": record["file_count"],
            "size_bytes": record["size_bytes"],
            "content_sha256": content_hash,
        }
    return upgraded, {
        "path": receipt_path.relative_to(ROOT).as_posix(),
        "sha256": receipt_sha256,
        "schema": "nino26-chirps-superseded-quarantine-v1",
    }


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
    parser.add_argument(
        "--inventory-receipt",
        type=Path,
        help=(
            "with --apply, reuse content hashes from a matching dry-run JSON "
            "after cheap path/size/mtime state checks"
        ),
    )
    parser.add_argument(
        "--upgrade-inventory-receipt",
        type=Path,
        help=(
            "upgrade a completed v1 dry-run receipt to v2 using only current "
            "path/size/mtime metadata; requires --report-json and never moves"
        ),
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help=(
            "dry-run inventory using path/type/size/mtime fingerprints only; "
            "intended for reversible quarantine of large superseded trees"
        ),
    )
    args = parser.parse_args(argv)
    if args.inventory_receipt is not None and not args.apply:
        parser.error("--inventory-receipt requires --apply")
    if args.upgrade_inventory_receipt is not None:
        if args.apply:
            parser.error("--upgrade-inventory-receipt is dry-run only")
        if args.report_json is None:
            parser.error("--upgrade-inventory-receipt requires --report-json")
        if args.inventory_receipt is not None:
            parser.error("inventory receipt options are mutually exclusive")
        upgrade_source = (
            args.upgrade_inventory_receipt
            if args.upgrade_inventory_receipt.is_absolute()
            else ROOT / args.upgrade_inventory_receipt
        ).resolve()
        upgrade_output = (
            args.report_json
            if args.report_json.is_absolute()
            else ROOT / args.report_json
        ).resolve()
        if upgrade_source == upgrade_output:
            parser.error("v2 report must not overwrite the v1 inventory receipt")
        if not _inside(upgrade_output, (ROOT / "data/audit",)):
            parser.error("upgraded v2 report must be inside data/audit")
    if args.metadata_only:
        if args.apply:
            parser.error("--metadata-only is dry-run only")
        if args.report_json is None:
            parser.error("--metadata-only requires --report-json")
        if args.inventory_receipt is not None or args.upgrade_inventory_receipt is not None:
            parser.error("--metadata-only cannot be combined with inventory receipts")

    active = _active_target_validation(args.validation_receipt)
    active_identity = _validation_identity(active)
    active_signature = str(active["block_signature_sha256"])
    active_block_root = (
        ROOT
        / "data/interim/chirps_weekly_native_blocks"
        / active_signature[:16]
    ).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination_root = QUARANTINE / stamp
    candidate_paths = _candidate_paths(active_block_root)
    receipt_records: dict[str, dict[str, object]] | None = None
    inventory_receipt_meta: dict[str, str] | None = None
    if args.inventory_receipt is not None:
        receipt_records, inventory_receipt_meta = _load_inventory_receipt(
            args.inventory_receipt,
            active_identity=active_identity,
            candidate_paths=candidate_paths,
        )
    upgrade_records: dict[str, dict[str, object]] | None = None
    upgraded_from_meta: dict[str, str] | None = None
    if args.upgrade_inventory_receipt is not None:
        upgrade_records, upgraded_from_meta = _upgrade_v1_inventory_receipt(
            args.upgrade_inventory_receipt,
            active_identity=active_identity,
            candidate_paths=candidate_paths,
        )
    records: list[dict[str, object]] = []
    for source in candidate_paths:
        resolved = source.resolve()
        if not _inside(resolved, ALLOWED_SOURCE_ROOTS):
            raise ValueError(f"Candidate outside allowed derived roots: {source}")
        if resolved == active_block_root or active_block_root in resolved.parents:
            raise RuntimeError(f"Refusing active CHIRPS v4 block root: {source}")
        if resolved == ACTIVE_TARGET.resolve():
            raise RuntimeError(f"Refusing active promoted target: {source}")
        destination = destination_root / source.relative_to(ROOT)
        if not _inside(destination, (QUARANTINE,)) or destination.exists():
            raise ValueError(f"Invalid quarantine destination: {destination}")
        source_key = source.relative_to(ROOT).as_posix()
        state = (
            receipt_records[source_key]
            if receipt_records is not None
            else (
                upgrade_records[source_key]
                if upgrade_records is not None
                else (
                    {
                        **_cheap_tree_state(source),
                        "content_sha256": None,
                        "verification_scope": METADATA_ONLY_MODE,
                    }
                    if args.metadata_only
                    else _tree_state(source)
                )
            )
        )
        records.append(
            {
                "source": source_key,
                "destination": destination.relative_to(ROOT).as_posix(),
                "reason": "superseded CHIRPS v1/v2/v3 build or rejected staging",
                "operation": "move; no deletion" if args.apply else "dry-run",
                "status": "planned" if args.apply else "observed",
                **state,
            }
        )

    effective_inventory_mode = (
        METADATA_ONLY_MODE
        if args.metadata_only
        or (
            inventory_receipt_meta is not None
            and inventory_receipt_meta.get("inventory_mode") == METADATA_ONLY_MODE
        )
        else "byte_content_plus_tree_state"
    )
    payload: dict[str, object] = {
        "schema": INVENTORY_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "apply": args.apply,
        "inventory_mode": effective_inventory_mode,
        "active_target_validation": active,
        "records": records,
    }
    if inventory_receipt_meta is not None:
        payload["inventory_receipt"] = inventory_receipt_meta
    if upgraded_from_meta is not None:
        payload["upgraded_from"] = upgraded_from_meta
    if args.report_json:
        report_path = (
            args.report_json
            if args.report_json.is_absolute()
            else ROOT / args.report_json
        )
        _write_json_atomic(report_path, payload)

    if not args.apply:
        for record in records:
            print(
                f"WOULD MOVE {record['source']} -> {record['destination']} "
                f"({record['size_bytes']} bytes)"
            )
        print(f"items={len(records)} apply=False")
        return 0

    manifest_path = destination_root / "manifest.json"
    _write_json_atomic(manifest_path, payload)
    for index, record in enumerate(records):
        source = ROOT / str(record["source"])
        destination = ROOT / str(record["destination"])
        try:
            current_state = _cheap_tree_state(source)
            if not _same_cheap_tree_state(record, current_state):
                raise RuntimeError(
                    f"Candidate changed after inventory; refusing move: {source}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            moved_state = _cheap_tree_state(destination)
            if not _same_cheap_tree_state(record, moved_state):
                raise RuntimeError(
                    f"Moved candidate state differs from inventory: {destination}"
                )
            records[index]["status"] = "moved"
            records[index]["moved_at_utc"] = datetime.now(timezone.utc).isoformat()
            records[index]["destination_tree_state"] = moved_state
        except Exception as exc:
            records[index]["status"] = "failed"
            records[index]["error"] = f"{type(exc).__name__}: {exc}"
            _write_json_atomic(manifest_path, payload)
            raise
        _write_json_atomic(manifest_path, payload)

    post_validation = _active_target_validation(args.validation_receipt)
    if not bool(post_validation.get("valid")):
        raise RuntimeError("Active target failed validation after quarantine moves.")
    if _validation_identity(post_validation) != active_identity:
        raise RuntimeError("Active CHIRPS v4 identity/state changed during quarantine.")
    payload["post_move_active_target_validation"] = post_validation
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(manifest_path, payload)
    print(f"moved={len(records)} manifest={manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
