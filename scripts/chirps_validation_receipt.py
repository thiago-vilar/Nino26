#!/usr/bin/env python3
"""Cheap, fail-closed reuse of a completed CHIRPS deep validation.

The deep validator hashes every CHIRPS data chunk and is intentionally
expensive.  A cleanup command may reuse its receipt only while all cheap
identity checks still match: file-state fingerprint, code hashes, immutable
build-manifest hash, promoted identities and the small external contracts.
No target data chunk is opened or hashed by this module.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
import re


SCHEMA_VERSION = "nino26-chirps-deep-validation/v1"
TARGET_CONTRACT_VERSION = "chirps-native-weekly-v4"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not _SHA256.fullmatch(text):
        raise RuntimeError(f"CHIRPS validation receipt has invalid {label}.")
    return text


def _load_mapping(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain one JSON object: {path}")
    return payload


def _canonical_relative(path: Path, *, root: Path, label: str) -> str:
    root_resolved = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(f"{label} is outside the project root: {path}") from exc
    return relative.as_posix()


def _resolve_recorded_path(value: object, *, root: Path, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"CHIRPS validation receipt/manifest lacks {label}.")
    candidate = Path(text)
    candidate = candidate if candidate.is_absolute() else root / candidate
    _canonical_relative(candidate, root=root, label=label)
    return candidate.resolve()


def _require_equal(actual: object, expected: object, *, label: str) -> None:
    if str(actual or "") != str(expected or ""):
        raise RuntimeError(f"CHIRPS validation receipt mismatch: {label}.")


def _current_zarr_state(path: Path) -> str:
    # Import the authoritative implementation.  Its source hash is checked
    # against both the receipt and manifest before the receipt is accepted.
    from scripts.build_phase4_chirps_targets import _zarr_state_fingerprint

    return _zarr_state_fingerprint(path)


def validate_deep_validation_receipt(
    receipt_path: Path,
    *,
    root: Path,
    active_target: Path,
    builder_script: Path,
    target_module: Path,
) -> dict[str, object]:
    """Return the original validation identity after cheap revalidation.

    This is deliberately stricter than merely trusting ``validation.valid``.
    Any missing field, path escape, stale state or hash/identity disagreement
    raises ``RuntimeError`` and therefore prevents mutation.
    """

    root = root.resolve()
    active_target = active_target.resolve()
    canonical_target = _canonical_relative(
        active_target, root=root, label="canonical CHIRPS target"
    )
    if not active_target.is_dir():
        raise RuntimeError(f"Canonical CHIRPS target is missing: {active_target}")

    receipt_path = receipt_path if receipt_path.is_absolute() else root / receipt_path
    receipt_path = receipt_path.resolve()
    receipt_relative = _canonical_relative(
        receipt_path, root=root, label="CHIRPS validation receipt"
    )
    audit_root = (root / "data/audit").resolve()
    try:
        receipt_path.relative_to(audit_root)
    except ValueError as exc:
        raise RuntimeError(
            "CHIRPS validation receipt must be inside data/audit."
        ) from exc

    receipt = _load_mapping(receipt_path, label="CHIRPS validation receipt")
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Unexpected CHIRPS deep-validation receipt schema.")
    _require_equal(
        receipt.get("target_path"), canonical_target, label="canonical target path"
    )

    validation_value = receipt.get("validation")
    if not isinstance(validation_value, Mapping):
        raise RuntimeError("CHIRPS validation receipt lacks validation object.")
    validation = dict(validation_value)
    if validation.get("valid") is not True:
        raise RuntimeError("CHIRPS deep-validation receipt is not valid.")
    if validation.get("problems") not in ([], ()):  # fail closed on ambiguity
        raise RuntimeError("CHIRPS deep-validation receipt records problems.")
    target_validation = validation.get("target_validation")
    if not isinstance(target_validation, Mapping) or target_validation.get("valid") is not True:
        raise RuntimeError("CHIRPS target validation inside receipt is not valid.")
    if target_validation.get("errors") not in ([], ()):  # deep schema contract
        raise RuntimeError("CHIRPS target validation inside receipt records errors.")

    manifest_path = _resolve_recorded_path(
        validation.get("manifest"), root=root, label="build manifest"
    )
    if not manifest_path.is_file():
        raise RuntimeError(f"CHIRPS build manifest is missing: {manifest_path}")
    manifest = _load_mapping(manifest_path, label="CHIRPS build manifest")

    receipt_manifest_hash = _require_sha256(
        receipt.get("build_manifest_sha256"), label="build_manifest_sha256"
    )
    _require_equal(
        _sha256_file(manifest_path),
        receipt_manifest_hash,
        label="current build manifest hash",
    )

    builder_hash = _require_sha256(
        receipt.get("builder_script_sha256"), label="builder_script_sha256"
    )
    module_hash = _require_sha256(
        receipt.get("target_module_sha256"), label="target_module_sha256"
    )
    if not builder_script.is_file() or not target_module.is_file():
        raise RuntimeError("Current CHIRPS builder or target module is missing.")
    _require_equal(
        _sha256_file(builder_script), builder_hash, label="current builder hash"
    )
    _require_equal(
        _sha256_file(target_module), module_hash, label="current target-module hash"
    )
    _require_equal(
        manifest.get("builder_script_sha256"),
        builder_hash,
        label="manifest builder hash",
    )
    _require_equal(
        manifest.get("target_module_sha256"),
        module_hash,
        label="manifest target-module hash",
    )

    if manifest.get("build_complete") is not True:
        raise RuntimeError("CHIRPS build manifest is not complete.")
    if manifest.get("promotion_status") != "promoted_after_deep_validation":
        raise RuntimeError("CHIRPS build manifest was not deeply validated/promoted.")
    if manifest.get("target_contract_version") != TARGET_CONTRACT_VERSION:
        raise RuntimeError("CHIRPS build manifest contract is not native weekly v4.")
    _require_equal(
        manifest.get("promoted_target"),
        canonical_target,
        label="manifest promoted target path",
    )

    state_hash = _require_sha256(
        receipt.get("target_data_state_sha256"),
        label="target_data_state_sha256",
    )
    _require_equal(
        manifest.get("promoted_target_data_state_sha256"),
        state_hash,
        label="manifest target state",
    )
    _require_equal(
        _current_zarr_state(active_target), state_hash, label="current target state"
    )

    build_id = str(validation.get("build_id") or "").strip()
    if not build_id:
        raise RuntimeError("CHIRPS validation receipt lacks build_id.")
    signature = _require_sha256(
        validation.get("block_signature_sha256"),
        label="validation.block_signature_sha256",
    )
    content_hash = _require_sha256(
        validation.get("target_data_content_sha256"),
        label="validation.target_data_content_sha256",
    )
    _require_equal(manifest.get("build_id"), build_id, label="build_id")
    _require_equal(
        manifest.get("signature_sha256"), signature, label="block signature"
    )
    _require_equal(
        manifest.get("promoted_target_data_content_sha256"),
        content_hash,
        label="target content hash",
    )

    expected_manifest = (
        root
        / "data/interim/chirps_weekly_native_blocks"
        / signature[:16]
        / "manifest.json"
    ).resolve()
    if manifest_path != expected_manifest:
        raise RuntimeError("CHIRPS build manifest path does not match block signature.")

    pixel_path = _resolve_recorded_path(
        validation.get("pixel_inventory"), root=root, label="pixel inventory"
    )
    manifest_pixel = _resolve_recorded_path(
        manifest.get("promoted_pixel_inventory"),
        root=root,
        label="manifest pixel inventory",
    )
    if pixel_path != manifest_pixel:
        raise RuntimeError("CHIRPS pixel-inventory identity differs from manifest.")
    pixel_hash = _require_sha256(
        manifest.get("promoted_pixel_inventory_sha256"),
        label="manifest promoted_pixel_inventory_sha256",
    )
    if not pixel_path.is_file():
        raise RuntimeError(f"CHIRPS pixel inventory is missing: {pixel_path}")
    _require_equal(
        _sha256_file(pixel_path), pixel_hash, label="current pixel-inventory hash"
    )

    contract_path = _resolve_recorded_path(
        manifest.get("target_variable_contract"),
        root=root,
        label="target variable contract",
    )
    contract_hash = _require_sha256(
        manifest.get("target_variable_contract_sha256"),
        label="manifest target_variable_contract_sha256",
    )
    if not contract_path.is_file():
        raise RuntimeError(f"CHIRPS target contract is missing: {contract_path}")
    _require_equal(
        _sha256_file(contract_path), contract_hash, label="current target-contract hash"
    )

    grid_hash = _require_sha256(
        target_validation.get("grid_hash"), label="target-validation grid_hash"
    )
    _require_equal(
        manifest.get("promoted_target_grid_hash_sha256"),
        grid_hash,
        label="target grid hash",
    )

    attrs_path = active_target / ".zattrs"
    attrs = _load_mapping(attrs_path, label="CHIRPS root Zarr attributes")
    if attrs.get("deep_validation_passed") is not True:
        raise RuntimeError("CHIRPS target lacks deep-validation stamp.")
    for key, expected in (
        ("build_id", build_id),
        ("block_signature_sha256", signature),
        ("target_contract_version", TARGET_CONTRACT_VERSION),
        ("grid_hash_sha256", grid_hash),
    ):
        _require_equal(attrs.get(key), expected, label=f"target attribute {key}")

    validation.update(
        {
            "validation_source": "deep_validation_receipt",
            "validation_receipt": receipt_relative,
            "target_data_state_sha256": state_hash,
            "build_manifest_sha256": receipt_manifest_hash,
        }
    )
    return validation

