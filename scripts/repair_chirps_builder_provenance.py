#!/usr/bin/env python3
"""Repair only the CHIRPS builder hash after a provenance-only code fix.

This utility never rebuilds or rewrites the promoted Zarr target, pixel
inventory or numeric contract.  It reuses a byte-level validation receipt tied
to the unchanged manifest/Zarr state, then cheaply rechecks every mutable
fingerprint.  The old builder hash must be supplied explicitly.  Default mode
is read-only; ``--apply`` writes a predeclared audit receipt and atomically
replaces the build manifest, then performs one full validation and rolls back
if it does not pass.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for _import_root in (ROOT, ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from nino_brasil.artifacts import sha256_file  # noqa: E402
from scripts.build_phase4_chirps_targets import (  # noqa: E402
    OUTPUT,
    _pixel_mask_fingerprint,
    _zarr_state_fingerprint,
    validate_promoted_target,
)


BUILDER = ROOT / "scripts/build_phase4_chirps_targets.py"
RECEIPT_ROOT = ROOT / "data/audit/provenance_repairs"
DEFAULT_VALIDATION_RECEIPT = ROOT / "data/audit/chirps_deep_validation.json"
REASON_CODE = "pixel_mask_fingerprint_float32_csv_roundtrip_v1"
EXPECTED_PRE_REPAIR_PROBLEMS = ["builder_script_fingerprint_mismatch"]
ACCEPTED_PRIOR_RECEIPT_PROBLEMS = {
    ("builder_script_fingerprint_mismatch",),
    ("promoted_pixel_mask_fingerprint_mismatch",),
}


def _inside_workspace(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return False
    return True


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    if not _inside_workspace(path):
        raise ValueError(f"Refusing provenance write outside workspace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _recorded_path(value: object, *, label: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise RuntimeError(f"Missing recorded path: {label}.")
    candidate = (Path(raw) if Path(raw).is_absolute() else ROOT / raw).resolve()
    if not _inside_workspace(candidate):
        raise RuntimeError(f"Recorded path escapes workspace ({label}): {candidate}")
    return candidate


def _manifest_path(validation: dict[str, object]) -> Path:
    value = str(validation.get("manifest") or "").strip()
    if not value:
        raise RuntimeError("Promoted-target validation did not identify its manifest.")
    candidate = _recorded_path(value, label="validation.manifest")
    if not candidate.is_file():
        raise RuntimeError(f"Invalid promoted-target manifest path: {candidate}")
    return candidate


def _preflight_from_validation_receipt(
    receipt_path: Path,
) -> tuple[dict[str, object], Path, bytes, dict[str, object], dict[str, object]]:
    """Reuse deep bytes only when current cheap invariants prove immutability."""

    path = receipt_path.resolve()
    if not _inside_workspace(path) or not path.is_file():
        raise RuntimeError(f"Invalid prior CHIRPS validation receipt: {path}")
    receipt_bytes = path.read_bytes()
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    validation = dict(receipt.get("validation") or {})
    prior_problems = tuple(validation.get("problems") or [])
    if validation.get("valid") is True or prior_problems not in (
        ACCEPTED_PRIOR_RECEIPT_PROBLEMS
    ):
        raise RuntimeError(
            "Prior deep receipt must have exactly the historical pixel-mask or "
            "current builder hash mismatch; got "
            f"{validation.get('problems')}."
        )
    if not bool((validation.get("target_validation") or {}).get("valid")):
        raise RuntimeError("Prior receipt did not pass the scientific target validator.")

    manifest_path = _manifest_path(validation)
    original_bytes = manifest_path.read_bytes()
    manifest = json.loads(original_bytes.decode("utf-8"))
    checks: dict[str, bool] = {
        "receipt_manifest_sha_matches": str(receipt.get("build_manifest_sha256"))
        == hashlib.sha256(original_bytes).hexdigest(),
        "receipt_target_path_matches": _recorded_path(
            receipt.get("target_path"), label="receipt.target_path"
        )
        == OUTPUT.resolve(),
        "target_state_matches_receipt": _zarr_state_fingerprint(OUTPUT)
        == str(receipt.get("target_data_state_sha256")),
        "target_state_matches_manifest": str(receipt.get("target_data_state_sha256"))
        == str(manifest.get("promoted_target_data_state_sha256")),
        "prior_content_matches_manifest": str(
            validation.get("target_data_content_sha256")
        )
        == str(manifest.get("promoted_target_data_content_sha256")),
        "build_id_matches": str(validation.get("build_id"))
        == str(manifest.get("build_id")),
        "block_signature_matches": str(validation.get("block_signature_sha256"))
        == str(manifest.get("signature_sha256")),
    }
    builder_hash = sha256_file(BUILDER)
    module_path = ROOT / "src/nino_brasil/targets/chirps_native.py"
    module_hash = sha256_file(module_path)
    checks.update(
        {
            "receipt_builder_matches_problem_epoch": (
                str(receipt.get("builder_script_sha256")).lower()
                == (
                    str(manifest.get("builder_script_sha256")).lower()
                    if prior_problems
                    == ("promoted_pixel_mask_fingerprint_mismatch",)
                    else builder_hash.lower()
                )
            ),
            "receipt_module_matches_current": str(
                receipt.get("target_module_sha256")
            ).lower()
            == module_hash.lower(),
            "manifest_module_matches_current": str(
                manifest.get("target_module_sha256")
            ).lower()
            == module_hash.lower(),
        }
    )
    target_path = _recorded_path(manifest.get("promoted_target"), label="promoted_target")
    pixel_path = _recorded_path(
        manifest.get("promoted_pixel_inventory"), label="promoted_pixel_inventory"
    )
    contract_path = _recorded_path(
        manifest.get("target_variable_contract"), label="target_variable_contract"
    )
    if not target_path.is_dir() or not pixel_path.is_file() or not contract_path.is_file():
        raise RuntimeError("A promoted CHIRPS target/inventory/contract path is missing.")
    pixels = pd.read_csv(pixel_path)
    checks.update(
        {
            "manifest_target_path_matches": target_path == OUTPUT.resolve(),
            "pixel_path_matches_prior_receipt": str(validation.get("pixel_inventory"))
            == pixel_path.relative_to(ROOT).as_posix(),
            "pixel_file_sha_matches": sha256_file(pixel_path)
            == str(manifest.get("promoted_pixel_inventory_sha256")),
            "pixel_logical_mask_matches": _pixel_mask_fingerprint(pixels)
            == str(manifest.get("pixel_mask_sha256")),
            "numeric_contract_sha_matches": sha256_file(contract_path)
            == str(manifest.get("target_variable_contract_sha256")),
        }
    )
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            "Prior deep receipt is not reusable; current invariants failed: "
            f"{failed}."
        )
    validation_before = {
        **validation,
        "valid": False,
        "problems": EXPECTED_PRE_REPAIR_PROBLEMS,
        "mode": "prior_deep_receipt_plus_current_cheap_invariants",
        "prior_validation_receipt": path.relative_to(ROOT).as_posix(),
        "prior_receipt_problems": list(prior_problems),
        "current_cheap_invariants": checks,
    }
    return validation_before, manifest_path, original_bytes, manifest, receipt


def _immutable_snapshot(manifest: dict[str, object]) -> dict[str, object]:
    """Fields whose bytes/identity the pre-repair validator already checked."""

    return {
        key: manifest.get(key)
        for key in (
            "build_id",
            "signature_sha256",
            "target_contract_version",
            "promoted_target",
            "promoted_target_data_state_sha256",
            "promoted_target_data_content_sha256",
            "promoted_pixel_inventory",
            "promoted_pixel_inventory_sha256",
            "pixel_mask_sha256",
            "target_variable_contract",
            "target_variable_contract_sha256",
            "target_module_sha256",
        )
    }


def repair(
    *,
    expected_old_builder_sha256: str,
    prior_validation_receipt: Path = DEFAULT_VALIDATION_RECEIPT,
    apply: bool,
) -> dict[str, object]:
    expected_old = expected_old_builder_sha256.strip().lower()
    if len(expected_old) != 64 or any(char not in "0123456789abcdef" for char in expected_old):
        raise ValueError("--expected-old-builder-sha256 must be 64 lowercase hex characters.")

    (
        validation_before,
        manifest_path,
        original_bytes,
        manifest,
        prior_receipt,
    ) = _preflight_from_validation_receipt(prior_validation_receipt)
    prior_receipt_path = prior_validation_receipt.resolve()
    prior_receipt_original_bytes = prior_receipt_path.read_bytes()
    old_builder = str(manifest.get("builder_script_sha256") or "").lower()
    new_builder = sha256_file(BUILDER).lower()

    if old_builder != expected_old:
        raise RuntimeError(
            "Manifest builder hash does not equal the explicitly expected old hash: "
            f"manifest={old_builder} expected={expected_old}."
        )
    if new_builder == old_builder:
        raise RuntimeError("Builder bytes did not change; no provenance repair is justified.")
    problems = list(validation_before.get("problems") or [])
    if validation_before.get("valid") is True or problems != EXPECTED_PRE_REPAIR_PROBLEMS:
        raise RuntimeError(
            "Refusing provenance repair because validation has problems beyond the "
            f"expected builder hash mismatch: {problems}."
        )

    now = datetime.now(timezone.utc)
    pre_manifest_sha = hashlib.sha256(original_bytes).hexdigest()
    repair_id = hashlib.sha256(
        f"{pre_manifest_sha}|{old_builder}|{new_builder}|{REASON_CODE}".encode("ascii")
    ).hexdigest()[:16]
    receipt_path = RECEIPT_ROOT / f"chirps_builder_{repair_id}.json"
    repair_entry: dict[str, object] = {
        "schema_version": "nino26-chirps-provenance-repair/v1",
        "repair_id": repair_id,
        "applied_at_utc": now.isoformat(),
        "reason_code": REASON_CODE,
        "old_builder_script_sha256": old_builder,
        "new_builder_script_sha256": new_builder,
        "pre_repair_manifest_sha256": pre_manifest_sha,
        "receipt": receipt_path.relative_to(ROOT).as_posix(),
        "immutable_artifacts_verified_unchanged": _immutable_snapshot(manifest),
    }
    updated_manifest = dict(manifest)
    previous_repairs = list(updated_manifest.get("provenance_repairs") or [])
    updated_manifest["provenance_repairs"] = [*previous_repairs, repair_entry]
    updated_manifest["builder_script_sha256"] = new_builder
    updated_bytes = _json_bytes(updated_manifest)
    post_manifest_sha = hashlib.sha256(updated_bytes).hexdigest()
    receipt: dict[str, object] = {
        **repair_entry,
        "status": "planned" if not apply else "prepared",
        "apply": apply,
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "post_repair_manifest_sha256": post_manifest_sha,
        "validation_before": validation_before,
        "prior_validation_receipt_sha256": hashlib.sha256(
            prior_receipt_original_bytes
        ).hexdigest(),
        "target_inventory_contract_bytes_rewritten": False,
    }
    if not apply:
        return receipt

    _atomic_write_bytes(receipt_path, _json_bytes(receipt))
    try:
        _atomic_write_bytes(manifest_path, updated_bytes)
        validation_after = validate_promoted_target(OUTPUT)
        if validation_after.get("valid") is not True:
            raise RuntimeError(
                "Full CHIRPS validation failed after manifest repair: "
                f"{validation_after.get('problems')}."
            )
        refreshed_deep_receipt = {
            "schema_version": "nino26-chirps-deep-validation/v1",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_path": OUTPUT.relative_to(ROOT).as_posix(),
            "target_data_state_sha256": _zarr_state_fingerprint(OUTPUT),
            "builder_script_sha256": sha256_file(BUILDER),
            "target_module_sha256": sha256_file(
                ROOT / "src/nino_brasil/targets/chirps_native.py"
            ),
            "build_manifest_sha256": sha256_file(manifest_path),
            "validation": validation_after,
            "refresh_source": receipt_path.relative_to(ROOT).as_posix(),
        }
        _atomic_write_bytes(prior_receipt_path, _json_bytes(refreshed_deep_receipt))
    except Exception as exc:
        _atomic_write_bytes(manifest_path, original_bytes)
        _atomic_write_bytes(prior_receipt_path, prior_receipt_original_bytes)
        receipt.update(
            {
                "status": "rolled_back",
                "error": f"{type(exc).__name__}: {exc}",
                "active_manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
            }
        )
        _atomic_write_bytes(receipt_path, _json_bytes(receipt))
        raise

    receipt.update(
        {
            "status": "applied_and_validated",
            "validation_after": validation_after,
            "refreshed_deep_validation_receipt": prior_receipt_path.relative_to(
                ROOT
            ).as_posix(),
            "refreshed_deep_validation_receipt_sha256": sha256_file(
                prior_receipt_path
            ),
            "active_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
        }
    )
    _atomic_write_bytes(receipt_path, _json_bytes(receipt))
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-old-builder-sha256", required=True)
    parser.add_argument(
        "--prior-validation-receipt",
        type=Path,
        default=DEFAULT_VALIDATION_RECEIPT,
        help=(
            "deep validation receipt tied to the unchanged target/manifest; "
            "default: data/audit/chirps_deep_validation.json"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the receipt and atomically repair the manifest; default is dry-run",
    )
    args = parser.parse_args(argv)
    result = repair(
        expected_old_builder_sha256=args.expected_old_builder_sha256,
        prior_validation_receipt=(
            args.prior_validation_receipt
            if args.prior_validation_receipt.is_absolute()
            else ROOT / args.prior_validation_receipt
        ),
        apply=args.apply,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
