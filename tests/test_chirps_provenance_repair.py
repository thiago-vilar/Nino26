from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import repair_chirps_builder_provenance as repair


def _configured_repair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    builder = tmp_path / "scripts/build_phase4_chirps_targets.py"
    builder.parent.mkdir(parents=True)
    builder.write_text("# float32 fingerprint fix\n", encoding="utf-8")
    target = tmp_path / "target.zarr"
    target.mkdir()
    module = tmp_path / "src/nino_brasil/targets/chirps_native.py"
    module.parent.mkdir(parents=True)
    module.write_text("# target contract\n", encoding="utf-8")
    pixels_path = tmp_path / "pixels.csv"
    pixels = pd.DataFrame(
        {
            "pixel_id": [0],
            "grid_row": [0],
            "grid_column": [0],
            "lat": pd.Series([-10.125], dtype="float32"),
            "lon": pd.Series([-50.125], dtype="float32"),
            "brazil_fraction": pd.Series([0.12345679], dtype="float32"),
            "brazil_center": [True],
            "grid_hash": ["g" * 64],
        }
    )
    pixels.to_csv(pixels_path, index=False)
    contract_path = tmp_path / "contract.csv"
    contract_path.write_text("variable\nprecip_weekly_mm\n", encoding="utf-8")
    manifest_path = tmp_path / "blocks/signature/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    old_hash = "a" * 64
    manifest = {
        "builder_script_sha256": old_hash,
        "build_id": "F4TARGET_test",
        "signature_sha256": "b" * 64,
        "target_contract_version": "chirps-native-weekly-v4",
        "promoted_target": "target.zarr",
        "promoted_target_data_state_sha256": repair._zarr_state_fingerprint(target),
        "promoted_target_data_content_sha256": "d" * 64,
        "promoted_pixel_inventory": "pixels.csv",
        "promoted_pixel_inventory_sha256": repair.sha256_file(pixels_path),
        "pixel_mask_sha256": repair._pixel_mask_fingerprint(pixels),
        "target_variable_contract": "contract.csv",
        "target_variable_contract_sha256": repair.sha256_file(contract_path),
        "target_module_sha256": repair.sha256_file(module),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    new_hash = hashlib.sha256(builder.read_bytes()).hexdigest()
    prior_receipt_path = tmp_path / "data/audit/chirps_deep_validation.json"
    prior_receipt_path.parent.mkdir(parents=True)
    prior_receipt = {
        "schema_version": "nino26-chirps-deep-validation/v1",
        "checked_at_utc": "2026-07-13T00:00:00+00:00",
        "target_path": "target.zarr",
        "target_data_state_sha256": repair._zarr_state_fingerprint(target),
        "builder_script_sha256": new_hash,
        "target_module_sha256": repair.sha256_file(module),
        "build_manifest_sha256": repair.sha256_file(manifest_path),
        "validation": {
            "valid": False,
            "problems": ["builder_script_fingerprint_mismatch"],
            "manifest": manifest_path.relative_to(tmp_path).as_posix(),
            "target_data_content_sha256": "d" * 64,
            "target_validation": {"valid": True},
            "build_id": "F4TARGET_test",
            "block_signature_sha256": "b" * 64,
            "pixel_inventory": "pixels.csv",
        },
    }
    prior_receipt_path.write_text(
        json.dumps(prior_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    monkeypatch.setattr(repair, "ROOT", tmp_path)
    monkeypatch.setattr(repair, "BUILDER", builder)
    monkeypatch.setattr(repair, "OUTPUT", target)
    monkeypatch.setattr(repair, "RECEIPT_ROOT", tmp_path / "data/audit/provenance_repairs")
    monkeypatch.setattr(repair, "DEFAULT_VALIDATION_RECEIPT", prior_receipt_path)

    def validate(_target):
        active = json.loads(manifest_path.read_text(encoding="utf-8"))
        current = active["builder_script_sha256"] == new_hash
        return {
            "valid": current,
            "problems": [] if current else ["builder_script_fingerprint_mismatch"],
            "manifest": manifest_path.relative_to(tmp_path).as_posix(),
            "target_data_content_sha256": "d" * 64,
        }

    monkeypatch.setattr(repair, "validate_promoted_target", validate)
    return builder, manifest_path, prior_receipt_path, old_hash, new_hash


def test_chirps_provenance_repair_is_dry_run_by_default(tmp_path, monkeypatch):
    _, manifest_path, prior_receipt_path, old_hash, _ = _configured_repair(
        tmp_path, monkeypatch
    )
    before = manifest_path.read_bytes()

    result = repair.repair(
        expected_old_builder_sha256=old_hash,
        prior_validation_receipt=prior_receipt_path,
        apply=False,
    )

    assert result["status"] == "planned"
    assert result["target_inventory_contract_bytes_rewritten"] is False
    assert manifest_path.read_bytes() == before
    assert not repair.RECEIPT_ROOT.exists()


def test_chirps_provenance_repair_is_receipted_atomic_and_validated(
    tmp_path, monkeypatch
):
    _, manifest_path, prior_receipt_path, old_hash, new_hash = _configured_repair(
        tmp_path, monkeypatch
    )

    result = repair.repair(
        expected_old_builder_sha256=old_hash,
        prior_validation_receipt=prior_receipt_path,
        apply=True,
    )

    active = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt_path = tmp_path / result["receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert active["builder_script_sha256"] == new_hash
    assert active["provenance_repairs"][-1]["old_builder_script_sha256"] == old_hash
    assert receipt["status"] == "applied_and_validated"
    assert receipt["validation_after"]["valid"] is True
    assert receipt["post_repair_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    refreshed = json.loads(prior_receipt_path.read_text(encoding="utf-8"))
    assert refreshed["validation"]["valid"] is True
    assert refreshed["build_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()


def test_chirps_provenance_repair_refuses_unexpected_old_hash(tmp_path, monkeypatch):
    _, manifest_path, prior_receipt_path, _, _ = _configured_repair(
        tmp_path, monkeypatch
    )
    before = manifest_path.read_bytes()

    with pytest.raises(RuntimeError, match="explicitly expected old hash"):
        repair.repair(
            expected_old_builder_sha256="9" * 64,
            prior_validation_receipt=prior_receipt_path,
            apply=True,
        )

    assert manifest_path.read_bytes() == before
    assert not repair.RECEIPT_ROOT.exists()


def test_chirps_provenance_repair_accepts_historical_mask_mismatch_receipt(
    tmp_path, monkeypatch
):
    _, manifest_path, prior_receipt_path, old_hash, _ = _configured_repair(
        tmp_path, monkeypatch
    )
    prior = json.loads(prior_receipt_path.read_text(encoding="utf-8"))
    prior["builder_script_sha256"] = old_hash
    prior["validation"]["problems"] = [
        "promoted_pixel_mask_fingerprint_mismatch"
    ]
    prior_receipt_path.write_text(
        json.dumps(prior, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result = repair.repair(
        expected_old_builder_sha256=old_hash,
        prior_validation_receipt=prior_receipt_path,
        apply=False,
    )

    assert result["status"] == "planned"
    assert result["validation_before"]["problems"] == [
        "builder_script_fingerprint_mismatch"
    ]
    assert result["validation_before"]["prior_receipt_problems"] == [
        "promoted_pixel_mask_fingerprint_mismatch"
    ]
    assert json.loads(manifest_path.read_text(encoding="utf-8"))[
        "builder_script_sha256"
    ] == old_hash
