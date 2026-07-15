from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from scripts import chirps_validation_receipt as receipt_validation
from scripts import quarantine_legacy_phase4_outputs as legacy
from scripts import quarantine_superseded_chirps as superseded
from scripts.build_phase4_chirps_targets import _zarr_state_fingerprint


SIGNATURE = "a" * 64
CONTENT = "b" * 64
GRID = "c" * 64
BUILD_ID = "F4TARGET_TEST_RECEIPT"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _validated_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    target = (
        tmp_path
        / "data/processed/zarr/features/chirps_native_weekly_targets.zarr"
    )
    chunk = target / "precip_weekly_mm/0.0.0"
    chunk.parent.mkdir(parents=True)
    chunk.write_bytes(b"native-chirps-data")
    _write_json(
        target / ".zattrs",
        {
            "build_id": BUILD_ID,
            "block_signature_sha256": SIGNATURE,
            "target_contract_version": "chirps-native-weekly-v4",
            "grid_hash_sha256": GRID,
            "deep_validation_passed": True,
        },
    )

    builder = tmp_path / "scripts/build_phase4_chirps_targets.py"
    module = tmp_path / "src/nino_brasil/targets/chirps_native.py"
    builder.parent.mkdir(parents=True)
    module.parent.mkdir(parents=True)
    builder.write_text("# frozen builder\n", encoding="utf-8")
    module.write_text("# frozen target contract\n", encoding="utf-8")

    pixels = tmp_path / "data/processed/parquet/features/native_pixels.csv"
    contract = tmp_path / "data/processed/parquet/statistics/target_contract.csv"
    pixels.parent.mkdir(parents=True)
    contract.parent.mkdir(parents=True)
    pixels.write_text("pixel_id,lat,lon\n0,-1,-40\n", encoding="utf-8")
    contract.write_text("variable,role\nprecip_weekly_mm,target\n", encoding="utf-8")

    state = _zarr_state_fingerprint(target)
    manifest_path = (
        tmp_path
        / "data/interim/chirps_weekly_native_blocks"
        / SIGNATURE[:16]
        / "manifest.json"
    )
    manifest = {
        "build_complete": True,
        "promotion_status": "promoted_after_deep_validation",
        "target_contract_version": "chirps-native-weekly-v4",
        "build_id": BUILD_ID,
        "signature_sha256": SIGNATURE,
        "promoted_target": target.relative_to(tmp_path).as_posix(),
        "promoted_target_data_state_sha256": state,
        "promoted_target_data_content_sha256": CONTENT,
        "promoted_target_grid_hash_sha256": GRID,
        "promoted_pixel_inventory": pixels.relative_to(tmp_path).as_posix(),
        "promoted_pixel_inventory_sha256": receipt_validation._sha256_file(pixels),
        "target_variable_contract": contract.relative_to(tmp_path).as_posix(),
        "target_variable_contract_sha256": receipt_validation._sha256_file(contract),
        "builder_script_sha256": receipt_validation._sha256_file(builder),
        "target_module_sha256": receipt_validation._sha256_file(module),
    }
    _write_json(manifest_path, manifest)

    receipt = tmp_path / "data/audit/chirps_deep_validation.json"
    _write_json(
        receipt,
        {
            "schema_version": "nino26-chirps-deep-validation/v1",
            "target_path": target.relative_to(tmp_path).as_posix(),
            "target_data_state_sha256": state,
            "builder_script_sha256": receipt_validation._sha256_file(builder),
            "target_module_sha256": receipt_validation._sha256_file(module),
            "build_manifest_sha256": receipt_validation._sha256_file(manifest_path),
            "validation": {
                "valid": True,
                "problems": [],
                "build_id": BUILD_ID,
                "block_signature_sha256": SIGNATURE,
                "target_data_content_sha256": CONTENT,
                "manifest": manifest_path.relative_to(tmp_path).as_posix(),
                "pixel_inventory": pixels.relative_to(tmp_path).as_posix(),
                "target_validation": {
                    "valid": True,
                    "errors": [],
                    "grid_hash": GRID,
                },
            },
        },
    )
    return target, receipt, chunk


def _configure_superseded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: Path
) -> Path:
    features = tmp_path / "data/processed/zarr/features"
    blocks = tmp_path / "data/interim/chirps_weekly_native_blocks"
    candidate = features / "chirps_native_weekly_targets.zarr.staging-old"
    candidate.mkdir(parents=True)
    (candidate / "chunk.bin").write_bytes(b"old")
    monkeypatch.setattr(superseded, "ROOT", tmp_path)
    monkeypatch.setattr(superseded, "ACTIVE_TARGET", target)
    monkeypatch.setattr(superseded, "ALLOWED_SOURCE_ROOTS", (features, blocks))
    monkeypatch.setattr(superseded, "CANDIDATES", (candidate,))
    monkeypatch.setattr(
        superseded,
        "QUARANTINE",
        tmp_path / "data/quarantine/phase4_chirps_superseded",
    )
    return candidate


def _configure_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: Path
) -> Path:
    statistics = tmp_path / "data/processed/parquet/statistics"
    zarr_statistics = tmp_path / "data/processed/zarr/statistics"
    statistics.mkdir(parents=True, exist_ok=True)
    zarr_statistics.mkdir(parents=True, exist_ok=True)
    candidate = statistics / "phase4C_old.csv"
    candidate.write_text("legacy", encoding="utf-8")
    monkeypatch.setattr(legacy, "ROOT", tmp_path)
    monkeypatch.setattr(legacy, "STATISTICS", statistics)
    monkeypatch.setattr(legacy, "ZARR_STATISTICS", zarr_statistics)
    monkeypatch.setattr(
        legacy, "LEGACY_ATLAS", zarr_statistics / "phase4C_atlas_pixel.zarr"
    )
    monkeypatch.setattr(legacy, "ACTIVE_TARGET", target)
    monkeypatch.setattr(
        legacy, "QUARANTINE", tmp_path / "data/quarantine/stale_derived"
    )
    return candidate


def test_superseded_cli_reuses_valid_receipt_without_deep_rehash(
    tmp_path, monkeypatch
):
    target, receipt, _ = _validated_fixture(tmp_path)
    candidate = _configure_superseded(tmp_path, monkeypatch, target)
    monkeypatch.setattr(
        superseded,
        "_validated_active_target",
        lambda: (_ for _ in ()).throw(AssertionError("deep validator was called")),
    )

    assert superseded.main(["--apply", "--validation-receipt", str(receipt)]) == 0

    assert not candidate.exists()
    manifest = next(superseded.QUARANTINE.glob("*/manifest.json"))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["active_target_validation"]["validation_source"] == (
        "deep_validation_receipt"
    )
    assert payload["post_move_active_target_validation"][
        "target_data_state_sha256"
    ] == payload["active_target_validation"]["target_data_state_sha256"]


def test_legacy_cli_reuses_valid_receipt_without_deep_rehash(tmp_path, monkeypatch):
    target, receipt, _ = _validated_fixture(tmp_path)
    candidate = _configure_legacy(tmp_path, monkeypatch, target)
    monkeypatch.setattr(
        legacy,
        "_validated_active_target",
        lambda: (_ for _ in ()).throw(AssertionError("deep validator was called")),
    )

    assert legacy.main(["--apply", "--validation-receipt", str(receipt)]) == 0

    assert not candidate.exists()
    manifest = next(legacy.QUARANTINE.glob("*/manifest.json"))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["active_target_validation"]["validation_source"] == (
        "deep_validation_receipt"
    )


def test_receipt_rejects_stale_target_state_before_move(tmp_path, monkeypatch):
    target, receipt, chunk = _validated_fixture(tmp_path)
    candidate = _configure_legacy(tmp_path, monkeypatch, target)
    chunk.write_bytes(b"changed-after-deep-validation")

    with pytest.raises(RuntimeError, match="current target state"):
        legacy.main(["--apply", "--validation-receipt", str(receipt)])

    assert candidate.exists()
    assert not legacy.QUARANTINE.exists()


@pytest.mark.parametrize("changed", ["builder", "manifest"])
def test_receipt_rejects_current_code_or_manifest_mismatch(
    tmp_path, monkeypatch, changed
):
    target, receipt, _ = _validated_fixture(tmp_path)
    candidate = _configure_superseded(tmp_path, monkeypatch, target)
    if changed == "builder":
        (tmp_path / "scripts/build_phase4_chirps_targets.py").write_text(
            "# code changed\n", encoding="utf-8"
        )
        expected = "current builder hash"
    else:
        manifest = (
            tmp_path
            / "data/interim/chirps_weekly_native_blocks"
            / SIGNATURE[:16]
            / "manifest.json"
        )
        manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        expected = "current build manifest hash"

    with pytest.raises(RuntimeError, match=expected):
        superseded.main(["--apply", "--validation-receipt", str(receipt)])

    assert candidate.exists()


def test_receipt_revalidates_state_after_moves_and_fails_closed(
    tmp_path, monkeypatch
):
    target, receipt, chunk = _validated_fixture(tmp_path)
    _configure_legacy(tmp_path, monkeypatch, target)
    original_move = shutil.move

    def move_then_change_target(source: str, destination: str):
        result = original_move(source, destination)
        chunk.write_bytes(b"target-mutated-during-cleanup")
        return result

    monkeypatch.setattr(legacy.shutil, "move", move_then_change_target)

    with pytest.raises(RuntimeError, match="current target state"):
        legacy.main(["--apply", "--validation-receipt", str(receipt)])

    manifest = next(legacy.QUARANTINE.glob("*/manifest.json"))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "post_move_validation_failed"
    assert "current target state" in payload["post_move_validation_error"]

