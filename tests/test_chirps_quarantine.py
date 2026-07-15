from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import quarantine_superseded_chirps as quarantine


def _configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    processed = tmp_path / "data/processed/zarr/features"
    blocks = tmp_path / "data/interim/chirps_weekly_native_blocks"
    candidate = processed / "old.staging"
    candidate.mkdir(parents=True)
    (candidate / "chunk.bin").write_bytes(b"old-v2")
    monkeypatch.setattr(quarantine, "ROOT", tmp_path)
    monkeypatch.setattr(
        quarantine,
        "QUARANTINE",
        tmp_path / "data/quarantine/phase4_chirps_superseded",
    )
    monkeypatch.setattr(
        quarantine,
        "ACTIVE_TARGET",
        processed / "chirps_native_weekly_targets.zarr",
    )
    monkeypatch.setattr(quarantine, "ALLOWED_SOURCE_ROOTS", (processed, blocks))
    monkeypatch.setattr(quarantine, "CANDIDATES", (candidate,))
    monkeypatch.setattr(
        quarantine,
        "_validated_active_target",
        lambda: {"valid": True, "block_signature_sha256": "a" * 64},
    )
    return candidate


def test_chirps_quarantine_default_is_read_only(tmp_path, monkeypatch):
    candidate = _configure(tmp_path, monkeypatch)

    assert quarantine.main([]) == 0
    assert candidate.exists()
    assert not quarantine.QUARANTINE.exists()


def test_chirps_quarantine_moves_only_after_validated_target(tmp_path, monkeypatch):
    candidate = _configure(tmp_path, monkeypatch)

    assert quarantine.main(["--apply"]) == 0
    assert not candidate.exists()
    manifests = list(quarantine.QUARANTINE.glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["records"][0]["status"] == "moved"
    moved = tmp_path / manifest["records"][0]["destination"]
    assert (moved / "chunk.bin").read_bytes() == b"old-v2"
    assert manifest["post_move_active_target_validation"]["valid"] is True


def test_chirps_quarantine_refuses_unvalidated_target(monkeypatch):
    monkeypatch.setattr(
        quarantine,
        "_validated_active_target",
        lambda: (_ for _ in ()).throw(RuntimeError("target invalid")),
    )

    with pytest.raises(RuntimeError, match="target invalid"):
        quarantine.main([])


def test_chirps_quarantine_discovers_stagings_and_excludes_active_block(
    tmp_path, monkeypatch
):
    processed = tmp_path / "data/processed/zarr/features"
    blocks = tmp_path / "data/interim/chirps_weekly_native_blocks"
    staging = processed / "chirps_native_weekly_targets.zarr.staging-deadbeef"
    inactive = blocks / "inactive"
    active = blocks / ("a" * 16)
    unrelated = processed / "chirps_native_daily_brazil_box.zarr"
    for path in (staging, inactive, active, unrelated):
        path.mkdir(parents=True)
    (inactive / "manifest.json").write_text("{}", encoding="utf-8")
    (active / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(quarantine, "ALLOWED_SOURCE_ROOTS", (processed, blocks))
    monkeypatch.setattr(quarantine, "CANDIDATES", ())

    candidates = quarantine._candidate_paths(active)

    assert set(candidates) == {staging, inactive}
    assert unrelated not in candidates


def _write_inventory_receipt(tmp_path: Path) -> Path:
    receipt = tmp_path / "data/audit/chirps_superseded_dry_run.json"
    assert quarantine.main(["--report-json", str(receipt)]) == 0
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["schema"] == quarantine.INVENTORY_SCHEMA
    assert payload["apply"] is False
    assert len(payload["records"]) == 1
    assert payload["records"][0]["tree_state_schema"] == (
        quarantine.TREE_STATE_SCHEMA
    )
    assert len(payload["records"][0]["tree_state_sha256"]) == 64
    return receipt


def _write_v1_inventory_receipt(tmp_path: Path, candidate: Path) -> Path:
    state = quarantine._tree_state(candidate)
    legacy_state = {
        key: state[key]
        for key in (
            "kind",
            "file_count",
            "size_bytes",
            "content_sha256",
            "last_modified_utc",
        )
    }
    receipt = tmp_path / "data/audit/chirps_superseded_dry_run_v1.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(
            {
                "schema": "nino26-chirps-superseded-quarantine-v1",
                "apply": False,
                "active_target_validation": quarantine._validated_active_target(),
                "records": [
                    {
                        "source": candidate.relative_to(tmp_path).as_posix(),
                        "destination": "data/quarantine/old/unused",
                        **legacy_state,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return receipt


def test_inventory_receipt_reuses_dry_run_content_hash_without_reading_tree_again(
    tmp_path, monkeypatch
):
    candidate = _configure(tmp_path, monkeypatch)
    receipt = _write_inventory_receipt(tmp_path)
    dry_run = json.loads(receipt.read_text(encoding="utf-8"))
    expected_hash = dry_run["records"][0]["content_sha256"]
    monkeypatch.setattr(
        quarantine,
        "_tree_state",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("candidate content was hashed again")
        ),
    )

    assert quarantine.main(["--apply", "--inventory-receipt", str(receipt)]) == 0

    assert not candidate.exists()
    manifest = next(quarantine.QUARANTINE.glob("*/manifest.json"))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["records"][0]["content_sha256"] == expected_hash
    assert payload["inventory_receipt"]["path"] == receipt.relative_to(
        tmp_path
    ).as_posix()
    assert len(payload["inventory_receipt"]["sha256"]) == 64


def test_inventory_receipt_rejects_stale_tree_state_before_move(
    tmp_path, monkeypatch
):
    candidate = _configure(tmp_path, monkeypatch)
    receipt = _write_inventory_receipt(tmp_path)
    chunk = candidate / "chunk.bin"
    stat = chunk.stat()
    os.utime(chunk, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    with pytest.raises(RuntimeError, match="candidate state changed"):
        quarantine.main(["--apply", "--inventory-receipt", str(receipt)])

    assert candidate.exists()
    assert not quarantine.QUARANTINE.exists()


def test_inventory_receipt_rejects_changed_candidate_list(tmp_path, monkeypatch):
    candidate = _configure(tmp_path, monkeypatch)
    receipt = _write_inventory_receipt(tmp_path)
    added = candidate.parent / "chirps_native_weekly_targets.zarr.staging-added"
    added.mkdir()
    (added / "chunk.bin").write_bytes(b"new stale staging")
    monkeypatch.setattr(quarantine, "CANDIDATES", (candidate, added))

    with pytest.raises(RuntimeError, match="candidate list changed"):
        quarantine.main(["--apply", "--inventory-receipt", str(receipt)])

    assert candidate.exists()
    assert added.exists()


def test_inventory_receipt_rejects_changed_chirps_identity(tmp_path, monkeypatch):
    candidate = _configure(tmp_path, monkeypatch)
    receipt = _write_inventory_receipt(tmp_path)
    monkeypatch.setattr(
        quarantine,
        "_validated_active_target",
        lambda: {"valid": True, "block_signature_sha256": "b" * 64},
    )

    with pytest.raises(RuntimeError, match="identity/state changed"):
        quarantine.main(["--apply", "--inventory-receipt", str(receipt)])

    assert candidate.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "obsolete-schema", "Unexpected.*schema"),
        ("apply", True, "apply=false"),
    ],
)
def test_inventory_receipt_requires_dry_run_schema(
    tmp_path, monkeypatch, field, value, message
):
    candidate = _configure(tmp_path, monkeypatch)
    receipt = _write_inventory_receipt(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload[field] = value
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        quarantine.main(["--apply", "--inventory-receipt", str(receipt)])

    assert candidate.exists()


def test_inventory_receipt_is_rechecked_immediately_before_move(
    tmp_path, monkeypatch
):
    candidate = _configure(tmp_path, monkeypatch)
    receipt = _write_inventory_receipt(tmp_path)
    original_load = quarantine._load_inventory_receipt

    def load_then_mutate(*args, **kwargs):
        result = original_load(*args, **kwargs)
        (candidate / "late.bin").write_bytes(b"late mutation")
        return result

    monkeypatch.setattr(quarantine, "_load_inventory_receipt", load_then_mutate)

    with pytest.raises(RuntimeError, match="changed after inventory"):
        quarantine.main(["--apply", "--inventory-receipt", str(receipt)])

    assert candidate.exists()


def test_v1_inventory_upgrade_preserves_content_hash_without_rehashing(
    tmp_path, monkeypatch
):
    candidate = _configure(tmp_path, monkeypatch)
    v1 = _write_v1_inventory_receipt(tmp_path, candidate)
    v1_payload = json.loads(v1.read_text(encoding="utf-8"))
    expected_hash = v1_payload["records"][0]["content_sha256"]
    v1_sha256 = quarantine.hashlib.sha256(v1.read_bytes()).hexdigest()
    v2 = tmp_path / "data/audit/chirps_superseded_dry_run_v2.json"
    monkeypatch.setattr(
        quarantine,
        "_tree_state",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("candidate bytes were hashed during v1 upgrade")
        ),
    )

    assert (
        quarantine.main(
            [
                "--upgrade-inventory-receipt",
                str(v1),
                "--report-json",
                str(v2),
            ]
        )
        == 0
    )

    payload = json.loads(v2.read_text(encoding="utf-8"))
    assert payload["schema"] == quarantine.INVENTORY_SCHEMA
    assert payload["apply"] is False
    assert payload["records"][0]["content_sha256"] == expected_hash
    assert payload["records"][0]["tree_state_schema"] == (
        quarantine.TREE_STATE_SCHEMA
    )
    assert payload["upgraded_from"] == {
        "path": v1.relative_to(tmp_path).as_posix(),
        "schema": "nino26-chirps-superseded-quarantine-v1",
        "sha256": v1_sha256,
    }
    assert candidate.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("file_count", 999),
        ("size_bytes", 999),
        ("last_modified_utc", "2000-01-01T00:00:00+00:00"),
    ],
)
def test_v1_inventory_upgrade_rejects_changed_legacy_state(
    tmp_path, monkeypatch, field, value
):
    candidate = _configure(tmp_path, monkeypatch)
    v1 = _write_v1_inventory_receipt(tmp_path, candidate)
    payload = json.loads(v1.read_text(encoding="utf-8"))
    payload["records"][0][field] = value
    v1.write_text(json.dumps(payload), encoding="utf-8")
    v2 = tmp_path / "data/audit/upgrade.json"

    with pytest.raises(RuntimeError, match="changed since v1 inventory"):
        quarantine.main(
            [
                "--upgrade-inventory-receipt",
                str(v1),
                "--report-json",
                str(v2),
            ]
        )

    assert candidate.exists()
    assert not v2.exists()


def test_v1_inventory_upgrade_rejects_links_during_stat_scan(tmp_path, monkeypatch):
    candidate = _configure(tmp_path, monkeypatch)
    v1 = _write_v1_inventory_receipt(tmp_path, candidate)
    v2 = tmp_path / "data/audit/upgrade.json"
    original = quarantine._is_link_or_junction
    monkeypatch.setattr(
        quarantine,
        "_is_link_or_junction",
        lambda path: path.name == "chunk.bin" or original(path),
    )

    with pytest.raises(ValueError, match="non-regular file"):
        quarantine.main(
            [
                "--upgrade-inventory-receipt",
                str(v1),
                "--report-json",
                str(v2),
            ]
        )

    assert candidate.exists()


def test_v1_inventory_upgrade_rejects_candidate_or_chirps_identity_change(
    tmp_path, monkeypatch
):
    candidate = _configure(tmp_path, monkeypatch)
    v1 = _write_v1_inventory_receipt(tmp_path, candidate)
    v2 = tmp_path / "data/audit/upgrade.json"
    monkeypatch.setattr(
        quarantine,
        "_validated_active_target",
        lambda: {"valid": True, "block_signature_sha256": "b" * 64},
    )

    with pytest.raises(RuntimeError, match="identity/state changed"):
        quarantine.main(
            [
                "--upgrade-inventory-receipt",
                str(v1),
                "--report-json",
                str(v2),
            ]
        )

    assert candidate.exists()


def test_v1_inventory_upgrade_is_dry_run_only_and_requires_new_report(
    tmp_path, monkeypatch
):
    candidate = _configure(tmp_path, monkeypatch)
    v1 = _write_v1_inventory_receipt(tmp_path, candidate)

    with pytest.raises(SystemExit):
        quarantine.main(["--upgrade-inventory-receipt", str(v1)])
    with pytest.raises(SystemExit):
        quarantine.main(
            [
                "--apply",
                "--upgrade-inventory-receipt",
                str(v1),
                "--report-json",
                str(tmp_path / "data/audit/v2.json"),
            ]
        )
    with pytest.raises(SystemExit):
        quarantine.main(
            [
                "--upgrade-inventory-receipt",
                str(v1),
                "--report-json",
                str(v1),
            ]
        )

    assert candidate.exists()
