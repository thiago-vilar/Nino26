from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import quarantine_legacy_phase4_outputs as quarantine


def _valid_target_report(*, content: str = "b" * 64) -> dict[str, object]:
    return {
        "valid": True,
        "problems": [],
        "build_id": "CHIRPS_NATIVE_20260713",
        "block_signature_sha256": "a" * 64,
        "target_data_content_sha256": content,
        "pixel_inventory": "data/processed/parquet/features/native_pixels.csv",
    }


def _configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    statistics = tmp_path / "data/processed/parquet/statistics"
    zarr_statistics = tmp_path / "data/processed/zarr/statistics"
    statistics.mkdir(parents=True)
    zarr_statistics.mkdir(parents=True)
    monkeypatch.setattr(quarantine, "ROOT", tmp_path)
    monkeypatch.setattr(quarantine, "STATISTICS", statistics)
    monkeypatch.setattr(quarantine, "ZARR_STATISTICS", zarr_statistics)
    monkeypatch.setattr(
        quarantine,
        "LEGACY_ATLAS",
        zarr_statistics / "phase4C_atlas_pixel.zarr",
    )
    monkeypatch.setattr(
        quarantine,
        "QUARANTINE",
        tmp_path / "data/quarantine/stale_derived",
    )
    monkeypatch.setattr(
        quarantine,
        "_validated_active_target",
        lambda: _valid_target_report(),
    )
    return statistics, zarr_statistics


def test_discovery_is_narrow_and_preserves_native_contracts_and_membership(
    tmp_path,
    monkeypatch,
):
    statistics, zarr_statistics = _configure(tmp_path, monkeypatch)
    expected_names = {
        "phase40_old.csv",
        "phase4A_old.csv",
        "phase4B_old.parquet",
        "phase4C_old.csv",
        "phase4D_old.csv",
    }
    for name in expected_names:
        (statistics / name).write_text(name, encoding="utf-8")
    protected = {
        "phase4C_native_best_lag_pixel.parquet",
        "phase4D_native_target_coverage.csv",
        "phase4C_target_contract.csv",
        "phase4D_contrato_metodo.csv",
        "phase4_chirps_native_brazil_membership.parquet",
        "phase4E_unrelated.csv",
    }
    for name in protected:
        (statistics / name).write_text("keep", encoding="utf-8")
    nested = statistics / "nested"
    nested.mkdir()
    (nested / "phase4C_nested.csv").write_text("keep", encoding="utf-8")
    (statistics / "phase4C_directory").mkdir()

    atlas = zarr_statistics / "phase4C_atlas_pixel.zarr"
    atlas.mkdir()
    (atlas / "chunk.bin").write_bytes(b"legacy-atlas")
    other_zarr = zarr_statistics / "phase4C_native_pixel_lags.zarr"
    other_zarr.mkdir()
    (other_zarr / "chunk.bin").write_bytes(b"native")

    candidates = quarantine.discover_candidates()
    discovered = {candidate.path.name for candidate in candidates}

    assert discovered == expected_names | {"phase4C_atlas_pixel.zarr"}
    assert not (discovered & protected)


def test_default_dry_run_does_not_move_or_create_quarantine(tmp_path, monkeypatch):
    statistics, _ = _configure(tmp_path, monkeypatch)
    candidate = statistics / "phase4C_best_lag_pixel.csv"
    candidate.write_bytes(b"legacy")

    assert quarantine.main([]) == 0

    assert candidate.read_bytes() == b"legacy"
    assert not quarantine.QUARANTINE.exists()


def test_dry_run_inventories_even_when_promoted_target_is_not_ready(
    tmp_path,
    monkeypatch,
    capsys,
):
    statistics, _ = _configure(tmp_path, monkeypatch)
    candidate = statistics / "phase4C_best_lag_pixel.csv"
    candidate.write_bytes(b"legacy")
    monkeypatch.setattr(
        quarantine,
        "_validated_active_target",
        lambda: (_ for _ in ()).throw(RuntimeError("CHIRPS v4 invalid")),
    )

    assert quarantine.main([]) == 0

    assert candidate.exists()
    assert "items=1 apply=False" in capsys.readouterr().out
    assert not quarantine.QUARANTINE.exists()


def test_dry_run_report_is_limited_to_audit_and_does_not_touch_sources(
    tmp_path,
    monkeypatch,
):
    statistics, _ = _configure(tmp_path, monkeypatch)
    candidate = statistics / "phase4C_best_lag_pixel.csv"
    candidate.write_bytes(b"legacy")
    report = tmp_path / "data/audit/phase4_legacy_inventory.json"

    assert quarantine.main(["--report-json", str(report)]) == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "dry_run_complete"
    assert payload["records"][0]["content_sha256"]
    assert candidate.read_bytes() == b"legacy"
    assert not quarantine.QUARANTINE.exists()


def test_apply_moves_with_hash_manifest_and_revalidates_target(tmp_path, monkeypatch):
    statistics, zarr_statistics = _configure(tmp_path, monkeypatch)
    table = statistics / "phase4D_cluster_ranking.csv"
    table.write_bytes(b"legacy-table")
    atlas = zarr_statistics / "phase4C_atlas_pixel.zarr"
    atlas.mkdir()
    (atlas / "chunk.bin").write_bytes(b"legacy-atlas")
    validations: list[dict[str, object]] = []

    def validate() -> dict[str, object]:
        report = _valid_target_report()
        validations.append(report)
        return report

    monkeypatch.setattr(quarantine, "_validated_active_target", validate)

    assert quarantine.main(["--apply"]) == 0

    assert len(validations) == 2
    assert not table.exists()
    assert not atlas.exists()
    manifests = list(quarantine.QUARANTINE.glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["post_move_active_target_validation"]["valid"] is True
    assert len(manifest["records"]) == 2
    for record in manifest["records"]:
        assert record["status"] == "moved"
        assert len(record["content_sha256"]) == 64
        assert record["destination_state"]["content_sha256"] == record["content_sha256"]
        moved = tmp_path / record["destination"]
        assert moved.exists()


def test_apply_refuses_when_promoted_target_is_invalid(tmp_path, monkeypatch):
    statistics, _ = _configure(tmp_path, monkeypatch)
    candidate = statistics / "phase4A_eventos.csv"
    candidate.write_text("legacy", encoding="utf-8")
    monkeypatch.setattr(
        quarantine,
        "_validated_active_target",
        lambda: (_ for _ in ()).throw(RuntimeError("CHIRPS v4 invalid")),
    )

    with pytest.raises(RuntimeError, match="CHIRPS v4 invalid"):
        quarantine.main(["--apply"])

    assert candidate.exists()
    assert not quarantine.QUARANTINE.exists()


def test_matching_link_candidate_is_rejected(tmp_path, monkeypatch):
    statistics, _ = _configure(tmp_path, monkeypatch)
    candidate = statistics / "phase4B_old.csv"
    candidate.write_text("legacy", encoding="utf-8")
    original = quarantine._is_link_or_junction
    monkeypatch.setattr(
        quarantine,
        "_is_link_or_junction",
        lambda path: path == candidate or original(path),
    )

    with pytest.raises(ValueError, match="link/junction"):
        quarantine.main(["--apply"])

    assert candidate.exists()


def test_post_move_validation_rejects_changed_active_target(tmp_path, monkeypatch):
    statistics, _ = _configure(tmp_path, monkeypatch)
    candidate = statistics / "phase40_old.csv"
    candidate.write_text("legacy", encoding="utf-8")
    reports = iter(
        [
            _valid_target_report(),
            _valid_target_report(content="c" * 64),
        ]
    )
    monkeypatch.setattr(quarantine, "_validated_active_target", lambda: next(reports))

    with pytest.raises(RuntimeError, match="identity changed"):
        quarantine.main(["--apply"])

    manifests = list(quarantine.QUARANTINE.glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "post_move_validation_failed"


def test_report_path_outside_workspace_is_rejected(tmp_path, monkeypatch):
    statistics, _ = _configure(tmp_path, monkeypatch)
    (statistics / "phase4C_old.csv").write_text("legacy", encoding="utf-8")
    outside = tmp_path.parent / "outside-phase4-report.json"

    with pytest.raises(ValueError, match="escapes workspace|outside allowed"):
        quarantine.main(["--report-json", str(outside)])

    assert not outside.exists()
