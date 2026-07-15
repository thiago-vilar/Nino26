from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import quarantine_incomplete_artifact_runs as quarantine


def _configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quarantine, "ROOT", tmp_path)
    monkeypatch.setattr(
        quarantine, "RUN_ROOT", tmp_path / "data/processed/runs"
    )
    monkeypatch.setattr(
        quarantine,
        "QUARANTINE",
        tmp_path / "data/quarantine/incomplete_runs",
    )
    monkeypatch.setattr(quarantine, "AUDIT_ROOT", tmp_path / "data/audit")


def _run_directory(
    root: Path, *, run_id: str, phase: int = 5, mode: str = "official"
) -> Path:
    directory = root / "data/processed/runs" / mode / f"fase{phase}" / run_id
    directory.mkdir(parents=True)
    return directory


def _make_incomplete(
    root: Path,
    *,
    run_id: str,
    phase: int = 5,
    mode: str = "official",
    status: str | None = None,
) -> Path:
    directory = _run_directory(root, run_id=run_id, phase=phase, mode=mode)
    tables = directory / "tables"
    tables.mkdir()
    (tables / "partial.csv").write_text("value\n1\n", encoding="utf-8")
    if status is not None:
        (directory / "run_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "nino26-run-v1",
                    "run_id": run_id,
                    "phase": phase,
                    "mode": mode,
                    "status": status,
                }
            ),
            encoding="utf-8",
        )
    return directory


def _make_complete(
    root: Path,
    *,
    run_id: str,
    phase: int = 5,
    mode: str = "official",
    scientific_gate_pass: bool = False,
) -> Path:
    directory = _run_directory(root, run_id=run_id, phase=phase, mode=mode)
    tables = directory / "tables"
    tables.mkdir()
    gate = tables / "scientific_gate.csv"
    gate.write_text(
        f"metric,scientific_gate_pass\nf1,{scientific_gate_pass}\n",
        encoding="utf-8",
    )
    tables_manifest = directory / "tables_manifest.csv"
    with tables_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=sorted(quarantine.TABLE_MANIFEST_COLUMNS)
        )
        writer.writeheader()
        writer.writerow(
            {
                "table": gate.name,
                "path": "tables/scientific_gate.csv",
                "rows": 1,
                "columns": 2,
                "sha256": quarantine._sha256_file(gate),
                "schema_sha256": "a" * 64,
                "description": "Gate may be false without invalidating the run.",
                "units_json": "{}",
                "dimensions_json": "{}",
                "methods_json": "{}",
                "primary_keys": "metric",
            }
        )
    parameters: dict[str, object] = {"model": "rf"}
    manifest = {
        "schema_version": "nino26-run-v1",
        "run_id": run_id,
        "phase": phase,
        "mode": mode,
        "status": "complete",
        "started_at": "2026-07-13T10:00:00+00:00",
        "finished_at": "2026-07-13T10:01:00+00:00",
        "seed": 42,
        "parameters": parameters,
        "parameters_sha256": quarantine._json_hash(parameters),
        "environment": {},
        "inputs": [],
        "configs": [],
        "files": [],
        "n_tables": 1,
        "tables_manifest_sha256": quarantine._sha256_file(tables_manifest),
    }
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return directory


def test_discovery_is_direct_f5_f8_only_and_protects_complete_false_gate(
    tmp_path,
    monkeypatch,
):
    _configure(tmp_path, monkeypatch)
    incomplete_f5 = _make_incomplete(tmp_path, run_id="F5_incomplete")
    incomplete_f8 = _make_incomplete(
        tmp_path, run_id="F8_incomplete", phase=8, mode="smoke"
    )
    complete_false = _make_complete(
        tmp_path, run_id="F5_complete_false_gate", scientific_gate_pass=False
    )
    _make_incomplete(tmp_path, run_id="F4_out_of_scope", phase=4)
    out_of_scope = (
        tmp_path
        / "data/processed/runs/official/fase9/F9_out_of_scope"
    )
    out_of_scope.mkdir(parents=True)
    phase_root_file = incomplete_f5.parent / "README.txt"
    phase_root_file.write_text("not a run directory", encoding="utf-8")

    candidates = quarantine.discover_candidates()

    assert {candidate.path for candidate in candidates} == {
        incomplete_f5,
        incomplete_f8,
    }
    assert quarantine.audit_completion(
        complete_false, phase=5, mode="official"
    ) == ()


def test_integrity_drift_and_noncomplete_status_are_candidates(
    tmp_path,
    monkeypatch,
):
    _configure(tmp_path, monkeypatch)
    drifted = _make_complete(tmp_path, run_id="F5_hash_drift")
    (drifted / "tables/scientific_gate.csv").write_text(
        "metric,scientific_gate_pass\nf1,True\n", encoding="utf-8"
    )
    failed = _make_incomplete(tmp_path, run_id="F7_failed", phase=7, status="failed")
    malformed = _make_incomplete(tmp_path, run_id="F6_bad_json", phase=6)
    (malformed / "run_manifest.json").write_text("{bad-json", encoding="utf-8")

    by_id = {candidate.run_id: candidate for candidate in quarantine.discover_candidates()}

    assert "table_hash_mismatch:scientific_gate.csv" in by_id[
        drifted.name
    ].completion_problems
    assert "run_status_not_complete" in by_id[failed.name].completion_problems
    assert "invalid_run_manifest_json" in by_id[malformed.name].completion_problems


def test_default_dry_run_never_creates_quarantine_or_moves(
    tmp_path,
    monkeypatch,
    capsys,
):
    _configure(tmp_path, monkeypatch)
    candidate = _make_incomplete(tmp_path, run_id="F5_dry_run")

    assert quarantine.main([]) == 0

    assert candidate.is_dir()
    assert not quarantine.QUARANTINE.exists()
    output = capsys.readouterr().out
    assert "WOULD QUARANTINE" in output
    assert "apply=False" in output


def test_apply_requires_explicit_current_candidate_allowlist(
    tmp_path,
    monkeypatch,
):
    _configure(tmp_path, monkeypatch)
    candidate = _make_incomplete(tmp_path, run_id="F5_candidate")
    complete = _make_complete(tmp_path, run_id="F5_complete")

    with pytest.raises(ValueError, match="requires at least one"):
        quarantine.main(["--apply"])
    with pytest.raises(ValueError, match="not current incomplete candidates"):
        quarantine.main(["--apply", "--run-id", complete.name])
    with pytest.raises(ValueError, match="not current incomplete candidates"):
        quarantine.main(["--apply", "--run-id", "F5_unknown"])
    with pytest.raises(ValueError, match="Unsafe --run-id"):
        quarantine.main(["--apply", "--run-id", "../escape"])

    assert candidate.is_dir()
    assert complete.is_dir()
    assert not quarantine.QUARANTINE.exists()


def test_apply_moves_only_selected_with_atomic_manifest_and_tree_receipt(
    tmp_path,
    monkeypatch,
):
    _configure(tmp_path, monkeypatch)
    selected = _make_incomplete(tmp_path, run_id="F5_selected")
    unselected = _make_incomplete(tmp_path, run_id="F5_unselected")

    assert quarantine.main(["--apply", "--run-id", selected.name]) == 0

    assert not selected.exists()
    assert unselected.is_dir()
    manifests = list(quarantine.QUARANTINE.glob("*/manifest.json"))
    assert len(manifests) == 1
    assert not list(quarantine.QUARANTINE.rglob("*.tmp-*"))
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["explicit_run_id_allowlist"] == [selected.name]
    assert manifest["selected_count"] == 1
    records = {record["run_id"]: record for record in manifest["records"]}
    moved = records[selected.name]
    retained = records[unselected.name]
    assert moved["status"] == "moved_and_verified"
    assert moved["source_tree_state"]["tree_sha256"] == moved[
        "destination_tree_state"
    ]["tree_sha256"]
    assert moved["source_tree_state"]["entries"]
    assert all(not Path(moved[key]).is_absolute() for key in ("source", "destination"))
    assert (tmp_path / moved["destination"]).is_dir()
    assert retained["status"] == "refused_not_allowlisted"
    assert retained["selected_by_explicit_allowlist"] is False


def test_apply_refuses_candidate_that_changes_after_inventory(
    tmp_path,
    monkeypatch,
):
    _configure(tmp_path, monkeypatch)
    candidate = _make_incomplete(tmp_path, run_id="F5_changing")
    original = quarantine._current_matching_candidate

    def change_then_read(record):
        (candidate / "changed.txt").write_text("concurrent write", encoding="utf-8")
        return original(record)

    monkeypatch.setattr(quarantine, "_current_matching_candidate", change_then_read)

    with pytest.raises(RuntimeError, match="changed after inventory"):
        quarantine.main(["--apply", "--run-id", candidate.name])

    assert candidate.is_dir()
    manifests = list(quarantine.QUARANTINE.glob("*/manifest.json"))
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload["status"] == "move_failed"


def test_link_or_junction_candidate_is_rejected(
    tmp_path,
    monkeypatch,
):
    _configure(tmp_path, monkeypatch)
    candidate = _make_incomplete(tmp_path, run_id="F5_linked")
    original = quarantine._is_link_or_junction
    monkeypatch.setattr(
        quarantine,
        "_is_link_or_junction",
        lambda path: path == candidate or original(path),
    )

    with pytest.raises(ValueError, match="link/junction"):
        quarantine.discover_candidates()

    assert candidate.is_dir()


def test_report_path_must_remain_below_data_audit(
    tmp_path,
    monkeypatch,
):
    _configure(tmp_path, monkeypatch)
    _make_incomplete(tmp_path, run_id="F5_report")
    outside = tmp_path / "outside.json"

    with pytest.raises(ValueError, match="outside allowed root"):
        quarantine.main(["--report-json", str(outside)])

    assert not outside.exists()
