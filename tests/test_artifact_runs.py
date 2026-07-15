from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import nino_brasil.artifacts as artifacts
from scripts.run_all_notebooks import _has_complete_run


def test_artifact_run_full_hash_and_tamper_detection(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(artifacts, "ROOT", tmp_path)
    monkeypatch.setattr(artifacts, "RUN_ROOT", tmp_path / "runs")
    config = tmp_path / "config.yaml"
    source = tmp_path / "input.csv"
    config.write_text("a: 1\n", encoding="utf-8")
    source.write_text("x\n1\n", encoding="utf-8")
    run = artifacts.start_artifact_run(
        5,
        mode="smoke",
        inputs=[source],
        config_paths=[config],
        parameters={"seed": 7},
    )
    table = run.write_table(
        "metrics",
        pd.DataFrame({"fold": ["f1"], "skill": [0.2]}),
        description="test",
        primary_keys=("fold",),
    )
    run.finalize()
    manifest = pd.read_csv(run.directory / "tables_manifest.csv")
    assert len(manifest.loc[0, "sha256"]) == 64
    assert artifacts.validate_artifact_run(run.directory).empty
    table.write_text("fold,skill\nf1,999\n", encoding="utf-8")
    problems = artifacts.validate_artifact_run(run.directory)
    assert "hash_mismatch" in set(problems["type"])


def test_artifact_validation_detects_input_drift(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(artifacts, "ROOT", tmp_path)
    monkeypatch.setattr(artifacts, "RUN_ROOT", tmp_path / "runs")
    source = tmp_path / "input.csv"
    source.write_text("x\n1\n", encoding="utf-8")
    run = artifacts.start_artifact_run(
        5,
        mode="smoke",
        inputs=[source],
        config_paths=(),
    )
    run.write_table("metrics", pd.DataFrame({"skill": [0.1]}), description="test")
    run.finalize()
    source.write_text("x\n2\n", encoding="utf-8")
    problems = artifacts.validate_artifact_run(run.directory)
    assert "input_hash_mismatch" in set(problems["type"])


def test_failed_empty_run_still_has_readable_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(artifacts, "ROOT", tmp_path)
    monkeypatch.setattr(artifacts, "RUN_ROOT", tmp_path / "runs")
    run = artifacts.start_artifact_run(8, mode="smoke", config_paths=())
    run.finalize(status="failed", notes="preflight")
    manifest = pd.read_csv(run.directory / "tables_manifest.csv")
    assert manifest.empty
    assert artifacts.validate_artifact_run(run.directory).empty


def test_notebook_preflight_rejects_incomplete_and_unmerged_f6_runs(tmp_path) -> None:
    incomplete = tmp_path / "F6_20260101_incomplete"
    incomplete.mkdir()
    (incomplete / "run_manifest.json").write_text(
        json.dumps({"phase": 6, "mode": "official", "status": "running"}),
        encoding="utf-8",
    )
    assert not _has_complete_run(tmp_path, phase=6, mode="official")

    shard = tmp_path / "F6_20260102_shard"
    shard.mkdir()
    (shard / "run_manifest.json").write_text(
        json.dumps(
            {
                "phase": 6,
                "mode": "official",
                "status": "complete",
                "parameters": {"role": "pixel_shard"},
            }
        ),
        encoding="utf-8",
    )
    (shard / "tables_manifest.csv").write_text("table\n", encoding="utf-8")
    assert not _has_complete_run(tmp_path, phase=6, mode="official")

    merged = tmp_path / "F6_20260103_merge"
    merged.mkdir()
    (merged / "run_manifest.json").write_text(
        json.dumps(
            {
                "phase": 6,
                "mode": "official",
                "status": "complete",
                "parameters": {"role": "merge_pixel_shards_and_field_gate"},
            }
        ),
        encoding="utf-8",
    )
    (merged / "tables_manifest.csv").write_text("table\n", encoding="utf-8")
    assert _has_complete_run(tmp_path, phase=6, mode="official")
