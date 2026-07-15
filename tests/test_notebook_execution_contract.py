from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nino_brasil.notebook_catalog import CANONICAL_NOTEBOOKS
from scripts import notebook_run_viewer


ROOT = Path(__file__).resolve().parents[1]


def _source(path: Path) -> tuple[dict, str, str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "markdown"
    )
    return notebook, code, markdown


@pytest.mark.parametrize("spec", CANONICAL_NOTEBOOKS, ids=lambda spec: spec.code)
def test_canonical_notebooks_are_clean_viewer_publishers(spec) -> None:
    notebook, source, markdown = _source(ROOT / spec.relative_path)
    metadata = notebook["metadata"]["nino26"]
    assert metadata["notebook_code"] == spec.code
    assert metadata["figure_precode"] == f"Fig{spec.code}"
    assert metadata["table_precode"] == f"Tab{spec.code}"
    assert metadata["default_run_pipeline"] == "0"
    assert metadata["persist_inline_outputs"] is spec.compact_source
    assert "NotebookWorkflow" in source
    assert "NINO26_RUN_PIPELINE" in source
    assert "raise RuntimeError" in source
    assert f"Fig{spec.code}1" in markdown
    assert f"Tab{spec.code}1" in markdown
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if spec.compact_source:
        inline_images = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if "image/png" in (output.get("data") or {})
        ]
        assert all(cell.get("execution_count") is not None for cell in code_cells)
        # Um notebook pode publicar uma unica figura canônica; o contrato é que
        # todo F3Nino persista ao menos uma imagem renderizada no próprio corpo.
        assert len(inline_images) >= 1
    else:
        assert all(cell.get("execution_count") is None for cell in code_cells)
        assert all(cell.get("outputs", []) == [] for cell in code_cells)


def test_run_all_notebooks_uses_catalog_and_writes_executed_copies() -> None:
    source = (ROOT / "scripts/run_all_notebooks.py").read_text(encoding="utf-8")
    assert "CANONICAL_NOTEBOOKS" in source
    assert 'nb.parent / "executed" / nb.name' in source
    assert '"--inplace"' not in source


def test_run_viewer_selects_by_validated_finished_time_not_directory_name(
    tmp_path, monkeypatch
):
    run_root = tmp_path / "data/processed/runs/official/fase5"
    older_lexical = run_root / "F5_ZZZ"
    newer_lexical = run_root / "F5_AAA"
    for directory, finished in (
        (older_lexical, "2026-01-01T00:00:00+00:00"),
        (newer_lexical, "2026-07-13T00:00:00+00:00"),
    ):
        directory.mkdir(parents=True)
        (directory / "run_manifest.json").write_text(
            json.dumps(
                {
                    "phase": 5,
                    "mode": "official",
                    "run_id": directory.name,
                    "status": "complete",
                    "finished_at": finished,
                    "parameters": {"model": "rf", "noise_copies": 0},
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        notebook_run_viewer,
        "validate_artifact_run",
        lambda _directory: pd.DataFrame(),
    )

    audit, selected = notebook_run_viewer.audit_artifact_runs(
        tmp_path, phase=5, mode="official"
    )
    assert len(audit) == 2
    assert len(selected) == 1
    assert selected[0]["run_id"] == "F5_AAA"


def test_run_viewer_keeps_latest_augmentation_on_and_off_arms(tmp_path, monkeypatch):
    run_root = tmp_path / "data/processed/runs/official/fase5"
    for suffix, copies in (("ON", 1), ("OFF", 0)):
        directory = run_root / f"F5_{suffix}"
        directory.mkdir(parents=True)
        (directory / "run_manifest.json").write_text(
            json.dumps(
                {
                    "phase": 5,
                    "mode": "official",
                    "run_id": directory.name,
                    "status": "complete",
                    "finished_at": f"2026-07-13T00:00:0{copies}+00:00",
                    "parameters": {
                        "model": "rf",
                        "noise_copies": copies,
                        "mixup_alpha": 0.4 if copies else None,
                    },
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        notebook_run_viewer,
        "validate_artifact_run",
        lambda _directory: pd.DataFrame(),
    )
    audit, selected = notebook_run_viewer.audit_artifact_runs(
        tmp_path, phase=5, mode="official"
    )
    assert len(audit) == 2
    assert len(selected) == 2
    assert {bool(item["augmentation"]) for item in selected} == {False, True}


@pytest.mark.parametrize(
    "phase,gate_table,gate_column,gate_values",
    [
        (5, "scientific_gate.csv", "gate_pass", [True, False]),
        (6, "field_gate.csv", "gate_pass", [True, True, False]),
        (7, "scientific_gate.csv", "scientific_gate_pass", [False]),
        (8, "confirmatory_gate_by_condition.csv", "gate_pass", [True, False]),
    ],
)
def test_run_viewer_reads_the_phase_specific_gate(
    tmp_path, monkeypatch, phase, gate_table, gate_column, gate_values
):
    directory = tmp_path / f"data/processed/runs/official/fase{phase}/F{phase}_RUN"
    (directory / "tables").mkdir(parents=True)
    (directory / "run_manifest.json").write_text(
        json.dumps(
            {
                "phase": phase,
                "mode": "official",
                "run_id": directory.name,
                "status": "complete",
                "finished_at": "2026-07-13T00:00:00+00:00",
                "parameters": {"model": "rf" if phase in {5, 6} else ""},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({gate_column: gate_values}).to_csv(
        directory / "tables" / gate_table, index=False
    )
    monkeypatch.setattr(
        notebook_run_viewer,
        "validate_artifact_run",
        lambda _directory: pd.DataFrame(),
    )
    audit, selected = notebook_run_viewer.audit_artifact_runs(
        tmp_path, phase=phase, mode="official"
    )
    assert len(selected) == 1
    row = audit.iloc[0]
    assert row["gate_table"] == gate_table
    assert row["gate_column"] == gate_column
    assert row["gate_rows"] == len(gate_values)
    assert row["gate_passes"] == sum(gate_values)
