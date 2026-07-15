from __future__ import annotations

import json
from pathlib import Path

from nino_brasil.notebook_catalog import specs_for_phase


ROOT = Path(__file__).resolve().parents[1]


def _code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_phase3_has_all_eleven_notebooks_for_each_isolated_signal() -> None:
    nino = specs_for_phase(3, "el_nino")
    nina = specs_for_phase(3, "la_nina")
    assert len(nino) == len(nina) == 11
    assert {spec.code[-1] for spec in nino} == set("ABCDEFGHIKL")
    assert {spec.code[-1] for spec in nina} == set("ABCDEFGHIKL")
    assert all((ROOT / spec.relative_path).is_file() for spec in (*nino, *nina))


def test_phase3_viewers_never_execute_or_mix_the_numeric_core() -> None:
    for enso_type in ("el_nino", "la_nina"):
        for spec in specs_for_phase(3, enso_type):
            source = _code(ROOT / spec.relative_path)
            assert "NotebookWorkflow" in source
            assert "NINO26_RUN_PIPELINE" in source
            assert "raise RuntimeError" in source
            assert "phase3_en_ln.py" not in source


def test_phase3_writer_parameters_are_complete_before_writer_and_top5_is_declared() -> None:
    source = (ROOT / "scripts/phase3_en_ln.py").read_text(encoding="utf-8")
    assert "parameters.update(" not in source
    assert source.index("signal_column =") < source.index("parameters = {")
    assert source.index('"excluded_target_aliases":') < source.index("writer = Writer(")
    assert '"bootstrap_top": args.bootstrap_top' in source
    assert "default=5" in source
    assert "estabilidade_dos_precursores_priorizados_nao_novo_screening" in source
    assert "PHASE3_TABLE_LAYOUT" in source
    assert "table_code(notebook, ordinal, slug=slug)" in source


def test_phase3_scoped_runner_uses_separate_directories_and_executed_copies() -> None:
    source = (ROOT / "scripts/run_fase3_all.py").read_text(encoding="utf-8")
    assert '"el_nino": "fase3_nino"' in source
    assert '"la_nina": "fase3_nina"' in source
    assert 'executed_dir = nbdir / "executed"' in source
    assert '"--output-dir", str(executed_dir)' in source
    assert "NINO26_PHASE3_FIGURE_ROOT" not in source


def test_downstream_models_use_the_explicit_f3_bridge() -> None:
    for relative in (
        "scripts/run_fase5_cycle_ml.py",
        "scripts/run_fase6_brazil_ml.py",
        "scripts/run_fase6_all_shards.py",
        "scripts/run_fase7_cycle_convlstm.py",
        "scripts/run_fase8_brazil_convlstm.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "f3_bridge" in source
        assert "phase3_fases_semanais_en_ln.csv" not in source
