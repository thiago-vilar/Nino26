from __future__ import annotations

import json
from pathlib import Path

from nino_brasil.notebook_catalog import specs_for_phase


ROOT = Path(__file__).resolve().parents[1]


def test_every_phase4_notebook_declares_its_public_fig_tab_precode() -> None:
    specs = specs_for_phase(4)
    assert {spec.code for spec in specs} == {
        "F4NinoC",
        "F4NinoD",
        "F4NinaC",
        "F4NinaD",
    }
    for spec in specs:
        notebook = json.loads((ROOT / spec.relative_path).read_text(encoding="utf-8"))
        metadata = notebook["metadata"]["nino26"]
        assert metadata["figure_precode"] == f"Fig{spec.code}"
        assert metadata["table_precode"] == f"Tab{spec.code}"
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        assert f"Fig{spec.code}1" in markdown
        assert f"Tab{spec.code}1" in markdown


def test_phase4_has_no_rejected_matrix_shortcut_or_fixed_2010_split() -> None:
    source = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in (
            "scripts/run_fase4c_regional.py",
            "scripts/run_fase4d_targets.py",
            "src/nino_brasil/stats/lag_analysis.py",
        )
    )
    assert "lagged_corr_pixel_matrix" not in source
    assert "summarize_peak_pixel_lags_by_region" in source
    for token in ("1993-2009", "1993–2009", "2010+", "2010-presente"):
        assert token not in source
