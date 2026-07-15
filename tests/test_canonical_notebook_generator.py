from __future__ import annotations

from scripts.generate_canonical_notebooks import build_notebook
from nino_brasil.notebook_catalog import CANONICAL_NOTEBOOKS


def test_generated_notebooks_are_clean_and_coded():
    for spec in CANONICAL_NOTEBOOKS:
        notebook = build_notebook(spec)
        metadata = notebook["metadata"]["nino26"]
        assert metadata["notebook_code"] == spec.code
        assert metadata["figure_precode"] == f"Fig{spec.code}"
        assert metadata["table_precode"] == f"Tab{spec.code}"
        assert metadata["persist_inline_outputs"] is spec.compact_source
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert all(cell["execution_count"] is None for cell in code_cells)
        assert all(cell["outputs"] == [] for cell in code_cells)
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        assert ("tl" + ";dr") not in source.casefold()
        assert f"NOTEBOOK_CODE = '{spec.code}'" in source
        assert f"Fig{spec.code}1" in source
        assert f"Tab{spec.code}1" in source
        assert notebook["cells"][0]["cell_type"] == "markdown"
        intro = "".join(notebook["cells"][0]["source"])
        headings = (
            "**TÍTULO**",
            "**CONTEXTO**",
            "**MOTIVAÇÃO**",
            "**METODOLOGIA**",
            "**RESULTADOS ESPERADOS**",
        )
        positions = [intro.index(heading) for heading in headings]
        assert positions == sorted(positions)
        assert notebook["cells"][-1]["cell_type"] == "markdown"
        assert "".join(notebook["cells"][-1]["source"]).startswith(
            "**REFERÊNCIAS BIBLIOGRÁFICAS**"
        )


def test_f3_nino_starts_with_complete_scientific_protocol():
    required = (
        "**TÍTULO**",
        "**CONTEXTO**",
        "**MOTIVAÇÃO**",
        "**METODOLOGIA**",
        "**RESULTADOS ESPERADOS**",
    )
    for spec in CANONICAL_NOTEBOOKS:
        if not spec.code.startswith("F3Nino"):
            continue
        notebook = build_notebook(spec)
        first_cell = "".join(notebook["cells"][0]["source"])
        assert all(section in first_cell for section in required)
        assert spec.context in first_cell
        assert spec.hypothesis_statement in first_cell
        assert spec.method_rationale in first_cell
        assert all(output in first_cell for output in spec.expected_outputs)
        last_cell = "".join(notebook["cells"][-1]["source"])
        assert last_cell.startswith("**REFERÊNCIAS BIBLIOGRÁFICAS**")
        assert all(reference in last_cell for reference in spec.references)
