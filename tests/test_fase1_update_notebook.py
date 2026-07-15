from scripts.generate_fase1_update_notebook import build_notebook


def test_fase1_update_notebook_is_guarded_and_documented():
    notebook = build_notebook()
    intro = notebook.cells[0].source
    for heading in (
        "**TÍTULO**",
        "**CONTEXTO**",
        "**MOTIVAÇÃO**",
        "**METODOLOGIA**",
        "**RESULTADOS ESPERADOS**",
    ):
        assert heading in intro
    assert notebook.cells[0].source.startswith("**COMANDO WSL2 — EXECUTAR FASE COMPLETA**")
    assert notebook.cells[0].source.index("**COMANDO WSL2") < notebook.cells[0].source.index("**TÍTULO**")
    assert notebook.cells[-1].source.startswith("**REFERÊNCIAS BIBLIOGRÁFICAS**")
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "NINO26_UPDATE_DATA" in source
    assert "run_full_download_pipeline.py" in source
    assert "audit_ibge_boundaries.py" in source
