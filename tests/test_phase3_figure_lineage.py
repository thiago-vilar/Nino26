from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import nino_brasil.viz as viz


def test_phase3_public_pair_uses_notebook_precode_and_run_lineage(tmp_path, monkeypatch):
    monkeypatch.setattr(viz, "FIG_ROOT", tmp_path / "figures")
    monkeypatch.setattr(viz, "NUM_ROOT", tmp_path / "numeric-tables")
    monkeypatch.setattr(viz, "MANIFEST", tmp_path / "figuras_manifesto.csv")
    monkeypatch.setattr(viz, "ROOT", tmp_path)
    fig, axis = plt.subplots()
    axis.plot([1, 2], [3, 4])
    pair = viz.registrar_par_notebook(
        fig,
        "F3NinoC",
        1,
        pd.DataFrame({"lag_semanas": [4], "r_pearson": [0.5]}),
        run_id="F3NINO_TEST",
    )
    plt.close(fig)
    assert pair.figure_path.name == "FigF3NinoC1.png"
    assert pair.table_path.name == "TabF3NinoC1.csv"
    manifest = json.loads(pair.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "F3NINO_TEST"
    assert manifest["pair_key"] == "F3NinoC:1"
    assert viz.validar_saidas(strict=False, require_semantic_lineage=True).empty


def test_phase3_public_pair_rejects_missing_run_id() -> None:
    fig, _axis = plt.subplots()
    try:
        with pytest.raises(ValueError, match="run_id"):
            viz.registrar_par_notebook(
                fig,
                "F3NinaC",
                1,
                pd.DataFrame({"x": [1]}),
                run_id="",
            )
    finally:
        plt.close(fig)
