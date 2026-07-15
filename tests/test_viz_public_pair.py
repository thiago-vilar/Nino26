from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd


class _FakeFigure:
    def __init__(self, payload: bytes = b"public-figure") -> None:
        self.payload = payload

    def savefig(self, path: str | Path, **_: object) -> None:
        Path(path).write_bytes(self.payload)


def _viz_em(tmp_path: Path):
    viz = importlib.import_module("nino_brasil.viz")
    viz.ROOT = tmp_path
    viz.FIG_ROOT = tmp_path / "data/processed/figures"
    viz.NUM_ROOT = tmp_path / "data/processed/numeric-tables"
    viz.MANIFEST = tmp_path / "data/processed/figuras_manifesto.csv"
    return viz


def test_publica_par_fig_tab_no_namespace_do_notebook(tmp_path: Path) -> None:
    viz = _viz_em(tmp_path)
    result = viz.registrar_par_notebook(
        _FakeFigure(),
        "F3NinoA",
        1,
        pd.DataFrame({"semana": [0, 1], "valor": [0.2, 0.4]}),
        slug="SSTA media",
        titulo="SSTA media",
        descricao="Tabela e figura publicas",
        notebook="notebooks/fase3_nino/F3NinoA.ipynb",
        run_id="F3_NINO_TEST",
    )

    assert result.figure_code == "FigF3NinoA1_ssta_media"
    assert result.table_code == "TabF3NinoA1_ssta_media"
    assert result.figure_path == (
        viz.FIG_ROOT / "fase3_nino/FigF3NinoA1_ssta_media.png"
    )
    assert result.table_path == (
        viz.NUM_ROOT / "fase3_nino/TabF3NinoA1_ssta_media.csv"
    )
    assert result.figure_path.is_file()
    assert result.table_path.is_file()
    assert result.manifest_path.is_file()
    assert not (viz.FIG_ROOT / "fase3_nino/fase3").exists()
    assert not (viz.NUM_ROOT / "fase3_nino/fase3").exists()

    pair_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert pair_manifest["figure_code"] == result.figure_code
    assert pair_manifest["table_code"] == result.table_code
    assert pair_manifest["notebook_code"] == "F3NinoA"
    assert pair_manifest["run_id"] == "F3_NINO_TEST"
    assert pair_manifest["hashes"]["figure_sha256"] == result.figure_sha256
    assert pair_manifest["hashes"]["table_sha256"] == result.table_sha256

    global_manifest = pd.read_csv(viz.MANIFEST)
    row = global_manifest.iloc[0]
    assert row["pair_key"] == "F3NinoA:1"
    assert row["figure_code"] == result.figure_code
    assert row["table_code"] == result.table_code
    assert row["notebook_code"] == "F3NinoA"
    assert row["namespace"] == "fase3_nino"
    assert row["figure_sha256"] == result.figure_sha256
    assert row["table_sha256"] == result.table_sha256
    assert viz.validar_saidas(strict=False).empty
    assert viz.validar_saidas(
        strict=False, require_semantic_lineage=True
    ).empty


def test_slug_nao_cria_nova_chave_nem_deixa_orfaos(tmp_path: Path) -> None:
    viz = _viz_em(tmp_path)
    first = viz.registrar_par_notebook(
        _FakeFigure(b"first"),
        "F4NinaC",
        2,
        pd.DataFrame({"lag": [1]}),
        slug="primeira versao",
        run_id="F4_NINA_1",
    )
    second = viz.registrar_par_notebook(
        _FakeFigure(b"second"),
        "F4NinaC",
        2,
        pd.DataFrame({"lag": [2]}),
        slug="versao final",
        run_id="F4_NINA_2",
    )

    assert not first.figure_path.exists()
    assert not first.table_path.exists()
    assert not first.manifest_path.exists()
    assert second.figure_path.exists()
    assert second.table_path.exists()
    manifest = pd.read_csv(viz.MANIFEST)
    assert len(manifest) == 1
    assert manifest.iloc[0]["pair_key"] == "F4NinaC:2"
    assert manifest.iloc[0]["run_id"] == "F4_NINA_2"
    assert viz.validar_saidas(strict=False).empty


def test_publicacao_por_path_registra_hash_da_fonte(tmp_path: Path) -> None:
    viz = _viz_em(tmp_path)
    source = tmp_path / "source.csv"
    source.write_text("x,y\n1,2\n", encoding="utf-8")
    result = viz.registrar_par_notebook(
        _FakeFigure(),
        "F5A",
        1,
        source,
        run_id="F5_TEST",
    )

    pair_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert pair_manifest["source"]["kind"] == "path"
    assert Path(pair_manifest["source"]["path"]) == source.resolve()
    assert pair_manifest["hashes"]["source_sha256"]
    assert viz.validar_saidas(strict=False).empty


def test_validador_detecta_tabela_publica_adulterada(tmp_path: Path) -> None:
    viz = _viz_em(tmp_path)
    result = viz.registrar_par_notebook(
        _FakeFigure(),
        "F8A",
        2,
        pd.DataFrame({"skill": [0.1]}),
        run_id="F8_TEST",
    )
    result.table_path.write_text("skill\n999\n", encoding="utf-8")

    problems = viz.validar_saidas(strict=False)
    assert "public_table_hash_divergente" in set(problems["tipo"])
