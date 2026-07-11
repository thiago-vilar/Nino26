"""Testes da co-geração figura+numeric-table (nino_brasil.viz)."""
from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest


def _viz_em(tmp_path: Path):
    viz = importlib.import_module("nino_brasil.viz")
    viz.ROOT = tmp_path
    viz.FIG_ROOT = tmp_path / "data/processed/figures"
    viz.NUM_ROOT = tmp_path / "data/processed/numeric-tables"
    viz.MANIFEST = tmp_path / "data/processed/figuras_manifesto.csv"
    return viz


def test_cogera_figura_tabela_manifesto(tmp_path):
    viz = _viz_em(tmp_path)
    df = pd.DataFrame({"lag": [0, 4, 8], "r": [0.1, 0.3, 0.2]})
    viz.registrar_figura(
        None, "Fig_3C01", fase=3, bloco="C", titulo="Precursores",
        descricao="t", hipotese="HIP0", notebook="nb", fontes={"tab_a": df, "tab_b": df},
    )
    d = viz.NUM_ROOT / "fase3/Fig_3C01"
    assert (d / "tab_a.csv").exists() and (d / "tab_b.csv").exists()
    assert (d / "manifest.csv").exists() and (d / "README.md").exists()
    man = pd.read_csv(viz.MANIFEST)
    assert len(man) == 1 and man.iloc[0]["codigo"] == "Fig_3C01" and man.iloc[0]["n_tabelas"] == 2


def test_sobreposicao_sem_orfao_e_sem_duplicar(tmp_path):
    viz = _viz_em(tmp_path)
    df = pd.DataFrame({"a": [1, 2]})
    kw = dict(fase=3, bloco="C", titulo="v", descricao="d", hipotese="HIP0", notebook="nb")
    viz.registrar_figura(None, "Fig_3C01", fontes={"x": df, "y": df}, **kw)
    viz.registrar_figura(None, "Fig_3C01", fontes={"x": df}, **{**kw, "titulo": "v2"})
    d = viz.NUM_ROOT / "fase3/Fig_3C01"
    assert not (d / "y.csv").exists()            # órfão removido
    man = pd.read_csv(viz.MANIFEST)
    assert len(man) == 1 and man.iloc[0]["titulo"] == "v2"   # upsert, não duplica


def test_codigo_invalido_rejeitado(tmp_path):
    viz = _viz_em(tmp_path)
    with pytest.raises(ValueError):
        viz.registrar_figura(None, "3C1", fase=3, bloco="C", titulo="x",
                             descricao="x", hipotese="HIP0", fontes={"a": pd.DataFrame({"a": [1]})})


def test_validador_pega_lixo_e_fora_do_padrao(tmp_path):
    viz = _viz_em(tmp_path)
    df = pd.DataFrame({"a": [1]})
    viz.registrar_figura(None, "Fig_3C01", fase=3, bloco="C", titulo="t",
                         descricao="d", hipotese="HIP0", fontes={"a": df})
    (viz.FIG_ROOT / "fase3").mkdir(parents=True, exist_ok=True)
    (viz.FIG_ROOT / "fase3/Fig_3C01.png").write_bytes(b"png")
    assert viz.validar_saidas(strict=False).empty          # coerente
    (viz.FIG_ROOT / "fase4").mkdir(parents=True, exist_ok=True)
    (viz.FIG_ROOT / "fase4/_tmp_x.png").write_bytes(b"x")
    (viz.FIG_ROOT / "fase4/phase4A_velho.png").write_bytes(b"x")
    prob = viz.validar_saidas(strict=False)
    assert {"lixo", "nome_fora_do_padrao"}.issubset(set(prob["tipo"]))
