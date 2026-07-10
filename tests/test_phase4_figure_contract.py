from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE4 = ROOT / "notebooks" / "fase4"
NOTEBOOKS = sorted(PHASE4.glob("4*.ipynb"))


def _code_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def test_every_phase4_figure_has_prefixed_name_and_interpretation() -> None:
    assert NOTEBOOKS
    pattern = re.compile(r"save_fig\([^,]+,\s*f?['\"]([^'\"]+)['\"]\)")
    valid_prefix = re.compile(r"^Fig_4(?:0|A|B|C|D)\d+_[A-Za-z0-9_{}]+\.png$")

    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            calls = pattern.findall(source)
            for name in calls:
                assert valid_prefix.fullmatch(name), f"nome de figura invalido em {path.name}: {name}"
                assert "Interpretacao:" in source, (
                    f"figura {name} em {path.name} nao traz interpretacao resumida na legenda"
                )


def test_phase4_has_no_rejected_matrix_shortcut_or_fixed_2010_split() -> None:
    source_4c = _code_source(PHASE4 / "4C_sinal_pixel_lags.ipynb")
    assert "lagged_corr_pixel_matrix" not in source_4c

    active_source = "\n".join(_code_source(path) for path in NOTEBOOKS)
    rejected = ("1993-2009", "1993–2009", "2010+", "2010-presente")
    for token in rejected:
        assert token not in active_source, f"corte temporal fixo ainda ativo: {token}"
