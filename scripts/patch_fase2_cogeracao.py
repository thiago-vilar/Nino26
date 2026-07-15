#!/usr/bin/env python3
"""Roteia as figuras do notebook 2Z (Fase 2) para a co-geração canônica.

O 2Z usava `fig.savefig(...)` cru (nomes `phase2_sanidade_*`, sem numeric-table).
Este patch troca essas chamadas por `nino_brasil.viz.cogerar_de_figura`, que
grava a figura sob o código canônico `Fig_2Z0N_<descritivo>` e co-gera a
numeric-table + manifesto. Idempotente (rodar de novo não duplica).

Uso (uma vez):
    .venv-wsl/bin/python scripts/patch_fase2_cogeracao.py
    # depois rode a Fase 2 de novo (ver README de execução)
"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks/fase2/2Z_sanidade_variaveis.ipynb"

IMPORT_OLD = "FIGS = ROOT/'data/processed/figures/fase2'; FIGS.mkdir(parents=True, exist_ok=True)"
IMPORT_NEW = (IMPORT_OLD + "; import sys as _s2; _s2.path.insert(0, str(ROOT/'src'))"
              "; from nino_brasil.viz import cogerar_de_figura as _cg")

NB_REF = "notebooks/fase2/2Z_sanidade_variaveis.ipynb"
SAVE_GRUPO_OLD = "fig.savefig(out, dpi=130, bbox_inches='tight'); plt.show()"
SAVE_GRUPO_NEW = (f"_cg(fig, f'phase2_sanidade_{{chave}}.png', fase=2, bloco='Z', "
                  f"notebook='{NB_REF}'); plt.show()")
SAVE_Z_OLD = "fig.savefig(FIGS/'phase2_sanidade_painel_z.png', dpi=130, bbox_inches='tight'); plt.show()"
SAVE_Z_NEW = (f"_cg(fig, 'phase2_sanidade_painel_z.png', fase=2, bloco='Z', "
              f"notebook='{NB_REF}'); plt.show()")

SUBS = [(IMPORT_OLD, IMPORT_NEW), (SAVE_GRUPO_OLD, SAVE_GRUPO_NEW), (SAVE_Z_OLD, SAVE_Z_NEW)]


def _load_tolerante(path: Path) -> dict:
    """Carrega o .ipynb ignorando lixo após o JSON (artefato de FS montado)."""
    txt = path.read_text(encoding="utf-8")
    return json.JSONDecoder().raw_decode(txt)[0]


def main() -> int:
    nb = _load_tolerante(NB)
    if any("_cg(" in "".join(c.get("source", [])) for c in nb["cells"]):
        print("[ok] 2Z já roteado para cogerar_de_figura (nada a fazer).")
        return 0
    trocas = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        novo = src
        for old, new in SUBS:
            if old in novo:
                novo = novo.replace(old, new); trocas += 1
        if novo != src:
            cell["source"] = novo.splitlines(keepends=True)
        cell["outputs"] = []          # limpa saídas pesadas (regeneradas ao rodar)
        cell["execution_count"] = None
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[ok] 2Z roteado ({trocas} substituições). Rode a Fase 2 de novo para gerar "
          "Fig_2Z01_oceano_superficie ... Fig_2Z05_painel_z + numeric-tables.")
    return 0 if trocas >= 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
