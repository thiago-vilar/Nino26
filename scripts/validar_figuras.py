#!/usr/bin/env python3
"""Valida a coerência tripla figura <-> numeric-table <-> manifesto (padrão NINO-BRASIL).

Substitui o antigo `export_numeric_tables_for_figures.py` (dicionário fixo que
divergia dos notebooks). Aqui não há registro paralelo: as figuras e tabelas são
co-geradas por `nino_brasil.viz.registrar_figura`, e este script só CONFERE.

Uso:
    python scripts/validar_figuras.py            # relatório
    python scripts/validar_figuras.py --strict --allow-render-extraction
                                                # compatibilidade estrutural
    python scripts/validar_figuras.py --strict   # gate de promoção semântica
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nino_brasil.viz import validar_saidas  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true", help="exit 1 se houver qualquer problema")
    ap.add_argument(
        "--allow-render-extraction",
        action="store_true",
        help="auditoria de migracao: aceita tabelas extraidas da renderizacao sem run_id",
    )
    ap.add_argument(
        "--max-details",
        type=int,
        default=None,
        help="limita linhas detalhadas impressas; o exit code ainda considera todas",
    )
    args = ap.parse_args()
    if args.max_details is not None and args.max_details < 0:
        ap.error("--max-details must be >= 0")

    semantic_gate = args.strict and not args.allow_render_extraction
    problemas = validar_saidas(
        strict=False,
        require_semantic_lineage=semantic_gate,
    )
    if problemas.empty:
        mode = "linhagem semântica" if semantic_gate else "compatibilidade estrutural"
        print(f"[ok] contrato figura/numeric-table/manifesto íntegro ({mode}).")
        return 0
    print(f"[falha] {len(problemas)} problema(s):")
    if args.max_details is not None:
        summary = (
            problemas.groupby("tipo", dropna=False)
            .size()
            .rename("n")
            .sort_values(ascending=False)
        )
        print(summary.to_string())
        if args.max_details:
            print("\nPrimeiros detalhes:")
            print(problemas.head(args.max_details).to_string(index=False))
        omitted = max(0, len(problemas) - args.max_details)
        if omitted:
            print(f"\n... {omitted} detalhe(s) omitido(s).")
    else:
        print(problemas.to_string(index=False))
    print(
        "\nDica: publique novamente pelo notebook canônico com "
        "registrar_par_notebook; os nomes devem ser Fig<CodigoNotebook><n> e "
        "Tab<CodigoNotebook><n>."
    )
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
