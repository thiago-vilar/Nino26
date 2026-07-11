#!/usr/bin/env python3
"""Valida a coerência tripla figura <-> numeric-table <-> manifesto (padrão NINO-BRASIL).

Substitui o antigo `export_numeric_tables_for_figures.py` (dicionário fixo que
divergia dos notebooks). Aqui não há registro paralelo: as figuras e tabelas são
co-geradas por `nino_brasil.viz.registrar_figura`, e este script só CONFERE.

Uso:
    python scripts/validar_figuras.py            # relatório
    python scripts/validar_figuras.py --strict   # falha (exit 1) se houver problema
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
    args = ap.parse_args()

    problemas = validar_saidas(strict=False)
    if problemas.empty:
        print("[ok] contrato figura/numeric-table/manifesto íntegro.")
        return 0
    print(f"[falha] {len(problemas)} problema(s):")
    print(problemas.to_string(index=False))
    print("\nDicas: use nino_brasil.viz.registrar_figura para (re)gerar cada figura; "
          "remova PNGs legados/_tmp; renomeie para Fig_<F><B><NN>.")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
