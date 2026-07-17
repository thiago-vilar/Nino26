#!/usr/bin/env python3
"""Executa a fase3_nino completa: regenera os 7 notebooks e roda em ordem.

Os notebooks executados são gravados em notebooks/fase3_nino/executed/;
figuras (600 dpi png+pdf) e tabelas numéricas saem em
data/processed/figures/fase3 e data/processed/numeric-tables/fase3.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks/fase3_nino"
EXECUTED_DIR = NOTEBOOK_DIR / "executed"

ORDER = [
    "F3N1_series_semanais_normalizacao.ipynb",
    "F3N2_eventos_el_nino.ipynb",
    "F3N3_ciclo_de_vida.ipynb",
    "F3N4_triagem_pca.ipynb",
    "F3N5_bjerknes_feedback.ipynb",
    "F3N6_ondas_kelvin.ipynb",
    "F3N7_hovmoller.ipynb",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="nino-brasil")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--skip-generate", action="store_true", help="Não regenerar os .ipynb fonte antes de executar.")
    parser.add_argument("--only", action="append", help="Executar somente os notebooks cujo nome contém este texto.")
    args = parser.parse_args(argv)

    if not args.skip_generate:
        subprocess.run([sys.executable, str(ROOT / "scripts/generate_fase3_nino_notebooks.py")], cwd=ROOT, check=True)

    EXECUTED_DIR.mkdir(parents=True, exist_ok=True)
    selected = [
        name for name in ORDER
        if not args.only or any(token.lower() in name.lower() for token in args.only)
    ]
    failures: list[str] = []
    for name in selected:
        source = NOTEBOOK_DIR / name
        print(f"\n=== executando {name} ===", flush=True)
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook", "--execute",
                f"--ExecutePreprocessor.timeout={args.timeout}",
                f"--ExecutePreprocessor.kernel_name={args.kernel}",
                "--output-dir", str(EXECUTED_DIR),
                "--output", name,
                str(source),
            ],
            cwd=ROOT,
        )
        if result.returncode != 0:
            failures.append(name)
            print(f"FALHA: {name}")
    print()
    if failures:
        print(f"fase3_nino: {len(failures)} falha(s): {failures}")
        return 1
    print(f"fase3_nino: {len(selected)} notebooks executados com sucesso -> {EXECUTED_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
