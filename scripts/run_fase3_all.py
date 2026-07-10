#!/usr/bin/env python3
"""Executa a Fase 3: gera artefatos EN/LN, notebooks, relatorio e numeric-tables."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NBDIR = ROOT / "notebooks" / "fase3"
NOTEBOOKS = [
    "3A_indices_fisicos_semanais.ipynb",
    "3B_alvo_eventos_ciclo_vida.ipynb",
    "3C_precursores_lags.ipynb",
    "3D_rigor_estatistico.ipynb",
    "3E_sensibilidade_temporal.ipynb",
    "3F_kelvin_sla.ipynb",
    "3G_compostos_ssta.ipynb",
    "3H_genese_precursores_classe.ipynb",
    "3K_pca_crescimento.ipynb",
    "3I_interpretacao_integrada.ipynb",
    "3L_en_ln_caracterizacao.ipynb",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--skip-report", action="store_true",
                        help="nao gerar RELATORIO_FINAL_FASE3.md apos os notebooks")
    args = parser.parse_args(argv)

    t0 = time.time()
    # Pre-etapa EN/LN: eventos, ciclo de vida, duracoes, discriminantes e PCA por
    # fase (El Nino E La Nina) a partir da matriz-mestre; roda antes dos notebooks.
    print(">>> phase3_en_ln.py", flush=True)
    subprocess.run([sys.executable, "scripts/phase3_en_ln.py"], cwd=ROOT, check=True)

    for name in NOTEBOOKS:
        nb = NBDIR / name
        if not nb.exists():
            raise FileNotFoundError(nb)
        print(f">>> {name}", flush=True)
        subprocess.run([
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            f"--ExecutePreprocessor.timeout={args.timeout}",
            f"--ExecutePreprocessor.kernel_name={args.kernel}",
            str(nb),
        ], cwd=ROOT, check=True)

    if not args.skip_report:
        print(">>> generate_phase3_report.py", flush=True)
        subprocess.run([sys.executable, "scripts/generate_phase3_report.py"], cwd=ROOT, check=True)
    print(">>> export_numeric_tables_for_figures.py", flush=True)
    subprocess.run([
        sys.executable, "scripts/export_numeric_tables_for_figures.py",
        "--force", "--strict",
    ], cwd=ROOT, check=True)
    print(f"Fase 3 completa em {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
