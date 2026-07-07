#!/usr/bin/env python3
"""Executa os notebooks da Fase 3 em ordem, com kernel novo por notebook."""
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
    "3E_estabilidade_subperiodos.ipynb",
    "3F_dhw_kelvin.ipynb",
    "3G_ciclo_vida_dhw.ipynb",
    "3H_genese_precursores_classe.ipynb",
    "3I_interpretacao_integrada.ipynb",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--kernel", default="python3")
    args = parser.parse_args(argv)

    t0 = time.time()
    for name in NOTEBOOKS:
        nb = NBDIR / name
        if not nb.exists():
            raise FileNotFoundError(nb)
        print(f">>> {name}", flush=True)
        command = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            f"--ExecutePreprocessor.timeout={args.timeout}",
            f"--ExecutePreprocessor.kernel_name={args.kernel}",
            str(nb),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"Fase 3 completa em {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
