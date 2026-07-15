"""Executa os notebooks científicos reconstruídos da FASE3-NINO."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "fase3_nino" / "cientifica"
NOTEBOOKS = {
    "F3NINO_01": "F3NINO_01_series_historicas_31_variaveis.ipynb",
    "F3NINO_02": "F3NINO_02_eventos_oni_p90_e_fases.ipynb",
    "F3NINO_03": "F3NINO_03_compostos_31_variaveis.ipynb",
    "F3NINO_04": "F3NINO_04_diagnostico_quatro_fases.ipynb",
    "F3NINO_05": "F3NINO_05_reducao_variaveis_ciclo_vida.ipynb",
    "F3NINO_06": "F3NINO_06_pca_comparativa_eof_mapas.ipynb",
    "F3NINO_07": "F3NINO_07_compostos_ssta_mapas.ipynb",
    "F3NINO_08": "F3NINO_08_kelvin_e_ventos.ipynb",
    "F3NINO_09": "F3NINO_09_hovmoller_ssta.ipynb",
    "F3NINO_10": "F3NINO_10_sensibilidade_e_incerteza.ipynb",
    "F3NINO_11": "F3NINO_11_sintese_e_referencias.ipynb",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=tuple(NOTEBOOKS), help="executa apenas os códigos selecionados")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--run-id", help="run_id comum para os artefatos")
    args = parser.parse_args(argv)
    selected = args.only or list(NOTEBOOKS)
    env = os.environ.copy()
    env["FASE3_NINO_RUN_ID"] = args.run_id or env.get("FASE3_NINO_RUN_ID", "FASE3_NINO_CIENTIFICA")
    for code in selected:
        path = NOTEBOOK_DIR / NOTEBOOKS[code]
        if not path.is_file():
            raise FileNotFoundError(path)
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
            str(path),
        ]
        print(f">>> {code} — {path.name}", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    print(f"[ok] FASE3-NINO concluída: {len(selected)} notebooks; saídas em data/processed/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
