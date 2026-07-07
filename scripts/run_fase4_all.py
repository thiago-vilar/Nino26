#!/usr/bin/env python3
"""Executa todos os notebooks da Fase 4 de uma vez (headless) e grava as
saidas (tabelas + graficos) dentro de cada .ipynb, para abrir e analisar depois.

Uso (no terminal, com o ambiente .venv NINO26 ativo):

    python scripts/run_fase4_all.py              # sequencial (recomendado)
    python scripts/run_fase4_all.py --parallel   # todos ao mesmo tempo (pesado)
    python scripts/run_fase4_all.py --only 0 4A   # so alguns notebooks

Cada notebook e executado in-place: ao terminar, abra-o no VS Code e role as
celulas para ver as tabelas (DataFrame) e figuras renderizadas. As figuras e
CSVs tambem ficam em data/processed/figures e data/processed/parquet/statistics.

Requer nbconvert (instale com:  pip install nbconvert ipykernel).
"""
from __future__ import annotations
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NBDIR = ROOT / "notebooks" / "fase4"
KERNEL = "python3"
TIMEOUT = 7200  # 2 h por notebook (a 1a execucao de B/C le ERA5/CHIRPS)

NOTEBOOKS = {
    "0": "0_fase4_sanidade_disponibilidade.ipynb",
    "I": "A_fase4_fontes_variaveis_series.ipynb",
    "4A": "4A_fase4_regionalizacao_chuva.ipynb",
    "4B": "4B_fase4_correlacao_regressao_defasada.ipynb",
    "4C": "4C_fase4_modos_acoplados.ipynb",
    "4D": "4D_fase4_atribuicao_composicoes.ipynb",
}


def cmd(nb: Path) -> list[str]:
    return [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute", "--inplace",
        # --allow-errors: grava as saidas MESMO se alguma celula falhar.
        # Sem isso, um unico erro faz o nbconvert descartar TODAS as saidas.
        "--allow-errors",
        f"--ExecutePreprocessor.timeout={TIMEOUT}",
        f"--ExecutePreprocessor.kernel_name={KERNEL}",
        str(nb),
    ]


def main(argv: list[str]) -> int:
    parallel = "--parallel" in argv
    only = None
    if "--only" in argv:
        i = argv.index("--only")
        only = [a.upper() for a in argv[i + 1:] if not a.startswith("--")]
    keys = [k for k in NOTEBOOKS if only is None or k.upper() in only]
    nbs = [NBDIR / NOTEBOOKS[k] for k in keys]
    missing = [str(p) for p in nbs if not p.exists()]
    if missing:
        print("Notebooks ausentes:", missing); return 1

    print(f"Executando {len(nbs)} notebook(s) em modo "
          f"{'PARALELO' if parallel else 'SEQUENCIAL'}:")
    for p in nbs:
        print("  -", p.name)
    print()

    t0 = time.time()
    results: dict[str, str] = {}
    if parallel:
        procs = {p.name: subprocess.Popen(cmd(p)) for p in nbs}
        for name, pr in procs.items():
            results[name] = "OK" if pr.wait() == 0 else "FALHOU"
    else:
        for p in nbs:
            print(f">>> {p.name} ...", flush=True)
            r = subprocess.run(cmd(p))
            results[p.name] = "OK" if r.returncode == 0 else "FALHOU"
            print(f"    {results[p.name]}\n", flush=True)

    print("=" * 60)
    for name, st in results.items():
        print(f"{st:8s} {name}")
    print(f"tempo total: {time.time()-t0:.0f}s")
    print("\nResultados gravados dentro de cada .ipynb (abra no VS Code).")
    print("Figuras: data/processed/figures/ | Tabelas: data/processed/parquet/statistics/")
    return 0 if all(v == "OK" for v in results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
