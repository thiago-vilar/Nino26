#!/usr/bin/env python3
"""Executa todos os notebooks da Fase 4 de uma vez (headless) e grava as
saidas (tabelas + graficos) dentro de cada .ipynb, para abrir e analisar depois.

Uso (no terminal, com o ambiente .venv NINO26 ativo):

    python scripts/run_fase4_all.py              # sequencial (recomendado)
    python scripts/run_fase4_all.py --only 4.0 4A # so alguns notebooks

Cada notebook e executado in-place: ao terminar, abra-o no VS Code e role as
celulas para ver as tabelas (DataFrame) e figuras renderizadas. As figuras e
CSVs tambem ficam em data/processed/figures e data/processed/parquet/statistics.
A camada obrigatoria figura->tabela fica em data/processed/numeric-tables.

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
    "4.0": "4_0_fase4_abertura.ipynb",
    "4A": "4A_ciclo_enso_fases.ipynb",
    "4B": "4B_variaveis_determinantes_fases.ipynb",
    "4C": "4C_sinal_pixel_lags.ipynb",
    "4D": "4D_clusters_alvo.ipynb",
}


def cmd(nb: Path) -> list[str]:
    return [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute", "--inplace",
        f"--ExecutePreprocessor.timeout={TIMEOUT}",
        f"--ExecutePreprocessor.kernel_name={KERNEL}",
        str(nb),
    ]


def main(argv: list[str]) -> int:
    if "--parallel" in argv:
        print(
            "ERRO: a Fase 4 tem dependencias 4.0 -> 4A -> 4B -> 4C -> 4D; "
            "execucao paralela nao e valida."
        )
        return 2
    only = None
    if "--only" in argv:
        i = argv.index("--only")
        only = [a.upper() for a in argv[i + 1:] if not a.startswith("--")]
    keys = [k for k in NOTEBOOKS if only is None or k.upper() in only]
    if only is not None and not keys:
        valid = ", ".join(NOTEBOOKS)
        print(f"ERRO: nenhum notebook reconhecido em --only. Opcoes: {valid}")
        return 2
    nbs = [NBDIR / NOTEBOOKS[k] for k in keys]
    missing = [str(p) for p in nbs if not p.exists()]
    if missing:
        print("Notebooks ausentes:", missing); return 1

    print(f"Executando {len(nbs)} notebook(s) em modo SEQUENCIAL:")
    for p in nbs:
        print("  -", p.name)
    print()

    t0 = time.time()
    results: dict[str, str] = {}
    for p in nbs:
        print(f">>> {p.name} ...", flush=True)
        r = subprocess.run(cmd(p))
        results[p.name] = "OK" if r.returncode == 0 else "FALHOU"
        print(f"    {results[p.name]}\n", flush=True)
        if r.returncode != 0:
            print("Execucao interrompida: corrija o primeiro notebook com erro.")
            break

    print("=" * 60)
    for name, st in results.items():
        print(f"{st:8s} {name}")
    ok = all(v == "OK" for v in results.values())
    if ok:
        print(">>> export_numeric_tables_for_figures.py", flush=True)
        subprocess.run([
            sys.executable, "scripts/export_numeric_tables_for_figures.py",
            "--force", "--strict",
        ], cwd=ROOT, check=True)
    print(f"tempo total: {time.time()-t0:.0f}s")
    print("\nResultados gravados dentro de cada .ipynb (abra no VS Code).")
    print("Figuras: data/processed/figures/ | Tabelas: data/processed/parquet/statistics/")
    print("Numeric-tables: data/processed/numeric-tables/")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
