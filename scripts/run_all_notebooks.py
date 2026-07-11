#!/usr/bin/env python3
"""Executa TODOS os notebooks do NINO-BRASIL na ordem de dependência (Fase 2 -> 8).

Cada notebook é executado com `jupyter nbconvert --execute --inplace`, de modo que
tabelas e figuras são reescritas por sobreposição dentro do próprio .ipynb e nas
pastas `data/processed/figures` e `numeric-tables` (via nino_brasil.viz). Ao final
roda o validador de contrato (figura <-> numeric-table <-> manifesto).

Exemplos:
    python scripts/run_all_notebooks.py                 # roda os 21 em ordem
    python scripts/run_all_notebooks.py --inputs        # gera insumos antes (master + fase3)
    python scripts/run_all_notebooks.py --only 3 4      # só as Fases 3 e 4
    python scripts/run_all_notebooks.py --continue-on-error
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

# Ordem de dependência (Fase 2 -> 8). Fase 3 antes da 4; 4 antes de 5/6; etc.
ORDEM = [
    "fase2/2Z_sanidade_variaveis.ipynb",
    "fase3/3A_indices_fisicos_semanais.ipynb",
    "fase3/3B_alvo_eventos_ciclo_vida.ipynb",
    "fase3/3C_precursores_lags.ipynb",
    "fase3/3D_rigor_estatistico.ipynb",
    "fase3/3E_sensibilidade_temporal.ipynb",
    "fase3/3F_kelvin_sla.ipynb",
    "fase3/3G_compostos_ssta.ipynb",
    "fase3/3H_genese_precursores_classe.ipynb",
    "fase3/3K_pca_crescimento.ipynb",
    "fase3/3I_interpretacao_integrada.ipynb",
    "fase3/3L_en_ln_caracterizacao.ipynb",
    "fase4/4_0_fase4_abertura.ipynb",
    "fase4/4A_ciclo_enso_fases.ipynb",
    "fase4/4B_variaveis_determinantes_fases.ipynb",
    "fase4/4C_sinal_pixel_lags.ipynb",
    "fase4/4D_clusters_alvo.ipynb",
    "fase5/5_ciclo_ml.ipynb",
    "fase6/6_brasil_ml.ipynb",
    "fase7/7_ciclo_convlstm.ipynb",
    "fase8/8_brasil_convlstm.ipynb",
]


def run(cmd: list[str]) -> int:
    print(">>>", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None,
                    help="filtra por prefixo de fase/código (ex.: 3 4  ou  3C 4C)")
    ap.add_argument("--inputs", action="store_true",
                    help="gera insumos antes (build_master_weekly + fase3_build_inputs + phase3_en_ln)")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--kernel", default="python3")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    if args.inputs:
        for pre in (["scripts/build_master_weekly.py", "--era5-years", "1981:2026"],
                    ["scripts/fase3_build_inputs.py", "--force"],
                    ["scripts/phase3_en_ln.py"]):
            if run([sys.executable, *pre]) != 0 and not args.continue_on_error:
                print(f"[parou] insumo falhou: {pre[0]}"); return 1

    def _match(rel: str, tokens: list[str]) -> bool:
        fase = rel.split("/")[0]            # ex.: "fase3"
        nome = Path(rel).name.upper()       # ex.: "3C_...IPYNB"
        return any(fase == f"fase{t}" or fase.endswith(t) or nome.startswith(t.upper())
                   for t in tokens)

    alvo = [n for n in ORDEM if args.only is None or _match(n, args.only)]

    falhas = []
    for i, rel in enumerate(alvo, 1):
        nb = NB / rel
        if not nb.exists():
            print(f"[aviso] ausente: {rel}"); continue
        print(f"\n[{i}/{len(alvo)}] {rel}", flush=True)
        rc = run([sys.executable, "-m", "jupyter", "nbconvert",
                  "--to", "notebook", "--execute", "--inplace",
                  f"--ExecutePreprocessor.timeout={args.timeout}",
                  f"--ExecutePreprocessor.kernel_name={args.kernel}", str(nb)])
        if rc != 0:
            falhas.append(rel)
            if not args.continue_on_error:
                print(f"[parou] falhou: {rel}"); return 1

    if not args.no_validate:
        run([sys.executable, "scripts/validar_figuras.py"])

    dt = time.time() - t0
    print(f"\nConcluído em {dt/60:.1f} min | executados={len(alvo)-len(falhas)} | falhas={len(falhas)}")
    if falhas:
        print("falharam:", ", ".join(falhas))
    return 1 if falhas and not args.continue_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
