#!/usr/bin/env python3
"""Executa a Fase 3: gera artefatos EN/LN, notebooks, relatorio e numeric-tables."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L")
DESCRIPTIVE_NAMES = {
    "A": "indices_fisicos_semanais",
    "B": "alvo_eventos_ciclo_vida",
    "C": "precursores_lags",
    "D": "rigor_estatistico",
    "E": "sensibilidade_temporal",
    "F": "kelvin_sla",
    "G": "compostos_ssta",
    "H": "genese_precursores_classe",
    "I": "interpretacao_integrada",
    "K": "pca_crescimento",
    "L": "caracterizacao_fases",
}
SCOPE_DIRECTORY = {"el_nino": "F3Nino", "la_nina": "F3Nina"}
NOTEBOOK_DIRECTORY = {"el_nino": "fase3_nino", "la_nina": "fase3_nina"}


def notebook_inventory(enso_type: str) -> dict[str, str]:
    prefix = SCOPE_DIRECTORY[enso_type]
    return {
        f"3{block}": f"{prefix}{block}_{DESCRIPTIVE_NAMES[block]}.ipynb"
        for block in BLOCKS
    }


def canonical_phase3_run_id(stats_dir: Path, enso_type: str) -> str:
    prefix = SCOPE_DIRECTORY[enso_type]
    manifest = stats_dir / f"Tab{prefix}B1_eventos.csv.manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifesto âncora F3 ausente: {manifest}")
    run_id = str(json.loads(manifest.read_text(encoding="utf-8")).get("run_id", "")).strip()
    if not run_id:
        raise RuntimeError(f"run_id ausente no manifesto âncora F3: {manifest}")
    return run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--kernel", default="nino-brasil")
    all_codes = tuple(f"3{block}" for block in BLOCKS)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--only",
        nargs="+",
        choices=all_codes,
        help="executa somente os codigos indicados, por exemplo --only 3G 3H",
    )
    selection.add_argument(
        "--start-at",
        choices=all_codes,
        help="retoma neste codigo e segue ate o ultimo notebook F3",
    )
    parser.add_argument(
        "--skip-core",
        action="store_true",
        help="retomar nos notebooks quando phase3_en_ln ja concluiu e foi verificada",
    )
    parser.add_argument(
        "--enso-type",
        choices=("el_nino", "la_nina"),
        default=None,
        help="executa a F3 isolada para El Nino ou La Nina",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="gera apenas as tabelas cientificas scoped, sem executar notebooks",
    )
    args = parser.parse_args(argv)

    if args.enso_type is None:
        parser.error("a F3 oficial exige --enso-type el_nino ou la_nina")

    notebook_codes = notebook_inventory(args.enso_type)
    notebooks = list(notebook_codes.values())
    nbdir = ROOT / "notebooks" / NOTEBOOK_DIRECTORY[args.enso_type]

    stats_dir = (
        ROOT / "data/processed/parquet/statistics" / SCOPE_DIRECTORY[args.enso_type]
    )

    t0 = time.time()
    # O núcleo novo depende apenas do master F2 e do ONI local. Os antigos
    # caches de mapas/Hovmöller não são mais reconstruídos por este runner.
    if not args.skip_core:
        print(">>> phase3_en_ln.py", flush=True)
        core_command = [sys.executable, "scripts/phase3_en_ln.py"]
        if args.enso_type:
            core_command.extend(["--enso-type", args.enso_type])
        subprocess.run(core_command, cwd=ROOT, check=True)

    if args.core_only:
        print(f"Fase 3 core concluida: {stats_dir.relative_to(ROOT)}")
        return 0

    notebook_environment = os.environ.copy()
    notebook_environment.update(
        {
            "NINO26_NOTEBOOK_MODE": "official",
            "NINO26_RUN_PIPELINE": "0",
            "NINO_RUN_ID": canonical_phase3_run_id(stats_dir, args.enso_type),
        }
    )
    local_jupyter = ROOT / ".venv" / "share" / "jupyter"
    if local_jupyter.exists():
        current = notebook_environment.get("JUPYTER_PATH", "")
        notebook_environment["JUPYTER_PATH"] = os.pathsep.join(
            value for value in (str(local_jupyter), current) if value
        )
    ipython_dir = ROOT / "data" / "interim" / "notebook-runtime" / "ipython"
    jupyter_runtime = ROOT / "data" / "interim" / "notebook-runtime" / "jupyter"
    ipython_dir.mkdir(parents=True, exist_ok=True)
    jupyter_runtime.mkdir(parents=True, exist_ok=True)
    notebook_environment["IPYTHONDIR"] = str(ipython_dir)
    notebook_environment["JUPYTER_RUNTIME_DIR"] = str(jupyter_runtime)
    # Todos os onze blocos possuem agora implementacao fisicamente orientada
    # ao sinal selecionado; La Nina nao reutiliza janelas quentes de El Nino.
    selected_notebooks = notebooks
    if args.only:
        selected_notebooks = [notebook_codes[code] for code in args.only]
    elif args.start_at:
        selected = notebook_codes[args.start_at]
        start = notebooks.index(selected)
        selected_notebooks = notebooks[start:]
    executed_dir = nbdir / "executed"
    executed_dir.mkdir(parents=True, exist_ok=True)
    for name in selected_notebooks:
        nb = nbdir / name
        if not nb.exists():
            raise FileNotFoundError(nb)
        print(f">>> {name}", flush=True)
        notebook_command = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute",
            f"--ExecutePreprocessor.timeout={args.timeout}",
            f"--ExecutePreprocessor.kernel_name={args.kernel}",
        ]
        if args.enso_type == "el_nino":
            # A F3Nino é um conjunto científico de leitura direta: todos os
            # notebooks entregues preservam tabelas e figuras inline no arquivo
            # que o pesquisador abre, além dos PNG/CSV auditáveis em disco.
            notebook_command.append("--inplace")
        else:
            notebook_command.extend(
                ["--output-dir", str(executed_dir), "--output", name]
            )
        notebook_command.append(str(nb))
        subprocess.run(
            notebook_command,
            cwd=ROOT,
            env=notebook_environment,
            check=True,
        )

    print(">>> verify_phase3_semantic_tables.py (escopo isolado)", flush=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/verify_phase3_semantic_tables.py",
            "--input-dir",
            str(stats_dir),
            "--pattern",
            "TabF3*.csv",
        ],
        cwd=ROOT,
        check=True,
    )
    print(">>> validar_notebooks.py --strict", flush=True)
    subprocess.run(
        [sys.executable, "scripts/validar_notebooks.py", "--strict"],
        cwd=ROOT,
        check=True,
    )
    print(">>> validar_figuras.py --strict", flush=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/validar_figuras.py",
            "--strict",
            "--max-details",
            "20",
        ],
        cwd=ROOT,
        check=True,
    )
    print(
        f"Fase 3 {args.enso_type} executada em {time.time() - t0:.0f}s | "
        f"tabelas verificadas em {stats_dir.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
