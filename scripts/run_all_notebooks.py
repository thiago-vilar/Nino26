#!/usr/bin/env python3
"""Executa TODOS os notebooks do NINO-BRASIL na ordem de dependência (Fase 2 -> 8).

Cada notebook é executado com `jupyter nbconvert --execute`. F5-F8 funcionam por
padrão como viewers de artefatos já concluídos (`NINO26_RUN_PIPELINE=0`); eles não
repetem treino. Ao final, o contrato estrutural é obrigatório e o gate separado
de linhagem semântica pode ser promovido com `--require-semantic-lineage`.

No WSL, escrever em `/mnt/c` pode falhar com `OSError: [Errno 22]` (bug do DrvFs).
Por isso o modo `--safe-write` (auto-ligado sob /mnt/) executa cada notebook num
diretório temporário do Linux e copia o resultado de volta com `cp`.

Exemplos:
    python scripts/run_all_notebooks.py                 # roda os 18 notebooks canonicos
    python scripts/run_all_notebooks.py --inputs        # gera insumos antes (master + fase3)
    python scripts/run_all_notebooks.py --only 3 4      # só as Fases 3 e 4
    python scripts/run_all_notebooks.py --continue-on-error
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.artifact_codes import parse_notebook_code
from nino_brasil.notebook_catalog import CANONICAL_NOTEBOOKS


ORDEM = [spec.relative_path.removeprefix("notebooks/") for spec in CANONICAL_NOTEBOOKS]
CODE_BY_PATH = {
    spec.relative_path.removeprefix("notebooks/"): spec.code
    for spec in CANONICAL_NOTEBOOKS
}


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> int:
    print(">>>", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def _copiar_de_volta(src: Path, dest: Path) -> bool:
    """Copia src->dest com `cp` (funciona no /mnt/c onde open('w') dá Errno 22)."""
    if subprocess.run(["cp", "-f", str(src), str(dest)]).returncode == 0:
        return True
    try:
        shutil.copyfile(src, dest)
        return True
    except OSError as exc:
        print(f"[erro] não consegui gravar {dest}: {exc}")
        return False


def executar_notebook(
    nb: Path,
    timeout: int,
    kernel: str,
    safe: bool,
    env: dict[str, str],
) -> int:
    """Executa em ``executed/`` e mantém o notebook-fonte sempre limpo."""
    base = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
            "--execute", f"--ExecutePreprocessor.timeout={timeout}",
            f"--ExecutePreprocessor.kernel_name={kernel}"]
    destination = nb.parent / "executed" / nb.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not safe:
        return run(
            base
            + [
                "--output-dir",
                str(destination.parent),
                "--output",
                destination.name,
                str(nb),
            ],
            env=env,
        )
    tmp = Path(tempfile.mkdtemp(prefix="nino_nb_"))
    try:
        rc = run(
            base + ["--output", nb.stem, "--output-dir", str(tmp), str(nb)],
            env=env,
        )
        out = tmp / f"{nb.stem}.ipynb"
        if rc == 0 and out.exists() and not _copiar_de_volta(out, destination):
            rc = 1
        return rc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _match(rel: str, tokens: list[str]) -> bool:
    code = CODE_BY_PATH[rel]
    parsed = parse_notebook_code(code)
    for raw in tokens:
        token = str(raw).strip().upper().replace("FASE", "F")
        if token.isdigit() and parsed.phase == int(token):
            return True
        if token == f"F{parsed.phase}":
            return True
        if code.upper().startswith(token):
            return True
    return False


def _has_complete_run(directory: Path, *, phase: int, mode: str) -> bool:
    """Accept only finalized artifacts; an abandoned run directory is not evidence."""

    for manifest_path in sorted(directory.glob(f"F{phase}_*/run_manifest.json"), reverse=True):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_phase = int(manifest.get("phase", -1))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if (
            manifest.get("status") != "complete"
            or manifest_phase != phase
            or manifest.get("mode") != mode
        ):
            continue
        if phase == 6 and manifest.get("parameters", {}).get("role") != "merge_pixel_shards_and_field_gate":
            continue
        if (manifest_path.parent / "tables_manifest.csv").exists():
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None,
                    help="filtra por fase/código (ex.: 3 4  ou  3C 4C)")
    ap.add_argument("--inputs", action="store_true",
                    help="gera somente o master F2; use os runners isolados para F3/F4")
    ap.add_argument("--build-chirps-target", action="store_true",
                    help="constroi/valida o alvo CHIRPS nativo antes dos notebooks (I/O longo)")
    ap.add_argument("--build-neural-input", action="store_true",
                    help="constroi o cubo espacial GLORYS semanal de F7/F8 (I/O longo)")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--kernel", default="nino-brasil")
    ap.add_argument("--mode", choices=("official", "smoke"), default="official")
    ap.add_argument(
        "--require-artifacts",
        action="store_true",
        help="falha se os artefatos correspondentes nao existirem antes do viewer",
    )
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument(
        "--require-semantic-lineage",
        action="store_true",
        help="promocao: falha se o gate figura<-tabela semantica<-run_id nao passar",
    )
    ap.add_argument("--safe-write", action=argparse.BooleanOptionalAction, default=None,
                    help="grava via tmp do Linux + cp (contorna Errno 22 do /mnt/c no WSL); "
                         "auto-ligado quando o projeto está sob /mnt/")
    args = ap.parse_args(argv)

    safe = args.safe_write if args.safe_write is not None else str(ROOT).startswith("/mnt/")
    if safe:
        print("[modo seguro] gravação via tmp do Linux + cp (evita Errno 22 do DrvFs em /mnt/c).")

    t0 = time.time()
    falhas: list[str] = []
    pendencias: list[str] = []
    if args.inputs:
        for pre in (["scripts/build_master_weekly.py", "--era5-years", "1981:2026", "--strict"],):
            if run([sys.executable, *pre]) != 0:
                falhas.append(f"input:{pre[0]}")
                if not args.continue_on_error:
                    print(f"[parou] insumo falhou: {pre[0]}")
                    return 1
    if args.build_chirps_target:
        if run([sys.executable, "scripts/build_phase4_chirps_targets.py"]) != 0:
            return 1
    if args.build_neural_input:
        if run([sys.executable, "scripts/build_phase7_pacific_cube.py"]) != 0:
            return 1

    alvo = [n for n in ORDEM if args.only is None or _match(n, args.only)]
    if not alvo:
        ap.error("--only did not match any canonical notebook")
    if args.require_artifacts:
        selected_phases = {parse_notebook_code(CODE_BY_PATH[rel]).phase for rel in alvo}
        requirements: dict[int, tuple[Path, ...]] = {
            4: (ROOT / "data/processed/zarr/features/chirps_native_weekly_targets.zarr",),
        }
        run_mode = "official" if args.mode == "official" else "smoke"
        for phase_no in (5, 6, 7, 8):
            requirements[phase_no] = (
                ROOT / f"data/processed/runs/{run_mode}/fase{phase_no}",
            )
        for phase_number in sorted(selected_phases):
            if phase_number == 3:
                scoped_namespaces = {
                    parse_notebook_code(CODE_BY_PATH[rel]).namespace
                    for rel in alvo
                    if parse_notebook_code(CODE_BY_PATH[rel]).phase == 3
                }
                for namespace in sorted(scoped_namespaces):
                    required = (
                        ROOT
                        / "data/processed/parquet/statistics"
                        / namespace
                        / f"Tab{namespace}B1_eventos.csv.manifest.json"
                    )
                    if not required.exists():
                        falhas.append(f"artifact:{required.relative_to(ROOT)}")
                continue
            for required in requirements.get(phase_number, ()):
                exists = required.exists()
                if exists and required.is_dir():
                    exists = _has_complete_run(
                        required,
                        phase=phase_number,
                        mode=run_mode,
                    )
                if not exists:
                    falhas.append(f"artifact:{required.relative_to(ROOT)}")
        if falhas and not args.continue_on_error:
            print("[preflight] artefatos ausentes:", ", ".join(falhas))
            return 1

    notebook_environment = os.environ.copy()
    notebook_environment.update(
        {"NINO26_NOTEBOOK_MODE": args.mode, "NINO26_RUN_PIPELINE": "0"}
    )
    current_pythonpath = notebook_environment.get("PYTHONPATH", "")
    notebook_environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(SRC), current_pythonpath) if value
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
    executed_ok = 0
    for i, rel in enumerate(alvo, 1):
        nb = NB / rel
        if not nb.exists():
            print(f"[erro] ausente: {rel}")
            falhas.append(rel)
            if not args.continue_on_error:
                return 1
            continue
        print(f"\n[{i}/{len(alvo)}] {rel}", flush=True)
        rc = executar_notebook(
            nb, args.timeout, args.kernel, safe, notebook_environment
        )
        if rc != 0:
            falhas.append(rel)
            if not args.continue_on_error:
                print(f"[parou] falhou: {rel}")
                return 1
        else:
            executed_ok += 1

    if not args.no_validate:
        if run([sys.executable, "scripts/validar_notebooks.py", "--strict"]) != 0:
            falhas.append("validar_notebooks.py --strict")
        if run([
            sys.executable,
            "scripts/validar_figuras.py",
            "--strict",
            "--allow-render-extraction",
            "--max-details",
            "20",
        ]) != 0:
            falhas.append("validar_figuras.py compatibility")
        lineage_rc = run([
            sys.executable,
            "scripts/validar_figuras.py",
            "--strict",
            "--max-details",
            "20",
        ])
        if lineage_rc != 0:
            pendencias.append("semantic figure lineage gate")
            print(
                "[linhagem pendente] notebooks/artefatos podem ser inspecionados, "
                "mas as figuras ainda nao passaram o gate semantico de promocao."
            )
            if args.require_semantic_lineage:
                falhas.append("semantic figure lineage gate")

    dt = time.time() - t0
    print(
        f"\nConcluído em {dt/60:.1f} min | executados={executed_ok} | "
        f"falhas={len(falhas)} | pendencias_de_promocao={len(pendencias)}"
    )
    if falhas:
        print("falharam:", ", ".join(falhas))
    if pendencias:
        print("pendentes de promocao:", ", ".join(pendencias))
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
