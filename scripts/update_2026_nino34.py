#!/usr/bin/env python3
"""Planeja ou executa a atualizacao 2026 do recorte Nino 3.4.

O objetivo e trazer cada fonte ate o maximo operacional permitido pela sua
latencia e depois reconstruir a Fase 3. Por padrao o script apenas imprime o
plano; use --execute para rodar downloads/processamentos externos.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def cmd(*parts: str) -> list[str]:
    return [str(PY), *parts]


def run_step(title: str, command: list[str], *, execute: bool) -> None:
    print(f"\n== {title}")
    print(" ".join(command))
    if execute:
        subprocess.run(command, cwd=ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="roda os comandos; sem isso e dry-run")
    parser.add_argument("--start-year", type=int, default=2026, help="ano inicial da atualizacao operacional")
    parser.add_argument("--glorys-operational-start", default="2026-05-27")
    parser.add_argument("--skip-downloads", action="store_true", help="reconstroi somente produtos da Fase 3")
    args = parser.parse_args(argv)

    today = date.today()
    print(f"Atualizacao Nino 3.4 gerada em {today.isoformat()} | modo={'EXECUTE' if args.execute else 'DRY-RUN'}")
    print("Latencias usadas pelo projeto: OISST/ERA5=7d, GLO12 operacional=1d, ORAS5=15d, in situ=3d.")
    print(f"Alvo operacional aproximado: OISST/ERA5 ate {today - timedelta(days=7)}, GLO12 ate {today - timedelta(days=1)}.")

    if not args.skip_downloads:
        run_step(
            "OISST 2026: baixa/curadoria e reprocessa SST/SSTA local",
            cmd(
                "scripts/curate_and_resume_downloads.py",
                "--source",
                "oisst",
                "--stage",
                "all",
                "--start-year",
                str(args.start_year),
                "--execute",
                "--limit",
                "0",
                "--retries",
                "5",
                "--retry-wait",
                "60",
                "--continue-on-error",
            ),
            execute=args.execute,
        )
        run_step(
            "ERA5 2026 Nino 3.4: meses completos disponiveis",
            cmd(
                "scripts/data_pipeline.py",
                "download-era5",
                "--start-year",
                str(args.start_year),
                "--kind",
                "both",
                "--region",
                "nino34",
                "--execute",
                "--continue-on-error",
            ),
            execute=args.execute,
        )
        run_step(
            "GLO12 operacional: cauda diaria de analise para Nino 3.4/Pacifico equatorial",
            cmd(
                "scripts/ocean_daily_pipeline.py",
                "ingest-glorys-operational",
                "--start-date",
                args.glorys_operational_start,
                "--delete-source-after-zarr",
                "--execute",
            ),
            execute=args.execute,
        )
        run_step(
            "ORAS5 mensal 2026: memoria mensal independente",
            cmd(
                "scripts/ocean_monthly_pipeline.py",
                "ingest",
                "--start-year",
                str(args.start_year),
                "--build-features",
                "--delete-raw-after-zarr",
                "--execute",
            ),
            execute=args.execute,
        )
        run_step(
            "TAO/TRITON e Argo: validacao in situ ate a data suportada pela API",
            cmd(
                "scripts/data_pipeline.py",
                "download-validation",
                "--source",
                "all",
                "--start-year",
                str(args.start_year),
                "--max-depth",
                "300",
                "--execute",
                "--continue-on-error",
            ),
            execute=args.execute,
        )

    rebuild = [
        ("Fase 3: indice diario Nino 3.4", cmd("scripts/data_pipeline.py", "build-nino34-daily-index")),
        ("Fase 3: referencia mensal OISST/ONI local", cmd("scripts/data_pipeline.py", "build-nino34-sst-reference")),
        ("Fase 3: diagnosticos fisicos", cmd("scripts/data_pipeline.py", "build-phase3-diagnostics")),
        ("Fase 3: auditoria", cmd("scripts/data_pipeline.py", "audit-phase3-diagnostics")),
        ("Fase 3: cache atmosferico ERA5", cmd("scripts/update_era5_nino34_atmo_cache.py", "--start-year", "1981", "--end-year", str(today.year))),
        ("Fase 3: insumos 3A-3I", cmd("scripts/fase3_build_inputs.py", "--force")),
        ("Painel executivo", cmd("scripts/update_painel_executivo.py")),
    ]
    for title, command in rebuild:
        run_step(title, command, execute=args.execute)

    print("\nDepois do --execute, rode: .venv\\Scripts\\python scripts\\run_fase3_all.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
