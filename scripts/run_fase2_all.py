#!/usr/bin/env python3
"""Trata dados locais da F1, audita a F2 e publica o notebook F2Z.

Este runner não baixa fontes. A atualização de OISST, ERA5, CHIRPS,
UFS+GLORYS, in situ e IBGE pertence exclusivamente à Fase 1.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.phase_runner_utils import execute_viewer, python, run


def ensure_processed_output_roots() -> None:
    """Mantém disponíveis os destinos oficiais mesmo após limpeza dos produtos."""
    for relative in ("data/processed/figures", "data/processed/numeric-tables"):
        (ROOT / relative).mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era5-years", default=f"1981:{datetime.now().year}")
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    ensure_processed_output_roots()
    command = python(
        "scripts/build_master_weekly.py",
        "--era5-years",
        args.era5_years,
        "--strict",
    )
    if args.validate_only:
        command.append("--validate-only")
    run(command)
    execute_viewer("F2Z", kernel=args.kernel, timeout=args.timeout)
    execute_viewer("F2V", kernel=args.kernel, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
