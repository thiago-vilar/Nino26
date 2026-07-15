#!/usr/bin/env python3
"""Atualiza todas as fontes da F1 e, após sucesso, reconstrói a base semanal F2."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--skip-fase1", action="store_true", help="Reconstrói somente a F2 com os dados locais já publicados.")
    args = parser.parse_args(argv)

    if not args.skip_fase1:
        subprocess.run(
            [sys.executable, "scripts/run_full_download_pipeline.py", "--execute", "--stop-on-error"],
            cwd=ROOT,
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            "scripts/run_fase2_all.py",
            "--kernel",
            args.kernel,
            "--timeout",
            str(args.timeout),
        ],
        cwd=ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
