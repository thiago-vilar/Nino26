#!/usr/bin/env python3
"""Executa FASE_4_NINO: F4C -> F4D -> notebooks auditaveis, em sequencia."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_fase4_all import main as run_fase4


def main(argv: list[str] | None = None) -> int:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    return run_fase4(
        ["--run-pipeline", *forwarded, "--enso-type", "el_nino"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
