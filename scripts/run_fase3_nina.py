#!/usr/bin/env python3
"""Executa a FASE_3_NINA completa no inventario cientificamente valido."""
from __future__ import annotations

import sys

from run_fase3_all import main


if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "--enso-type", "la_nina"]))
