#!/usr/bin/env python3
"""Executa F8 ConvLSTM no CHIRPS nativo e publica o notebook F8A."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.phase_runner_utils import ROOT, execute_viewer, python, run


CUBE = ROOT / "data/processed/zarr/modeling/phase7_pacific_weekly.zarr"
CHIRPS = ROOT / "data/processed/zarr/features/chirps_native_weekly_targets.zarr"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="nino-brasil")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    run(python("scripts/build_phase3_modeling_bridge.py"))
    if not CUBE.exists():
        run(python("scripts/build_phase7_pacific_cube.py"))
    if not CHIRPS.exists():
        raise FileNotFoundError(
            "CHIRPS nativo validado ausente; rode scripts/build_phase4_chirps_targets.py"
        )
    run(
        python(
            "scripts/run_fase8_brazil_convlstm.py",
            "--mode",
            "official",
            "--device",
            args.device,
        )
    )
    execute_viewer("F8A", kernel=args.kernel, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
