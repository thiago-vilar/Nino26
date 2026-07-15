#!/usr/bin/env python3
"""Executa F7 com augmentation ON/OFF, compara os braços e publica F7A."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.phase_runner_utils import (
    ROOT,
    execute_viewer,
    latest_complete_run,
    python,
    run,
)


CUBE = ROOT / "data/processed/zarr/modeling/phase7_pacific_weekly.zarr"


def _augmented(manifest: dict[str, object]) -> bool:
    return bool((manifest.get("parameters") or {}).get("augmentation", False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="nino-brasil")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rebuild-cube", action="store_true")
    args = parser.parse_args(argv)
    run(python("scripts/build_phase3_modeling_bridge.py"))
    if args.rebuild_cube or not CUBE.exists():
        cube_command = python("scripts/build_phase7_pacific_cube.py")
        if args.rebuild_cube:
            cube_command.append("--overwrite")
        run(cube_command)
    run(
        python(
            "scripts/run_fase7_cycle_convlstm.py",
            "--mode",
            "official",
            "--device",
            args.device,
        )
    )
    augmented = latest_complete_run(7, _augmented)
    run(
        python(
            "scripts/run_fase7_cycle_convlstm.py",
            "--mode",
            "official",
            "--device",
            args.device,
            "--no-augmentation",
        )
    )
    unaugmented = latest_complete_run(7, lambda manifest: not _augmented(manifest))
    output = ROOT / "data/audit/augmentation_ablation" / (
        f"F7A_{augmented.name}_{unaugmented.name}"
    )
    run(
        python(
            "scripts/compare_augmentation_ablation.py",
            "--with-augmentation",
            str(augmented),
            "--without-augmentation",
            str(unaugmented),
            "--output-dir",
            str(output),
            "--write",
        )
    )
    execute_viewer("F7A", kernel=args.kernel, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
