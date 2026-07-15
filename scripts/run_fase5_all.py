#!/usr/bin/env python3
"""Executa F5 RF/XGBoost, ablação augmentation ON/OFF e notebook F5A."""
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


def _augmentation_active(manifest: dict[str, object]) -> bool:
    parameters = manifest.get("parameters") or {}
    return bool(
        int(parameters.get("noise_copies", 0) or 0) > 0
        or float(parameters.get("mixup_alpha", 0.0) or 0.0) > 0
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="nino-brasil")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--model", choices=("rf", "xgb"), action="append")
    args = parser.parse_args(argv)
    models = tuple(dict.fromkeys(args.model or ("rf", "xgb")))
    run(python("scripts/build_phase3_modeling_bridge.py"))
    for model in models:
        run(python("scripts/run_fase5_cycle_ml.py", "--mode", "official", "--model", model))
        augmented = latest_complete_run(
            5,
            lambda manifest, selected=model: (
                str((manifest.get("parameters") or {}).get("model")) == selected
                and _augmentation_active(manifest)
            ),
        )
        run(
            python(
                "scripts/run_fase5_cycle_ml.py",
                "--mode",
                "official",
                "--model",
                model,
                "--no-augmentation",
            )
        )
        unaugmented = latest_complete_run(
            5,
            lambda manifest, selected=model: (
                str((manifest.get("parameters") or {}).get("model")) == selected
                and not _augmentation_active(manifest)
            ),
        )
        output = ROOT / "data/audit/augmentation_ablation" / (
            f"F5A_{model}_{augmented.name}_{unaugmented.name}"
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
    execute_viewer("F5A", kernel=args.kernel, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
