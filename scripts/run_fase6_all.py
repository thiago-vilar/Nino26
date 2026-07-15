#!/usr/bin/env python3
"""Executa todos os pixels CHIRPS em F6 para RF/XGBoost e publica F6A."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.phase_runner_utils import execute_viewer, python, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="nino-brasil")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--model", choices=("rf", "xgb"), action="append")
    parser.add_argument("--shard-size", type=int, default=500)
    args = parser.parse_args(argv)
    models = tuple(dict.fromkeys(args.model or ("rf", "xgb")))
    run(python("scripts/build_phase3_modeling_bridge.py"))
    for model in models:
        run(
            python(
                "scripts/run_fase6_all_shards.py",
                "--model",
                model,
                "--shard-size",
                str(args.shard_size),
            )
        )
    execute_viewer("F6A", kernel=args.kernel, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
