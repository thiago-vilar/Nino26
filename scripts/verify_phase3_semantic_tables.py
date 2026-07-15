#!/usr/bin/env python3
"""Verify Phase 3 semantic-table hashes and scientific lineage inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nino_brasil.stats.semantic_tables import verify_semantic_csv  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="data/processed/parquet/statistics",
        help="Diretorio com CSVs e manifests da F3.",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help=(
            "Glob opcional de CSVs. Por padrao, somente CSVs registrados por "
            "um manifest semantico phase3_*.manifest.json sao auditados."
        ),
    )
    parser.add_argument(
        "--include-unregistered",
        action="store_true",
        help=(
            "inclui todo phase3_*.csv, inclusive produtos legados sem manifest; "
            "use apenas em auditoria de migracao"
        ),
    )
    parser.add_argument("--skip-input-hashes", action="store_true")
    parser.add_argument("--report", default=None, help="CSV opcional com o resultado da auditoria.")
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    if not input_dir.exists():
        parser.error(f"input directory does not exist: {input_dir}")
    if args.pattern:
        tables = sorted(input_dir.glob(args.pattern))
    elif args.include_unregistered:
        tables = sorted(input_dir.glob("phase3_*.csv"))
    else:
        manifests = sorted(input_dir.glob("phase3_*.manifest.json"))
        tables = [
            manifest.with_name(
                manifest.name.removesuffix(".manifest.json")
            )
            for manifest in manifests
        ]
    if not tables:
        selection = args.pattern or (
            "phase3_*.csv" if args.include_unregistered else "registered phase3 manifests"
        )
        parser.error(f"no table matched {selection!r} in {input_dir}")

    rows = []
    for table in tables:
        result = verify_semantic_csv(table, verify_inputs=not args.skip_input_hashes)
        rows.append(
            {
                "table": table.name,
                "run_id": result.get("run_id"),
                "valid": result["valid"],
                "artifact_hash_ok": result["artifact_hash_ok"],
                "inputs_hash_ok": result["inputs_hash_ok"],
                "manifest_exists": result["manifest_exists"],
                "current_sha256": result.get("current_sha256"),
                "expected_sha256": result.get("expected_sha256"),
            }
        )
    report = pd.DataFrame(rows)
    if args.report:
        output = Path(args.report)
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output, index=False)
        print(f"[report] {output}")
    invalid = int((~report["valid"]).sum())
    print(report[["table", "valid", "artifact_hash_ok", "inputs_hash_ok"]].to_string(index=False))
    print(f"Phase 3 semantic audit: tables={len(report)} invalid={invalid}")
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
