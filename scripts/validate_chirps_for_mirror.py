#!/usr/bin/env python3
"""Validate CHIRPS for the project mirror.

The default performs the expensive byte-level validation and refreshes its
receipt.  ``--reuse-valid-receipt`` is the fail-closed path for later mirror
checks: it accepts the prior deep result only while the target state, code,
manifest and promoted identities still match, without re-reading every data
chunk.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from nino_brasil.artifacts import sha256_file  # noqa: E402
from scripts.build_phase4_chirps_targets import (  # noqa: E402
    OUTPUT,
    _zarr_state_fingerprint,
    validate_promoted_target,
)


DEFAULT_REPORT = ROOT / "data/audit/chirps_deep_validation.json"


def write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--reuse-valid-receipt",
        type=Path,
        help=(
            "cheaply revalidate an existing deep receipt and do not rewrite it; "
            "fails closed instead of falling back to a full content hash"
        ),
    )
    args = parser.parse_args(argv)
    if args.reuse_valid_receipt is not None:
        from scripts.chirps_validation_receipt import (
            validate_deep_validation_receipt,
        )

        receipt_path = (
            args.reuse_valid_receipt
            if args.reuse_valid_receipt.is_absolute()
            else ROOT / args.reuse_valid_receipt
        )
        validation = validate_deep_validation_receipt(
            receipt_path,
            root=ROOT,
            active_target=OUTPUT,
            builder_script=ROOT / "scripts/build_phase4_chirps_targets.py",
            target_module=ROOT / "src/nino_brasil/targets/chirps_native.py",
        )
        print(
            json.dumps(
                {
                    "mode": "deep_receipt_plus_current_cheap_invariants",
                    "receipt": receipt_path.resolve().relative_to(ROOT).as_posix(),
                    "valid": True,
                    "validation": validation,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    result = validate_promoted_target(OUTPUT)
    manifest_value = str(result.get("manifest") or "").strip()
    manifest_path = ROOT / manifest_value if manifest_value else None
    payload: dict[str, object] = {
        "schema_version": "nino26-chirps-deep-validation/v1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_path": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "target_data_state_sha256": (
            _zarr_state_fingerprint(OUTPUT) if OUTPUT.is_dir() else None
        ),
        "builder_script_sha256": sha256_file(
            ROOT / "scripts/build_phase4_chirps_targets.py"
        ),
        "target_module_sha256": sha256_file(
            ROOT / "src/nino_brasil/targets/chirps_native.py"
        ),
        "build_manifest_sha256": (
            sha256_file(manifest_path)
            if manifest_path is not None and manifest_path.is_file()
            else None
        ),
        "validation": result,
    }
    write_report(report_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if result.get("valid") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
