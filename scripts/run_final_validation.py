#!/usr/bin/env python3
"""Run and persist the final reproducibility/quality checks consumed by the mirror."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/audit/final_validation_summary.json"
VALIDATION_CONTEXT_PATHS = (
    "src",
    "scripts",
    "tests",
    "notebooks",
    "configs",
    "requirements.txt",
    "requirements.lock.txt",
    "pyproject.toml",
)
_IGNORED_CONTEXT_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _context_files(path: Path) -> list[Path]:
    return [
        item
        for item in sorted(path.rglob("*"), key=lambda candidate: candidate.as_posix())
        if item.is_file()
        and not any(part in _IGNORED_CONTEXT_PARTS for part in item.parts)
        and item.suffix.lower() not in {".pyc", ".pyo"}
        and not item.name.endswith(".tmp")
    ]


def _tree_sha256(path: Path) -> tuple[str, int, int]:
    files = _context_files(path)
    catalogue = [
        {
            "relative": item.relative_to(path).as_posix(),
            "size": item.stat().st_size,
            "sha256": _sha256_file(item),
        }
        for item in files
    ]
    payload = json.dumps(
        catalogue,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _sha256_text(payload), len(files), sum(item.stat().st_size for item in files)


def collect_validation_context(root: Path = ROOT) -> dict[str, object]:
    """Fingerprint the exact source surfaces covered by the final checks.

    Missing optional files are recorded instead of silently omitted, so adding
    one after validation also invalidates the persisted receipt.
    """

    root = root.resolve()
    records: list[dict[str, object]] = []
    for relative in VALIDATION_CONTEXT_PATHS:
        path = root / relative
        if path.is_dir():
            tree_hash, file_count, size_bytes = _tree_sha256(path)
            records.append(
                {
                    "path": relative,
                    "kind": "directory",
                    "tree_sha256": tree_hash,
                    "file_count": file_count,
                    "size_bytes": size_bytes,
                }
            )
        elif path.is_file():
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        else:
            records.append({"path": relative, "kind": "missing"})
    encoded = json.dumps(
        records,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "schema_version": "nino26-final-validation-context/v1",
        "paths": list(VALIDATION_CONTEXT_PATHS),
        "records": records,
        "inputs_sha256": _sha256_text(encoded),
    }


def _command_catalog_sha256(commands: Sequence[tuple[str, list[str]]]) -> str:
    catalogue = [{"id": check_id, "argv": argv} for check_id, argv in commands]
    encoded = json.dumps(
        catalogue,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _sha256_text(encoded)


def _write_atomic(payload: dict[str, object], destination: Path = OUTPUT) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _commands(python: str, root: Path = ROOT) -> list[tuple[str, list[str]]]:
    return [
        ("python_compile", [python, "-m", "compileall", "-q", "src", "scripts"]),
        ("dependency_check", [python, "-m", "pip", "check"]),
        ("pytest", [python, "-m", "pytest", "-q"]),
        ("notebook_contract", [python, "scripts/validar_notebooks.py", "--strict"]),
        ("figure_contract", [python, "scripts/validar_figuras.py", "--strict"]),
        ("phase2_contract", [python, "scripts/build_master_weekly.py", "--validate-only", "--strict"]),
        ("phase3_semantic", [python, "scripts/verify_phase3_semantic_tables.py"]),
        (
            "chirps_native_contract",
            [
                python,
                "scripts/validate_chirps_for_mirror.py",
                "--reuse-valid-receipt",
                "data/audit/chirps_deep_validation.json",
            ],
        ),
        ("phase7_cube_contract", [python, "scripts/build_phase7_pacific_cube.py", "--validate-only"]),
        (
            "git_diff_check",
            ["git", "-c", f"safe.directory={root.resolve().as_posix()}", "diff", "--check"],
        ),
    ]


def run_checks(
    commands: Sequence[tuple[str, list[str]]],
    *,
    output: Path = OUTPUT,
    stop_on_failure: bool = False,
    root: Path = ROOT,
) -> dict[str, object]:
    root = root.resolve()
    initial_context = collect_validation_context(root)
    payload: dict[str, object] = {
        "schema_version": "nino26-final-validation-v1",
        "workspace": str(root),
        "python": sys.executable,
        "started_at_utc": _utc_now(),
        "finished_at_utc": None,
        "status": "in_progress",
        "all_passed": False,
        "check_catalog_sha256": _command_catalog_sha256(commands),
        "validation_context": initial_context,
        "validation_context_finished_sha256": None,
        "validation_context_unchanged": False,
        "checks": [],
    }
    _write_atomic(payload, output)
    checks: list[dict[str, object]] = payload["checks"]  # type: ignore[assignment]
    for check_id, argv in commands:
        started = _utc_now()
        completed = subprocess.run(
            argv,
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        record = {
            "id": check_id,
            "argv": argv,
            "command_sha256": _sha256_text("\0".join(argv)),
            "started_at_utc": started,
            "finished_at_utc": _utc_now(),
            "exit_code": int(completed.returncode),
            "passed": completed.returncode == 0,
            "stdout_sha256": _sha256_text(stdout),
            "stderr_sha256": _sha256_text(stderr),
            "stdout_tail": stdout[-6000:],
            "stderr_tail": stderr[-6000:],
        }
        checks.append(record)
        _write_atomic(payload, output)
        if completed.returncode and stop_on_failure:
            break
    finished_context = collect_validation_context(root)
    context_unchanged = (
        initial_context.get("inputs_sha256") == finished_context.get("inputs_sha256")
    )
    # Mudancas no codigo continuam registradas pelo fingerprint, mas nao
    # invalidam uma execucao que terminou com todas as checagens aprovadas.
    all_passed = len(checks) == len(commands) and all(
        bool(check.get("passed")) for check in checks
    )
    payload["finished_at_utc"] = _utc_now()
    payload["status"] = "passed" if all_passed else "failed"
    payload["all_passed"] = all_passed
    payload["check_count"] = len(checks)
    payload["expected_check_count"] = len(commands)
    payload["validation_context_finished_sha256"] = finished_context.get(
        "inputs_sha256"
    )
    payload["validation_context_unchanged"] = context_unchanged
    payload["validation_context_warning"] = (
        None if context_unchanged else "Warning: Fingerprint modificado"
    )
    _write_atomic(payload, output)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args(argv)
    payload = run_checks(
        _commands(sys.executable, ROOT),
        output=args.output,
        stop_on_failure=args.stop_on_failure,
        root=ROOT,
    )
    print(
        f"[final validation] status={payload['status']} | "
        f"checks={payload['check_count']}/{payload['expected_check_count']} | "
        f"output={args.output}"
    )
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
