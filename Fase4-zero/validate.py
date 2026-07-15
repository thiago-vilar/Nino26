#!/usr/bin/env python3
"""Cria ou valida o inventario imutavel da restauracao Fase4-zero."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


CAPSULE = Path(__file__).resolve().parent
MANIFEST = CAPSULE / "RESTORATION_MANIFEST.json"
EXPECTED_NOTEBOOKS = (
    "4_0_fase4_abertura.ipynb",
    "4A_ciclo_enso_fases.ipynb",
    "4B_variaveis_determinantes_fases.ipynb",
    "4C_sinal_pixel_lags.ipynb",
    "4D_clusters_alvo.ipynb",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory() -> tuple[list[dict[str, object]], str]:
    rows: list[dict[str, object]] = []
    for path in sorted(CAPSULE.rglob("*")):
        if not path.is_file() or path == MANIFEST:
            continue
        relative = path.relative_to(CAPSULE).as_posix()
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    tree = hashlib.sha256()
    for row in rows:
        tree.update(
            f"{row['path']}\0{row['size_bytes']}\0{row['sha256']}\n".encode("utf-8")
        )
    return rows, tree.hexdigest()


def notebook_inventory() -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    problems: list[str] = []
    notebook_root = CAPSULE / "notebooks"
    observed = {path.name for path in notebook_root.glob("*.ipynb")}
    problems.extend(
        f"missing_notebook:{name}"
        for name in sorted(set(EXPECTED_NOTEBOOKS) - observed)
    )
    problems.extend(
        f"unexpected_notebook:{name}"
        for name in sorted(observed - set(EXPECTED_NOTEBOOKS))
    )
    for name in EXPECTED_NOTEBOOKS:
        path = notebook_root / name
        if not path.exists():
            continue
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"invalid_notebook:{name}:{exc}")
            continue
        cells = notebook.get("cells", [])
        outputs = [
            output
            for cell in cells
            if cell.get("cell_type") == "code"
            for output in cell.get("outputs", [])
        ]
        errors = sum(output.get("output_type") == "error" for output in outputs)
        if errors:
            problems.append(f"stored_notebook_error:{name}:{errors}")
        rows.append(
            {
                "name": name,
                "cells": len(cells),
                "markdown_cells": sum(cell.get("cell_type") == "markdown" for cell in cells),
                "code_cells": sum(cell.get("cell_type") == "code" for cell in cells),
                "stored_outputs": len(outputs),
                "stored_errors": errors,
                "sha256": sha256_file(path),
            }
        )
    return rows, problems


def count_files(path: Path) -> int:
    return len([item for item in path.rglob("*") if item.is_file()])


def build_manifest() -> dict[str, object]:
    files, tree_sha256 = inventory()
    notebooks, problems = notebook_inventory()
    if problems:
        raise RuntimeError("; ".join(problems))
    return {
        "schema_version": "nino26-fase4-zero-restoration-v1",
        "label": "Fase4-zero",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "quarantine": "data/quarantine/output_reset_20260713T220000BRT",
            "pre_review_commit": "61ae278",
            "restoration_policy": "byte_preserving_copy_no_scientific_reinterpretation",
        },
        "counts": {
            "notebooks": len(notebooks),
            "figures": count_files(CAPSULE / "outputs/figures"),
            "numeric_table_files": count_files(CAPSULE / "outputs/numeric-tables"),
            "feature_files": count_files(CAPSULE / "outputs/parquet/features"),
            "modeling_files": count_files(CAPSULE / "outputs/parquet/modeling"),
            "statistics_files": count_files(CAPSULE / "outputs/parquet/statistics"),
            "zarr_files": count_files(CAPSULE / "outputs/zarr"),
            "all_files_excluding_manifest": len(files),
            "total_bytes_excluding_manifest": sum(int(row["size_bytes"]) for row in files),
        },
        "tree_sha256": tree_sha256,
        "notebooks": notebooks,
        "files": files,
    }


def validate() -> list[str]:
    if not MANIFEST.exists():
        return ["missing_restoration_manifest"]
    try:
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid_restoration_manifest:{exc}"]
    observed, tree_sha256 = inventory()
    expected_files = {str(row["path"]): row for row in expected.get("files", [])}
    observed_files = {str(row["path"]): row for row in observed}
    problems: list[str] = []
    problems.extend(
        f"missing_file:{path}"
        for path in sorted(set(expected_files) - set(observed_files))
    )
    problems.extend(
        f"unexpected_file:{path}"
        for path in sorted(set(observed_files) - set(expected_files))
    )
    for path in sorted(set(expected_files) & set(observed_files)):
        if expected_files[path] != observed_files[path]:
            problems.append(f"file_hash_or_size_mismatch:{path}")
    if expected.get("tree_sha256") != tree_sha256:
        problems.append("tree_sha256_mismatch")
    _, notebook_problems = notebook_inventory()
    problems.extend(notebook_problems)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args(argv)
    if args.write_manifest:
        payload = build_manifest()
        MANIFEST.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[gravado] {MANIFEST}")
    problems = validate()
    if problems:
        for problem in problems:
            print(f"[erro] {problem}")
        return 1
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    counts = payload["counts"]
    print(
        "[ok] Fase4-zero integra: "
        f"notebooks={counts['notebooks']}, figuras={counts['figures']}, "
        f"numeric_files={counts['numeric_table_files']}, "
        f"statistics_files={counts['statistics_files']}, "
        f"zarr_files={counts['zarr_files']}, "
        f"tree_sha256={payload['tree_sha256'][:16]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

