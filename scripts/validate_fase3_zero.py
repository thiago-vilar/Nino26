#!/usr/bin/env python3
"""Cria ou valida o inventario imutavel da restauracao Fase3-zero."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CAPSULE = ROOT / "Fase3-zero"
MANIFEST = CAPSULE / "RESTORATION_MANIFEST.json"
EXPECTED_NOTEBOOKS = (
    "3A_indices_fisicos_semanais.ipynb",
    "3B_alvo_eventos_ciclo_vida.ipynb",
    "3C_precursores_lags.ipynb",
    "3D_rigor_estatistico.ipynb",
    "3E_sensibilidade_temporal.ipynb",
    "3F_kelvin_sla.ipynb",
    "3G_compostos_ssta.ipynb",
    "3H_genese_precursores_classe.ipynb",
    "3I_interpretacao_integrada.ipynb",
    "3K_pca_crescimento.ipynb",
    "3L_en_ln_caracterizacao.ipynb",
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
    missing = sorted(set(EXPECTED_NOTEBOOKS) - observed)
    unexpected = sorted(observed - set(EXPECTED_NOTEBOOKS))
    problems.extend(f"missing_notebook:{name}" for name in missing)
    problems.extend(f"unexpected_notebook:{name}" for name in unexpected)
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
        error_count = sum(output.get("output_type") == "error" for output in outputs)
        if error_count:
            problems.append(f"stored_notebook_error:{name}:{error_count}")
        rows.append(
            {
                "name": name,
                "cells": len(cells),
                "markdown_cells": sum(cell.get("cell_type") == "markdown" for cell in cells),
                "code_cells": sum(cell.get("cell_type") == "code" for cell in cells),
                "stored_outputs": len(outputs),
                "stored_errors": error_count,
                "sha256": sha256_file(path),
            }
        )
    return rows, problems


def build_manifest() -> dict[str, object]:
    files, tree_sha256 = inventory()
    notebooks, problems = notebook_inventory()
    if problems:
        raise RuntimeError("; ".join(problems))
    return {
        "schema_version": "nino26-fase3-zero-restoration-v1",
        "label": "Fase3-zero",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "quarantine": "data/quarantine/output_reset_20260713T220000BRT",
            "pre_review_commit": "61ae278",
            "restoration_policy": "byte_preserving_copy_no_scientific_reinterpretation",
        },
        "counts": {
            "notebooks": len(notebooks),
            "figures": len(list((CAPSULE / "outputs/figures").rglob("*.png"))),
            "numeric_table_files": len(
                [path for path in (CAPSULE / "outputs/numeric-tables").rglob("*") if path.is_file()]
            ),
            "statistics_files": len(
                [path for path in (CAPSULE / "outputs/statistics").glob("*") if path.is_file()]
            ),
            "all_files_excluding_manifest": len(files),
            "total_bytes_excluding_manifest": sum(int(row["size_bytes"]) for row in files),
        },
        "tree_sha256": tree_sha256,
        "notebooks": notebooks,
        "files": files,
    }


def validate() -> list[str]:
    problems: list[str] = []
    if not MANIFEST.exists():
        return ["missing_restoration_manifest"]
    try:
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid_restoration_manifest:{exc}"]
    files, tree_sha256 = inventory()
    expected_files = {str(row["path"]): row for row in expected.get("files", [])}
    observed_files = {str(row["path"]): row for row in files}
    for missing in sorted(set(expected_files) - set(observed_files)):
        problems.append(f"missing_file:{missing}")
    for unexpected in sorted(set(observed_files) - set(expected_files)):
        problems.append(f"unexpected_file:{unexpected}")
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
        print(f"[gravado] {MANIFEST.relative_to(ROOT)}")
    problems = validate()
    if problems:
        for problem in problems:
            print(f"[erro] {problem}")
        return 1
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    counts = payload["counts"]
    print(
        "[ok] Fase3-zero integra: "
        f"notebooks={counts['notebooks']}, figuras={counts['figures']}, "
        f"numeric_files={counts['numeric_table_files']}, "
        f"statistics_files={counts['statistics_files']}, "
        f"tree_sha256={payload['tree_sha256'][:16]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

