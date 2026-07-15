#!/usr/bin/env python3
"""Quarentena reversível das saídas derivadas e notebooks legados NINO26.

O modo padrão é apenas inventário. ``--apply`` move os itens para uma pasta
datada em ``data/quarantine``; nada é apagado. Dados brutos e os alvos CHIRPS
nativos são protegidos por uma verificação fail-closed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.notebook_catalog import CANONICAL_NOTEBOOKS


PROTECTED = (
    ROOT / "data/raw",
    ROOT / "data/processed/zarr/features/chirps_native_daily_brazil_box.zarr",
    ROOT / "data/processed/zarr/features/chirps_native_weekly_targets.zarr",
    ROOT / "data/processed/parquet/features/phase4_chirps_native_pixels.csv",
)

WHOLE_PATHS = (
    "data/processed/figures",
    "data/processed/numeric-tables",
    "data/processed/runs",
    "data/processed/parquet/modeling",
    "data/processed/zarr/statistics",
    "data/processed/zarr/modeling",
    "data/audit/augmentation_ablation",
    "data/interim/notebook-runtime",
    "notebooks/fase3",
    "notebooks/fase4",
    "executed",
)

FILES = (
    "data/processed/figuras_manifesto.csv",
    "data/processed/figuras_manifesto_F3Nino.csv",
    "data/processed/figuras_manifesto_F3Nina.csv",
    "data/processed/parquet/features/nino34_master_weekly.csv",
    "data/processed/parquet/features/nino34_master_weekly_source_adjusted_v1.csv",
    "data/processed/parquet/features/phase3_indices_semanais.csv",
    "data/processed/parquet/features/phase3_indices_semanais.csv.manifest.json",
    "data/processed/parquet/features/phase4_chirps_pixels.csv",
    "data/processed/parquet/features/phase4_chirps_weekly_zanom.parquet",
    "notebooks/fase2/2Z_sanidade_variaveis.ipynb",
    "notebooks/fase5/5_ciclo_ml.ipynb",
    "notebooks/fase6/6_brasil_ml.ipynb",
    "notebooks/fase7/7_ciclo_convlstm.ipynb",
    "notebooks/fase8/8_brasil_convlstm.ipynb",
)

PRESERVE_IN_STATISTICS = {
    ".gitkeep",
    "phase4_chirps_native_brazil_membership.parquet",
    "phase4_chirps_target_variable_contract.csv",
}


def _inside_workspace(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    return resolved == ROOT.resolve() or ROOT.resolve() in resolved.parents


def _assert_safe(source: Path, destination: Path) -> None:
    if not _inside_workspace(source) or not _inside_workspace(destination):
        raise ValueError(f"caminho fora do workspace: {source} -> {destination}")
    resolved = source.resolve(strict=False)
    for protected in PROTECTED:
        protected_resolved = protected.resolve(strict=False)
        if (
            resolved == protected_resolved
            or resolved in protected_resolved.parents
            or protected_resolved in resolved.parents
        ):
            raise ValueError(f"movimento recusado: {source} protege {protected}")


def _inventory(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file():
        return 1, path.stat().st_size
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def _move(
    source: Path,
    quarantine: Path,
    *,
    apply: bool,
    rows: list[dict[str, object]],
) -> None:
    if not source.exists():
        return
    relative = source.relative_to(ROOT)
    destination = quarantine / relative
    _assert_safe(source, destination)
    count, size = _inventory(source)
    rows.append(
        {
            "source": relative.as_posix(),
            "destination": destination.relative_to(ROOT).as_posix(),
            "files": count,
            "bytes": size,
        }
    )
    if not apply:
        return
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def _clean_canonical_notebooks(*, apply: bool) -> list[str]:
    changed: list[str] = []
    for spec in CANONICAL_NOTEBOOKS:
        path = ROOT / spec.relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        notebook = nbformat.read(path, as_version=4)
        dirty = False
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            if cell.get("execution_count") is not None or cell.get("outputs", []):
                cell.execution_count = None
                cell.outputs = []
                dirty = True
        if dirty:
            changed.append(spec.relative_path)
            if apply:
                nbformat.write(notebook, path)
    return changed


def reset(*, apply: bool, stamp: str) -> dict[str, object]:
    missing_protected = [str(path.relative_to(ROOT)) for path in PROTECTED if not path.exists()]
    # data/raw pode estar logicamente vazio em uma instalação de teste, mas
    # os dois targets e o inventário nativo devem existir nesta limpeza oficial.
    required_protected = PROTECTED[1:]
    if missing := [path for path in required_protected if not path.exists()]:
        raise FileNotFoundError(f"proteções CHIRPS ausentes: {missing}")

    quarantine = ROOT / "data/quarantine" / f"output_reset_{stamp}"
    if apply and quarantine.exists():
        raise FileExistsError(quarantine)
    rows: list[dict[str, object]] = []

    for relative in WHOLE_PATHS:
        _move(ROOT / relative, quarantine, apply=apply, rows=rows)
    for relative in FILES:
        _move(ROOT / relative, quarantine, apply=apply, rows=rows)

    statistics = ROOT / "data/processed/parquet/statistics"
    if statistics.is_dir():
        for child in sorted(statistics.iterdir()):
            if child.name in PRESERVE_IN_STATISTICS:
                continue
            _move(child, quarantine, apply=apply, rows=rows)

    # Executados dos novos escopos não devem contaminar os fontes limpos.
    for directory in sorted((ROOT / "notebooks").glob("fase*/executed")):
        _move(directory, quarantine, apply=apply, rows=rows)

    cleaned_notebooks = _clean_canonical_notebooks(apply=apply)
    if apply:
        for relative in (
            "data/processed/figures",
            "data/processed/numeric-tables",
            "data/processed/runs",
            "data/processed/parquet/modeling",
            "data/processed/zarr/statistics",
            "data/processed/zarr/modeling",
        ):
            directory = ROOT / relative
            directory.mkdir(parents=True, exist_ok=True)
            (directory / ".gitkeep").touch(exist_ok=True)
        for protected in required_protected:
            if not protected.exists():
                raise RuntimeError(f"proteção desapareceu durante a limpeza: {protected}")

    payload: dict[str, object] = {
        "schema_version": 1,
        "mode": "apply" if apply else "dry-run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quarantine": quarantine.relative_to(ROOT).as_posix(),
        "moved_or_planned": rows,
        "totals": {
            "paths": len(rows),
            "files": sum(int(row["files"]) for row in rows),
            "bytes": sum(int(row["bytes"]) for row in rows),
        },
        "canonical_notebooks_with_embedded_outputs": cleaned_notebooks,
        "protected_paths": [path.relative_to(ROOT).as_posix() for path in PROTECTED],
        "missing_noncritical_protected_paths": missing_protected,
    }
    receipt_dir = ROOT / "data/audit/output_resets"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = receipt_dir / f"output_reset_{stamp}_{payload['mode']}.json"
    receipt.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--stamp",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args(argv)
    payload = reset(apply=args.apply, stamp=args.stamp)
    total = payload["totals"]
    print(
        f"[{payload['mode']}] caminhos={total['paths']} arquivos={total['files']} "
        f"GB={total['bytes'] / 1024**3:.3f}"
    )
    print(f"quarentena: {payload['quarantine']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
