#!/usr/bin/env python3
"""Inventory or quarantine explicitly known stale derived outputs.

The default is a read-only dry-run.  ``--apply`` moves only figures and their
numeric tables to a timestamped quarantine; raw, interim, Zarr, parquet and run
artifacts are deliberately outside this script's authority.  Nothing is ever
deleted.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
QUARANTINE = ROOT / "data/quarantine/stale_derived"
FIGURE_REGISTRY = ROOT / "data/processed/figuras_manifesto.csv"
ALLOWED_SOURCE_ROOTS = (
    ROOT / "data/processed/figures",
    ROOT / "data/processed/numeric-tables",
)


@dataclass(frozen=True)
class Candidate:
    relative: str
    category: str
    reason: str
    apply_allowed: bool = True


KNOWN_STALE = (
    Candidate("data/processed/figures/fase2/phase2_sanidade_atmosfera_bjerknes.png", "legacy_figure", "legacy F2 figure without semantic run lineage"),
    Candidate("data/processed/figures/fase2/phase2_sanidade_oceano_recarga.png", "legacy_figure", "legacy F2 figure without semantic run lineage"),
    Candidate("data/processed/figures/fase2/phase2_sanidade_oceano_superficie.png", "legacy_figure", "legacy F2 figure without semantic run lineage"),
    Candidate("data/processed/figures/fase2/phase2_sanidade_oceano_temp_perfil.png", "legacy_figure", "legacy F2 figure without semantic run lineage"),
    Candidate("data/processed/figures/fase2/phase2_sanidade_painel_z.png", "legacy_figure", "legacy F2 figure without semantic run lineage"),
    Candidate("data/processed/figures/fase4/Fig_4001.png", "historical_figure", "historical F4.0 output, excluded from the canonical F4 runner"),
    Candidate("data/processed/figures/fase4/Fig_4A01.png", "historical_figure", "historical F4A output, excluded from the canonical native-CHIRPS F4 runner"),
    Candidate("data/processed/figures/fase4/Fig_4A02.png", "historical_figure", "historical F4A output, excluded from the canonical native-CHIRPS F4 runner"),
    Candidate("data/processed/figures/fase4/Fig_4A03.png", "historical_figure", "historical F4A output, excluded from the canonical native-CHIRPS F4 runner"),
    Candidate("data/processed/figures/fase4/Fig_4B01.png", "invalid_target_pilot", "pilot generated from the superseded CHIRPS target"),
    Candidate("data/processed/figures/fase4/Fig_4C01.png", "invalid_target_pilot", "pilot generated from the superseded CHIRPS target"),
    Candidate("data/processed/figures/fase4/Fig_4C02.png", "invalid_target_pilot", "pilot generated from the superseded CHIRPS target"),
    Candidate("data/processed/figures/fase4/Fig_4C03.png", "invalid_target_pilot", "pilot generated from the superseded CHIRPS target"),
    Candidate("data/processed/numeric-tables/fase4/Fig_4001", "historical_numeric_table", "numeric extraction for historical F4.0 output"),
    Candidate("data/processed/numeric-tables/fase4/Fig_4A01", "historical_numeric_table", "numeric extraction for historical F4A output"),
    Candidate("data/processed/numeric-tables/fase4/Fig_4A02", "historical_numeric_table", "numeric extraction for historical F4A output"),
    Candidate("data/processed/numeric-tables/fase4/Fig_4A03", "historical_numeric_table", "numeric extraction for historical F4A output"),
    Candidate("data/processed/numeric-tables/fase4/Fig_4B01", "invalid_target_pilot", "numeric extraction generated from the superseded CHIRPS target"),
    Candidate("data/processed/numeric-tables/fase4/Fig_4C03", "invalid_target_pilot", "numeric extraction generated from the superseded CHIRPS target"),
    Candidate("data/processed/figures/fase5/Fig_5A01.png", "unversioned_pilot", "unversioned F5 pilot; run-scoped smoke artifacts supersede it"),
    Candidate("data/processed/figures/fase5/Fig_5A02.png", "unversioned_pilot", "unversioned F5 pilot; run-scoped smoke artifacts supersede it"),
    Candidate("data/processed/numeric-tables/fase5/Fig_5A01", "unversioned_pilot", "unversioned F5 pilot without a complete run manifest"),
    Candidate("data/processed/numeric-tables/fase5/Fig_5A02", "unversioned_pilot", "unversioned F5 pilot without a complete run manifest"),
    Candidate("data/processed/figures/fase6/Fig_6A01.png", "invalid_target_pilot", "unversioned F6 pilot generated before the native CHIRPS target"),
    Candidate("data/processed/numeric-tables/fase6/Fig_6A01", "invalid_target_pilot", "unversioned F6 pilot generated before the native CHIRPS target"),
)


def _inside(path: Path, parents: Iterable[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == parent.resolve() or parent.resolve() in resolved.parents for parent in parents)


def _legacy_f3_duplicates() -> list[Candidate]:
    """Find only unsuffixed F3 predecessors that have a descriptive sibling."""
    figures = ROOT / "data/processed/figures/fase3"
    tables = ROOT / "data/processed/numeric-tables/fase3"
    found: list[Candidate] = []
    if not figures.exists():
        return found
    for old in sorted(figures.glob("Fig_3[A-Z][0-9][0-9].png")):
        replacements = sorted(figures.glob(old.stem + "_*.png"))
        if not replacements:
            continue
        reason = (
            "unsuffixed F3 predecessor duplicated by: "
            + ", ".join(item.name for item in replacements)
            + "; inventory only until the replacement has semantic_source lineage"
        )
        found.append(Candidate(old.relative_to(ROOT).as_posix(), "legacy_duplicate", reason, False))
        old_table = tables / old.stem
        replacement_tables = [tables / item.stem for item in replacements]
        if old_table.exists() and any(item.exists() for item in replacement_tables):
            found.append(Candidate(old_table.relative_to(ROOT).as_posix(), "legacy_duplicate", reason, False))
    return found


def _state(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError(f"Refusing symbolic link/reparse candidate: {path}")
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    total = 0
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                total += len(chunk)
                digest.update(chunk)
    return {
        "kind": "file" if path.is_file() else "directory",
        "file_count": len(files),
        "size_bytes": total,
        "content_sha256": digest.hexdigest(),
        "last_modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
    }


def _write_json_atomic(path: Path, payload: object) -> None:
    if not _inside(path, (ROOT,)):
        raise ValueError(f"Refusing report outside workspace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv_atomic(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    if not _inside(path, (ROOT,)):
        raise ValueError(f"Refusing CSV outside workspace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _reconcile_quarantined_registry(destination_root: Path) -> dict[str, object]:
    """Remove only registry rows whose known-stale PNG is already quarantined."""

    if not FIGURE_REGISTRY.is_file():
        return {"status": "registry_missing", "removed_codes": []}
    stale_figures = {
        Path(candidate.relative).stem: candidate.relative
        for candidate in KNOWN_STALE
        if candidate.relative.startswith("data/processed/figures/")
        and candidate.relative.endswith(".png")
    }
    eligible_codes: set[str] = set()
    for code, relative in stale_figures.items():
        active = ROOT / relative
        quarantined = list(QUARANTINE.glob(f"*/{relative}"))
        if not active.exists() and any(path.is_file() for path in quarantined):
            eligible_codes.add(code)
    with FIGURE_REGISTRY.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "codigo" not in fieldnames:
        raise ValueError(f"Figure registry lacks codigo column: {FIGURE_REGISTRY}")
    removed = [row for row in rows if str(row.get("codigo") or "") in eligible_codes]
    if not removed:
        return {"status": "no_quarantined_rows", "removed_codes": []}
    kept = [row for row in rows if str(row.get("codigo") or "") not in eligible_codes]
    destination_root.mkdir(parents=True, exist_ok=True)
    backup = destination_root / "figuras_manifesto_before.csv"
    removed_path = destination_root / "figuras_manifesto_quarantined_rows.csv"
    shutil.copy2(FIGURE_REGISTRY, backup)
    before_sha256 = _sha256_file(backup)
    _write_csv_atomic(removed_path, removed, fieldnames)
    _write_csv_atomic(FIGURE_REGISTRY, kept, fieldnames)
    return {
        "status": "reconciled",
        "removed_codes": sorted(str(row["codigo"]) for row in removed),
        "removed_count": len(removed),
        "kept_count": len(kept),
        "registry_before": backup.relative_to(ROOT).as_posix(),
        "registry_before_sha256": before_sha256,
        "quarantined_rows": removed_path.relative_to(ROOT).as_posix(),
        "quarantined_rows_sha256": _sha256_file(removed_path),
        "registry_after_sha256": _sha256_file(FIGURE_REGISTRY),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform moves; default is read-only dry-run")
    parser.add_argument("--report-json", type=Path, help="optional dry-run inventory/report path inside the workspace")
    args = parser.parse_args(argv)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination_root = QUARANTINE / stamp
    candidates = list(KNOWN_STALE) + _legacy_f3_duplicates()
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.relative in seen:
            continue
        seen.add(candidate.relative)
        source = ROOT / candidate.relative
        if not source.exists():
            continue
        if not _inside(source, ALLOWED_SOURCE_ROOTS):
            raise ValueError(f"Refusing source outside derived figure roots: {source}")
        destination = destination_root / candidate.relative
        if not _inside(destination, (QUARANTINE,)):
            raise ValueError(f"Refusing destination outside quarantine: {destination}")
        if destination.exists():
            raise FileExistsError(f"Refusing existing quarantine destination: {destination}")
        records.append(
            {
                "source": candidate.relative,
                "destination": destination.relative_to(ROOT).as_posix(),
                "category": candidate.category,
                "reason": candidate.reason,
                "operation": "move; no deletion" if args.apply else "dry-run; no mutation",
                "apply_allowed": candidate.apply_allowed,
                "status": (
                    "planned"
                    if args.apply and candidate.apply_allowed
                    else "blocked_unpromoted_replacement"
                    if not candidate.apply_allowed
                    else "observed"
                ),
                **_state(source),
            }
        )

    payload = {
        "schema": "nino26-derived-quarantine-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "apply": args.apply,
        "allowed_source_roots": [item.relative_to(ROOT).as_posix() for item in ALLOWED_SOURCE_ROOTS],
        "records": records,
    }
    if args.report_json:
        report = args.report_json if args.report_json.is_absolute() else ROOT / args.report_json
        _write_json_atomic(report, payload)

    movable = [record for record in records if bool(record["apply_allowed"])]
    if args.apply:
        manifest = destination_root / "manifest.json"
        _write_json_atomic(manifest, payload)
        for index, record in enumerate(records):
            if not bool(record["apply_allowed"]):
                print(f"KEEP [{record['category']}] {record['source']} ({record['status']})")
                continue
            source = ROOT / str(record["source"])
            destination = ROOT / str(record["destination"])
            print(f"MOVE {record['source']} -> {record['destination']}")
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                records[index]["status"] = "moved"
                records[index]["moved_at_utc"] = datetime.now(timezone.utc).isoformat()
            except Exception as exc:
                records[index]["status"] = "failed"
                records[index]["error"] = f"{type(exc).__name__}: {exc}"
                _write_json_atomic(manifest, payload)
                raise
            _write_json_atomic(manifest, payload)
        payload["figure_registry_reconciliation"] = _reconcile_quarantined_registry(
            destination_root
        )
        payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(manifest, payload)
    else:
        for record in records:
            action = "WOULD MOVE" if bool(record["apply_allowed"]) else "KEEP/INVENTORY"
            print(f"{action} [{record['category']}] {record['source']} -> {record['destination']}")

    registry_removed = len(
        (payload.get("figure_registry_reconciliation") or {}).get(
            "removed_codes", []
        )
    )
    print(
        f"items={len(records)} movable={len(movable)} "
        f"registry_removed={registry_removed} apply={args.apply}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
