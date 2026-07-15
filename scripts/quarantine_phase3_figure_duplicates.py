#!/usr/bin/env python3
"""Quarantine superseded slug variants of canonical Phase 3 figure codes."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "data/processed/figures/fase3"
TABLES = ROOT / "data/processed/numeric-tables/fase3"
MANIFEST = ROOT / "data/processed/figuras_manifesto.csv"
PATTERN = re.compile(r"^(Fig_3[0-9A-Z]\d{2})_[a-z0-9_]+$")


def main() -> int:
    if not MANIFEST.is_file():
        return 0
    manifest = pd.read_csv(MANIFEST)
    phase3_codes = set(
        manifest.loc[pd.to_numeric(manifest["fase"], errors="coerce").eq(3), "codigo"].astype(str)
    )
    stale: list[str] = []
    for code in sorted(phase3_codes):
        match = PATTERN.match(code)
        if not match:
            continue
        predecessor = match.group(1)
        predecessor_rows = manifest.loc[manifest["codigo"].astype(str).eq(predecessor)]
        predecessor_run_ids = predecessor_rows.get(
            "run_id", pd.Series(dtype=str)
        ).fillna("").astype(str).str.strip()
        predecessor_semantic = predecessor_rows.get(
            "audit_level", pd.Series(dtype=str)
        ).astype(str).eq("semantic_source")
        local_manifest = TABLES / predecessor / "manifest.csv"
        try:
            local = pd.read_csv(local_manifest)
            local_semantic = bool(
                local["semantic_source"].astype(str).str.lower().isin({"true", "1"}).all()
                and local["audit_level"].astype(str).eq("semantic_source").all()
                and local["run_id"].fillna("").astype(str).str.strip().ne("").all()
            )
        except (FileNotFoundError, KeyError, pd.errors.ParserError):
            local_semantic = False
        if (
            predecessor in phase3_codes
            and (FIGURES / f"{predecessor}.png").is_file()
            and (TABLES / predecessor).is_dir()
            and bool(predecessor_semantic.all())
            and bool(predecessor_run_ids.ne("").all())
            and local_semantic
        ):
            stale.append(code)
    if not stale:
        print("[F3 figuras] nenhum duplicado transitório para quarentena")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "data/quarantine/phase3_figure_lineage" / stamp
    (destination / "figures").mkdir(parents=True, exist_ok=False)
    (destination / "numeric-tables").mkdir(parents=True, exist_ok=False)
    manifest.loc[manifest["codigo"].astype(str).isin(stale)].to_csv(
        destination / "manifest_rows.csv", index=False
    )
    for code in stale:
        png = FIGURES / f"{code}.png"
        table_dir = TABLES / code
        if png.exists():
            shutil.move(str(png), destination / "figures" / png.name)
        if table_dir.exists():
            shutil.move(str(table_dir), destination / "numeric-tables" / table_dir.name)
    manifest = manifest.loc[~manifest["codigo"].astype(str).isin(stale)]
    manifest.sort_values("codigo").to_csv(MANIFEST, index=False)
    print(f"[F3 figuras] {len(stale)} duplicados preservados em {destination.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
