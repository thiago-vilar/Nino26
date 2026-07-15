#!/usr/bin/env python3
"""Audita as malhas oficiais IBGE consumidas pelas fases espaciais do NINO26."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed/parquet/statistics/phase1_ibge_boundaries_audit.csv"

PRODUCTS = {
    "estados_ufs": (
        ROOT / "data/interim/ibge/BR_UF_2024/BR_UF_2024.shp",
        27,
        {"CD_UF", "NM_UF", "SIGLA_UF", "CD_REGIA", "NM_REGIA"},
        "IBGE malha territorial 2024",
    ),
    "regioes": (
        ROOT / "data/interim/ibge/BR_Regioes_2024/BR_Regioes_2024.shp",
        5,
        {"CD_REGIA", "NM_REGIA", "SIGLA_RG"},
        "IBGE malha territorial 2024",
    ),
    "biomas": (
        ROOT / "data/interim/ibge/Biomas_2025/lml_bioma_e250k_v20250911_A.shp",
        6,
        {"CD_BIOMA", "NM_BIOMA"},
        "IBGE biomas 1:250.000 revisao 2025",
    ),
}


def bundle_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for component in sorted(path.parent.glob(path.stem + ".*")):
        if not component.is_file():
            continue
        digest.update(component.name.encode("utf-8"))
        digest.update(component.read_bytes())
    return digest.hexdigest()


def audit() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for product, (path, expected_count, required_columns, version) in PRODUCTS.items():
        exists = path.is_file()
        row: dict[str, object] = {
            "produto": product,
            "versao": version,
            "caminho": str(path.relative_to(ROOT)),
            "existe": exists,
            "esperado_geometrias": expected_count,
            "geometrias": 0,
            "crs": "",
            "geometrias_invalidas": 0,
            "geometrias_vazias": 0,
            "colunas_obrigatorias_ok": False,
            "bundle_sha256": "",
            "aprovado": False,
            "problemas": "arquivo ausente" if not exists else "",
        }
        if exists:
            frame = gpd.read_file(path)
            missing = sorted(required_columns.difference(frame.columns))
            invalid = int((~frame.geometry.is_valid).sum())
            empty = int(frame.geometry.is_empty.sum())
            crs = frame.crs.to_epsg() if frame.crs else None
            problems = []
            if len(frame) != expected_count:
                problems.append(f"contagem={len(frame)} esperado={expected_count}")
            if crs != 4674:
                problems.append(f"crs={frame.crs} esperado=EPSG:4674")
            if invalid:
                problems.append(f"invalidas={invalid}")
            if empty:
                problems.append(f"vazias={empty}")
            if missing:
                problems.append("colunas_ausentes=" + ",".join(missing))
            row.update(
                geometrias=len(frame),
                crs=str(frame.crs or ""),
                geometrias_invalidas=invalid,
                geometrias_vazias=empty,
                colunas_obrigatorias_ok=not missing,
                bundle_sha256=bundle_sha256(path),
                aprovado=not problems,
                problemas=";".join(problems),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    result = audit()
    if not args.validate_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"[gravado] {args.output.relative_to(ROOT)}")
    print(result.to_string(index=False))
    return 0 if bool(result["aprovado"].all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
