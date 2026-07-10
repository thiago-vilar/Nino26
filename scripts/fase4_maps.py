"""Prepare and audit official IBGE boundaries used by Phase 4C.

Downloads are never implicit during analysis.  Run this script once with
``--download``; ZIP files and SHA-256 metadata are kept under ``data/raw/ibge``
and versioned extractions under ``data/interim/ibge``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen
import zipfile

import geopandas as gpd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))

from nino_brasil.maps.spatial_support import (  # noqa: E402
    IBGE_BIOMES_2025_URL,
    IBGE_REGIONS_2024_URL,
    build_analysis_units,
    load_ibge_biomes,
    load_ibge_regions,
)


RAW_DIR = ROOT / "data" / "raw" / "ibge" / "fase4_limites"
INTERIM_DIR = ROOT / "data" / "interim" / "ibge"
REGIONS_ZIP = RAW_DIR / "BR_Regioes_2024.zip"
BIOMES_ZIP = RAW_DIR / (
    "2025_Biomas-e-Sistema-Costeiro-Marinho-do-Brasil-1-250000_shp.zip"
)
REGIONS_DIR = INTERIM_DIR / "BR_Regioes_2024"
BIOMES_DIR = INTERIM_DIR / "Biomas_2025"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    # Metadata is written atomically so an interrupted download cannot produce
    # a valid-looking provenance record.
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False, dir=path.parent
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def download_official_zip(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and zipfile.is_zipfile(destination):
        return destination
    request = Request(url, headers={"User-Agent": "NINO26/Phase4C boundary audit"})
    with tempfile.NamedTemporaryFile(delete=False, suffix=".part", dir=destination.parent) as tmp:
        temporary = Path(tmp.name)
        with urlopen(request, timeout=120) as response:  # noqa: S310 - fixed official URLs
            shutil.copyfileobj(response, tmp)
    if not zipfile.is_zipfile(temporary):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file is not a ZIP archive: {url}")
    os.replace(temporary, destination)
    metadata = {
        "url": url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "filename": destination.name,
        "size_bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }
    _write_metadata(destination.with_suffix(".metadata.json"), metadata)
    return destination


def safe_extract_zip(archive: Path, destination: Path) -> Path:
    """Extract without allowing ZIP members to escape the intended directory."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and any(destination.rglob("*.shp")):
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")
        package.extractall(destination)
    return destination


def prepare_official_boundaries() -> tuple[Path, Path]:
    regions_archive = download_official_zip(IBGE_REGIONS_2024_URL, REGIONS_ZIP)
    biomes_archive = download_official_zip(IBGE_BIOMES_2025_URL, BIOMES_ZIP)
    safe_extract_zip(regions_archive, REGIONS_DIR)
    safe_extract_zip(biomes_archive, BIOMES_DIR)
    return find_regions_shapefile(), find_biomes_shapefile()


def _single_shapefile(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {pattern!r} below {directory}; found {len(matches)}. "
            "Run scripts/fase4_maps.py --download."
        )
    return matches[0]


def find_regions_shapefile() -> Path:
    return _single_shapefile(REGIONS_DIR, "BR_Regioes_2024.shp")


def find_biomes_shapefile() -> Path:
    return _single_shapefile(BIOMES_DIR, "*.shp")


def load_analysis_units() -> gpd.GeoDataFrame:
    return build_analysis_units(
        load_ibge_regions(find_regions_shapefile()),
        load_ibge_biomes(find_biomes_shapefile()),
    )


def add_brazil_boundaries(
    ax,
    *,
    regions: bool = True,
    biomes: bool = False,
    linewidth: float = 0.8,
):
    """Overlay unsimplified official boundaries on a longitude/latitude axis."""

    units = load_analysis_units()
    if regions:
        units[units["tipo_unidade"].eq("regiao")].boundary.plot(
            ax=ax, color="#111827", linewidth=linewidth, zorder=5
        )
    if biomes:
        units[units["tipo_unidade"].eq("bioma")].boundary.plot(
            ax=ax, color="#4b5563", linewidth=linewidth, linestyle="--", zorder=5
        )
    ax.set_xlim(-74.2, -28.5)
    ax.set_ylim(-34.2, 6.2)
    return ax


def audit_boundaries() -> None:
    units = load_analysis_units()
    table = units.drop(columns="geometry").copy()
    table["geometry_valid"] = units.geometry.is_valid.to_numpy()
    table["geometry_empty"] = units.geometry.is_empty.to_numpy()
    print(table.to_string(index=False))
    print(f"CRS: {units.crs}; unidades: {len(units)}")
    print(f"regioes: {(units.tipo_unidade == 'regiao').sum()}")
    print(f"biomas: {(units.tipo_unidade == 'bioma').sum()}")
    print(f"recortes especiais: {(units.tipo_unidade == 'recorte_bioma_regiao').sum()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download official ZIPs to raw and extract versioned vectors to interim.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Validate and print every Phase 4C spatial unit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download:
        prepare_official_boundaries()
    if args.audit or not args.download:
        audit_boundaries()


if __name__ == "__main__":
    main()
