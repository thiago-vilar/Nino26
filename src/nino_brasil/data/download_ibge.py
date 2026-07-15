from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from nino_brasil.data.download_http import download_url


IBGE_MUNICIPAL_2024_BASE_URL = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/"
    "malhas_territoriais/malhas_municipais/municipio_2024/Brasil"
)


@dataclass(frozen=True)
class IbgeProduct:
    product_id: str
    filename: str
    description: str
    source_url: str
    reference_year: int
    extraction_dir: str


IBGE_PRODUCTS: dict[str, IbgeProduct] = {
    "uf": IbgeProduct(
        product_id="uf",
        filename="BR_UF_2024.zip",
        description="Unidades da Federacao do Brasil, malha municipal 2024.",
        source_url=f"{IBGE_MUNICIPAL_2024_BASE_URL}/BR_UF_2024.zip",
        reference_year=2024,
        extraction_dir="BR_UF_2024",
    ),
    "municipios": IbgeProduct(
        product_id="municipios",
        filename="BR_Municipios_2024.zip",
        description="Municipios do Brasil, malha municipal 2024.",
        source_url=f"{IBGE_MUNICIPAL_2024_BASE_URL}/BR_Municipios_2024.zip",
        reference_year=2024,
        extraction_dir="BR_Municipios_2024",
    ),
    "regioes": IbgeProduct(
        product_id="regioes",
        filename="BR_Regioes_2024.zip",
        description="Grandes Regioes do Brasil, malha territorial 2024.",
        source_url=f"{IBGE_MUNICIPAL_2024_BASE_URL}/BR_Regioes_2024.zip",
        reference_year=2024,
        extraction_dir="BR_Regioes_2024",
    ),
    "biomas": IbgeProduct(
        product_id="biomas",
        filename="2025_Biomas-e-Sistema-Costeiro-Marinho-do-Brasil-1-250000_shp.zip",
        description="Biomas do Brasil, escala 1:250.000, revisao 2025.",
        source_url=(
            "https://geoftp.ibge.gov.br/informacoes_ambientais/estudos_ambientais/"
            "biomas/vetores/2025_Biomas-e-Sistema-Costeiro-Marinho-do-Brasil-"
            "1-250000_shp.zip"
        ),
        reference_year=2025,
        extraction_dir="Biomas_2025",
    ),
}


def _write_metadata(
    metadata_path: Path,
    *,
    product: IbgeProduct,
    url: str,
    raw_path: Path,
    extracted_path: Path | None,
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset_id": f"ibge_{product.product_id}_{product.reference_year}",
        "name": product.description,
        "institution": "Instituto Brasileiro de Geografia e Estatistica",
        "source_url": url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(raw_path.as_posix()),
        "extracted_path": str(extracted_path.as_posix()) if extracted_path else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def download_ibge(
    *,
    product_id: str,
    raw_dir: Path,
    interim_dir: Path | None = None,
    extract: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Path:
    if product_id not in IBGE_PRODUCTS:
        valid = ", ".join(sorted(IBGE_PRODUCTS))
        raise ValueError(f"Produto IBGE invalido: {product_id}. Use: {valid}.")

    product = IBGE_PRODUCTS[product_id]
    url = product.source_url
    raw_path = raw_dir / product.filename

    if dry_run:
        print(f"DRY RUN ibge {product_id}: {url} -> {raw_path}")
        return raw_path

    download_url(url, raw_path, overwrite=overwrite)

    extracted_path: Path | None = None
    if extract:
        if interim_dir is None:
            raise ValueError("interim_dir e obrigatorio quando extract=True.")
        extracted_path = interim_dir / product.extraction_dir
        extracted_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(raw_path) as zf:
            zf.extractall(extracted_path)
        print(f"extracted: {extracted_path}")

    _write_metadata(
        raw_dir / f"{raw_path.stem}.metadata.json",
        product=product,
        url=url,
        raw_path=raw_path,
        extracted_path=extracted_path,
    )

    return raw_path
