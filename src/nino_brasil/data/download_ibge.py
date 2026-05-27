from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


IBGE_MUNICIPAL_2024_BASE_URL = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/"
    "malhas_territoriais/malhas_municipais/municipio_2024/Brasil"
)


@dataclass(frozen=True)
class IbgeProduct:
    product_id: str
    filename: str
    description: str


IBGE_PRODUCTS: dict[str, IbgeProduct] = {
    "uf": IbgeProduct(
        product_id="uf",
        filename="BR_UF_2024.zip",
        description="Unidades da Federacao do Brasil, malha municipal 2024.",
    ),
    "municipios": IbgeProduct(
        product_id="municipios",
        filename="BR_Municipios_2024.zip",
        description="Municipios do Brasil, malha municipal 2024.",
    ),
}


def _download_url(url: str, output_path: Path, overwrite: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        print(f"exists: {output_path}")
        return

    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    with urllib.request.urlopen(url, timeout=60) as response, temp_path.open("wb") as fh:
        shutil.copyfileobj(response, fh)

    temp_path.replace(output_path)
    print(f"downloaded: {output_path}")


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
        "dataset_id": f"ibge_{product.product_id}_2024",
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
) -> Path:
    if product_id not in IBGE_PRODUCTS:
        valid = ", ".join(sorted(IBGE_PRODUCTS))
        raise ValueError(f"Produto IBGE invalido: {product_id}. Use: {valid}.")

    product = IBGE_PRODUCTS[product_id]
    url = f"{IBGE_MUNICIPAL_2024_BASE_URL}/{product.filename}"
    raw_path = raw_dir / product.filename

    _download_url(url, raw_path, overwrite=overwrite)

    extracted_path: Path | None = None
    if extract:
        if interim_dir is None:
            raise ValueError("interim_dir e obrigatorio quando extract=True.")
        extracted_path = interim_dir / raw_path.stem
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
