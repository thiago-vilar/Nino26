from __future__ import annotations

import shutil
from pathlib import Path

import requests


def download_url(url: str, output_path: Path, *, overwrite: bool = False) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        print(f"exists: {output_path}")
        return output_path

    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temp_path.open("wb") as fh:
            shutil.copyfileobj(response.raw, fh)

    temp_path.replace(output_path)
    print(f"downloaded: {output_path}")
    return output_path
