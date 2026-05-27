from __future__ import annotations

from pathlib import Path

import requests
from tqdm import tqdm


def download_url(
    url: str,
    output_path: Path,
    *,
    overwrite: bool = False,
    resume: bool = True,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        print(f"exists: {output_path}")
        return output_path

    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    if temp_path.exists() and (overwrite or not resume):
        temp_path.unlink()

    existing_size = temp_path.stat().st_size if temp_path.exists() else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size and resume else None

    with requests.get(url, stream=True, timeout=(30, 120), headers=headers) as response:
        response.raise_for_status()

        mode = "ab" if existing_size and response.status_code == 206 else "wb"
        if mode == "wb":
            existing_size = 0

        content_length = int(response.headers.get("Content-Length", 0))
        total = existing_size + content_length if content_length else None

        with temp_path.open(mode) as fh:
            with tqdm(
                total=total,
                initial=existing_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=output_path.name,
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    bar.update(len(chunk))

    temp_path.replace(output_path)
    print(f"downloaded: {output_path}")
    return output_path
