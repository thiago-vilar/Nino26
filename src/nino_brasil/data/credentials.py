from __future__ import annotations

import os
from pathlib import Path

from nino_brasil.config import project_path


DEFAULT_CDS_API_URL = "https://cds.climate.copernicus.eu/api"


def load_local_env(path: Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from a local .env file without overwriting env vars."""
    env_path = path or project_path(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def mask_secret(value: str | None) -> str:
    if not value:
        return "missing"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def cds_credentials_status() -> dict[str, str]:
    load_local_env()
    key = os.environ.get("CDS_API_KEY")
    url = os.environ.get("CDS_API_URL", DEFAULT_CDS_API_URL)
    return {
        "CDS_API_KEY": mask_secret(key),
        "CDS_API_URL": url,
        "ready": "yes" if key else "no",
    }
