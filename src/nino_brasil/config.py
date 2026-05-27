from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "project.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load a YAML project configuration file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_path(*parts: str) -> Path:
    """Return an absolute path inside the project root."""
    return ROOT.joinpath(*parts)
