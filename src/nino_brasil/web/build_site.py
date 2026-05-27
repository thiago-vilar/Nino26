from __future__ import annotations

from pathlib import Path


def ensure_site_dirs(site_dir: str | Path = "docs") -> None:
    """Create expected static-site directories."""
    root = Path(site_dir)
    (root / "assets" / "maps").mkdir(parents=True, exist_ok=True)
    (root / "assets" / "data").mkdir(parents=True, exist_ok=True)
