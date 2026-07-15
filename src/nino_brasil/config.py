from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "project.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load a YAML project configuration file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def confirmatory_fdr_alpha(
    config: dict[str, Any] | None = None,
    *,
    fallback: float = 0.05,
) -> float:
    """Return the single project-wide confirmatory BH-FDR threshold.

    ``configs/project.yaml`` is authoritative.  The explicit 0.05 fallback is
    reserved for a missing legacy key; malformed/out-of-range values fail
    loudly so an official analysis cannot silently change its error rate.
    """

    if not 0.0 < float(fallback) < 1.0:
        raise ValueError("fallback FDR alpha must lie in (0, 1)")
    resolved = load_config() if config is None else config
    try:
        value = resolved["modeling"]["phase_3_diagnostics"]["fdr_alpha"]
    except (KeyError, TypeError):
        warnings.warn(
            "modeling.phase_3_diagnostics.fdr_alpha is missing; "
            f"using the explicit compatibility fallback {fallback:g}",
            RuntimeWarning,
            stacklevel=2,
        )
        value = fallback
    alpha = float(value)
    if not 0.0 < alpha < 1.0:
        raise ValueError("confirmatory FDR alpha must lie in (0, 1)")
    return alpha


def project_path(*parts: str) -> Path:
    """Return an absolute path inside the project root."""
    return ROOT.joinpath(*parts)
