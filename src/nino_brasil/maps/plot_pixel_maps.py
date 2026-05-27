from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr


def save_pixel_map(
    da: xr.DataArray,
    output_path: str | Path,
    title: str,
    cmap: str = "RdBu",
) -> Path:
    """Save a simple pixel map from a lat/lon DataArray."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    da.plot(ax=ax, cmap=cmap)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
