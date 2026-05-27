from __future__ import annotations

from pathlib import Path

import geopandas as gpd


def save_choropleth(
    gdf: gpd.GeoDataFrame,
    column: str,
    output_path: str | Path,
    title: str,
) -> Path:
    """Save a basic choropleth map."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ax = gdf.plot(column=column, legend=True, figsize=(8, 8), edgecolor="0.4", linewidth=0.3)
    ax.set_axis_off()
    ax.set_title(title)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    return output
