from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd


def _label_columns(gdf: gpd.GeoDataFrame, column: str) -> list[str]:
    candidates = [
        "NM_UF",
        "nome_uf",
        "uf",
        "NM_MUN",
        "nome_municipio",
        "municipio",
        "name",
    ]
    labels = [candidate for candidate in candidates if candidate in gdf.columns]
    return [*labels, column]


def export_choropleth_table(
    gdf: gpd.GeoDataFrame,
    column: str,
    output_path: str | Path,
    *,
    ascending_rank: bool = False,
    round_decimals: int = 4,
) -> Path:
    """Export choropleth source values as CSV without geometry."""
    if column not in gdf.columns:
        raise KeyError(f"GeoDataFrame must contain {column!r}.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame = gdf[_label_columns(gdf, column)].copy()
    frame = frame.drop(columns=["geometry"], errors="ignore")
    frame[column] = frame[column].round(round_decimals)
    frame["rank"] = frame[column].rank(ascending=ascending_rank, method="dense").astype("Int64")
    frame.to_csv(output, index=False)
    return output


def save_choropleth(
    gdf: gpd.GeoDataFrame,
    column: str,
    output_path: str | Path,
    title: str,
    *,
    table_output_path: str | Path | None = None,
    export_table: bool = True,
) -> Path:
    """Save a basic choropleth map."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if export_table:
        export_choropleth_table(gdf, column, table_output_path or output.with_suffix(".csv"))
    ax = gdf.plot(column=column, legend=True, figsize=(8, 8), edgecolor="0.4", linewidth=0.3)
    ax.set_axis_off()
    ax.set_title(title)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    return output
