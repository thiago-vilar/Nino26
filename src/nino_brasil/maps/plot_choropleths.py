from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt

from nino_brasil.maps.plot_pixel_maps import (
    FIGURE_DPI,
    NO_DATA_COLOR,
    add_interpretive_caption,
    yellow_neutral_diverging_cmap,
)


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
    interpretation: str | None = None,
    metadata: str | None = None,
    cmap=None,
) -> Path:
    """Save a spacious, high-resolution choropleth with a numeric companion."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if export_table:
        export_choropleth_table(gdf, column, table_output_path or output.with_suffix(".csv"))
    chosen_cmap = cmap or yellow_neutral_diverging_cmap()
    ax = gdf.plot(
        column=column,
        legend=True,
        figsize=(13.5, 11.5),
        edgecolor="#374151",
        linewidth=0.7,
        cmap=chosen_cmap,
        missing_kwds={"color": NO_DATA_COLOR, "label": "Sem dado"},
    )
    ax.set_axis_off()
    ax.set_title(title, fontsize=18, pad=14, fontweight="bold")
    fig = ax.get_figure()
    if interpretation:
        add_interpretive_caption(
            fig,
            metadata=metadata or f"Variável: {column}",
            interpretation=interpretation,
        )
    fig.tight_layout(rect=(0, 0.08 if interpretation else 0, 1, 1))
    fig.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output
