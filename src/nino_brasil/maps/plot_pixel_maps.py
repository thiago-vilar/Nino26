from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr


def _coord_name(da: xr.DataArray, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in da.coords:
            return candidate
    raise KeyError(f"DataArray must contain one of these coordinates: {candidates}")


def export_pixel_table(
    da: xr.DataArray,
    output_path: str | Path,
    *,
    value_name: str = "value",
    round_decimals: int = 4,
) -> Path:
    """Export a lat/lon DataArray to CSV before any map rendering."""
    lat_name = _coord_name(da, ("lat", "latitude", "y"))
    lon_name = _coord_name(da, ("lon", "longitude", "x"))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame = da.to_dataframe(name=value_name).reset_index()
    frame = frame[[lat_name, lon_name, value_name]].dropna()
    frame = frame.rename(columns={lat_name: "lat", lon_name: "lon"})
    frame = frame.round(round_decimals)
    frame.to_csv(output, index=False)
    return output


def save_pixel_map(
    da: xr.DataArray,
    output_path: str | Path,
    title: str,
    cmap: str = "RdBu",
    *,
    table_output_path: str | Path | None = None,
    export_table: bool = True,
    value_name: str = "value",
) -> Path:
    """Save a simple pixel map from a lat/lon DataArray."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if export_table:
        export_pixel_table(da, table_output_path or output.with_suffix(".csv"), value_name=value_name)

    fig, ax = plt.subplots(figsize=(8, 7))
    da.plot(ax=ax, cmap=cmap)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
