"""High-resolution pixel maps for Brazilian signal/response analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import xarray as xr


NO_DATA_COLOR = "#d1d5db"
YELLOW_NEUTRAL = "#ffd84d"
FIGURE_DPI = 360


def _coord_name(da: xr.DataArray, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in da.coords:
            return candidate
    raise KeyError(f"DataArray must contain one of these coordinates: {candidates}")


def yellow_neutral_diverging_cmap() -> LinearSegmentedColormap:
    """Blue (negative) -> yellow (zero) -> red (positive), with no white."""

    cmap = LinearSegmentedColormap.from_list(
        "blue_yellow_red_phase4c",
        ["#21409a", "#2f80c1", "#66c2a5", YELLOW_NEUTRAL, "#f28e52", "#d1495b", "#8e1538"],
        N=256,
    )
    cmap.set_bad(NO_DATA_COLOR)
    return cmap


def yellow_origin_lag_cmap() -> LinearSegmentedColormap:
    """Sequential lag palette whose simultaneous response (zero) is yellow."""

    cmap = LinearSegmentedColormap.from_list(
        "yellow_orange_purple_lag_phase4c",
        [YELLOW_NEUTRAL, "#f4a261", "#e76f51", "#9b5de5", "#3a0ca3"],
        N=256,
    )
    cmap.set_bad(NO_DATA_COLOR)
    return cmap


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


def _figure_save(fig, output: Path, *, dpi: int = FIGURE_DPI) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def save_pixel_map(
    da: xr.DataArray,
    output_path: str | Path,
    title: str,
    cmap: str | LinearSegmentedColormap | None = None,
    *,
    table_output_path: str | Path | None = None,
    export_table: bool = True,
    value_name: str = "value",
    interpretation: str | None = None,
) -> Path:
    """Save a large, publication-resolution map from a lat/lon DataArray."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if export_table:
        export_pixel_table(
            da, table_output_path or output.with_suffix(".csv"), value_name=value_name
        )
    chosen_cmap = cmap or yellow_neutral_diverging_cmap()
    if isinstance(chosen_cmap, str):
        chosen_cmap = plt.get_cmap(chosen_cmap).copy()
        chosen_cmap.set_bad(NO_DATA_COLOR)
    fig, ax = plt.subplots(figsize=(13.5, 10.5))
    da.plot(ax=ax, cmap=chosen_cmap)
    ax.set_title(title, fontsize=17, pad=12)
    ax.tick_params(labelsize=11)
    if interpretation:
        add_interpretive_caption(
            fig,
            metadata=f"Variável: {value_name} | arquivo numérico: {output.with_suffix('.csv').name}",
            interpretation=interpretation,
        )
    fig.tight_layout(rect=(0, 0.08 if interpretation else 0, 1, 1))
    return _figure_save(fig, output)


def _pixel_grid(
    pixels: pd.DataFrame,
    values: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(pixels) != len(values):
        raise ValueError("pixels and values have different lengths.")
    frame = pixels[["lat", "lon"]].copy()
    frame["value"] = np.asarray(values, dtype=float)
    if frame.duplicated(["lat", "lon"]).any():
        raise ValueError("Pixel coordinates are not unique.")
    grid = frame.pivot(index="lat", columns="lon", values="value").sort_index()
    return (
        grid.columns.to_numpy(dtype=float),
        grid.index.to_numpy(dtype=float),
        grid.to_numpy(dtype=float),
    )


def plot_pixel_field(
    ax,
    pixels: pd.DataFrame,
    values: Sequence[float],
    *,
    brazil_geometry,
    boundaries=None,
    significant: Sequence[bool] | None = None,
    cmap: LinearSegmentedColormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "",
    extent: tuple[float, float, float, float] = (-74.2, -28.5, -34.2, 6.2),
):
    """Plot all valid effects; encode FDR independently as black stippling."""

    chosen_cmap = cmap or yellow_neutral_diverging_cmap()
    # Grey is drawn only within Brazil, underneath valid coloured cells.  Thus
    # grey cannot be mistaken for a zero or a non-significant effect.
    brazil_geometry.plot(ax=ax, color=NO_DATA_COLOR, edgecolor="none", zorder=0)
    lon, lat, grid = _pixel_grid(pixels, values)
    mesh = ax.pcolormesh(
        lon,
        lat,
        np.ma.masked_invalid(grid),
        cmap=chosen_cmap,
        vmin=vmin,
        vmax=vmax,
        shading="nearest",
        rasterized=True,
        zorder=2,
    )
    if significant is not None:
        _, _, sig_grid = _pixel_grid(pixels, np.asarray(significant, dtype=float))
        finite = np.isfinite(grid)
        hatched = np.where(finite, sig_grid, np.nan)
        if np.nanmax(hatched, initial=0.0) >= 1.0:
            ax.contourf(
                lon,
                lat,
                hatched,
                levels=[0.5, 1.5],
                colors="none",
                hatches=["...."],
                zorder=3,
            )
    if boundaries is not None:
        boundaries.boundary.plot(ax=ax, color="#111827", linewidth=0.65, zorder=4)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=15, pad=9, fontweight="semibold")
    ax.set_xlabel("Longitude (°)", fontsize=11)
    ax.set_ylabel("Latitude (°)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(color="#6b7280", alpha=0.15, linewidth=0.5)
    return mesh


def add_interpretive_caption(
    fig,
    *,
    metadata: str,
    interpretation: str,
) -> None:
    """Add readable two-line metadata and a literal result-based interpretation."""

    clean = " ".join(str(interpretation).split())
    if not clean:
        raise ValueError("An interpretive caption cannot be empty.")
    fig.text(
        0.5,
        0.045,
        metadata,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#374151",
        wrap=True,
    )
    fig.text(
        0.5,
        0.012,
        f"Interpretação: {clean}",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#111827",
        fontweight="semibold",
        wrap=True,
    )


def save_phase_pixel_panels(
    pixels: pd.DataFrame,
    values_by_phase: Mapping[str, Sequence[float]],
    significant_by_phase: Mapping[str, Sequence[bool]],
    output_path: str | Path,
    *,
    brazil_geometry,
    boundaries,
    title: str,
    colorbar_label: str,
    interpretation: str,
    metadata: str,
    phase_order: Sequence[str] = ("genese", "crescimento", "pico", "decaimento"),
    kind: str = "effect",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """Save the four El Niño source phases as spacious 2x2 map panels."""

    missing = set(phase_order).difference(values_by_phase)
    if missing:
        raise KeyError(f"Missing phase panels: {sorted(missing)}")
    cmap = yellow_neutral_diverging_cmap() if kind == "effect" else yellow_origin_lag_cmap()
    fig, axes = plt.subplots(2, 2, figsize=(22, 16), constrained_layout=False)
    mesh = None
    phase_labels = {
        "genese": "Gênese",
        "crescimento": "Crescimento/acoplamento",
        "pico": "Pico",
        "decaimento": "Decaimento",
    }
    for ax, phase in zip(axes.ravel(), phase_order, strict=True):
        mesh = plot_pixel_field(
            ax,
            pixels,
            values_by_phase[phase],
            brazil_geometry=brazil_geometry,
            boundaries=boundaries,
            significant=significant_by_phase.get(phase),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            title=f"Fase-fonte em t−lag: {phase_labels.get(phase, phase)}",
        )
    fig.suptitle(title, fontsize=21, fontweight="bold", y=0.985)
    if mesh is None:
        raise RuntimeError("No phase panel was plotted.")
    colorbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", fraction=0.035, pad=0.055)
    colorbar.set_label(colorbar_label, fontsize=12)
    colorbar.ax.tick_params(labelsize=10)
    legend = [
        Patch(facecolor=NO_DATA_COLOR, edgecolor="#6b7280", label="Sem dado / fora da máscara"),
        Patch(facecolor="none", edgecolor="#111827", hatch="....", label="FDR BH q<0,10"),
        Patch(facecolor=YELLOW_NEUTRAL, edgecolor="#6b7280", label="Valor neutro/zero (não é ausência)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.075), fontsize=10)
    add_interpretive_caption(fig, metadata=metadata, interpretation=interpretation)
    fig.subplots_adjust(left=0.055, right=0.98, top=0.94, bottom=0.145, wspace=0.13, hspace=0.16)
    return _figure_save(fig, Path(output_path))


def derive_phase_map_interpretation(
    phase_table: pd.DataFrame,
    *,
    value_column: str,
    lag_column: str,
    fdr_column: str,
) -> str:
    """Create a compact caption from actual mapped values, not boilerplate."""

    parts: list[str] = []
    labels = {
        "genese": "gênese",
        "crescimento": "crescimento",
        "pico": "pico",
        "decaimento": "decaimento",
    }
    for phase in ("genese", "crescimento", "pico", "decaimento"):
        group = phase_table[phase_table["fase_fonte_em_t_menos_lag"].eq(phase)]
        values = pd.to_numeric(group[value_column], errors="coerce")
        lags = pd.to_numeric(group[lag_column], errors="coerce")
        significant = group[fdr_column].fillna(False).astype(bool)
        valid = values.notna() & lags.notna()
        if not valid.any():
            parts.append(f"{labels[phase]} sem pixels válidos")
            continue
        positive_fraction = float((values[valid] > 0).mean())
        median_lag = float(lags[valid].median())
        significant_fraction = float(significant[valid].mean())
        sign_text = "predomínio úmido" if positive_fraction >= 0.5 else "predomínio seco"
        parts.append(
            f"{labels[phase]}: {sign_text}, lag mediano {median_lag:.0f} sem e "
            f"{100 * significant_fraction:.1f}% dos pixels com FDR"
        )
    return "; ".join(parts) + "."


def save_unit_lag_heatmap(
    table: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
    interpretation: str,
    metadata: str,
    unit_order: Sequence[str] | None = None,
    phase_order: Sequence[str] = ("genese", "crescimento", "pico", "decaimento"),
    max_lag: float = 78.0,
) -> Path:
    """Map direct aggregate-series lags; annotations retain effect and FDR."""

    required = {
        "nome_unidade",
        "fase_fonte_em_t_menos_lag",
        "lag_sem",
        "r",
        "fdr_bh_0_10",
    }
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"Unit lag table is missing {sorted(missing)}")
    units = list(unit_order or pd.unique(table["nome_unidade"]))
    phase_labels = ["Gênese", "Crescimento", "Pico", "Decaimento"]
    lag_grid = np.full((len(units), len(phase_order)), np.nan)
    annotations = np.full(lag_grid.shape, "", dtype=object)
    for i, unit in enumerate(units):
        for j, phase in enumerate(phase_order):
            row = table[
                table["nome_unidade"].eq(unit)
                & table["fase_fonte_em_t_menos_lag"].eq(phase)
            ]
            if row.empty:
                continue
            record = row.iloc[0]
            lag_grid[i, j] = float(record["lag_sem"])
            marker = "●" if bool(record["fdr_bh_0_10"]) else "○"
            annotations[i, j] = f"{record['lag_sem']:.0f} sem\nr={record['r']:+.2f} {marker}"

    fig_height = max(7.5, 1.15 * len(units) + 3.5)
    fig, ax = plt.subplots(figsize=(15.5, fig_height))
    image = ax.imshow(
        np.ma.masked_invalid(lag_grid),
        cmap=yellow_origin_lag_cmap(),
        norm=Normalize(vmin=0, vmax=max_lag),
        aspect="auto",
    )
    ax.set_xticks(range(len(phase_order)), phase_labels, fontsize=12)
    ax.set_yticks(range(len(units)), units, fontsize=11)
    for i in range(len(units)):
        for j in range(len(phase_order)):
            if annotations[i, j]:
                ax.text(j, i, annotations[i, j], ha="center", va="center", fontsize=10, color="#111827")
    ax.set_title(title, fontsize=19, fontweight="bold", pad=15)
    ax.set_xlabel("Fase do El Niño na semana-fonte t−lag", fontsize=12, labelpad=10)
    ax.set_ylabel("Unidade espacial oficial", fontsize=12)
    ax.set_xticks(np.arange(-0.5, len(phase_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(units), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Melhor lag descritivo da correlação direta (semanas)", fontsize=11)
    fig.text(0.5, 0.095, "● FDR BH q<0,10  |  ○ efeito estimado sem rejeição FDR", ha="center", fontsize=10)
    add_interpretive_caption(fig, metadata=metadata, interpretation=interpretation)
    fig.subplots_adjust(left=0.22, right=0.91, top=0.89, bottom=0.17)
    return _figure_save(fig, Path(output_path))
