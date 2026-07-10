"""Official Brazilian regions/biomes and area-weighted Phase 4C supports."""

from __future__ import annotations

from pathlib import Path
import re
import unicodedata
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


IBGE_REGIONS_2024_URL = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/"
    "malhas_municipais/municipio_2024/Brasil/BR_Regioes_2024.zip"
)
IBGE_BIOMES_2025_URL = (
    "https://geoftp.ibge.gov.br/informacoes_ambientais/estudos_ambientais/"
    "biomas/vetores/"
    "2025_Biomas-e-Sistema-Costeiro-Marinho-do-Brasil-1-250000_shp.zip"
)

EXPECTED_REGIONS: tuple[str, ...] = (
    "Norte",
    "Nordeste",
    "Centro-Oeste",
    "Sudeste",
    "Sul",
)
EXPECTED_BIOMES: tuple[str, ...] = (
    "Amazônia",
    "Caatinga",
    "Cerrado",
    "Mata Atlântica",
    "Pampa",
    "Pantanal",
)


def _ascii_slug(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text


def _normalise_name(value: object) -> str:
    return _ascii_slug(value).replace("_", "")


def _read_vector(path: str | Path) -> gpd.GeoDataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    # Honour .cpg/default detection first, then choose the decoding with the
    # least replacement/mojibake.  Some archived DBFs and their .cpg disagree,
    # so merely completing a read is not sufficient validation.
    candidates: list[tuple[int, gpd.GeoDataFrame]] = []
    for encoding in (None, "utf-8", "latin1"):
        try:
            candidate = (
                gpd.read_file(path)
                if encoding is None
                else gpd.read_file(path, encoding=encoding)
            )
        except (UnicodeDecodeError, LookupError):
            continue
        text_columns = [
            column for column in candidate.columns if candidate[column].dtype == object
        ]
        score = sum(
            int(candidate[column].astype(str).str.count("�|Ã").sum())
            for column in text_columns
        )
        candidates.append((score, candidate))
    if not candidates:
        raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode {path}")
    gdf = min(candidates, key=lambda item: item[0])[1]
    if gdf.crs is None:
        raise ValueError(f"Official boundary file has no CRS: {path}")
    return gdf.to_crs("EPSG:4326")


def load_ibge_regions(path: str | Path) -> gpd.GeoDataFrame:
    """Load and validate the five official IBGE 2024 macroregions."""

    raw = _read_vector(path)
    required = {"CD_REGIA", "NM_REGIA", "geometry"}
    missing = required.difference(raw.columns)
    if missing:
        raise KeyError(f"IBGE regions file is missing {sorted(missing)}")
    out = raw[["CD_REGIA", "NM_REGIA", "geometry"]].copy()
    canonical = {_normalise_name(name): name for name in EXPECTED_REGIONS}
    out["nome_unidade"] = out["NM_REGIA"].map(
        lambda value: canonical.get(_normalise_name(value), str(value))
    )
    missing_names = set(EXPECTED_REGIONS).difference(out["nome_unidade"])
    if missing_names or len(out) != 5:
        raise ValueError(
            "IBGE regions must contain exactly the five macroregions; "
            f"missing={sorted(missing_names)}, n={len(out)}."
        )
    out["tipo_unidade"] = "regiao"
    out["id_unidade"] = "regiao_" + out["nome_unidade"].map(_ascii_slug)
    out["versao_geometria"] = "IBGE Regioes 2024"
    out["fonte_geometria"] = IBGE_REGIONS_2024_URL
    return gpd.GeoDataFrame(
        out[
            [
                "id_unidade",
                "tipo_unidade",
                "nome_unidade",
                "versao_geometria",
                "fonte_geometria",
                "geometry",
            ]
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )


def load_ibge_biomes(path: str | Path) -> gpd.GeoDataFrame:
    """Load and validate the six continental IBGE biomes (2025 revision)."""

    raw = _read_vector(path)
    required = {"CD_BIOMA", "NM_BIOMA", "geometry"}
    missing = required.difference(raw.columns)
    if missing:
        raise KeyError(f"IBGE biomes file is missing {sorted(missing)}")
    out = raw[["CD_BIOMA", "NM_BIOMA", "geometry"]].copy()
    canonical = {_normalise_name(name): name for name in EXPECTED_BIOMES}
    out["nome_unidade"] = out["NM_BIOMA"].map(
        lambda value: canonical.get(_normalise_name(value), str(value))
    )
    missing_names = set(EXPECTED_BIOMES).difference(out["nome_unidade"])
    if missing_names or len(out) != 6:
        raise ValueError(
            "IBGE biomes must contain exactly the six continental biomes; "
            f"missing={sorted(missing_names)}, n={len(out)}."
        )
    out["tipo_unidade"] = "bioma"
    out["id_unidade"] = "bioma_" + out["nome_unidade"].map(_ascii_slug)
    out["versao_geometria"] = "IBGE Biomas 1:250.000 revisao 2025"
    out["fonte_geometria"] = IBGE_BIOMES_2025_URL
    return gpd.GeoDataFrame(
        out[
            [
                "id_unidade",
                "tipo_unidade",
                "nome_unidade",
                "versao_geometria",
                "fonte_geometria",
                "geometry",
            ]
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )


def build_analysis_units(
    regions: gpd.GeoDataFrame,
    biomes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Combine regions, biomes, and the two requested Northeast intersections."""

    regions = regions.to_crs("EPSG:4326").copy()
    biomes = biomes.to_crs("EPSG:4326").copy()
    northeast_rows = regions[regions["nome_unidade"].eq("Nordeste")]
    if len(northeast_rows) != 1:
        raise ValueError("Exactly one official Nordeste geometry is required.")
    northeast = northeast_rows.geometry.iloc[0]

    cuts: list[dict[str, object]] = []
    for biome_name, cut_name in (
        ("Caatinga", "Caatinga do Nordeste"),
        ("Mata Atlântica", "Mata Atlântica do Nordeste"),
    ):
        biome_rows = biomes[biomes["nome_unidade"].eq(biome_name)]
        if len(biome_rows) != 1:
            raise ValueError(f"Exactly one {biome_name} geometry is required.")
        geometry = biome_rows.geometry.iloc[0].intersection(northeast)
        if geometry.is_empty:
            raise ValueError(f"The official {biome_name} x Nordeste intersection is empty.")
        cuts.append(
            {
                "id_unidade": "recorte_" + _ascii_slug(cut_name),
                "tipo_unidade": "recorte_bioma_regiao",
                "nome_unidade": cut_name,
                "versao_geometria": "intersecao IBGE Regioes 2024 x Biomas 2025",
                "fonte_geometria": f"{IBGE_REGIONS_2024_URL} | {IBGE_BIOMES_2025_URL}",
                "geometry": geometry,
            }
        )
    cuts_gdf = gpd.GeoDataFrame(cuts, geometry="geometry", crs="EPSG:4326")
    units = pd.concat([regions, biomes, cuts_gdf], ignore_index=True)
    units = gpd.GeoDataFrame(units, geometry="geometry", crs="EPSG:4326")
    if units["id_unidade"].duplicated().any():
        raise ValueError("Analysis unit identifiers are not unique.")
    return units


def build_pixel_membership(
    pixels: pd.DataFrame,
    units: gpd.GeoDataFrame,
    *,
    equal_area_crs: str = "EPSG:6933",
    boundary_method: str = "area",
) -> pd.DataFrame:
    """Intersect every CHIRPS cell with official geometries and area-weight it.

    Boundary fractions are measured after projection to an equal-area CRS.
    Aggregation uses ``cos(latitude) * cell_fraction``: the conventional area
    correction for a regular latitude/longitude grid, refined at unit borders by
    the exact official-geometry overlap.  ``centro_na_unidade`` remains available
    for pixel maps, where each cell must receive a single visible classification.
    """

    required_pixels = {"pixel_id", "lat", "lon"}
    missing_pixels = required_pixels.difference(pixels.columns)
    if missing_pixels:
        raise KeyError(f"pixels is missing columns {sorted(missing_pixels)}")
    required_units = {
        "id_unidade",
        "tipo_unidade",
        "nome_unidade",
        "versao_geometria",
        "fonte_geometria",
        "geometry",
    }
    missing_units = required_units.difference(units.columns)
    if missing_units:
        raise KeyError(f"units is missing columns {sorted(missing_units)}")

    if boundary_method not in {"area", "centroid"}:
        raise ValueError("boundary_method must be 'area' or 'centroid'.")
    if boundary_method == "centroid":
        # Fast, topology-safe assignment: each pixel belongs wholly to the unit
        # that contains its centre (fraction 1.0). Standard for teleconnection
        # aggregation on a fine grid; avoids the exact per-cell overlay cost.
        point_frame = gpd.GeoDataFrame(
            pixels[["pixel_id", "lat", "lon"]].copy(),
            geometry=gpd.points_from_xy(pixels["lon"], pixels["lat"]),
            crs="EPSG:4326",
        )
        unit_columns = [
            "id_unidade",
            "tipo_unidade",
            "nome_unidade",
            "versao_geometria",
            "fonte_geometria",
            "geometry",
        ]
        joined = gpd.sjoin(
            point_frame,
            units[unit_columns].to_crs("EPSG:4326"),
            how="inner",
            predicate="within",
        ).drop(columns=["geometry", "index_right"], errors="ignore")
        joined["fracao_pixel_na_unidade"] = 1.0
        joined["area_intersecao_km2_equal_area"] = np.nan
        joined["centro_na_unidade"] = True
        joined["peso_area_coslat"] = np.cos(np.deg2rad(joined["lat"].astype(float)))
        joined["peso_agregacao"] = joined["peso_area_coslat"]
        joined["regra_atribuicao_pixel"] = (
            "centroide: pixel atribuido a unidade que contem seu centro; fraction=1"
        )
        joined["crs_calculo_fracao"] = "n/a (centroide)"
        return joined.sort_values(
            ["tipo_unidade", "id_unidade", "pixel_id"], kind="mergesort"
        ).reset_index(drop=True)

    lon_values = np.sort(pd.unique(pd.to_numeric(pixels["lon"], errors="raise")))
    lat_values = np.sort(pd.unique(pd.to_numeric(pixels["lat"], errors="raise")))
    if len(lon_values) < 2 or len(lat_values) < 2:
        raise ValueError("At least two unique latitudes and longitudes are required.")
    delta_lon = float(np.median(np.diff(lon_values)))
    delta_lat = float(np.median(np.diff(lat_values)))
    if delta_lon <= 0 or delta_lat <= 0:
        raise ValueError("Pixel coordinates do not define a positive regular grid.")
    if not np.allclose(np.diff(lon_values), delta_lon, rtol=0, atol=1e-7):
        raise ValueError("Longitude spacing is not regular.")
    if not np.allclose(np.diff(lat_values), delta_lat, rtol=0, atol=1e-7):
        raise ValueError("Latitude spacing is not regular.")

    cell_frame = gpd.GeoDataFrame(
        pixels[["pixel_id", "lat", "lon"]].copy(),
        geometry=[
            box(
                float(lon) - delta_lon / 2.0,
                float(lat) - delta_lat / 2.0,
                float(lon) + delta_lon / 2.0,
                float(lat) + delta_lat / 2.0,
            )
            for lat, lon in zip(pixels["lat"], pixels["lon"], strict=True)
        ],
        crs="EPSG:4326",
    )
    cell_area = cell_frame[["pixel_id", "geometry"]].to_crs(equal_area_crs)
    cell_area_lookup = pd.Series(
        cell_area.geometry.area.to_numpy(dtype=float),
        index=cell_area["pixel_id"],
    )

    unit_columns = [
        "id_unidade",
        "tipo_unidade",
        "nome_unidade",
        "versao_geometria",
        "fonte_geometria",
        "geometry",
    ]
    intersections = gpd.overlay(
        cell_frame,
        units[unit_columns].to_crs("EPSG:4326"),
        how="intersection",
        keep_geom_type=False,
    )
    intersections = intersections[~intersections.geometry.is_empty].copy()
    intersection_area_m2 = intersections.to_crs(equal_area_crs).geometry.area.to_numpy(
        dtype=float
    )
    full_area_m2 = intersections["pixel_id"].map(cell_area_lookup).to_numpy(dtype=float)
    intersections["fracao_pixel_na_unidade"] = np.clip(
        intersection_area_m2 / full_area_m2, 0.0, 1.0
    )
    intersections["area_intersecao_km2_equal_area"] = intersection_area_m2 / 1_000_000.0

    point_frame = gpd.GeoDataFrame(
        pixels[["pixel_id", "lat", "lon"]].copy(),
        geometry=gpd.points_from_xy(pixels["lon"], pixels["lat"]),
        crs="EPSG:4326",
    )
    centre_join = gpd.sjoin(
        point_frame,
        units[["id_unidade", "geometry"]].to_crs("EPSG:4326"),
        how="inner",
        predicate="within",
    )
    centre_keys = pd.MultiIndex.from_frame(centre_join[["pixel_id", "id_unidade"]])
    intersection_keys = pd.MultiIndex.from_frame(
        intersections[["pixel_id", "id_unidade"]]
    )
    intersections["centro_na_unidade"] = intersection_keys.isin(centre_keys)

    membership = pd.DataFrame(intersections.drop(columns="geometry"))
    membership["peso_area_coslat"] = np.cos(
        np.deg2rad(membership["lat"].astype(float))
    )
    membership["peso_agregacao"] = (
        membership["peso_area_coslat"] * membership["fracao_pixel_na_unidade"]
    )
    membership["regra_atribuicao_pixel"] = (
        "intersecao pixel-geometria em EPSG:6933; peso cos(lat)*fracao; "
        "centro_na_unidade reservado ao mapa"
    )
    membership["crs_calculo_fracao"] = equal_area_crs
    membership = membership.sort_values(
        ["tipo_unidade", "id_unidade", "pixel_id"], kind="mergesort"
    ).reset_index(drop=True)

    # Fractions within each base partition may split a border cell, but cannot
    # exceed one apart from tiny projection/topology tolerances.
    base = membership[membership["tipo_unidade"].isin(["regiao", "bioma"])]
    fraction_sum = base.groupby(["tipo_unidade", "pixel_id"])[
        "fracao_pixel_na_unidade"
    ].sum()
    if (fraction_sum > 1.001).any():
        raise ValueError("Official-unit overlap assigned more than one full pixel.")
    return membership


def brazil_pixel_ids(membership: pd.DataFrame) -> pd.Index:
    """Return pixel centres inside Brazil according to the region partition."""

    ids = membership.loc[
        membership["tipo_unidade"].eq("regiao")
        & membership["centro_na_unidade"].astype(bool),
        "pixel_id",
    ]
    return pd.Index(pd.unique(ids), name="pixel_id")


def _response_column_lookup(response: pd.DataFrame) -> dict[str, object]:
    lookup: dict[str, object] = {}
    for column in response.columns:
        key = str(column)
        if key in lookup:
            raise ValueError(f"Response columns collide after string conversion: {key}")
        lookup[key] = column
    return lookup


def aggregate_area_weighted_response(
    response: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    min_valid_area_fraction: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate weekly pixel anomalies directly for every spatial unit."""

    if not 0.0 < min_valid_area_fraction <= 1.0:
        raise ValueError("min_valid_area_fraction must lie in (0, 1].")
    lookup = _response_column_lookup(response)
    series: dict[str, np.ndarray] = {}
    coverage: dict[str, np.ndarray] = {}
    for unit_id, members in membership.groupby("id_unidade", sort=False):
        keys = members["pixel_id"].astype(str)
        available = keys.isin(lookup)
        members = members.loc[available].copy()
        if members.empty:
            series[str(unit_id)] = np.full(len(response), np.nan)
            coverage[str(unit_id)] = np.zeros(len(response))
            continue
        columns = [lookup[str(pixel_id)] for pixel_id in members["pixel_id"]]
        values = response.loc[:, columns].to_numpy(dtype=float)
        weights = members["peso_agregacao"].to_numpy(dtype=float)
        valid = np.isfinite(values)
        valid_weight = valid * weights[None, :]
        denominator = valid_weight.sum(axis=1)
        total_weight = float(weights.sum())
        fraction = denominator / total_weight
        numerator = np.where(valid, values * weights[None, :], 0.0).sum(axis=1)
        aggregate = np.divide(
            numerator,
            denominator,
            out=np.full(len(response), np.nan, dtype=float),
            where=denominator > 0,
        )
        aggregate[fraction < min_valid_area_fraction] = np.nan
        series[str(unit_id)] = aggregate
        coverage[str(unit_id)] = fraction
    return (
        pd.DataFrame(series, index=response.index),
        pd.DataFrame(coverage, index=response.index),
    )


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Iterable[float],
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.asarray(list(quantiles), dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.full(len(quantiles), np.nan)
    order = np.argsort(values[valid], kind="mergesort")
    sorted_values = values[valid][order]
    sorted_weights = weights[valid][order]
    positions = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / sorted_weights.sum()
    return np.interp(quantiles, positions, sorted_values)


def summarize_pixel_best_lags_by_unit(
    pixel_best: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Complement direct unit correlations with weighted pixel-lag distributions."""

    required = {
        "pixel_id",
        "variavel",
        "condicao_fonte",
        "best_lag_sem_descritivo",
        "r_no_best_lag_descritivo",
        "best_lag_sem_fdr",
    }
    missing = required.difference(pixel_best.columns)
    if missing:
        raise KeyError(f"pixel_best is missing columns {sorted(missing)}")
    merged = membership.merge(pixel_best, on="pixel_id", how="inner", validate="many_to_many")
    rows: list[dict[str, object]] = []
    group_columns = [
        "variavel",
        "condicao_fonte",
        "tipo_enso_fonte",
        "fase_fonte_em_t_menos_lag",
        "id_unidade",
        "tipo_unidade",
        "nome_unidade",
        "versao_geometria",
        "fonte_geometria",
    ]
    for keys, group in merged.groupby(group_columns, sort=False, dropna=False):
        row = dict(zip(group_columns, keys, strict=True))
        weight = group["peso_agregacao"].to_numpy(dtype=float)
        lag_all = group["best_lag_sem_descritivo"].to_numpy(dtype=float)
        r_all = group["r_no_best_lag_descritivo"].to_numpy(dtype=float)
        lag_sig = group["best_lag_sem_fdr"].to_numpy(dtype=float)
        q_all = _weighted_quantile(lag_all, weight, [0.25, 0.50, 0.75])
        q_sig = _weighted_quantile(lag_sig, weight, [0.25, 0.50, 0.75])
        valid_all = np.isfinite(lag_all)
        significant = np.isfinite(lag_sig)
        total_weight = weight[valid_all].sum()
        significant_weight = weight[significant].sum()
        positive = valid_all & (r_all > 0)
        negative = valid_all & (r_all < 0)
        row.update(
            {
                "n_pixels": int(valid_all.sum()),
                "n_pixels_com_lag_fdr": int(significant.sum()),
                "fracao_area_com_lag_fdr": (
                    significant_weight / total_weight if total_weight > 0 else np.nan
                ),
                "lag_descritivo_p25_sem": q_all[0],
                "lag_descritivo_mediano_sem": q_all[1],
                "lag_descritivo_p75_sem": q_all[2],
                "lag_fdr_p25_sem": q_sig[0],
                "lag_fdr_mediano_sem": q_sig[1],
                "lag_fdr_p75_sem": q_sig[2],
                "fracao_area_r_positivo": (
                    weight[positive].sum() / total_weight if total_weight > 0 else np.nan
                ),
                "fracao_area_r_negativo": (
                    weight[negative].sum() / total_weight if total_weight > 0 else np.nan
                ),
                "camada_analitica": (
                    "distribuicao_pixelar_complementar; inferencia primaria na serie "
                    "espacial agregada"
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
