from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
ZARR_ROOT = ROOT / "data" / "processed" / "zarr"
OUT_DIR = ROOT / "data" / "processed" / "parquet" / "statistics"


METADATA_VARS = {
    "time",
    "year",
    "month",
    "day",
    "source",
    "source_url",
    "climatology",
    "month_start",
    "ocean_source_code",
    "ocean_feature_source",
}


def _open_zarr(path: Path) -> xr.Dataset | None:
    try:
        return xr.open_zarr(path, consolidated=False)
    except Exception:
        try:
            return xr.open_zarr(path)
        except Exception:
            return None


def _time_array(ds: xr.Dataset) -> Any:
    if "time" in ds.coords:
        return ds["time"].values
    if "time" in ds:
        return ds["time"].values
    return None


def _time_dim(ds: xr.Dataset) -> str | None:
    if "time" in ds.coords and ds["time"].dims:
        return str(ds["time"].dims[0])
    if "time" in ds and ds["time"].dims:
        return str(ds["time"].dims[0])
    return None


def _valid_time_extent(ds: xr.Dataset, var: str) -> tuple[str, str, int]:
    values = _time_array(ds)
    dim = _time_dim(ds)
    if values is None or dim is None or dim not in ds[var].dims:
        return "", "", 0

    times = pd.to_datetime(values)
    if not len(times):
        return "", "", 0

    try:
        mask = ds[var].notnull()
        reduce_dims = [name for name in mask.dims if name != dim]
        if reduce_dims:
            mask = mask.any(dim=reduce_dims)
        valid = pd.Series(mask.values.astype(bool))
        if valid.any():
            valid_times = times[valid.to_numpy()]
            return str(valid_times.min().date()), str(valid_times.max().date()), int(valid.sum())
    except Exception:
        return str(times.min().date()), str(times.max().date()), int(len(times))
    return "", "", 0


def _frequency_from_times(start: str, end: str, count: int) -> str:
    if not start or not end or not count:
        return "indefinido"
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    days = max((end_ts - start_ts).days, 0)
    if count <= 1:
        return "pontual/irregular"
    step = days / max(count - 1, 1)
    if step <= 1.5:
        return "diario"
    if 25 <= step <= 35:
        return "mensal"
    return "irregular"


def _history_label(start: str, end: str) -> str:
    if start and end:
        return f"{start} a {end}"
    return start or end


def _years_from_path(path: Path) -> list[int]:
    return [int(match.group(0)) for match in re.finditer(r"(?:19|20)\d{2}", str(path))]


def _source_from_path(path: Path) -> str:
    rel = path.relative_to(ZARR_ROOT)
    parts = rel.parts
    text = str(rel).replace("\\", "/")
    if text.startswith("brazil_precipitation/"):
        return "CHIRPS - precipitacao Brasil"
    if text.startswith("regridded/chirps"):
        return "CHIRPS - regridado 0.25"
    if text.startswith("regridded/noaa_oisst"):
        return "NOAA OISST - regridado 0.25"
    if text.startswith("cpc_noaa/oisst"):
        return "NOAA OISST - campo SST diario"
    if text.startswith("features/nino34_monthly_oisst"):
        return "NOAA OISST - Nino 3.4 mensal local"
    if text.startswith("features/nino34_daily_oisst"):
        return "NOAA OISST - Nino 3.4 diario"
    if text.startswith("features/nino34_thermocline"):
        return "Fase 3 - termoclina/OHC Nino 3.4"
    if text.startswith("features/nino34_physical"):
        return "Fase 3 - sinal fisico Nino 3.4"
    if text.startswith("features/nino34"):
        return "Fase 3 - derivados Nino 3.4"
    if text.startswith("features/ocean_daily/glorys12_operational"):
        return "Copernicus GLO12 operacional - features oceanicas"
    if text.startswith("features/ocean_daily/glorys12"):
        return "Copernicus GLORYS12 - features oceanicas"
    if text.startswith("features/ocean_daily/noaa_ufs"):
        return "NOAA UFS - features oceanicas"
    if text.startswith("features/ocean_monthly"):
        return "ORAS5 - features mensais"
    if text.startswith("ocean_daily/glorys12_operational"):
        return "Copernicus GLO12 operacional - cubo oceanico"
    if text.startswith("ocean_daily/glorys12"):
        return "Copernicus GLORYS12 - cubo oceanico"
    if text.startswith("ocean_daily/noaa_ufs"):
        return "NOAA UFS - cubo oceanico"
    if text.startswith("ocean_monthly/oras5"):
        return "ORAS5 - cubo oceanico mensal"
    if text.startswith("era5/single_levels"):
        return "ERA5 single levels"
    if text.startswith("era5/pressure_levels"):
        return "ERA5 pressure levels"
    if text.startswith("ctd_noaa"):
        return "NOAA WOD CTD - validacao"
    if text.startswith("validation/tao_triton"):
        return "TAO/TRITON - validacao"
    if text.startswith("validation/argo"):
        return "Argo - validacao"
    if text.startswith("distributions"):
        return "Distribuicoes derivadas"
    if text.startswith("statistics"):
        return "Estatisticas derivadas"
    if parts:
        return parts[0]
    return "desconhecido"


def _path_template(path: Path) -> str:
    rel = str(path.relative_to(ZARR_ROOT)).replace("\\", "/")
    return re.sub(r"(?:19|20)\d{2}", "{year}", rel)


def _is_scientific_data_store(path: Path) -> bool:
    rel = str(path.relative_to(ZARR_ROOT)).replace("\\", "/")
    if rel.startswith("statistics/"):
        return False
    if rel.startswith("modeling/"):
        return False
    return True


def _variable_rows(path: Path, group_paths: list[Path] | None = None) -> list[dict[str, Any]]:
    group_paths = group_paths or [path]
    ds = _open_zarr(path)
    if ds is None:
        return []
    rows: list[dict[str, Any]] = []
    try:
        years: list[int] = []
        for item in group_paths:
            years.extend(_years_from_path(item))
        year_start = min(years) if years else ""
        year_end = max(years) if years else ""
        for var in ds.data_vars:
            var_name = str(var)
            if var_name in METADATA_VARS:
                continue
            start, end, count = _valid_time_extent(ds, var_name)
            if not start and year_start:
                start = str(year_start)
            if not end and year_end:
                end = str(year_end)
            interval = _frequency_from_times(start, end, count)
            if len(group_paths) > 1 and year_start and year_end:
                start = str(year_start)
                end = str(year_end)
            if interval == "indefinido":
                if "monthly" in str(path).lower() or "mensal" in str(path).lower() or "oras5" in str(path).lower():
                    interval = "mensal"
                elif "daily" in str(path).lower() or "diario" in str(path).lower() or re.search(r"(?:19|20)\d{2}", path.name):
                    interval = "diario"
            rows.append(
                {
                    "fonte": _source_from_path(path),
                    "variavel": var_name,
                    "intervalo": interval,
                    "inicio": start,
                    "fim": end,
                    "n_tempos": count,
                    "n_stores_grupo": len(group_paths),
                    "produto_zarr": str(path.relative_to(ROOT)),
                    "padrao_zarr": _path_template(path),
                    "dimensoes": json.dumps({str(k): int(v) for k, v in ds[var_name].sizes.items()}, sort_keys=True),
                }
            )
    finally:
        ds.close()
    return rows


def build_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(path for path in ZARR_ROOT.rglob("*.zarr") if _is_scientific_data_store(path))
    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped_paths[_path_template(path)].append(path)
    raw_rows: list[dict[str, Any]] = []
    for _, group in sorted(grouped_paths.items()):
        raw_rows.extend(_variable_rows(group[0], group_paths=group))
    raw = pd.DataFrame(raw_rows)
    if raw.empty:
        return raw, raw

    grouped_rows: list[dict[str, Any]] = []
    for (source, variable, interval), group in raw.groupby(["fonte", "variavel", "intervalo"], dropna=False):
        starts = [str(value) for value in group["inicio"].dropna() if str(value)]
        ends = [str(value) for value in group["fim"].dropna() if str(value)]
        start = min(starts) if starts else ""
        end = max(ends) if ends else ""
        grouped_rows.append(
            {
                "fonte": source,
                "variavel": variable,
                "intervalo": interval,
                "serie_historica": _history_label(start, end),
                "n_stores": int(group["produto_zarr"].nunique()),
                "n_stores_grupo": int(group["n_stores_grupo"].sum()),
                "exemplo_zarr": group["produto_zarr"].iloc[0],
            }
        )
    grouped = pd.DataFrame(grouped_rows).sort_values(["fonte", "variavel", "intervalo"]).reset_index(drop=True)
    return raw.sort_values(["fonte", "variavel", "produto_zarr"]).reset_index(drop=True), grouped


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw, grouped = build_inventory()
    raw_path = OUT_DIR / "phase4_all_processed_variables_detail.csv"
    grouped_path = OUT_DIR / "phase4_all_processed_variables.csv"
    raw.to_csv(raw_path, index=False)
    grouped.to_csv(grouped_path, index=False)
    print(f"Detalhe: {raw_path} ({len(raw)} linhas)")
    print(f"Tabela: {grouped_path} ({len(grouped)} linhas)")
    if not grouped.empty:
        print(grouped[["fonte", "variavel", "intervalo", "serie_historica"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
