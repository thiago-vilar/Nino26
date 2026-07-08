"""Materializa os insumos dos notebooks da Fase 3 (3A-3G).

Gera, a partir dos dados locais (OISST global bruto/Zarr, GLORYS12, feature
stores da Fase 3):

1. features/tropical_atlantic_sst_daily.csv  - ATL3/ATL4/TNA/TSA SST + SSTA
   (climatologia dia-do-ano 1991-2020, janela 15 d - mesma convencao Nino 3.4)
2. features/equatorial_pacific_ssta_weekly_by_lon.parquet - SSTA semanal
   (W-SUN) da banda 2S-2N por longitude (120E-280E)
3. features/ssh_equatorial_daily_by_lon_events.parquet - SSH 1S-1N por
   longitude nos anos de evento (Kelvin)
4. features/nino34_dhw_daily.csv - DHW ONI local 12 semanas (HotSpot >=0.5 C)
5. interim/fase3_map_cache/*.nc - campos mensais Nov/Dez 1991-2020 + picos
   (mapas compostos do 3B)

Uso: .venv\\Scripts\\python scripts\\fase3_build_inputs.py
Reexecucao e idempotente (pula o que ja existe; use --force para refazer).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nino_brasil.data.anomalies import daily_anomaly, dayofyear_climatology
from nino_brasil.features.dhw import degree_heating_weeks, thermal_window_mean
from nino_brasil.features.nino import tropical_atlantic_sst_indices

FEAT = ROOT / "data/processed/parquet/features"
MAPC = ROOT / "data/interim/fase3_map_cache"
OISST_ZARR = ROOT / "data/processed/zarr/cpc_noaa/oisst"
OISST_RAW = ROOT / "data/raw/cpc_noaa/oisst"
OCEAN_DAILY = ROOT / "data/processed/zarr/ocean_daily"
EVENT_YEARS = [1997, 1998, 2015, 2016, 2023, 2024, 2025, 2026]
PEAKS = [(1982, 12), (1997, 11), (2015, 11)]


def open_oisst_year(year: int) -> xr.Dataset:
    """Zarr v2 (consolidated=False) com fallback ao netCDF bruto (stores v3)."""
    zpath = OISST_ZARR / f"sst.day.mean.{year}.zarr"
    try:
        return xr.open_zarr(zpath)
    except Exception:
        pass
    try:
        return xr.open_zarr(zpath, consolidated=False)
    except Exception:
        return xr.open_dataset(OISST_RAW / f"sst.day.mean.{year}.nc")


def oisst_years() -> list[int]:
    years = set()
    for p in OISST_ZARR.glob("sst.day.mean.*.zarr"):
        years.add(int(p.name.split(".")[-2]))
    for p in OISST_RAW.glob("sst.day.mean.*.nc"):
        years.add(int(p.name.split(".")[-2]))
    return sorted(years)


def build_atlantic_and_band(force: bool) -> None:
    out_atl = FEAT / "tropical_atlantic_sst_daily.csv"
    out_band = FEAT / "equatorial_pacific_ssta_weekly_by_lon.parquet"
    if out_atl.exists() and out_band.exists() and not force:
        print("[skip] atlantico/banda ja materializados")
        return
    atl_frames, band_frames = [], []
    for year in oisst_years():
        ds = open_oisst_year(year)
        sst = ds["sst"]
        atl_frames.append(tropical_atlantic_sst_indices(sst).to_dataframe().reset_index())
        band = sst.sel(lat=slice(-2, 2), lon=slice(120, 280)).mean("lat").to_pandas()
        band.columns = [f"{c:.3f}" for c in band.columns]
        band_frames.append(band)
        ds.close()
        print(f"  oisst {year} ok")
    atl = pd.concat(atl_frames, ignore_index=True)
    atl["time"] = pd.to_datetime(atl["time"])
    atl = atl.sort_values("time").drop_duplicates("time").set_index("time")
    for col in ["atl3_sst", "atl4_sst", "tna_sst", "tsa_sst"]:
        da = xr.DataArray(atl[col].values, coords={"time": atl.index}, dims=("time",))
        clim = dayofyear_climatology(da.sel(time=slice("1991-01-01", "2020-12-31")), 15, "time")
        atl[col.replace("_sst", "_ssta")] = daily_anomaly(da, climatology=clim, window_days=15, time_name="time").values
    atl.reset_index().to_csv(out_atl, index=False)
    print(f"[ok] {out_atl.relative_to(ROOT)} {atl.shape}")

    band = pd.concat(band_frames).sort_index()
    band.index = pd.to_datetime(band.index)
    doy = band.index.dayofyear.values
    ref = band.loc["1991":"2020"]
    doy_ref = ref.index.dayofyear.values
    clim = np.full((367, band.shape[1]), np.nan)
    for d in range(1, 367):
        m = doy_ref == d
        if m.any():
            clim[d] = np.nanmean(ref.values[m], axis=0)
    c = pd.DataFrame(clim[1:367])
    cc = pd.concat([c.iloc[-15:], c, c.iloc[:15]]).rolling(15, center=True, min_periods=1).mean().iloc[15:-15]
    clim[1:367] = cc.values
    ssta = pd.DataFrame(band.values - clim[doy], index=band.index, columns=band.columns)
    ssta.resample("W-SUN").mean().to_parquet(out_band)
    print(f"[ok] {out_band.relative_to(ROOT)}")


def build_ssh_events(force: bool) -> None:
    out = FEAT / "ssh_equatorial_daily_by_lon_events.parquet"
    if out.exists() and not force:
        print("[skip] ssh eventos ja materializado")
        return
    frames = []
    for year in EVENT_YEARS:
        for sub in ("glorys12", "glorys12_operational"):
            for spath in sorted((OCEAN_DAILY / sub / str(year)).glob("*.zarr")):
                try:
                    ds = xr.open_zarr(spath)
                except Exception:
                    ds = xr.open_zarr(spath, consolidated=False)
                if "sea_surface_height" in ds:
                    ssh = ds["sea_surface_height"].sel(lat=slice(-1, 1)).mean("lat").to_pandas()
                    ssh.columns = [f"{c:.3f}" for c in ssh.columns]
                    frames.append(ssh)
                ds.close()
        print(f"  ssh {year} ok")
    ssh = pd.concat(frames).sort_index()
    ssh.index = pd.to_datetime(ssh.index)
    ssh.to_parquet(out)
    print(f"[ok] {out.relative_to(ROOT)} {ssh.shape}")


def build_dhw(force: bool) -> None:
    """DHW Nino 3.4 oficializado para a Fase 3.

    Regra: HotSpot diario = SSTA quando SSTA >=0.5 C; valores menores viram
    zero. O DHW e a soma movel em 12 semanas, expressa em C-weeks. A validacao
    temporal adicional usa a media movel de 12 semanas da SSTA e marca
    consolidacao quando essa media fica >=0.5 C por pelo menos 20 semanas
    consecutivas.
    """
    out = FEAT / "nino34_dhw_daily.csv"
    out_var = FEAT / "nino34_dhw_variants.csv"
    if out.exists() and out_var.exists() and not force:
        print("[skip] dhw ja materializado")
        return
    d = pd.read_csv(FEAT / "nino34_daily_oisst.csv", parse_dates=["time"])
    ssta = xr.DataArray(d["nino34_ssta"].values, coords={"time": d["time"].values}, dims=("time",))
    ssta_values = d["nino34_ssta"].to_numpy(dtype=float)
    hotspot = np.where(ssta_values >= 0.5, ssta_values, 0.0)
    dhw = degree_heating_weeks(ssta, threshold_c=0.5, window_weeks=12)
    mean12 = thermal_window_mean(ssta, window_weeks=12)
    mean_ge = pd.Series(mean12.values >= 0.5, index=d.index).fillna(False)
    run_days = mean_ge.groupby((mean_ge != mean_ge.shift(fill_value=False)).cumsum()).cumcount() + 1
    run_days = run_days.where(mean_ge, 0).astype(int)
    table = pd.DataFrame(
        {
            "time": d["time"],
            "nino34_ssta": d["nino34_ssta"],
            "hotspot_ge_0p5_c": hotspot,
            "dhw_cweek_0p5_12w": dhw.values,
            "oni_12w_mean_c": mean12.values,
            "oni_12w_ge_0p5": mean_ge.astype(int).values,
            "oni_12w_run_ge_0p5_days": run_days.values,
            "oni_12w_run_ge_0p5_weeks": (run_days / 7.0).round(2).values,
            "elnino_thermal_persistent_20w": (run_days >= 140).astype(int).values,
        }
    )
    table.to_csv(out, index=False)
    table.to_csv(out_var, index=False)
    print(f"[ok] {out.relative_to(ROOT)} e {out_var.relative_to(ROOT)} (HotSpot >=0.5 C; janela=12 sem; persistencia=20 sem)")


def build_map_cache(force: bool) -> None:
    MAPC.mkdir(parents=True, exist_ok=True)
    jobs = [(y, m) for y in range(1991, 2021) for m in (11, 12)] + PEAKS
    for y, m in jobs:
        f = MAPC / f"sst_month_{y}_{m:02d}.nc"
        if f.exists() and not force:
            continue
        ds = open_oisst_year(y)
        field = ds["sst"].sel(time=f"{y}-{m:02d}").mean("time").sel(lat=slice(-30, 30)).load()
        field.to_netcdf(f)
        ds.close()
        print(f"  mapa {y}-{m:02d} ok")
    print(f"[ok] {MAPC.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="refaz mesmo se ja existir")
    args = ap.parse_args()
    build_dhw(args.force)
    build_atlantic_and_band(args.force)
    build_ssh_events(args.force)
    build_map_cache(args.force)
    print("insumos da Fase 3 prontos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
