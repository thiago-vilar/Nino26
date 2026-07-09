#!/usr/bin/env python3
"""build_master_weekly.py - Matriz-mestre semanal unificada (Fase 2 -> Fase 3/4).

1. SERIE OCEANICA UNIFICADA 1981-2026: UFS(1981-92) -> GLORYS12(1993+) -> GLO12
   (cauda), reduzida ao eixo semanal W-SUN a partir do sinal fisico diario
   `nino34_physical_signal.csv`. `ocean_source_code` (1/2/3) e metadado de fonte.
2. ATMOSFERA ERA5 SEMANAL (14 variaveis, caixa Nino 3.4) para o feedback de
   Bjerknes. Extracao cacheada em `era5_nino34_daily_cache.parquet` (reexecucao
   rapida) e gravacao final ATOMICA (nunca deixa arquivo truncado).
3. Dados mensais so calibram; nunca entram em estatistica avancada.
4. Validacao CTD (WOD) da termoclina.

Saidas:
  data/processed/parquet/features/nino34_master_weekly.csv
  data/processed/parquet/statistics/phase2_master_audit.csv
  data/processed/parquet/statistics/phase2_master_validation.csv
  data/processed/parquet/statistics/phase2_ctd_validation.csv

Uso:
  .venv-wsl/bin/python scripts/build_master_weekly.py --era5-years 1981:2026
  .venv-wsl/bin/python scripts/build_master_weekly.py --ocean-only   # sem ERA5
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FEAT = ROOT / "data/processed/parquet/features"
STATS = ROOT / "data/processed/parquet/statistics"
ERA5 = ROOT / "data/processed/zarr/era5"
CTD = ROOT / "data/processed/zarr/ctd_noaa/wod"
STATS.mkdir(parents=True, exist_ok=True)

RHO_AIR = 1.2
CD_NEUTRAL = 1.3e-3
NINO34 = {"lat": (-5.0, 5.0), "lon": (-170.0, -120.0)}
CLIM = ("1991-01-01", "2020-12-31")
ERA5_DAILY_CACHE = FEAT / "era5_nino34_daily_cache.parquet"

OCEAN_COLS = {
    "nino34_ssta": "nino34_ssta",
    "d20_nino34_mean_m": "d20_m",
    "thermocline_tilt_m": "tilt_m",
    "thermocline_tilt_slope_m_per_degree": "tilt_slope",
    "ohc_0_100_nino34_j_m2": "ohc_0_100",
    "ohc_0_300_nino34_j_m2": "ohc_0_300",
    "ohc_0_700_nino34_j_m2": "ohc_0_700",
    "ohc_300_700_nino34_j_m2": "ohc_300_700",
    "ssh_nino34_mean_m": "ssh_m",
    "wwv_equatorial_pacific_m3": "wwv",
    "temperature_50m_nino34_c": "t50m",
    "temperature_100m_nino34_c": "t100m",
    "temperature_150m_nino34_c": "t150m",
    "temperature_200m_nino34_c": "t200m",
    "temperature_300m_nino34_c": "t300m",
    "temperature_500m_nino34_c": "t500m",
    "temperature_700m_nino34_c": "t700m",
}
ERA5_SINGLE = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "mslp",
    "total_column_water_vapour": "tcwv",
    "surface_latent_heat_flux": "slhf",
    "surface_sensible_heat_flux": "sshf",
    "surface_net_solar_radiation": "ssr",
    "surface_net_thermal_radiation": "str",
}
ERA5_PRESSURE = {
    "u_component_of_wind": [("u850", 850.0), ("u200", 200.0)],
    "vertical_velocity": [("omega850", 850.0), ("omega500", 500.0)],
    "divergence": [("div850", 850.0)],
}
ATMO_ANOM = ["u10", "v10", "mslp", "tcwv", "slhf", "sshf", "ssr", "str",
             "u850", "u200", "omega850", "omega500", "div850"]


def _open(path):
    import xarray as xr
    try:
        return xr.open_zarr(path)
    except Exception:
        return xr.open_zarr(path, consolidated=False)


def _box_mean(da):
    """Media espacial; os stores ERA5 `*nino34*` ja sao recortados na caixa."""
    return da.mean([d for d in da.dims if d != "time"])


def _atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Grava CSV atomico: escreve num .tmp e so entao substitui o alvo.
    Uma execucao interrompida NUNCA deixa o arquivo final truncado."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp)
    os.replace(tmp, path)


def _monthly_kind_frames(year: int) -> pd.DataFrame:
    """Extrai o ano a partir dos stores mensais por tipo (layout operacional
    recente: era5_single_nino34_YYYYMM_daily.zarr + era5_pressure_...).
    Retorna DataFrame diario com as 13 colunas de ATMO_ANOM (vazio se nada)."""
    single_map = {v: k for k, v in ERA5_SINGLE.items()}  # out -> era5 name
    rows = []
    for month in range(1, 13):
        sp = ERA5 / f"single_levels/{year}/era5_single_nino34_{year}{month:02d}_daily.zarr"
        pp = ERA5 / f"pressure_levels/{year}/era5_pressure_nino34_{year}{month:02d}_daily.zarr"
        if not sp.exists() or not pp.exists():
            continue
        ds, dp = _open(sp), _open(pp)
        frame = pd.DataFrame({out: _box_mean(ds[var]).to_pandas()
                              for out, var in single_map.items() if var in ds})
        lvln = "pressure_level" if "pressure_level" in dp.coords else "level"
        for var, levels in ERA5_PRESSURE.items():
            if var not in dp:
                continue
            for out, lvl in levels:
                frame[out] = _box_mean(dp[var].sel({lvln: lvl})).to_pandas()
        ds.close(); dp.close()
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows).sort_index()
    out.index = pd.to_datetime(out.index)
    return out


def extract_era5(years: range) -> pd.DataFrame:
    """Serie diaria da caixa Nino 3.4 para as variaveis ERA5. Cacheia em parquet;
    se o cache cobre os anos pedidos, reusa (reexecucao instantanea). Anos sem o
    layout anual por variavel sao completados pelos stores mensais por tipo."""
    cached = None
    if ERA5_DAILY_CACHE.exists():
        try:
            cached = pd.read_parquet(ERA5_DAILY_CACHE)
            cached.index = pd.to_datetime(cached.index)
        except Exception as exc:
            print(f"  [era5] cache ignorado ({exc}); reextraindo")
            cached = None
    if cached is not None:
        missing = [y for y in years if y not in set(cached.index.year)]
        if not missing:
            cov = f"{cached.index.min().year}-{cached.index.max().year}"
            print(f"  [era5] cache reusado ({cov}); sem reabrir zarr")
            return cached.loc[f"{min(years)}-01-01":f"{max(years)}-12-31"].sort_index()
        # completa apenas os anos ausentes via stores mensais por tipo
        extra = [f for f in (_monthly_kind_frames(y) for y in missing) if not f.empty]
        if extra:
            df = pd.concat([cached] + extra).sort_index()
            df = df[~df.index.duplicated(keep="last")]
            try:
                df.to_parquet(ERA5_DAILY_CACHE)
                print(f"  [era5] cache estendido com {[int(y) for y in missing]}: {df.index.max().date()}")
            except Exception as exc:
                print(f"  [era5] nao consegui gravar cache ({exc})")
            return df.loc[f"{min(years)}-01-01":f"{max(years)}-12-31"].sort_index()
    cols = {}
    for var, out in ERA5_SINGLE.items():
        parts = []
        for y in years:
            fs = glob.glob(str(ERA5 / f"single_levels/{y}/{var}/*nino34*.zarr"))
            if not fs:
                continue
            d = _open(fs[0])
            name = [str(v) for v in d.data_vars][0]
            parts.append(_box_mean(d[name]).to_pandas())
        if parts:
            cols[out] = pd.concat(parts).sort_index()
            print(f"  [era5] {out} ok ({len(cols[out])} dias)")
    for var, levels in ERA5_PRESSURE.items():
        per_year = {out: [] for out, _ in levels}
        for y in years:
            fs = glob.glob(str(ERA5 / f"pressure_levels/{y}/{var}/*nino34*.zarr"))
            if not fs:
                continue
            d = _open(fs[0])
            name = [str(v) for v in d.data_vars][0]
            lvln = "pressure_level" if "pressure_level" in d.coords else "level"
            for out, lvl in levels:
                if lvln in d[name].dims:
                    per_year[out].append(_box_mean(d[name].sel({lvln: lvl})).to_pandas())
        for out, _ in levels:
            if per_year[out]:
                cols[out] = pd.concat(per_year[out]).sort_index()
                print(f"  [era5] {out} ok ({len(cols[out])} dias)")
    df = pd.DataFrame(cols)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    missing = [y for y in years if df.empty or y not in set(df.index.year)]
    extra = [f for f in (_monthly_kind_frames(y) for y in missing) if not f.empty]
    if extra:
        df = pd.concat([df] + extra).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        print(f"  [era5] anos completados por stores mensais: {[int(y) for y in missing]}")
    try:
        df.to_parquet(ERA5_DAILY_CACHE)
        print(f"  [era5] cache gravado: {ERA5_DAILY_CACHE.name} {df.shape}")
    except Exception as exc:
        print(f"  [era5] nao consegui gravar cache ({exc})")
    return df


def doy_anomaly(s: pd.Series) -> pd.Series:
    s = s.astype(float).sort_index()
    base = s.loc[CLIM[0]:CLIM[1]]
    clim = base.groupby(base.index.dayofyear).mean()
    c = clim.reindex(range(1, 367)).interpolate().bfill().ffill()
    c = pd.concat([c.iloc[-15:], c, c.iloc[:15]]).rolling(15, center=True, min_periods=1).mean().iloc[15:-15]
    return (s - s.index.dayofyear.map(c)).astype(float)


def build(era5_years: range, ocean_only: bool = False) -> pd.DataFrame:
    phys = pd.read_csv(FEAT / "nino34_physical_signal.csv", parse_dates=["time"],
                       usecols=["time"] + list(OCEAN_COLS) + ["ocean_source_code"]).set_index("time")
    ocean = phys[list(OCEAN_COLS)].rename(columns=OCEAN_COLS)
    ocean_w = ocean.resample("W-SUN").mean()
    src_w = phys["ocean_source_code"].resample("W-SUN").agg(lambda x: x.mode().iloc[0] if len(x.dropna()) else np.nan)

    if ocean_only:
        atmo_w = pd.DataFrame(index=ocean_w.index)
        p3 = FEAT / "phase3_indices_semanais.csv"
        if p3.exists():
            t = pd.read_csv(p3, parse_dates=["week_ending_sunday"]).set_index("week_ending_sunday")
            if "tau_x_anom_nino34_pa" in t.columns:
                atmo_w["tau_x_anom"] = t["tau_x_anom_nino34_pa"]
    else:
        atmo = extract_era5(era5_years)
        if not atmo.empty:
            tau_raw = (RHO_AIR * CD_NEUTRAL * atmo["u10"].abs() * atmo["u10"]).rename("tau_x_raw")
            atmo["tau_x_anom"] = doy_anomaly(tau_raw)
            for c in [c for c in ATMO_ANOM if c in atmo.columns]:
                atmo[f"{c}_anom"] = doy_anomaly(atmo[c])
            keep = ["tau_x_anom"] + [f"{c}_anom" for c in ATMO_ANOM if c in atmo.columns]
            atmo_w = atmo[keep].resample("W-SUN").mean()
        else:
            atmo_w = pd.DataFrame(index=ocean_w.index)

    master = ocean_w.join(atmo_w, how="outer")
    master["ocean_source_code"] = src_w
    master = master[~master.index.duplicated(keep="first")].sort_index()
    full = pd.date_range(master.index.min(), master.index.max(), freq="W-SUN")
    master = master.reindex(full)
    master.index.name = "week_ending_sunday"
    return master


def validate(master: pd.DataFrame) -> pd.DataFrame:
    checks = []
    idx = master.index
    checks.append(("indice_monotonico_crescente", bool(idx.is_monotonic_increasing)))
    checks.append(("sem_semanas_duplicadas", bool(not idx.duplicated().any())))
    checks.append(("grade_semanal_W-SUN_regular",
                   bool((idx.to_series().diff().dropna() == pd.Timedelta(weeks=1)).all())))
    checks.append(("eixo_1981_a_2026", bool(idx.min().year <= 1981 and idx.max().year >= 2026)))
    vazias = [c for c in master.columns if c != "ocean_source_code" and master[c].notna().sum() == 0]
    checks.append(("nenhuma_variavel_totalmente_vazia", len(vazias) == 0))
    src = master["ocean_source_code"].dropna()
    trans = src[src.diff() != 0]
    checks.append(("transicoes_fonte_oceanica_ordenadas",
                   bool(list(trans.dropna().values) == sorted(set(trans.dropna().values)))))
    out = pd.DataFrame(checks, columns=["checagem", "passou"])
    if vazias:
        out = pd.concat([out, pd.DataFrame([{"checagem": f"vazia:{v}", "passou": False} for v in vazias])],
                        ignore_index=True)
    return out


def audit(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in master.columns:
        if c == "ocean_source_code":
            continue
        s = master[c]
        rows.append({
            "variavel": c,
            "inicio": s.first_valid_index().date() if s.notna().any() else None,
            "fim": s.last_valid_index().date() if s.notna().any() else None,
            "semanas_validas": int(s.notna().sum()),
            "cobertura_%": round(100 * s.notna().mean(), 1),
            "maior_lacuna_semanas": int(s.isna().astype(int).groupby(s.notna().cumsum()).sum().max()) if s.isna().any() else 0,
        })
    return pd.DataFrame(rows)


def ctd_validation(master: pd.DataFrame, years: range) -> pd.DataFrame:
    rows = []
    for y in years:
        f = CTD / f"{y}/wod_ctd_{y}.zarr"
        i