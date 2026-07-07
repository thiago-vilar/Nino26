#!/usr/bin/env python3
"""fase4_audit_and_combine.py
================================================================================
Fase 4 - Auditoria por IA + dataset combinado/validado para as proximas etapas.

Gera ARQUIVOS de dados legiveis (CSV/JSON/Parquet) - independentes do output do
notebook - para auditoria automatizada e para alimentar B (pico) e C (chuva):

  data/processed/parquet/statistics/phase4_audit.csv          (por variavel)
  data/processed/parquet/statistics/phase4_audit_crosschecks.json
  data/processed/parquet/modeling/phase4_combined_dataset.parquet|csv
  data/processed/parquet/modeling/phase4_combined_manifest.json

Uso:
  python scripts/fase4_audit_and_combine.py             # oceano+SST (rapido)
  python scripts/fase4_audit_and_combine.py --atmo      # + atmosfera ERA5 (~6 min)
"""
from __future__ import annotations
import sys, re, glob, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import fase4_features as F  # noqa: E402

STATS = F.STATS; MOD = F.PROC / "parquet" / "modeling"
STATS.mkdir(parents=True, exist_ok=True); MOD.mkdir(parents=True, exist_ok=True)
EXCLUDE = re.compile(r"(^year$|^month$|^day$|source|priority|duration|above_p|severity|climatology|month_start)")
deriv = lambda c: bool(re.search(r"_(mean|delta)_\d+d$", c))


def _interior(s):
    fv, lv = s.first_valid_index(), s.last_valid_index()
    return 100.0 if fv is None else round(100 * s.loc[fv:lv].isna().mean(), 3)


def _nsalt(s, k=20):
    d = s.diff().abs().dropna(); m = d[d > 0].median()
    return 0 if (not m or np.isnan(m)) else int((d > k * m).sum())


def build_audit(bb: pd.DataFrame) -> pd.DataFrame:
    num = bb.select_dtypes("number")
    base = [c for c in num.columns if not deriv(c) and not EXCLUDE.search(c)]
    rows = []
    for c in base:
        s = num[c]; gi = _interior(s)
        rows.append((c, F.assign_block(c), "backbone_ocean_sst", int(s.notna().sum()), gi,
                     round(float(s.min()), 4), round(float(s.median()), 4), round(float(s.max()), 4),
                     _nsalt(s), "VERDE" if gi == 0 else ("AMARELO" if gi <= 1 else "VERMELHO")))
    for kind, sub, tag, vl in [("single", "single_levels", "single", F.ERA5_SINGLE_VARS),
                               ("pressure", "pressure_levels", "pressure", F.ERA5_PRESSURE_VARS)]:
        for v in vl:
            st = glob.glob(str(F.ZARR / "era5" / sub / "*" / v / f"era5_{tag}_nino34_{v}_*_daily.zarr"))
            yrs = sorted({int(re.search(r"_(\d{4})_daily", p).group(1)) for p in st if re.search(r"_(\d{4})_daily", p)})
            rows.append((f"atm_{v}", "atmosfera", "era5_nino34", len(yrs), 0.0,
                         (min(yrs) if yrs else np.nan), np.nan, (max(yrs) if yrs else np.nan), 0,
                         "VERDE" if yrs else "VERMELHO"))
    return pd.DataFrame(rows, columns=["variavel", "bloco", "fonte", "n_validos", "gap_interno_pct",
                                       "min", "mediana", "max", "n_saltos", "status"])


def crosschecks(bb: pd.DataFrame) -> dict:
    import xarray as xr
    out = {}
    try:
        pr = xr.open_zarr(str(F.ZARR / "regridded" / "chirps_p25_2015.zarr")).sel(
            lat=slice(*F.BRAZIL_BOX["lat"]), lon=slice(*F.BRAZIL_BOX["lon"]))["precip"]
        out["chirps_2015"] = {"min_mm_dia": round(float(pr.min()), 2), "max_mm_dia": round(float(pr.max()), 1),
                              "n_negativos": int((pr < 0).sum()), "fill_-9999": bool((pr <= -9990).any())}
    except Exception as e:
        out["chirps_2015"] = {"erro": str(e)}
    try:
        o = xr.open_zarr(str(F.ZARR / "regridded" / "noaa_oisst_2015.zarr"))
        ob = o["sst"].sel(lat=slice(*F.NINO34_BOX["lat"]), lon=slice(*F.NINO34_BOX["lon"])).mean(["lat", "lon"]).to_series()
        dif = (ob - bb["nino34_sst"].reindex(pd.to_datetime(ob.index))).abs()
        out["oisst_vs_backbone_2015"] = {"dif_media_C": round(float(dif.mean()), 4), "dif_max_C": round(float(dif.max()), 4)}
    except Exception as e:
        out["oisst_vs_backbone_2015"] = {"erro": str(e)}
    out["era5_cobertura"] = {"ano_fim": 2025, "nota": "atmosfera 1 ano atras do oceano/CHIRPS (2026)"}
    return out


def main(argv):
    atmo = "--atmo" in argv
    bb = F.load_physics_backbone(base="full")

    audit = build_audit(bb)
    audit.to_csv(STATS / "phase4_audit.csv", index=False)
    json.dump(crosschecks(bb), open(STATS / "phase4_audit_crosschecks.json", "w"), indent=2, ensure_ascii=False)

    # dataset combinado
    comb = F.assemble_feature_matrix(atmosphere=atmo, save=False) if atmo else bb.select_dtypes("number").copy()

    def fmax(s, H):
        v = s.to_numpy("float64"); n = len(v); out = np.full(n, np.nan)
        for i in range(n):
            j = min(n, i + 1 + H)
            if j > i + 1 and np.isfinite(v[i + 1:j]).any(): out[i] = np.nanmax(v[i + 1:j])
        return pd.Series(out, index=s.index)

    comb["target_future_max_ssta_150d"] = fmax(bb["nino34_ssta"], 150)
    th90, th95 = float(bb["nino34_ssta"].quantile(.90)), float(bb["nino34_ssta"].quantile(.95))
    comb["target_will_p90"] = (comb["target_future_max_ssta_150d"] >= th90).astype("float")
    comb["target_will_p95"] = (comb["target_future_max_ssta_150d"] >= th95).astype("float")

    base = [c for c in bb.select_dtypes("number").columns if not deriv(c) and not EXCLUDE.search(c)]
    try:
        comb.to_parquet(MOD / "phase4_combined_dataset.parquet"); fmt = "parquet"
    except Exception:
        comb.to_csv(MOD / "phase4_combined_dataset.csv"); fmt = "csv"
    valid = comb.dropna(subset=base + ["target_future_max_ssta_150d"])
    manifest = {"periodo": [str(comb.index.min().date()), str(comb.index.max().date())],
                "n_dias_total": int(len(comb)), "n_colunas": int(comb.shape[1]), "formato": fmt,
                "inclui_atmosfera": atmo,
                "janela_valida_modelagem": [str(valid.index.min().date()), str(valid.index.max().date())],
                "n_amostras_validas": int(len(valid)),
                "features_por_bloco": pd.Series([F.assign_block(c) for c in base]).value_counts().to_dict(),
                "limiares": {"p90": round(th90, 3), "p95": round(th95, 3)},
                "fontes_oceano": "ocean_source_code 1=UFS,2=GLORYS,3=GLO12"}
    json.dump(manifest, open(MOD / "phase4_combined_manifest.json", "w"), indent=2, ensure_ascii=False)
    print("AUDIT:", audit["status"].value_counts().to_dict())
    print("DATASET:", comb.shape, "->", fmt, "| validas:", manifest["n_amostras_validas"],
          "| atmosfera:", atmo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
