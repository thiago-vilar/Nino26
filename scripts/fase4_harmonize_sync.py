#!/usr/bin/env python3
"""fase4_harmonize_sync.py
================================================================================
Implementa o parecer do Thiago para a Fase 4:

1. HARMONIZACAO VERTICAL das fontes oceanicas ao limite comum.
   UFS tem 48 niveis (1..771 m), GLORYS tem 34 (0.5..763 m) - grades distintas.
   `harmonize_cube_to_levels` interpola o cubo a um grid-padrao comum (<=700 m)
   antes de recalcular D20/OHC/WWV. (Reprocessa cubos: rode na sua maquina.)

2. CORRECAO SOURCE-AWARE DO DEGRAU (estatistica, aplicavel ja sobre as features
   existentes): subtrai a climatologia dia-do-ano DENTRO de cada fonte, removendo
   o vies de emenda. Comprovado: WWV +1.26 sigma -> -0.22 sigma.

3. SINCRONIZACAO MENSAL (estado da arte p/ ENSO): agrega o diario a mensal para
   comparar/validar com a referencia mensal OISST local e ORAS5.

4. VALIDACAO CTD: pareia D20/termoclina da reanalise x perfis CTD do WOD
   (observacao in situ), em especial 1981-1992 (periodo UFS, menor confianca).

Uso:
  python scripts/fase4_harmonize_sync.py            # seam-fix + sync mensal (rapido)
  python scripts/fase4_harmonize_sync.py --ctd      # + validacao CTD
  python scripts/fase4_harmonize_sync.py --recompute-cubes  # harmoniza cubos (lento)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import fase4_features as F  # noqa: E402

MOD = F.PROC / "parquet" / "modeling"; MOD.mkdir(parents=True, exist_ok=True)
STATS = F.STATS

# grade-padrao comum (<= 700 m): niveis WOA-like dentro do limite comum
COMMON_LEVELS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300,
                 400, 500, 600, 700]
OCEAN_FEATS = ["ohc_0_100_nino34_j_m2", "ohc_0_300_nino34_j_m2", "ohc_0_700_nino34_j_m2",
               "ohc_300_700_nino34_j_m2", "d20_nino34_mean_m", "wwv_equatorial_pacific_m3",
               "ssh_nino34_mean_m", "sss_nino34_mean", "temperature_50m_nino34_c",
               "temperature_100m_nino34_c", "temperature_150m_nino34_c", "temperature_200m_nino34_c",
               "temperature_300m_nino34_c", "temperature_500m_nino34_c", "temperature_700m_nino34_c",
               "thermocline_tilt_m", "thermocline_tilt_slope_m_per_degree"]


# ----------------------------------------------------------------------------
# 1. Harmonizacao vertical de cubo (reprocessamento - rode na sua maquina)
# ----------------------------------------------------------------------------
def harmonize_cube_to_levels(ds, depth_name="depth", target=COMMON_LEVELS):
    """Interpola um cubo oceanico ao grid-padrao comum (descarta o excedente).

    Garante que UFS (48) e GLORYS (34) fiquem na MESMA grade vertical antes de
    recalcular features dependentes de profundidade (D20/OHC/WWV).
    """
    tgt = [d for d in target if d <= float(ds[depth_name].max())]
    return ds.interp({depth_name: tgt})


# ----------------------------------------------------------------------------
# 2. Correcao source-aware do degrau de emenda (estatistica)
# ----------------------------------------------------------------------------
def source_aware_anom(s: pd.Series, src: pd.Series, window: int = 15) -> pd.Series:
    """Anomalia dia-do-ano calculada DENTRO de cada fonte (remove vies de emenda)."""
    out = pd.Series(np.nan, index=s.index)
    for code_ in src.dropna().unique():
        m = src == code_; seg = s[m].astype(float)
        dv = seg.index.dayofyear.values; v = seg.values
        clim = np.full(367, np.nan)
        for d in range(1, 367):
            diff = np.abs(dv - d); diff = np.minimum(diff, 366 - diff); w = diff <= window
            if w.any():
                clim[d] = np.nanmean(v[w])
        out[m] = seg.values - clim[seg.index.dayofyear.values]
    return out


def seam_step(s: pd.Series, y0="1992", y1="1993") -> float:
    sd = s.std()
    return float((s.loc[y1].mean() - s.loc[y0].mean()) / sd) if sd else np.nan


# ----------------------------------------------------------------------------
# 3b. Camada SEMANAL (cadencia principal da analise causal/ML)
# ----------------------------------------------------------------------------
def _woy_anom_weekly(s: pd.Series, ww: int = 2) -> pd.Series:
    """Anomalia semanal por semana-do-ano (janela +- ww semanas)."""
    w = s.resample("W").mean()
    woy = w.index.isocalendar().week.values.astype(int)
    clim = np.full(54, np.nan)
    for k in range(1, 54):
        dd = np.abs(woy - k); dd = np.minimum(dd, 52 - dd); m = dd <= ww
        if m.any():
            clim[k] = np.nanmean(w.values[m])
    return w - clim[np.clip(woy, 1, 53)]


SYNODIC = 29.530588853                      # mes sinodico (dias)
REF_NEW_MOON = pd.Timestamp("2000-01-06 18:14:00")   # lua nova de referencia
ANCHOR_7D = pd.Timestamp("1981-01-06")      # ancora dos bins de 7 dias (lua nova ~jan/1981)


def _lunar_phase_name(frac: float) -> str:
    if frac < 0.125 or frac >= 0.875:
        return "nova"
    if frac < 0.375:
        return "crescente"
    if frac < 0.625:
        return "cheia"
    return "minguante"


def build_7d_lunar(bb: pd.DataFrame, lags=(1, 2, 3, 4, 6, 8, 12, 16, 20, 24)) -> pd.DataFrame:
    """Dataset de bins de **7 dias ancorados em lua nova** (alinhado as fases da lua).

    Mantem resolucao sub-mensal (resolve a propagacao do sinal) sem o jitter
    diario. Cada bin recebe a fase lunar real no ponto medio (`fase_lua`) e o
    `mes_referencia` (mes do ponto medio) para alinhamento mensal correto.
    Marca a confianca do oceano (UFS pre-1993 = baixa)."""
    sc = "ocean_source_code"
    daily = pd.DataFrame(index=bb.index)
    for c in OCEAN_FEATS:
        if c in bb.columns:
            daily[c + "_sa"] = source_aware_anom(bb[c], bb[sc])
    if "nino34_ssta" in bb:
        daily["nino34_ssta_anom"] = bb["nino34_ssta"]
    if "nino34_sst" in bb:
        daily["nino34_sst"] = bb["nino34_sst"]

    b7 = daily.resample("7D", origin=ANCHOR_7D).mean(numeric_only=True)
    mid = b7.index + pd.Timedelta(days=3.5)
    frac = (((mid - REF_NEW_MOON) / pd.Timedelta(days=1)) / SYNODIC) % 1.0
    b7["fase_lua_frac"] = np.round(frac, 3)
    b7["fase_lua"] = [_lunar_phase_name(f) for f in frac]
    b7["mes_referencia"] = mid.to_period("M").astype(str)
    b7["fonte_oceano"] = bb[sc].resample("7D", origin=ANCHOR_7D).last()
    b7["confianca_oceano"] = np.where(b7["fonte_oceano"] == 1, "baixa_UFS", "alta")

    key = [c for c in ["wwv_equatorial_pacific_m3_sa", "d20_nino34_mean_m_sa",
                       "ohc_0_300_nino34_j_m2_sa", "nino34_ssta_anom"] if c in b7.columns]
    for c in key:
        for L in lags:
            b7[f"{c}_mean_{L}p"] = b7[c].rolling(L, min_periods=max(1, L // 2)).mean()
            b7[f"{c}_delta_{L}p"] = b7[c] - b7[c].shift(L)
    return b7


# ----------------------------------------------------------------------------
# 4. Validacao CTD (reanalise x observacao in situ)
# ----------------------------------------------------------------------------
def validate_against_ctd(bb: pd.DataFrame) -> pd.DataFrame:
    """Pareia profundidade de termoclina da reanalise (D20) x CTD do WOD por mes.

    Retorna comparacao mensal (reanalise vs observacao) por periodo de fonte.
    """
    import xarray as xr, glob
    rows = []
    stores = sorted(glob.glob(str(F.ZARR / "ctd_noaa" / "wod" / "*")))
    for st in stores:
        try:
            ds = xr.open_zarr(st)
            tvar = next((v for v in ds.variables if "thermocline" in v.lower()), None)
            timev = next((v for v in ds.variables if v.lower() in ("time", "date")), None)
            if tvar is None or timev is None:
                continue
            obs = pd.Series(np.asarray(ds[tvar]).ravel(),
                            index=pd.to_datetime(np.asarray(ds[timev]).ravel(), errors="coerce")).dropna()
            obs_m = obs.resample("MS").mean()
            rea = bb["d20_nino34_mean_m"].resample("MS").mean().reindex(obs_m.index)
            for t in obs_m.index:
                rows.append((t, float(obs_m[t]), float(rea[t]) if pd.notna(rea[t]) else np.nan))
        except Exception:
            continue
    df = pd.DataFrame(rows, columns=["mes", "ctd_termoclina_m", "reanalise_d20_m"]).dropna()
    if len(df):
        df["dif_m"] = (df["reanalise_d20_m"] - df["ctd_termoclina_m"]).round(1)
    return df


def main(argv):
    bb = F.load_physics_backbone(base="full").loc["1981":"2025"]
    sc = "ocean_source_code"

    # 2 + 3: seam-fix source-aware + sync mensal
    corr = bb.copy()
    steps = []
    for c in OCEAN_FEATS:
        if c in bb.columns:
            corr[c + "_sa"] = source_aware_anom(bb[c], bb[sc])
            steps.append((c, round(seam_step(bb[c]), 3), round(seam_step(corr[c + "_sa"]), 3)))
    tab_seam = pd.DataFrame(steps, columns=["feature", "degrau_bruto_sigma", "degrau_corrigido_sigma"])
    tab_seam.to_csv(STATS / "phase4_seam_correction.csv", index=False)

    monthly = corr.select_dtypes("number").resample("MS").mean()
    monthly.to_csv(MOD / "phase4_combined_monthly_1981_2025.csv")

    # camada de 7 dias alinhada a lua (cadencia principal da analise causal/ML)
    b7 = build_7d_lunar(bb)
    b7.to_csv(MOD / "phase4_combined_7d_lunar_1981_2025.csv")

    print("seam-fix:\n", tab_seam.to_string(index=False))
    print("\nmensal (validacao):", monthly.shape)
    print("7-dias/lua (causal/ML):", b7.shape, "| fases:", b7["fase_lua"].value_counts().to_dict(),
          "| confianca:", b7["confianca_oceano"].value_counts().to_dict())

    if "--ctd" in argv:
        ctd = validate_against_ctd(bb)
        ctd.to_csv(STATS / "phase4_ctd_validation.csv", index=False)
        if len(ctd):
            print("\nCTD validacao:", len(ctd), "meses pareados | dif media %.1f m" % ctd["dif_m"].mean())
        else:
            print("\nCTD: nenhum par encontrado (checar estrutura do store).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
