#!/usr/bin/env python3
"""Fase 4C - lags sinal/resposta por FASE do ciclo, por PIXEL, REGIAO e BIOMA.

Complementa o mapa pixel-a-pixel com a distribuicao espaco-temporal agregada em
unidades oficiais: as 5 regioes IBGE 2024, os 6 biomas IBGE 2025 e os recortes
Caatinga do Nordeste e Mata Atlantica do Nordeste. Para cada preditor do Pacifico
(t-lag) mede-se a correlacao com a anomalia padronizada de chuva (t), condicionada
a fase do ciclo (genese, crescimento, pico, decaimento) de El Nino e La Nina.

Camadas:
  1. serie agregada por unidade (cos(lat) x fracao de area) -> lag direto por
     unidade e fase, com N_eff/FDR (inferencia primaria por unidade);
  2. distribuicao pixelar dos melhores lags por unidade (complementar).

Saidas: tabelas em data/processed/parquet/statistics/ e figuras codificadas
(Fig_4C*) com legenda interpretativa em data/processed/figures/fase4/.

Uso:
    python scripts/run_fase4c_regional.py            # todos os preditores selecionados
    python scripts/run_fase4c_regional.py --quick    # so nino34_ssta (validacao rapida)
    python scripts/run_fase4c_regional.py --exact-membership  # overlay exato por area
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.maps.figure_registry import figure_code, register_only  # noqa: E402
from nino_brasil.maps.plot_pixel_maps import save_unit_lag_heatmap  # noqa: E402
from nino_brasil.maps.spatial_support import (  # noqa: E402
    aggregate_area_weighted_response,
    build_analysis_units,
    build_pixel_membership,
    load_ibge_biomes,
    load_ibge_regions,
)
from nino_brasil.stats.lag_analysis import (  # noqa: E402
    PHASE_ORDER,
    best_from_long_table,
    build_source_conditions,
    harmonic_deseasonalize_predictors,
    lagged_correlation_exact,
    load_selected_predictors,
    result_to_long_table,
)

FEAT = ROOT / "data/processed/parquet/features"
STATS = ROOT / "data/processed/parquet/statistics"
FIGS = ROOT / "data/processed/figures/fase4"
IBGE = ROOT / "data/interim/ibge"
REG_SHP = IBGE / "BR_Regioes_2024/BR_Regioes_2024.shp"
BIO_SHP = IBGE / "Biomas_2025/lml_bioma_e250k_v20250911_A.shp"
MEMBERSHIP = STATS / "phase4C_pixel_membership.parquet"
LAGS = list(range(0, 79, 2))
KEY_PREDICTOR = "nino34_ssta"


def pixel_best_lag_fast(x, response, lags):
    """Melhor lag por pixel (|r| max) via correlacao vetorizada rapida."""
    import numpy as _np
    R = response.to_numpy(dtype="float64")
    best_r = _np.full(R.shape[1], _np.nan); best_ab = _np.full(R.shape[1], -1.0)
    best_lag = _np.full(R.shape[1], _np.nan)
    for L in lags:
        xl = x.shift(L).reindex(response.index).to_numpy()
        m = _np.isfinite(xl)
        if m.sum() < 60:
            continue
        xv = xl[m]; Rm = R[m]
        nok = _np.isfinite(Rm).sum(0)
        xz = (xv - xv.mean()) / (xv.std() or 1.0)
        mu = _np.nanmean(Rm, 0); sd = _np.nanstd(Rm, 0); sd[sd == 0] = _np.nan
        Rz = (Rm - mu) / sd
        r = _np.nansum(_np.where(_np.isfinite(Rz), Rz, 0) * xz[:, None], 0) / _np.where(nok > 0, nok, 1)
        r = _np.where(nok >= 60, r, _np.nan)
        upd = _np.abs(r) > best_ab
        best_r = _np.where(upd, r, best_r); best_ab = _np.where(upd, _np.abs(r), best_ab)
        best_lag = _np.where(upd, float(L), best_lag)
    return best_r, best_lag


def load_units():
    return build_analysis_units(load_ibge_regions(REG_SHP), load_ibge_biomes(BIO_SHP))


def load_membership(units, pixels, *, exact: bool) -> pd.DataFrame:
    if MEMBERSHIP.exists():
        return pd.read_parquet(MEMBERSHIP)
    method = "area" if exact else "centroid"
    membership = build_pixel_membership(pixels, units, boundary_method=method)
    membership.to_parquet(MEMBERSHIP)
    return membership


def unit_name_lookup(units) -> pd.Series:
    return units.set_index("id_unidade")["nome_unidade"]


def compute_unit_lags(
    predictors: pd.DataFrame,
    unit_series: pd.DataFrame,
    phase_table: pd.DataFrame,
    conditions: dict,
) -> pd.DataFrame:
    """Direct per-unit lag table for every predictor/condition/unit."""

    deseason, _ = harmonic_deseasonalize_predictors(predictors)
    tables: list[pd.DataFrame] = []
    for name in deseason.columns:
        for condition in conditions.values():
            result = lagged_correlation_exact(
                deseason[name], unit_series, LAGS, condition, phase_table
            )
            tables.append(
                result_to_long_table(
                    result,
                    predictor_name=name,
                    condition=condition,
                    column_name="id_unidade",
                )
            )
    return pd.concat(tables, ignore_index=True)


def heatmap_table_for(long_table: pd.DataFrame, predictor: str, event_type: str,
                      names: pd.Series) -> pd.DataFrame:
    """Best per (unit, phase) lag for one predictor and ENSO type, for the heatmap."""

    condition_names = [f"{event_type}_{phase}" for phase in PHASE_ORDER]
    subset = long_table[
        long_table["variavel"].eq(predictor)
        & long_table["condicao_fonte"].isin(condition_names)
    ]
    best = best_from_long_table(
        subset,
        group_columns=["id_unidade", "fase_fonte_em_t_menos_lag"],
        require_fdr=False,
    )
    if best.empty:
        return best
    best = best.copy()
    best["nome_unidade"] = best["id_unidade"].map(names)
    return best


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="so o preditor-chave")
    parser.add_argument("--exact-membership", action="store_true",
                        help="overlay exato por area (lento; recomendado fora do sandbox)")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    FIGS.mkdir(parents=True, exist_ok=True)

    response = pd.read_parquet(FEAT / "phase4_chirps_weekly_zanom.parquet")
    response.index = pd.to_datetime(response.index)
    response = response.sort_index()
    pixels = pd.read_csv(FEAT / "phase4_chirps_pixels.csv")

    master = pd.read_csv(FEAT / "nino34_master_weekly.csv",
                         parse_dates=["week_ending_sunday"]).set_index("week_ending_sunday")
    phase_table = pd.read_csv(STATS / "phase4A_fases_semanais.csv",
                              parse_dates=["week_ending_sunday"]).set_index("week_ending_sunday")
    phase_table = phase_table.reindex(response.index).fillna(
        {"fase": "neutro", "tipo": "neutro", "event_id": ""}
    )

    available = [c for c in master.columns if master[c].notna().any()]
    if args.quick:
        predictor_names, contract = [KEY_PREDICTOR], "quick:nino34_ssta"
    else:
        predictor_names, contract = load_selected_predictors(
            STATS, available, allow_all_fallback=True
        )
    if KEY_PREDICTOR not in predictor_names:  # o indice canonico sempre entra no mapa/heatmap
        predictor_names = [KEY_PREDICTOR] + list(predictor_names)
    predictors = master[[c for c in predictor_names if c in master.columns]].reindex(response.index)
    print(f"[4C] preditores={list(predictors.columns)} (contrato={contract})")

    units = load_units()
    membership = load_membership(units, pixels, exact=args.exact_membership)
    names = unit_name_lookup(units)
    unit_series, coverage = aggregate_area_weighted_response(response, membership)
    unit_series = unit_series.loc[:, [c for c in unit_series.columns if unit_series[c].notna().any()]]
    print(f"[4C] unidades com serie valida={unit_series.shape[1]} | semanas={len(unit_series)}")

    conditions = build_source_conditions(phase_table)
    long_table = compute_unit_lags(predictors, unit_series, phase_table, conditions)
    long_table["nome_unidade"] = long_table["id_unidade"].map(names)
    long_table["tipo_unidade"] = long_table["id_unidade"].map(
        units.set_index("id_unidade")["tipo_unidade"]
    )
    out_table = STATS / "phase4C_lags_por_unidade.csv"
    long_table.to_csv(out_table, index=False)
    print(f"[tabela] {out_table.relative_to(ROOT)} ({len(long_table)} linhas)")

    # Figura-chave: heatmap unidade x fase do lag de resposta (El Nino), preditor-chave.
    unit_order = list(
        units.sort_values(["tipo_unidade", "nome_unidade"])["nome_unidade"].unique()
    )
    for event_type in ("el_nino", "la_nina"):
        table = heatmap_table_for(long_table, KEY_PREDICTOR, event_type, names)
        if table.empty:
            print(f"[aviso] sem lags para {event_type}")
            continue
        order = [u for u in unit_order if u in set(table["nome_unidade"])]
        interp = (
            f"Lag de resposta da chuva ao sinal {KEY_PREDICTOR} em {event_type.replace('_',' ')} "
            f"por regiao IBGE e bioma; celulas com marcador cheio passam FDR BH q<0,10. "
            f"Compara o tempo de atuacao entre Nordeste, Sul, Caatinga e Mata Atlantica."
        )
        idx = 1 if event_type == "el_nino" else 2
        code = figure_code(4, "C", idx, f"lags_regiao_bioma_{event_type}")
        out_png = FIGS / f"{code}.png"
        meta = f"Fonte: CHIRPS 0.25 + OISST | unidades IBGE 2024/2025 | lags {LAGS[0]}-{LAGS[-1]} sem | {KEY_PREDICTOR}"
        # save_unit_lag_heatmap monta e salva a figura (com legenda interpretativa);
        # register_only apenas indexa o codigo/legenda no registro central.
        save_unit_lag_heatmap(
            table,
            out_png,
            title=f"4C - Lag por regiao e bioma | {KEY_PREDICTOR} | {event_type.replace('_',' ')}",
            interpretation=interp,
            metadata=meta,
            unit_order=order,
        )
        register_only(code, out_png.name, phase=4, block="C", interpretation=interp,
                      metadata=meta, registry_dir=FIGS,
                      title=f"Lag por regiao e bioma ({event_type})")
        print(f"[figura] {out_png.name} (heatmap unidade x fase, {event_type})")
    # Mapa pixel-a-pixel do preditor canonico (grade preenchida, amarelo neutro) + CSV para 4D.
    try:
        from nino_brasil.maps.plot_pixel_maps import plot_pixel_field, yellow_neutral_diverging_cmap
        import matplotlib.pyplot as _plt
        best_r, best_lag = pixel_best_lag_fast(master[KEY_PREDICTOR].reindex(response.index), response, LAGS)
        pixel_tbl = pixels[["pixel_id", "lat", "lon"]].copy()
        pixel_tbl["r_no_best_lag"] = best_r
        pixel_tbl["best_lag_sem"] = best_lag
        pixel_tbl.to_csv(STATS / "phase4C_best_lag_pixel.csv", index=False)
        brazil = units[units["tipo_unidade"].eq("regiao")].dissolve()
        regs = units[units["tipo_unidade"].eq("regiao")]
        fig, ax = _plt.subplots(figsize=(13, 11))
        mesh = plot_pixel_field(ax, pixels[["lat", "lon"]], best_r, brazil_geometry=brazil,
                                boundaries=regs, cmap=yellow_neutral_diverging_cmap(),
                                vmin=-0.5, vmax=0.5,
                                title=f"r no melhor lag | {KEY_PREDICTOR} -> anomalia de chuva CHIRPS")
        cb = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("r (amarelo=neutro; azul=seco; vermelho=umido)")
        from nino_brasil.maps.figure_registry import save_registered_figure
        save_registered_figure(fig, phase=4, block="C", index=3, slug="mapa_pixel_r_melhor_lag",
            interpretation="r por pixel no melhor lag entre " + KEY_PREDICTOR + " e a anomalia de chuva; "
                           "cinza=fora da mascara (nao e zero); amarelo=efeito neutro; grade preenchida sem buracos.",
            metadata=f"Fonte: CHIRPS 0.25 + OISST | Brasil pixel-a-pixel | lags {LAGS[0]}-{LAGS[-1]} sem",
            figures_dir=FIGS, reserve_bottom=0.12)
        print("[figura] Fig_4C3_mapa_pixel_r_melhor_lag.png + phase4C_best_lag_pixel.csv")
    except Exception as exc:
        print(f"[aviso] mapa pixelar 4C nao gerado: {exc}")
    print("[4C] concluido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
