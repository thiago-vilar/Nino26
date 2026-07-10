#!/usr/bin/env python3
"""Fase 6 - distribuicao no Brasil com RF/XGBoost + XAI (analoga a Fase 4C, com ML).

Para cada unidade oficial (regiao IBGE, bioma, recortes Caatinga/Mata Atlantica do
Nordeste) treina RF/XGBoost prevendo a anomalia de chuva agregada a partir da janela
deslizante de variaveis do Pacifico, com validacao cronologica e baseline. Ranqueia
variaveis por ganho. Usa o cache de membership da Fase 4C.

Uso:  python scripts/run_fase6_brazil_ml.py [--model rf|xgb] [--quick]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import nino_brasil.models.phase5_cycle_ml as p5  # noqa: E402
import nino_brasil.models.phase6_brazil_ml as p6  # noqa: E402
from nino_brasil.maps.figure_registry import save_registered_figure  # noqa: E402
from nino_brasil.maps.spatial_support import aggregate_area_weighted_response  # noqa: E402

FEAT = ROOT / "data/processed/parquet/features"
STATS = ROOT / "data/processed/parquet/statistics"
FIGS = ROOT / "data/processed/figures/fase6"
PREDICTORS = ["nino34_ssta", "d20_m", "ohc_0_300", "ssh_m", "tau_x_anom", "wwv", "tilt_m"]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["rf", "xgb"], default="rf")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    FIGS.mkdir(parents=True, exist_ok=True)

    membership = pd.read_parquet(STATS / "phase4C_pixel_membership.parquet")
    if args.quick:  # reduz custo: so unidades do NE e so as colunas necessarias
        alvo = membership["id_unidade"].astype(str).str.contains("nordeste")
        membership = membership[alvo] if alvo.any() else membership
        cols = [str(pid) for pid in membership["pixel_id"].unique()]
        response = pd.read_parquet(FEAT / "phase4_chirps_weekly_zanom.parquet", columns=cols)
    else:
        response = pd.read_parquet(FEAT / "phase4_chirps_weekly_zanom.parquet")
    response.index = pd.to_datetime(response.index); response = response.sort_index()
    unit_series, _ = aggregate_area_weighted_response(response, membership)
    unit_series = unit_series.loc[:, [c for c in unit_series.columns if unit_series[c].notna().any()]]
    if args.quick:
        keep = [c for c in unit_series.columns if "nordeste" in c][:2] or list(unit_series.columns[:2])
        unit_series = unit_series[keep]

    master = pd.read_csv(FEAT / "nino34_master_weekly.csv",
                         parse_dates=["week_ending_sunday"]).set_index("week_ending_sunday")
    phase_table = pd.read_csv(STATS / "phase4A_fases_semanais.csv",
                              parse_dates=["week_ending_sunday"]).set_index("week_ending_sunday")
    phase_table = phase_table.reindex(response.index).fillna({"fase": "neutro", "tipo": "neutro", "event_id": ""})
    preds = [c for c in PREDICTORS if c in master.columns]
    lags = range(4, 29, 8) if args.quick else range(4, 53, 4)
    X = p5.build_lagged_features(master[preds], preds, lags=lags).reindex(response.index)

    conditions = ["todas"] if args.quick else ["todas", "el_nino_pico", "la_nina_pico"]
    result = p6.fit_unit_teleconnection(X, unit_series, phase_table, model=args.model,
                                        conditions=conditions, n_splits=4)
    result.skill.to_csv(STATS / f"phase6_skill_{args.model}.csv", index=False)
    p6.top_importances_by_unit(result.importances).to_csv(
        STATS / f"phase6_importancias_{args.model}.csv", index=False)
    print(f"[6] {args.model.upper()} teleconexao ML: {len(result.skill)} (unidade x condicao)")

    if not result.skill.empty:
        sub = result.skill[result.skill["condicao"] == "todas"].sort_values("r_ml")
        fig, ax = plt.subplots(figsize=(13, max(6, 0.5 * len(sub) + 3)))
        ax.barh(sub["id_unidade"], sub["r_ml"], color="#2f80c1")
        ax.set_title(f"6 - Skill ML por unidade ({args.model.upper()}): r fora-da-amostra", fontsize=14)
        ax.set_xlabel("correlacao r (validacao cronologica)")
        interp = ("Correlacao fora-da-amostra do RF/XGB prevendo a anomalia de chuva de cada "
                  "regiao/bioma a partir do Pacifico defasado; positivo = ha sinal aprendivel.")
        save_registered_figure(fig, phase=6, block="A", index=1, slug=f"skill_ml_unidades_{args.model}",
            interpretation=interp,
            metadata=f"Fonte: CHIRPS 0.25 + master semanal | unidades IBGE | {args.model.upper()} TimeSeriesSplit",
            figures_dir=FIGS, title="6A - Skill da teleconexao por unidade (ML)")
        print("[figura] Fig_6A1_skill_ml_unidades_%s.png" % args.model)
    print("[6] concluido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
