#!/usr/bin/env python3
"""Fase 5 - ciclo ENSO com Random Forest / XGBoost + XAI (SHAP, PDP).

Executa: janela deslizante (lags 4-52) -> classificacao das 4 fases com validacao
cronologica (TimeSeriesSplit) -> ranking de variaveis por ganho -> RFECV ->
dependencia parcial (PDP) dos limiares de Bjerknes; e alvos por evento
(Y_pico, Y_tempo_para_pico, Y_duracao) com leave-one-event-out. SHAP e opcional.

Uso:
    python scripts/run_fase5_cycle_ml.py             # RF (padrao)
    python scripts/run_fase5_cycle_ml.py --model xgb
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
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT), str(ROOT / "notebooks" / "fase4")):
    if p not in sys.path:
        sys.path.insert(0, p)

import nino_brasil.models.phase5_cycle_ml as p5  # noqa: E402
from nino_brasil.maps.figure_registry import save_registered_figure  # noqa: E402

FEAT = ROOT / "data/processed/parquet/features"
STATS = ROOT / "data/processed/parquet/statistics"
FIGS = ROOT / "data/processed/figures/fase5"
PREDICTORS = [
    "nino34_ssta", "d20_m", "tilt_m", "ohc_0_100", "ohc_0_300", "ssh_m", "wwv",
    "t100m", "t150m", "tau_x_anom", "u850_anom", "mslp_anom",
]
LAGS = list(range(4, 53, 2))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["rf", "xgb"], default="rf")
    parser.add_argument("--rfecv", action="store_true", help="roda RFECV (mais lento)")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    FIGS.mkdir(parents=True, exist_ok=True)
    STATS.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(FEAT / "nino34_master_weekly.csv",
                         parse_dates=["week_ending_sunday"]).set_index("week_ending_sunday")
    phase_table = pd.read_csv(STATS / "phase4A_fases_semanais.csv",
                              parse_dates=["week_ending_sunday"]).set_index("week_ending_sunday")

    predictors = [c for c in PREDICTORS if c in master.columns]
    X = p5.build_lagged_features(master[predictors], predictors, lags=LAGS)
    result = p5.fit_phase_classifier(X, phase_table["fase"], model=args.model)
    result.importances.to_csv(STATS / f"phase5_importancia_{args.model}.csv", index=False)
    result.cv_scores.to_csv(STATS / f"phase5_cv_{args.model}.csv", index=False)
    print(f"[5] {args.model.upper()} classificador de fases | F1-macro medio="
          f"{result.cv_scores['f1_macro'].mean():.3f} | features={len(result.features)}")

    # Alvos por evento (projecao do ciclo) + leave-one-event-out.
    import fase4_utils as u  # noqa
    events = u.enso_events(u.load_oni_monthly())
    targets = p5.build_event_targets(events)
    targets.to_csv(STATS / "phase5_alvos_por_evento.csv", index=False)
    print(f"[5] alvos por evento: {len(targets)} eventos | LOO folds="
          f"{len(p5.leave_one_event_out_indices(targets['event_id']))}")

    if args.rfecv:
        labels = phase_table["fase"]
        selected, selector = p5.rfecv_select(
            X.join(labels.rename("__f")).dropna().drop(columns="__f"),
            labels.reindex(X.index).loc[X.dropna().index],
            task="classification",
        )
        pd.Series(selected, name="variavel").to_csv(STATS / "phase5_rfecv_selecionadas.csv", index=False)
        print(f"[5] RFECV selecionou {len(selected)} de {X.shape[1]} features")

    # Figura 1: top variaveis por ganho (com destaque para precursoras de recarga).
    top = result.importances.head(18).iloc[::-1]
    cores = ["#d1495b" if "__recharge" in v else "#2f80c1" for v in top["variavel"]]
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.barh(top["variavel"], top["importancia_ganho"], color=cores)
    ax.set_title(f"5 - Importancia por ganho ({args.model.upper()}) na classificacao das 4 fases", fontsize=14)
    ax.set_xlabel("importancia (ganho/impureza)")
    interp = (
        f"Variaveis mais determinantes das 4 fases do ciclo segundo {args.model.upper()} "
        f"(F1-macro medio {result.cv_scores['f1_macro'].mean():.2f}, TimeSeriesSplit). "
        f"Barras vermelhas = precursoras de recarga (OHC/SSH/D20/tau_x) nos lags 15-20 sem."
    )
    save_registered_figure(fig, phase=5, block="A", index=1, slug=f"importancia_fases_{args.model}",
        interpretation=interp,
        metadata=f"Fonte: master semanal NINO26 | janela {LAGS[0]}-{LAGS[-1]} sem | validacao cronologica",
        figures_dir=FIGS, title="5A - Variaveis determinantes das fases (ML)")
    print("[figura] Fig_5A1_importancia_fases_%s.png" % args.model)

    # Figura 2: PDP dos limiares nao-lineares das top variaveis fisicas.
    try:
        top_feats = list(result.importances["variavel"].head(4))
        pdp = p5.partial_dependence_frame(result.model, X, top_feats)
        pdp.to_csv(STATS / f"phase5_pdp_{args.model}.csv", index=False)
        fig, ax = plt.subplots(figsize=(13, 8))
        for feat, sub in pdp.groupby("variavel"):
            ax.plot(sub["valor"], sub["resposta_pdp"], marker="o", ms=3, label=feat)
        ax.set_title("5A - Dependencia parcial (PDP): limiares nao-lineares", fontsize=14)
        ax.set_xlabel("valor padronizado da variavel"); ax.set_ylabel("resposta parcial media")
        ax.legend(fontsize=8)
        save_registered_figure(fig, phase=5, block="A", index=2, slug=f"pdp_limiares_{args.model}",
            interpretation="Curvas PDP revelam limiares nao-lineares do acoplamento de Bjerknes: inflexoes indicam gatilhos de mudanca de fase.",
            metadata=f"Fonte: {args.model.upper()} Fase 5 | PDP sklearn",
            figures_dir=FIGS, title="5A - Limiares fisicos por dependencia parcial")
        print("[figura] Fig_5A2_pdp_limiares_%s.png" % args.model)
    except Exception as exc:  # PDP e best-effort
        print(f"[aviso] PDP nao gerado: {exc}")

    # SHAP (opcional): summary global se a biblioteca estiver instalada.
    try:
        import shap  # noqa
        values, sample = p5.shap_summary_values(result.model, X)
        print("[5] SHAP summary calculado (biblioteca disponivel).")
    except ImportError:
        print("[5] SHAP nao instalado (opcional): 'pip install shap' para summary/force/waterfall.")
    print("[5] concluido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
