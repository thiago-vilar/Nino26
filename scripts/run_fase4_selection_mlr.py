#!/usr/bin/env python3
"""Fase 4.0 (reducao de variaveis) + Fase 4B (MLR multivariada por fase).

4.0  Reduz o vetor de estado do Pacifico sem contar varios proxies da mesma
     dimensao fisica como evidencia independente: anomalia harmonica (fit no
     treino) -> filtro de redundancia por |r| -> PCA 90% -> estabilidade em
     janelas temporais expansivas. Gera o CONTRATO de variaveis selecionadas
     (phase40_variaveis_selecionadas.csv) consumido pela Fase 4C.

4B   Regressao linear multivariada (MLR) padronizada por fase do ciclo, com
     erros-padrao HAC limitados ao evento, VIF e validacao walk-forward por
     evento inteiro (sem vazamento). Alvo = ONI local continuo.

Saidas: tabelas em data/processed/parquet/statistics/ e figuras Fig_40*/Fig_4B*.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.maps.figure_registry import save_registered_figure  # noqa: E402
from nino_brasil.stats.phase4_selection import (  # noqa: E402
    temporal_pca_stability,
    walk_forward_phase_mlr,
)

FEAT = ROOT / "data/processed/parquet/features"
STATS = ROOT / "data/processed/parquet/statistics"
FIGS = ROOT / "data/processed/figures/fase4"
PHASE_ORDER = ("genese", "crescimento", "pico", "decaimento")
# Metadados de fonte, nao variaveis fisicas.
NON_PHYSICAL = {"ocean_source_code"}


def load_pacific() -> pd.DataFrame:
    w = pd.read_csv(FEAT / "nino34_master_weekly.csv", parse_dates=["week_ending_sunday"])
    w = w.set_index("week_ending_sunday").sort_index()
    cols = [c for c in w.columns if c not in NON_PHYSICAL]
    return w[cols].astype(float)


def weekly_oni(index: pd.DatetimeIndex) -> pd.Series:
    monthly = pd.read_csv(FEAT / "nino34_monthly_oisst.csv", parse_dates=["time"])
    monthly = monthly.set_index("time")["oni_local_c"].astype(float)
    key = pd.PeriodIndex(index, freq="M")
    lookup = monthly.copy()
    lookup.index = pd.PeriodIndex(lookup.index, freq="M")
    return pd.Series(key.map(lookup), index=index, name="oni_local_c")


def run_selection() -> pd.DataFrame:
    X = load_pacific()
    result = temporal_pca_stability(X, n_folds=5, min_train_years=8)
    selected = result.selected_variables.rename(
        columns={"selecionada": "selecionada"}
    )
    selected.to_csv(STATS / "phase40_variaveis_selecionadas.csv", index=False)
    result.reference_fit.audit.to_csv(STATS / "phase40_reducao_variaveis.csv", index=False)
    result.reference_fit.pca_variance.to_csv(STATS / "phase40_pca_variancia.csv", index=False)
    result.fold_metrics.to_csv(STATS / "phase40_estabilidade_folds.csv", index=False)
    n_sel = int(selected["selecionada"].sum())
    print(f"[4.0] variaveis de entrada={X.shape[1]} -> representantes estaveis selecionados={n_sel}")

    # Figura: variancia explicada acumulada da PCA + variaveis selecionadas.
    var = result.reference_fit.pca_variance
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    axes[0].bar(var["componente"], var["variancia_explicada"], color="#2f80c1")
    axes[0].plot(var["componente"], var["variancia_acumulada"], "o-", color="#d1495b")
    axes[0].axhline(0.90, ls="--", color="#6b7280")
    axes[0].set_title("PCA: variancia explicada e acumulada", fontsize=13)
    axes[0].set_ylabel("fracao da variancia"); axes[0].tick_params(axis="x", rotation=45, labelsize=8)
    sel = selected[selected["selecionada"]].sort_values("estabilidade", ascending=True)
    axes[1].barh(sel["variavel"], sel["estabilidade"], color="#059669")
    axes[1].set_title("Representantes estaveis (fracao de folds)", fontsize=13)
    axes[1].set_xlabel("estabilidade temporal"); axes[1].set_xlim(0, 1.02)
    interp = (
        f"De {X.shape[1]} variaveis do Pacifico, o filtro de redundancia + PCA(90%) + "
        f"estabilidade temporal retem {n_sel} representantes nao redundantes; "
        f"as demais sao proxies colineares da mesma dimensao fisica."
    )
    save_registered_figure(
        fig, phase=4, block="0", index=1, slug="selecao_variaveis_pca",
        interpretation=interp,
        metadata="Fonte: master semanal NINO26 | anomalia harmonica no treino + filtro |r| + PCA + estabilidade",
        figures_dir=FIGS, title="4.0 - Reducao dimensional do vetor do Pacifico",
    )
    print("[figura] Fig_401_selecao_variaveis_pca.png")
    return selected


def run_mlr() -> None:
    X = load_pacific()
    phase_table = pd.read_csv(
        STATS / "phase4A_fases_semanais.csv", parse_dates=["week_ending_sunday"]
    ).set_index("week_ending_sunday")
    phase_table = phase_table.reindex(X.index).fillna(
        {"fase": "neutro", "tipo": "neutro", "event_id": ""}
    )
    oni = weekly_oni(X.index)
    result = walk_forward_phase_mlr(
        X, oni, phase_table, phase_order=PHASE_ORDER
    )
    result.coefficients.to_csv(STATS / "phase4B_mlr_coeficientes.csv", index=False)
    result.metrics.to_csv(STATS / "phase4B_mlr_metricas.csv", index=False)
    result.diagnostics.to_csv(STATS / "phase4B_mlr_diagnosticos.csv", index=False)
    result.selection.to_csv(STATS / "phase4B_mlr_selecao.csv", index=False)
    print(f"[4B] MLR walk-forward: {len(result.metrics)} estratos tipo x fase; "
          f"{len(result.coefficients)} coeficientes de referencia")

    if not result.coefficients.empty:
        coef = result.coefficients[result.coefficients["termo"] != "intercepto"].copy()
        el = coef[coef["tipo"] == "el_nino"]
        if not el.empty:
            fig, ax = plt.subplots(figsize=(15, 8))
            for phase, sub in el.groupby("fase"):
                ax.errorbar(
                    sub["termo"], sub["beta_padronizado"],
                    yerr=(sub["beta_padronizado"] - sub["ic95_hac_inferior"]).abs(),
                    fmt="o", capsize=3, label=phase,
                )
            ax.axhline(0, color="#6b7280", lw=1)
            ax.set_title("4B - MLR El Nino: coeficientes padronizados por fase (IC95 HAC)", fontsize=13)
            ax.set_ylabel("beta padronizado (por desvio-padrao de ONI)")
            ax.tick_params(axis="x", rotation=60, labelsize=8); ax.legend(title="fase")
            interp = (
                "Coeficientes padronizados da MLR por fase do El Nino explicam o ONI local a partir "
                "das variaveis selecionadas; barras cruzando zero nao sao distinguiveis de nulo (HAC por evento)."
            )
            save_registered_figure(
                fig, phase=4, block="B", index=1, slug="mlr_coeficientes_el_nino",
                interpretation=interp,
                metadata="Fonte: master semanal + ONI local | MLR padronizada; IC95 Newey-West por evento",
                figures_dir=FIGS, title="4B - Regressao linear multivariada por fase",
            )
            print("[figura] Fig_4B1_mlr_coeficientes_el_nino.png")


def main(argv: list[str]) -> int:
    warnings.filterwarnings("ignore")
    FIGS.mkdir(parents=True, exist_ok=True)
    steps = [a for a in argv if a in {"selection", "mlr"}] or ["selection", "mlr"]
    if "selection" in steps:
        run_selection()
    if "mlr" in steps:
        run_mlr()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
