# Fase 6 - Distribuicao no Brasil com Machine Learning (RF/XGBoost) + XAI

**Coluna B da matriz metodologica, linha ML.** Repete o estudo espaco-temporal da
Fase 4C com ML. Modulo: `src/nino_brasil/models/phase6_brazil_ml.py`; script:
`scripts/run_fase6_brazil_ml.py`.

## Desenho
- Alvo: anomalia padronizada de chuva agregada por **unidade oficial** (5 regioes
  IBGE 2024, 6 biomas IBGE 2025, recortes Caatinga e Mata Atlantica do Nordeste),
  via agregacao area-ponderada (`maps.spatial_support.aggregate_area_weighted_response`).
- Preditores: janela deslizante do Pacifico (lags 4-52 sem), reaproveitando
  `phase5_cycle_ml.build_lagged_features`.
- Condicoes: `todas` e as 4 fases de El Nino e La Nina.
- Modelos: RF/XGBoost por unidade x condicao; **validacao cronologica**
  (`TimeSeriesSplit`) e baseline de media expansiva.
- XAI: importancia por ganho (`top_importances_by_unit`) e SHAP opcional.

## Leitura
`skill_rmse_vs_baseline > 0` e `r` fora-da-amostra positivo indicam sinal aprendivel
do Pacifico sobre a chuva da unidade. Comparar por regiao e bioma (o Brasil e grande
e nao pode ser generalizado): Nordeste/Caatinga (seca em El Nino) vs Sul (chuva).

## Gate G3
RF/XGBoost so avancam se superarem a triagem estatistica da Fase 4 e os baselines.
