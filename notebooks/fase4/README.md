# Fase 4 - Notebooks

Esta pasta e o local oficial para executar a Fase 4 no VS Code.

## Notebook ativo

| Ordem | Notebook | O que faz |
|---|---|---|
| 0 | `0_fase4_sanidade_disponibilidade.ipynb` | **Sanidade**: reune e trata todas as variaveis e plota tabelas/graficos (cobertura temporal, outliers, series, distribuicoes, transicoes de fonte oceanica, ERA5, CHIRPS, OISST, quadro-resumo). Responde se ha buracos de disponibilidade e outliers. |
| 0 | `0_fase4_sanidade_disponibilidade.ipynb` | **Pre-flight**: disponibilidade, cobertura temporal e outliers. |
| I | `A_fase4_fontes_variaveis_series.ipynb` | **Pre-flight**: inventario de variaveis processadas. Nao conta como 4A cientifico. |
| 4A | `4A_fase4_regionalizacao_chuva.ipynb` | Regionalizacao da chuva CHIRPS com EOF/REOF e clusters. |
| 4B | `4B_fase4_correlacao_regressao_defasada.ipynb` | Correlacao/regressao defasada pixel-a-pixel e por cluster, com FDR de campo. |
| 4C | `4C_fase4_modos_acoplados.ipynb` | Modos acoplados EOF/MCA/SVD/CCA para SST(Pacifico+Atlantico) x chuva. |
| 4D | `4D_fase4_atribuicao_composicoes.ipynb` | Atribuicao Pacifico vs Atlantico, composicoes, estabilidade e gate para ML. |

Modulo compartilhado: `scripts/fase4_features.py` (montagem da matriz ampla,
anomalias, lags, blocos fisicos, CV temporal com embargo, z-score source-aware).
Protocolo resumido: `docs/FASE4_PLANO.md`.

Os notebooks antigos de status, auditoria e contratos foram removidos para nao
parecerem resultados cientificos. As proximas analises devem ser criadas aqui
apenas quando tiverem calculos reais e saidas numericas verificaveis.

Kernel no VS Code:

```text
Python 3 (.venv NINO26)
```

Para treino neural/GPU nas proximas fases, use VS Code Remote WSL e selecione:

```text
Python 3 (.venv-wsl NINO26 GPU)
```

Diagnostico completo: `docs/AMBIENTE_KERNEL_GPU.md`.

## Saidas

O notebook A grava:

```text
data\processed\parquet\statistics\phase4_all_processed_variables.csv
data\processed\parquet\statistics\phase4_all_processed_variables_detail.csv
```

O notebook B grava (apos a 1a execucao, que monta a matriz ampla lendo ERA5):

```text
data\processed\parquet\modeling\phase4_feature_matrix_daily.parquet
data\processed\parquet\statistics\phase4B_best_lag_per_feature.csv
data\processed\parquet\statistics\phase4B_feature_importance.csv
data\processed\parquet\statistics\phase4B_skill_incremental.csv
data\processed\parquet\statistics\phase4B_collinearity_blocks.csv
```

O notebook C grava:

```text
data\processed\zarr\statistics\phase4C_rain_nino_corr_maps.nc
data\processed\parquet\statistics\phase4C_index_rain_ranking.csv
data\processed\figures\phase4C_ssta_rain_corr.png
data\processed\figures\phase4C_ssta_rain_sig.png
```

O notebook 4D deve gravar, quando atualizado para a nova especificacao:

```text
data\processed\parquet\statistics\phase4d_atribuicao_pacifico_atlantico.csv
data\processed\zarr\statistics\phase4d_enso_composites.zarr
data\processed\parquet\statistics\phase4d_stability_subperiods.csv
data\processed\parquet\statistics\phase4d_predictor_gate_for_ml.csv
data\processed\parquet\statistics\phase4d_questions_answers.md
```
