# Fase 4 - Plano curto (protocolo)

Pre-analise estatistica exploratoria usando **todas** as variaveis processadas,
mas com salvaguardas de inferencia: janelas reais por fonte, FDR, graus de
liberdade efetivos, colinearidade por bloco e transformacoes ajustadas apenas
no treino de cada fold.

## Arquivos

| Arquivo | Papel |
|---|---|
| `scripts/fase4_features.py` | Montagem da matriz ampla + utilidades (anomalia, lags, blocos, CV temporal, z-score source-aware). |
| `notebooks/fase4/0_fase4_sanidade_disponibilidade.ipynb` | Pre-flight: sanidade de disponibilidade e cobertura. |
| `notebooks/fase4/A_fase4_fontes_variaveis_series.ipynb` | Pre-flight: inventario de fontes/variaveis; nao conta como 4A cientifico. |
| `notebooks/fase4/4A_fase4_regionalizacao_chuva.ipynb` | Regionalizacao da chuva CHIRPS: clusters/regioes-alvo. |
| `notebooks/fase4/4B_fase4_correlacao_regressao_defasada.ipynb` | Correlacao/regressao defasada por pixel e cluster. |
| `notebooks/fase4/4C_fase4_modos_acoplados.ipynb` | Modos acoplados EOF/MCA/SVD/CCA. |
| `notebooks/fase4/4D_fase4_atribuicao_composicoes.ipynb` | Atribuicao Pacifico vs Atlantico, composicoes, estabilidade e gate para ML. |

## Tratamento de cada variavel (para "usar todas")

1. **Media diaria** - ERA5 (horario) agregado para diario; CHIRPS ja diario;
   features oceanicas Nino 3.4 ja reduzidas por media espacial na caixa.
2. **Calendario mestre** - reindex no diario 1981-latest; lacunas ficam NaN
   auditavel (sem `fillna` silencioso).
3. **Anomalia** - no eixo semanal canonico, climatologia harmonica com 2-3
   harmonicos anuais, sempre ajustada **so no treino de cada fold** para
   ranking, regressao, EOF/MCA/CCA e skill. A climatologia diaria por
   dia-do-ano fica para insumos diarios/descritivos. A base 1991-2020 fica
   apenas como referencia descritiva sem claim preditivo. Para fluxos e
   precipitacao usar **anomalia padronizada**. Alternativa trend-aware:
   `detrend_series`.
4. **Normalizacao** - z-score ajustado **so no treino de cada fold**; oceano
   **source-aware** (media/desvio por fonte: UFS=1, GLORYS=2, GLO12=3).
5. **Lags/derivadas** - medias e deltas em 7/30/90/180 d; lags 7..168 d.
6. **Janelas reais** - CHIRPS/OISST/ERA5 sustentam 1981-latest; GLORYS12 diario
   comeca em 1993; Argo/TAO entram como validacao com cobertura util mais forte
   pos-2000. Subsuperficie deve reportar sensibilidade 1993+ e 2000+.
7. **Controles de teleconexao** - incluir `ATL4` como controle atlantico
   prioritario para o Nordeste, junto de `ATL3`, `TNA`, `TSA` como covariaveis
   obrigatorias e `IOD/DMI` como recomendado; comparar Pacifico-only contra
   Pacifico+Atlantico para evitar variavel omitida.
8. **Validacao fisica** - TAO/TRITON, CTD WOD, ARGO conferem D20/termoclina das
   reanalises contra observacao in situ (camada de validacao, nao feature).
9. **Sanidade (visual)** - serie com eventos sombreados; ciclo climatologico;
   histograma/QQ da anomalia; Hovmoller/mapa num evento forte; reanalise x boia.

## Reducao de colinearidade (decisao: por bloco fisico)

OHC, temperaturas 50-700 m, D20, WWV e termoclina medem o mesmo bloco fisico de
recarga/subsuperficie. Blocos:
`sst`, `ocean_heat`, `thermocline_tilt`, `sea_level`, `salinity`, `wind`,
`pressure`, `convection`, `heat_flux`, `tropical_atlantic`, `iod`. Dentro de
cada bloco: cluster por correlacao (corte |r|>0.9) e mantem 1 representante,
ou PCA/ridge/elastic-net para sensibilidade. Importancia reportada **por bloco**
e, quando houver redundancia, nao interpretar coeficientes redundantes como
efeitos independentes.

## Teleconexao Brasil (objetivo da Fase 4)

- Alvo CHIRPS regridado, recortado ao Brasil, semanal, com anomalia
  padronizada por pixel e eventos locais P10/P25/P75/P90.
- Primeiro regionalizar a chuva (4A), depois testar forcantes por pixel/cluster
  (4B), extrair modos acoplados (4C) e fechar atribuicao Pacifico vs Atlantico
  (4D).
- Correlacao de Pearson/Spearman e regressao por pixel/cluster vs indices Nino
  e covariaveis Atlantico/IOD (`ATL4`, `ATL3`, `TNA`, `TSA`, `DMI`) em lags
  semanais.
- Mapa de melhor-lag + significancia de campo (p-valor por pixel + FDR, com
  `N_eff` por autocorrelacao).
- MLR/partial correlation deve comparar Pacifico-only contra Pacifico+Atlantico.
- Composicoes por fase/tipo de ENSO entram no 4D.

## Validacao temporal (obrigatoria nos dois)

Split temporal expansivo com **embargo >= maior lag (168 d)**; climatologia e
z-score ajustados **dentro** do treino de cada fold. Nunca k-fold aleatorio.

## Saidas

```
data/processed/parquet/modeling/phase4_feature_matrix_daily.parquet
data/processed/parquet/statistics/phase4a_clusters_brasil.csv
data/processed/zarr/statistics/phase4a_data_driven_regions.zarr
data/processed/zarr/statistics/phase4b_pixel_correlation_field.zarr
data/processed/zarr/statistics/phase4b_pixel_regression_slope.zarr
data/processed/zarr/statistics/phase4b_pixel_mlr_diagnostics.zarr
data/processed/zarr/statistics/phase4c_mca_coupled_modes.zarr
data/processed/parquet/statistics/phase4c_independent_variable_selection.csv
data/processed/parquet/statistics/phase4d_atribuicao_pacifico_atlantico.csv
data/processed/parquet/statistics/phase4d_predictor_gate_for_ml.csv
data/processed/parquet/statistics/phase4d_questions_answers.md
```
