# Fase 4 - Metodologia revisada: teleconexao El Nino -> Brasil

**Projeto NINO-BRASIL - Oceanografia Fisica UFPE - Thiago Vilar**

Este documento aplica o parecer de reorganizacao das Fases 3 e 4. A fronteira
fica assim:

- **Fase 3:** alvo = Nino 3.4. Diagnostica pico, memoria e precursores do ENOS.
- **Fase 4:** alvo = chuva no Brasil. Diagnostica a teleconexao e a atribuicao
  Pacifico vs. Atlantico para chuva, seca e extremos regionais.

Os indices Atlanticos aparecem nas duas fases com papeis diferentes. Na Fase 3,
ATL3/ATL4/TNA/TSA podem testar influencia sobre o ENOS. Na Fase 4, eles entram
como forcantes diretas e controles obrigatorios da chuva no Brasil, especialmente
para NEB/Amazonia. Isso nao e duplicacao: e a mesma serie em hipoteses distintas.

## 1. Politica temporal da Fase 4

O eixo padrao da Fase 4 e **semanal de 7 dias** (`week_ending_sunday`).

| Nivel | Uso | Fontes compativeis |
|---|---|---|
| Diario | insumo bruto, acumulacao do DHW, velocidade de fase Kelvin | OISST, CHIRPS, ERA5, GLORYS12 |
| Semanal | eixo canonico de precursores, correlacao defasada e teleconexao | derivado do diario |
| Mensal | excecao para serie nativamente mensal e sensibilidade | ORAS5, Argo em grade, WOD, NOAA PSL/ONI |

Consequencias:

- GLORYS12 diario e a fonte primaria para D20/OHC/WWV no eixo semanal.
- ORAS5 mensal fica como cross-check de sensibilidade, sem promocao artificial
  para diario/semanal.
- CHIRPS em pentada deve ser reamostrado para a semana canonica de 7 dias, com
  o pequeno efeito de borda documentado.
- Climatologia semanal deve usar 2-3 harmonicos anuais ajustados somente no
  treino, nao media crua de 52 semanas.
- Todo p-valor e IC usa `N_eff`; o N nominal semanal nao deve ser tratado como
  amostras independentes.

## 2. Estrutura oficial: 4A-4D

```
4A  Regionalizacao da chuva          -> define clusters/regioes-alvo do Brasil
4B  Correlacao/regressao defasada    -> testa forcantes por pixel, cluster, lag e estacao
4C  Modos acoplados                  -> extrai padroes SST(Pacifico+Atlantico) x chuva
4D  Atribuicao + composicoes         -> separa Pacifico vs Atlantico e fecha as respostas
```

Inventario, sanidade de disponibilidade e auditoria de variaveis sao pre-flight:
eles podem existir como notebooks de apoio, mas nao contam como 4A-4D.

## 3. Fase 4A - Regionalizacao da chuva

Pergunta: **onde o Brasil responde de forma homogenea?**

Metodos:

- usar CHIRPS no Brasil, em grade 0.25 grau e eixo semanal;
- construir anomalias e eventos locais (`P10`, `P25`, `P75`, `P90`) com
  climatologia ajustada apenas no treino;
- aplicar EOF/REOF e clusterizacao espacial dos pixels;
- comparar clusters estatisticos com regioes fisicas conhecidas, sem impor NEB
  e Sul antes do teste;
- gerar indice regional por media ponderada de area ou PC1.

Saidas:

```text
data/processed/parquet/statistics/phase4a_clusters_brasil.csv
data/processed/zarr/statistics/phase4a_data_driven_regions.zarr
docs/assets/maps/phase4a_regions_*.png
```

Resultado esperado: regioes-alvo para 4B/4D e Fase 5B, incluindo NEB,
Amazonia, Sul e outras regioes que surgirem dos dados.

## 4. Fase 4B - Correlacao/regressao defasada

Pergunta: **quais forcantes explicam cada cluster, onde e quando?**

Metodos:

- correlacao Pearson/Spearman pixel-a-pixel e por cluster;
- regressao linear com tamanho de efeito fisico, por exemplo mm por grau C;
- MLR pixel-a-pixel e por cluster comparando:
  - Pacifico-only;
  - Pacifico + Atlantico tropical;
  - superficie oceanica;
  - subsuperficie;
  - atmosfera;
  - oceano + atmosfera;
- lags semanais canonicos, com 0-78 semanas para varreduras equivalentes a
  0-18 meses;
- estacoes DJF/MAM/JJA/SON e janelas chuvosas regionais;
- `N_eff` de Bretherton para toda significancia;
- FDR Benjamini-Hochberg/Wilks sobre o conjunto completo de pixels, lags,
  estacoes e preditores;
- bootstrap de blocos quando houver composicoes ou incerteza espacial.

Controles obrigatorios:

```text
ATL4, ATL3, TNA, TSA, gradiente TNA-TSA, IOD/DMI recomendado
```

Saidas:

```text
data/processed/zarr/statistics/phase4b_pixel_correlation_field.zarr
data/processed/zarr/statistics/phase4b_pixel_regression_slope.zarr
data/processed/zarr/statistics/phase4b_pixel_mlr_coefficients.zarr
data/processed/zarr/statistics/phase4b_pixel_mlr_diagnostics.zarr
data/processed/parquet/statistics/phase4b_cluster_lag_regression.csv
```

Resultado esperado: mapas e tabelas por pixel/cluster com sinal, lag, estacao,
tamanho de efeito, IC, `p_FDR` e fracao de area significativa.

## 5. Fase 4C - Modos acoplados

Pergunta: **quais padroes conjuntos SST -> chuva dominam?**

Metodos:

- EOF dos campos de chuva para validar as regioes da 4A;
- EOF dos campos SST/SSHA/D20/OHC/vento no Pacifico e Atlantico tropical;
- MCA/SVD da covariancia cruzada SST(Pacifico+Atlantico) x chuva Brasil;
- CCA como sensibilidade;
- MCA defasada para lead semanal otimo;
- EOF/MCA/CCA ajustados apenas dentro do bloco de treino quando usados para
  ranking inferencial ou skill;
- selecao ciente de colinearidade com correlacao parcial, VIF, LASSO/elastic-net
  e stability selection.

Saidas:

```text
data/processed/zarr/statistics/phase4c_rainfall_eof_modes.zarr
data/processed/zarr/statistics/phase4c_pacific_atlantic_eof_modes.zarr
data/processed/zarr/statistics/phase4c_mca_coupled_modes.zarr
data/processed/parquet/statistics/phase4c_independent_variable_selection.csv
```

Resultado esperado: modos acoplados fisicamente interpretaveis, series de
expansao e variaveis com informacao independente.

## 6. Fase 4D - Atribuicao, composicoes e estabilidade

Pergunta: **quanto do sinal e Pacifico, quanto e Atlantico, e como isso varia por
tipo de evento e periodo?**

Metodos:

- regressao parcial Pacifico controlando Atlantico e Atlantico controlando
  Pacifico;
- composicoes de chuva por El Nino/La Nina/Neutro, intensidade, EP vs CP e
  estacao-alvo;
- repetir 4B/4C em 1993-2009 e 2010-presente para testar estabilidade temporal
  da teleconexao;
- registrar quando um sinal Pacifico-only desaparece ao incluir ATL4/ATL3/TNA/TSA;
- classificar variaveis para Fases 5-6 como `preditor`, `controle`,
  `diagnostico` ou `excluido`.

Saidas:

```text
data/processed/parquet/statistics/phase4d_atribuicao_pacifico_atlantico.csv
data/processed/zarr/statistics/phase4d_enso_composites.zarr
data/processed/parquet/statistics/phase4d_stability_subperiods.csv
data/processed/parquet/statistics/phase4d_predictor_gate_for_ml.csv
data/processed/parquet/statistics/phase4d_questions_answers.md
```

Resultado esperado: respostas finais da Fase 4 para Q2, com controle de variavel
omitida, e uma lista auditavel de candidatos para Fase 5B/6C.

## 7. Criterios de aceite

- Fase 4A define regioes por dados antes da modelagem regional.
- Fase 4B aplica `N_eff` e FDR em todo mapa.
- Fase 4C nao usa PCA de indices como substituto de EOF/MCA dos campos.
- Fase 4D testa atribuicao Pacifico vs Atlantico e estabilidade por subperiodo.
- CHIRPS/extremos Amazonicos aparecem com ressalva metodologica, pois CHIRPS pode
  subestimar extremos convectivos locais.
- Nenhuma figura e publicada sem saida numerica rastreavel.
