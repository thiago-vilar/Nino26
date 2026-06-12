# Projeto de previsão climática El NINO e seus impactos no Brasil

**Programa de Pós Graduação em Oceanografia UFPE**  
**Área:** Oceanografia Física  
**Responsável:** Thiago Vilar  

## Plano Diretor do Projeto NINO-BRASIL

**Tema:** quantificação dos pesos oceanográficos e atmosféricos associados ao aquecimento do Pacífico e seus impactos na precipitação do Brasil.  
**Período:** 1981-01-01 até o último dado disponível por fonte.  
**Arquitetura:** Python.  

## 1. Pergunta de pesquisa

Qual é o impacto, em termos de peso relativo, das variáveis oceanográficas e atmosféricas associadas ao aquecimento do Pacífico sobre as anomalias diárias de precipitação no Brasil, considerando sua distribuição espacial e temporal entre 1981 e o último dado disponível por fonte?

O projeto deve responder quais variáveis, profundidades, níveis atmosféricos, regiões do Pacífico e lags temporais têm maior poder explicativo sobre cada pixel do Brasil.

## 2. Escopo

```text
Pacífico: 35S a 30N / 120E a 70W
Brasil: território nacional em grade pixel-a-pixel
Frequência: diária sempre que disponível
Período: 1981-01-01 até o último dado disponível por fonte
Horizontes: previsão semanal de 1 a 24 semanas, equivalentes a 7-168 dias
```

As análises regionais devem incluir Brasil inteiro, Nordeste, Semiárido, Sul, Norte, Centro-Oeste e Sudeste, sem substituir a análise principal pixel-a-pixel.

Hipótese física prioritária: El Nino alto tende a aumentar o risco de seca no Nordeste/Semiárido e de chuva acima do normal no Sul do Brasil. O projeto deve quantificar esse sinal por pixel, região, estação e horizonte semanal.

## 2.1 Fases de execução

Fase 1 consolida a fundação local do projeto: estrutura Git, documentação metodológica, catálogo, scripts de ingestão, IBGE, CHIRPS 0.25° inicial, OISST diário e base reprodutível para os demais downloads.

Fase 2 calibra e padroniza os dados na grade comum de 0.25°: controle de qualidade, anomalias, climatologia sem vazamento, acumulados, percentis P10/P25/P75/P90, horizontes semanais, ERA5, ORAS5, CTD/WOD e Zarrs regridados.

Fase 3 executa o diagnostico fisico do sinal Nino 3.4: alinhamento de anomalias, volume/grau de agua quente anomala, D20/termoclina, slope longitudinal/vertical, duracao do sinal correlacionado e comparacao com os picos de El Nino mais impactantes da serie historica.

Fase 4 executa pre-analises estatisticas experimentais: regressao multipla, PCA/EOF, KNN, correlacoes defasadas e triagem de variaveis Nino 3.4 individuais e combinadas para seca no Nordeste e chuva no Sul.

Fase 5 executa machine learning classico e XAI: Ridge, Random Forest, XGBoost/LightGBM, cabecas de regressao e classificacao, ablations A/B/C/D/E/F, walk-forward, permutation importance, SHAP, pesos por grupo e mapas analiticos.

Fase 6 executa Redes Neurais + XAI e experimentos de memoria: CNN, ConvLSTM, U-Net, Transformer espaco-temporal, saliency maps, occlusion maps, attention maps, comparacao contra fases anteriores e teste opcional de Memory Caching (arXiv:2602.24281).

Fase 7 publica e operacionaliza o produto: GitHub Pages em `docs/`, comparacao previsao-observado, drift, recalibracao, mapas de confianca, relatorio automatico, rotina recorrente e experimentos CHIRPS 0.05°.

## 3. Etapa 1 - Download, armazenamento, tratamento e disponibilização

### 3.1 Download e armazenamento local

| Dado | Fonte | Pasta bruta | Justificativa |
|---|---|---|---|
| Oceano reanalisado | ORAS/ORAS5 | `data/raw/oras/` | Campo contínuo de temperatura e salinidade por profundidade para representar a memória térmica do oceano. |
| Oceano observado | CTD NOAA/WOD | `data/raw/ctd_noaa/wod/` e `data/processed/zarr/ctd_noaa/wod/` | Perfis in situ com QC WOD e TEOS-10 para validar temperatura, salinidade, densidade, termoclina, haloclina, picnoclina e conteúdo de calor. |
| Validacao in situ | TAO/TRITON/Argo | `data/raw/tao_triton/`, `data/raw/argo/` e `data/processed/zarr/validation/` | Camada auxiliar para validar ORAS5/CTD-WOD no Nino 3.4; nao substitui OISST/ORAS5 e nao decide antecipadamente se subsuperficie melhora o modelo. |
| Atmosfera | ERA5 | `data/raw/era5/` | Variáveis atmosféricas necessárias para representar o transporte do sinal do Pacífico em direção ao Brasil. |
| Precipitação | CHIRPS 0.25° diário nas Fases 1 a 7; CHIRPS 0.05° reservado para experimento futuro de alta resolução | `data/raw/chirps/p25/` e `data/raw/chirps/p05/` | Campo observado de chuva para calcular anomalias, seca e chuva acima do normal. |
| Limites territoriais | IBGE | `data/raw/ibge/` | Máscara oficial do Brasil e limites para mapas coropléticos por UF, região e recortes territoriais. |

### 3.2 Tratamento mínimo

1. Validar arquivos, datas, unidades e cobertura temporal.
2. Padronizar nomes de variáveis e coordenadas.
3. Corrigir longitude, calendário e grade.
4. Recortar Pacífico e Brasil.
5. Calcular climatologia diária.
6. Calcular anomalias diárias.
7. Criar acumulados de precipitação.
8. Criar gradientes de precipitação por percentis locais P10/P25/P75/P90.
9. Gerar datasets com horizontes semanais de 1 a 24 semanas.
10. Registrar tudo em `data/catalog/datasets.yaml`.
11. Salvar perfis CTD processados em `.zarr` anual depois de QC, TEOS-10 e calculo de estratificacao.
12. Estimar climatologia, desvio padrão e percentis somente dentro do bloco de treino de cada fold walk-forward.
13. Regridar Pacífico, Brasil e ponte atmosférica para a grade comum declarada em `configs/project.yaml` antes da matriz de modelagem.

## 4. Etapa 2 - Disponibilização dos dados

### 4.1 Aquecimento do Pacífico: satélite, ORAS e CTD

Variáveis principais:

- `SST`: temperatura da superfície do mar.
- `SSTA`: anomalia da temperatura da superfície do mar.
- `SSHA`: anomalia da altura da superfície do mar.
- `T(z)`: temperatura por profundidade.
- `S(z)`: salinidade por profundidade.
- `D20`: profundidade da isoterma de 20 graus Celsius.
- `MLD`: profundidade da camada de mistura.
- `OHC 0-300 m`: conteúdo de calor oceânico até 300 m.
- `OHC 0-700 m`: conteúdo de calor oceânico até 700 m.
- `u/v oceânico`: correntes zonal e meridional.
- `clorofila-a`: variável auxiliar superficial.

### 4.2 Atmosfera Pacífico -> Brasil

Variáveis principais:

- `SLP`: pressão ao nível do mar.
- `tau_x/tau_y`: tensão de vento zonal e meridional.
- `u10/v10`: vento zonal e meridional a 10 m.
- `u850/v850/q850`: vento e umidade em baixos níveis.
- `u500/v500/q500/z500/omega500`: circulação, umidade, geopotencial e movimento vertical em níveis médios.
- `u200/v200/z200/div200`: circulação, geopotencial e divergência em altos níveis.
- `OLR`: radiação de onda longa emitida.
- `TCWV`: vapor d'água total integrado na coluna atmosférica.

### 4.3 Precipitação Brasil

Variáveis principais:

- precipitação diária.
- anomalia diária.
- acumulados de 3, 5, 7, 15 e 30 dias.
- seco extremo abaixo de `P10`.
- abaixo do quartil inferior `P25`.
- acima do quartil superior `P75`.
- chuva extrema acima de `P90`.
- faixa interquartil `P25-P75` apenas como referência estatística, não como classe "normal".

### 4.4 Mapa do Brasil

O dado do IBGE será usado para:

- recortar a área continental brasileira.
- criar máscara territorial.
- agregar resultados por UF, região, bacia, bioma e Semiárido.
- gerar mapas coropléticos no produto web.

## 5. Etapa 3 - Diagnostico fisico, estatistica, modelagem, mapas e XAI

### 5.1 Formulação

```text
X[t] = variáveis oceanográficas + variáveis atmosféricas
Y[t + lag] = anomalia de precipitação no Brasil
```

Saídas:

- anomalia prevista de precipitação.
- probabilidade de seca.
- probabilidade de chuva acima do normal.
- peso relativo das variáveis oceanográficas.
- peso relativo das variáveis atmosféricas.

### 5.2 Diagnostico fisico e modelos

Fase 3:

- alinhamento de anomalias entre Nino 3.4 e chuva no Brasil.
- calculo de volume/grau de agua quente anomala no Nino 3.4.
- D20, termoclina, gradiente vertical e profundidade do maior gradiente.
- slope longitudinal/vertical e duracao do sinal.
- comparacao de slope, duracao e intensidade com picos historicos de El Nino.

Fase 4:

- regressao multipla.
- PCA/EOF.
- KNN para analogos historicos.
- correlacao defasada por lag semanal.
- triagem de variaveis individuais e combinadas para Nordeste e Sul.

Baselines:

- climatologia.
- persistência.
- correlação defasada.
- regressão regularizada.
- Random Forest.
- XGBoost.

Modelos da Fase 6, Redes Neurais + XAI:

- CNN.
- ConvLSTM.
- U-Net.
- Transformer espaço-temporal.

### 5.3 Dimensionamento de pesos

Experimentos mínimos:

```text
Modelo A: apenas variáveis oceanográficas
Modelo B: apenas variáveis atmosféricas
Modelo C: variáveis oceanográficas + atmosféricas
Modelo D: sem subsuperfície oceânica
Modelo E: sem altos níveis atmosféricos
Modelo F: sem umidade atmosférica
```

Técnicas:

- importância por permutação.
- SHAP.
- ablation study.
- mapas de atenção.
- diferença de skill entre modelos.

Artefatos mínimos da Fase 3:

- `nino34_physical_signal.zarr`.
- `nino34_thermocline_diagnostics.zarr`.
- `nino34_peak_signal_comparison.zarr`.
- `nino34_signal_slope_duration.zarr`.

Artefatos mínimos da Fase 4:

- `phase4_variable_screening.zarr`.
- `phase4_multiple_regression.zarr`.
- `phase4_pca_modes.zarr`.
- `phase4_knn_similarity.zarr`.

Artefatos mínimos da Fase 5:

- `walk_forward_metrics.zarr`.
- `walk_forward_predictions.zarr`.
- `walk_forward_importances.zarr`.
- `walk_forward_group_weights.zarr`.
- `distribution_diagnostics.zarr` quando houver diagnóstico de cauda.

Artefatos mínimos da Fase 6:

- `neural_training_runs.zarr`.
- `neural_predictions.zarr`.
- `neural_xai_maps.zarr`.
- `neural_skill_comparison.zarr`.

### 5.4 Validação

Regras:

- não usar split aleatório.
- usar blocos temporais.
- aplicar walk-forward validation.
- avaliar por lag, região, estação do ano e fonte de precipitação.

Métricas:

- correlação.
- MAE.
- RMSE.
- bias.
- Brier Score.
- ROC-AUC.
- precision.
- recall.
- hit rate.
- false alarm rate.
- F1-score.

### 5.5 Mapas

Mapas obrigatórios:

- anomalia prevista de precipitação.
- probabilidade de seca.
- probabilidade de chuva acima do normal.
- lag dominante.
- peso oceanográfico.
- peso atmosférico.
- variável dominante.
- erro histórico.
- confiança.

## 6. Fase 6 - Redes neurais, XAI e teste com Memory Caching

Avaliação da técnica de test-time memorization publicada pelo Google Research em "Memory Caching: RNNs with Growing Memory" (Behrouz et al., arXiv:2602.24281), como experimento interno da Fase 6, depois das redes neurais basicas e antes da operação.

Motivação: RNNs comprimem o passado em memória de tamanho fixo (custo O(L)), enquanto Transformers mantêm memória crescente (custo O(L²)). Memory Caching guarda checkpoints do estado de memória por segmento da sequência e os consulta na recuperação, interpolando entre os dois regimes com custo O(NL) — alinhado ao objetivo do projeto de experimentar diferentes modelos buscando o melhor equilíbrio entre skill e eficiência.

Escopo experimental:

1. Segmentar as sequências diárias (1981-presente) em blocos e cachear os estados de memória por segmento.
2. Implementar as quatro variantes de agregação do artigo: Residual Memory, Gated Residual Memory (gating dependente do contexto), Memory Soup (interpolação de parâmetros das memórias) e Sparse Selective Caching (roteador top-k estilo Mixture-of-Experts).
3. Aplicar sobre backbones recorrentes (linear attention e memórias profundas) para os mesmos alvos das Fases 5 e 6: anomalia de precipitação e eventos P10/P25/P75/P90.
4. Manter o protocolo metodológico inegociável: frequência diária, walk-forward em blocos temporais, climatologia/percentis apenas no treino de cada fold, grade comum 0.25°.
5. Medir o trade-off skill × custo: métricas das Fases 5/6 mais throughput de treino, memória de pico e custo de inferência por horizonte (1-24 semanas), variando número de segmentos N entre os regimes O(L) e O(L²).
6. Comparar contra os campeões das Fases 5 e 6 e registrar a recomendação de modelo operacional.

Artefatos mínimos da Fase 6:

- `memory_caching_runs.zarr`.
- `memory_caching_skill_comparison.zarr`.
- `memory_caching_efficiency.zarr`.

## 7. Fase 7 - Operação e deploy no GitHub Pages

O deploy será feito em `docs/`, usando GitHub Pages.

```text
pipeline Python
|
gera dados processados
|
gera mapas PNG/HTML
|
salva em docs/
|
GitHub Pages publica
```

## 8. Entregáveis

1. Base local diária 1981-latest_available.
2. Catálogo de fontes.
3. Cubos processados.
4. Datasets com lags.
5. Modelos baseline.
6. Modelos ML validados.
7. Mapas pixel-a-pixel.
8. Mapas coropléticos.
9. Relatório de métricas.
10. Relatório XAI.
11. Relatório do experimento Memory Caching (skill × eficiência).
12. Rotina de correção recorrente.
13. Deploy no GitHub Pages.
