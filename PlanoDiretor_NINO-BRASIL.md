# Projeto de previsão climática El NINO e seus impactos no Brasil

**Programa de Pós Graduação em Oceanografia UFPE**  
**Área:** Oceanografia Física  
**Responsável:** Thiago Vilar  

## Plano Diretor do Projeto NINO-BRASIL

**Tema:** quantificação dos pesos oceanográficos e atmosféricos associados ao aquecimento do Pacífico e seus impactos na precipitação do Brasil.  
**Período:** 1980 até a data presente.  
**Arquitetura:** Python.  

## 1. Pergunta de pesquisa

Qual é o impacto, em termos de peso relativo, das variáveis oceanográficas e atmosféricas associadas ao aquecimento do Pacífico sobre as anomalias de precipitação no Brasil, considerando sua distribuição espacial e temporal entre 1980 e a data presente?

O projeto deve responder quais variáveis, profundidades, níveis atmosféricos, regiões do Pacífico e lags temporais têm maior poder explicativo sobre cada pixel do Brasil.

## 2. Escopo

```text
Pacífico: 35S a 30N / 120E a 70W
Brasil: território nacional em grade pixel-a-pixel
Frequência: diária sempre que disponível
Lags: 0, 7, 15, 30, 45, 60, 90, 120 e 180 dias
```

As análises regionais devem incluir Brasil inteiro, Nordeste, Semiárido, Sul, Norte, Centro-Oeste e Sudeste, sem substituir a análise principal pixel-a-pixel.

## 3. Etapa 1 - Download, armazenamento, tratamento e disponibilização

### 3.1 Download e armazenamento local

| Dado | Fonte | Pasta bruta | Justificativa |
|---|---|---|---|
| Oceano reanalisado | ORAS/ORAS5 | `data/raw/oras/` | Campo contínuo de temperatura e salinidade por profundidade para representar a memória térmica do oceano. |
| Oceano observado | CTD NOAA/WOD | `data/raw/ctd_noaa/` | Perfis in situ para validar e corrigir temperatura, salinidade, densidade, termoclina, camada de mistura e conteúdo de calor. |
| Atmosfera | ERA5 | `data/raw/era5/` | Variáveis atmosféricas necessárias para representar o transporte do sinal do Pacífico em direção ao Brasil. |
| Precipitação | CPC/NOAA | `data/raw/cpc_noaa/` | Campo observado de chuva para calcular anomalias, seca e chuva acima do normal. |
| Limites territoriais | IBGE | `data/raw/ibge/` | Máscara oficial do Brasil e limites para mapas coropléticos por UF, região e recortes territoriais. |

### 3.2 Tratamento mínimo

1. Validar arquivos, datas, unidades e cobertura temporal.
2. Padronizar nomes de variáveis e coordenadas.
3. Corrigir longitude, calendário e grade.
4. Recortar Pacífico e Brasil.
5. Calcular climatologia diária.
6. Calcular anomalias diárias.
7. Criar acumulados de precipitação.
8. Criar eventos secos e chuva acima do normal por percentis locais.
9. Gerar datasets com lags temporais.
10. Registrar tudo em `data/catalog/datasets.yaml`.

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
- evento seco abaixo de `P10`.
- chuva acima do normal acima de `P90`.

### 4.4 Mapa do Brasil

O dado do IBGE será usado para:

- recortar a área continental brasileira.
- criar máscara territorial.
- agregar resultados por UF, região, bacia, bioma e Semiárido.
- gerar mapas coropléticos no produto web.

## 5. Etapa 3 - Machine Learning, validação, mapas e XAI

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

### 5.2 Modelos

Baselines:

- climatologia.
- persistência.
- correlação defasada.
- regressão regularizada.
- Random Forest.
- XGBoost.

Modelos posteriores:

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

## 6. Etapa 4 - Deploy no GitHub Pages

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

## 7. Entregáveis

1. Base local 1980-presente.
2. Catálogo de fontes.
3. Cubos processados.
4. Datasets com lags.
5. Modelos baseline.
6. Modelos ML validados.
7. Mapas pixel-a-pixel.
8. Mapas coropléticos.
9. Relatório de métricas.
10. Relatório XAI.
11. Rotina de correção recorrente.
12. Deploy no GitHub Pages.
