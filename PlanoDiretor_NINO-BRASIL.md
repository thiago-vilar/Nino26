# Projeto de previsão climática El NINO e seus impactos no Brasil

**Programa de Pós Graduação em Oceanografia UFPE**  
**Área:** Oceanografia Física  
**Responsável:** Thiago Vilar  

## Plano Diretor do Projeto NINO-BRASIL

### Relação entre aquecimento do Pacífico e anomalias de precipitação no Brasil

**Período de estudo:** 1980 até a data presente  
**Fontes principais:** ERA5, ORAS/ORAS5, CPC/NOAA  
**Arquitetura:** Python  

---

## 1. Pergunta de pesquisa

Como o aquecimento diário do Pacífico, observado por satélite, CTD e reanálises oceânicas, se relaciona com anomalias de precipitação no Brasil entre 1980 e a data presente?

Perguntas derivadas:

- Quais sinais do Pacífico antecedem seca ou chuva abaixo do normal no Nordeste?
- Quais sinais do Pacífico antecedem chuva acima do normal ou chuva extrema no Sul?
- Como a relação Pacífico -> Brasil se distribui espacialmente em pixels?
- Quais variáveis, profundidades e defasagens apresentam maior poder explicativo?

---

## 2. Escopo físico e temporal

### 2.1 Período

```text
1980 até presente data
```

Cada fonte deve registrar a cobertura temporal real. Quando uma variável não cobrir todo o período, o pipeline deve usar apenas a interseção temporal válida ou marcar a lacuna explicitamente.

### 2.2 Domínio do Pacífico

```text
35S a 30N
120E a 70W
frequência diária sempre que disponível
```

### 2.3 Domínio de impacto

```text
Brasil inteiro
grade pixel-a-pixel
agregação por estado, região, bacia, semiárido e bioma
```

---

## 3. Etapa 1 - Download, armazenamento local, tratamento e disponibilização de dados

### 3.1 Objetivo

Criar uma base local, rastreável e reprodutível com dados do Pacífico, dados atmosféricos do corredor Pacífico -> Brasil e precipitação no Brasil.

### 3.2 Estrutura local

```text
NINO26/
  data/
    raw/
      era5/
      oras/
      cpc_noaa/
      auxiliary/
    interim/
      pacific_warming/
      atmosphere_bridge/
      brazil_precipitation/
    processed/
      zarr/
      parquet/
      geotiff/
    catalog/
      datasets.yaml
  src/
    data/
      download_era5.py
      download_oras.py
      download_cpc_noaa.py
      standardize.py
      quality_control.py
      anomalies.py
      build_lagged_dataset.py
    features/
      ocean_heat.py
      thermocline.py
      atmospheric_bridge.py
      precipitation_events.py
    models/
      baselines.py
      train.py
      validate.py
      explain.py
    maps/
      plot_pixel_maps.py
      plot_choropleths.py
    web/
      build_site.py
```

### 3.3 Formatos

- **NetCDF:** arquivos originais em grade.
- **Zarr:** cubos processados para leitura rápida com `xarray`.
- **Parquet:** tabelas de treino, amostras, métricas e resultados.
- **GeoTIFF:** rasters georreferenciados para mapas.
- **YAML:** catálogo de dados com fonte, variável, unidade, período e caminho local.

### 3.4 Tratamento mínimo

1. Baixar dados brutos.
2. Validar arquivos, datas e variáveis.
3. Padronizar nomes e unidades.
4. Recortar domínio do Pacífico.
5. Recortar domínio do Brasil.
6. Corrigir longitude, calendário e grade.
7. Registrar cobertura temporal real por variável.
8. Calcular climatologia diária.
9. Calcular anomalias diárias.
10. Criar acumulados de precipitação.
11. Criar eventos por percentis locais.
12. Criar datasets com defasagens temporais.

### 3.5 Defasagens iniciais

```text
0, 7, 15, 30, 45, 60, 90, 120 e 180 dias
```

Pergunta operacional:

```text
O estado do Pacífico no dia t ajuda a explicar ou prever a precipitação no Brasil em t + lag?
```

---

## 4. Etapa 2 - Disponibilização de dados

Os dados devem ser organizados em três blocos.

---

### 4.1 Bloco A - Aquecimento do Pacífico: satélite + CTD + ORAS

Objetivo: representar o aquecimento superficial, subsuperficial e a propagação do calor no Pacífico.

#### 4.1.1 Variáveis de satélite e CPC/NOAA

- **SST:** temperatura da superfície do mar.
- **SSTA:** anomalia da temperatura da superfície do mar.
- **SSHA:** anomalia da altura da superfície do mar, quando disponível.
- **Temperatura de brilho:** se canais radiométricos forem usados diretamente.
- **Clorofila-a:** variável auxiliar para ressurgência, mistura e resposta biológica no Pacífico Leste.

#### 4.1.2 Variáveis CTD e ORAS/ORAS5

Variáveis observadas por CTD:

- **T(z):** temperatura em função da profundidade.
- **S(z):** salinidade em função da profundidade.

Variáveis calculadas e padronizadas a partir de `T(z)` e `S(z)`:

- densidade potencial;
- **D20:** profundidade da isoterma de 20 graus Celsius;
- profundidade da termoclina;
- profundidade da camada de mistura;
- temperatura média 0-300 m;
- temperatura média 0-700 m;
- conteúdo de calor oceânico 0-300 m;
- conteúdo de calor oceânico 0-700 m;
- gradiente vertical de temperatura.

#### 4.1.3 Dinâmica oceânica

Adicionar a partir de ORAS/ORAS5 quando disponível:

- corrente zonal oceânica `u`;
- corrente meridional oceânica `v`;
- velocidade vertical ou proxy de ressurgência;
- ondas de Kelvin oceânicas;
- ondas de Rossby oceânicas;
- rajadas de vento de oeste;
- gradiente zonal de temperatura;
- gradiente meridional de temperatura;
- propagação longitudinal via diagrama Hovmoller.

#### 4.1.4 Fluxos de calor oceano-atmosfera

Adicionar a partir de ERA5 quando disponível:

- fluxo líquido de calor na superfície;
- fluxo de calor latente;
- fluxo de calor sensível;
- radiação de onda curta;
- radiação de onda longa.

---

### 4.2 Bloco B - Dados atmosféricos Pacífico -> Brasil

Objetivo: representar a ponte atmosférica entre o Pacífico aquecido e a resposta de chuva no Brasil.

#### 4.2.1 Núcleo mínimo

- **SLP:** pressão ao nível do mar.
- **tau_x:** tensão de vento zonal na superfície.
- **tau_y:** tensão de vento meridional na superfície.
- **u10:** vento zonal a 10 m.
- **v10:** vento meridional a 10 m.
- **OLR:** radiação de onda longa emitida.
- **TCWV:** vapor d'água total integrado na coluna atmosférica.

#### 4.2.2 Estrutura vertical da atmosfera

Baixos níveis:

- **u850:** vento zonal em 850 hPa.
- **v850:** vento meridional em 850 hPa.
- **q850:** umidade específica em 850 hPa.

Níveis médios:

- **u500:** vento zonal em 500 hPa.
- **v500:** vento meridional em 500 hPa.
- **q500:** umidade específica em 500 hPa.
- **z500:** geopotencial em 500 hPa.
- **omega500:** movimento vertical em 500 hPa.

Altos níveis:

- **u200:** vento zonal em 200 hPa.
- **v200:** vento meridional em 200 hPa.
- **z200:** geopotencial em 200 hPa.
- **div200:** divergência em 200 hPa.

Essas variáveis representam circulação de grande escala, transporte de umidade, subsidência/ascensão, convecção tropical e teleconexão entre Pacífico e Brasil.

---

### 4.3 Bloco C - Dados de precipitação Brasil

Objetivo: representar a resposta observada no território brasileiro.

Fontes:

- CPC/NOAA Daily Precipitation.
- CHIRPS diário, como fonte complementar.
- MERGE/INPE diário, como fonte nacional complementar.
- GPM IMERG diário, para extremos recentes.

Variáveis:

- precipitação diária;
- anomalia diária;
- acumulado de 3, 5, 7, 15 e 30 dias;
- evento seco: abaixo do P10 local;
- chuva alta: acima do P90 local;
- chuva extrema: acima do P95 ou P99 local.

---

## 5. Etapa 3 - Arquitetura Machine Learning, validação, mapas, XAI e correção recorrente

### 5.1 Arquitetura em Python

Bibliotecas principais:

- `xarray`, `numpy`, `pandas`, `zarr`, `netCDF4`;
- `geopandas`, `rasterio`, `rioxarray`;
- `scikit-learn`, `xgboost`, `lightgbm`;
- `torch`, `pytorch-lightning`;
- `matplotlib`, `plotly`, `folium`;
- `shap` para explicabilidade.

### 5.2 Problema de modelagem

Entrada:

```text
variáveis do Pacífico + variáveis atmosféricas em t
```

Saída:

```text
anomalia de precipitação no Brasil em t + lag
probabilidade de seca
probabilidade de chuva extrema
```

### 5.3 Baselines obrigatórios

- climatologia diária local;
- persistência;
- correlação defasada pixel-a-pixel;
- regressão regularizada;
- Random Forest;
- XGBoost;
- Pacífico embaralhado no tempo;
- lag invertido.

### 5.4 Modelos

Primeira fase:

- regressão regularizada;
- Random Forest;
- XGBoost;
- LightGBM.

Segunda fase:

- CNN;
- ConvLSTM;
- U-Net;
- Transformer espaço-temporal.

### 5.5 Validação

Regras:

- não usar split aleatório;
- usar treino, validação e teste por blocos temporais;
- usar walk-forward validation;
- avaliar por lag;
- avaliar por região;
- avaliar por estação do ano;
- avaliar por período El Nino, La Nina e neutro apenas como análise auxiliar.

Regiões:

- Brasil inteiro;
- Nordeste;
- Semiárido;
- Sul;
- Norte;
- Centro-Oeste;
- Sudeste.

### 5.6 Métricas

Regressão:

- correlação;
- MAE;
- RMSE;
- bias;
- erro por pixel;
- erro por região;
- erro por lag.

Eventos:

- Brier Score;
- ROC-AUC;
- precision;
- recall;
- hit rate;
- false alarm rate;
- F1-score.

Mapas:

- correlação espacial;
- área de acerto;
- área de falso alarme;
- FSS quando aplicável.

### 5.7 XAI

Perguntas:

- qual região do Pacífico foi mais importante?
- qual variável foi mais importante?
- qual profundidade CTD foi mais importante?
- qual lag foi mais importante?
- a resposta é diferente no Nordeste e no Sul?
- o sinal é fisicamente coerente?

Técnicas:

- mapas de correlação defasada;
- permutation importance;
- SHAP;
- occlusion maps;
- saliency maps;
- attention maps;
- ablation study.

### 5.8 Plotagem dos mapas

Mapas:

- anomalia prevista de precipitação;
- probabilidade de seca;
- probabilidade de chuva extrema;
- erro histórico;
- confiança;
- lag dominante;
- importância das variáveis;
- diferença Nordeste-Sul.

### 5.9 Correção recorrente automatizada

A cada nova rodada:

1. Comparar previsto vs observado.
2. Atualizar métricas por pixel, região e lag.
3. Detectar drift.
4. Recalibrar probabilidades quando necessário.
5. Atualizar mapas de confiança.
6. Gerar relatório automático.

---

## 6. Etapa 4 - Deploy dos mapas coropléticos na web com GitHub Pages

### 6.1 Objetivo

Publicar mapas e métricas em página estática no GitHub Pages.

### 6.2 Fluxo

```text
pipeline Python
|
gera NetCDF/Zarr/GeoTIFF/Parquet
|
gera mapas HTML e PNG
|
atualiza pasta docs/ ou site/
|
GitHub Pages publica
```

### 6.3 Mapas

Coropléticos:

- risco médio por estado;
- risco médio por região;
- risco médio por bacia;
- risco médio no semiárido;
- risco médio por bioma.

Pixel-a-pixel:

- anomalia prevista;
- seca;
- chuva extrema;
- confiança;
- erro histórico;
- importância do Pacífico.

### 6.4 Stack

- Python;
- GeoPandas;
- Xarray;
- Rasterio;
- Plotly;
- Folium;
- GitHub Actions;
- GitHub Pages.

---

## 7. Entregáveis

1. Base local 1980-presente.
2. Catálogo de fontes.
3. Cubos Zarr processados.
4. Datasets com lags.
5. Modelos baseline.
6. Modelos ML validados.
7. Mapas pixel-a-pixel.
8. Mapas coropléticos.
9. Relatório de métricas.
10. Relatório XAI.
11. Rotina de correção recorrente.
12. Deploy no GitHub Pages.

