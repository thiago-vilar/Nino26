# Projeto de previsão climática El NINO e seus impactos no Brasil

**Programa de Pós Graduação em Oceanografia UFPE**  
**Área:** Oceanografia Física  
**Responsável:** Thiago Vilar  

## NINO-BRASIL

O **NINO-BRASIL** é um projeto de pesquisa e desenvolvimento em Python para estudar e prever como o aquecimento do Pacífico associado ao El Niño se conecta, por meio do acoplamento oceano-atmosfera, às anomalias de precipitação no Brasil.

O objetivo central é produzir mapas de influência e previsão **pixel-a-pixel**, mostrando onde e com qual defasagem temporal os sinais do Pacífico se relacionam com seca, chuva acima do normal e chuva extrema no Brasil.

---

## 1. Ideia científica central

O projeto parte do seguinte fluxo físico:

```text
Aquecimento do Pacífico
SST, SSTA, SSHA, T(z), S(z), D20, conteúdo de calor
|
v
Acoplamento oceano-atmosfera
ventos, pressão, OLR, vapor d'água, fluxos de calor
|
v
Ponte atmosférica Pacífico -> Brasil
circulação em baixos, médios e altos níveis
|
v
Resposta no Brasil
anomalias de precipitação, seca, chuva extrema
|
v
Mapas pixel-a-pixel e mapas coropléticos
previsão, erro, confiança e explicabilidade
```

Em vez de olhar apenas para um índice agregado, o projeto busca entender a **distribuição espacial da influência**:

```text
Qual região do Pacífico influencia qual pixel do Brasil?
Com qual lag temporal?
Com qual variável física?
Com qual intensidade?
Com qual grau de confiança?
```

---

## 2. Pergunta de pesquisa

Como as variáveis oceanográficas e atmosféricas associadas ao aquecimento do Pacífico se relacionam com anomalias de precipitação no Brasil entre 1980 e a data presente?

Pergunta operacional:

```text
O estado do Pacífico no dia t ajuda a explicar ou prever anomalias de precipitação no Brasil em t + lag?
```

Perguntas específicas:

- Quais sinais do Pacífico antecedem seca ou chuva abaixo do normal no Nordeste?
- Quais sinais do Pacífico antecedem chuva acima do normal ou chuva extrema no Sul?
- Quais regiões do Brasil respondem mais fortemente ao aquecimento do Pacífico?
- Quais variáveis são mais importantes: superfície, subsuperfície, vento, pressão, umidade ou circulação vertical?
- Qual defasagem temporal gera maior relação estatística e física?

---

## 3. O que são lags temporais

Um **lag temporal** é uma defasagem entre o estado do Pacífico e a resposta de chuva no Brasil.

Exemplo:

```text
Pacífico em 1 de janeiro
|
lag de 30 dias
v
Chuva no Brasil em 31 de janeiro
```

O projeto testa vários lags:

```text
0, 7, 15, 30, 45, 60, 90, 120 e 180 dias
```

Isso permite responder:

```text
O aquecimento do Pacífico hoje se relaciona com chuva no Brasil daqui a quantos dias?
```

Em termos de modelagem:

```text
X[t] = variáveis do Pacífico e da atmosfera no tempo t
Y[t + lag] = precipitação ou anomalia de precipitação no Brasil
```

---

## 4. Distribuição espacial pixel-a-pixel

O projeto não deve produzir apenas uma média nacional. A meta é mapear a resposta espacial em grade.

Cada pixel do Brasil pode ter:

- correlação diferente com o Pacífico;
- lag dominante diferente;
- erro histórico diferente;
- confiança diferente;
- variável explicativa dominante diferente.

Exemplo de saída:

```text
Pixel A, semiárido:
maior relação com SSTA do Pacífico em lag de 90 dias

Pixel B, Sul:
maior relação com vento em 850 hPa e TCWV em lag de 30 dias

Pixel C, Amazônia:
relação fraca ou instável
```

Essa abordagem permite construir:

- mapas de anomalia prevista;
- mapas de probabilidade de seca;
- mapas de probabilidade de chuva extrema;
- mapas de confiança;
- mapas de lag dominante;
- mapas de importância das variáveis.

---

## 5. Variáveis centrais

### 5.1 Oceano Pacífico

- `SST`: temperatura da superfície do mar.
- `SSTA`: anomalia da temperatura da superfície do mar.
- `SSHA`: anomalia da altura da superfície do mar.
- `T(z)`: temperatura por profundidade.
- `S(z)`: salinidade por profundidade.
- `D20`: profundidade da isoterma de 20 graus Celsius.
- profundidade da termoclina.
- profundidade da camada de mistura.
- conteúdo de calor oceânico `0-300 m`.
- conteúdo de calor oceânico `0-700 m`.
- correntes oceânicas zonal e meridional.
- fluxos de calor na superfície.

### 5.2 Atmosfera Pacífico -> Brasil

- `SLP`: pressão ao nível do mar.
- `tau_x`: tensão de vento zonal na superfície.
- `tau_y`: tensão de vento meridional na superfície.
- `u10`, `v10`: vento zonal e meridional a 10 m.
- `u850`, `v850`, `q850`: vento e umidade em baixos níveis.
- `u500`, `v500`, `q500`, `z500`, `omega500`: circulação e movimento vertical em níveis médios.
- `u200`, `v200`, `z200`, `div200`: circulação e divergência em altos níveis.
- `OLR`: radiação de onda longa emitida.
- `TCWV`: vapor d'água total integrado na coluna atmosférica.

### 5.3 Precipitação no Brasil

- precipitação diária.
- anomalia diária.
- acumulado de 3, 5, 7, 15 e 30 dias.
- evento seco abaixo do `P10`.
- chuva alta acima do `P90`.
- chuva extrema acima do `P95` ou `P99`.

---

## 6. Estrutura do projeto

```text
NINO26/
  configs/
    project.yaml
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
    nino_brasil/
      data/
      features/
      models/
      maps/
      web/
  scripts/
    smoke_test.py
  docs/
    index.html
    assets/
      maps/
```

### O que vai em cada pasta

`data/raw/`
Arquivos brutos baixados das fontes. Não devem ser editados manualmente.

`data/interim/`
Dados já recortados, padronizados ou convertidos, mas ainda não prontos para modelagem.

`data/processed/`
Dados prontos para análise, treino e mapas. Aqui entram `Zarr`, `Parquet` e `GeoTIFF`.

`data/catalog/`
Catálogo de rastreabilidade dos datasets.

`src/nino_brasil/data/`
Scripts de download, padronização, qualidade, anomalias e lags.

`src/nino_brasil/features/`
Cálculo de variáveis derivadas: conteúdo de calor, termoclina, eventos de chuva e ponte atmosférica.

`src/nino_brasil/models/`
Modelos baseline, validação e treino.

`src/nino_brasil/maps/`
Geração de mapas pixel-a-pixel e coropléticos.

`docs/`
Pasta que será publicada no GitHub Pages.

---

## 7. Como rodar na sua IDE

As instruções abaixo funcionam no terminal da sua IDE, como VS Code, PyCharm ou terminal integrado do Cursor.

### 7.1 Abrir a pasta do projeto

Abra a pasta:

```text
C:\DEV\NINO26
```

No terminal da IDE:

```powershell
cd C:\DEV\NINO26
```

### 7.2 Criar o ambiente virtual

Se o ambiente ainda não existir:

```powershell
python -m venv .venv
```

Ativar no Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear ativação, use:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

sem ativar manualmente.

### 7.3 Instalar dependências

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### 7.4 Rodar o teste inicial

```powershell
.\.venv\Scripts\python .\scripts\smoke_test.py
```

Saída esperada:

```text
Smoke test OK
lagged X time: 1065
lagged Y time: 1065
```

Esse teste usa dados sintéticos e confirma que o pipeline básico funciona:

- anomalias;
- eventos de chuva;
- lag temporal;
- conteúdo de calor;
- D20;
- mapa pixel-a-pixel.

O mapa gerado aparece em:

```text
docs/assets/maps/smoke_precip_anomaly.png
```

---

## 8. Fluxo completo do projeto

### Etapa 1 - Download dos dados

Objetivo:

```text
baixar dados brutos e salvar em data/raw/
```

Scripts previstos:

```text
src/nino_brasil/data/download_era5.py
src/nino_brasil/data/download_oras.py
src/nino_brasil/data/download_cpc_noaa.py
```

Exemplo futuro:

```powershell
.\.venv\Scripts\python .\src\nino_brasil\data\download_cpc_noaa.py
```

Estado atual:

```text
os scripts existem como scaffolds
o próximo passo é implementar o download real de NOAA OISST e precipitação
```

---

### Etapa 2 - Salvamento e organização dos dados

Os dados baixados devem ser salvos em:

```text
data/raw/era5/
data/raw/oras/
data/raw/cpc_noaa/
```

Regra:

```text
data/raw/ nunca deve ser alterado manualmente
```

Depois, os dados tratados seguem para:

```text
data/interim/
```

E os dados prontos para modelagem seguem para:

```text
data/processed/zarr/
data/processed/parquet/
data/processed/geotiff/
```

---

### Etapa 3 - Tratamento dos dados

Tratamentos principais:

1. Padronizar nomes de variáveis.
2. Padronizar unidades.
3. Corrigir longitude.
4. Recortar o Pacífico.
5. Recortar o Brasil.
6. Calcular climatologia diária.
7. Calcular anomalias.
8. Criar acumulados de chuva.
9. Criar eventos de seca e extremos.
10. Criar datasets com lags.

Módulos envolvidos:

```text
src/nino_brasil/data/standardize.py
src/nino_brasil/data/anomalies.py
src/nino_brasil/data/build_lagged_dataset.py
src/nino_brasil/features/precipitation_events.py
```

---

### Etapa 4 - Construção das variáveis físicas

Variáveis oceânicas:

```text
D20
termoclina
camada de mistura
conteúdo de calor 0-300 m
conteúdo de calor 0-700 m
```

Módulos:

```text
src/nino_brasil/features/ocean_heat.py
src/nino_brasil/features/thermocline.py
```

Variáveis atmosféricas:

```text
vento por nível
umidade por nível
geopotencial
omega
divergência
```

Módulo:

```text
src/nino_brasil/features/atmospheric_bridge.py
```

---

### Etapa 5 - Montagem do dataset de modelagem

Formato conceitual:

```text
X[t] = Pacífico + atmosfera no tempo t
Y[t + lag] = precipitação no Brasil
```

Exemplo:

```text
SSTA do Pacífico em 01/01/2000
lag de 30 dias
anomalia de chuva no Brasil em 31/01/2000
```

Módulo:

```text
src/nino_brasil/data/build_lagged_dataset.py
```

---

### Etapa 6 - Treino dos modelos

Começar sempre pelos baselines:

```text
climatologia
persistência
correlação defasada
regressão regularizada
Random Forest
XGBoost
```

Depois:

```text
CNN
ConvLSTM
U-Net
Transformer espaço-temporal
```

Módulos:

```text
src/nino_brasil/models/baselines.py
src/nino_brasil/models/validate.py
```

---

### Etapa 7 - Validação e XAI

Validação:

```text
treino por anos iniciais
validação por anos intermediários
teste por anos finais
walk-forward validation
```

Métricas:

```text
correlação
MAE
RMSE
Brier Score
ROC-AUC
FSS
erro por pixel
erro por região
erro por lag
```

XAI deve responder:

```text
qual região do Pacífico importou mais?
qual variável importou mais?
qual lag foi dominante?
qual profundidade oceânica teve mais peso?
o Nordeste e o Sul respondem de forma diferente?
```

---

### Etapa 8 - Geração dos mapas

Mapas pixel-a-pixel:

```text
anomalia prevista
probabilidade de seca
probabilidade de chuva extrema
erro histórico
confiança
lag dominante
importância das variáveis
```

Mapas coropléticos:

```text
risco por estado
risco por região
risco por bacia
risco no semiárido
risco por bioma
```

Módulos:

```text
src/nino_brasil/maps/plot_pixel_maps.py
src/nino_brasil/maps/plot_choropleths.py
```

---

### Etapa 9 - Publicação na web

O GitHub Pages publica a pasta:

```text
docs/
```

Fluxo:

```text
modelo gera previsão
|
salva mapas em docs/assets/maps/
|
docs/index.html apresenta resultados
|
GitHub Pages publica a página
```

Módulo:

```text
src/nino_brasil/web/build_site.py
```

---

## 9. Comandos mais usados

Rodar teste sintético:

```powershell
.\.venv\Scripts\python .\scripts\smoke_test.py
```

Ver arquivos Git:

```powershell
git status
```

Salvar alterações:

```powershell
git add .
git commit -m "Mensagem do commit"
git push
```

---

## 10. Próximo passo recomendado

O próximo passo técnico é implementar o primeiro download real:

```text
NOAA OISST
precipitação diária CPC/NOAA ou CHIRPS
```

Com esses dois dados já será possível gerar o primeiro produto científico:

```text
mapa de correlação defasada entre SSTA do Pacífico e anomalia de precipitação no Brasil
para lags de 30, 60 e 90 dias
```

Esse será o primeiro marco real do projeto.
