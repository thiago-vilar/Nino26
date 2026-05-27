# Projeto de previsão climática El NINO e seus impactos no Brasil

**Programa de Pós Graduação em Oceanografia UFPE**  
**Área:** Oceanografia Física  
**Responsável:** Thiago Vilar  

## NINO-BRASIL

O **NINO-BRASIL** é um projeto em Python para investigar como variáveis oceanográficas e atmosféricas associadas ao aquecimento do Pacífico se relacionam, no espaço e no tempo, com anomalias de precipitação no Brasil.

O foco não é usar apenas um índice climático agregado. O foco é medir o peso relativo das variáveis oceânicas e atmosféricas, identificar onde esses pesos aparecem no território brasileiro e avaliar em quais defasagens temporais eles têm maior poder explicativo.

## 1. Pergunta de pesquisa

Qual é o impacto, em termos de peso relativo, das variáveis oceanográficas e atmosféricas associadas ao aquecimento do Pacífico sobre as anomalias de precipitação no Brasil, considerando sua distribuição espacial e temporal entre 1980 e a data presente?

De forma operacional, o projeto pergunta:

```text
Quais variáveis do Pacífico e da ponte atmosférica no dia t explicam melhor
as anomalias de precipitação no Brasil em t + lag?
```

## 2. Hipótese física

O aquecimento do Pacífico altera a estrutura oceânica superficial e subsuperficial. Essa alteração afeta o acoplamento oceano-atmosfera, modifica vento, pressão, umidade, movimento vertical e convecção, e esse sinal pode se propagar até o Brasil por teleconexões atmosféricas.

O projeto avalia essa cadeia física em três blocos:

```text
Aquecimento do Pacífico -> Ponte atmosférica -> Precipitação no Brasil
```

## 3. Domínios espaciais do projeto

O projeto trabalha com três áreas espaciais, cada uma com uma função física e computacional diferente.

| Domínio | Fonte principal | Recorte | Função |
|---|---|---|---|
| Pacífico oceânico | ORAS/ORAS5 + CPC/NOAA | 35S a 30N / 120E a 70W | Caracterizar o aquecimento associado ao El NINO por variáveis oceânicas superficiais e subsuperficiais. |
| Brasil | IBGE + precipitação em grade | 35S a 7N / 75W a 30W | Representar o território brasileiro, aplicar máscara espacial, agregar por recortes oficiais e avaliar anomalias de precipitação. |
| Ponte atmosférica | ERA5 | área abrangendo Pacífico oceânico e Brasil | Medir como vento, pressão, umidade, geopotencial, movimento vertical e divergência conectam o Pacífico aquecido à resposta de chuva no Brasil. |

O `IBGE` fornece os shapefiles, limites oficiais e máscaras territoriais. A série pluviométrica 1980-presente deve vir de uma base climática em grade, como `CPC/NOAA`, `CHIRPS`, `MERGE/INPE` ou outra fonte definida no catálogo.

## 4. Escopo espacial e temporal

```text
Período: 1980 até a data presente
Pacífico: 35S a 30N / 120E a 70W
Brasil: território nacional em grade pixel-a-pixel
Atmosfera: domínio amplo cobrindo Pacífico oceânico e Brasil
Frequência preferencial: diária
Lags iniciais: 0, 7, 15, 30, 45, 60, 90, 120 e 180 dias
```

## 5. Dados e justificativa

| Bloco | Fonte principal | Download local | Justificativa |
|---|---|---|---|
| Oceano Pacífico | ORAS/ORAS5 + CPC/NOAA | `data/raw/oras/` e `data/raw/cpc_noaa/` | Fornece temperatura, salinidade, SST/SSTA e variáveis derivadas para caracterizar o aquecimento do Pacífico. |
| Oceano observado | CTD NOAA/WOD | `data/raw/ctd_noaa/` | Fornece perfis observacionais de temperatura e salinidade para validar, comparar e corrigir a estrutura vertical representada pelo ORAS. |
| Atmosfera | ERA5 | `data/raw/era5/` | Representa a ponte atmosférica Pacífico -> Brasil por vento, pressão, umidade, geopotencial, movimento vertical, divergência e fluxos de calor. |
| Mapa do Brasil | IBGE | `data/raw/ibge/` | Fornece limites oficiais para recorte, máscara territorial, agregação regional e mapas coropléticos. |
| Precipitação Brasil | CPC/NOAA, CHIRPS, MERGE/INPE ou fonte definida | `data/raw/cpc_noaa/` ou pasta específica | Fornece a resposta observada de chuva usada para calcular anomalias, eventos secos e chuva acima do normal no Brasil. |

## 6. Variáveis centrais

### Oceanográficas

- `SST`: temperatura da superfície do mar.
- `SSTA`: anomalia da temperatura da superfície do mar.
- `SSHA`: anomalia da altura da superfície do mar.
- `T(z)`: temperatura por profundidade.
- `S(z)`: salinidade por profundidade.
- `D20`: profundidade da isoterma de 20 graus Celsius.
- `MLD`: profundidade da camada de mistura.
- `OHC 0-300 m`: conteúdo de calor oceânico entre a superfície e 300 m.
- `OHC 0-700 m`: conteúdo de calor oceânico entre a superfície e 700 m.
- `u/v oceânico`: correntes oceânicas zonal e meridional.
- `clorofila-a`: variável auxiliar de resposta biogeoquímica superficial.

### Atmosféricas

- `SLP`: pressão ao nível do mar.
- `tau_x/tau_y`: tensão de vento zonal e meridional.
- `u10/v10`: vento zonal e meridional a 10 m.
- `u850/v850/q850`: vento e umidade em baixos níveis.
- `u500/v500/q500/z500/omega500`: circulação, umidade, geopotencial e movimento vertical em níveis médios.
- `u200/v200/z200/div200`: circulação, geopotencial e divergência em altos níveis.
- `OLR`: radiação de onda longa emitida.
- `TCWV`: vapor d'água total integrado na coluna atmosférica.

### Precipitação no Brasil

- precipitação diária.
- anomalia diária.
- acumulados de 3, 5, 7, 15 e 30 dias.
- evento seco abaixo de `P10`.
- chuva acima do normal acima de `P90`.

## 7. Fluxo e produto final

```text
download dos dados brutos
|
padronização, recorte espacial e controle de qualidade
|
cálculo de anomalias, variáveis derivadas e lags temporais
|
treinamento e validação dos modelos
|
dimensionamento dos pesos oceânicos e atmosféricos
|
geração de mapas pixel-a-pixel e coropléticos
|
publicação no GitHub Pages
```

Produto final no GitHub Pages:

- anomalia prevista de precipitação.
- probabilidade de seca.
- probabilidade de chuva acima do normal.
- lag dominante.
- peso oceanográfico.
- peso atmosférico.
- variável dominante.
- erro histórico.
- confiança da previsão.

## 8. Entrada principal de dados

O pipeline de dados começa pelo script:

```text
scripts/data_pipeline.py
```

Comandos disponíveis:

```text
python scripts/data_pipeline.py status
python scripts/data_pipeline.py init
python scripts/data_pipeline.py plan
python scripts/data_pipeline.py download-ibge --product uf
python scripts/data_pipeline.py download-ibge --product municipios
python scripts/data_pipeline.py download-chirps --start-year 1981 --end-year 1981
python scripts/data_pipeline.py download-oisst --start-year 1981 --end-year 1981
python scripts/data_pipeline.py download-era5 --start-year 1981 --end-year 1981 --month 1
python scripts/data_pipeline.py download-oras --start-year 1981 --end-year 1981 --month 1
python scripts/data_pipeline.py download-all --start-year 1981 --end-year 1981
```

O primeiro bloco real baixa e extrai shapefiles oficiais do IBGE em:

```text
data/raw/ibge/
data/interim/ibge/
```

Por padrão, os comandos de bases climáticas rodam em modo `dry-run`. Para baixar de verdade, adicione:

```text
--execute
```

Para incluir ERA5 e ORAS no `download-all`, adicione:

```text
--include-cds
```

## 9. Estrutura e execução na IDE

```text
NINO26/
  configs/
  data/
    raw/
      era5/
      oras/
      ctd_noaa/
      cpc_noaa/
      ibge/
    interim/
    processed/
    catalog/
  src/
    nino_brasil/
      data/
      features/
      models/
      maps/
      web/
  scripts/
  docs/
```

Abra a pasta do projeto:

```text
C:\DEV\NINO26
```

Crie o ambiente virtual:

```powershell
python -m venv .venv
```

Instale as dependências:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Execute o teste inicial:

```powershell
.\.venv\Scripts\python .\scripts\smoke_test.py
```

O teste confirma que a estrutura Python está funcional e gera um primeiro mapa sintético em:

```text
docs/assets/maps/smoke_precip_anomaly.png
```

Para uso com placa de vídeo no VSCode via WSL, siga:

```text
WSL_GPU_SETUP.md
```
