# Projeto de previsao climatica El NINO e seus impactos no Brasil

**Programa de Pos Graduacao em Oceanografia UFPE**  
**Area:** Oceanografia Fisica  
**Responsavel:** Thiago Vilar

# DATA_SOURCES

Este documento define, de forma operacional, quais bases serao baixadas, quais variaveis entram em cada requisicao, onde os arquivos ficam salvos e qual ETL gera os produtos padronizados em Zarr.

## 1. Regras fixas

- Periodo do projeto: `1981-01-01` ate o ultimo dado disponivel por fonte.
- Frequencia mestre da modelagem: diaria.
- Downloads grandes devem ser executados por ano ou por mes, nunca como bloco unico.
- Todo dado bruto entra por `data/raw/`; ERA5/ORAS podem usar raw como cache temporario e apagar o arquivo pesado depois do Zarr validado.
- Todo dado temporario ou extraido fica em `data/interim/`.
- Todo dado tratado para modelagem fica em `data/processed/`.
- Saidas tabulares de modelagem e diagnostico ficam em stores `.zarr` sob `data/processed/zarr/`.
- Todo produto multidimensional tratado deve ser salvo em `.zarr`.
- Toda etapa critica deve registrar evento em `data/audit/ledger.jsonl`.
- A latencia de cada fonte deve ser registrada no ledger; `--end-year` e opcional e e limitado por `latest_available`.
- Arquivo `.part` indica download interrompido e pode ser retomado pelo mesmo comando.
- O uso de `--overwrite` deve ficar restrito a arquivo corrompido ou reprocessamento deliberado.

## 2. Dominios espaciais

### 2.1 Pacifico associado ao El NINO

Uso: aquecimento do Pacifico, estrutura vertical, memoria oceanica e acoplamento oceano-atmosfera.

```text
Latitude: -35 a 30
Longitude: 120E a 70W
Longitude em 0-360: 120 a 290
```

### 2.2 Brasil

Uso: precipitacao diaria, anomalias de precipitacao, mascara territorial e mapas pixel-a-pixel.

```text
Mascara oficial: IBGE
Recorte aproximado: -35 a 7 latitude, -75 a -30 longitude
```

IBGE nao e fonte de chuva. IBGE fornece limite territorial. A precipitacao vem de CHIRPS e depois e recortada pela mascara do Brasil.

### 2.3 Ponte atmosferica Pacifico-Brasil

Uso: variaveis atmosfericas que conectam o aquecimento do Pacifico a resposta de chuva no Brasil.

```text
west_pacific:
  latitude: -35 a 30
  longitude: 120E a 180E

east_pacific_brazil:
  latitude: -35 a 30
  longitude: 180W a 30W
```

O dominio atmosferico e dividido em dois blocos para evitar erro no cruzamento do antimeridiano.

## 3. Estrutura local

```text
data/
  raw/
    chirps/
    cpc_noaa/
      oisst/
    ctd_noaa/
      wod/
    era5/
      single_levels/
      pressure_levels/
    ibge/
    oras/
  interim/
    brazil_precipitation/
    ctd_noaa/
    ibge/
    nino34/
    oras/
  processed/
    zarr/
      brazil_precipitation/
      cpc_noaa/
      ctd_noaa/
      distributions/
      era5/
      features/
      modeling/
      oras/
      regridded/
      statistics/
    geotiff/
      ibge/
  audit/
    ledger.jsonl
  state/
```

## 4. Inventario operacional de variaveis

### 4.1 ORAS5

Fonte: ECMWF / Copernicus Climate Data Store.  
Produto: ORAS5 global ocean reanalysis monthly data.  
Link: https://cds.climate.copernicus.eu/datasets/reanalysis-oras5  
Uso: oceano subsuperficial do Pacifico.

Variaveis solicitadas pelo script no CDS:

```text
potential_temperature
salinity
ocean_heat_content_for_the_upper_300m
ocean_heat_content_for_the_upper_700m
```

Significado fisico:

```text
potential_temperature: temperatura potencial do oceano por profundidade
salinity: salinidade por profundidade
ocean_heat_content_for_the_upper_300m: conteudo de calor oceanico 0-300 m
ocean_heat_content_for_the_upper_700m: conteudo de calor oceanico 0-700 m
```

Frequencia e formato:

```text
fonte: mensal
download historico: anual agrupado por ano no modo --annual-zarr --request-mode annual-kind
fallback: anual por variavel ou mensal para ano em aberto
bruto: ZIP/NetCDF agrupado em cache temporario
processado: Zarr diario anual por variavel
```

Destino local:

```text
data/raw/oras/<ano>/_annual_kind/oras5_all_variables_<ano>.zip
data/interim/oras/<ano>/_annual_kind/oras5_all_variables_<ano>/
data/processed/zarr/oras/<ano>/<variavel>/oras5_<variavel>_<ano>_daily.zarr
```

Derivados calculados depois do download:

```text
anomalia de temperatura oceanica
anomalia de salinidade
D20: profundidade da isoterma de 20 graus Celsius
termoclina
camada de mistura
densidade potencial
conteudo de calor 0-300 m
conteudo de calor 0-700 m
```

### 4.2 CTD NOAA WOD

Fonte: NOAA / NCEI.  
Produto: World Ocean Database, CTD annual NetCDF.  
Catalogo: https://www.ncei.noaa.gov/thredds-ocean/catalog/ncei/wod/catalog.html  
Produto WOD: https://www.ncei.noaa.gov/products/world-ocean-database  
Uso: perfis in situ para validar a estrutura vertical do Pacifico e corrigir vieses de ORAS5.

Arquivo bruto anual baixado:

```text
wod_ctd_<ano>.nc
```

Variaveis lidas pelo ETL:

```text
lat
lon
time
date
wod_unique_cast
z
z_row_size
z_WODflag
Temperature
Temperature_row_size
Temperature_WODflag
Temperature_WODprofileflag
Salinity
Salinity_row_size
Salinity_WODflag
Salinity_WODprofileflag
```

No modo padrao focado em termoclina, temperatura e profundidade sao obrigatorias; salinidade e usada quando passa QC. Use `--require-salinity` para exigir perfis completos T/S.

Significado fisico:

```text
lat: latitude do perfil
lon: longitude do perfil
time/date: data do perfil
wod_unique_cast: identificador unico do perfil no WOD
z: profundidade observada em metros
Temperature: temperatura in situ por profundidade
Salinity: salinidade pratica por profundidade
*_row_size: tamanho de cada perfil no formato ragged-array do WOD
*_WODflag: flag de controle de qualidade por observacao
*_WODprofileflag: flag de controle de qualidade do perfil inteiro
```

Politica de controle de qualidade:

```text
padrao atual: aceitar somente flag 0
flag 0: accepted
aplicacao: profundidade, temperatura e perfil de temperatura; salinidade quando disponivel
```

ETL termodinamico TEOS-10:

```text
pressure: pressao calculada a partir de profundidade e latitude
absolute_salinity: Salinidade Absoluta, calculada por TEOS-10
conservative_temperature: Temperatura Conservativa, calculada por TEOS-10
sigma0: anomalia de densidade potencial na referencia de 0 dbar
```

TEOS-10 depende de salinidade valida; quando o perfil e apenas de temperatura, as variaveis termodinamicas ficam ausentes/NaN e a termoclina ainda e calculada pela temperatura in situ.

Estratificacao calculada:

```text
thermocline_depth: profundidade de maior gradiente vertical de temperatura in situ
halocline_depth: profundidade de maior gradiente vertical de salinidade absoluta
pycnocline_depth: profundidade de maior gradiente vertical de sigma0
```

Frequencia e formato:

```text
fonte: perfis irregulares por data e local
download: anual
bruto: NetCDF ragged-array
processado: Zarr anual em grade vertical regular
grade vertical do ETL: 0 a 300 m, passo de 5 m
```

Destino local:

```text
data/raw/ctd_noaa/wod/<ano>/wod_ctd_<ano>.nc
data/processed/zarr/ctd_noaa/wod/<ano>/wod_ctd_<ano>.zarr
```

Observacao tecnica:

CTD nao e grade diaria continua. Cada perfil tem data, posicao e profundidade propria. No projeto, CTD entra como observacao vertical para controle, validacao, comparacao com ORAS5 e possivel correcao de vies.

### 4.3 TAO/TRITON/Argo como validacao in situ

Esta camada nao muda a direcao do projeto NINO26. Ela entra como validacao independente para testar, ao longo do desenvolvimento, se a subsuperficie do Pacifico Nino 3.4 acrescenta informacao aos modelos de chuva/seca no Brasil alem da SST/SSTA superficial.

Fontes candidatas:

```text
TAO/TRITON/GTMBA: fundeios equatoriais do Pacifico tropical com temperatura subsuperficial e, em parte da rede/periodo, salinidade.
Argo GDAC: perfis autonomos globais de temperatura, salinidade e pressao, mais forte a partir dos anos 2000.
WOD/GTSPP: agregadores de perfis in situ que podem ajudar na auditoria, com cuidado para duplicatas.
```

Uso metodologico:

```text
1. Validar ORAS5 em pontos/periodos com observacao in situ.
2. Verificar se D20, termoclina, OHC 0-300 m e temperatura media 0-300 m sao fisicamente consistentes.
3. Marcar anos/periodos em que CTD/WOD esta vazio no Nino 3.4, sem preencher lacunas artificialmente.
4. Comparar modelos com e sem subsuperficie, mantendo a pergunta em aberto.
```

Papel por fonte:

```text
OISST: referencia diaria de SST/SSTA superficial.
ORAS5: campo continuo para memoria oceanica subsuperficial.
CTD/WOD: perfis observados irregulares para controle vertical.
TAO/TRITON: serie temporal de fundeios para validacao equatorial.
Argo: perfis T/S independentes para validacao pos-2000.
```

Produtos baixados/preparados:

```text
data/raw/tao_triton/temperature/tao_triton_temperature_<ano>.csv
data/raw/tao_triton/salinity/tao_triton_salinity_<ano>.csv
data/raw/argo/argo_nino34_<ano>.csv
data/processed/zarr/validation/tao_triton/
data/processed/zarr/validation/argo/
```

Estado operacional atual:

```text
documentado como camada de validacao;
download bruto implementado em scripts/data_pipeline.py download-validation;
ETL Zarr de validacao ainda e etapa posterior;
nao bloquear o fechamento do CTD/WOD, ERA5 ou ORAS5.
```

Comando Windows cmd:

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py download-validation --source all --start-year 1981 --max-depth 300 --execute --continue-on-error
```

### 4.4 NOAA OISST

Fonte: NOAA.  
Produto: Optimum Interpolation Sea Surface Temperature v2.1.  
Link: https://www.ncei.noaa.gov/products/optimum-interpolation-sst  
Espelho usado pelo script: https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres  
Uso: aquecimento superficial diario do Pacifico.

Variavel baixada:

```text
sst
```

Significado fisico:

```text
sst: temperatura da superficie do mar
```

Frequencia e formato:

```text
fonte: diaria
download: anual
bruto: NetCDF anual
processado: Zarr depois do recorte e calculo de anomalias
```

Destino local:

```text
data/raw/cpc_noaa/oisst/sst.day.mean.<ano>.nc
data/processed/zarr/cpc_noaa/oisst/
```

Derivados calculados depois do download:

```text
SSTA: anomalia diaria de temperatura da superficie do mar
media movel de SSTA
campos defasados por lag
```

### 4.5 ERA5 single levels

Fonte: ECMWF / Copernicus Climate Data Store.  
Produto: ERA5 single levels.  
Link: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels  
Uso: superficie, fluxos e umidade integrada da ponte atmosferica.

Variaveis solicitadas pelo script no CDS:

```text
mean_sea_level_pressure
10m_u_component_of_wind
10m_v_component_of_wind
total_column_water_vapour
surface_latent_heat_flux
surface_sensible_heat_flux
surface_net_solar_radiation
surface_net_thermal_radiation
```

Significado fisico:

```text
mean_sea_level_pressure: pressao ao nivel medio do mar
10m_u_component_of_wind: componente zonal do vento a 10 m
10m_v_component_of_wind: componente meridional do vento a 10 m
total_column_water_vapour: vapor d'agua total integrado na coluna
surface_latent_heat_flux: fluxo turbulento de calor latente na superficie
surface_sensible_heat_flux: fluxo turbulento de calor sensivel na superficie
surface_net_solar_radiation: radiacao solar liquida na superficie
surface_net_thermal_radiation: radiacao termica liquida na superficie
```

Frequencia e formato:

```text
fonte: horaria
download do projeto: 00:00, 06:00, 12:00, 18:00
download: anual por regiao/tipo com variaveis agrupadas no modo --annual-zarr --request-mode annual-kind
bruto: NetCDF anual agrupado em cache temporario
processado: Zarr diario anual por regiao/variavel
fallback: annual-variable ou NetCDF/Zarr mensal apenas com --request-mode monthly-kind ou sem --annual-zarr
```

Destino local:

```text
data/raw/era5/single_levels/<ano>/_annual_kind/era5_single_<regiao>_all_variables_<ano>.nc
data/processed/zarr/era5/single_levels/<ano>/<variavel>/era5_single_<regiao>_<variavel>_<ano>_daily.zarr
```

### 4.6 ERA5 pressure levels

Fonte: ECMWF / Copernicus Climate Data Store.  
Produto: ERA5 pressure levels.  
Link: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels  
Uso: circulacao vertical e horizontal em baixos, medios e altos niveis.

Variaveis solicitadas pelo script no CDS:

```text
u_component_of_wind
v_component_of_wind
specific_humidity
geopotential
vertical_velocity
divergence
```

Niveis de pressao solicitados:

```text
850 hPa
500 hPa
200 hPa
```

Significado fisico:

```text
u_component_of_wind: vento zonal em nivel de pressao
v_component_of_wind: vento meridional em nivel de pressao
specific_humidity: umidade especifica em nivel de pressao
geopotential: geopotencial, usado para altura geopotencial e padroes de circulacao
vertical_velocity: velocidade vertical em coordenada de pressao
divergence: divergencia horizontal do vento
850 hPa: baixos niveis, transporte de umidade
500 hPa: medios niveis, subsidencia/ascensao e bloqueios
200 hPa: altos niveis, divergencia e jatos superiores
```

Frequencia e formato:

```text
fonte: horaria
download do projeto: 00:00, 06:00, 12:00, 18:00
download: anual por regiao/tipo com variaveis agrupadas no modo --annual-zarr --request-mode annual-kind
bruto: NetCDF anual agrupado em cache temporario
processado: Zarr diario anual por regiao/variavel
fallback: annual-variable ou NetCDF/Zarr mensal apenas com --request-mode monthly-kind ou sem --annual-zarr
```

Destino local:

```text
data/raw/era5/pressure_levels/<ano>/_annual_kind/era5_pressure_<regiao>_all_variables_<ano>.nc
data/processed/zarr/era5/pressure_levels/<ano>/<variavel>/era5_pressure_<regiao>_<variavel>_<ano>_daily.zarr
```

### 4.7 CHIRPS precipitacao diaria

Fonte: Climate Hazards Center / UC Santa Barbara.  
Produto: CHIRPS v2.0 global daily 0.25 degree nas Fases 1 a 7; 0.05 degree reservado para experimento futuro de alta resolucao.
Link: https://www.chc.ucsb.edu/data/chirps  
Download usado pelo script na Fase 1: https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p25/  
Uso: resposta observada de chuva no Brasil.

Variavel baixada:

```text
precip
```

Significado fisico:

```text
precip: precipitacao diaria acumulada
```

Frequencia e formato:

```text
fonte: diaria
download: anual
resolucao: 0.25 grau nas Fases 1 a 7; 0.05 grau em experimento futuro
bruto: NetCDF anual
processado: Zarr depois do recorte pelo Brasil
```

Destino local:

```text
data/raw/chirps/p25/chirps-v2.0.<ano>.days_p25.nc
data/processed/zarr/brazil_precipitation/
```

Derivados calculados depois do download:

```text
anomalia diaria de precipitacao
acumulado de 3 dias
acumulado de 5 dias
acumulado de 7 dias
acumulado de 15 dias
acumulado de 30 dias
evento seco por percentil local
chuva abaixo do quartil inferior por P25 local
chuva acima do quartil superior por P75 local
chuva acima do normal por percentil local
```

### 4.8 IBGE

Fonte: Instituto Brasileiro de Geografia e Estatistica.  
Produto: malhas territoriais oficiais.  
Link: https://www.ibge.gov.br/geociencias/organizacao-do-territorio/malhas-territoriais.html  
Uso: mascara do Brasil, estados, municipios e mapas coropleticos.

Produtos baixados:

```text
uf
municipios
```

Variaveis/cartografia usadas:

```text
geometria territorial
codigo da unidade territorial
nome da unidade territorial
sigla da unidade federativa
```

Destino local:

```text
data/raw/ibge/BR_UF_2024.zip
data/raw/ibge/BR_Municipios_2024.zip
data/interim/ibge/BR_UF_2024/
data/interim/ibge/BR_Municipios_2024/
data/processed/geotiff/ibge/
```

## 5. Comandos de download e ETL

Todos os comandos devem ser executados na raiz do projeto usando o Python do ambiente WSL:

```bash
./.venv-wsl/bin/python scripts/data_pipeline.py <comando>
```

### 5.1 Criar pastas

```bash
./.venv-wsl/bin/python scripts/data_pipeline.py init
```

### 5.2 Conferir credencial Copernicus

```bash
./.venv-wsl/bin/python scripts/data_pipeline.py check-cds
```

### 5.3 Baixar IBGE

```bash
./.venv-wsl/bin/python scripts/data_pipeline.py download-ibge --product uf
./.venv-wsl/bin/python scripts/data_pipeline.py download-ibge --product municipios
```

### 5.4 Baixar CHIRPS 0.25 diario com retomada

```bash
./.venv-wsl/bin/python scripts/curate_and_resume_downloads.py --source chirps --stage all --start-year 1981 --execute --limit 0 --retries 5 --retry-wait 60 --continue-on-error
```

### 5.5 Baixar NOAA OISST diario com retomada

```bash
./.venv-wsl/bin/python scripts/curate_and_resume_downloads.py --source oisst --stage all --start-year 1981 --execute --limit 0 --retries 5 --retry-wait 60 --continue-on-error
```

### 5.6 Baixar CTD NOAA WOD e gerar Zarr para termoclina

Este comando baixa o NetCDF anual do WOD, aplica filtro espacial Nino 3.4, aceita somente flags WOD aprovadas, usa grade 0-300 m para termoclina e grava Zarr anual. Anos sem perfis de temperatura no Nino 3.4 sao registrados como `ano - sem dados` e o lote continua.

```bash
./.venv-wsl/bin/python scripts/data_pipeline.py download-ctd --start-year 1981 --max-depth 300 --min-levels 3 --execute --continue-on-error
```

Para baixar somente o bruto e deixar o ETL para depois:

```bash
./.venv-wsl/bin/python scripts/data_pipeline.py download-ctd --start-year 1981 --raw-only --execute --continue-on-error
```

Para processar CTD bruto ja baixado:

```bash
./.venv-wsl/bin/python scripts/data_pipeline.py etl-ctd --start-year 1981 --max-depth 300 --min-levels 3 --execute --continue-on-error
```

### 5.7 Baixar ERA5 e converter para Zarr

```bash
./.venv-wsl/bin/python scripts/data_pipeline.py download-era5 --start-year 1981 --kind both --region nino34 --region brazil --annual-zarr --request-mode annual-kind --delete-raw-after-zarr --execute --continue-on-error
```

### 5.8 Baixar ORAS5 e converter para Zarr

```bash
./.venv-wsl/bin/python scripts/data_pipeline.py download-oras --start-year 1981 --annual-zarr --request-mode annual-kind --delete-raw-after-zarr --execute --continue-on-error
```

### 5.9 Baixar TAO/TRITON/Argo para validacao in situ

```bash
./.venv-wsl/bin/python scripts/data_pipeline.py download-validation --source all --start-year 1981 --max-depth 300 --execute --continue-on-error
```

TAO/TRITON usa PMEL ERDDAP (`pmelTaoDyT` e `pmelTaoDyS`) no recorte Nino 3.4. Argo usa Ifremer ERDDAP (`ArgoFloats`) no mesmo recorte, com `--argo-start-year 1999` como padrao interno. Se `--end-date` for omitido, o script usa a data local do dia da execucao.

## 6. Continuidade depois de erro

```text
1. Nao reiniciar tudo do zero.
2. Usar os comandos de retomada com --continue-on-error.
3. O download HTTP retoma arquivo .part quando o servidor aceita Range.
4. O ETL pula Zarr valido quando ele ja existe.
5. O ledger registra ok, erro, fonte ausente ou ano sem dado util.
6. Rodar audit ao fim do lote.
7. Usar --overwrite somente quando houver arquivo corrompido ou decisao clara de reprocessar.
```

Comando de auditoria:

```bash
./.venv-wsl/bin/python scripts/data_pipeline.py audit
```

## 7. Criterio minimo de base pronta

Uma base so deve ser considerada pronta quando:

- o arquivo bruto existe em `data/raw/`;
- o Zarr existe em `data/processed/zarr/`, quando houver ETL;
- o Zarr abre com `xarray.open_zarr`;
- o dominio espacial foi aplicado ou documentado;
- a frequencia temporal foi preservada ou agregada com regra explicita;
- a auditoria registra `status="ok"`;
- nao ha `.part` sendo tratado como produto final.

## 8. Periodo por fonte

```text
CHIRPS: 1981-latest_available, diario, 0.25 grau nas Fases 1 a 7; 0.05 grau reservado para experimento futuro de alta resolucao
NOAA OISST: 1981-latest_available, diario, 0.25 grau, SST/SSTA principal
CTD NOAA WOD: 1981-latest_available, perfis irregulares, filtrados no Pacifico
TAO/TRITON/Argo: validacao in situ baixavel por ERDDAP; bruto em CSV anual, ETL Zarr posterior
ERA5: 1981-latest_available, subdiario baixado em 4 horarios e agregado depois para diario
ORAS5: 1981-latest_available, fonte mensal baixada no historico por ano/variavel, reservado para memoria subsuperficial oceanica
IBGE: cartografia estatica
```
