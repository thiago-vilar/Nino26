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
- Todo dado bruto fica em `data/raw/`.
- Todo dado temporario ou extraido fica em `data/interim/`.
- Todo dado tratado para modelagem fica em `data/processed/`.
- Saidas tabulares de modelagem e diagnostico ficam em `data/processed/parquet/`.
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
    atmosphere_bridge/
    brazil_precipitation/
    ctd_noaa/
    ibge/
    oras/
    pacific_warming/
  processed/
    zarr/
      atmosphere_bridge/
      brazil_precipitation/
      cpc_noaa/
      ctd_noaa/
      era5/
      oras/
      pacific_warming/
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
sea_surface_temperature
ocean_heat_content_for_the_upper_300m
ocean_heat_content_for_the_upper_700m
```

Significado fisico:

```text
potential_temperature: temperatura potencial do oceano por profundidade
salinity: salinidade por profundidade
sea_surface_temperature: temperatura da superficie do mar
ocean_heat_content_for_the_upper_300m: conteudo de calor oceanico 0-300 m
ocean_heat_content_for_the_upper_700m: conteudo de calor oceanico 0-700 m
```

Frequencia e formato:

```text
fonte: mensal
download: mensal
bruto: ZIP/NetCDF
processado: Zarr mensal
```

Destino local:

```text
data/raw/oras/<ano>/oras5_<ano><mes>.zip
data/interim/oras/<ano>/oras5_<ano><mes>/
data/processed/zarr/oras/<ano>/oras5_<ano><mes>.zarr
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
aplicacao: profundidade, temperatura, salinidade e perfil
```

ETL termodinamico TEOS-10:

```text
pressure: pressao calculada a partir de profundidade e latitude
absolute_salinity: Salinidade Absoluta, calculada por TEOS-10
conservative_temperature: Temperatura Conservativa, calculada por TEOS-10
sigma0: anomalia de densidade potencial na referencia de 0 dbar
```

Estratificacao calculada:

```text
thermocline_depth: profundidade de maior gradiente vertical de temperatura conservativa
halocline_depth: profundidade de maior gradiente vertical de salinidade absoluta
pycnocline_depth: profundidade de maior gradiente vertical de sigma0
```

Frequencia e formato:

```text
fonte: perfis irregulares por data e local
download: anual
bruto: NetCDF ragged-array
processado: Zarr anual em grade vertical regular
grade vertical do ETL: 0 a 700 m, passo de 5 m
```

Destino local:

```text
data/raw/ctd_noaa/wod/<ano>/wod_ctd_<ano>.nc
data/processed/zarr/ctd_noaa/wod/<ano>/wod_ctd_<ano>.zarr
```

Observacao tecnica:

CTD nao e grade diaria continua. Cada perfil tem data, posicao e profundidade propria. No projeto, CTD entra como observacao vertical para controle, validacao, comparacao com ORAS5 e possivel correcao de vies.

### 4.3 NOAA OISST

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

### 4.4 ERA5 single levels

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
download: mensal por regiao
bruto: NetCDF mensal
processado: Zarr mensal
produto posterior: agregado diario
```

Destino local:

```text
data/raw/era5/single_levels/<ano>/era5_single_<regiao>_<ano><mes>.nc
data/processed/zarr/era5/single_levels/<ano>/era5_single_<regiao>_<ano><mes>.zarr
```

### 4.5 ERA5 pressure levels

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
download: mensal por regiao
bruto: NetCDF mensal
processado: Zarr mensal
produto posterior: agregado diario
```

Destino local:

```text
data/raw/era5/pressure_levels/<ano>/era5_pressure_<regiao>_<ano><mes>.nc
data/processed/zarr/era5/pressure_levels/<ano>/era5_pressure_<regiao>_<ano><mes>.zarr
```

### 4.6 CHIRPS precipitacao diaria

Fonte: Climate Hazards Center / UC Santa Barbara.  
Produto: CHIRPS v2.0 global daily 0.25 degree na Fase 1; 0.05 degree na Fase 2.  
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
resolucao: 0.25 grau na Fase 1; 0.05 grau na Fase 2
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
chuva acima do normal por percentil local
```

### 4.7 IBGE

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

### 5.4 Baixar CHIRPS 0.25 diario, ano a ano

```bash
for y in $(seq 1981 $(date +%Y)); do
  ./.venv-wsl/bin/python scripts/data_pipeline.py download-chirps --start-year "$y" --end-year "$y" --resolution p25 --execute
  ./.venv-wsl/bin/python scripts/data_pipeline.py audit
done
```

### 5.5 Baixar NOAA OISST diario, ano a ano

```bash
for y in $(seq 1981 $(date +%Y)); do
  ./.venv-wsl/bin/python scripts/data_pipeline.py download-oisst --start-year "$y" --end-year "$y" --execute
  ./.venv-wsl/bin/python scripts/data_pipeline.py audit
done
```

### 5.6 Baixar CTD NOAA WOD e gerar Zarr TEOS-10

Este comando baixa o NetCDF anual do WOD, aplica filtro espacial do Pacifico, aceita somente flags WOD aprovadas, calcula TEOS-10 e grava Zarr anual.

```bash
for y in $(seq 1981 $(date +%Y)); do
  ./.venv-wsl/bin/python scripts/data_pipeline.py download-ctd --start-year "$y" --end-year "$y" --execute
  ./.venv-wsl/bin/python scripts/data_pipeline.py audit
done
```

Para baixar somente o bruto e deixar o ETL para depois:

```bash
for y in $(seq 1981 $(date +%Y)); do
  ./.venv-wsl/bin/python scripts/data_pipeline.py download-ctd --start-year "$y" --end-year "$y" --raw-only --execute
  ./.venv-wsl/bin/python scripts/data_pipeline.py audit
done
```

Para processar CTD bruto ja baixado:

```bash
for y in $(seq 1981 $(date +%Y)); do
  ./.venv-wsl/bin/python scripts/data_pipeline.py etl-ctd --start-year "$y" --end-year "$y" --execute
  ./.venv-wsl/bin/python scripts/data_pipeline.py audit
done
```

### 5.7 Baixar ERA5 e converter para Zarr

```bash
for y in $(seq 1981 $(date +%Y)); do
  ./.venv-wsl/bin/python scripts/data_pipeline.py download-era5 --start-year "$y" --end-year "$y" --kind both --execute
  ./.venv-wsl/bin/python scripts/data_pipeline.py audit
done
```

### 5.8 Baixar ORAS5 e converter para Zarr

```bash
for y in $(seq 1981 $(date +%Y)); do
  ./.venv-wsl/bin/python scripts/data_pipeline.py download-oras --start-year "$y" --end-year "$y" --execute
  ./.venv-wsl/bin/python scripts/data_pipeline.py audit
done
```

## 6. Continuidade depois de erro

```text
1. Nao reiniciar tudo do zero.
2. Rodar audit.
3. Repetir o mesmo comando do ano ou mes que falhou.
4. O download HTTP retoma arquivo .part quando o servidor aceita Range.
5. O ETL pula Zarr valido quando ele ja existe.
6. Usar --overwrite somente quando houver arquivo corrompido ou decisao clara de reprocessar.
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
CHIRPS: 1981-latest_available, diario, 0.25 grau na Fase 1 e 0.05 grau na Fase 2
NOAA OISST: 1981-latest_available, diario, 0.25 grau, SST/SSTA principal
CTD NOAA WOD: 1981-latest_available, perfis irregulares, filtrados no Pacifico
ERA5: 1981-latest_available, subdiario baixado em 4 horarios e agregado depois para diario
ORAS5: 1981-latest_available, mensal, reservado para memoria subsuperficial oceanica
IBGE: cartografia estatica
```
