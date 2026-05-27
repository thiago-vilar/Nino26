# Projeto de previsão climática El NINO e seus impactos no Brasil

**Programa de Pós Graduação em Oceanografia UFPE**  
**Área:** Oceanografia Física  
**Responsável:** Thiago Vilar  

## DATA SOURCES

Este arquivo registra a origem dos dados, a área de download, o uso no projeto e a justificativa científica de cada fonte.

Todo dataset deve registrar:

- instituição.
- link oficial.
- variável.
- período disponível.
- resolução temporal.
- resolução espacial.
- data de download.
- versão do produto.
- licença ou termos de uso.
- caminho bruto.
- caminho processado.

## 1. ORAS/ORAS5

**Instituição:** ECMWF / Copernicus.  
**Download:** https://cds.climate.copernicus.eu/datasets/reanalysis-oras5  
**Referência:** https://www.ecmwf.int/en/forecasts/dataset/ocean-reanalysis-system-5  
**Pasta bruta:** `data/raw/oras/`  
**Pasta processada:** `data/processed/zarr/oras/`  

Uso no projeto:

- temperatura por profundidade.
- salinidade por profundidade.
- correntes oceânicas zonal e meridional, quando disponíveis.
- cálculo de densidade potencial.
- cálculo de `D20`, termoclina, camada de mistura e conteúdo de calor oceânico.

Justificativa:

ORAS/ORAS5 fornece uma representação griddada e contínua do oceano, necessária para acompanhar a memória térmica do Pacífico e a estrutura vertical do aquecimento. É a base principal para variáveis subsuperficiais, especialmente `OHC 0-300 m`, `OHC 0-700 m`, `D20`, termoclina e camada de mistura.

## 2. CTD NOAA / World Ocean Database

**Instituição:** NOAA / National Centers for Environmental Information.  
**Download:** https://www.ncei.noaa.gov/products/world-ocean-database  
**Pasta bruta:** `data/raw/ctd_noaa/`  
**Pasta processada:** `data/processed/parquet/ctd_noaa/`  

Uso no projeto:

- `T(z)`: temperatura observada por profundidade.
- `S(z)`: salinidade observada por profundidade.
- validação da estrutura vertical do ORAS.
- comparação e correção de viés em perfis oceânicos.

Justificativa:

CTD é dado observacional in situ. Ele não substitui o ORAS como campo espacial contínuo, mas é essencial para conferir se a estrutura vertical reanalisada é fisicamente coerente. O projeto usa CTD para avaliar temperatura, salinidade, densidade potencial, termoclina, camada de mistura e conteúdo de calor calculados a partir do ORAS.

## 3. ERA5

**Instituição:** Copernicus Climate Change Service / ECMWF.  
**ERA5 single levels:** https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels  
**ERA5 pressure levels:** https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels  
**Portal CDS:** https://cds.climate.copernicus.eu/  
**Pasta bruta:** `data/raw/era5/`  
**Pasta processada:** `data/processed/zarr/era5/`  

Uso no projeto:

- `SLP`: pressão ao nível do mar.
- `tau_x/tau_y`: tensão de vento zonal e meridional.
- `u10/v10`: vento zonal e meridional a 10 m.
- `u850/v850/q850`: vento e umidade em baixos níveis.
- `u500/v500/q500/z500/omega500`: circulação, umidade, geopotencial e movimento vertical em níveis médios.
- `u200/v200/z200/div200`: circulação, geopotencial e divergência em altos níveis.
- `TCWV`: vapor d'água total integrado.
- fluxos de calor líquido, latente, sensível, onda curta e onda longa.

Justificativa:

ERA5 é a fonte principal para descrever a ponte atmosférica entre o Pacífico aquecido e o Brasil. As variáveis de pressão, vento, umidade, geopotencial, movimento vertical e divergência permitem avaliar o transporte e a reorganização da circulação que podem favorecer seca, chuva abaixo do normal ou chuva acima do normal.

## 4. CPC/NOAA

**Instituição:** NOAA / Climate Prediction Center e NOAA/NCEI.  
**NOAA OISST:** https://www.ncei.noaa.gov/products/optimum-interpolation-sst  
**NOAA OLR:** https://psl.noaa.gov/data/gridded/data.interp_OLR.html  
**CPC Global Precipitation:** https://beta.cpc.noaa.gov/observations/global-precipitation  
**CPC Precipitation Monitoring:** https://www.cpc.ncep.noaa.gov/products/Precip_Monitoring/gl_obs.shtml  
**Pasta bruta:** `data/raw/cpc_noaa/`  
**Pasta processada:** `data/processed/zarr/cpc_noaa/`  

Uso no projeto:

- `SST`: temperatura da superfície do mar.
- `SSTA`: anomalia da temperatura da superfície do mar.
- `OLR`: radiação de onda longa emitida.
- precipitação diária em grade.

Justificativa:

CPC/NOAA fornece produtos observacionais fundamentais para o estado superficial do oceano, convecção tropical e precipitação. A precipitação é a variável-alvo do projeto, enquanto SST/SSTA e OLR ajudam a caracterizar o aquecimento superficial e a convecção associada.

## 5. IBGE

**Instituição:** Instituto Brasileiro de Geografia e Estatística.  
**Geociências / malhas territoriais:** https://www.ibge.gov.br/geociencias/organizacao-do-territorio/malhas-territoriais.html  
**Pasta bruta:** `data/raw/ibge/`  
**Pasta processada:** `data/processed/geotiff/ibge/`  

Uso no projeto:

- limite nacional do Brasil.
- limites estaduais.
- limites regionais.
- recortes territoriais auxiliares.
- máscaras espaciais.
- mapas coropléticos.

Justificativa:

IBGE é a fonte oficial para limites territoriais brasileiros. Esses dados não entram como preditores climáticos, mas são necessários para recortar o Brasil, mascarar pixels fora do território nacional, agregar resultados por unidades territoriais e publicar mapas interpretáveis.

## 6. Fontes complementares de precipitação

Estas fontes podem ser usadas para validação cruzada ou comparação de sensibilidade:

- CHIRPS: https://www.chc.ucsb.edu/data/chirps
- MERGE/INPE: https://ftp.cptec.inpe.br/modelos/tempo/MERGE/
- GPM IMERG: https://gpm.nasa.gov/data/imerg

## 7. Catálogo local

Arquivo:

```text
data/catalog/datasets.yaml
```

Campos mínimos:

```yaml
dataset_id:
  name:
  institution:
  source_url:
  license:
  version:
  variable:
  units:
  temporal_resolution:
  spatial_resolution:
  period_start:
  period_end:
  downloaded_at:
  raw_path:
  processed_path:
  notes:
```
