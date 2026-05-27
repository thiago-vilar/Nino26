# Projeto de previsão climática El NINO e seus impactos no Brasil

**Programa de Pós Graduação em Oceanografia UFPE**  
**Área:** Oceanografia Física  
**Responsável:** Thiago Vilar  

## DATA SOURCES

### Princípio de rastreabilidade

Toda variável usada no projeto deve registrar:

- nome da fonte;
- instituição;
- link oficial;
- variável;
- período disponível;
- resolução temporal;
- resolução espacial;
- data de download;
- versão do produto;
- licença ou termos de uso;
- caminho local bruto;
- caminho local processado.

O período-alvo é 1980 até a data presente, mas cada dataset deve declarar sua cobertura real.

---

## 1. ERA5

Uso no projeto:

- SLP: pressão ao nível do mar.
- tau_x: tensão de vento zonal na superfície.
- tau_y: tensão de vento meridional na superfície.
- u10 e v10: vento zonal e meridional a 10 m.
- u850, v850, q850.
- u500, v500, q500, z500, omega500.
- u200, v200, z200, div200.
- TCWV: vapor d'água total integrado.
- fluxos de calor: líquido, latente, sensível, onda curta e onda longa.
- OLR, se for necessário manter consistência com a mesma reanálise.

Instituição:

- Copernicus Climate Change Service / ECMWF.

Links oficiais:

- ERA5 single levels: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
- ERA5 pressure levels: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels
- Portal CDS: https://cds.climate.copernicus.eu/

Cobertura:

- ERA5 cobre o período necessário para 1980-presente.

Formato esperado:

- NetCDF ou GRIB bruto.
- Zarr processado.

Termos de uso:

- Licença Copernicus. Registrar aceite e versão da licença no catálogo local.

---

## 2. ORAS/ORAS5

Uso no projeto:

- temperatura por profundidade;
- salinidade por profundidade;
- densidade potencial calculada;
- D20 calculado;
- profundidade da termoclina calculada;
- camada de mistura calculada;
- conteúdo de calor oceânico 0-300 m;
- conteúdo de calor oceânico 0-700 m;
- correntes oceânicas u e v, quando disponíveis.

Instituição:

- ECMWF / Copernicus.

Links oficiais:

- ORAS5 CDS: https://cds.climate.copernicus.eu/datasets/reanalysis-oras5
- ECMWF ORAS5: https://www.ecmwf.int/en/forecasts/dataset/ocean-reanalysis-system-5

Cobertura:

- ORAS5 possui série histórica longa e será recortado para 1980-presente.

Formato esperado:

- NetCDF bruto.
- Zarr processado.

Termos de uso:

- Licença Copernicus. Registrar termos no catálogo.

---

## 3. CPC/NOAA

Uso no projeto:

- SST/SSTA do Pacífico.
- OLR observacional.
- precipitação diária para o Brasil, quando usada fonte CPC/NOAA.
- produtos oceanográficos NOAA/NCEI complementares.

Instituição:

- NOAA / Climate Prediction Center.
- NOAA / National Centers for Environmental Information.

Links oficiais:

- NOAA OISST: https://www.ncei.noaa.gov/products/optimum-interpolation-sst
- NOAA Physical Sciences Laboratory OLR: https://psl.noaa.gov/data/gridded/data.interp_OLR.html
- CPC Global Precipitation: https://beta.cpc.noaa.gov/observations/global-precipitation
- CPC Precipitation Monitoring: https://www.cpc.ncep.noaa.gov/products/Precip_Monitoring/gl_obs.shtml

Cobertura:

- OISST v2.1 inicia em 1981-09-01. A lacuna 1980-1981 deve ser registrada ou preenchida com fonte alternativa documentada.
- Produtos CPC/NOAA variam por produto. Registrar cobertura real no catálogo.

Formato esperado:

- NetCDF, GRIB, ASCII ou binário conforme produto.
- Converter para NetCDF/Zarr padronizado.

Termos de uso:

- Produtos NOAA são, em geral, públicos, mas a página de cada produto deve ser citada e registrada.

---

## 4. CTD / World Ocean Database

Uso no projeto:

- T(z): temperatura por profundidade.
- S(z): salinidade por profundidade.
- validação da estrutura vertical;
- comparação com ORAS/ORAS5.

Instituição:

- NOAA / National Centers for Environmental Information.

Link oficial:

- https://www.ncei.noaa.gov/products/world-ocean-database

Cobertura:

- Histórica, variável por campanha e região.

Formato esperado:

- NetCDF, CSV ou formato WOD.
- Converter para tabela padronizada.

Termos de uso:

- Registrar citação e termos indicados pelo NOAA/NCEI.

---

## 5. Fontes complementares de precipitação

### CHIRPS

Link:

- https://www.chc.ucsb.edu/data/chirps

Uso:

- precipitação diária em grade.

### MERGE/INPE

Link:

- https://ftp.cptec.inpe.br/modelos/tempo/MERGE/

Uso:

- precipitação diária nacional.

### GPM IMERG

Link:

- https://gpm.nasa.gov/data/imerg

Uso:

- precipitação subdiária/diária para eventos recentes e extremos.

---

## 6. Catálogo local esperado

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

