# Data README

Este diretorio guarda apenas dados locais do pipeline. Dados grandes nao entram no Git; a estrutura de pastas e mantida por `.gitkeep`.

## Regra de ouro

O dado so esta pronto para modelagem quando:

1. foi baixado ou extraido sem `.part`;
2. aparece no catalogo em `data/catalog/datasets.yaml`;
3. tem cobertura temporal registrada no ledger;
4. foi convertido para Zarr quando multidimensional;
5. foi normalizado para a grade comum antes de entrar em `scripts/model_pipeline.py`.

## Estrutura

```text
data/
  raw/        arquivos exatamente como vieram da fonte
  interim/    extracoes e padronizacoes intermediarias
  processed/  Zarr, Parquet e GeoTIFF prontos para analise
  catalog/    metadados declarativos
  audit/      ledger jsonl de execucao
  state/      estado auxiliar de pipeline
```

## O que existe hoje

- `raw/ibge/`: zips oficiais IBGE de UF e municipios.
- `interim/ibge/`: shapefiles extraidos.
- `raw/chirps/p25/chirps-v2.0.1981.days_p25.nc`: valido para a Fase 1.
- Demais pastas climaticas ainda estao vazias ou so com `.gitkeep`.

O arquivo CHIRPS `p25` nao deve ser removido. A Fase 1 usa grade `0.25` grau; `p05` fica para Fase 2.

## raw/

Use `raw/` para dados baixados sem modificacao.

Subpastas esperadas:

- `raw/chirps/p25/`: CHIRPS diario 0.25 grau da Fase 1.
- `raw/chirps/p05/`: CHIRPS diario 0.05 grau da Fase 2.
- `raw/cpc_noaa/oisst/`: NOAA OISST diario.
- `raw/era5/`: ERA5 single e pressure levels.
- `raw/oras/`: ORAS5 mensal.
- `raw/ctd_noaa/wod/`: WOD CTD anual.
- `raw/ibge/`: zips oficiais do IBGE.

Nao edite arquivos em `raw/`. Se um arquivo estiver corrompido, apague apenas esse arquivo ou use `--overwrite` no comando de origem.

## interim/

Use `interim/` para extracao, filtros e padronizacoes que ainda nao sao produto final.

Exemplos:

- `interim/ibge/BR_UF_2024/`
- `interim/ibge/BR_Municipios_2024/`
- `interim/oras/<ano>/`
- `interim/ctd_noaa/`

Produtos em `interim/` podem ser refeitos a partir de `raw/`.

## processed/

Use `processed/` para artefatos prontos para analise:

- `processed/zarr/`: cubos multidimensionais.
- `processed/zarr/regridded/`: cubos na grade comum de modelagem.
- `processed/parquet/modeling/`: metricas, previsoes, importancias e pesos por grupo.
- `processed/parquet/distributions/`: diagnosticos de cauda.
- `processed/geotiff/`: mascaras e rasters auxiliares.

`model_pipeline.py` deve consumir preferencialmente `processed/zarr/regridded/`.

## Catalogo

Arquivo principal:

```text
data/catalog/datasets.yaml
```

Ao incluir uma fonte, registre:

- nome e instituicao;
- URL de origem;
- variaveis;
- resolucao temporal e espacial;
- `period_start`;
- `period_end: latest_available`;
- `latency_days`;
- caminhos `raw_path` e `processed_path`;
- nota de uso metodologico.

## Auditoria

Ledger:

```text
data/audit/ledger.jsonl
```

Ver resumo:

```powershell
python scripts\data_pipeline.py audit
```

Cada download/ETL critico deve registrar `status`, `dataset`, caminho de entrada/saida e latencia quando aplicavel.

## Ordem operacional dos dados

1. Inicializar estrutura:

```powershell
python scripts\data_pipeline.py init
```

2. Baixar IBGE:

```powershell
python scripts\data_pipeline.py download-ibge --product uf
python scripts\data_pipeline.py download-ibge --product municipios
```

3. Baixar CHIRPS Fase 1:

```powershell
python scripts\data_pipeline.py download-chirps --start-year 1981 --resolution p25 --execute
```

4. Baixar OISST:

```powershell
python scripts\data_pipeline.py download-oisst --start-year 1981 --execute
```

5. Baixar ERA5/ORAS5/CTD quando houver credenciais e espaco:

```powershell
python scripts\data_pipeline.py check-cds
python scripts\data_pipeline.py download-era5 --start-year 1981 --kind both --execute
python scripts\data_pipeline.py download-oras --start-year 1981 --execute
python scripts\data_pipeline.py download-ctd --start-year 1981 --execute
```

6. Regridar cubos para modelagem:

```powershell
python scripts\data_pipeline.py regrid-zarr --input <zarr_entrada> --output data/processed/zarr/regridded/<nome>.zarr --dataset <dataset>
```

7. Rodar diagnostico distribucional quando util:

```powershell
python scripts\data_pipeline.py diagnose-distributions --input <zarr_regridded> --dataset <dataset> --variable <variavel>
```

8. Rodar modelagem:

```powershell
python scripts\model_pipeline.py --predictor-zarr <zarr_predictor> --target-zarr <zarr_alvo> --target-var precip
```

## Limpeza

Pode limpar:

- `.part` de download interrompido;
- Zarr de saida corrompido antes de reprocessar;
- produtos `interim/` que possam ser refeitos;
- dados fora da resolucao declarada em `configs/project.yaml`, se nao forem parte de uma fase futura.

Nao limpar:

- CHIRPS `p25` existente;
- IBGE bruto ou extraido;
- `.gitkeep`;
- Parquets de metricas/importancias sem antes registrar qual rodada eles representam.

## Checklist antes de modelar

- `configs/project.yaml` esta em `period.end: latest_available`.
- `modeling.chirps_resolution` esta em `p25` para Fase 1.
- OISST foi tratado como SST/SSTA principal.
- ORAS5 nao esta duplicando SST mensal.
- Todos os Zarr de entrada estao regridados.
- O ledger tem registro de latencia por fonte.
- `python -m unittest discover -s tests -v` passa.
