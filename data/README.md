# Data README

Este diretorio guarda apenas dados locais do pipeline. Dados grandes nao entram no Git; a estrutura de pastas e mantida por `.gitkeep`.

## Regra De Ouro

O dado so esta pronto para o diagnostico fisico da Fase 3 quando:

1. foi baixado ou extraido sem `.part`;
2. aparece no catalogo em `data/catalog/datasets.yaml`;
3. tem cobertura temporal registrada no ledger;
4. foi convertido para Zarr quando multidimensional;
5. foi normalizado para a grade comum quando a analise exigir comparacao espacial.

O escopo ativo encerra na Fase 3. A referencia ENSO mensal, os eventos e os
picos P90 metricos sao derivados da propria SST/SSTA OISST baixada. Graficos
oficiais do Nino 3.4 podem ser mantidos em `docs/assets` apenas como comparativo
visual externo.

## Estrutura

```text
data/
  raw/        arquivos exatamente como vieram da fonte
  interim/    extracoes e padronizacoes intermediarias
  processed/  Zarr, CSV e GeoTIFF prontos para analise
  catalog/    metadados declarativos
  audit/      ledger jsonl de execucao
  state/      estado auxiliar de pipeline
```

## O Que Existe Hoje

- `raw/ibge/`: zips oficiais IBGE de UF e municipios.
- `interim/ibge/`: shapefiles extraidos.
- `raw/cpc_noaa/oisst/`: SST diaria OISST baixada.
- `processed/zarr/features/`: diagnosticos fisicos Nino 3.4 da Fase 3.
- `processed/parquet/features/`: tabelas CSV auditaveis da Fase 3.
- O status local vivo fica em `painel_executivo.md`, gerado por `scripts/update_painel_executivo.py`.
- As pastas climaticas podem estar em estados diferentes entre maquinas; dados grandes nao entram no Git.

O arquivo CHIRPS `p25` nao deve ser removido. Ele pode permanecer em disco, mas chuva no Brasil nao entra no escopo ativo da Fase 3.

## raw/

Use `raw/` para dados baixados sem modificacao. No fluxo compacto de ERA5/GLORYS12, o raw pode ser apenas cache temporario e ser apagado depois do Zarr validado.

Subpastas esperadas:

- `raw/chirps/p25/`: CHIRPS diario 0.25 grau mantido como dado local.
- `raw/chirps/p05/`: CHIRPS diario 0.05 grau reservado.
- `raw/cpc_noaa/oisst/`: OISST diario.
- `raw/era5/`: ERA5 single e pressure levels.
- `raw/ocean_daily/glorys12/`: cache Zarr nativo diario do GLORYS12.
- `raw/ocean_monthly/oras5/`: cache ZIP/NetCDF mensal agrupado por resolucao vertical.
- `raw/ctd_noaa/wod/`: WOD CTD anual.
- `raw/ibge/`: zips oficiais do IBGE.

Nao edite arquivos em `raw/`. Se um arquivo estiver corrompido, apague apenas esse arquivo ou use `--overwrite` no comando de origem.

## processed/

Use `processed/` para artefatos prontos para analise:

- `processed/zarr/`: cubos multidimensionais.
- `processed/zarr/regridded/`: cubos na grade comum 0.25 grau.
- `processed/zarr/features/`: artefatos da Fase 3.
- `processed/zarr/distributions/`: diagnosticos de cauda opcionais.
- `processed/geotiff/`: mascaras e rasters auxiliares.

Produtos centrais da Fase 3:

```text
data/processed/parquet/features/nino34_daily_oisst.csv
data/processed/parquet/features/nino34_monthly_oisst.csv
data/processed/parquet/features/nino34_oisst_event_reference.csv
data/processed/parquet/features/nino34_oisst_p90_peaks.csv
data/processed/parquet/features/nino34_physical_signal.csv
data/processed/parquet/features/nino34_event_table_monthly.csv
data/processed/parquet/features/nino34_peak_comparison.csv
data/processed/parquet/physics_precalc_timeseries.csv
```

Comparativo visual oficial, fora de `data/processed`:

```text
docs/assets/figures/official_nino34/noaa_psl_nino34_timeseries.png
docs/assets/figures/official_nino34/noaa_psl_nino34_event_panel.png
docs/assets/figures/official_nino34/official_nino34_visuals_manifest.json
```

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

## Ordem Operacional Dos Dados

1. Inicializar estrutura:

```powershell
python scripts\data_pipeline.py init
```

2. Baixar IBGE:

```powershell
python scripts\data_pipeline.py download-ibge --product uf
python scripts\data_pipeline.py download-ibge --product municipios
```

3. Baixar OISST:

```powershell
python scripts\data_pipeline.py download-oisst --start-year 1981 --execute
```

4. Montar o indice diario Nino 3.4 com a OISST baixada:

```powershell
python scripts\data_pipeline.py build-nino34-daily-index
```

5. Gerar referencia mensal, eventos locais e P90 a partir da propria OISST:

```powershell
python scripts\data_pipeline.py build-nino34-sst-reference
python scripts\data_pipeline.py build-nino34-p90-peaks
python scripts\data_pipeline.py sync-official-nino34-visuals
```

6. Rodar a Fase 3:

```powershell
python scripts\data_pipeline.py build-phase3-diagnostics
python scripts\data_pipeline.py audit-phase3-diagnostics
python scripts\update_painel_executivo.py
```

Para curadoria e retomada automatica do ponto pendente:

```powershell
python scripts\curate_and_resume_downloads.py
python scripts\curate_and_resume_downloads.py --source oisst --stage raw --execute
```

## Limpeza

Pode limpar:

- `.part` de download interrompido;
- Zarr de saida corrompido antes de reprocessar;
- produtos `interim/` que possam ser refeitos;
- dados fora da resolucao declarada em `configs/project.yaml`, se nao forem parte do diagnostico atual.

Nao limpar:

- OISST diario baixado;
- CHIRPS `p25` existente;
- IBGE bruto ou extraido;
- `.gitkeep`;
- produtos numericos da Fase 3 sem antes registrar qual rodada representam.

## Checklist Da Fase 3

- `configs/project.yaml` esta em `period.end: latest_available`.
- OISST diario foi tratado como SST/SSTA principal.
- `nino34_monthly_oisst.csv` existe e foi derivado de `nino34_daily_oisst.csv`.
- Eventos e picos P90 usam apenas a SST/SSTA local.
- Graficos oficiais Nino 3.4, quando presentes, sao apenas comparativo visual e ficam fora de `data/processed`.
- O oceano subsuperficial nao esta duplicando nem promovendo SST mensal.
- Todos os Zarr de entrada que exigem grade comum estao regridados.
- O ledger tem registro de latencia por fonte.
- `python -m unittest discover -s tests -v` passa.
