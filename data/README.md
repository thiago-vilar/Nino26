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
  processed/  Zarr e GeoTIFF prontos para analise
  catalog/    metadados declarativos
  audit/      ledger jsonl de execucao
  state/      estado auxiliar de pipeline
```

## O que existe hoje

- `raw/ibge/`: zips oficiais IBGE de UF e municipios.
- `interim/ibge/`: shapefiles extraidos.
- O status local vivo fica em `painel_executivo.md`, gerado por `scripts/update_painel_executivo.py`.
- As pastas climaticas podem estar em estados diferentes entre maquinas; dados grandes nao entram no Git.

O arquivo CHIRPS `p25` nao deve ser removido. As Fases 1 a 7 usam grade `0.25` grau; `p05` fica para experimento futuro de alta resolucao.

## raw/

Use `raw/` para dados baixados sem modificacao. No fluxo compacto de ERA5/ORAS, o raw pode ser apenas cache temporario e ser apagado depois do Zarr validado.

Subpastas esperadas:

- `raw/chirps/p25/`: CHIRPS diario 0.25 grau das Fases 1 a 7.
- `raw/chirps/p05/`: CHIRPS diario 0.05 grau reservado para experimento futuro de alta resolucao.
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
- `processed/zarr/features/`: Fase 3, diagnosticos fisicos Nino 3.4.
- `processed/zarr/statistics/`: Fase 4, pre-analises estatisticas experimentais.
- `processed/zarr/modeling/`: Fase 5, metricas, previsoes, importancias e pesos por grupo.
- `processed/zarr/distributions/`: diagnosticos de cauda.
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

Para curadoria e retomada automatica do ponto pendente:

```powershell
python scripts\curate_and_resume_downloads.py
python scripts\curate_and_resume_downloads.py --source oisst --stage raw --execute
```

O primeiro comando apenas enumera o que esta OK e o que esta pendente. O segundo executa a proxima pendencia de download do OISST conforme o estado real da maquina consultada.

Para rodar o pipeline completo em piloto automatico, seguindo a ordem das fontes, retomando do ponto valido e gerando relatorio final:

```powershell
python scripts\run_full_download_pipeline.py --execute
```

Este e o comando recomendado quando a intencao for dar apenas um play e deixar o computador baixando de `1981` ate o ultimo dado disponivel por fonte. Ele pula arquivos ja validos, retoma pendencias, muda de fonte na ordem do pipeline e grava log/relatorio em `data/state/pipeline_reports/`.

A ordem operacional agora e fonte por fonte, e dentro de cada fonte por ano. No ERA5, o recorte oficial tem apenas duas regioes: `nino34` e `brazil`.

1. CHIRPS diario;
2. OISST diario;
3. ERA5 subdiario convertido para diario;
4. fontes semanais, quando forem adicionadas ao catalogo;
5. ORAS5 mensal convertido para calendario diario;
6. CTD/WOD observacional irregular.

ERA5 e processado por ano, regiao e tipo (`single_levels` ou `pressure_levels`). O cache bruto fica em `data/raw/era5/` por mes, e o produto pronto fica em `data/processed/zarr/era5/` como um Zarr diario anual por regiao/tipo. ORAS5 continua mensal por variavel por ser fonte oceanica de memoria subsuperficial.

Para ver o plano sem executar:

```powershell
python scripts\run_full_download_pipeline.py
```

Para testar uma janela pequena antes de deixar a maquina trabalhando por horas:

```powershell
python scripts\run_full_download_pipeline.py --end-year 1981 --month 1
```

O pipeline completo roda IBGE, CHIRPS, OISST, ERA5, ORAS5, CTD/WOD, conversao diaria para Zarr, regrid e atualizacao do painel. Por padrao, ele continua depois de falhas individuais e registra o que falhou no relatorio final. Use `--skip-cds` ou `--skip-ctd` se quiser pular temporariamente ERA5/ORAS5 ou CTD.

5. Baixar ERA5/ORAS5/CTD quando houver credenciais e espaco:

```powershell
python scripts\data_pipeline.py check-cds
python scripts\data_pipeline.py download-era5 --start-year 1981 --kind both --region nino34 --region brazil --annual-zarr --delete-raw-after-zarr --execute --continue-on-error
python scripts\data_pipeline.py download-oras --start-year 1981 --annual-zarr --delete-raw-after-zarr --execute --continue-on-error
python scripts\data_pipeline.py etl-ctd --start-year 1981 --max-depth 300 --min-levels 3 --execute --continue-on-error
```

Para testar uma unica variavel CDS sem iniciar tudo:

```powershell
python scripts\data_pipeline.py download-era5 --start-year 1981 --end-year 1981 --month 1 --kind single --region nino34
python scripts\data_pipeline.py download-oras --start-year 1981 --end-year 1981 --month 1 --variable salinity
```

Sem `--execute`, esses comandos apenas mostram o cache bruto e o Zarr diario que seriam gerados.

6. Regridar cubos para modelagem:

```powershell
python scripts\data_pipeline.py regrid-zarr --input <zarr_entrada> --output data/processed/zarr/regridded/<nome>.zarr --dataset <dataset>
```

7. Rodar diagnostico distribucional quando util:

```powershell
python scripts\data_pipeline.py diagnose-distributions --input <zarr_regridded> --dataset <dataset> --variable <variavel>
```

8. Rodar Fase 3 e Fase 4 quando os cubos de entrada estiverem fechados:

```text
Fase 3: gerar diagnosticos fisicos Nino 3.4 em processed/zarr/features/
Fase 4: gerar triagem estatistica experimental em processed/zarr/statistics/
```

9. Rodar modelagem da Fase 5:

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
- Stores Zarr de metricas/importancias sem antes registrar qual rodada eles representam.

## Checklist antes de modelar

- `configs/project.yaml` esta em `period.end: latest_available`.
- `modeling.chirps_resolution` esta em `p25` para as Fases 1 a 7.
- OISST foi tratado como SST/SSTA principal.
- ORAS5 nao esta duplicando SST mensal.
- Todos os Zarr de entrada estao regridados.
- O ledger tem registro de latencia por fonte.
- `python -m unittest discover -s tests -v` passa.
