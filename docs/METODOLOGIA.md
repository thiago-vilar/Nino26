# Metodologia Ativa

## Escopo

O NINO-BRASIL fica restrito, neste recorte, as Fases 1 a 3:

1. Ingestao auditavel das fontes locais.
2. Padronizacao, anomalias, Zarr e grade comum quando necessaria.
3. Diagnostico fisico do Nino 3.4.

Nao ha etapa ativa de teleconexao Brasil, modelagem estatistica, ML, redes
neurais ou rotulo ENSO externo.

Graficos oficiais NOAA/PSL do Nino 3.4 podem ser mantidos no repositorio como
comparativo visual externo. Eles nao definem eventos, nao entram no P90, nao
alimentam diagnosticos e nao substituem a SSTA OISST calculada localmente.

## Fonte ENSO

A fonte de verdade para superficie e a SST diaria OISST ja baixada no projeto.
O indice diario Nino 3.4 e calculado por media ponderada por latitude na caixa
5S-5N, 170W-120W. A anomalia usa climatologia fixa local 1991-2020, aplicada
sobre a propria serie OISST.

Produtos:

```text
data/processed/parquet/features/nino34_daily_oisst.csv
data/processed/zarr/features/nino34_daily_oisst.zarr
```

## Eventos E Picos

Eventos mensais e picos dependem apenas da OISST local. O procedimento ativo e:

1. Agregar `nino34_daily_oisst.csv` para media mensal.
2. Calcular media movel trimestral da SSTA mensal.
3. Identificar eventos quentes por SSTA trimestral local >= 0,5 C por pelo menos 5 meses.
4. Calcular P90 sobre a SSTA mensal OISST valida e destacar o pico de cada sequencia acima desse limiar.

Produtos:

```text
data/processed/parquet/features/nino34_monthly_oisst.csv
data/processed/zarr/features/nino34_monthly_oisst.zarr
data/processed/parquet/features/nino34_oisst_event_reference.csv
data/processed/zarr/features/nino34_oisst_event_reference.zarr
data/processed/parquet/features/nino34_oisst_p90_peaks.csv
data/processed/zarr/features/nino34_oisst_p90_peaks.zarr
docs/assets/figures/nino34_oisst_p90_peaks.png
```

## Comparativo Visual Oficial

O comando abaixo espelha graficos oficiais NOAA/PSL do Nino 3.4 em `docs/assets`
para comparacao visual no parecer:

```powershell
.\.venv\Scripts\python scripts\data_pipeline.py sync-official-nino34-visuals
```

Produtos:

```text
docs/assets/figures/official_nino34/noaa_psl_nino34_timeseries.png
docs/assets/figures/official_nino34/noaa_psl_nino34_event_panel.png
docs/assets/figures/official_nino34/official_nino34_visuals_manifest.json
```

Regra: esses PNGs sao `visual_reference_only`; a metrica continua sendo
`nino34_monthly_oisst.csv`.

## Diagnostico Fisico Da Fase 3

A Fase 3 cruza a trajetoria diaria OISST Nino 3.4 com variaveis oceanicas
diarias ja processadas:

- SSTA, medias moveis e tendencias.
- D20 e tendencia da termoclina.
- OHC por camadas.
- WWV.
- SSH quando disponivel.
- Tilt da termoclina e slope.

O objetivo e produzir um parecer fisico auditavel sobre formacao, memoria,
persistencia e pico do sinal Nino 3.4. O resultado nao e um preditor para ML; e
uma caracterizacao fisica.

Produtos:

```text
data/processed/parquet/features/nino34_physical_signal.csv
data/processed/zarr/features/nino34_physical_signal.zarr
data/processed/zarr/features/nino34_thermocline_diagnostics.zarr
data/processed/zarr/features/nino34_peak_signal_comparison.zarr
data/processed/zarr/features/nino34_signal_slope_duration.zarr
data/processed/parquet/features/nino34_event_table_monthly.csv
data/processed/parquet/features/nino34_peak_comparison.csv
data/processed/parquet/physics_precalc_timeseries.csv
data/audit/phase3_diagnostics_audit.json
```

## Regras De Qualidade

- Nao misturar datasets de indice: OISST diario gera o diario, o mensal, os eventos e o P90.
- Graficos oficiais do Nino 3.4 sao comparativo visual externo, nao fonte de metrica.
- Nao promover fonte mensal para dado diario.
- Declarar janela real de subsuperficie por fonte.
- Toda figura analitica do projeto precisa de CSV/Zarr anterior.
- A auditoria da Fase 3 precisa passar com `errors=0`.
