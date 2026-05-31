# NINO-BRASIL

Projeto Python para medir o peso relativo de variaveis oceanograficas e atmosfericas associadas ao aquecimento do Pacifico sobre anomalias diarias de precipitacao no Brasil.

O fluxo oficial preserva a janela historica completa: `1981-01-01` ate o ultimo dado disponivel por fonte. O fim do periodo e dinamico porque CHIRPS, OISST, ERA5, ORAS5 e CTD/WOD tem latencias diferentes.

## Decisoes metodologicas fixas

- Frequencia de modelagem: sempre diaria.
- Lags: `0, 30, 60, 90, 120, 150, 180` dias.
- Validacao: somente blocos temporais e walk-forward; nao usar split aleatorio.
- Anti-vazamento: climatologia diaria, desvio padrao, P10 e P90 sao estimados no bloco de treino de cada fold e reaplicados a validacao/teste.
- Grade comum da Fase 1: `0.25` grau, longitude `0_360`, declarada em `configs/project.yaml`.
- CHIRPS da Fase 1: `p25`, coerente com a grade comum e com o arquivo local ja baixado.
- CHIRPS `p05`: reservado para Fase 2, quando o custo pixel-a-pixel for validado.
- SST/SSTA principal: NOAA OISST diario.
- ORAS5: reservado para memoria oceanica subsuperficial, nao para duplicar SST mensal.

## Estado local dos dados

Atualmente ha IBGE e um arquivo CHIRPS `p25` de 1981 em `data/raw/chirps/p25/`. Esse arquivo nao e erro: ele esta alinhado com a escolha oficial da Fase 1. Nao apague `p25` salvo se o arquivo estiver corrompido.

Dados que ainda precisam ser ingeridos para rodar ponta a ponta:

- NOAA OISST historico diario.
- CHIRPS `p25` historico diario.
- ERA5 single e pressure levels.
- ORAS5 subsuperficial.
- CTD/WOD para validacao vertical.
- Cubos Zarr regridados em `data/processed/zarr/regridded/`.

Arquivos grandes em `data/raw/`, `data/interim/` e `data/processed/` nao devem ser commitados. Versione codigo, configs, catalogo, docs e testes.

## Instalacao

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Teste rapido:

```powershell
.\.venv\Scripts\python .\scripts\smoke_test.py
python -m unittest discover -s tests -v
```

## Ordem do pipeline

Use `plan` antes de executar downloads longos:

```powershell
python scripts\data_pipeline.py plan
python scripts\data_pipeline.py status
```

### 1. Estrutura local

```powershell
python scripts\data_pipeline.py init
```

### 2. IBGE

```powershell
python scripts\data_pipeline.py download-ibge --product uf
python scripts\data_pipeline.py download-ibge --product municipios
```

Saidas esperadas:

- `data/raw/ibge/*.zip`
- `data/interim/ibge/BR_UF_2024/`
- `data/interim/ibge/BR_Municipios_2024/`

### 3. CHIRPS Fase 1

Dry-run:

```powershell
python scripts\data_pipeline.py download-chirps --start-year 1981 --resolution p25
```

Execucao real ate o ultimo ano disponivel pela regra de latencia:

```powershell
python scripts\data_pipeline.py download-chirps --start-year 1981 --resolution p25 --execute
```

Para rodar ano a ano:

```powershell
for ($y = 1981; $y -le (Get-Date).Year; $y++) {
  python scripts\data_pipeline.py download-chirps --start-year $y --end-year $y --resolution p25 --execute
  python scripts\data_pipeline.py audit
}
```

### 4. OISST

```powershell
python scripts\data_pipeline.py download-oisst --start-year 1981 --execute
```

OISST e a fonte oficial de SST/SSTA diaria. Depois do download, o cubo precisa ser padronizado e regridado antes da modelagem.

### 5. ERA5 e ORAS5

Verifique credenciais CDS:

```powershell
python scripts\data_pipeline.py check-cds
```

ERA5:

```powershell
python scripts\data_pipeline.py download-era5 --start-year 1981 --kind both --execute
```

ORAS5:

```powershell
python scripts\data_pipeline.py download-oras --start-year 1981 --execute
```

Os comandos CDS geram Zarr mensal quando executados sem `--raw-only`.

### 6. CTD/WOD

```powershell
python scripts\data_pipeline.py download-ctd --start-year 1981 --execute
```

Para baixar bruto primeiro:

```powershell
python scripts\data_pipeline.py download-ctd --start-year 1981 --raw-only --execute
python scripts\data_pipeline.py etl-ctd --start-year 1981 --execute
```

### 7. Regridding para a grade comum

Todo cubo que entrar em `model_pipeline.py` deve estar reconciliado para a grade de `configs/project.yaml`.

Exemplo:

```powershell
python scripts\data_pipeline.py regrid-zarr `
  --input data/processed/zarr/noaa_oisst.zarr `
  --output data/processed/zarr/regridded/noaa_oisst.zarr `
  --dataset noaa_oisst `
  --dry-run
```

Remova `--dry-run` quando o caminho de entrada existir e estiver validado.

### 8. Diagnostico de distribuicoes

Use em recortes ou cubos manejaveis para caracterizar caudas de chuva, extremos, OHC e SSTA:

```powershell
python scripts\data_pipeline.py diagnose-distributions `
  --input data/processed/zarr/regridded/brazil_precipitation.zarr `
  --dataset chirps_precipitation_p25 `
  --variable precip `
  --tail upper
```

Saida:

- `data/processed/parquet/distributions/distribution_diagnostics.parquet`
- evento correspondente em `data/audit/ledger.jsonl`

O diagnostico estima `alpha`, `xmin`, distancia KS e compara power law contra lognormal/exponencial. A comparacao e obrigatoria porque chuva e OHC podem preferir lognormal.

### 9. Modelagem walk-forward

Dry-run:

```powershell
python scripts\model_pipeline.py `
  --predictor-zarr data/processed/zarr/regridded/noaa_oisst.zarr `
  --target-zarr data/processed/zarr/regridded/brazil_precipitation.zarr `
  --target-var precip `
  --dry-run
```

Execucao Fase 1 agregada:

```powershell
python scripts\model_pipeline.py `
  --predictor-zarr data/processed/zarr/regridded/noaa_oisst.zarr `
  --target-zarr data/processed/zarr/regridded/brazil_precipitation.zarr `
  --target-var precip `
  --predictor-strategy mean `
  --target-strategy mean
```

Saidas:

- `data/processed/parquet/modeling/walk_forward_metrics.parquet`
- `data/processed/parquet/modeling/walk_forward_predictions.parquet`
- `data/processed/parquet/modeling/walk_forward_importances.parquet`
- `data/processed/parquet/modeling/walk_forward_group_weights.parquet`

Para SHAP em modelos de arvore:

```powershell
python scripts\model_pipeline.py ... --shap --shap-max-rows 1000
```

### 10. Validacao pixel-a-pixel

`flatten` e Fase 2. Antes de escalar para todo o Brasil, rode em recorte pequeno:

```powershell
python scripts\model_pipeline.py `
  --predictor-zarr data/processed/zarr/regridded/noaa_oisst_small.zarr `
  --target-zarr data/processed/zarr/regridded/brazil_precipitation_small.zarr `
  --target-var precip `
  --predictor-strategy flatten `
  --target-strategy flatten `
  --regression-model ridge `
  --classification-model logistic `
  --importance-repeats 2
```

Registre tempo, memoria, numero de features, numero de pixels e tamanho dos Parquets antes de expandir.

## Auditoria e limpeza segura

Resumo do ledger:

```powershell
python scripts\data_pipeline.py audit
```

Pode apagar:

- arquivos `.part` comprovadamente interrompidos, se o mesmo comando sera reexecutado;
- Zarr de saida corrompido, usando depois o mesmo comando com `--overwrite`;
- dados que nao correspondam a `configs/project.yaml`, apos confirmar que ha copia ou fonte para baixar novamente.

Nao apagar:

- `data/raw/chirps/p25/chirps-v2.0.1981.days_p25.nc`, pois e valido na Fase 1;
- shapefiles IBGE ja extraidos;
- `.gitkeep`, pois mantem a estrutura versionada.

## Versionamento recomendado

Antes de commit:

```powershell
python -m unittest discover -s tests -v
python -m compileall src scripts tests
git status --short
```

Commitar codigo, testes, configs e documentacao. Nao commitar dados brutos/intermediarios/processados grandes.
