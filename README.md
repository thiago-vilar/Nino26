# NINO-BRASIL

Projeto Python para medir o peso relativo de variaveis oceanograficas e atmosfericas associadas ao aquecimento do Pacifico sobre anomalias diarias de precipitacao no Brasil.

O fluxo oficial preserva a janela historica completa: `1981-01-01` ate o ultimo dado disponivel por fonte. O fim do periodo e dinamico porque CHIRPS, OISST, ERA5, ORAS5 e CTD/WOD tem latencias diferentes.

## Decisoes metodologicas fixas

- Frequencia de modelagem: sempre diaria.
- Horizontes de previsao: semanal, de 1 a 24 semanas (`7` a `168` dias).
- Validacao: somente blocos temporais e walk-forward; nao usar split aleatorio.
- Anti-vazamento: climatologia diaria, desvio padrao, P10, P25, P75 e P90 sao estimados no bloco de treino de cada fold e reaplicados a validacao/teste.
- Grade comum das Fases 1 a 7: `0.25` grau, declarada em `configs/project.yaml`, com foco em duas regioes: `nino34` e `brazil`.
- CHIRPS oficial das Fases 1 a 7: `p25`, coerente com a grade comum.
- CHIRPS `p05`: reservado para experimento futuro de alta resolucao, depois que o fluxo `0.25` grau estiver validado.
- SST/SSTA principal: NOAA OISST diario.
- ORAS5: reservado para memoria oceanica subsuperficial, nao para duplicar SST mensal.
- TAO/TRITON/Argo: camada auxiliar de validacao in situ; nao substitui OISST/ORAS5/CTD-WOD e nao decide antecipadamente se a subsuperficie melhora a resposta de chuva.

Hipotese fisica regional a testar: El Nino alto tende a aumentar o risco de seca no Nordeste/Semiarido e chuva acima do normal no Sul do Brasil. O projeto deve medir a intensidade, o lag dominante e a confianca desse sinal por pixel, regiao e estacao.

Classes de precipitacao por percentis locais:

- `<= P10`: seco extremo.
- `<= P25`: abaixo do quartil inferior, incluindo seca moderada.
- `P25-P75`: faixa interquartil de referencia, sem ser rotulada como "normal".
- `>= P75`: acima do quartil superior, incluindo chuva moderadamente elevada.
- `>= P90`: umido extremo.

## Painel de fases

| Fase | Foco | Marco de conclusao |
|---|---|---|
| Fase 1 | Fundacao local, estrutura Git, catalogo, scripts de ingestao e base diaria inicial | Dados essenciais baixaveis/reprodutiveis e estrutura versionada |
| Fase 2 | Calibracao, padronizacao, anomalias, lags e regridding para a grade comum `0.25` grau | Cubos Zarr reconciliados em `data/processed/zarr/regridded/` |
| Fase 3 | Diagnostico fisico Nino 3.4: alinhamento de anomalias, volume/grau de termoclina, slope, duracao de sinal e picos historicos de El Nino | Feature store fisico com variaveis de superficie/subsuperficie, slope e duracao de sinal |
| Fase 4 | Pre-analises estatisticas experimentais: regressao multipla, PCA, KNN, correlacoes defasadas e triagem de variaveis combinadas | Ranking experimental das variaveis Nino 3.4 mais associadas a seca no Nordeste e chuva no Sul |
| Fase 5 | Machine learning classico e XAI: Ridge, Random Forest, XGBoost/LightGBM, walk-forward, permutation importance, SHAP e pesos por grupo | Stores Zarr de metricas, previsoes, importancias, pesos por grupo e mapas analiticos |
| Fase 6 | Redes neurais, XAI e memoria experimental: CNN, ConvLSTM, U-Net, Transformer e Memory Caching | Stores Zarr de treino/inferencia neural, explicabilidade neural e comparacao contra fases anteriores |
| Fase 7 | Publicacao, operacao recorrente, experimentos `p05` e relatorios automaticos | Produto em `docs/`, rotina recorrente e painel publico |

## Estado local dos dados

O status local vivo deve ser consultado pelo painel executivo gerado em `painel_executivo.md`. Esse arquivo e local, ignorado pelo Git, e pode diferir entre maquinas.

Dados que ainda precisam ser ingeridos para encerrar a base da Fase 1 e entrar com seguranca na Fase 2:

- NOAA OISST historico diario.
- CHIRPS `p25` historico diario.
- ERA5 single e pressure levels.
- ORAS5 subsuperficial.
- CTD/WOD para validacao vertical.
- TAO/TRITON/Argo como validacao complementar da subsuperficie no Pacifico equatorial.
- Cubos Zarr regridados em `data/processed/zarr/regridded/`.

Arquivos grandes em `data/raw/`, `data/interim/` e `data/processed/` nao devem ser commitados. Versione codigo, configs, catalogo, docs e testes.

Para trabalhar em mais de uma maquina, use o GitHub como fonte do codigo: antes de iniciar, rode `git pull --ff-only origin main`; antes de trocar de maquina, rode os testes, faca commit e `git push origin main`. Dados grandes e `.env` ficam locais em cada maquina.

## Checklist ECMWF resumido

O projeto ja esta alinhado com ERA5/ORAS5/OISST, anomalias sem vazamento, walk-forward, RF/XGBoost, SHAP e permutation importance. Antes de declarar aderencia operacional meteorologica, faltam: `2m_temperature`, temperatura em niveis de pressao, bulbo umido ou proxy, correntes oceanicas U/V, dipolo do Atlantico, altura geopotencial derivada, calibracao probabilistica/reliability e analise de sensibilidade.

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

### 3. Politica de retomada

Regra operacional: checar o que ja existe, auditar no ledger, baixar/processar apenas pendencias e seguir para o proximo item quando um ano/mes isolado falhar. Use `--overwrite` somente para arquivo corrompido ou reprocessamento deliberado.

No Windows `cmd`, os comandos recomendados sao:

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\curate_and_resume_downloads.py --source chirps --stage all --start-year 1981 --execute --limit 0 --retries 5 --retry-wait 60 --continue-on-error

cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\curate_and_resume_downloads.py --source oisst --stage all --start-year 1981 --execute --limit 0 --retries 5 --retry-wait 60 --continue-on-error

cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py etl-ctd --start-year 1981 --max-depth 300 --min-levels 3 --execute --continue-on-error

cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py download-era5 --start-year 1981 --kind both --region nino34 --region brazil --annual-zarr --request-mode annual-kind --delete-raw-after-zarr --execute --continue-on-error

cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py download-oras --start-year 1981 --annual-zarr --request-mode annual-kind --delete-raw-after-zarr --execute --continue-on-error

cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py download-validation --source all --start-year 1981 --max-depth 300 --execute --continue-on-error
```

Runbook completo para usar em outra maquina: [docs/DOWNLOAD_CMD_NINO26.md](docs/DOWNLOAD_CMD_NINO26.md).

### 4. CHIRPS Fase 1

Dry-run:

```powershell
python scripts\data_pipeline.py download-chirps --start-year 1981 --resolution p25
```

Execucao real ate o ultimo ano disponivel pela regra de latencia:

```powershell
python scripts\data_pipeline.py download-chirps --start-year 1981 --resolution p25 --execute --continue-on-error
```

Curadoria completa recomendada:

```powershell
python scripts\curate_and_resume_downloads.py --source chirps --stage all --start-year 1981 --execute --limit 0 --retries 5 --retry-wait 60 --continue-on-error
```

### 5. OISST

```powershell
python scripts\curate_and_resume_downloads.py --source oisst --stage all --start-year 1981 --execute --limit 0 --retries 5 --retry-wait 60 --continue-on-error
```

OISST e a fonte oficial de SST/SSTA diaria. Depois do download, o cubo precisa ser padronizado e regridado antes da modelagem.

### 6. ERA5 e ORAS5

Verifique credenciais CDS:

```powershell
python scripts\data_pipeline.py check-cds
```

ERA5:

```powershell
python scripts\data_pipeline.py download-era5 --start-year 1981 --kind both --region nino34 --region brazil --annual-zarr --request-mode annual-kind --delete-raw-after-zarr --execute --continue-on-error
```

ORAS5:

```powershell
python scripts\data_pipeline.py download-oras --start-year 1981 --annual-zarr --request-mode annual-kind --delete-raw-after-zarr --execute --continue-on-error
```

Os comandos CDS usam bruto como cache temporario quando combinados com `--delete-raw-after-zarr`; no ERA5, o fluxo oficial `--annual-zarr --request-mode annual-kind` faz uma requisicao por ano/regiao/tipo (`single` ou `pressure`) e gera Zarr diario anual separado por variavel. No ORAS5, `annual-kind` faz uma requisicao por ano com todas as variaveis selecionadas e alinha a fonte mensal ao calendario diario. Use `annual-variable` ou modo mensal apenas como fallback.

### 7. CTD/WOD

```powershell
python scripts\data_pipeline.py download-ctd --start-year 1981 --max-depth 300 --min-levels 3 --execute --continue-on-error
```

Para baixar bruto primeiro:

```powershell
python scripts\data_pipeline.py download-ctd --start-year 1981 --raw-only --execute --continue-on-error
python scripts\data_pipeline.py etl-ctd --start-year 1981 --max-depth 300 --min-levels 3 --execute --continue-on-error
```

### 8. Regridding para a grade comum

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

- `data/processed/zarr/distributions/distribution_diagnostics.zarr`
- evento correspondente em `data/audit/ledger.jsonl`

O diagnostico estima `alpha`, `xmin`, distancia KS e compara power law contra lognormal/exponencial. A comparacao e obrigatoria porque chuva e OHC podem preferir lognormal.

### 9. Fase 3 - diagnostico fisico Nino 3.4

A Fase 3 produz variaveis fisicas explicaveis antes da modelagem: alinhamento de anomalias, D20/termoclina, volume/grau de aquecimento, slope longitudinal/vertical, duracao do sinal e comparacao com os picos historicos de El Nino mais impactantes.

Saidas previstas:

- `data/processed/zarr/features/nino34_physical_signal.zarr`
- `data/processed/zarr/features/nino34_thermocline_diagnostics.zarr`
- `data/processed/zarr/features/nino34_peak_signal_comparison.zarr`

### 10. Fase 4 - pre-analises estatisticas

A Fase 4 e experimental e serve para sentir o terreno antes dos modelos finais: regressao multipla, PCA, KNN, correlacoes defasadas e ablations simples de variaveis individuais/combinadas.

Saidas previstas:

- `data/processed/zarr/statistics/phase4_variable_screening.zarr`
- `data/processed/zarr/statistics/phase4_pca_modes.zarr`
- `data/processed/zarr/statistics/phase4_knn_similarity.zarr`

### 11. Fase 5 - modelagem walk-forward

Dry-run:

```powershell
python scripts\model_pipeline.py `
  --predictor-zarr data/processed/zarr/regridded/noaa_oisst.zarr `
  --target-zarr data/processed/zarr/regridded/brazil_precipitation.zarr `
  --target-var precip `
  --dry-run
```

Execucao agregada da Fase 5:

```powershell
python scripts\model_pipeline.py `
  --predictor-zarr data/processed/zarr/regridded/noaa_oisst.zarr `
  --target-zarr data/processed/zarr/regridded/brazil_precipitation.zarr `
  --target-var precip `
  --predictor-strategy mean `
  --target-strategy mean
```

Saidas:

- `data/processed/zarr/modeling/walk_forward_metrics.zarr`
- `data/processed/zarr/modeling/walk_forward_predictions.zarr`
- `data/processed/zarr/modeling/walk_forward_importances.zarr`
- `data/processed/zarr/modeling/walk_forward_group_weights.zarr`

Para SHAP em modelos de arvore:

```powershell
python scripts\model_pipeline.py ... --shap --shap-max-rows 1000
```

### 12. Validacao pixel-a-pixel em recorte

`flatten` e uma validacao controlada da Fase 5 em `0.25` grau. Antes de escalar para todo o Brasil, rode em recorte pequeno:

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

Registre tempo, memoria, numero de features, numero de pixels e tamanho dos stores Zarr antes de expandir.

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
