# NINO-BRASIL

Projeto Python para medir o peso relativo de variaveis oceanograficas e atmosfericas associadas ao aquecimento do Pacifico sobre anomalias diarias de precipitacao no Brasil.

O fluxo oficial preserva a janela historica completa: `1981-01-01` ate o ultimo dado disponivel por fonte. O fim do periodo e dinamico porque CHIRPS, OISST, ERA5, ORAS5 e CTD/WOD tem latencias diferentes.

## Decisoes metodologicas fixas

- Frequencia de modelagem: sempre diaria.
- Horizontes de previsao: semanal, de 1 a 24 semanas (`7` a `168` dias).
- Validacao: somente blocos temporais e walk-forward; nao usar split aleatorio.
- Anti-vazamento: climatologia diaria, desvio padrao, P10, P25, P75 e P90 sao estimados no bloco de treino de cada fold e reaplicados a validacao/teste.
- Grade comum das Fases 1, 2, 3 e 4: `0.25` grau, longitude `0_360`, declarada em `configs/project.yaml`.
- CHIRPS oficial das Fases 1, 2, 3 e 4: `p25`, coerente com a grade comum e com o arquivo local ja baixado.
- CHIRPS `p05`: reservado para experimento futuro de alta resolucao, depois que o fluxo `0.25` grau estiver validado.
- SST/SSTA principal: NOAA OISST diario.
- ORAS5: reservado para memoria oceanica subsuperficial, nao para duplicar SST mensal.

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
| Fase 3 | Machine learning classico e XAI: Ridge, Random Forest, XGBoost, walk-forward, permutation importance, SHAP e pesos por grupo | Stores Zarr de metricas, previsoes, importancias, pesos por grupo e mapas analiticos |
| Fase 4 | Redes neurais e XAI: CNN, ConvLSTM, U-Net, Transformer espaco-temporal, saliency, occlusion e attention maps | Stores Zarr de treino/inferencia neural, explicabilidade neural e comparacao contra a Fase 3 |
| Fase 5 | Teste da tecnica Memory Caching do Google Research (arXiv:2602.24281): RNNs com memoria crescente via cache de estados, variantes Residual/Gated/Soup/SSC, sob o mesmo walk-forward | Stores Zarr de runs, comparacao de skill contra Fases 3/4 e relatorio skill x eficiencia |
| Fase 6 | Publicacao, operacao recorrente, experimentos `p05` e relatorios automaticos | Produto em `docs/`, rotina recorrente e painel publico |

## Estado local dos dados

Atualmente ha IBGE e um arquivo CHIRPS `p25` de 1981 em `data/raw/chirps/p25/`. Esse arquivo nao e erro: ele esta alinhado com a escolha oficial da Fase 1. Nao apague `p25` salvo se o arquivo estiver corrompido.

Dados que ainda precisam ser ingeridos para encerrar a base da Fase 1 e entrar com seguranca na Fase 2:

- NOAA OISST historico diario.
- CHIRPS `p25` historico diario.
- ERA5 single e pressure levels.
- ORAS5 subsuperficial.
- CTD/WOD para validacao vertical.
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

Os comandos CDS mantem o bruto em cache e geram Zarr diario por variavel quando executados sem `--raw-only`.

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

- `data/processed/zarr/distributions/distribution_diagnostics.zarr`
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

Execucao agregada da Fase 3:

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

### 10. Validacao pixel-a-pixel em recorte

`flatten` e uma validacao controlada da Fase 3 em `0.25` grau. Antes de escalar para todo o Brasil, rode em recorte pequeno:

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
