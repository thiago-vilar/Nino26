# NINO26 - Runbook de Download via Windows cmd

Atualizado em: 2026-06-12. Esta e a data da pagina; os comandos dinamicos usam `date.today()` no momento da execucao.
Raiz do projeto: `C:\DEV\NINO26`.

## 1. Ideia operacional

Use sempre o Python da venv diretamente:

```cmd
.venv\Scripts\python
```

Assim, nao e obrigatorio ativar a venv. Ativar a venv so e necessario se voce quiser digitar `python` sem o prefixo `.venv\Scripts\`.

Politica do projeto:

```text
1. Checar o que ja existe.
2. Baixar/processar apenas pendencias.
3. Manter bruto permanente onde ele e leve/necessario; em ERA5/GLORYS12 usar bruto como cache temporario validado.
4. Gerar Zarr quando houver ETL e apagar raw pesado validado com `--delete-raw-after-zarr`.
5. Continuar depois de falha isolada com --continue-on-error.
6. Usar --overwrite somente em reprocessamento deliberado ou arquivo corrompido.
7. Para anos em aberto, deixar o script calcular meses completos com `date.today()` e a latencia de cada fonte.
```

## 2. Preparar outra maquina

```cmd
cd /d C:\DEV\NINO26

.venv\Scripts\python scripts\data_pipeline.py init

.venv\Scripts\python scripts\data_pipeline.py status

.venv\Scripts\python scripts\data_pipeline.py check-cds
```

Se a venv ainda nao existir nessa maquina:

```cmd
cd /d C:\DEV\NINO26

python -m venv .venv

.venv\Scripts\python -m pip install --upgrade pip

.venv\Scripts\python -m pip install -r requirements.txt
```

## 3. Sequencia principal recomendada

### 3.1 CHIRPS

CHIRPS p25 e a precipitacao diaria do Brasil na grade oficial do projeto.

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\curate_and_resume_downloads.py --source chirps --stage all --start-year 1981 --execute --limit 0 --retries 5 --retry-wait 60 --continue-on-error
```

### 3.2 OISST

OISST e a fonte principal de SST/SSTA diaria.

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\curate_and_resume_downloads.py --source oisst --stage all --start-year 1981 --execute --limit 0 --retries 5 --retry-wait 60 --continue-on-error
```

### 3.3 CTD/WOD

Se os brutos `data\raw\ctd_noaa\wod\<ano>\wod_ctd_<ano>.nc` ja existem, nao baixe de novo. Reprocesse localmente para termoclina 0-300 m:

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py etl-ctd --start-year 1981 --max-depth 300 --min-levels 3 --overwrite --execute --continue-on-error
```

Se a maquina ainda nao tiver os brutos CTD:

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py download-ctd --start-year 1981 --max-depth 300 --min-levels 3 --execute --continue-on-error
```

Anos sem perfil valido no Nino 3.4 imprimem:

```text
ANO - sem dados
```

## 4. ERA5

### 4.1 Historico anual fechado

ERA5 usa Copernicus CDS. As credenciais precisam estar configuradas antes. No modo oficial do NINO26, `single-level` usa `--annual-zarr --request-mode annual-kind`, pois o payload anual agrupado cabe no limite do CDS. Para `pressure-level`, use `annual-auto`: ele tenta o request anual agrupado e divide automaticamente quando o CDS rejeita 6 variaveis x 3 niveis por custo/payload.

Para evitar que um `HTTP 500` temporario do CDS prenda a fila por horas, os downloads usam tres variaveis opcionais de controle:

```cmd
set "NINO_CDS_RETRY_MAX=5"
set "NINO_CDS_SLEEP_MAX=60"
set "NINO_CDS_TIMEOUT=120"
```

Com `--continue-on-error`, uma tarefa que continuar falhando depois dessas tentativas e registrada como erro e a fila segue para a proxima pendencia.

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py check-cds

cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py download-era5 --start-year 1981 --kind single --region nino34 --region brazil --annual-zarr --request-mode annual-kind --delete-raw-after-zarr --execute --continue-on-error
```

Para pressure-level, use `annual-auto`. O pacote completo tem `42 anos x 2 regioes x 6 variaveis = 504 produtos` entre 1984 e 2025, mas esse modo nao abre 504 requisicoes de saida: ele tenta baixar o maior grupo anual por regiao e so divide quando o CDS rejeita por custo/payload.

Comando unico recomendado:

```cmd
cd /d C:\DEV\NINO26 && set "NINO_CDS_RETRY_MAX=5" && set "NINO_CDS_SLEEP_MAX=60" && set "NINO_CDS_TIMEOUT=120" && .venv\Scripts\python scripts\data_pipeline.py download-era5 --start-year 1984 --end-year 2025 --kind pressure --region nino34 --region brazil --annual-zarr --request-mode annual-auto --delete-raw-after-zarr --execute --continue-on-error
```

Se quiser limitar a primeira rodada ao nucleo dinamico vertical, mantendo a mesma logica automatica:

```cmd
cd /d C:\DEV\NINO26 && set "NINO_CDS_RETRY_MAX=5" && set "NINO_CDS_SLEEP_MAX=60" && set "NINO_CDS_TIMEOUT=120" && .venv\Scripts\python scripts\data_pipeline.py download-era5 --start-year 1984 --end-year 2025 --kind pressure --region nino34 --region brazil --variable u_component_of_wind --variable v_component_of_wind --variable geopotential --annual-zarr --request-mode annual-auto --delete-raw-after-zarr --execute --continue-on-error
```

Esse comando anual baixa apenas anos completos. Se a data atual ainda estiver no meio do ano, o ano em aberto e ignorado automaticamente para evitar meses futuros. A flag `--delete-raw-after-zarr` transforma o NetCDF bruto em cache temporario: depois que o Zarr anual por variavel e validado, o `.nc` bruto daquela tarefa e apagado.

Fallback ainda mais fino, so se o CDS rejeitar uma requisicao anual por variavel:

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py download-era5 --start-year 1984 --end-year 2025 --kind pressure --region nino34 --region brazil --annual-zarr --request-mode monthly-kind --delete-raw-after-zarr --execute --continue-on-error
```

### 4.2 ERA5 do ano em aberto

Nao use `--annual-zarr` para o ano em aberto. Sem `--month`, o script calcula automaticamente os meses completos disponiveis pela latencia do ERA5. No estado atual do projeto, o primeiro ano em aberto e 2026:

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py download-era5 --start-year 2026 --kind both --region nino34 --region brazil --execute --continue-on-error
```

## 5. Oceano diario e memoria mensal

UFS+GLORYS permanece originalmente diário; os nomes individuais aparecem apenas
nos comandos de ingestão de cada componente. ORAS5 é fonte mensal independente,
sem promoção para diário. Execute o plano completo em
[RUNBOOK_FASE2_OCEANO.md](RUNBOOK_FASE2_OCEANO.md).
## 6. TAO/TRITON/Argo

TAO/TRITON/Argo e camada de validacao in situ do Nino 3.4. Ela nao substitui OISST, os cubos oceânicos diarios ou CTD/WOD. Serve para validar a estrutura vertical e manter aberta a pergunta cientifica: a subsuperficie melhora a previsao de seca/chuva no Brasil, ou SST/SSTA basta?

Fonte:

```text
TAO/TRITON: NOAA PMEL ERDDAP, temperatura e salinidade por profundidade.
Argo: Ifremer ERDDAP, perfis T/S/PRES, mais forte a partir de 1999/2000.
```

Recorte:

```text
Nino 3.4: 5S-5N, 170W-120W.
Profundidade/pressao maxima: 300 m/dbar.
```

Baixar tudo disponivel ate o dia da execucao:

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py download-validation --source all --start-year 1981 --max-depth 300 --execute --continue-on-error
```

Quando `--end-date` e omitido, o subcomando usa `date.today()` internamente. Quando `--end-year` e omitido, usa o ano da data final calculada.

Baixar apenas TAO/TRITON:

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py download-validation --source tao_triton --start-year 1981 --max-depth 300 --execute --continue-on-error
```

Baixar apenas Argo:

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py download-validation --source argo --start-year 1981 --max-depth 300 --execute --continue-on-error
```

Saidas:

```text
data/raw/tao_triton/temperature/tao_triton_temperature_<ano>.csv
data/raw/tao_triton/salinity/tao_triton_salinity_<ano>.csv
data/raw/argo/argo_nino34_<ano>.csv
```

## 7. Auditoria

Depois de cada bloco grande:

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py audit
```

Status de pastas e espaco:

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py status
```

## 8. Sequencia compacta para esta maquina

Pelo estado local verificado:

```text
CHIRPS: completo 1981-2026.
OISST: completo 1981-2026 no produto regrid.
CTD/WOD: bruto completo 1981-2025; reprocessar Zarr 0-300 m.
ERA5: 1981-1984 completo; retomar 1985 ate o ultimo ano anual completo e depois meses completos do ano em aberto.
Oceano: NOAA UFS 1981-1992, GLORYS/GLO12 diario e ORAS5 mensal ainda precisam ser ingeridos conforme o runbook da Fase 2.
TAO/TRITON/Argo: baixar validacao in situ ate o dia da execucao.
```

Comandos:

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py etl-ctd --start-year 1981 --max-depth 300 --min-levels 3 --overwrite --execute --continue-on-error

cd /d C:\DEV\NINO26 && set "NINO_CDS_RETRY_MAX=5" && set "NINO_CDS_SLEEP_MAX=60" && set "NINO_CDS_TIMEOUT=120" && .venv\Scripts\python scripts\data_pipeline.py download-era5 --start-year 1984 --end-year 2025 --kind pressure --region nino34 --region brazil --annual-zarr --request-mode annual-auto --delete-raw-after-zarr --execute --continue-on-error

cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py download-era5 --start-year 2026 --kind both --region nino34 --region brazil --execute --continue-on-error

cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\ocean_daily_pipeline.py ingest-ufs --start-year 1981 --end-year 1992 --build-features --execute

cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\ocean_daily_pipeline.py ingest-glorys --start-year 1993 --end-year 2025 --delete-source-after-zarr --execute

cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py download-validation --source all --start-year 1981 --max-depth 300 --execute --continue-on-error
```
