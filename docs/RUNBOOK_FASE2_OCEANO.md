# Execucao completa da Fase 2 oceanica

## Comando unico retomavel

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\run_ocean_phase2.py --execute --continue-on-error
```

Objetivo: executar, na ordem, UFS principal e sobreposicao, GLORYS multiyear, cauda GLO12, ORAS5 mensal, auditoria e painel. O comando e retomavel porque cada pipeline valida e ignora stores existentes. As secoes abaixo mostram as mesmas etapas individualmente para controle fino.

## Contrato

- UFS e GLORYS/GLO12 entram somente com valores originalmente diarios.
- ORAS5 entra somente como media mensal e nunca e promovido a observacao diaria.
- Temperatura, salinidade e SSH sao preservados; D20, OHC, SSS, WWV e Tilt sao derivados com formula e proveniencia.
- O limite de requisicao e 800 m; ele nao muda a metrica cientifica. Na grade vertical GLORYS, isso retém o centro nativo de aproximadamente 763,33 m (o seguinte fica em aproximadamente 902,34 m). O OHC continua integrado exatamente entre 0 e 700 m, recortando a espessura da camada no limite de 700 m.
- Todos os cubos processados usam os mesmos nos canonicos de 0,25 grau (`5S-5N`, `120E-280E`, extremos inclusos). GLORYS/GLO12 usa media alinhada de blocos 3x3 da grade 1/12 grau. UFS usa interpolacao bilinear da grade nativa de aproximadamente 1 grau. ORAS5 mensal usa interpolacao linear da grade curvilinea. As interpolacoes alinham coordenadas para comparacao, mas nao criam informacao espacial nova.
- Dados de previsao GLO12 nao entram na serie historica.

## 1. Preparar estrutura e credenciais

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py init
```

Objetivo: criar os novos caminhos `ocean_daily`, `ocean_monthly`, features e auditoria sem alterar produtos existentes.

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py check-cds
```

Objetivo: validar a credencial CDS usada exclusivamente pelo ORAS5 mensal.

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\copernicusmarine.exe login --check-credentials-valid
```

Objetivo: validar a credencial Copernicus Marine usada pelo GLORYS/GLO12. Ela e separada da chave CDS.

Se aparecer `No credentials found`:

1. Crie gratuitamente a conta em <https://data.marine.copernicus.eu/register> e confirme o e-mail.
2. Execute o login abaixo e informe e-mail/usuario e senha interativamente. A senha nao deve ser colocada na linha de comando, no repositorio ou neste documento.

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\copernicusmarine.exe login
```

3. Repita `login --check-credentials-valid`. Somente inicie o comando completo quando a credencial for declarada valida.

O projeto tambem aceita `COPERNICUSMARINE_SERVICE_USERNAME` e `COPERNICUSMARINE_SERVICE_PASSWORD` no `.env` local, que e ignorado pelo Git. O orquestrador carrega esse arquivo, faz a verificacao antes da primeira transferencia e aborta sem baixar nada se a conta Marine estiver ausente ou invalida.

## 2. Conferir planos sem baixar

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\ocean_daily_pipeline.py plan --start-year 1981 --end-year 2026
```

Objetivo: exibir fontes diarias, transicao de 1993 e limite de requisicao de 800 m para sustentar a integracao cientifica exata ate 700 m.

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\ocean_monthly_pipeline.py plan --start-year 1981 --end-year 2026 --end-month 5
```

Objetivo: confirmar no maximo 92 requisicoes CDS ORAS5: duas por ano, agrupadas por resolucao vertical.

## 3. NOAA UFS diario: serie principal 1981-1992

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\ocean_daily_pipeline.py ingest-ufs --start-year 1981 --end-year 1992 --build-features --execute
```

Objetivo: ler remotamente somente os NetCDF diarios oceanicos desses anos, recortar T(z), S(z), SSH e espessura ate 800 m, interpolar para os nos canonicos de 0,25 grau e produzir Zarr/features anuais. Stores validos sao ignorados na retomada.

## 4. NOAA UFS de sobreposicao: auditoria UFS-GLORYS

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\ocean_daily_pipeline.py ingest-ufs --start-year 1993 --end-year 1995 --build-features --execute
```

Objetivo: produzir tres anos que nao entram como fonte principal depois de 1992, mas permitem medir sazonalidade, vies e RMSE da mudanca UFS para GLORYS.

## 5. GLORYS12 multiyear diario: 1993-26/05/2026

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\ocean_daily_pipeline.py ingest-glorys --start-year 1993 --end-year 2026 --end-date 2026-05-26 --delete-source-after-zarr --execute
```

Objetivo: fazer uma requisicao anual contendo `thetao+so+zos`, incluir um halo horizontal minimo, reduzir a grade nativa 1/12 grau por blocos 3x3 alinhados aos mesmos nos de 0,25 grau, calcular features e apagar apenas o cache nativo anual ja validado.

Quantidade maxima: 34 requisicoes Copernicus Marine. Nao use `--overwrite` em retomada normal.

## 6. Cauda operacional diaria: 27/05/2026-19/06/2026

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\ocean_daily_pipeline.py ingest-glorys-operational --start-date 2026-05-27 --end-date 2026-06-19 --delete-source-after-zarr --execute
```

Objetivo: fazer tres requisicoes, uma para cada dataset operacional (`thetao`, `so`, `zos`), unir somente os dias historicos solicitados e rejeitar hoje/datas futuras. Em atualizacoes posteriores, omita `--end-date`; o comando usa ontem como limite conservador.

## 7. ORAS5 mensal independente: 1981-maio/2026

```cmd
cd /d C:\DEV\NINO26 && set "NINO_CDS_RETRY_MAX=5" && set "NINO_CDS_SLEEP_MAX=60" && set "NINO_CDS_TIMEOUT=120" && .venv\Scripts\python scripts\ocean_monthly_pipeline.py ingest --start-year 1981 --end-year 2026 --end-month 5 --build-features --delete-raw-after-zarr --execute --continue-on-error
```

Objetivo: por ano, tentar somente duas requisicoes CDS: uma com D20/OHC300/OHC700/SSH/SSS e outra com temperatura/salinidade 3D. Cada variavel e interpolada da grade curvilinea ORAS5 para os mesmos nos de 0,25 grau e vira Zarr mensal anual; WWV e Tilt permanecem mensais. O raw so e apagado depois de todos os Zarr do grupo abrirem e validarem.

Se algum ano falhar, execute exatamente o mesmo comando. Anos com todos os stores validos sao ignorados.

## 8. Auditoria ORAS5 mensal

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\ocean_monthly_pipeline.py audit --start-year 1981 --end-year 2026 --end-month 5
```

Objetivo: exigir 12 registros mensais por ano fechado, cinco em 2026, sete variaveis e features mensais, recusando qualquer store rotulado como diario.

## 9. Auditoria integrada e transicoes

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\audit_ocean_phase2.py --glorys-my-end 2026-05-26 --operational-end 2026-06-19 --oras-end 2026-05-01 --overlap-year 1993 --overlap-year 1994 --overlap-year 1995
```

Objetivo: verificar continuidade diaria 1981-19/06/2026, integridade mensal ORAS5, variaveis obrigatorias, ausencia de previsoes e calcular vies/RMSE UFS-GLORYS. A Fase 2 somente fecha se o comando retornar codigo zero e `status=complete`.

Saidas:

```text
data/audit/ocean_phase2_audit.json
data/processed/parquet/ocean_source_transition_audit.csv
```

## 10. Atualizar o painel

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\update_painel_executivo.py
```

Objetivo: refletir cobertura, espaco e resultado final da auditoria no painel executivo.
