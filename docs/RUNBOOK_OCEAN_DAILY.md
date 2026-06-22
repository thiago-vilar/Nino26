# Runbook do oceano diario

## Contrato cientifico

O pipeline aceita somente observacoes ou medias originalmente diarias. Nao interpola, repete nem promove produto mensal ou semanal para o eixo diario.

| Periodo usado | Fonte | Resolucao temporal original | Grade armazenada | Papel |
|---|---|---|---|---|
| 1981-1992 | NOAA UFS Marine Reanalysis | analise diaria | 0,25 grau por interpolacao da fonte nominal de 1 grau | ponte historica |
| 1993-26/05/2026 | Copernicus Marine GLORYS12V1 | media diaria | 0,25 grau, por media de blocos da grade 1/12 grau | fonte principal |
| apos 26/05/2026 | Copernicus Marine GLO12 operacional | media diaria de analise | 0,25 grau | cauda sem previsoes |

As fontes permanecem em stores separados. A mudanca de fonte em 1993 deve entrar como metadado/covariavel e as anomalias devem ser ajustadas somente no bloco de treino de cada fold. Nao se substituem valores entre fontes.

Os cubos processados das tres fontes usam exatamente as mesmas coordenadas de 0,25 grau. GLORYS/GLO12 e reduzido por media de blocos 3x3 alinhados; UFS e interpolado bilinearmente. A interpolacao UFS serve para compatibilidade de grade e nao deve ser interpretada como aumento de resolucao observacional.

Dominio solicitado: `5S-5N`, `120E-80W` (`120-280` na convencao 0-360). A requisicao vai ate 800 m apenas para incluir suporte vertical abaixo de 700 m; no GLORYS, o ultimo centro nativo retido e aproximadamente 763,33 m. A metrica OHC e integrada exatamente ate 700 m, nao ate 763 nem 800 m. Variaveis: temperatura potencial `thetao`, salinidade `so` e nivel do mar `zos`. D20, OHC por camadas, WWV e inclinacao da termoclina sao calculados localmente a partir desses campos.

## 1. Auditar o plano sem baixar

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\ocean_daily_pipeline.py plan --start-year 1981 --end-year 2025
```

Objetivo: mostrar a divisao temporal, o dominio e a quantidade de tarefas. Nao acessa nem altera dados.

## 2. Validar a credencial Copernicus Marine

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\copernicusmarine.exe login --check-credentials-valid
```

Objetivo: verificar a credencial salva sem iniciar download. Se ainda nao houver credencial, execute `.venv\Scripts\copernicusmarine.exe login` uma vez e informe usuario/senha interativamente; nao coloque senha no comando nem no repositorio.

Essa conta e separada da chave CDS. Se aparecer `No credentials found`, cadastre-se gratuitamente em <https://data.marine.copernicus.eu/register>, confirme o e-mail, execute o login interativo e repita a verificacao.

## 3. Ingerir NOAA UFS, 1981-1992

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\ocean_daily_pipeline.py ingest-ufs --start-year 1981 --end-year 1992 --build-features --execute
```

Objetivo: abrir remotamente o ZIP publico da NOAA, ler somente os 4.383 NetCDF oceanicos diarios de 1981-1992, recortar T(z), S(z), SSH e espessura para o Pacifico equatorial/0-800 m e gravar Zarr anual comprimido. O ZIP completo de aproximadamente 320 GB nao e salvo localmente. A transferencia estimada dos membros selecionados e aproximadamente 86 GiB.

Retomada: stores anuais ja existentes e validos sao ignorados. Nao use `--overwrite` em retomada normal.

## 4. Ingerir GLORYS12V1, 1993-26/05/2026

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\ocean_daily_pipeline.py ingest-glorys --start-year 1993 --end-year 2026 --end-date 2026-05-26 --delete-source-after-zarr --execute
```

Objetivo: para cada ano, fazer uma unica requisicao Copernicus Marine contendo simultaneamente `thetao`, `so` e `zos`, baixar diretamente em Zarr, validar a continuidade diaria, agregar espacialmente a 0,25 grau, calcular as features fisicas, validar os produtos e apagar somente o cache nativo daquele ano.

Quantidade: 34 requisicoes anuais Copernicus Marine. O ORAS5 mensal usa um fluxo CDS separado e nunca entra como observacao diaria. Uma requisicao multidecadal seria numericamente menor, mas formaria uma transferencia muito grande e fragil; a divisao anual permite retomada e liberacao de espaco ano a ano.

Retomada: por padrao, `--skip-existing` evita novo download do Zarr nativo existente; os Zarr processados tambem sao validados e reutilizados. Use `--overwrite` somente se uma auditoria provar corrupcao.

## 5. Separar download e processamento, se necessario

Download agrupado, sem processar:

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\ocean_daily_pipeline.py download-glorys --start-year 1993 --end-year 2026 --end-date 2026-05-26 --execute
```

Objetivo: obter os Zarr nativos diarios. Use apenas se quiser preservar o cache original antes do processamento.

Processamento do cache ja baixado:

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\ocean_daily_pipeline.py process-glorys --start-year 1993 --end-year 2026 --build-features --delete-source-after-zarr
```

Objetivo: processar o que ja existe, sem requisitar dados, e liberar o cache nativo somente depois da validacao do Zarr 0,25 grau e das features.

## 6. Atualizar a auditoria executiva

```cmd
cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\update_painel_executivo.py
```

Objetivo: recontar cobertura anual e uso de disco e atualizar `painel_executivo.md`.

## Saidas

```text
data/processed/zarr/ocean_daily/noaa_ufs/<ano>/noaa_ufs_equatorial_pacific_<ano>_daily.zarr
data/processed/zarr/ocean_daily/glorys12/<ano>/glorys12_equatorial_pacific_<ano>_daily_0p25.zarr
data/processed/zarr/features/ocean_daily/<fonte>/<ano>/<fonte>_ocean_features_<ano>_daily.zarr
```

Para a execucao integrada com ORAS5 mensal e auditorias, use [RUNBOOK_FASE2_OCEANO.md](RUNBOOK_FASE2_OCEANO.md).
