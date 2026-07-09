# Fase 2 - Notebook de sanidade

`2Z_sanidade_variaveis.ipynb` - inspecao sensorial de **todas** as variaveis da
matriz-mestre semanal unificada (`nino34_master_weekly.csv`), em graficos grandes,
com El Nino (vermelho) e La Nina (azul) sombreados. Roda ao fim da Fase 2, antes
de qualquer analise da Fase 3/4.

Pre-requisito: gerar a matriz-mestre na maquina de origem:

```cmd
.venv\Scripts\python scripts\build_master_weekly.py --era5-years 1981:2026
```

Isso materializa `data/processed/parquet/features/nino34_master_weekly.csv`
(17 oceanicas unificadas UFS/GLORYS/GLO12 + 14 atmosfericas ERA5 semanais para o
feedback de Bjerknes), mais as auditorias `phase2_master_audit.csv`,
`phase2_master_validation.csv` e a validacao in situ `phase2_ctd_validation.csv`.

Para uma base de teste rapida (so oceano + tau_x, sem extrair ERA5):

```cmd
.venv\Scripts\python scripts\build_master_weekly.py --ocean-only
```
