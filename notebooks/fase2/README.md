# Fase 2 — disponibilização semanal e validação

- `F2Z_sanidade_variaveis.ipynb`: disponibilidade, continuidade e sanidade de todas as variáveis contratadas na matriz semanal; a quantidade é calculada durante a execução.
- `F2V_validacao_insitu.ipynb`: inventário e validação independente com CTD/WOD, TAO/TRITON e Argo; semanas sem observação não são preenchidas.

Execução completa no WSL2:

```bash
cd /mnt/c/DEV/NINO26 && .venv-wsl/bin/python scripts/run_fase2_all.py
```

A fonte canônica do ERA5 é o conjunto de Zarrs diários produzido pela Fase 1. A Fase 2 lê esses Zarrs diretamente, agrega os dados em períodos `W-SUN` e publica `data/processed/zarr/features/nino34_master_weekly.zarr`. O CSV semanal permanece como exportação de compatibilidade para consumidores ainda não migrados; o antigo cache diário em Parquet não é necessário.
