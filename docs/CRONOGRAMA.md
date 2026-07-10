# Cronograma e fases - NINO-BRASIL

**Projeto NINO-BRASIL - Oceanografia Fisica UFPE - Thiago Vilar**
**Revisao:** 2026-07-09.

As **diretrizes canonicas** de cada fase estao em
[`DIRETRIZES_FASES.md`](DIRETRIZES_FASES.md) (fonte de verdade). Este documento
resume o estado real em disco e a ordem de execucao.

## 1. Estado real por fase

| Fase | Titulo | Estado | Evidencia |
|---|---|---|---|
| 1 | Base local e ingestao bruta | **Concluida** | CHIRPS, OISST, ERA5, oceano UFS/GLORYS12, ORAS5, validacao in situ. |
| 2 | Padronizacao, anomalias, Zarr e grade comum | **Concluida** | Matriz-mestre semanal `nino34_master_weekly.csv` (31 variaveis fisicas: 17 oceanicas + 14 ERA5, mais `ocean_source_code` como metadado), auditoria+validacao (tudo True), validacao CTD, notebook de sanidade. |
| 3 | Diagnostico fisico do sinal Nino 3.4 (sem ML/RN) | **Concluida** | Notebooks 3A-3L executados; eventos EN/LN, quatro periodos por evento, duracoes, discriminantes, PCA por fase, Kelvin/Bjerknes/mapas e relatorio final. |
| 4 | Teleconexao ENSO -> chuvas extremas/secas no Brasil (sem ML/RN) | **Em execucao** | Sinal pixel-a-pixel + lags semanais; metrica P90 do periodo de aquecimento a consolidar. |
| 5 | Mesmo estudo da Fase 4, com ML (Random Forest e XGBoost) + XAI | **Nao iniciada** | series semanais e diarias. |
| 6 | Redes neurais nativas + XAI | **Nao iniciada** | so apos vencer Fases 4 e 5. |
| WEB | Publicacao e operacao (FaseWEB) | **Esqueleto** | painel/web + rotina de recalibracao. |

## 2. Ordem de execucao (bash / WSL)

```bash
cd /mnt/c/DEV/NINO26

# Fase 2 - preparar dados (uma vez; extrai ERA5 completo, ~25-30 min)
.venv-wsl/bin/python scripts/build_master_weekly.py --era5-years 1981:2026
.venv-wsl/bin/python scripts/fase3_build_inputs.py --force
.venv-wsl/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  notebooks/fase2/2Z_sanidade_variaveis.ipynb

# Fase 3 e Fase 4
.venv-wsl/bin/python scripts/run_fase3_all.py
.venv-wsl/bin/python scripts/run_fase4_all.py
```

No CMD do Windows, troque `.venv-wsl/bin/python` por `.venv\Scripts\python`.

## 3. Gates de decisao

| Gate | Quando | Criterio |
|---|---|---|
| G1 | fim da Fase 4 | teleconexao significativa por campo (N_eff/FDR), efeito interpretavel, estabilidade temporal e lags defensaveis |
| G2 | fim da Fase 5 | RF/XGBoost + XAI superam a triagem estatistica da Fase 4 e os baselines |
| G3 | fim da Fase 6 | redes neurais superam climatologia, persistencia, Fase 4 e Fase 5 |

Regra de ouro: nada avanca sem vencer climatologia e persistencia.
