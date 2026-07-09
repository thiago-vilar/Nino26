# Notebooks

Indice operacional dos notebooks ativos. A fonte canonica das fases fica em
`../docs/DIRETRIZES_FASES.md`.

| Pasta | Estado | Papel |
|---|---|---|
| `fase2/` | Concluida | Sanidade da matriz semanal `nino34_master_weekly.csv`: todas as variaveis, cobertura, lacunas, CTD e sombreamento EN/LN. |
| `fase3/` | Em consolidacao/rebuild | Diagnostico fisico Nino 3.4, sem ML/RN: eventos EN/LN, ciclo de vida, Kelvin, Bjerknes, PCA/EOF e quatro periodos. |
| `fase4/` | Em execucao | Teleconexao ENSO -> chuva no Brasil, sem ML/RN: CHIRPS semanal, P90, lags e testes pixel-a-pixel com N_eff/FDR. |

Ordem curta:

```cmd
.venv\Scripts\python scripts\build_master_weekly.py --era5-years 1981:2026
.venv\Scripts\python scripts\fase3_build_inputs.py --force
.venv\Scripts\python scripts\run_fase3_all.py
.venv\Scripts\python scripts\run_fase4_all.py
```
