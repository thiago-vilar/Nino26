# Fase 3 - Notebooks 3A-3I

Implementacao executavel do protocolo de `docs/FASE3_RECOMENDACOES.md`.
Todos os notebooks: (i) declaram a pergunta que respondem e a metodologia,
(ii) gravam tabelas numericas em `data/processed/parquet/statistics/`,
(iii) gravam figuras e mapas em `data/processed/figures/fase3/`.
Nenhuma figura sem saida numerica correspondente.

## Pre-requisitos

```cmd
.venv\Scripts\python scripts\data_pipeline.py build-nino34-daily-index
.venv\Scripts\python scripts\data_pipeline.py build-nino34-sst-reference
.venv\Scripts\python scripts\data_pipeline.py build-nino34-p90-peaks
.venv\Scripts\python scripts\data_pipeline.py build-nino34-p95-peaks
.venv\Scripts\python scripts\data_pipeline.py build-phase3-diagnostics
.venv\Scripts\python scripts\data_pipeline.py audit-phase3-diagnostics
.venv\Scripts\python scripts\fase3_build_inputs.py
.venv\Scripts\python -m pytest -q
```

`fase3_build_inputs.py` materializa ATL3/ATL4/TNA/TSA, banda equatorial por
longitude, SSH de eventos, DHW diario, variantes DHW e o cache de mapas. Use
`--force` quando OISST/oceano forem atualizados.

Execucao headless completa:

```cmd
.venv\Scripts\python scripts\run_fase3_all.py
```

## Notebooks (ordem de execucao)

| NB | Pergunta | Saidas principais |
|---|---|---|
| 3A | Quais series fisicas descrevem o sistema e em que janelas reais? | `phase3_indices_semanais.csv` (matriz canonica W-SUN), cobertura, series, Hovmoller panorama |
| 3B | Como eventos nascem, crescem, picam e decaem? Quanta memoria ha? | taxas por evento, trajetorias compostas, e-folding, mapa composto de pico |
| 3C | O que antecede o pico do Nino 3.4, e com quantas semanas? | ranking preliminar de lags, heatmap, mapa lon x lag |
| 3D | O que sobrevive a N_eff + FDR + IC95? | testes completos, ranking significativo, forest plot, mapa FDR |
| 3E | O sinal vale em 1993-2009 E 2010-presente? | tabela de estabilidade, scatter r1 x r2, mapas por subperiodo |
| 3F | DHW agrega leitura alem de SSTA/WWV/OHC? Kelvin e visivel? | correlacao parcial por horizonte, Hov