# Fase 3 - Notebooks 3A-3I/3K

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
| 3F | DHW agrega leitura alem de SSTA/WWV/OHC? Kelvin e visivel? | correlacao parcial por horizonte, Hovmoller SSH |
| 3G | (extra) Como o ciclo de vida se relaciona com o DHW C-week? | metricas por fase de evento, composto duplo, escalonamento, mapa DHW-lon |
| 3H | (extra) Que estado fisico precede o onset? A genese separa `forte_p90` de `super_p95`? | compostos onset-alinhados, retrato precursor por classe, separacao Spearman |
| 3K | (extra) Quais variaveis explicam o crescimento pre-pico? | PCA, loadings e conjunto indispensavel para crescimento |
| 3I | Qual e a interpretacao integrada da Fase 3? Como comparar P90/P95 e 2026? | conclusoes executivas, tabela P90/P95, estado 2026, texto para parecer |

**Regra de corte do parecer:** so entra o que sobrevive a **3D e 3E**.
O 3F tem regra extra de nao-redundancia; o 3G caracteriza severidade acumulada;
o 3I consolida a interpretacao, mas nao cria evidencia nova.

## DHW

O DHW principal da Fase 3 e `dhw_12w`: acumulo de C-week em 12 semanas acima
do limiar P90 diario local da SSTA OISST. `dhw_26w_p90` permanece como
sensibilidade na escala do evento, pois sua janela se aproxima do e-folding
observado da SSTA (~27 semanas). Ambas sao metricas de calor
acumulado/severidade; nao devem ser chamadas de previsao nem definicao de
El Nino.

Classificacao executiva: `forte_p90` = pico mensal >P90 e <P95; `super_p95` =
pico mensal >P95. A media executiva dos eventos >P90 fica em
`phase3I_media_eventos_gt_p90.csv`.

Kernel no VS Code: `Python 3 (.venv NINO26)`.
Modulo compartilhado: `fase3_utils.py` (caminhos, matriz semanal, eventos, salvamento padrao).
