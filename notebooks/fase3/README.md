# Fase 3 - Notebooks 3A-3G

Implementacao executavel do protocolo de `docs/FASE3_RECOMENDACOES.md`.
Todos os notebooks: (i) declaram a pergunta que respondem e a metodologia,
(ii) gravam tabelas numericas em `data/processed/parquet/statistics/`,
(iii) gravam figuras e mapas em `data/processed/figures/fase3/`.
Nenhuma figura sem saida numerica correspondente.

## Pre-requisito (uma vez)

```cmd
.venv\Scripts\python scripts\fase3_build_inputs.py
```

Materializa ATL3/ATL4/TNA/TSA, banda equatorial por longitude, SSH de eventos,
DHW diario e o cache de mapas. Idempotente.

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
| 3H | (extra) Que estado fisico precede o onset? A genese separa fortes de fracos? | compostos onset-alinhados, retrato precursor por classe, separacao Spearman |

**Regra de corte do parecer:** so entra o que sobrevive a **3D e 3E**.
O 3F tem regra extra de nao-redundancia; o 3G caracteriza severidade acumulada.

Kernel no VS Code: `Python 3 (.venv NINO26)`.
Modulo compartilhado: `fase3_utils.py` (caminhos, matriz semanal, eventos, salvamento padrao).
