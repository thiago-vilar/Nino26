# Fase 4 - Notebooks (reconstrucao 2026-07-08, expandida)

Teleconexao ENSO -> anomalia de chuva no Brasil. Protocolo completo em
`docs/METODOLOGIA_FASE4.md`. Escopo estritamente Pacifico -> Brasil.
Comece pelo 4.0 (abertura/inventario). Ordem cientifica obrigatoria:
fases do ciclo -> determinantes -> sinal pixel-a-pixel -> alvos clusterizados.

| Ordem | Notebook | O que responde |
|---|---|---|
| 4.0 | `4_0_fase4_abertura.ipynb` | **Abertura (pre-flight):** justificativa, objetivos e metodologia da Fase 4; inventario de todos os dados disponiveis (forcantes do Pacifico + anomalia de chuva CHIRPS + serie ONI/eventos) com cobertura temporal; resumo das perguntas que 4A-4D respondem. Nao produz resultado cientifico. |
| 4A | `4A_ciclo_enso_fases.ipynb` | Marca cada domingo com tipo/fase/evento, classe, semana relativa ao onset e semana relativa ao pico; tambem resume a cronologia e duracao de cada fase por evento. |
| 4B | `4B_variaveis_determinantes_fases.ipynb` | Estudo puramente estatistico com todas as variaveis numericas do master: Cliff/Mann-Whitney, Kruskal/epsilon2, Spearman e relacoes entre pares. |
| 4C | `4C_sinal_pixel_lags.ipynb` | Sinal pixel-a-pixel: `Pacifico(t-L)` x anomalia padronizada de chuva CHIRPS 0,25(t), lags 0-78, N_eff+FDR; Brasil inteiro, NEB e Sul. |
| 4D | `4D_clusters_alvo.ipynb` | So depois do pixel: alvos clusterizados descritivamente a partir dos perfis r(lag) de todas as variaveis, estabilidade 1993-2009 vs 2010+ e gate estatistico da hipotese NEB seco / Sul umido em El Nino. |

Modulo compartilhado: `fase4_utils.py` (eventos/fases ENSO, anomalia harmonica
CHIRPS por pixel, inventario de janelas de lag, correlacao defasada vetorizada
com N_eff+FDR, N_eff generico, mapas com recorte regional).

Execucao headless:

```cmd
.venv\Scripts\python scripts\run_fase4_all.py
```

Notas operacionais:

- A primeira execucao do 4C monta o cache
  `phase4_chirps_weekly_zanom.parquet` lendo os 46 zarr anuais do CHIRPS
  (etapa pesada; ~10-30 min conforme o disco). Use
  `u.build_chirps_weekly_zanom(force=True)` se o CHIRPS for atualizado.
- 4B e 4C dependem do rotulo de fases do 4A; 4D depende do atlas do 4C.
  Execute em ordem.
- Notebooks legados da Fase 4 foram removidos; suas saidas nao devem ser
  citadas.

Kernel no VS Code: `Python 3 (.venv NINO26)`.
