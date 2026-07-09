# Fase 4 - Notebooks (reconstrucao 2026-07-08, expandida)

Teleconexao ENSO -> chuva no Brasil. Protocolo completo em
`docs/METODOLOGIA_FASE4.md`. Escopo estritamente Pacifico -> Brasil.
Comece pelo 4.0 (abertura/inventario). Ordem cientifica obrigatoria:
fases do ciclo -> determinantes -> sinal pixel-a-pixel -> alvos clusterizados.

| Ordem | Notebook | O que responde |
|---|---|---|
| 4.0 | `4_0_fase4_abertura.ipynb` | **Abertura (pre-flight):** justificativa, objetivos e metodologia da Fase 4; inventario de todos os dados disponiveis (forcantes do Pacifico + chuva CHIRPS + serie ONI/eventos) com cobertura temporal; resumo das perguntas que 4A-4D respondem. Nao produz resultado cientifico. |
| 4A | `4A_ciclo_enso_fases.ipynb` | Separacao logica/estatistica das 4 fases (I. genese, II. crescimento/acoplamento, III. pico, IV. decaimento) para El Nino E La Nina; duracoes, dispersao e ONI por fase; rotulo semanal do ciclo. |
| 4B | `4B_variaveis_determinantes_fases.ipynb` | Estudo puramente estatistico: quais variaveis do Pacifico mais determinam cada fase (Cliff/Mann-Whitney + Kruskal/epsilon2 + Spearman genese->pico), para EN e LN. |
| 4C | `4C_sinal_pixel_lags.ipynb` | Sinal pixel-a-pixel: conjunto Pacifico x chuva CHIRPS em lags semanais 0-78; Brasil inteiro primeiro, depois recorte NEB e recorte Sul; quanto tempo o sinal demora por regiao e tipo de sinal. |
| 4D | `4D_clusters_alvo.ipynb` | So depois do pixel: alvos clusterizados mais afetados, lag de atuacao por tipo de sinal, estabilidade 1993-2009 vs 2010+ e gate G1. |

Modulo compartilhado: `fase4_utils.py` (eventos/fases ENSO, anomalia harmonica
CHIRPS por pixel, correlacao defasada vetorizada com N_eff+FDR, N_eff generico,
mapas com recorte regional).

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
