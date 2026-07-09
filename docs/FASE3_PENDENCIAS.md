# Fase 3 - Fechamento contra a diretriz

**Data:** 2026-07-09 (atualizado apos execucao completa)  
**Base:** notebooks `3A-3L`, tabelas em `data/processed/parquet/statistics/`,
gerador `scripts/phase3_en_ln.py` e diretriz canonica
`docs/DIRETRIZES_FASES.md`.

## Parecer curto

A Fase 3 esta **concluida como diagnostico fisico semanal do Nino 3.4**. A
matriz-mestre semanal foi regenerada com 17 variaveis oceanicas + 14
atmosfericas ERA5 + `ocean_source_code`; os artefatos **El Nino e La Nina** ja
existem por evento e por fase; e o notebook `3L_en_ln_caracterizacao.ipynb`
fecha a camada visual/interpretativa EN/LN.

## Estado por item da diretriz

| Pedido da diretriz | Estado | Evidencia |
|---|---|---|
| Separar eventos El Nino e La Nina 1981-2026 | **Feito** | `phase3_events_en_ln.csv` (12 EN + 11 LN, ONI local simetrico +/-0.5) |
| Duracao media por tipo/classe, EN e LN | **Feito** | `phase3_duracao_por_tipo_classe.csv` |
| Classificar genese/crescimento/pico/decaimento por evento, EN e LN | **Feito** | `phase3_event_lifecycle_en_ln.csv`, `phase3_fases_semanais_en_ln.csv` |
| Variaveis que delimitam os 4 periodos: nivel, volatilidade semanal, discriminancia | **Feito** | `phase3_fase_stats_variaveis.csv`, `phase3_discriminantes_por_periodo.csv` |
| PCA por ciclo de vida, por fase e por sinal | **Feito** | `phase3_pca_por_fase.csv`, `phase3_pca_loadings_por_fase.csv` |
| Figuras EN/LN de ciclo, duracao, discriminantes e PCA por fase | **Feito** | `phase3L_ciclo_vida_en_ln.png`, `phase3L_duracao_fases_en_ln.png`, `phase3L_discriminantes_heatmap.png`, `phase3L_pca_por_fase.png` |
| Sem ML/RN | **OK** | 3I/3K usam LOO/nested-LOO como diagnostico estatistico exploratorio, nao modelo operacional |

## Arquivos-chave

```text
data/processed/parquet/statistics/phase3_events_en_ln.csv
data/processed/parquet/statistics/phase3_event_lifecycle_en_ln.csv
data/processed/parquet/statistics/phase3_duracao_por_tipo_classe.csv
data/processed/parquet/statistics/phase3_fase_stats_variaveis.csv
data/processed/parquet/statistics/phase3_discriminantes_por_periodo.csv
data/processed/parquet/statistics/phase3_pca_por_fase.csv
data/processed/parquet/statistics/phase3_pca_loadings_por_fase.csv
notebooks/fase3/3L_en_ln_caracterizacao.ipynb
```

## Observacao cientifica

O produto fecha a Fase 3 como caracterizacao fisica e estatistica semanal do
Pacifico/Nino 3.4. EOF espacial completo pode ser uma extensao futura de
visualizacao/modos espaciais, mas nao bloqueia o fechamento da Fase 3 porque a
diretriz central de eventos EN/LN, quatro fases, duracoes, discriminantes e PCA
por fase foi atendida.
