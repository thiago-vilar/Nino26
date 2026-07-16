# Fase 2 — disponibilização semanal e validação

- `F2Z_sanidade_variaveis.ipynb`: disponibilidade, continuidade e sanidade de todas as variáveis contratadas na matriz semanal. Expõe frequência nativa, cobertura e variáveis diretas/calculadas de UFS+GLORYS e comprova por timestamps que os Zarr ORAS5 são médias mensais excluídas do master.
- `F2V_validacao_insitu.ipynb`: validação independente de temperatura e salinidade com CTD/WOD, TAO/TRITON e Argo, além da comparação mensal UFS+GLORYS × ORAS5. Publica séries sobrepostas, correlações em nível e anomalias, tamanho amostral efetivo, viés, MAE, RMSE, IC95% bootstrap e FDR BH; observações e ORAS5 não preenchem a matriz principal.

UFS+GLORYS entra com temperatura potencial, salinidade e altura da superfície do mar diárias. Antes da agregação semanal são calculados D20, OHC 0–100 m, OHC 0–300 m, OHC 0–700 m, OHC 300–700 m, WWV e inclinação da termoclina. Apenas semanas completas com fechamento `W-SUN` são publicadas.

Execução completa no WSL2:

```bash
make fase2
```

A fonte canônica do ERA5 é o conjunto de Zarrs diários produzido pela Fase 1. A Fase 2 lê esses Zarrs diretamente, agrega os dados em períodos `W-SUN` e publica `data/processed/zarr/features/nino34_master_weekly.zarr`. O CSV semanal permanece como exportação de compatibilidade para consumidores ainda não migrados; o antigo cache diário em Parquet não é necessário.
