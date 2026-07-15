# Notebooks NINO26

Cada pasta representa uma fase independente. Notebooks apresentam resultados;
runners em `scripts/` produzem os núcleos numéricos quando aplicável.

| Pasta | Papel |
|---|---|
| `fase1/` | atualização retomável, inventário e auditoria das fontes |
| `fase2/` | disponibilização e gráficos de sanidade no tempo, incluindo seção de validação in situ |
| `fase3_nino/`, `fase3_nina/` | análises puramente estatísticas do ciclo |
| `fase4_nino/`, `fase4_nina/` | relação estatística, lags e distribuição espaço-temporal no Brasil |
| `fase5/` | RF/XGBoost para antecipar fases e faixa de pico |
| `fase6/` | RF/XGBoost para distribuição espaço-temporal no Brasil |
| `fase7/` | ConvLSTM para antecipar fases e faixa de pico |
| `fase8/` | ConvLSTM para distribuição espaço-temporal no Brasil |

Nenhum notebook depende cientificamente da aprovação de outra fase. Reutilização
de dados ou comparação deve ser declarada. Toda figura analítica possui tabela
numérica correspondente; JSONs ficam na árvore de metadados.
