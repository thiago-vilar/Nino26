# Painel descritivo do Parecer 2

Data: 2026-06-13

## Veredito tecnico

O Parecer 2 procede. A regra "saida numerica antes da visual" melhora auditoria, reprodutibilidade e analise colaborativa. Apliquei as mudancas de baixo risco que nao exigem reestruturar a pipeline: plots agora geram CSV, a Fase 5 exporta resumos tabulares, a Fase 3 ganhou tabela mensal Nino 3.4, e os diagnosticos de distribuicao/MHW ganharam contratos numericos.

## Painel de modificacoes aplicadas

| Metodo | Recomendacao | Decisao | Modificacao aplicada |
|---|---|---|---|
| M4 Walk-forward | Gerar resumo texto/CSV de metricas | Procede | `model_pipeline.py` grava `walk_forward_metrics_summary.csv` |
| M4 Baselines | Exportar comparacao contra persistencia/climatologia | Parcial | `walk_forward_baseline_comparison.csv` e gerado com cabecalho estavel; fica vazio ate os baselines entrarem como linhas de metricas |
| M5 Importance | Exportar top-20 por lag/modelo/tarefa | Procede | `importance_top20_by_lag_model.csv` |
| M5 Guard flatten | Registrar skip de permutation importance | Procede | Linha `method="skipped"` com `reason`, `n_features` e `feature_limit` |
| M6 SHAP | Resumo SHAP por grupo e acordo SHAP x permutation | Procede | `shap_group_summary.csv` e `xai_method_agreement.csv` |
| M8 Nino 3.4 | Criar tabela mensal de eventos historicos | Procede | Novo `features/nino34_event_table.py` com export mensal e comparacao de anos de pico |
| M9 Marine Heatwave | Criar catalogo MHW tabular | Procede como primeiro contrato | Novo `features/marine_heatwave.py` com `detect_mhw` e `export_mhw_catalog` |
| M10 Distribuicoes | Exportar CSV e recomendacao acionavel | Procede | `diagnose-distributions` grava CSV e adiciona `recommended_distribution`, `recommendation_confidence`, `physical_note` |
| M12 Mapas | Cada PNG deve ter CSV correspondente | Procede | `save_pixel_map` e `save_choropleth` exportam `.csv` antes do PNG por padrao |
| Headless/WSL | Plots opcionais nao devem depender de display | Procede | Backend matplotlib `Agg` nos modulos de mapa |
| Documentacao | Registrar convencao numeric-first | Procede | README atualizado com a regra e os novos artefatos |

## Backlog mantido

| Prioridade | Item | Motivo |
|---|---|---|
| Alta | `lagged_correlation_table` com Zarr/CSV por regiao | Metodo Fase 4 ainda nao existe no codigo |
| Alta | PCA/EOF numerico completo | Metodo Fase 4 ainda nao existe no codigo |
| Alta | Wrapper de ablation A-F com `ablation_comparison.csv` | Depende do wrapper de ablation mencionado no Parecer 1 |
| Media | SHAP por instancia de eventos historicos | Exige salvar valores SHAP por amostra, nao apenas media global |
| Media | MHW cruzado com anomalias NEB/Sul nos 90 dias seguintes | Depende dos cubos de precipitacao completos e convencao regional |
| Media | Flag global `--plot` em scripts analiticos | Ainda nao ha scripts de plot alem dos helpers de mapa |

## Validacao executada

- `.\.venv\Scripts\python.exe -m pytest -q`

Resultado: 21 testes passaram.
