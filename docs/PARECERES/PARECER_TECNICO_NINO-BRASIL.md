# Parecer técnico — NINO-BRASIL (crítica construtiva para correção)

> Instruções ao agente corretor: este documento lista, em ordem de prioridade, as correções a aplicar no repositório `C:\DEV\NINO26`. Trate cada item como uma tarefa acionável. Não descarte nenhum objetivo específico do plano original — itens de grande porte foram reorganizados em fases, não removidos. Preserve a arquitetura existente (`data/features/models/maps/web`, `configs/project.yaml`, catálogo, ledger de auditoria, modo dry-run e Zarr).

## Contexto do estado atual

Fundação sólida: scaffold, camada de download/ETL (CHIRPS, OISST, CTD/WOD com TEOS-10, ERA5/ORAS via CDS, IBGE) e feature engineering (climatologia, anomalias, OHC, D20, termoclina, MLD, eventos por percentil, alinhamento X(t)→Y(t+lag)) estão implementados e fisicamente coerentes. Baselines Ridge/RF e split temporal existem. **Lacuna principal:** a camada científica (montagem da matriz de features, treino, walk-forward, classificação, XGBoost, XAI) ainda não existe em código. Em dados, só IBGE e 1 ano de CHIRPS (em resolução divergente) foram baixados.

## 1. Correções metodológicas (corrigir antes de treinar qualquer modelo)

1. **Vazamento de informação (crítico).** `dayofyear_climatology` e os limiares de percentil P10/P90 são calculados sobre todo o registro. Como anomalias e eventos viram features/alvos, isso vaza o teste para o treino. **Correção:** estimar climatologia e percentis **apenas no bloco de treino** e reaplicá-los a validação/teste dentro do loop walk-forward.
2. **Inconsistência de padronização.** `standardized_anomaly` recalcula média/desvio sem a suavização por janela usada em `dayofyear_climatology`. **Correção:** padronizar a mesma base climatológica suavizada.
3. **Bug de RMSE (quebra em execução).** `mean_squared_error(..., squared=False)` foi removido no scikit-learn ≥ 1.4. **Correção:** usar `root_mean_squared_error`.
4. **Grades não reconciliadas.** Pacífico, Brasil e ponte atmosférica usam grades/convenções de longitude distintas. **Correção:** adicionar etapa explícita de *regridding* (usar `xesmf`, já nas dependências) para uma grade-alvo comum, definida em `project.yaml`, antes de montar a matriz de modelagem.
5. **Redundância de SST.** OISST (diário) e ORAS5 (SST mensal) cobrem SST. **Correção:** definir OISST como SST/SSTA principal e reservar ORAS apenas para subsuperfície.

## 2. Correções de dados e escopo

6. **Resolução do CHIRPS divergente.** Plano e catálogo pedem p05 (0.05°), mas o único arquivo baixado é p25 (0.25°). **Correção:** alinhar docs/código/catálogo a uma escolha única e coerente com a grade-alvo de modelagem (ver fases).
7. **Janela temporal sem dado futuro.** O fim fixo em 2026-12-31 é irreal (estamos em 05/2026) e reanálises/CHIRPS têm latência. **Correção:** tornar o fim da janela **dinâmico** ("até o último disponível por fonte") e registrar a latência de cada fonte no ledger. **Não reduzir** a extensão histórica da janela.

## 3. Camada de modelagem ausente (implementar — maior prioridade de código)

8. `build_feature_matrix`: montar X a partir dos cubos Zarr já reconciliados na grade comum.
9. Loop de **walk-forward** com blocos temporais (sem split aleatório), avaliando por lag, região e estação.
10. **XGBoost/LightGBM** além de Ridge/RF; **cabeça de classificação** para seca (≤P10) e chuva acima do normal (≥P90).
11. Métricas de classificação ausentes: Brier, ROC-AUC, precision/recall, hit rate, false alarm rate, F1.
12. **XAI:** permutation importance + SHAP para o dimensionamento de pesos oceano × atmosfera.
13. **Testes** das funções de feature (OHC, D20, anomalia) com dados sintéticos.
14. **Sanidade física:** incluir índice Niño 3.4 pronto como baseline e validação cruzada do sinal aprendido.

## 4. Reestruturação do plano em fases (preservando todos os objetivos)

Nenhum objetivo do plano é descartado; são distribuídos em fases para viabilizar execução individual. O ciclo ponta-a-ponta (dado → modelo → mapa → GitHub Pages) deve fechar já na Fase 1.

### 4.1 Fase 1 — MVP correto e ponta-a-ponta
- Resolução de modelagem reduzida (0.25°) ou agregação regional/UF como primeiro alvo; a pergunta de pesquisa permanece intacta.
- Dados: OISST + CHIRPS + IBGE (já permitem o ciclo completo).
- Modelos: Ridge, RF e **XGBoost** + permutation/SHAP.
- Ablations iniciais: A (oceano), B (atmosfera), C (oceano+atmosfera).
- **Manter a janela temporal histórica completa 1981–presente (não limitar).**

### 4.2 Resolução temporal da análise
- **Trabalhar sempre em resolução diária.** Não introduzir análise em escala mensal — é inútil para o objetivo de machine learning deste projeto.

### 4.3 Estrutura de lags
- **Definir os lags em base mensal** (ex.: 0, 1, 2, 3, 4, 5, 6 meses → 0, 30, 60, 90, 120, 150, 180 dias), coerente com a memória oceânica do ORAS5, mantendo o alvo e os preditores em resolução diária.

### 4.4 Fase 2 — Expansão de variáveis e resolução (objetivos preservados)
- Incorporar ERA5 completo (single + pressure levels) e ORAS5 (subsuperfície).
- Ablations D (sem subsuperfície), E (sem altos níveis), F (sem umidade).
- Escalar a modelagem para 0.05° pixel-a-pixel (produto final do plano).

### 4.5 Fase 4 — Redes Neurais + XAI avançado (objetivos preservados)
- Modelos: CNN, ConvLSTM, U-Net, Transformer espaço-temporal.
- XAI avançado: occlusion maps, saliency maps, attention maps, ablation study completo.

### 4.6 Fase 5 — Teste com Memory Caching (Google Research)
- Avaliar a técnica de test-time memorization "Memory Caching: RNNs with Growing Memory" (Behrouz et al., arXiv:2602.24281): cache de estados de memória por segmento, variantes Residual Memory, Gated Residual Memory, Memory Soup e Sparse Selective Caching.
- Comparar contra os campeões das Fases 3 e 4 sob o mesmo protocolo walk-forward, medindo o trade-off skill × eficiência computacional (interpolação O(L) a O(L²)).

### 4.7 Fase 6 — Publicação e operação
- Correção recorrente automatizada: comparação previsão×observado, drift, recalibração, mapas de confiança, relatório automático.
- Publicação em `docs/` via GitHub Pages.

### 4.8 Janela temporal
- **Não limitar a janela temporal.** Manter 1981 até o último dado disponível em todas as fases; aplicar apenas o fim dinâmico do item 7 (latência de fonte), sem recortar o período histórico.

## Conclusão

A fundação e a metodologia estão essencialmente corretas; o risco é de escopo e execução, não conceitual. Aplicar primeiro as correções das seções 1 e 2 (vazamento, padronização, RMSE, regridding, resolução do CHIRPS, janela dinâmica), depois implementar a camada de modelagem da seção 3, e conduzir o desenvolvimento pelas fases da seção 4 — sem descartar nenhum objetivo específico do plano original.
