# Correções aplicadas ao código - revisão científica de 2026-07-10

Implementação dos itens P0/P1 do `PARECER_REVISAO_CIENTIFICA_2026-07-10.md`.
Todas as mudanças foram validadas por testes unitários novos
(`tests/test_phase5_phase6_fixes.py`, 9 testes) e por verificação numérica
contra os dados reais do projeto.

## 1. Bug do alvo de duração (Fase 5) - corrigido

`src/nino_brasil/models/phase5_cycle_ml.py::build_event_targets` agora calcula
`Y_duracao_sem` por diferença de datas `(fim - onset)/7`, com fallback
`duracao_estacoes x 4.348` (semanas por mês) para tabelas legadas sem `fim`.
Constante documentada: `WEEKS_PER_MONTH`. Verificação com a tabela real de
eventos:

| Evento | Antes (x13) | Depois (datas) |
|---|---:|---:|
| el_nino_1982_1983 | 143 sem | 47,7 sem |
| el_nino_1997_1998 | 143 sem | 47,6 sem |
| el_nino_2014_2016 | 247 sem | 82,4 sem |
| la_nina_1998_2001 | 442 sem | 147,7 sem |

`phase5_alvos_por_evento.csv` deve ser regenerado (`run_fase5_cycle_ml.py`).

## 2. Anomalização + detrend das variáveis oceânicas cruas

Novo `stats/climatology.py::harmonic_anomaly_matrix`: remove ciclo anual
(Fourier, 3 harmônicos) e tendência linear, ajustados APENAS na base 1991-2020
e extrapolados deterministicamente (sem vazamento do período de teste, conforme
parecer de 2026-05-26). Aplicado em três pontos:

1. `fase4_utils.load_pacific_weekly(anomalize_ocean=True)` - default novo; o
   notebook 4D (gate) e o inventário herdam automaticamente. A leitura antiga
   fica disponível com `anomalize_ocean=False`. `pacific_variable_inventory`
   ganhou a coluna `tratamento` para declarar a proveniência de cada variável.
2. `phase5_cycle_ml.prepare_pacific_predictors` - usado pelos runners das
   Fases 5 e 6 antes da janela deslizante.
3. A camada por unidade do 4C (`run_fase4c_regional.py`) já deseasonalizava via
   `stats/lag_analysis.harmonic_deseasonalize_predictors`; com o novo default a
   passagem dupla é matematicamente inócua (coeficientes ~0 na segunda), e o
   detrend passa a estar presente a montante.

Verificação no master real (amplitude sazonal antes -> depois): `d20_m`
24,9 m -> 3,4 m; `t200m` 1,23 °C -> 0,16 °C; `ohc_0_300` -85%. Sinal interanual
preservado (corr com SSTA: OHC0-300 0,77; D20 0,49) e tendência residual de
`t200m` ~ -0,007 °C/década (nula).

## 3. Baselines obrigatórios dos gates G2/G3

- **Fase 5** (`fit_phase_classifier`): cada fold reporta
  `f1_baseline_semana_do_ano` (classe modal por semana-do-ano, fit só no
  treino - captura o phase-locking) e `f1_baseline_persistencia` (rótulo da
  semana-calendário anterior). Legenda da Fig_5A1 reformulada para
  "caracterização diagnóstica" (rótulos post hoc não são previsão).
- **Fase 6** (`fit_unit_teleconnection`): além da média expansiva, reporta
  `rmse_persistencia`, `rmse_clim_semana_do_ano`, `melhor_baseline`,
  `rmse_melhor_baseline` e `skill_rmse_vs_melhor_baseline`. Critério do gate
  G3 nas próprias tabelas: `skill_rmse_vs_melhor_baseline > 0`.

## 4. Regressões por evento da Fase 5 - agora treinadas

Novo `phase5_cycle_ml.fit_event_regressions`: leave-one-event-out para
`Y_pico`, `Y_tempo_para_pico_sem` e `Y_duracao_sem`, com jitter gaussiano
APENAS no treino, baseline de climatologia dos eventos de treino (mesmo
protocolo da Fase 3), RF regularizado (`max_features='sqrt'`,
`min_samples_leaf=2`) ou XGBoost. O runner grava
`phase5_regressao_eventos_{modelo}.csv`, `phase5_regressao_predicoes_{modelo}.csv`
e `phase5_regressao_importancias_{modelo}.csv`.

## 5. Escrita atômica de CSVs

Novo `src/nino_brasil/io_utils.py::write_csv_atomic` (tmp + `os.replace`),
adotado nos runners das Fases 5 e 6 - o leitor nunca vê CSV parcial, como o
`phase6_skill_rf.csv` truncado encontrado na revisão.

## 6. Como reexecutar (ordem recomendada)

```powershell
.\.venv\Scripts\python -m pytest tests\test_phase5_phase6_fixes.py -q
.\.venv\Scripts\python scripts\run_fase4c_regional.py            # 4C com detrend a montante
.\.venv\Scripts\python scripts\run_fase4_all.py                  # 4D/gate com anomalias
.\.venv\Scripts\python scripts\export_numeric_tables_for_figures.py --force --strict
.\.venv\Scripts\python scripts\run_fase5_cycle_ml.py             # + --model xgb
.\.venv\Scripts\python scripts\run_fase6_brazil_ml.py            # + --model xgb
```

Só depois disso reler o gate G1 (em especial o `contraria_hipotese` do NEB,
suspeito de confundimento sazonal/tendência na leitura antiga) e formalizar a
decisão em documento.

## 7. O que NÃO foi alterado

Nada da Fase 3 (madura), nada da máquina 4C pixel (`lagged_corr_pixel_matrix`,
N_eff/FDR), nenhum notebook `.ipynb` (herdam as mudanças via `fase4_utils` e
runners), nenhum dado processado (as tabelas antigas continuam em disco até a
reexecução). A comparação com a leitura antiga permanece reprodutível via
`anomalize_ocean=False`.
