# Fase 5 - Ciclo ENSO com Machine Learning (RF/XGBoost) + XAI

**Coluna A da matriz metodologica, linha ML.** Repete o mecanismo da Fase 3
(estatistica) com Random Forest e XGBoost. Modulo: `src/nino_brasil/models/phase5_cycle_ml.py`;
script: `scripts/run_fase5_cycle_ml.py`.

## Pre-processamento
- **Janela deslizante** (`build_lagged_features`): cada semana `t` recebe as
  variaveis do Pacifico em `t-L`, para `L` de 4 a 52 semanas. So informacao passada
  entra em `t` (causalidade fisica).
- **Coerencia fisica** (`convert_precip_m_per_day_to_mm_month`): precipitacao `tp`
  (m/dia do ERA5) e convertida para mm acumulados antes do modelo (1 m/dia = 1000
  mm/dia; x dias do periodo), preservando o balanco hidrico.
- **Precursores de recarga**: `OHC0-300`, `SSH`, `D20`, `tau_x` nos lags 15-20 sem
  sao marcados (`__recharge`) e monitorados como eixo de recarga subsuperficial.

## Alvos (arquitetura preditiva dupla)
- Regressao: `Y_pico` (|ONI| maximo) e `Y_tempo_para_pico` (semanas de antecedencia).
- Duracao: `Y_duracao`, periodo continuo com ONI/OISST >= +0.5 C por >=5 estacoes
  moveis sobrepostas na regiao Nino 3.4.
- Classificacao semanal das 4 fases (genese/crescimento/pico/decaimento).

## Selecao e validacao
- **RFECV** (`rfecv_select`): eliminacao recursiva com validacao cruzada por
  ganho/impureza; nao assume importancia a priori.
- **Validacao cronologica exclusiva**: `TimeSeriesSplit` e leave-one-event-out
  (`leave_one_event_out_indices`). Proibido split aleatorio (vazamento futuro->passado).
- **Desbalanceamento** (~12 eventos severos): jitter gaussiano (`gaussian_jitter`) e
  SMOGN (`smogn_augment`, opcional) para regularizacao/reamostragem sintetica.

## Explicabilidade (XAI)
- **SHAP** (`shap_summary_values`): summary global (dominancia de SST, termoclina e
  vento) e force/waterfall locais para eventos atipicos (quedas em T100m/T150m/OHC
  sinalizando decaimento precoce antes da superficie).
- **PDP** (`partial_dependence_frame`): limiares nao-lineares do acoplamento de Bjerknes.

## Gate G2
RF/XGBoost so avancam se superarem a caracterizacao da Fase 3 e os baselines de
climatologia/persistencia.
