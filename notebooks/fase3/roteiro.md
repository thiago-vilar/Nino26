# Roteiro local da Fase 3

Projeto NINO-BRASIL - Diagnostico fisico do Nino 3.4.

## Execucao

```cmd
cd /d C:\DEV\NINO26
.venv\Scripts\python scripts\fase3_build_inputs.py --force
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3A_indices_fisicos_semanais.ipynb --inplace --ExecutePreprocessor.timeout=1200
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3B_alvo_eventos_ciclo_vida.ipynb --inplace --ExecutePreprocessor.timeout=1200
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3C_precursores_lags.ipynb --inplace --ExecutePreprocessor.timeout=1200
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3D_rigor_estatistico.ipynb --inplace --ExecutePreprocessor.timeout=1200
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3E_estabilidade_subperiodos.ipynb --inplace --ExecutePreprocessor.timeout=1200
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3F_dhw_kelvin.ipynb --inplace --ExecutePreprocessor.timeout=1200
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3G_ciclo_vida_dhw.ipynb --inplace --ExecutePreprocessor.timeout=1200
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3H_genese_precursores_classe.ipynb --inplace --ExecutePreprocessor.timeout=1200
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3K_pca_crescimento.ipynb --inplace --ExecutePreprocessor.timeout=1200
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3I_interpretacao_integrada.ipynb --inplace --ExecutePreprocessor.timeout=1200
.venv\Scripts\python scripts\generate_phase3_report.py
.venv\Scripts\python scripts\audit_phase3_temporal_integrity.py
.venv\Scripts\python -m pytest -q
```

## Regra NOAA/ONI local

Evento El Nino = media movel de 3 meses da SSTA Nino 3.4 >= +0.5 C por 5+
estacoes moveis sobrepostas. Intensidade:

| Classe | Pico ONI local |
|---|---|
| fraco | 0.5 C <= pico < 1.0 C |
| moderado | 1.0 C <= pico < 1.5 C |
| forte | 1.5 C <= pico < 2.0 C |
| muito_forte / super | pico >= 2.0 C |

## DHW canonico

Use somente `dhw_cweek_0p5_12w`.

1. HotSpot diario = SSTA diaria quando SSTA >= +0.5 C.
2. Valores abaixo de +0.5 C sao descartados como 0.
3. Acumulo = janela movel de 12 semanas em C-weeks.
4. O valor so e reportado depois de 12 semanas diarias consecutivas com SSTA >= +0.5 C.
5. `oni_12w_mean_c` permanece apenas como coluna de auditoria, nao como preditor.

## Leitura por etapa

| Notebook | O que responde |
|---|---|
| 3A | Cobertura e forma das variaveis; Hovmoller SSTA e Hovmoller SLA/tau_x. |
| 3B | Eventos NOAA/ONI locais, ciclo de vida e autocorrelacao/persistencia da SSTA. |
| 3C | Ranking bruto de lags, ordenado pelo maior abs(r), e mapa longitude-lag. |
| 3D | Relacoes que sobrevivem a N_eff, FDR e IC95. |
| 3E | Estabilidade das relacoes entre 1993-2009 e 2010-presente. |
| 3F | DHW como severidade/memoria e leitura qualitativa de ondas de Kelvin por SLA/SSH. |
| 3G | DHW por classe e comparacao dos fortes/super com a formacao atual 2025/26. |
| 3H | Genese por classe e ciclo de vida alinhado ao pico real. |
| 3I | Sintese: quais variaveis antecipam o aquecimento maximo e como ler 2025/26. |
| 3K | PCA para reduzir redundancia entre variaveis fisicas. |

## Padrao de figuras

Cada notebook gera ao menos uma figura padronizada em
`data/processed/figures/fase3/`. A convencao e `3A1_...png`: Fase 3, notebook
A, figura 1. Se o notebook precisar de mais imagens do mesmo subcontexto, use
`3A2`, `3A3` etc. O indice completo e regenerado em
`notebooks/fase3/INDICE_FIGURAS_FASE3.md`.

## Coordenadas

Nos mapas e Hovmollers, o eixo x segue 120E -> 80W, oeste para leste:
120E, 140E, 160E, 180, 160W, 140W, 120W, 100W, 80W. A faixa Nino 3.4 fica
sombreada em 170W-120W.

## Interpretacao executiva

O conjunto mais importante para antecipar o aquecimento maximo e o bloco de
recarga/subsuperficie: D20, SSH, OHC, WWV e tilt. `tau_x_anom_nino34_pa` mede
acoplamento vento-superficie. DHW confirma persistencia e severidade acumulada,
mas nao e previsor longo isolado. A Fase 3 agora entrega uma projecao
condicional exploratoria por nested LOO; a previsao operacional de timing +
amplitude fica para a Fase 5.
