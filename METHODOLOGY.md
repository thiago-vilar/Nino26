# Projeto de previsão climática El NINO e seus impactos no Brasil

**Programa de Pós Graduação em Oceanografia UFPE**  
**Área:** Oceanografia Física  
**Responsável:** Thiago Vilar  

## METHODOLOGY

O objetivo metodológico é medir, modelar e dimensionar o peso relativo das variáveis oceanográficas e atmosféricas associadas ao aquecimento do Pacífico sobre as anomalias de precipitação no Brasil.

## 1. Padronização temporal

A janela temporal mestre do projeto é diária, de `1981-01-01` até o último dado disponível por fonte.

Regras:

- não usar informação futura em simulação operacional.
- todo produto de modelagem deve estar alinhado ao eixo diário 1981-latest_available.
- registrar a cobertura temporal real de cada variável antes do alinhamento.
- registrar lacunas de fonte sem preenchimento silencioso.
- converter ERA5 subdiário para estatísticas diárias.
- alinhar ORAS5 mensal, baixado no historico por ano/variavel, como variavel de memoria oceanica com regra explicita de lag.
- avaliar previsões em horizontes semanais de 1 a 24 semanas, equivalentes a 7-168 dias.
- documentar qualquer reamostragem temporal.
- registrar a latência de cada fonte no ledger de auditoria.

## 2. Climatologia e anomalia

Climatologia diária:

```text
clim[doy, pixel] = média histórica do valor observado no dia do ano
```

Anomalia diária:

```text
anomalia[t, pixel] = valor[t, pixel] - clim[doy(t), pixel]
```

Anomalia padronizada:

```text
anomalia_padronizada = (valor - média_climatológica) / desvio_padrão_climatológico
```

Regra anti-vazamento:

```text
climatologia, desvio padrão climatológico, P10, P25, P75 e P90 são estimados apenas no bloco de treino de cada fold walk-forward e reaplicados à validação/teste.
```

## 3. Eventos de precipitação

Para cada pixel do Brasil:

```text
P10 = percentil 10 da chuva local
P25 = percentil 25 da chuva local
P75 = percentil 75 da chuva local
P90 = percentil 90 da chuva local
```

Classificação:

```text
seco_extremo = chuva <= P10
abaixo_quartil_inferior = chuva <= P25
faixa_interquartil = P25 < chuva < P75
acima_quartil_superior = chuva >= P75
umido_extremo = chuva >= P90
```

A faixa `P25-P75` é uma referência interquartil da distribuição local, não uma classe meteorológica chamada "normal".

Acumulados:

```text
chuva_3d, chuva_5d, chuva_7d, chuva_15d, chuva_30d
```

## 4. Variáveis oceanográficas

Entradas:

- `SST`: temperatura da superfície do mar.
- `SSTA`: anomalia da temperatura da superfície do mar.
- `SSHA`: anomalia da altura da superfície do mar.
- `T(z)`: temperatura por profundidade.
- `S(z)`: salinidade por profundidade.
- `u/v oceânico`: correntes zonal e meridional.
- `clorofila-a`: variável auxiliar superficial.

Variáveis derivadas:

- densidade potencial.
- `D20`: profundidade da isoterma de 20 graus Celsius.
- termoclina.
- `MLD`: profundidade da camada de mistura.
- temperatura média `0-300 m`.
- temperatura média `0-700 m`.
- `OHC 0-300 m`: conteúdo de calor oceânico até 300 m.
- `OHC 0-700 m`: conteúdo de calor oceânico até 700 m.
- gradiente vertical de temperatura.

Conteúdo de calor oceânico:

```text
OHC_0-H = integral[0,H] rho * cp * T(z) dz
```

## 4.1 CTD NOAA WOD e TEOS-10

Os perfis CTD do WOD entram como observacao vertical irregular, nao como grade diaria continua.

Fluxo computacional:

```text
1. baixar wod_ctd_<ano>.nc do NOAA WOD;
2. filtrar perfis no Pacifico 35S-30N / 120E-70W;
3. aceitar somente flags WOD aprovadas;
4. interpolar temperatura e salinidade para 0-700 m com passo de 5 m;
5. calcular pressao a partir de profundidade e latitude;
6. aplicar TEOS-10;
7. salvar Zarr anual.
```

Variaveis TEOS-10:

```text
SA = Salinidade Absoluta
CT = Temperatura Conservativa
sigma0 = anomalia de densidade potencial referida a 0 dbar
```

Estratificacao:

```text
termoclina = profundidade do maior gradiente vertical de CT
haloclina = profundidade do maior gradiente vertical de SA
picnoclina = profundidade do maior gradiente vertical de sigma0
```

## 4.2 Validacao TAO/TRITON/Argo

TAO/TRITON/Argo nao muda a direcao do projeto e nao substitui OISST, ORAS5 ou CTD/WOD. Essa camada entra como validacao independente para manter aberta a pergunta cientifica: a subsuperficie do Pacifico Nino 3.4 melhora a explicacao das anomalias de chuva/seca no Brasil, ou SST/SSTA superficial e suficiente?

Uso metodologico:

```text
1. comparar ORAS5 contra observacoes in situ no Nino 3.4;
2. validar D20, termoclina, OHC 0-300 m e temperatura media 0-300 m;
3. identificar periodos em que CTD/WOD e escasso sem preencher lacunas artificialmente;
4. rodar ablation com e sem variaveis subsuperficiais em walk-forward.
```

Papel das fontes:

```text
TAO/TRITON/GTMBA = fundeios equatoriais, bons para serie temporal subsuperficial;
Argo GDAC = perfis T/S independentes, mais fortes a partir dos anos 2000;
WOD/GTSPP = agregadores uteis para auditoria, com cuidado para duplicatas.
```

## 5. Variáveis atmosféricas

Grupos:

```text
Superfície: SLP, tau_x, tau_y, u10, v10
Baixos níveis: u850, v850, q850
Níveis médios: u500, v500, q500, z500, omega500
Altos níveis: u200, v200, z200, div200
Umidade e convecção: TCWV, OLR
```

Essas variáveis descrevem circulação, transporte de umidade, subsidência, ascensão, teleconexões e convecção tropical.

## 6. Cruzamento Pacífico -> Brasil

Para cada lag:

```text
X[t] = variáveis oceanográficas + variáveis atmosféricas
Y[t + lag] = anomalia de precipitação no Brasil
```

Lags iniciais:

```text
1 a 24 semanas: 7, 14, 21, ..., 168 dias
```

As grades do Pacífico, Brasil e ponte atmosférica são reconciliadas antes da montagem da matriz de modelagem. As Fases 1 a 7 usam grade comum `0.25°` em longitude `0_360`; CHIRPS `0.05°` fica reservado para experimento futuro de alta resolução depois que o fluxo `0.25°` estiver validado.

Saídas:

```text
Y_regressao = anomalia de precipitação
Y_seca_extrema = chuva <= P10
Y_abaixo_quartil = chuva <= P25
Y_acima_quartil = chuva >= P75
Y_chuva_extrema = chuva >= P90
```

Hipótese regional prioritária: El Nino alto tende a aumentar a probabilidade de seca no Nordeste/Semiárido e a probabilidade de chuva acima do quartil superior/extremos úmidos no Sul do Brasil. Essa hipótese deve ser testada por lag semanal, estação do ano e pixel/região.

## 7. Modelagem e dimensionamento de pesos

Grupos avaliados:

```text
Oceano superficial: SST, SSTA, SSHA
Oceano subsuperficial: T(z), S(z), D20, OHC, termoclina, MLD
Atmosfera de superfície: SLP, tau_x, tau_y, u10, v10
Atmosfera em baixos níveis: u850, v850, q850
Atmosfera em níveis médios: u500, v500, q500, z500, omega500
Atmosfera em altos níveis: u200, v200, z200, div200
Umidade e convecção: TCWV, OLR
```

Experimentos mínimos:

```text
Modelo A: apenas variáveis oceanográficas
Modelo B: apenas variáveis atmosféricas
Modelo C: oceanográficas + atmosféricas
Modelo D: sem subsuperfície oceânica
Modelo E: sem altos níveis atmosféricos
Modelo F: sem umidade atmosférica
```

Fase 3 - diagnostico fisico Nino 3.4:

- alinhamento temporal de anomalias oceanicas e atmosfericas.
- volume/grau de agua quente anomala no Nino 3.4.
- D20, termoclina, gradiente vertical e profundidade de maior estratificacao.
- slope longitudinal/vertical do sinal termico.
- duracao do sinal anomalo por evento.
- comparacao com os picos historicos de El Nino mais impactantes da serie.

Fase 4 - pre-analises estatisticas experimentais:

- regressao multipla.
- PCA/EOF para modos dominantes de covariacao.
- KNN para similaridade de eventos e analogos historicos.
- correlacao defasada por lag semanal.
- ablation screening de variaveis individuais e combinadas.

Modelos da Fase 5:

- climatologia.
- persistência.
- correlação defasada.
- regressão regularizada.
- Random Forest.
- XGBoost.

Modelos da Fase 6, Redes Neurais + XAI:

- CNN.
- ConvLSTM.
- U-Net.
- Transformer espaço-temporal.

## 8. Validação

Regras:

- não usar split aleatório.
- usar treino, validação e teste por blocos temporais.
- aplicar walk-forward validation.
- avaliar por lag.
- avaliar por região.
- avaliar por estação do ano.
- comparar fontes de precipitação quando houver mais de uma.

Métricas:

- correlação.
- MAE.
- RMSE.
- bias.
- Brier Score.
- ROC-AUC.
- precision.
- recall.
- hit rate.
- false alarm rate.
- F1-score.

## 9. XAI e diagnóstico físico

Perguntas de XAI:

- qual região do Pacífico teve maior peso?
- qual grupo teve maior peso: oceanográfico ou atmosférico?
- qual variável oceanográfica foi mais importante?
- qual variável atmosférica foi mais importante?
- qual profundidade foi mais importante?
- qual nível atmosférico foi mais importante?
- qual lag foi mais importante?
- o sinal é coerente fisicamente?

Técnicas:

- correlação defasada.
- permutation importance.
- SHAP.
- occlusion maps.
- saliency maps.
- attention maps.
- ablation study.

Saídas operacionais da Fase 3:

```text
nino34_physical_signal.zarr
nino34_thermocline_diagnostics.zarr
nino34_peak_signal_comparison.zarr
nino34_signal_slope_duration.zarr
```

Esses arquivos alimentam a Fase 4 e mantem explicita a pergunta: a subsuperficie do Nino 3.4 melhora a explicacao das anomalias de chuva no Brasil ou a SST/SSTA superficial e suficiente?

Saídas operacionais da Fase 4:

```text
phase4_variable_screening.zarr
phase4_multiple_regression.zarr
phase4_pca_modes.zarr
phase4_knn_similarity.zarr
```

Esses stores Zarr geram um ranking experimental das variaveis Nino 3.4, individuais e combinadas, associadas a seca no Nordeste e chuva no Sul.

Saídas operacionais da Fase 5:

```text
walk_forward_metrics.zarr
walk_forward_predictions.zarr
walk_forward_importances.zarr
walk_forward_group_weights.zarr
```

Esses arquivos alimentam os mapas de peso oceanográfico, peso atmosférico e variável dominante.

Saídas operacionais da Fase 6:

```text
neural_training_runs.zarr
neural_predictions.zarr
neural_xai_maps.zarr
neural_skill_comparison.zarr
```

Esses stores Zarr alimentam a comparação entre redes neurais, ML clássico e XAI físico.

## 9.1 Diagnóstico distribucional

Algumas variáveis físicas podem ter caudas pesadas. O diagnóstico de cauda é separado do treino e serve para QC/EDA:

```text
variáveis prioritárias: chuva diária, extremos >= P90, OHC, SSTA
ajustes: power law, lognormal, exponencial
critérios: alpha, xmin, KS, razão de log-verossimilhança
```

A comparação contra lognormal e exponencial é obrigatória; uma cauda com aparência de lei de potência não deve ser assumida sem teste comparativo.

## 10. Correção recorrente automatizada

A cada nova rodada:

1. Comparar previsão com observado.
2. Atualizar métricas.
3. Atualizar mapas de erro.
4. Detectar drift.
5. Recalibrar probabilidades.
6. Atualizar mapas de confiança.
7. Gerar relatório automático.
