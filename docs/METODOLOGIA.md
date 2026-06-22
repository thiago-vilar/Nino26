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
- usar NOAA UFS/GLORYS como oceano originalmente diario e ORAS5 como memoria mensal independente, mantendo fonte, frequencia e resolucao explicitamente identificadas.
- nunca promover ORAS5 mensal a observacao diaria; para modelos diarios, liberar apenas o ultimo mes ja publicado com latencia causal.
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

TAO/TRITON/Argo nao muda a direcao do projeto e nao substitui OISST, os cubos oceânicos diarios ou CTD/WOD. Essa camada entra como validacao independente para manter aberta a pergunta cientifica: a subsuperficie do Pacifico Nino 3.4 melhora a explicacao das anomalias de chuva/seca no Brasil, ou SST/SSTA superficial e suficiente?

Uso metodologico:

```text
1. comparar NOAA UFS, GLORYS12V1 e ORAS5 contra observacoes in situ no Nino 3.4, respeitando a frequencia de cada fonte;
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

## 6. Aprendizado em duas etapas

A fase de IA fica separada em duas perguntas. Essa separacao evita que o
modelo tente aprender ao mesmo tempo a dinamica do El Nino e o impacto
regional no Brasil.

Etapa 5A - Modelo de progressao ENSO:

```text
trajetoria diaria oceano-atmosfera do Nino3.4 em t -> pico futuro do El Nino
```

O sinal dinamico principal da Etapa 5A deve vir da SSTA diaria Nino3.4
extraida do OISST para 1981-latest_available. O indice mensal NOAA PSL/ERSST
v6 entra como referencia oficial de rotulo/pico, nao como unica variavel de
aprendizado. A serie diaria preserva rampa, aceleracao, persistencia,
reversoes curtas e duracao do aquecimento.

Alvos minimos:

```text
future_peak_ssta
days_to_peak
future_peak_class
will_el_nino
will_strong_el_nino
will_super_el_nino
```

O treino deve ser evento-centrado. O modelo precisa aprender trajetorias de
aquecimento, recarga de calor, D20, OHC, WWV, slope, ventos e duracao do sinal
que antecedem picos historicos, especialmente Super El Nino.

Etapa 5B - Modelo de teleconexao Nino3.4 -> Brasil:

```text
estado/fase/intensidade do Nino3.4 em t -> eventos climaticos em clusters de pixels do Brasil em t + lag
```

O alvo regional nao deve ser apenas media do Brasil. Primeiro os pixels sao
clusterizados por comportamento de eventos secos/umidos; depois cada cluster
vira uma unidade de previsao com taxa/probabilidade de evento.

Saidas numericas minimas:

```text
data/processed/zarr/features/noaa_psl_nino34_reference_peaks.zarr
data/processed/zarr/features/nino34_daily_oisst.zarr
data/processed/zarr/modeling/enso_peak_progression.zarr
data/processed/zarr/modeling/nino34_cluster_progression.zarr
```

## 6.1 Cruzamento Pacífico -> Brasil

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

Modelos da Fase 5A - progressao ate pico ENSO:

- climatologia e persistencia como baselines.
- regressao regularizada para intensidade futura do pico.
- Random Forest e LightGBM para classe/probabilidade de El Nino forte ou Super El Nino.
- leave-one-event-out para eventos principais.
- XAI por variavel fisica, horizonte mensal e evento historico.

Modelos da Fase 5B - Nino3.4 para clusters de pixels no Brasil:

- climatologia.
- persistencia.
- correlacao defasada.
- regressao regularizada.
- Random Forest.
- LightGBM/XGBoost.
- clusterizacao de pixels por frequencia, sazonalidade e intensidade de eventos secos/umidos.
- avaliacao por cluster, lag, estacao e classe de evento.

Modelos legados da Fase 5 para comparacao:

- climatologia.
- persistência.
- correlação defasada.
- regressão regularizada.
- Random Forest.
- XGBoost.

Modelos da Fase 6, Redes Neurais + XAI:

Fase 6A - CNN espacial nativa NINO-BRASIL:

- objetivo: criar um baseline neural nativo para prever a progressao ate o pico ENSO.
- arquitetura: encoder CNN espacial multihorizonte com cabecas densas para regressao/classificacao evento-centrada.
- entrada: campos diarios OISST/SSTA, trajetoria diaria Nino3.4, ERA5 ponte atmosferica e oceano diario/D20/OHC/WWV quando disponiveis.
- janelas candidatas: 90, 180, 365 e 540 dias antes da origem.
- alvos: `future_peak_ssta`, `days_to_peak`, `future_peak_class`, `will_strong_el_nino`, `will_super_el_nino` e estado `P90`/MHW da SSTA Nino3.4.

Fase 6B - memoria espaco-temporal da progressao ENSO:

- objetivo: aprender a rampa, aceleracao, persistencia, reversoes curtas e duracao do aquecimento ate o topo do El Nino.
- modelos candidatos: ConvLSTM, Temporal CNN e 1D-Transformer sobre series fisicas agregadas.
- variaveis-chave: SSTA, tendencia de SSTA, medias moveis 7/30/90/180 dias, duracao acima de limiar, excedencia `P90`, D20, OHC, WWV, ventos e SLP.
- validacao: blocked walk-forward, leave-one-major-event-out e leave-one-super-el-nino-out.
- pergunta central: o modelo antecipa o pico por processo fisico ou apenas memoriza anos fortes?

Fase 6C - teleconexoes neurais Nino3.4 -> Brasil:

- objetivo: usar o estado latente ENSO aprendido em 6A/6B para prever impactos por clusters de pixels no Brasil.
- entrada: estado ENSO, trajetoria Nino3.4, ponte atmosferica ERA5, sazonalidade e metadados dos clusters.
- alvos por cluster: probabilidade `P10` seco extremo, `P25` chuva baixa, `P75` chuva alta, `P90` chuva extrema e anomalia de precipitacao.
- o alvo `P90` e prioritario para chuva extrema no Sul e para avaliar se a rede entende a cauda superior da distribuicao local.
- avaliacao: Brier, ROC-AUC, hit rate, false alarm rate, skill por lag, por estacao e por cluster.

Fallback de pre-treino/fine-tuning CMIP6:

- CMIP6 entra apenas se 6A/6B/6C nao superarem climatologia, persistencia e Fase 5 em validacao evento-centrada.
- qualquer uso de CMIP6 exige anomalias e correcao de vies por modelo, pre-treino separado, fine-tuning em observacao/reanalise e relatorio separado de skill nativo vs fine-tuned.
- split aleatorio continua proibido.

Fase 8 - exploracao adicional Ham2019:

- objetivo: testar pesos salvos, tensores compativeis e dados associados a Ham2019/reproducao em uma bancada separada.
- a Fase 8 nao alimenta o treino principal da Fase 6 e nao substitui os modelos nativos.
- usos permitidos: inventario de SavedModels, leitura de shapes/assinaturas, inferencia congelada se houver compatibilidade de entrada, comparacao de skill contra Fases 5/6 e relatorio de limites de transferencia.
- os pesos baixados ficam em `data/raw/external/ham2019_reproduction_weights`.
- qualquer dado externo adicional de Ham2019 deve ser marcado como exploratorio, ter licenca revisada e ficar em namespace separado.
- resultados devem ser reportados como `ham2019_exploratory_*`, nunca misturados com `neural_6a/6b/6c`.

## 8. Validação

Regras:

- não usar split aleatório.
- usar treino, validação e teste por blocos temporais.
- aplicar walk-forward validation.
- aplicar leave-one-event-out na Etapa 5A para picos El Nino/Super El Nino.
- avaliar a Etapa 5B por clusters de pixels, sem tratar pixels do mesmo cluster como amostras independentes.
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
noaa_psl_nino34_reference_peaks.zarr
nino34_daily_oisst.zarr
enso_peak_progression.zarr
nino34_cluster_progression.zarr
walk_forward_metrics.zarr
walk_forward_predictions.zarr
walk_forward_importances.zarr
walk_forward_group_weights.zarr
```

Esses arquivos alimentam os mapas de peso oceanográfico, peso atmosférico e variável dominante.

Os dois primeiros stores da Fase 5 sao os contratos principais da nova fase:
primeiro progressao ate pico ENSO; depois teleconexao Nino3.4 -> clusters de
pixels no Brasil. Os stores walk-forward ficam como avaliacao, comparacao e XAI.

Saídas operacionais da Fase 6:

```text
neural_training_runs.zarr
neural_predictions.zarr
neural_xai_maps.zarr
neural_skill_comparison.zarr
neural_6a_enso_cnn_training_runs.zarr
neural_6a_enso_cnn_predictions.zarr
neural_6b_enso_memory_training_runs.zarr
neural_6b_enso_memory_predictions.zarr
neural_6c_brazil_cluster_training_runs.zarr
neural_6c_brazil_cluster_predictions.zarr
neural_6c_brazil_cluster_p90_metrics.zarr
```

Esses stores Zarr alimentam a comparação entre redes neurais, ML clássico e XAI físico.

Saidas operacionais da Fase 8:

```text
ham2019_exploratory_weight_inventory.zarr
ham2019_exploratory_predictions.zarr
ham2019_exploratory_skill_comparison.zarr
ham2019_exploratory_report.csv
```

Esses arquivos sao exploratorios e nao definem a habilidade operacional do NINO-BRASIL.

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
