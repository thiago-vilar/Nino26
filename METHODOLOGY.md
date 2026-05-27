# Projeto de previsão climática El NINO e seus impactos no Brasil

**Programa de Pós Graduação em Oceanografia UFPE**  
**Área:** Oceanografia Física  
**Responsável:** Thiago Vilar  

## METHODOLOGY

O objetivo metodológico é medir, modelar e dimensionar o peso relativo das variáveis oceanográficas e atmosféricas associadas ao aquecimento do Pacífico sobre as anomalias de precipitação no Brasil.

## 1. Padronização temporal

A frequência operacional será diária sempre que a fonte permitir.

Regras:

- não usar informação futura em simulação operacional.
- registrar a cobertura temporal real de cada variável.
- manter apenas a interseção temporal válida quando necessário.
- documentar qualquer reamostragem temporal.

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

## 3. Eventos de precipitação

Para cada pixel do Brasil:

```text
P10 = percentil 10 da chuva local
P90 = percentil 90 da chuva local
```

Classificação:

```text
evento_seco = chuva <= P10
chuva_acima_normal = chuva >= P90
```

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
0, 7, 15, 30, 45, 60, 90, 120, 180 dias
```

Saídas:

```text
Y_regressao = anomalia de precipitação
Y_seca = evento_seco
Y_chuva_acima_normal = chuva_acima_normal
```

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

Modelos:

- climatologia.
- persistência.
- correlação defasada.
- regressão regularizada.
- Random Forest.
- XGBoost.
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

## 10. Correção recorrente automatizada

A cada nova rodada:

1. Comparar previsão com observado.
2. Atualizar métricas.
3. Atualizar mapas de erro.
4. Detectar drift.
5. Recalibrar probabilidades.
6. Atualizar mapas de confiança.
7. Gerar relatório automático.
