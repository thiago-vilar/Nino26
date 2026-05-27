# Projeto de previsão climática El NINO e seus impactos no Brasil

**Programa de Pós Graduação em Oceanografia UFPE**  
**Área:** Oceanografia Física  
**Responsável:** Thiago Vilar  

## METHODOLOGY

### Objetivo metodológico

Medir e modelar a relação entre variáveis do Pacífico, variáveis atmosféricas e anomalias de precipitação no Brasil no período 1980-presente.

---

## 1. Padronização temporal

### 1.1 Frequência base

A frequência operacional será diária sempre que a fonte permitir.

Quando uma variável estiver em frequência mensal, usar uma das duas estratégias:

1. Agregar a chuva para escala mensal.

```text
chuva_mensal = soma da chuva diária dentro do mês
anomalia_mensal = chuva_mensal - climatologia_mensal
```

2. Expandir a métrica mensal para os dias do mês.

```text
valor_mensal_oceano -> repetido em todos os dias daquele mês
```

Regra:

- Para análise mensal, agregar a chuva.
- Para modelo diário com feature mensal auxiliar, repetir o valor mensal nos dias do mês.
- Nunca usar informação futura em simulação operacional.

---

## 2. Climatologia e anomalia

### 2.1 Climatologia diária

Para cada pixel e cada dia do ano:

```text
clim[doy, pixel] = média histórica do valor observado naquele dia do ano
```

Para reduzir ruído:

```text
clim[doy] = média dos valores entre doy - 7 e doy + 7 ao longo dos anos
```

### 2.2 Anomalia diária

```text
anomalia[t, pixel] = valor[t, pixel] - climatologia[doy(t), pixel]
```

Para SST:

```text
SSTA[t, x, y] = SST[t, x, y] - climatologia_SST[doy(t), x, y]
```

Para precipitação:

```text
anomalia_chuva[t, x, y] = chuva[t, x, y] - climatologia_chuva[doy(t), x, y]
```

### 2.3 Anomalia padronizada

```text
anomalia_padronizada = (valor - média_climatológica) / desvio_padrão_climatológico
```

---

## 3. Eventos de precipitação

Para cada pixel do Brasil:

```text
P10 = percentil 10 da chuva local
P90 = percentil 90 da chuva local
P95 = percentil 95 da chuva local
P99 = percentil 99 da chuva local
```

Classificação:

```text
evento_seco = chuva <= P10
chuva_alta = chuva >= P90
chuva_extrema = chuva >= P95 ou chuva >= P99
```

Acumulados:

```text
chuva_3d = soma dos últimos 3 dias
chuva_5d = soma dos últimos 5 dias
chuva_7d = soma dos últimos 7 dias
chuva_15d = soma dos últimos 15 dias
chuva_30d = soma dos últimos 30 dias
```

---

## 4. Variáveis CTD e ORAS/ORAS5

Entradas observadas:

```text
T(z) = temperatura por profundidade
S(z) = salinidade por profundidade
```

Variáveis calculadas:

```text
densidade potencial
D20
termoclina
camada de mistura
temperatura média 0-300 m
temperatura média 0-700 m
conteúdo de calor oceânico 0-300 m
conteúdo de calor oceânico 0-700 m
gradiente vertical de temperatura
```

Conteúdo de calor oceânico:

```text
OHC_0-H = integral[0,H] rho * cp * T(z) dz
```

Onde:

- `rho` é a densidade da água do mar.
- `cp` é o calor específico da água do mar.
- `H` é 300 m ou 700 m.

---

## 5. Variáveis atmosféricas

Núcleo mínimo:

```text
SLP, tau_x, tau_y, u10, v10, OLR, TCWV
```

Estrutura vertical:

```text
u850, v850, q850
u500, v500, q500, z500, omega500
u200, v200, z200, div200
```

Essas variáveis descrevem:

- circulação de baixos níveis;
- transporte de umidade;
- subsidência ou ascensão;
- jatos e teleconexões em altos níveis;
- convecção tropical;
- caminho atmosférico Pacífico -> Brasil.

---

## 6. Cruzamento Pacífico -> Brasil

Para cada lag:

```text
X[t] = variáveis do Pacífico e atmosfera no tempo t
Y[t + lag] = precipitação ou anomalia no Brasil
```

Lags iniciais:

```text
0, 7, 15, 30, 45, 60, 90, 120, 180 dias
```

Saídas:

```text
Y_regressao = anomalia de precipitação
Y_seca = evento_seco
Y_extremo = chuva_extrema
```

Escalas espaciais:

- pixel;
- estado;
- região;
- bacia;
- semiárido;
- Brasil inteiro.

---

## 7. Modelagem

Baselines:

- climatologia;
- persistência;
- correlação defasada;
- regressão regularizada;
- Random Forest;
- XGBoost;
- Pacífico embaralhado;
- lag invertido.

Modelos posteriores:

- CNN;
- ConvLSTM;
- U-Net;
- Transformer espaço-temporal.

---

## 8. Validação

Não usar split aleatório.

Usar:

```text
treino: anos iniciais
validação: anos intermediários
teste: anos finais
```

Walk-forward:

```text
treina até ano Y
testa ano Y + 1
avança a janela
repete
```

Avaliar por:

- lag;
- região;
- estação do ano;
- tipo de alvo;
- fonte de precipitação.

---

## 9. Métricas

Regressão:

- correlação;
- MAE;
- RMSE;
- bias.

Eventos:

- Brier Score;
- ROC-AUC;
- precision;
- recall;
- hit rate;
- false alarm rate;
- F1-score.

Mapas:

- correlação espacial;
- área de acerto;
- área de falso alarme;
- FSS quando aplicável.

---

## 10. XAI e diagnóstico físico

Perguntas:

- qual região do Pacífico foi mais importante?
- qual variável foi mais importante?
- qual profundidade foi mais importante?
- qual lag foi mais importante?
- o sinal é coerente fisicamente?

Técnicas:

- correlação defasada;
- permutation importance;
- SHAP;
- occlusion maps;
- saliency maps;
- attention maps;
- ablation study.

---

## 11. Correção recorrente automatizada

A cada nova rodada:

1. Comparar previsão com observado.
2. Atualizar métricas.
3. Atualizar mapas de erro.
4. Detectar drift.
5. Recalibrar probabilidades.
6. Atualizar mapas de confiança.
7. Gerar relatório automático.

