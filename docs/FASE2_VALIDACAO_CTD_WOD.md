# Fase 2 - Validacao CTD/WOD da serie oceanografica

**Data:** 2026-07-09  
**Arquivo testado:** `data/processed/parquet/statistics/phase2_ctd_validation.csv`  
**Comparacao:** `diferenca_m = termoclina_ctd_media_m - d20_reanalise_media_m`

## Pergunta

As diferencas entre a termoclina observada por CTD/WOD e o D20 da reanalise sao
estatisticamente significativas?

## Resultado estatistico

| Periodo | n anos | Vies medio CTD-D20 (m) | IC95% (m) | p t-teste | p Wilcoxon | MAE (m) | Parecer |
|---|---:|---:|---:|---:|---:|---:|---|
| 1981-2023 | 33 | -5.15 | -12.70 a 2.39 | 0.1738 | 0.1831 | 17.40 | sem vies medio significativo |
| UFS 1981-1992 | 12 | -11.81 | -25.19 a 1.57 | 0.0781 | 0.0923 | 18.58 | vies negativo sugerido, mas nao significativo a 5% |
| GLORYS/GLO12 1993-2023 | 21 | -1.35 | -10.89 a 8.19 | 0.7705 | 0.7412 | 16.72 | sem vies medio significativo |
| Periodo moderno 2000-2023 | 16 | -7.66 | -17.39 a 2.07 | 0.1142 | 0.0879 | 15.53 | sem vies medio significativo |

## Parecer de validacao

Na escala anual agregada disponivel, as diferencas CTD/WOD - D20 **nao sao
estatisticamente significativas a 5%**. Portanto, nao ha evidencia estatistica
de um vies medio sistematico da reanalise contra CTD/WOD.

Isso valida a serie oceanografica para uso como **diagnostico fisico semanal,
anomalias, lags e ciclo de vida ENSO**, com ressalvas de incerteza vertical.
Nao valida a profundidade absoluta da termoclina com precisao fina.

## Tolerancia pratica

| Margem de equivalencia | Resultado |
|---|---|
| +/- 10 m | equivalencia nao demonstrada |
| +/- 20 m | equivalencia demonstrada |
| +/- 25 m | equivalencia demonstrada |

Leitura: a serie e defensavel se a tolerancia cientifica aceitar erro vertical
da ordem de **~20 m** para D20/termoclina. Se o trabalho exigir precisao melhor
que **10 m**, a validacao atual ainda nao basta.

## Anos de alerta

| Ano | Perfis CTD | CTD medio (m) | D20 reanalise (m) | Diferenca (m) |
|---:|---:|---:|---:|---:|
| 1981 | 6 | 83.3 | 139.0 | -55.6 |
| 1995 | 10 | 154.0 | 123.7 | 30.3 |
| 1996 | 5 | 166.0 | 127.5 | 38.5 |
| 2012 | 16 | 74.7 | 124.8 | -50.2 |

Esses anos devem ser tratados como pontos de alerta, nao como falha global da
serie. Tres deles tem poucos perfis CTD, o que aumenta a incerteza amostral.

## Limite da validacao atual

Esta validacao usa medias anuais. Ela testa vies medio anual, mas nao substitui
uma validacao ponto-a-ponto. Para validacao forte, o proximo passo e parear cada
perfil CTD com a reanalise no mesmo dia/localidade, calcular bias, MAE, RMSE,
correlacao, IC95% por bootstrap e graficos Bland-Altman por fonte
UFS/GLORYS/GLO12.
