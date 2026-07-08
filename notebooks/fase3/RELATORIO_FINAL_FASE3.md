# Relatorio final interpretativo - Fase 3 NINO26

Gerado em: 2026-07-08 16:40

## Veredito executivo

A Fase 3 esta completa como diagnostico fisico do Pacifico equatorial/Nino 3.4.
O conjunto de variaveis que melhor antecipa o aquecimento maximo do El Nino e o
bloco de **recarga/subsuperficie**:

- `ohc_0_300`: melhor preditor individual no hindcast; representa calor armazenado nos 0-300 m.
- `ssh_m`: proxy dinamico de expansao/recarga da coluna d'agua.
- `tau_x_anom_nino34_pa`: acoplamento vento-superficie; anomalias de oeste favorecem downwelling Kelvin e aquecimento.
- `ohc_0_700`, `tilt_m` e `d20_m`: confirmam profundidade/inclinacao da termoclina e memoria subsuperficial.
- `wwv`: variavel fisica classica de recarga basinwide; entra com ressalva local porque perdeu significancia em 2010-presente.
- `dhw_cweek_0p5_12w`: nao e precursor longo; mede persistencia e severidade acumulada apos o aquecimento se consolidar.

## Integridade temporal dos dados

Resumo da auditoria: **ok=7, warning=3**. Alertas `warning` indicam defasagem ou
cobertura a acompanhar; `error` indicaria quebra de integridade regular.

| artifact | scope | expected_freq | start | end | rows | freshness_days | max_key_null_pct | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oisst_diario_nino34 | fase3 | D | 1981-09-01 | 2026-06-09 | 16353 | 29 | 0.0 | warning | defasagem 29d > 21d |
| oisst_mensal_nino34 | fase3 | MS | 1981-09-01 | 2026-06-01 | 538 | 37 | 0.37 | ok |  |
| sinal_fisico_nino34 | fase3 | D | 1981-09-01 | 2026-06-09 | 16353 | 29 | 0.0 | warning | defasagem 29d > 21d |
| era5_atmo_nino34 | fase3 | D | 1981-01-01 | 2026-06-30 | 16617 | 8 | 0.0 | ok |  |
| dhw_variantes_nino34 | fase3 | D | 1981-09-01 | 2026-06-09 | 16353 | 29 | 0.51 | warning | defasagem 29d > 21d |
| matriz_semanal_fase3 | fase3 | W-SUN | 1981-01-04 | 2026-07-05 | 2375 | 3 | 2.11 | ok |  |
| pacifico_equatorial_lon_weekly | fase3 | W-SUN | 1981-09-06 | 2026-06-14 | 2337 | 24 | 0.0 | ok |  |
| ssh_kelvin_eventos | fase3 | event_windows | 1997-01-01 | 2026-07-07 | 2810 | 1 | 0.0 | ok |  |
| atlantico_tropical_legacy | legacy | D | 1981-09-01 | 2026-06-09 | 16353 | 29 | 0.0 | ok |  |
| eventos_elnino_referencia | fase3 | events | 1983-01-01 | 2023-12-01 | 12 | 950 | 0.0 | ok |  |

## Metodologia preditiva adotada

O 3I/3K usa **nested leave-one-event-out**. O loop interno escolhe o candidato
apenas nos eventos de treino; o loop externo preve o evento retido. Isso reduz o
vies otimista do LOO simples quando ele tambem escolhe o melhor modelo.

Referencias metodologicas: Jin (1997), Meinen & McPhaden (2000), WMO SVSLRF,
Barnston et al. (2012), Ambroise & McLachlan (2002), Cawley & Talbot (2010).

### Resultado nested LOO do 3I

| n_eventos | r_nested_loo | mae_nested_loo_c | rmse_nested_loo_c | mae_climatologia_c | skill_vs_climatologia | residuo_std_c | protocolo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 0.738 | 0.379 | 0.465 | 0.556 | 0.318 | 0.482 | nested leave-one-event-out: inner LOO seleciona candidato; outer LOO avalia evento retido |

### Projecao condicional 2025/26

| pico_projetado_c | ic95_baixo_c | ic95_alto_c | modelo | variaveis | horizonte_sem | r_loo | mae_loo_c | skill_vs_climatologia | leitura |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.93 | 0.985 | 2.875 | ohc300_20w | ohc_0_300 | 20 | 0.738 | 0.379 | 0.318 | projecao condicional exploratoria: amplitude do pico assumindo que o estado atual e precursor de um pico ~H semanas a frente |

Leitura: a projecao estima amplitude condicional dado o estado recente. Ela ainda
nao e previsao operacional de timing; isso fica para a Fase 5 com walk-forward,
embargo temporal, barreira de primavera e baseline de persistencia amortecida.

### Resultado nested LOO do 3K/PCA

| n_eventos | r_nested_loo | mae_nested_loo_c | rmse_nested_loo_c | mae_climatologia_c | skill_vs_climatologia | residuo_std_c | protocolo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 0.801 | 0.283 | 0.389 | 0.556 | 0.491 | 0.395 | nested leave-one-event-out: inner LOO seleciona candidato; outer LOO avalia evento retido |

## Melhores preditores por variavel (triagem flat LOO)

| variavel | rotulo | lead_semanas | r_loo | mae_loo_c | skill_vs_climatologia |
| --- | --- | --- | --- | --- | --- |
| ohc_0_300 | OHC0-300 | 20 | 0.866 | 0.214 | 0.615 |
| ssh_m | SSH | 20 | 0.793 | 0.24 | 0.568 |
| tau_x_anom_nino34_pa | tau_x anom. | 15 | 0.851 | 0.276 | 0.504 |
| ohc_0_700 | OHC0-700 | 20 | 0.801 | 0.278 | 0.501 |
| tilt_m | Tilt | 15 | 0.726 | 0.353 | 0.366 |
| d20_m | D20 | 20 | 0.629 | 0.379 | 0.319 |
| wwv | WWV | 12 | -0.204 | 0.518 | 0.069 |

## Estabilidade por subperiodo

| variavel | lag_semanas | r_full | r_1993_2009 | p_1993_2009 | r_2010_hoje | p_2010_hoje | estavel |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tilt_m | 0 | 0.772 | 0.803 | 0.0006 | 0.807 | 0.0008 | True |
| ssh_m | 6 | 0.755 | 0.781 | 0.0002 | 0.757 | 0.0005 | True |
| ohc_0_300 | 6 | 0.738 | 0.745 | 0.0011 | 0.728 | 0.0019 | True |
| ohc_0_700 | 6 | 0.701 | 0.713 | 0.0015 | 0.689 | 0.0024 | True |
| dhw_cweek_0p5_12w | 0 | 0.638 | 0.524 | 0.0135 | 0.724 | 0.0031 | True |
| d20_m | 15 | 0.545 | 0.546 | 0.0323 | 0.527 | 0.0404 | True |
| wwv | 20 | 0.516 | 0.558 | 0.0479 | 0.483 | 0.1095 | False |
| tau_x_anom_nino34_pa | 1 | 0.478 | 0.475 | 0.0 | 0.525 | 0.0 | True |

## Classes NOAA/ONI locais

| grupo | rotulo_curto | rotulo | definicao | n_eventos | oni_pico_medio_c | oni_pico_min_c | oni_pico_max_c | duracao_media_estacoes_oni | crescimento_medio_c_mes | decaimento_medio_c_mes | dhw_pico_ssta_medio_c_weeks | dhw_maximo_medio_c_weeks | defasagem_pico_dhw_media_sem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fraco | Fraco | El Nino fraco (0.5 <= ONI < 1.0 C) | pico da media movel de 3 meses na Nino 3.4 entre +0.5 e +0.9 C | 4 | 0.858 | 0.652 | 0.968 | 6.0 | 0.224 | -0.027 | 0.0 | 3.95 | 3.75 |
| moderado | Moderado | El Nino moderado (1.0 <= ONI < 1.5 C) | pico da media movel de 3 meses na Nino 3.4 entre +1.0 e +1.4 C | 2 | 1.212 | 1.211 | 1.213 | 12.0 | 0.11 | -0.098 | 0.0 | 15.0 | 10.5 |
| forte | Forte | El Nino forte (1.5 <= ONI < 2.0 C) | pico da media movel de 3 meses na Nino 3.4 entre +1.5 e +1.9 C | 3 | 1.706 | 1.586 | 1.907 | 10.67 | 0.247 | -0.192 | 6.67 | 20.85 | 8.0 |
| muito_forte | Muito forte | El Nino muito forte / super (ONI >= 2.0 C) | pico da media movel de 3 meses na Nino 3.4 igual ou acima de +2.0 C | 3 | 2.286 | 2.12 | 2.592 | 13.67 | 0.245 | -0.367 | 24.75 | 27.69 | 8.33 |

## O que cada notebook responde

| notebook | faz | pergunta | leitura |
| --- | --- | --- | --- |
| 3A | Materializa matriz semanal, cobertura e Hovmollers. | Pergunta: quais variaveis existem, em que forma e janela real? | Base fisica pronta; nem tudo e anomalia, por isso z-score e usado em comparacoes. |
| 3B | Define eventos NOAA/ONI locais, classes, ciclo e memoria. | Pergunta: como eventos nascem, crescem, picam e decaem? | Autocorrelacao vira baseline de persistencia. |
| 3C | Faz triagem bruta de lags preditivos. | Pergunta: quem antecede a SSTA e com quantas semanas? | Ranking bruto guia, mas nao basta sem rigor. |
| 3D | Aplica N_eff, FDR e IC95. | Pergunta: o que sobrevive ao controle estatistico? | Reduz falsos positivos e define evidencias robustas. |
| 3E | Testa estabilidade entre subperiodos. | Pergunta: o sinal vale antes e depois de 2010? | WWV fica com ressalva; OHC/SSH/tilt seguem fortes. |
| 3F | Avalia DHW e leitura qualitativa de Kelvin. | Pergunta: calor acumulado agrega informacao? | DHW mede severidade; Kelvin e diagnostico visual, nao detector automatico. |
| 3G | Compara DHW por classe e com 2025/26. | Pergunta: como severidade acumulada escala com eventos? | DHW ajuda a comparar persistencia e intensidade acumulada. |
| 3H | Mostra genese e ciclo de vida fisico. | Pergunta: o estado pre-onset separa classes? | Recarga cresce antes do pico e descarrega depois. |
| 3K | Reduz variaveis por PCA e testa skill. | Pergunta: quais variaveis sao redundantes? | PC1/OHC0-300 representa eixo de recarga com parcimonia. |
| 3I | Integra parecer e nested LOO. | Pergunta: quais variaveis predizem o pico e como ler 2025/26? | Entrega projecao condicional exploratoria, nao operacional. |

## Catalogo de figuras e tabelas

| notebook | figura | titulo | interpreta | tabela_referencia | figura_existe | tabela_existe |
| --- | --- | --- | --- | --- | --- | --- |
| 3A | 3A1_series_semanais.png | Series semanais | Cobertura, unidades e comparacao visual das variaveis fisicas. | phase3A_cobertura_variaveis.csv | True | True |
| 3A | 3A2_hovmoller_ssta.png | Hovmoller SSTA | Mostra propagacao longitudinal da anomalia de SST na faixa equatorial. | phase3A_fontes_variaveis.csv | True | True |
| 3A | 3A3_hovmoller_sla_taux.png | Hovmoller SLA + vento | Conecta SLA local e setas de tau_x para leitura qualitativa de Kelvin/acoplamento. | phase3A_fontes_variaveis.csv | True | True |
| 3B | 3B1_trajetorias_compostas.png | Trajetorias por classe | Compara a evolucao media da SSTA por classe NOAA/ONI. | phase3B_trajetorias_compostas.csv | True | True |
| 3B | 3B2_autocorrelacao.png | Autocorrelacao | Mede memoria/persistencia da SSTA que qualquer previsao deve superar. | phase3B_autocorrelacao.csv | True | True |
| 3B | 3B3_mapa_composto_pico.png | Mapa composto do pico | Mostra a assinatura espacial media no pico dos eventos. | phase3B_mapa_composto_resumo.csv | True | True |
| 3C | 3C1_heatmap_lags.png | Heatmap de lags | Triagem bruta de quais variaveis antecedem a SSTA e em que lag. | phase3C_ranking_lags.csv | True | True |
| 3C | 3C2_mapa_lon_lag.png | Longitude x lag | Mostra onde no Pacifico equatorial o sinal antecedente aparece por longitude. | phase3C_lag_correlacoes.csv | True | True |
| 3D | 3D1_forest_ic95.png | Forest IC95 | Aplica N_eff, FDR e IC95 para reduzir falsos positivos. | phase3D_ranking_significativo.csv | True | True |
| 3D | 3D2_mapa_lon_lag_fdr.png | Mapa FDR | Mostra regioes longitude-lag que sobrevivem ao controle estatistico. | phase3D_testes_completos.csv | True | True |
| 3E | 3E1_scatter_estabilidade.png | Estabilidade | Compara correlacoes 1993-2009 vs 2010-presente. | phase3E_estabilidade.csv | True | True |
| 3E | 3E2_mapa_lon_lag_subperiodos.png | Subperiodos | Testa se o padrao longitudinal se repete em regimes diferentes. | phase3E_estabilidade.csv | True | True |
| 3F | 3F1_dhw_serie.png | DHW serie | Mostra DHW canonico como severidade acumulada/persistencia. | phase3F_dhw_redundancia.csv | True | True |
| 3F | 3F2_hovmoller_ssh_kelvin.png | Kelvin SSH | Diagnostico visual de propagacao por SLA/SSH em eventos fortes. | phase3F_dhw_redundancia.csv | True | True |
| 3G | 3G1_composto_ssta_dhw.png | SSTA x DHW | Compara aquecimento e calor acumulado por classe NOAA/ONI. | phase3G_composto_ssta_dhw_classes_noaa.csv | True | True |
| 3G | 3G2_escalonamento_dhw.png | Escalonamento DHW | Relaciona DHW maximo com pico e duracao do evento. | phase3G_escalonamento.csv | True | True |
| 3G | 3G3_mapa_dhw_lon.png | DHW longitude | Compara fortes/super historicos com a formacao atual 2025/26. | phase3G_mapa_dhw_lon_eventos_forte_super.csv | True | True |
| 3H | 3H1_compostos_onset.png | Onset por classe | Mostra quais variaveis se separam na genese dos eventos. | phase3H_estado_precursor_por_classe.csv | True | True |
| 3H | 3H2_ciclo_vida.png | Ciclo de vida | Resume genese, crescimento, pico e decaimento com variaveis em z-score. | phase3H_ciclo_vida_media.csv | True | True |
| 3I | 3I1_sintese_parecer.png | Sintese do parecer | Organiza quais evidencias entram, entram com ressalva ou ficam fora. | phase3I_conclusoes_decisao.csv | True | True |
| 3I | 3I2_antecipacao_pico.png | Antecipacao | Mostra variaveis candidatas para antecipar o aquecimento maximo. | phase3I_conjunto_antecipacao_pico.csv | True | True |
| 3I | 3I3_previsao_condicional_nested.png | Nested LOO | Avalia selecao+ajuste por nested LOO e gera projecao condicional. | phase3I_nested_loo_metricas.csv | True | True |
| 3K | 3K1_skill_loo_nested.png | Skill PCA | Testa se PCA reduz redundancia sem perder skill preditivo. | phase3K_previsao_pico_nested_loo_metricas.csv | True | True |
| 3K | 3K2_scree.png | Scree PCA | Mostra quantos componentes explicam a variancia de crescimento. | phase3K_pca_variancia.csv | True | True |
| 3K | 3K3_biplot.png | Biplot PCA | Mostra agrupamentos fisicos e colinearidade entre variaveis. | phase3K_pca_loadings.csv | True | True |

## Galeria padronizada

![Galeria Fase 3](../../data/processed/figures/fase3/3R1_galeria_figuras_fase3.png)


## Interpretacao para cientistas

O aquecimento maximo do El Nino nao e explicado apenas pela SSTA superficial.
A SSTA e a resposta observavel final; antes dela, o sistema precisa acumular
calor e alterar a estrutura vertical/oceanica. OHC0-300, SSH, D20 e tilt medem
essa recarga e a geometria da termoclina. O `tau_x_anom` representa o elo de
acoplamento com a atmosfera: anomalias de oeste reduzem/alteram os alisios,
favorecem ondas Kelvin de downwelling e aprofundam a termoclina no centro-leste
do Pacifico. O WWV e teoricamente central no oscilador de recarga, mas nesta
implementacao local fica menos estavel nos subperiodos; por isso entra com
ressalva. O DHW e util para severidade acumulada e comparacao de eventos, mas
por construcao responde depois de persistencia termica e nao deve ser vendido
como precursor principal do pico.

## Interpretacao para pessoas comuns

Pense no El Nino como uma panela grande. A temperatura da superficie e o que se
ve por cima, mas o pico depende do calor ja guardado embaixo e de como o vento
empurra esse calor pelo Pacifico. As melhores pistas sao: quanto calor ha nos
primeiros 300 m, se o nivel do mar/coluna d'agua indica recarga, se a termoclina
esta mais profunda/inclinada e se o vento esta ajudando o calor a ir para leste.
Quando essas pistas aparecem juntas, a chance de um pico maior aumenta. O DHW
mede por quanto tempo o aquecimento ficou acumulando; ele confirma severidade,
mas nao e a primeira pista.
