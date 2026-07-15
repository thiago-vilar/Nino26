# Relatorio final interpretativo - Fase 3 NINO26

Gerado em: 2026-07-13 12:54

## Veredito executivo

A Fase 3 esta **concluida como diagnostico fisico semanal do Nino 3.4** pela
diretriz canonica atual. O pipeline agora separa **El Nino e La Nina**, delimita
genese, crescimento, pico e decaimento por evento, calcula duracoes por tipo e
classe, materializa discriminantes por periodo e entrega PCA por fase.

O conjunto de variaveis que melhor antecipa o aquecimento maximo do El Nino, no
estado atual do pipeline, e o bloco de **recarga/subsuperficie**:

- `ohc_0_300`: melhor preditor individual no hindcast; representa calor armazenado nos 0-300 m.
- `ssh_m`: proxy dinamico de expansao/recarga da coluna d'agua.
- `tau_x_anom_nino34_pa`: acoplamento vento-superficie; anomalias de oeste favorecem downwelling Kelvin e aquecimento.
- `ohc_0_700`, `tilt_m` e `d20_m`: confirmam profundidade/inclinacao da termoclina e memoria subsuperficial.
- `wwv`: volume acima da isoterma de 20 C integrado em area no Pacifico equatorial; candidato do bloco de recarga, derivado de D20 e nao representante obrigatorio.

## Fechamento contra a diretriz atual da Fase 3

| item_pedido | estado_atual | evidencia | produto |
| --- | --- | --- | --- |
| Separar eventos El Nino e La Nina 1981-2026 | feito | phase3_events_en_ln.csv | 12 El Nino + 11 La Nina pela regra ONI local simetrica (+/-0.5 C) |
| Duracao media por tipo, incluindo fortes/super, para El Nino e La Nina | feito | phase3_duracao_por_tipo_classe.csv | duracao media por tipo, classe e fase do ciclo de vida |
| Classificar genese, crescimento, pico e decaimento de cada evento | feito | phase3_event_lifecycle_en_ln.csv e phase3_fases_semanais_en_ln.csv | quatro periodos por evento, para El Nino e La Nina |
| Hovmoller, Bjerknes, Kelvin, mapas e ciclos de vida | feito | 3A/3F/3G/3H + 3L | Hovmoller/Kelvin/mapas do pacote fisico e figuras EN/LN de ciclo e duracao |
| PCA e EOF por ciclo de vida | feito | phase3_pca_por_fase.csv e phase3_pca_loadings_por_fase.csv | PCA por fase e por sinal; EOF espacial fica como extensao, nao bloqueio da Fase 3 |
| Quais variaveis delimitam os quatro periodos | feito | phase3_fase_stats_variaveis.csv e phase3_discriminantes_por_periodo.csv | nivel, volatilidade semanal e poder discriminante por periodo |

## Integridade temporal dos dados

Resumo da auditoria: **ok=7, warning=2**. Alertas `warning` indicam defasagem ou
cobertura a acompanhar; `error` indicaria quebra de integridade regular.

| artifact | scope | expected_freq | start | end | rows | freshness_days | max_key_null_pct | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oisst_diario_nino34 | fase3 | D | 1981-09-01 | 2026-06-09 | 16353 | 29 | 0.0 | warning | defasagem 29d > 21d |
| oisst_mensal_nino34 | fase3 | MS | 1981-09-01 | 2026-06-01 | 538 | 37 | 0.37 | ok |  |
| sinal_fisico_nino34 | fase3 | D | 1981-09-01 | 2026-06-09 | 16353 | 29 | 0.0 | warning | defasagem 29d > 21d |
| era5_atmo_nino34 | fase3 | D | 1981-01-01 | 2026-06-30 | 16617 | 8 | 0.0 | ok |  |
| matriz_semanal_fase3 | fase3 | W-SUN | 1981-01-04 | 2026-07-05 | 2375 | 3 | 1.6 | ok |  |
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
| 12 | 0.738 | 0.379 | 0.465 | 0.556 | 0.319 | 0.482 | nested leave-one-event-out: inner LOO seleciona candidato; outer LOO avalia evento retido |

### Projecao condicional 2025/26

| pico_projetado_c | ic95_baixo_c | ic95_alto_c | modelo | variaveis | horizonte_sem | r_loo | mae_loo_c | skill_vs_climatologia | leitura |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.93 | 0.985 | 2.875 | ohc300_20w | ohc_0_300 | 20 | 0.738 | 0.379 | 0.319 | projecao condicional exploratoria: amplitude do pico assumindo que o estado atual e precursor de um pico ~H semanas a frente |

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
| t100m | T100m | 20 | 0.837 | 0.247 | 0.557 |
| ohc_0_100 | OHC0-100 | 4 | 0.865 | 0.248 | 0.555 |
| u850_anom | u850 anom. | 15 | 0.781 | 0.258 | 0.537 |
| tau_x_anom_nino34_pa | tau_x anom. | 15 | 0.85 | 0.278 | 0.501 |
| ohc_0_700 | OHC0-700 | 20 | 0.801 | 0.278 | 0.501 |
| t50m | T50m | 26 | 0.77 | 0.288 | 0.482 |
| u10_anom | u10 anom. | 15 | 0.82 | 0.298 | 0.464 |
| tcwv_anom | TCWV anom. | 20 | 0.692 | 0.307 | 0.448 |
| tilt_m | Tilt | 15 | 0.726 | 0.353 | 0.366 |
| mslp_anom | MSLP anom. | 12 | 0.652 | 0.366 | 0.342 |
| d20_m | D20 | 20 | 0.629 | 0.379 | 0.319 |
| tilt_slope | Tilt slope | 8 | 0.624 | 0.403 | 0.275 |
| t150m | T150m | 15 | 0.475 | 0.467 | 0.16 |
| t200m | T200m | 15 | 0.422 | 0.469 | 0.157 |
| v10_anom | v10 anom. | 4 | 0.153 | 0.47 | 0.156 |
| t300m | T300m | 15 | 0.161 | 0.471 | 0.154 |
| wwv | WWV | 12 | -0.204 | 0.518 | 0.069 |
| ohc_300_700 | OHC300-700 | 20 | -0.209 | 0.522 | 0.062 |

## Sensibilidade temporal sem breakpoint

| variavel | lag_semanas | r_full | bootstrap_ic95_inf_envelope | bootstrap_ic95_sup_envelope | bootstrap_min_fracao_mesmo_sinal | loo_eventos_n | loo_eventos_r_min | loo_eventos_r_max | loo_evento_maior_influencia | loo_max_delta_r | papel_3E |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ohc_0_100 | 1 | 0.9011273440498032 | 0.8650958262983888 | 0.9286592983727524 | 1.0 | 17 | 0.8920043679915188 | 0.9038356583113184 | el_nino_2014_2016 | 0.0091229760582842 | sensibilidade; nao filtro e nao breakpoint |
| t50m | 0 | 0.8870888363817137 | 0.8452146877772106 | 0.9173272714338172 | 1.0 | 17 | 0.8772477917218165 | 0.8879412828178471 | el_nino_2014_2016 | 0.0098410446598971 | sensibilidade; nao filtro e nao breakpoint |
| tilt_m | 0 | 0.7716180623939575 | 0.6859132597539398 | 0.8383406825507499 | 1.0 | 17 | 0.738367731628422 | 0.7919881653479556 | el_nino_1997_1998 | 0.0332503307655355 | sensibilidade; nao filtro e nao breakpoint |
| ssh_m | 6 | 0.7550882065024468 | 0.6283847063591926 | 0.8353922792718872 | 1.0 | 17 | 0.7085610347811975 | 0.7835715791376043 | el_nino_2014_2016 | 0.0465271717212493 | sensibilidade; nao filtro e nao breakpoint |
| tcwv_anom | 0 | 0.754309301461754 | 0.6381369679555702 | 0.8291514430719445 | 1.0 | 17 | 0.7128701550292745 | 0.7591353365901268 | el_nino_2014_2016 | 0.0414391464324794 | sensibilidade; nao filtro e nao breakpoint |
| tilt_slope | 0 | 0.7435991388912172 | 0.6556409394478068 | 0.8147047418359621 | 1.0 | 17 | 0.7162138691130583 | 0.7828536485919183 | la_nina_1998_2001 | 0.0392545097007011 | sensibilidade; nao filtro e nao breakpoint |
| ohc_0_300 | 6 | 0.7378748716208154 | 0.6319149915683508 | 0.812392742402634 | 1.0 | 17 | 0.7009624933556625 | 0.7573874010916146 | el_nino_2014_2016 | 0.0369123782651529 | sensibilidade; nao filtro e nao breakpoint |
| sshf_anom | 0 | -0.7327297009741838 | -0.8014862622770242 | -0.6317487465023918 | 1.0 | 17 | -0.7442138112062039 | -0.701039548893116 | el_nino_2014_2016 | 0.0316901520810677 | sensibilidade; nao filtro e nao breakpoint |
| ohc_0_700 | 6 | 0.701191752632237 | 0.5787130270471067 | 0.7892879344697545 | 1.0 | 17 | 0.6606547957929064 | 0.7241010238668134 | el_nino_2014_2016 | 0.0405369568393305 | sensibilidade; nao filtro e nao breakpoint |
| t100m | 7 | 0.6972796083645344 | 0.5652437918494642 | 0.7796482932811621 | 1.0 | 17 | 0.6581402526988291 | 0.7144787202217118 | el_nino_2014_2016 | 0.0391393556657052 | sensibilidade; nao filtro e nao breakpoint |
| omega850_anom | 0 | -0.6775532342936517 | -0.7615455873408785 | -0.5315882128564895 | 1.0 | 17 | -0.6906448443156772 | -0.6269119270970992 | el_nino_1997_1998 | 0.0506413071965524 | sensibilidade; nao filtro e nao breakpoint |
| u850_anom | 1 | 0.6380052447528572 | 0.5238604433329367 | 0.7119270393667368 | 1.0 | 17 | 0.5914333713268135 | 0.6420899969134546 | el_nino_1997_1998 | 0.0465718734260437 | sensibilidade; nao filtro e nao breakpoint |
| ssr_anom | 0 | -0.6051958175363698 | -0.7074161120371585 | -0.4371590857540165 | 1.0 | 17 | -0.6129286036068726 | -0.5520020341337019 | el_nino_2014_2016 | 0.0531937834026678 | sensibilidade; nao filtro e nao breakpoint |
| u200_anom | 0 | -0.5922666981715741 | -0.6657932071880893 | -0.488122660923491 | 1.0 | 17 | -0.6027859386980076 | -0.5552918356421997 | el_nino_1997_1998 | 0.0369748625293743 | sensibilidade; nao filtro e nao breakpoint |
| omega500_anom | 0 | -0.5524333783663236 | -0.667217593746322 | -0.3618222757416271 | 1.0 | 17 | -0.5726431366276363 | -0.4869482637294499 | el_nino_1997_1998 | 0.0654851146368736 | sensibilidade; nao filtro e nao breakpoint |
| d20_m | 15 | 0.5448633312768661 | 0.3836516387105952 | 0.6617180053380557 | 1.0 | 17 | 0.4787594919295452 | 0.5730722005622646 | la_nina_1998_2001 | 0.0661038393473208 | sensibilidade; nao filtro e nao breakpoint |
| t300m | 0 | 0.52479912104495 | 0.3728147632453252 | 0.6469716848946216 | 1.0 | 17 | 0.4734018497325895 | 0.5438365611891595 | el_nino_2014_2016 | 0.0513972713123604 | sensibilidade; nao filtro e nao breakpoint |
| wwv | 20 | 0.5161023343556326 | 0.3174695765936827 | 0.6644869744364795 | 1.0 | 17 | 0.424300135824424 | 0.57457069882279 | la_nina_1998_2001 | 0.0918021985312085 | sensibilidade; nao filtro e nao breakpoint |
| u10_anom | 1 | 0.5034384172588451 | 0.3135729245532102 | 0.6191490174319956 | 1.0 | 17 | 0.4183403939372053 | 0.5564711739330653 | el_nino_1997_1998 | 0.0850980233216397 | sensibilidade; nao filtro e nao breakpoint |
| slhf_anom | 0 | -0.5033972789981966 | -0.6102333086618874 | -0.4014775180657176 | 1.0 | 17 | -0.5234452896976898 | -0.4481818949120394 | la_nina_1998_2001 | 0.0552153840861572 | sensibilidade; nao filtro e nao breakpoint |

## Classes El Nino NOAA/ONI locais

| grupo | rotulo_curto | rotulo | definicao | n_eventos | oni_pico_medio_c | oni_pico_min_c | oni_pico_max_c | duracao_media_estacoes_oni | crescimento_medio_c_mes | decaimento_medio_c_mes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fraco | Fraco | El Nino fraco (0.5 <= ONI < 1.0 C) | pico da media movel de 3 meses na Nino 3.4 entre +0.5 e +0.9 C | 4 | 0.858 | 0.652 | 0.968 | 6.0 | 0.224 | -0.027 |
| moderado | Moderado | El Nino moderado (1.0 <= ONI < 1.5 C) | pico da media movel de 3 meses na Nino 3.4 entre +1.0 e +1.4 C | 2 | 1.212 | 1.211 | 1.213 | 12.0 | 0.11 | -0.098 |
| forte | Forte | El Nino forte (1.5 <= ONI < 2.0 C) | pico da media movel de 3 meses na Nino 3.4 entre +1.5 e +1.9 C | 3 | 1.706 | 1.586 | 1.907 | 10.67 | 0.247 | -0.192 |
| muito_forte | Muito forte | El Nino muito forte / super (ONI >= 2.0 C) | pico da media movel de 3 meses na Nino 3.4 igual ou acima de +2.0 C | 3 | 2.286 | 2.12 | 2.592 | 13.67 | 0.245 | -0.367 |

## O que cada notebook responde

| notebook | faz | pergunta | leitura |
| --- | --- | --- | --- |
| 3A | Materializa matriz semanal, cobertura e Hovmollers. | Pergunta: quais variaveis existem, em que forma e janela real? | Base fisica pronta; nem tudo e anomalia, por isso z-score e usado em comparacoes. |
| 3B | Define eventos NOAA/ONI locais, classes, ciclo e memoria. | Pergunta: como eventos nascem, crescem, picam e decaem? | Autocorrelacao vira baseline de persistencia. |
| 3C | Faz triagem bruta de lags preditivos. | Pergunta: quem antecede a SSTA e com quantas semanas? | Ranking bruto guia, mas nao basta sem rigor. |
| 3D | Aplica N_eff, FDR e IC95. | Pergunta: o que sobrevive ao controle estatistico? | Reduz falsos positivos e define evidencias robustas. |
| 3E | Quantifica sensibilidade sem breakpoint. | Pergunta: a relacao depende da autocorrelacao ou de um evento ENSO isolado? | Bootstrap movel + LOO por evento informam incerteza; nao criam gate. |
| 3F | Avalia leitura qualitativa de ondas de Kelvin por SLA/SSH e vento. | Pergunta: ha propagacao dinamica compativel com Kelvin? | Kelvin e diagnostico visual/dinamico, nao detector automatico. |
| 3G | Compara SSTA por classe NOAA/ONI e com 2025/26. | Pergunta: como intensidade, duracao e propagacao longitudinal se organizam? | SSTA por classe mostra escala termica sem metricas acumuladas auxiliares. |
| 3H | Mostra genese e ciclo de vida fisico. | Pergunta: o estado pre-onset separa classes? | Recarga cresce antes do pico e descarrega depois. |
| 3K | Reduz variaveis por PCA e testa skill. | Pergunta: quais variaveis sao redundantes? | PC1/OHC0-300 representa eixo de recarga com parcimonia. |
| 3I | Integra parecer e nested LOO. | Pergunta: quais variaveis predizem o pico e como ler 2025/26? | Entrega projecao condicional exploratoria, nao operacional. |
| 3L | Caracteriza El Nino e La Nina por evento/fase. | Pergunta: o protocolo completo EN/LN fecha a diretriz da Fase 3? | Eventos, fases, duracoes, discriminantes e PCA por fase ficam materializados. |

## Catalogo de figuras e tabelas

| notebook | figura | titulo | interpreta | tabela_referencia | figura_existe | tabela_existe |
| --- | --- | --- | --- | --- | --- | --- |
| 3A | Fig_3A01.png | Series semanais | Cobertura, unidades e comparacao visual das variaveis fisicas. | phase3A_cobertura_variaveis.csv | True | True |
| 3A | Fig_3A02.png | Hovmoller SSTA | Mostra propagacao longitudinal da anomalia de SST na faixa equatorial. | phase3A_fontes_variaveis.csv | True | True |
| 3A | Fig_3A03.png | Hovmoller SLA + vento | Conecta SLA local e setas de tau_x para leitura qualitativa de Kelvin/acoplamento. | phase3A_fontes_variaveis.csv | True | True |
| 3B | Fig_3B01.png | Trajetorias por classe | Compara a evolucao media da SSTA por classe NOAA/ONI. | phase3B_trajetorias_compostas.csv | True | True |
| 3B | Fig_3B02.png | Autocorrelacao | Mede memoria/persistencia da SSTA que qualquer previsao deve superar. | phase3B_autocorrelacao.csv | True | True |
| 3B | Fig_3B03.png | Mapa composto do pico | Mostra a assinatura espacial media no pico dos eventos. | phase3B_mapa_composto_resumo.csv | True | True |
| 3B | Fig_3B04.png | Sensibilidade da faixa de pico | Compara a duração da faixa sob 80%, 90% e 95% do extremo. | phase3_peak_band_sensitivity.csv | True | True |
| 3C | Fig_3C01.png | Heatmap de lags | Triagem bruta de quais variaveis antecedem a SSTA e em que lag. | phase3C_ranking_lags.csv | True | True |
| 3C | Fig_3C02.png | Longitude x lag | Mostra onde no Pacifico equatorial o sinal antecedente aparece por longitude. | phase3C_lag_correlacoes.csv | True | True |
| 3D | Fig_3D01.png | Forest IC95 | Aplica N_eff, FDR e IC95 para reduzir falsos positivos. | phase3D_ranking_significativo.csv | True | True |
| 3D | Fig_3D02.png | Mapa FDR | Mostra regioes longitude-lag que sobrevivem ao controle estatistico. | phase3D_testes_completos.csv | True | True |
| 3E | Fig_3E01.png | Sensibilidade sem breakpoint | Compara r completo, envelope IC95 do bootstrap movel e faixa leave-one-event-out; nao define exclusao. | phase3E_sensibilidade_resumo.csv | True | True |
| 3E | Fig_3E02.png | Influencia de eventos | Mostra quanto r muda ao retirar cada evento EN/LN, sem dividir a serie por ano. | phase3E_leave_one_event_out.csv | True | True |
| 3F | Fig_3F01.png | Kelvin por SLA | Diagnostico visual de propagacao oeste-leste por SLA/SSH em eventos fortes. | phase3F_kelvin_eventos_resumo.csv | True | True |
| 3F | Fig_3F02.png | Vento e SLA | Resume tau_x_anom na Nino 3.4 junto ao sinal de SLA por evento. | phase3F_kelvin_eventos_resumo.csv | True | True |
| 3G | Fig_3G01.png | SSTA por classe | Compara a evolucao termica media por classe NOAA/ONI. | phase3G_composto_ssta_classes_noaa.csv | True | True |
| 3G | Fig_3G02.png | Escalonamento termico | Relaciona pico ONI local, duracao e taxas de crescimento/decaimento. | phase3G_escalonamento_ssta.csv | True | True |
| 3G | Fig_3G03.png | SSTA longitude | Compara fortes/super historicos com a formacao atual 2025/26 por longitude. | phase3G_mapa_ssta_lon_eventos_forte_super.csv | True | True |
| 3H | Fig_3H01.png | Onset por classe | Mostra quais variaveis se separam na genese dos eventos. | phase3H_estado_precursor_por_classe.csv | True | True |
| 3H | Fig_3H02.png | Ciclo de vida | Resume genese, crescimento, pico e decaimento com variaveis em z-score. | phase3H_ciclo_vida_media.csv | True | True |
| 3H | Fig_3H03.png | Ciclo físico ampliado | Resume subsuperfície e atmosfera nas quatro fases. | phase3H_ciclo_vida_media_subsuperficie_atmosfera.csv | True | True |
| 3I | Fig_3I01.png | Sintese do parecer | Organiza evidencias do 3D por bloco fisico, lag e metricas continuas de sensibilidade do 3E. | phase3I_conclusoes_decisao.csv | True | True |
| 3I | Fig_3I02.png | Antecipacao | Mostra variaveis candidatas para antecipar o aquecimento maximo. | phase3I_conjunto_antecipacao_pico.csv | True | True |
| 3I | Fig_3I03.png | Nested LOO | Avalia selecao+ajuste por nested LOO e gera projecao condicional. | phase3I_nested_loo_metricas.csv | True | True |
| 3K | Fig_3K01.png | Skill PCA | Testa se PCA reduz redundancia sem perder skill preditivo. | phase3K_previsao_pico_nested_loo_metricas.csv | True | True |
| 3K | Fig_3K02.png | Scree PCA | Mostra quantos componentes explicam a variancia de crescimento. | phase3K_pca_variancia.csv | True | True |
| 3K | Fig_3K03.png | Biplot PCA | Mostra agrupamentos fisicos e colinearidade entre variaveis. | phase3K_pca_loadings.csv | True | True |
| 3L | Fig_3L01.png | Ciclo de vida EN/LN | Compara genese, crescimento, pico e decaimento de El Nino e La Nina. | phase3_event_lifecycle_en_ln.csv | True | True |
| 3L | Fig_3L03.png | Duracao por fase EN/LN | Resume a duracao media por tipo, classe e fase do ciclo de vida. | phase3_duracao_por_tipo_classe.csv | True | True |
| 3L | Fig_3L02.png | Friedman pareado e Kendall W | Mostra Kendall W para todas as variaveis e marca como confirmadas somente as com qBH<=0,05, em familia separada por tipo ENSO. | phase3_discriminantes_por_periodo.csv | True | True |
| 3L | Fig_3L04.png | PCA por fase | Resume a estrutura multivariada por genese, crescimento, pico e decaimento. | phase3_pca_por_fase.csv | True | True |

## Galeria padronizada

![Galeria Fase 3](../../data/processed/figures/fase3/Fig_3I04.png)


## Interpretacao para cientistas

O aquecimento maximo do El Nino nao e explicado apenas pela SSTA superficial.
A SSTA e a resposta observavel final; antes dela, o sistema precisa acumular
calor e alterar a estrutura vertical/oceanica. OHC0-300, SSH, D20 e tilt medem
essa recarga e a geometria da termoclina. O `tau_x_anom` representa o elo de
acoplamento com a atmosfera: anomalias de oeste reduzem/alteram os alisios,
favorecem ondas Kelvin de downwelling e aprofundam a termoclina no centro-leste
do Pacifico. O WWV e um indicador classico do oscilador de recarga, mas e
derivado de D20 e compartilha informacao com D20/OHC/SSH/tilt; por isso fica
como candidato, sem privilegio a priori. O 3E usa bootstrap em blocos e
leave-one-event-out apenas como sensibilidade, sem corte cronologico. A Fase 3 nao usa metricas acumuladas artificiais como preditoras:
o foco interpretativo fica em SSTA, recarga/subsuperficie e acoplamento do vento.

## Interpretacao para pessoas comuns

Pense no El Nino como uma panela grande. A temperatura da superficie e o que se
ve por cima, mas o pico depende do calor ja guardado embaixo e de como o vento
empurra esse calor pelo Pacifico. As melhores pistas sao: quanto calor ha nos
primeiros 300 m, se o nivel do mar/coluna d'agua indica recarga, se a termoclina
esta mais profunda/inclinada e se o vento esta ajudando o calor a ir para leste.
Quando essas pistas aparecem juntas, a chance de um pico maior aumenta.
