# Relatorio de caracterizacao cronologica do El Nino - Fase 3

**Projeto:** NINO-BRASIL / NINO26  
**Data da sintese:** 2026-07-09  
**Escopo:** caracterizacao fisica do El Nino no Pacifico equatorial, com base nos
artefatos da Fase 3 (`3A` a `3L`), sem redes neurais e sem usar rotulo externo
como variavel-alvo.  
**Alvo termico:** ONI local OISST/Nino 3.4, media movel de 3 meses; evento El
Nino = anomalia >= +0.5 C por 5+ estacoes moveis sobrepostas.  
**Serie principal:** matriz semanal Nino 3.4 / Pacifico equatorial, 1981-2026.

## 1. Parecer executivo

A Fase 3 sustenta uma cronologia fisica coerente para o El Nino:

1. **Pre-condicionamento/recarga:** cerca de 6 meses antes do pico, a memoria da
   propria SSTA ja e relevante (`e-folding` = 27 semanas), mas a informacao
   fisicamente mais util vem do oceano subsuperficial: OHC, SSH, D20, WWV e
   estrutura de termoclina.
2. **Nascimento espaco-temporal:** o sinal termico organizado aparece no
   Pacifico central, perto de **150W**, com maior forca entre 0 e 14 semanas de
   antecedencia. Em leads mais longos, o maximo desloca para o Pacifico oeste
   (~170E) e enfraquece, indicando recarga/memoria de longo prazo, nao SSTA
   local pronta.
3. **Crescimento:** depois do onset, o evento cresce por acoplamento Bjerknes:
   calor acumulado em subsuperficie, SSH/OHC elevados, termoclina inclinada,
   vento zonal anomalo e pulsos Kelvin de downwelling.
4. **Antecipacao do pico:** a melhor familia preditiva e recarga
   subsuperficial. O melhor modelo simples foi **OHC0-300 a 20 semanas**:
   `r_LOO=0.866`, `MAE=0.214 C`, `skill=0.615` contra climatologia LOO. O melhor
   conjunto curto foi **OHC0-300 + SSH + tau_x** a 15 semanas:
   `r_LOO=0.891`, `MAE=0.223 C`, `skill=0.599`.
5. **Pico:** o pico nao deve ser tratado como ponto unico. Em eventos fortes e
   muito fortes, a faixa de pico dura tipicamente 1-2 meses; em eventos fracos,
   pode durar 5-6 meses. No composto semanal, a faixa visual de pico fica em
   torno de -1 a +5 semanas do pico real.
6. **Decaimento:** a queda e precedida/acompanhada por descarga da recarga
   oceanica: T100m, T150m, D20, WWV, OHC0-300/OHC0-700 e SSH caem fortemente do
   pico para o decaimento. A atmosfera ainda pode manter sinal residual, mas a
   base energetica subsuperficial ja descarrega.

## 2. Base de eventos e denominadores

Depois da reexecucao dos artefatos EN/LN, a base consolidada tem:

| tipo | classes | n |
|---|---|---:|
| El Nino | fraco=4, moderado=2, forte=3, muito_forte=3 | 12 |
| La Nina | fraca=4, moderada=5, forte=1, muito_forte=1 | 11 |

Eventos El Nino detectados localmente:

| evento | classe | onset | pico | fim | duracao_meses | ONI pico C |
|---|---|---:|---:|---:|---:|---:|
| 1982/83 | muito_forte | 1982-07 | 1983-01 | 1983-05 | 11 | 2.120 |
| 1986/88 | moderado | 1986-10 | 1987-08 | 1988-01 | 16 | 1.211 |
| 1991/92 | forte | 1991-09 | 1992-01 | 1992-06 | 10 | 1.625 |
| 1994/95 | fraco | 1994-10 | 1994-12 | 1995-02 | 5 | 0.968 |
| 1997/98 | muito_forte | 1997-06 | 1997-12 | 1998-04 | 11 | 2.146 |
| 2002/03 | moderado | 2002-07 | 2002-11 | 2003-02 | 8 | 1.213 |
| 2004 | fraco | 2004-08 | 2004-09 | 2004-12 | 5 | 0.652 |
| 2006/07 | fraco | 2006-09 | 2006-12 | 2007-01 | 5 | 0.920 |
| 2009/10 | forte | 2009-07 | 2009-12 | 2010-04 | 10 | 1.586 |
| 2014/16 | muito_forte | 2014-10 | 2015-12 | 2016-04 | 19 | 2.592 |
| 2018/19 | fraco | 2018-10 | 2018-11 | 2019-06 | 9 | 0.892 |
| 2023/24 | forte | 2023-05 | 2023-12 | 2024-04 | 12 | 1.907 |

Fontes principais: `phase3_events_en_ln.csv`,
`phase3B_eventos_taxas.csv`, `phase3H_proveniencia_eventos.csv`.

## 3. Cronologia media do ciclo de vida

| fase | janela media | duracao media | leitura fisica |
|---|---:|---:|---|
| Genese | antes do onset; ate ~49 semanas antes do pico no composto | 24.6 sem; mediana 26 sem | pre-condicionamento oceanico e inicio da organizacao termica |
| Crescimento | onset ate pico | 21.5 sem; mediana 18 sem | acoplamento Bjerknes, recarga vira aquecimento de superficie |
| Pico | faixa de maturacao | 13.5 sem; mediana 13 sem | evento organizado; maxima coerencia multivariada |
| Decaimento | pos-pico | 11.6 sem; mediana 12 sem | descarga subsuperficial e enfraquecimento da sustentacao termica |

Por classe, eventos muito fortes crescem por mais tempo: crescimento medio
~30.7 semanas, contra ~6.0 semanas nos fracos. Essa diferenca e importante:
evento forte nao e apenas evento mais quente; ele fica mais tempo acoplando
recarga, vento e termoclina antes do pico.

Fontes: `phase3_event_lifecycle_en_ln.csv`,
`phase3_duracao_por_tipo_classe.csv`, `3H2_ciclo_vida.png`,
`3H3_ciclo_vida_subsuperficie_atmosfera.png`.

## 4. Nascimento espaco-temporal

### 4.1 Onde nasce o sinal?

O mapa longitude-lag (`3C2`) mostra maxima organizacao no Pacifico central,
perto de **150W**:

| janela de lead | lag do maximo | longitude do maximo | correlacao |
|---|---:|---:|---:|
| 0-4 semanas | 0 sem | 151.6W | 0.957 |
| 5-12 semanas | 6 sem | 147.4W | 0.846 |
| 13-26 semanas | 14 sem | 142.4W | 0.680 |
| 18-28 semanas | 18 sem | 133.9W | 0.569 |
| 27-40 semanas | 28 sem | 170.9E | 0.382 |

Leitura: a genese observavel por SSTA longitude-lag se organiza perto de
150W, sobretudo em lags de 1 a 3 meses. Quando se exige antecedencia de 6 meses
ou mais, o maximo enfraquece e aparece mais a oeste (~170E), o que e coerente
com recarga/remocao dinamica de longo prazo, nao com uma anomalia de superficie
ja madura na Nino 3.4.

### 4.2 Antecedencia perto de 150W

| lag | maximo global | r global | maximo 150W +/-5 | r perto de 150W |
|---:|---:|---:|---:|---:|
| 0 sem | 151.6W | 0.957 | 151.6W | 0.957 |
| 4 sem | 149.4W | 0.882 | 149.4W | 0.882 |
| 8 sem | 147.4W | 0.808 | 147.4W | 0.808 |
| 12 sem | 143.6W | 0.725 | 145.1W | 0.723 |
| 14 sem | 142.4W | 0.680 | 149.9W | 0.675 |
| 20 sem | 142.4W | 0.519 | 145.1W | 0.511 |
| 24 sem | 170.9E | 0.420 | 152.6W | 0.398 |
| 28 sem | 170.9E | 0.382 | 152.6W | 0.291 |

Assim, a resposta tecnica e: **sim, o nascimento/organizacao termica do El
Nino aparece por volta de 150W com antecedencia**, mas essa antecedencia e mais
defensavel em **4-14 semanas**. Para 20-28 semanas, o projeto deve falar em
recarga/precursor de fundo, nao em nascimento termico local.

Fontes: `phase3C_mapa_lon_lag.csv`, `3C2_mapa_lon_lag.png`,
`3D2_mapa_lon_lag_fdr.png`.

## 5. Rigor estatistico

A Fase 3 nao aceita apenas correlacao bruta. O protocolo aplicou:

- correcao por tamanho amostral efetivo (`N_eff`);
- intervalos de confianca Fisher-z;
- FDR em mapas longitude-lag;
- sensibilidade temporal sem breakpoint, por bootstrap movel e leave-one-event-out;
- avaliacao LOO/nested LOO para previsao de pico.

Resultados:

| teste | resultado |
|---|---:|
| celulas longitude-lag validas | 25.600 |
| sobreviventes FDR | 9.103 |
| proporcao sobrevivente | 35.6% |
| p critico FDR | 0.0178 |
| bootstrap temporal | blocos pareados de 26, 52 e 78 semanas |
| influencia de extremos | retirada de um evento El Nino ou La Nina por vez |

As variaveis que sobrevivem ao controle inferencial do 3D com maior forca e interpretacao fisica sao:
OHC0-100, T50m, tilt, SSH, TCWV, tilt_slope, OHC0-300, OHC0-700,
omega850, u850, SSR, u200, omega500, SLHF, u10, tau_x e D20. O 3E nao
adiciona um gate binario: ele informa IC bootstrap e dependencia de eventos.
WWV permanece candidato fisico do bloco de recarga, sem privilegio sobre
D20/OHC/SSH/tilt e sem ressalva artificial baseada em um corte em 2010.

Fontes: `phase3D_ranking_significativo.csv`,
`phase3D_mapa_fdr_resumo.csv`, `phase3E_estabilidade.csv`,
`phase3E_sensibilidade_resumo.csv`, `phase3E_bootstrap_blocos.csv` e
`phase3E_leave_one_event_out.csv`.

## 6. O que melhor informa cada fase

### 6.1 Discriminantes globais entre fases

Ranking por `epsilon2` de Kruskal entre genese, crescimento, pico e decaimento:

| variavel | epsilon2 |
|---|---:|
| SSTA Nino 3.4 | 0.649 |
| T100m | 0.525 |
| tilt_m | 0.509 |
| OHC0-300 | 0.503 |
| SSH | 0.490 |
| OHC0-700 | 0.473 |
| OHC0-100 | 0.460 |
| tilt_slope | 0.417 |
| D20 | 0.394 |
| TCWV | 0.381 |

Leitura: a SSTA separa as fases porque define a maturacao termica, mas o
arcabouco fisico que explica a transicao e subsuperficial: T100m, tilt, OHC e
SSH. Isso e a assinatura classica de recarga/descarga do Pacifico equatorial.

### 6.2 Importancia percentual descritiva por fase

Percentual calculado como `abs(nivel_z_medio)` normalizado dentro da fase. Ele
descreve a assinatura da fase, nao causalidade isolada.

| fase | principais variaveis |
|---|---|
| Genese | T300m 6.42%, T50m 5.85%, OHC0-100 5.61%, v10 5.15%, WWV 5.12%, omega500 4.83%, tilt_slope 4.81%, SSR 4.81% |
| Crescimento | OHC0-100 5.17%, SSH 5.12%, OHC0-300 4.94%, T100m 4.92%, OHC0-700 4.84%, T50m 4.74%, SSTA 4.73%, tilt 4.21% |
| Pico | tilt 4.92%, SSTA 4.86%, OHC0-300 4.57%, OHC0-700 4.40%, SSH 4.39%, tilt_slope 4.37%, T100m 4.35%, OHC0-100 4.32% |
| Decaimento | SSR 6.69%, omega500 6.15%, TCWV 5.86%, SSTA 5.48%, tilt_slope 5.40%, omega850 5.29%, u200 5.05%, STR 4.73% |

Fontes: `phase3_discriminantes_por_periodo.csv`,
`phase3_fase_stats_variaveis.csv`, `phase3L_discriminantes_heatmap.png`.

## 7. Genese: nascimento fisico

Na genese, SSTA ainda nao e o melhor separador de intensidade futura. A
estatistica de separacao pre-onset (-26..0 semanas contra pico ONI) mostra:

| variavel | Spearman rho | p bruto | n |
|---|---:|---:|---:|
| D20 | 0.545 | 0.067 | 12 |
| OHC0-300 | 0.343 | 0.276 | 12 |
| WWV | 0.280 | 0.379 | 12 |
| tilt | 0.217 | 0.499 | 12 |
| SSH | 0.189 | 0.557 | 12 |
| tau_x | 0.140 | 0.665 | 12 |
| SSTA Nino 3.4 | 0.063 | 0.846 | 12 |

Interpretacao: para estimar severidade futura, a SSTA de genese sozinha e
fraca. O melhor sinal fisico inicial e a profundidade/estrutura da termoclina
(D20), seguida por recarga (OHC/WWV) e altura dinamica (SSH). O p-valor de D20
fica em zona sugestiva, nao conclusiva, porque a amostra de eventos e pequena
(12 eventos), mas a direcao fisica e coerente.

## 8. Crescimento e acoplamento Bjerknes

O crescimento e a fase em que a recarga vira superficie. As variaveis que mais
crescem da genese para o crescimento sao:

| variavel | delta z genese->crescimento |
|---|---:|
| tilt_m | 1.241 |
| SSH | 1.127 |
| T300m | 1.080 |
| OHC0-700 | 1.072 |
| TCWV | 1.070 |
| SSTA Nino 3.4 | 1.054 |
| tilt_slope | 1.048 |
| OHC0-300 | 1.033 |
| T100m | 1.004 |
| OHC0-100 | 0.932 |

Leitura fisica:

- **OHC/SSH/D20/tilt** mostram recarga e aprofundamento/achatamento dinamico da
  termoclina no centro-leste do Pacifico.
- **tau_x/u10/u850** representam o componente atmosmerico do acoplamento:
  anomalias de oeste reduzem a ressurgencia fria e favorecem propagacao de
  ondas Kelvin de downwelling.
- **TCWV, omega, SSR e fluxos turbulentos** expressam a resposta atmosferica:
  conveccao/nuvens/vapor acompanham a maturacao do evento.

Os mapas e Hovmollers indicam que os sinais longitudinais evoluem de oeste para
leste: a recarga/pulso dinamico aparece no oeste/centro antes de aquecer
plenamente o centro-leste.

## 9. Ondas Kelvin e vento

O diagnostico 3F identifica pulsos de SLA/SSH compativeis com downwelling
Kelvin, usando velocidade assumida de 2.4 m/s:

| janela | max SLA m | longitude max | tau_x medio Pa | frac tau_x oeste | leitura |
|---|---:|---:|---:|---:|---|
| 1997/98 | 0.326 | 80W | 0.022 | 0.832 | forte suporte dinamico |
| 2015/16 | 0.233 | 80W | 0.014 | 0.764 | suporte diagnostico |
| 2023/24 | 0.241 | 129W | -0.002 | 0.470 | diagnostico, menos persistente |
| 2025/26 atual | 0.214 | 100W | -0.010 | 0.290 | pulso visivel, vento medio desfavoravel |

Pulsos oeste-leste marcados:

| janela | pulso oeste | chegada estimada leste | SLA oeste m |
|---|---:|---:|---:|
| 1997/98 | 1997-01-05 | 1997-03-11 | 0.219 |
| 1997/98 | 1997-03-24 | 1997-05-28 | 0.226 |
| 2015/16 | 2015-01-07 | 2015-03-13 | 0.074 |
| 2015/16 | 2015-03-24 | 2015-05-28 | 0.105 |
| 2023/24 | 2023-01-21 | 2023-03-27 | 0.092 |
| 2023/24 | 2023-06-04 | 2023-08-08 | 0.080 |

Interpretacao: Kelvin e evidencia dinamica, nao detector automatico. Ela ajuda
a explicar o crescimento quando aparece junto com OHC/SSH/D20 e vento de oeste.
Sozinha, nao deve ser usada como previsao operacional.

Fontes: `phase3F_kelvin_eventos_resumo.csv`, `phase3F_kelvin_setas.csv`,
`3F1_hovmoller_sla_kelvin.png`, `3F2_taux_sla_eventos.png`.

## 10. O que antecede e prediz o pico

### 10.1 Ranking fisico-estatistico de antecedentes

Correlacao defasada bruta e FDR indicam:

| variavel | lag | r | leitura |
|---|---:|---:|---|
| OHC0-100 | 1 sem | 0.901 | estado/apoio curto |
| SSH | 6 sem | 0.755 | precursor curto |
| OHC0-300 | 6 sem | 0.738 | precursor curto |
| OHC0-700 | 6 sem | 0.701 | precursor curto |
| T100m | 7 sem | 0.697 | precursor curto |
| D20 | 15 sem | 0.545 | precursor antecipado |
| WWV | 20 sem | 0.516 | candidato basinwide de recarga; parcialmente redundante com D20/OHC/SSH/tilt |
| tau_x | 1 sem | 0.478 | vento/acoplamento curto |

### 10.2 Skill preditivo por hindcast LOO

| modelo | horizonte | variaveis | r_LOO | MAE C | skill |
|---|---:|---|---:|---:|---:|
| ohc300_20w | 20 sem | OHC0-300 | 0.866 | 0.214 | 0.615 |
| wind_recharge_15w | 15 sem | OHC0-300 + SSH + tau_x | 0.891 | 0.223 | 0.599 |
| ssh_20w | 20 sem | SSH | 0.793 | 0.240 | 0.568 |
| tau_x_15w | 15 sem | tau_x | 0.850 | 0.278 | 0.501 |
| recharge_core_20w | 20 sem | OHC0-300 + SSH + D20 | 0.717 | 0.331 | 0.405 |

Resultado nested LOO:

| protocolo | r | MAE C | RMSE C | skill |
|---|---:|---:|---:|---:|
| 3I nested LOO candidato completo | 0.738 | 0.379 | 0.465 | 0.319 |
| 3K nested LOO PCA/representante PC1 | 0.801 | 0.283 | 0.389 | 0.491 |

Interpretacao: o melhor preditor da amplitude do pico nao e a SSTA atual, mas
o **eixo de recarga subsuperficial**, representado de forma parcimoniosa por
OHC0-300. O modelo com muitas variaveis pode parecer fisicamente rico, mas a
amostra de eventos e pequena; por isso o PCA/representante de PC1 e mais
conservador.

## 11. Pico

O pico e a fase de maior organizacao multivariada:

| PCA por fase | PC1 | PC1+PC2 |
|---|---:|---:|
| genese | 28.8% | 48.1% |
| crescimento | 31.1% | 56.7% |
| pico | 43.3% | 70.0% |
| decaimento | 33.3% | 61.1% |

No pico, PC1 combina atmosfera e superficie:
TCWV, omega850/omega500, SSR, tilt_slope, T50m, SSHF e tilt. PC2 preserva o
eixo oceanico subsuperficial. Isso mostra que, no pico, o sistema esta mais
integrado: superficie, conveccao, radiacao, calor turbulento, termoclina e OHC
se movem como um modo acoplado.

O pico espacial dos eventos fortes/muito fortes tende a puxar para o centro-
leste/leste do Pacifico. Epicentros notaveis:

| evento | classe | longitude epicentro | SSTA max C |
|---|---|---:|---:|
| 1982/83 | muito_forte | 127W | 5.44 |
| 1997/98 | muito_forte | 106W | 5.39 |
| 2014/16 | muito_forte | 123W | 4.19 |
| 2023/24 | forte | 102W | 3.45 |

Fonte: `phase3A_picos_epicentros.csv`, `3B3_mapa_composto_pico.png`,
`3G3_mapa_ssta_lon.png`.

## 12. Decaimento

As variaveis que mais caem do pico para o decaimento sao:

| variavel | queda z pico->decaimento |
|---|---:|
| T100m | 1.764 |
| D20 | 1.734 |
| T150m | 1.733 |
| OHC0-300 | 1.579 |
| WWV | 1.554 |
| OHC0-700 | 1.538 |
| SSH | 1.496 |
| T200m | 1.295 |

Leitura: o decaimento e fisicamente uma **descarga da recarga oceanica**. A
SSTA ainda pode permanecer positiva por algumas semanas/meses, mas D20, OHC,
WWV e SSH ja recuam. Isso e consistente com a ideia de que o pico termico e
mantido pela subsuperficie; quando essa base colapsa, a superficie perde
sustentacao.

Na assinatura percentual do decaimento, aparecem SSR, omega500, TCWV, SSTA,
tilt_slope, omega850, u200, STR e fluxos de calor. Essas variaveis descrevem a
reorganizacao atmosferica apos a maturacao, mas o aviso precoce de queda vem
principalmente do oceano subsuperficial.

## 13. Sintese cronologica integrada

| tempo relativo | local/fenomeno | variaveis-chave | evidencia | interpretacao |
|---:|---|---|---|---|
| -52 a -26 sem | Pacifico oeste/central; recarga ampla | WWV, OHC, D20, SSH | memoria SSTA 27 sem; WWV lag 20; D20 lag 15 | pre-condicionamento; nao e pico certo |
| -26 a -14 sem | centro-oeste para centro | D20, OHC0-300, OHC0-700, SSH | D20 r=0.545 em 15 sem; OHC/SSH estaveis | termoclina e recarga comecam a diferenciar severidade |
| -14 a -4 sem | ~150W a 145W | SSTA longitude-lag, OHC, SSH, T100 | r=0.68 a 0.88 perto de 150W | nascimento termico observavel |
| -23 a 0 sem | Nino 3.4, centro-leste | OHC, SSH, tilt, tau_x, TCWV | crescimento medio 21-23 sem | acoplamento Bjerknes e Kelvin |
| -20 a -15 sem | subsuperficie Nino 3.4 | OHC0-300, SSH, tau_x | skill OHC0-300 20w=0.615; wind_recharge 15w=0.599 | melhor janela para prever amplitude |
| -1 a +5 sem | centro-leste/leste | SSTA, tilt, OHC, atmosfera | faixa visual de pico; PC1+PC2=70% | maturacao acoplada |
| +5 a +18 sem | Nino 3.4 e subsuperficie | T100, D20, T150, OHC, WWV, SSH | maiores quedas pico->decaimento | descarga e fim do evento |

## 14. Limitacoes e cuidados

1. **Correlacao nao e causalidade.** A inferencia causal vem da coerencia
   fisica Bjerknes/Kelvin/recarga, nao apenas de `r`.
2. **Amostra de eventos e pequena.** Mesmo com serie semanal longa, o numero de
   eventos independentes e ~12. Por isso o projeto usa LOO/nested LOO.
3. **SSTA tem persistencia forte.** Qualquer precursor precisa superar o
   baseline de memoria da propria SSTA, que e ~27 semanas.
4. **WWV e util fisicamente, mas ficou regime-dependente na estabilidade.**
   Deve entrar como apoio/ressalva, nao como preditor isolado absoluto.
5. **Kelvin e diagnostico visual/dinamico.** O Hovmoller valida coerencia de
   propagacao, mas nao substitui o conjunto OHC/SSH/D20/tau_x.
6. **Pico deve ser faixa.** Em especial nos eventos fracos, o pico pontual e
   instavel; em eventos fortes/muito fortes, a janela de maturacao e mais
   estreita e fisicamente mais robusta.

## 15. Conclusao

A caracterizacao cronologica da Fase 3 indica que o El Nino nasce como uma
combinacao de recarga subsuperficial e organizacao termica no Pacifico central.
O sinal de superficie aparece de forma defensavel perto de **150W** com
antecedencia de **4 a 14 semanas**, enquanto a previsao de amplitude do pico
depende de sinais mais antecipados de **recarga oceanica** em **15 a 20
semanas**, principalmente **OHC0-300**, **SSH**, **D20** e **tau_x**.

O crescimento e sustentado por acoplamento Bjerknes e por pulsos Kelvin de
downwelling quando vento de oeste e SLA/SSH sao coerentes. O pico representa a
fase de maior coerencia multivariada, e o decaimento comeca quando a
subsuperficie descarrega: T100m, D20, T150m, OHC, WWV e SSH caem antes ou junto
da perda da SSTA. Portanto, para caracterizar e antecipar El Nino no projeto,
o eixo central nao deve ser "SSTA isolada", mas sim **recarga + termoclina +
altura dinamica + vento**, com a SSTA como expressao final e criterio de
classificacao.

## 16. Artefatos usados

- `data/processed/parquet/statistics/phase3_events_en_ln.csv`
- `data/processed/parquet/statistics/phase3_event_lifecycle_en_ln.csv`
- `data/processed/parquet/statistics/phase3_duracao_por_tipo_classe.csv`
- `data/processed/parquet/statistics/phase3B_memoria_persistencia.csv`
- `data/processed/parquet/statistics/phase3C_mapa_lon_lag.csv`
- `data/processed/parquet/statistics/phase3C_ranking_lags.csv`
- `data/processed/parquet/statistics/phase3D_ranking_significativo.csv`
- `data/processed/parquet/statistics/phase3D_mapa_fdr_resumo.csv`
- `data/processed/parquet/statistics/phase3E_estabilidade.csv`
- `data/processed/parquet/statistics/phase3E_sensibilidade_resumo.csv`
- `data/processed/parquet/statistics/phase3E_bootstrap_blocos.csv`
- `data/processed/parquet/statistics/phase3E_leave_one_event_out.csv`
- `data/processed/parquet/statistics/phase3F_kelvin_eventos_resumo.csv`
- `data/processed/parquet/statistics/phase3F_kelvin_setas.csv`
- `data/processed/parquet/statistics/phase3H_separacao_genese.csv`
- `data/processed/parquet/statistics/phase3I_skill_horizontes.csv`
- `data/processed/parquet/statistics/phase3I_nested_loo_metricas.csv`
- `data/processed/parquet/statistics/phase3K_previsao_pico_nested_loo_metricas.csv`
- `data/processed/parquet/statistics/phase3_discriminantes_por_periodo.csv`
- `data/processed/parquet/statistics/phase3_fase_stats_variaveis.csv`
- `data/processed/parquet/statistics/phase3_pca_por_fase.csv`
- `data/processed/parquet/statistics/phase3_pca_loadings_por_fase.csv`
- `data/processed/figures/fase3/3C2_mapa_lon_lag.png`
- `data/processed/figures/fase3/3D2_mapa_lon_lag_fdr.png`
- `data/processed/figures/fase3/3F1_hovmoller_sla_kelvin.png`
- `data/processed/figures/fase3/3H2_ciclo_vida.png`
- `data/processed/figures/fase3/3H3_ciclo_vida_subsuperficie_atmosfera.png`
- `data/processed/figures/fase3/phase3L_pca_por_fase.png`
