# O pico do El Nino como FAIXA, nao como ponto: bibliografia e motivacao pratica

Data: 2026-07-09. Documento de fundamentacao para a decisao metodologica do NINO-BRASIL
de delimitar o pico dos eventos El Nino por uma **janela (faixa) de pico**, e nao por uma
unica semana/mes de maximo.

## 1. O problema

A serie semanal (e mesmo mensal) de SSTA na caixa Nino 3.4 e contaminada por ruido
intrassazonal: MJO, rajadas de vento de oeste (WWBs), ondas de instabilidade tropical (TIWs)
e a propria incerteza da climatologia de base. Perto do maximo de um evento, a curva fica
"achatada": varios meses consecutivos diferem entre si por menos que a incerteza tipica do
indice (~0.1 C). Escolher UMA semana como "o pico" e, nesse regime, uma decisao instavel:
muda com o dataset (OISST x ERSST), com a climatologia (1991-2020 x 1971-2000), com a
suavizacao adotada e ate com a reprocessagem da mesma fonte.

Evidencia interna do projeto (tabela `phase3B_faixa_pico_eventos.csv`): com o criterio
"meses com ONI local >= pico - 0.1 C", os eventos fortes/muito fortes tem faixa de pico
estreita (1-2 meses: 1982/83, 1997/98, 2015/16, 2023/24), enquanto eventos fracos tem
faixa larga e mal definida (2004: 5 meses; 2018/19: 6 meses). Ou seja: quanto mais fraco o
evento, menos significa "o mes do pico" - e mais necessaria e a faixa.

## 2. O que a literatura internacional sustenta

1. **O proprio indice operacional ja e uma janela.** O ONI da NOAA/CPC e definido como
   media movel de 3 meses da SSTA Nino 3.4 justamente para filtrar variabilidade
   intrassazonal; eventos exigem 5 estacoes moveis consecutivas acima do limiar. A decisao
   do CPC de nunca declarar pico em base semanal e um reconhecimento institucional de que
   o maximo pontual nao e robusto. (NOAA CPC, ONI v5; NCAR Climate Data Guide, Nino SST
   Indices.)

2. **Phase-locking sazonal: a maturacao trava no inverno boreal.** Desde o composto
   canonico de Rasmusson & Carpenter (1982), sabe-se que eventos El Nino amadurecem
   preferencialmente em novembro-janeiro (NDJ); a desvio-padrao da SSTA Nino 3.4 maximiza
   em NDJ. A literatura de mecanismos (Tziperman et al. 1998; An & Wang 2001, "Mechanisms
   of locking of the El Nino and La Nina mature phases to boreal winter", J. Climate;
   Stein et al. 2014; Li et al. 2025, GRL) trata o pico como uma **estacao de maturacao**
   ("mature phase"), nao como uma data. Estudos de phase-locking medem o fenomeno por
   **histogramas do mes de pico** - a dispersao desses histogramas e em si a prova de que o
   pico pontual carrega incerteza de 1-3 meses.

3. **Verificacao de previsao usa alvo sazonal.** Os protocolos de verificacao
   (WMO SVSLRF; Barnston et al. 2012, BAMS; plumas do IRI/CPC) definem o alvo como a
   **estacao DJF/NDJ**, e a habilidade e medida contra a media de 3 meses. Prever "a semana
   do maximo" nao e um alvo verificavel operacionalmente com n pequeno de eventos.

4. **O pico pode nem ser unico - no tempo e no espaco.** O El Nino 2023/24 teve "pico
   espacial duplo" documentado (Communications Earth & Environment, 2024, "On the spatial
   double peak of the 2023-2024 El Nino"); eventos como 2014-16 tiveram maximos locais
   multiplos ao longo de 19 estacoes. Santoso et al. (2017, Reviews of Geophysics), na
   revisao dos extremos de ENSO, comparam eventos pela **evolucao da fase madura**, nao por
   um instante. Alem disso, a ECMWF (2026, "Measuring the strength of El Nino - relative
   Nino indices") mostra que a propria AMPLITUDE do pico muda com a referencia usada -
   mais uma razao para tolerancia ao redor do maximo.

## 3. Motivacao pratica para o NINO-BRASIL

- **Alvo de previsao defensavel (Fases 3I/5).** Com 12 eventos, prever amplitude + uma
  janela de pico e estatisticamente viavel; prever a semana exata do maximo nao e. A
  projecao condicional 2025/26 ja entrega "janela condicional" - a faixa e o conceito
  coerente com isso.
- **Robustez do rotulo.** Compostos alinhados "ao pico" (3B1, 3G, 3H, 3L) mudam se o pico
  pontual mudar de mes por ruido; com a faixa, o alinhamento e estavel e auditavel.
- **Monitoramento em tempo real.** Em 2025/26 nao se sabera "a semana do pico" senao
  meses depois; mas e possivel declarar em tempo quase real que o evento **entrou na faixa
  de pico** (ONI para de crescer dentro da tolerancia e o calendario entra na janela de
  phase-locking NDJ). Isso e acionavel para a teleconexao com a chuva no Brasil (Fase 4),
  que responde a forcante sustentada da estacao, nao a uma semana.
- **Comunicacao honesta.** "Pico de 1.9 C em nov/2023-dez/2023" comunica a incerteza real;
  "pico em 3 de dezembro" sugere precisao que os dados nao tem.

## 4. Definicao adotada no projeto

- **Pico central**: mes de maximo do ONI local (media movel de 3 meses, OISST local,
  climatologia 1991-2020).
- **Faixa de pico**: conjunto contiguo de meses com `ONI local >= pico - 0.1 C`.
  A tolerancia de 0.1 C corresponde a escala de incerteza tipica do indice
  (dataset/climatologia/suavizacao).
- **No eixo semanal dos compostos**: faixa = semanas com SSTA composta `>= max - 0.1 C`
  (usada em 3H2/3H3 como faixa dourada em torno da semana 0).

Saidas que materializam a decisao:

| Saida | Conteudo |
|---|---|
| `phase3B_faixa_pico_eventos.csv` | pico central, inicio/fim e largura da faixa por evento |
| `3B4_faixa_pico_oni.png` | ONI local 1981-2026 com faixas de pico sombreadas e mes/ano central |
| `3H2_ciclo_vida.png` / `3H3_...png` | faixa de pico composta (max-0.1 C) sobre o ciclo de vida |
| `3A2_hovmoller_ssta.png` + `phase3A_picos_epicentros.csv` | epicentro espacial (longitude) do pico com mes/ano |

## 5. Referencias

- NOAA/CPC. *Oceanic Nino Index (ONI) v5* - definicao operacional do indice e de evento.
  https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
- NCAR Climate Data Guide. *Nino SST Indices (Nino 1+2, 3, 3.4, 4; ONI and TNI)*.
  https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni
- Rasmusson, E. M., & Carpenter, T. H. (1982). Variations in tropical sea surface
  temperature and surface wind fields associated with the Southern Oscillation/El Nino.
  *Monthly Weather Review*, 110, 354-384.
- An, S.-I., & Wang, B. (2001). Mechanisms of locking of the El Nino and La Nina mature
  phases to boreal winter. *Journal of Climate*, 14, 2164-2176.
  https://journals.ametsoc.org/view/journals/clim/14/9/1520-0442_2001_014_2164_molote_2.0.co_2.xml
- Tziperman, E., et al. (1998). Locking of El Nino's peak time to the end of the calendar
  year in the delayed oscillator picture of ENSO. *Journal of Climate*, 11, 2191-2199.
- Li, X., et al. (2025). Why does El Nino tend to peak in boreal winter: I. The role of the
  ocean-atmosphere coupling strength. *Geophysical Research Letters*.
  https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024GL114403
- Santoso, A., McPhaden, M. J., & Cai, W. (2017). The defining characteristics of ENSO
  extremes and the strong 2015/2016 El Nino. *Reviews of Geophysics*, 55, 1079-1129.
  https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017RG000560
- Communications Earth & Environment (2024). On the spatial double peak of the 2023-2024
  El Nino event. https://www.nature.com/articles/s43247-024-01870-1
- Barnston, A. G., et al. (2012). Skill of real-time seasonal ENSO model predictions during
  2002-11. *BAMS*, 93, 631-651. https://doi.org/10.1175/BAMS-D-11-00111.1
- WMO Lead Centre for SVSLRF - verificacao de previsao sazonal por estacao-alvo.
  https://wmolc.org
- ECMWF (2026). Measuring the strength of El Nino - introducing relative Nino indices.
  https://www.ecmwf.int/en/about/media-centre/science-blog/2026/measuring-strength-el-nino
- IRI/CPC. ENSO Quick Look e plumas de previsao por estacao de 3 meses.
  https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/
