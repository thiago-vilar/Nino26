# Fase 3 - Relatorio condensado de recomendacoes

Substitui e consolida: parecer da Fase 3 e recomendacoes por escrito.

## 1. Escala Temporal Da Fase 3

A Fase 3 passa a ter **escala operacional semanal**. A matriz principal,
correlacoes, persistencia, DHW e checagens fisicas sao calculados por semana
(`week_ending_sunday`). A escala mensal continua apenas para:

- resumir a propria SSTA OISST diaria baixada;
- comparar eventos historicos dentro da mesma base de SST do projeto;
- suavizar crescimento/decaimento de eventos em 3 meses.

Janelas:

| Escopo | Janela | Uso |
|---|---|---|
| Comum da Fase 3 | 1993-presente | Compara superficie e subsuperficie na mesma base, limitada por GLORYS12 |
| So superficie | 1982-presente | Sensibilidade OISST/SSTA/DHW sem subsuperficie |
| Estabilidade | 1993-2009 e 2010-presente | Testa mudanca de regime/lead |

DHW so e valido apos a janela de acumulacao. No caso padrao de 12 semanas, a
serie efetiva comeca cerca de 3 meses apos o primeiro dado.

## 2. Perguntas Dos Notebooks

A Fase 3 e o bloco fisico do Pacifico e encerra o escopo ativo atual. Ela nao
entra em chuva no Brasil, teleconexao regional ou ML. A Fase 3 responde como o
sinal do Nino 3.4 se forma, quanto tempo ele guarda memoria e quais relacoes
fisicas sao defensaveis em um parecer auditavel.

| Notebook | Pergunta | Metodo | Saida de decisao |
|---|---|---|---|
| **3A - Indices** | Quais series fisicas descrevem o sistema ENOS? | Matriz semanal de SSTA, WWV, D20, OHC, termoclina, SSH, DHW C-week >=0,5 C e `tau_x_anom`. | `indices_semanais` com cobertura, fonte e unidade. |
| **3B - Alvo** | Como os eventos nascem, crescem, atingem pico e decaem? | Eventos mensais derivados da SSTA OISST local, trajetoria semanal do Nino 3.4, taxas de crescimento/decaimento e persistencia. | Tabela de eventos, fases do evento e matriz de memoria. |
| **3C - Precursores** | O que antecede o pico do Nino 3.4? | Correlacoes defasadas semanais, preditor liderando, lags 0-78 semanas. | Ranking preliminar de lead e forca. |
| **3D - Rigor** | O que sobrevive estatisticamente? | `N_eff`, teste-t, IC95 Fisher-z e FDR sobre o conjunto total de testes. | Ranking filtrado por significancia robusta. |
| **3E - Estabilidade** | O sinal vale antes e depois da mudanca recente de regime? | Repetir 3C/3D em 1993-2009 e 2010-presente. | Apenas relacoes estaveis seguem para o parecer final. |
| **3F - DHW + Kelvin** | Calor superficial integrado acrescenta leitura fisica alem de SSTA e WWV/OHC? | HotSpots diarios -> DHW diario -> reducao semanal; redundancia contra SSTA/WWV/OHC e Hovmoller D20/SSH para direcao Kelvin. | DHW entra no parecer apenas se tiver coerencia fisica e nao for redundante. |

**Regra de corte:** so entra no parecer final da Fase 3 o que sobrevive a D **e** E.
Para DHW, ha uma regra extra: so entra se a leitura fisica nao for redundante com
SSTA e com o estado de recarga WWV/OHC.

**Limite de escopo:** o projeto ativo para na Fase 3. O antigo apendice DHW fica
absorvido como **3F**. Nao ha etapa ativa de ML, redes neurais ou teleconexao
Brasil neste recorte.

## 3. Regioes

Referencias oficiais NOAA/CPC devem ser comunicadas em W/E. A notacao numerica
continua pode aparecer apenas em calculos internos, nunca como legenda
executiva.

| Indice | Caixa | Nota |
|---|---|---|
| Nino 3.4 (alvo) | 5N-5S, 170W-120W | OISST; alvo das series, eventos NOAA/ONI locais e parecer |
| Nino 4 (`u10_anom` / `tau_x_anom`) | 5N-5S, 160E-150W | ERA5; referencia desejada para WWB/Kelvin |
| Banda equatorial diagnostica | 2S-2N, 120E-80W | Hovmoller e mapas longitude x lag; nao e caixa oficial Nino |

## 4. Recomendacoes Essenciais

1. **Anti-vazamento:** configuracao unica para A-E; climatologia, anomalias e z-scores fitados so no periodo de treino. No eixo semanal, usar climatologia harmonica com 2-3 harmonicos anuais, nao 52 medias semanais cruas.
2. **Subsuperficie:** D20 por interpolacao linear da isoterma de 20 C (depth em m, positivo para baixo; paralelizar so no tempo); OHC 0-300 m = rho * cp * integral(T dz); WWV = integral de area da D20. Fonte primaria: ORAS5 ou GLORYS12, com a outra como sensibilidade. Janela real: 1993-presente.
3. **Eventos (B):** criterio local OISST compativel com NOAA/ONI (media movel trimestral da SSTA Nino 3.4 >= +0,5 C por >= 5 estacoes moveis sobrepostas); intensidade pelo pico ONI local: fraco, moderado, forte e muito forte; taxas de crescimento/decaimento em serie suavizada de 3 meses; descartar aceleracao bruta. O evento mensal e projetado na grade semanal para analises de lead.
4. **Persistencia (B):** matriz semanal por mes inicial e lead de 1-52 semanas, com resumo 12x12 mes inicial x lead mensal equivalente; quantifica memoria fisica e barreira de primavera, sem uso de ML.
5. **Correlacoes defasadas (C/D):** lags semanais de 0-78 semanas, preditor liderando; `N_eff = N * (1 - r1_x*r1_y) / (1 + r1_x*r1_y)`; teste-t e IC95 (Fisher-z) com `N_eff`; FDR Benjamini-Hochberg (`alpha=0,10`) sobre o conjunto total de testes.
6. **Controles inter-bacia:** ficam fora do parecer da Fase 3; entram apenas em fases de teleconexao Brasil.
7. **Diagrama de fase (C):** ocupacao dos quadrantes WWV x SSTA; sequencia esperada do oscilador: recarregado/frio -> recarregado/quente -> descarregado/quente -> descarregado/frio.
8. **Estabilidade (E):** repetir correlacoes defasadas por subperiodo; mudanca no lead do WWV reproduz McPhaden (2012) e entra como limite fisico do parecer.
9. **Colinearidade:** D20 ~= OHC ~= WWV ~= SSH/tilt em parte do sinal; escolher representantes por estabilidade estatistica e interpretabilidade fisica, sem contar o mesmo bloco como evidencias independentes.

## 5. Notebook 3F - Hipotese DHW E Ondas De Kelvin

O DHW (Degree Heating Weeks) e a acumulacao, em C-semana, de anomalias de TSM
acima de um limiar dentro de uma janela movel. A hipotese testada nao e "SST
importa"; e se **calor de superficie integrado no tempo** adiciona informacao
alem de SSTA instantanea e do bloco de recarga WWV/OHC.

### 5.1 Construir O Indice DHW Da Via Equatorial

- Ordem obrigatoria: calcular HotSpots e DHW na resolucao diaria; so depois
  reduzir para a semana canonica de 7 dias.
- Caixa principal: Nino 3.4 (5N-5S, 170W-120W); mapas longitudinais usam a
  banda diagnostica 2S-2N, 120E-80W.
- Metrica publicada unica: `dhw_cweek_0p5_12w`.
- Limiar: HotSpot diario entra no somatorio apenas quando SSTA Nino 3.4 >= +0,5 C.
- Janela: 12 semanas para preservar a leitura C-week de calor recente; a
  validacao temporal auxiliar exige media movel de 12 semanas >= +0,5 C por
  20 semanas consecutivas; nao publicar janelas concorrentes na Fase 3.
- Unidade: C-semana.

Formula operacional:

```text
DHW_diario(t) = soma_movel(SSTA_diaria(t) se SSTA_diaria(t) >= 0,5 C, senao 0, janela_dias) / 7
DHW_semanal = max_ou_media_semanal(DHW_diario)
```

### 5.2 Correlacao Parcial Contra Preditores Estabelecidos

Teste principal:

```text
DHW(t) -> pico futuro do Nino 3.4
controlando por WWV e SSTA instantanea
```

Se a correlacao parcial desaparece, DHW e redundante com recarga/SSTA. Se
sobrevive a `N_eff`, IC95 e FDR, ele e candidato a sinal novo.

### 5.3 Redundancia Contra SSTA E Recarga

Sem ML neste escopo. O DHW e avaliado como diagnostico fisico: ele precisa ter
temporizacao coerente, nao ser apenas uma reembalagem da SSTA instantanea, e nao
contradizer o estado de recarga indicado por WWV/OHC.

### 5.4 Direcao Causal Com Ondas De Kelvin

- DHW oeste deve anteceder atividade de Kelvin/downwelling se for precursor.
- DHW leste seguindo a chegada da onda indica consequencia, nao causa.
- Controlar rajadas de vento de oeste (`WWB`) e estado atual da termoclina.

## 6. Testes De Sanidade

Falha nesses itens e tratada como bug de pipeline antes de virar interpretacao cientifica.

1. Lead otimo WWV -> Nino 3.4 em aproximadamente 6-9 meses; WWV-W > WWV total.
2. Persistencia despenca cruzando maio-junho.
3. `tau_x_anom` deve ser interpretado como vento/acoplamento, nao como definicao de El Nino.
4. DHW de 12 semanas so tem valor valido apos 12 semanas de acumulacao.
5. DHW so e aceito no parecer se acrescentar leitura fisica nao redundante sobre SSTA e WWV/OHC.

## 7. Saidas Auditaveis

```text
data/processed/parquet/features/phase3_indices_semanais.csv
data/processed/parquet/features/phase3_eventos.csv
data/processed/parquet/features/phase3_persistencia_semanal.csv
data/processed/parquet/features/phase3_lag_correlacoes.csv
data/processed/parquet/features/phase3_fases_recarga.csv
data/processed/parquet/features/phase3_estabilidade_subperiodos.csv
data/processed/parquet/features/phase3f_dhw_indices_semanais.csv
data/processed/parquet/features/phase3f_dhw_partial_correlations.csv
data/processed/parquet/features/phase3f_dhw_redundancy_checks.csv
data/processed/parquet/features/phase3f_dhw_physical_incremental_summary.csv
data/processed/parquet/features/phase3f_kelvin_leadlag.csv
data/processed/parquet/features/nino34_monthly_oisst.csv
data/processed/parquet/features/nino34_oisst_p90_peaks.csv
```

Campos minimos de `phase3_lag_correlacoes.csv`: `r`, `N`, `N_eff`, `p`, `p_FDR`, `IC95`.
Nao ha saida de modelo, validacao preditiva ou metrica de habilidade neste escopo ativo.

## 8. Referencias-Chave

- Jin 1997 - oscilador de recarga.
- Meinen & McPhaden 2000 - WWV e ENSO.
- McPhaden 2003; McPhaden 2012 (`doi:10.1029/2012GL051826`).
- Zhao 2021 (`doi:10.1029/2021GL094366`).
- Rodriguez-Fonseca 2009 (`doi:10.1029/2009GL040048`).
- Ham 2013 (`doi:10.1038/ngeo1686`).
- Martin-Rey 2014 (`doi:10.1007/s00382-014-2305-3`).
- Bretherton 1999 - `N_eff`.
- Wilks 2006; Wilks 2016 (`doi:10.1175/BAMS-D-15-00267.1`) - FDR e significancia de campo.
