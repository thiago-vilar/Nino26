# Fase 3 - Relatorio condensado de recomendacoes

Substitui e consolida: parecer da Fase 3 e recomendacoes por escrito.

## 1. Escala Temporal Da Fase 3

A Fase 3 passa a ter **escala operacional semanal**. A matriz principal,
correlacoes, persistencia e checagens fisicas sao calculados por semana
(`week_ending_sunday`). A escala mensal continua apenas para:

- resumir a propria SSTA OISST diaria baixada;
- comparar eventos historicos dentro da mesma base de SST do projeto;
- suavizar crescimento/decaimento de eventos em 3 meses.

Janelas:

| Escopo | Janela | Uso |
|---|---|---|
| Comum da Fase 3 | 1993-presente | Compara superficie e subsuperficie na mesma base, limitada por GLORYS12 |
| So superficie | 1982-presente | Sensibilidade OISST/SSTA sem subsuperficie |
| Estabilidade | 1993-2009 e 2010-presente | Testa mudanca de regime/lead |

## 2. Perguntas Dos Notebooks

A Fase 3 e o bloco fisico do Pacifico e encerra o escopo ativo atual. Ela nao
entra em chuva no Brasil, teleconexao regional ou ML. A Fase 3 responde como o
sinal do Nino 3.4 se forma, quanto tempo ele guarda memoria e quais relacoes
fisicas sao defensaveis em um parecer auditavel.

| Notebook | Pergunta | Metodo | Saida de decisao |
|---|---|---|---|
| **3A - Indices** | Quais series fisicas descrevem o sistema ENOS? | Matriz semanal de SSTA, WWV, D20, OHC, termoclina, SSH e `tau_x_anom`. | `indices_semanais` com cobertura, fonte e unidade. |
| **3B - Alvo** | Como os eventos nascem, crescem, atingem pico e decaem? | Eventos mensais derivados da SSTA OISST local, trajetoria semanal do Nino 3.4, taxas de crescimento/decaimento e persistencia. | Tabela de eventos, fases do evento e matriz de memoria. |
| **3C - Precursores** | O que antecede o pico do Nino 3.4? | Correlacoes defasadas semanais, preditor liderando, lags 0-78 semanas. | Ranking preliminar de lead e forca. |
| **3D - Rigor** | O que sobrevive estatisticamente? | `N_eff`, teste-t, IC95 Fisher-z e FDR sobre o conjunto total de testes. | Ranking filtrado por significancia robusta. |
| **3E - Estabilidade** | O sinal vale antes e depois da mudanca recente de regime? | Repetir 3C/3D em 1993-2009 e 2010-presente. | Apenas relacoes estaveis seguem para o parecer final. |
| **3F - Kelvin/SLA** | A dinamica equatorial mostra propagacao compativel com ondas Kelvin? | Hovmoller SLA/SSH e `tau_x_anom` por janela de evento. | Evidencia dinamica entra como diagnostico fisico, nao como detector automatico. |

**Regra de corte:** so entra no parecer final da Fase 3 o que sobrevive a D **e** E.
O 3F e evidencia dinamica qualitativa; nao cria variavel preditora nova.

**Limite de escopo:** o projeto ativo para na Fase 3. Nao ha etapa ativa de ML, redes neurais ou teleconexao
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
6. **Escopo de bacia:** o projeto e estritamente Pacifico -> Brasil; nenhuma covariavel de outra bacia entra no parecer.
7. **Diagrama de fase (C):** ocupacao dos quadrantes WWV x SSTA; sequencia esperada do oscilador: recarregado/frio -> recarregado/quente -> descarregado/quente -> descarregado/frio.
8. **Estabilidade (E):** repetir correlacoes defasadas por subperiodo; mudanca no lead do WWV reproduz McPhaden (2012) e entra como limite fisico do parecer.
9. **Colinearidade:** D20 ~= OHC ~= WWV ~= SSH/tilt em parte do sinal; escolher representantes por estabilidade estatistica e interpretabilidade fisica, sem contar o mesmo bloco como evidencias independentes.

## 5. Notebook 3F - Ondas De Kelvin Por SLA/SSH E Vento

A hipotese testada em 3F e dinamica: se a banda equatorial mostra faixas de SLA
positivo propagando de oeste para leste, coerentes com ondas Kelvin de
downwelling, e se essa leitura e acompanhada por anomalias de oeste em
`tau_x_anom`.

### 5.1 Hovmoller SLA/SSH

- Usar SSH/SLA na faixa equatorial 2S-2N, 120E-80W.
- Mostrar longitude oficial 120E -> 80W; Nino 3.4 sombreado em 170W-120W.
- Remover a media da janela por longitude para destacar SLA local.
- Interpretar faixas inclinadas oeste->leste como evidencia qualitativa de
  propagacao Kelvin.

### 5.2 Coerencia Com Vento

- `tau_x_anom` positivo indica anomalia de oeste no proxy local.
- A leitura e mais forte quando SLA positivo no centro-leste aparece junto de
  anomalias de oeste antes/durante o crescimento do evento.
- O resultado e diagnostico fisico, nao uma declaracao operacional automatica.

## 6. Testes De Sanidade

Falha nesses itens e tratada como bug de pipeline antes de virar interpretacao cientifica.

1. Lead otimo WWV -> Nino 3.4 em aproximadamente 6-9 meses; WWV-W > WWV total.
2. Persistencia despenca cruzando maio-junho.
3. `tau_x_anom` deve ser interpretado como vento/acoplamento, nao como definicao de El Nino.
4. Mapas longitudinais devem manter longitude oficial 120E -> 80W, com Nino 3.4 sombreado.
5. A evidencia de Kelvin deve ser descrita como qualitativa/diagnostica.

## 7. Saidas Auditaveis

Figuras da Fase 3 seguem a convencao `3A1_...png`: Fase 3, notebook A,
figura 1. Quando um notebook precisa de mais imagens do mesmo subcontexto,
incrementa-se o indice (`3A2`, `3A3`). O catalogo e o relatorio final ficam em
`notebooks/fase3/INDICE_FIGURAS_FASE3.md` e
`notebooks/fase3/RELATORIO_FINAL_FASE3.md`.

```text
data/processed/parquet/features/phase3_indices_semanais.csv
data/processed/parquet/features/phase3_eventos.csv
data/processed/parquet/features/phase3_persistencia_semanal.csv
data/processed/parquet/features/phase3_fases_recarga.csv
data/processed/parquet/features/phase3_estabilidade_subperiodos.csv
data/processed/parquet/features/phase3f_kelvin_leadlag.csv
data/processed/parquet/features/nino34_monthly_oisst.csv
data/processed/parquet/statistics/phase3C_lag_correlacoes.csv
```

Campos minimos de `phase3C_lag_correlacoes.csv`: `r`, `N`, `N_eff`, `p`, `p_FDR`, `IC95`.

### 7.1 Aferimento Preditivo (3I/3K)

A Fase 3 encerra com aferimento preditivo exploratorio, sem ML pesado. O LOO
simples continua como triagem por variavel, mas a selecao de modelo/horizonte
passa a ser avaliada por **nested leave-one-event-out**: em cada evento externo,
o loop interno escolhe o candidato apenas nos eventos de treino, e o loop externo
mede o erro no evento retido. O baseline e climatologia LOO dos picos de treino;
`skill = 1 - MAE_modelo/MAE_clim`. A projecao 2025/26 e condicional, nao
operacional: assume que o estado recente representa um precursor a `H` semanas do
pico. Saidas:

```text
data/processed/parquet/statistics/phase3I_skill_por_variavel.csv
data/processed/parquet/statistics/phase3I_modelos_candidatos.csv
data/processed/parquet/statistics/phase3I_skill_horizontes.csv
data/processed/parquet/statistics/phase3I_nested_loo_eventos.csv
data/processed/parquet/statistics/phase3I_nested_loo_metricas.csv
data/processed/parquet/statistics/phase3I_nested_loo_selecao.csv
data/processed/parquet/statistics/phase3I_projecao_pico_2026.csv
data/processed/parquet/statistics/phase3K_previsao_pico_loo.csv
data/processed/parquet/statistics/phase3K_previsao_pico_nested_loo_metricas.csv
```

Limites explicitos: n=12 eventos, IC95 indicativo; o hindcast e centrado em
eventos e estima amplitude condicionada a um lead conhecido. A validacao
walk-forward continua, a barreira de primavera, persistencia amortecida e
previsao conjunta de timing + amplitude pertencem a Fase 5.

## 8. Referencias-Chave

- Jin 1997 - oscilador de recarga.
- Meinen & McPhaden 2000 - WWV e ENSO.
- Barnston et al. 2012 (`doi:10.1175/BAMS-D-11-00111.1`) - skill de previsoes ENSO.
- WMO SVSLRF - verificacao por hindcast, climatologia cross-validada e metricas de skill.
- Ambroise & McLachlan 2002 - vies de selecao quando a escolha de variaveis fica fora da validacao.
- Cawley & Talbot 2010 - overfitting na selecao de modelo e vies na avaliacao de desempenho.
- McPhaden 2003; McPhaden 2012 (`doi:10.1029/2012GL051826`).
- Zhao 2021 (`doi:10.1029/2021GL094366`).
- Bretherton 1999 - `N_eff`.
- Wilks 2006; Wilks 2016 (`doi:10.1175/BAMS-D-15-00267.1`) - FDR e significancia de campo.
