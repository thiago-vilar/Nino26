# Fase 4 - Metodologia revisada: teleconexao ENSO -> chuva no Brasil

**Projeto NINO-BRASIL - Oceanografia Fisica UFPE - Thiago Vilar**
**Revisao:** 2026-07-08 (expansao por parecer do pesquisador). Escopo
estritamente **Pacifico -> Brasil**.

A fronteira entre fases permanece:

- **Fase 3:** alvo = Nino 3.4. Diagnostica pico, memoria e precursores do ENOS.
- **Fase 4:** alvo = **anomalia padronizada de chuva por pixel no Brasil**.
  Diagnostica o ciclo ENSO em fases, seus determinantes, o tempo de resposta
  (lags semanais) e os alvos regionais.

## 0. Principios herdados

- Eixo canonico semanal `week_ending_sunday` (W-SUN); CHIRPS diario e somado
  por semana; a chuva e convertida em **anomalia padronizada por pixel** com
  climatologia harmonica (3 harmonicos anuais) na base 1991-2020, nunca 52
  medias cruas.
- Toda significancia usa `N_eff` (Bretherton) e FDR Benjamini-Hochberg
  (alfa=0,10) sobre o conjunto completo de testes.
- Nenhuma figura sem saida numerica rastreavel.
- Nenhum rotulo ENSO externo: eventos e fases derivam do ONI local OISST.

## 1. Ordem cientifica obrigatoria

```
4.0 Abertura (pre-flight)     -> justificativa, objetivos, metodologia,
                                  inventario dos dados e roteiro das perguntas
4A  Ciclo ENSO em 4 fases     -> genese, crescimento/acoplamento, pico e
                                  decaimento, para El Nino E La Nina
4B  Determinantes das fases   -> estudo puramente estatistico com todas as
                                  variaveis numericas do master semanal
4C  Sinal pixel-a-pixel       -> Pacifico x anomalia CHIRPS por pixel, lags
                                  0-78, Brasil inteiro -> NEB -> Sul
4D  Alvos clusterizados       -> SO DEPOIS do pixel: alvos mais afetados, lag
                                  por tipo de sinal, estabilidade e gate
                                  estatistico da hipotese
```

## 2. Fase 4A - Ciclo ENSO em 4 fases (El Nino e La Nina)

E conhecido no projeto que o ENSO tem 4 fases: **I. Genese, II.
Crescimento/acoplamento, III. Pico, IV. Decaimento**. A 4A formaliza a
separacao logica e estatistica dessas fases, simetrica para EN e LN:

- Eventos: ONI local >= +0,5 C (EN) ou <= -0,5 C (LN) por >= 5 estacoes moveis
  sobrepostas; intensidade pelo pico de |ONI| (classes simetricas).
- Fases por evento: **pico** = plateau com |ONI| >= 90% do maximo do evento;
  **crescimento** = onset -> plateau; **decaimento** = plateau -> fim;
  **genese** = 26 semanas pre-onset (janela de organizacao dos precursores,
  3C/3H), apenas sobre semanas neutras.
- Avaliacao expandida: duracao media e dispersao de cada fase por tipo/classe
  e distribuicao do ONI dentro de cada fase.
- **Plano de fase (assinatura relacional):** cada fase e uma regiao no plano
  WWV x SSTA (e D20 x SSTA); os centroides das 4 fases tracam a trajetoria do
  oscilador de recarga (Jin 1997), simetrica para EN e LN
  (`phase4A_plano_fase.png`).
- Saida central: rotulo semanal `phase4A_fases_semanais.csv`, com uma linha por
  domingo (`tipo`, `fase`, `event_id`, semana relativa ao onset/pico), e
  cronologia por evento em `phase4A_evento_fases_cronologia.csv`.

## 3. Fase 4B - Que variaveis determinam cada fase (puramente estatistico)

Tres testes nao parametricos, sem ML, sobre **todas as variaveis numericas** da
matriz-mestre semanal `nino34_master_weekly.csv` em z-score:

1. **Separacao fase vs neutro:** Mann-Whitney U por fase x variavel x tipo,
   tamanho de efeito = delta de Cliff, FDR BH por tipo.
2. **Discriminancia entre fases:** Kruskal-Wallis sobre as 4 fases dentro de
   cada tipo, com epsilon-quadrado (quem melhor distingue a fase do ciclo).
3. **Determinacao da intensidade:** Spearman entre o estado medio da variavel
   na genese/crescimento de cada evento e o |ONI| do pico (n pequeno:
   leitura indicativa).

E, alem do nivel isolado, **relacoes entre variaveis** (a pergunta: dentre
todas as duplas, qual *relacao* determina a fase?):

4. **Pares (bivariado):** para todos os pares de variaveis disponiveis, tres
   relacoes por semana -
   correlacao movel (13 sem), co-movimento `z_i*z_j` e log da razao de
   volatilidade `sigma_i/sigma_j` - testadas por Kruskal-Wallis/epsilon^2
   entre as 4 fases, com FDR. Ranking em `phase4B_relacoes_pares_fases.csv`.
5. **Estrutura multivariada (complexo):** matriz de correlacao NxN do conjunto
   completo por fase e distancia de Frobenius entre fases -> cada fase tem uma
   assinatura de covariacao propria (`phase4B_matriz_correlacao_por_fase.csv`,
   `phase4B_estrutura_correlacao_distancias.csv`).

## 4. Fase 4C - Sinal pixel-a-pixel com lags semanais (Brasil -> NEB -> Sul)

Pergunta-chave: **quanto tempo demora para o sinal do El Nino (e da La Nina)
aparecer nas anomalias de chuva do NEB e do Sul?**

- Alvo: CHIRPS 0,25 no Brasil, soma semanal, anomalia padronizada por pixel
  (climatologia harmonica). Cache em `phase4_chirps_weekly_zanom.parquet`.
- Correlacao de Pearson `Pacifico(t-L) -> anomalia_chuva_pixel(t)` por pixel,
  variavel e lag (0-78, passo 2), em tres condicoes: todas as semanas, semanas
  El Nino e semanas La Nina (fases do 4A). `N_eff` por pixel; FDR por
  condicao/variavel.
- Janelas de lag: `phase4C_janelas_lag_variavel.csv` registra, para cada
  variavel/condicao/lag, o inicio/fim da janela-alvo, o inicio/fim da janela do
  Pacifico deslocada e o numero de pares semanais usados.
- **Organizacao obrigatoria:** primeiro o Brasil inteiro (atlas completo),
  depois o recorte NEB e o recorte Sul com mapas ampliados, resumo por
  variavel e distribuicao dos lags por regiao.
- Resposta direta em `phase4C_lag_resposta_neb_sul.csv` (lag mediano + IQR por
  regiao, variavel e tipo de sinal).
- Atlas auditavel: `phase4C_atlas_pixel.zarr`.
- Ressalva: CHIRPS pode subestimar extremos convectivos locais.

## 5. Fase 4D - Alvos clusterizados, estabilidade e gate estatistico da hipotese

- Vetor de resposta por pixel: perfis r(lag) de todas as variaveis do master
  nas condicoes EN e LN (lags 0-52, passo 4); agrupamento k-medias usado apenas
  como sintese espacial descritiva dos resultados pixel-a-pixel, nao como
  modelo preditivo.
- Clusters ranqueados por |r| maximo medio e fracao FDR-significativa ->
  **alvos mais afetados**; por alvo e tipo de sinal: lag mediano, IQR, sentido
  (umido/seco) e forca.
- Sensibilidade temporal **sem breakpoint** da correlacao do alvo no lag
  otimo: bootstrap movel por blocos + leave-one-event-out, com `N_eff`. O
  corte 1993-2009/2010+ foi removido (arbitrario; ver FASE3_WWV_SENSIBILIDADE_TEMPORAL.md).
- Gate estatistico da hipotese (`phase4D_gate_hipotese.csv`): classifica cada
  alvo/variavel/sinal como `suporta_hipotese`, `suporta_com_ressalva`,
  `diagnostico_curto` (lag 0-6) ou `nao_suporta`, sempre com sentido
  seco/umido, lag, `N_eff`, significancia e estabilidade.

## 6. Saidas oficiais

```text
data/processed/parquet/statistics/phase40_inventario_pacifico.csv
data/processed/parquet/statistics/phase40_inventario_alvo.csv
data/processed/figures/fase4/phase40_cobertura_dados.png
data/processed/parquet/statistics/phase4A_eventos_enso.csv
data/processed/parquet/statistics/phase4A_fases_semanais.csv
data/processed/parquet/statistics/phase4A_evento_fases_cronologia.csv
data/processed/parquet/statistics/phase4A_fases_resumo.csv
data/processed/parquet/statistics/phase4A_duracao_fases_por_evento.csv
data/processed/figures/fase4/phase4A_plano_fase.png
data/processed/parquet/statistics/phase4B_determinantes_fases.csv
data/processed/parquet/statistics/phase4B_discriminancia_fases.csv
data/processed/parquet/statistics/phase4B_fase_intensidade_pico.csv
data/processed/parquet/statistics/phase4B_relacoes_pares_fases.csv
data/processed/parquet/statistics/phase4B_matriz_correlacao_por_fase.csv
data/processed/parquet/statistics/phase4B_estrutura_correlacao_distancias.csv
data/processed/parquet/features/phase4_chirps_weekly_zanom.parquet
data/processed/parquet/features/phase4_chirps_pixels.csv
data/processed/zarr/statistics/phase4C_atlas_pixel.zarr
data/processed/parquet/statistics/phase4C_janelas_lag_variavel.csv
data/processed/parquet/statistics/phase4C_best_lag_pixel.csv
data/processed/parquet/statistics/phase4C_lag_resposta_neb_sul.csv
data/processed/parquet/statistics/phase4D_clusters_pixels.csv
data/processed/parquet/statistics/phase4D_cluster_ranking.csv
data/processed/parquet/statistics/phase4D_cluster_lags_por_sinal.csv
data/processed/parquet/statistics/phase4D_estabilidade.csv
data/processed/parquet/statistics/phase4D_gate_hipotese.csv
```

## 7. Criterios de aceite

1. 4A separa as 4 fases com criterio estatistico explicito e simetrico EN/LN,
   com avaliacao expandida (duracoes, dispersao, ONI por fase).
2. 4B responde, com testes nao parametricos e FDR, quais variaveis determinam
   a genese, o crescimento, o pico e o decaimento - para EN e LN.
3. 4C publica o sinal pixel-a-pixel do Brasil inteiro ANTES dos recortes; NEB
   e Sul ganham detalhamento proprio; a pergunta do tempo de resposta e
   respondida em semanas (mediana + IQR) por tipo de sinal.
4. 4D deriva os alvos dos dados, mede estabilidade e preenche o gate
   estatistico da hipotese NEB seco / Sul umido em El Nino.
5. Escopo estritamente Pacifico -> Brasil.
6. Execucao: `python scripts/run_fase4_all.py` (4A -> 4B -> 4C -> 4D).
