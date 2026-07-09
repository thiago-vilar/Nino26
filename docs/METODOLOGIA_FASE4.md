# Fase 4 - Metodologia revisada: teleconexao ENSO -> chuva no Brasil

**Projeto NINO-BRASIL - Oceanografia Fisica UFPE - Thiago Vilar**
**Revisao:** 2026-07-08 (expansao por parecer do pesquisador). Escopo
estritamente **Pacifico -> Brasil**.

A fronteira entre fases permanece:

- **Fase 3:** alvo = Nino 3.4. Diagnostica pico, memoria e precursores do ENOS.
- **Fase 4:** alvo = chuva no Brasil. Diagnostica o ciclo ENSO em fases, seus
  determinantes, o tempo de resposta (lags semanais) e os alvos regionais.

## 0. Principios herdados

- Eixo canonico semanal `week_ending_sunday` (W-SUN); CHIRPS diario e somado
  por semana; climatologia harmonica (3 harmonicos anuais) na base 1991-2020,
  nunca 52 medias cruas.
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
4B  Determinantes das fases   -> estudo puramente estatistico: quais variaveis
                                  do Pacifico mais determinam cada fase
4C  Sinal pixel-a-pixel       -> conjunto Pacifico x chuva CHIRPS, lags 0-78,
                                  Brasil inteiro -> recorte NEB -> recorte Sul
4D  Alvos clusterizados       -> SO DEPOIS do pixel: alvos mais afetados, lag
                                  por tipo de sinal, estabilidade e gate G1
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
- Saida central: rotulo semanal `phase4A_fases_semanais.csv`, condicionante de
  4B-4D.

## 3. Fase 4B - Que variaveis determinam cada fase (puramente estatistico)

Tres testes nao parametricos, sem ML, sobre a matriz semanal do Pacifico
(SSTA, D20, OHC 0-300/0-700, SSH, tilt, WWV, tau_x) em z-score:

1. **Separacao fase vs neutro:** Mann-Whitney U por fase x variavel x tipo,
   tamanho de efeito = delta de Cliff, FDR BH por tipo.
2. **Discriminancia entre fases:** Kruskal-Wallis sobre as 4 fases dentro de
   cada tipo, com epsilon-quadrado (quem melhor distingue a fase do ciclo).
3. **Determinacao da intensidade:** Spearman entre o estado medio da variavel
   na genese/crescimento de cada evento e o |ONI| do pico (n pequeno:
   leitura indicativa).

E, alem do nivel isolado, **relacoes entre variaveis** (a pergunta: dentre
todas as duplas, qual *relacao* determina a fase?):

4. **Pares (bivariado):** para cada um dos 28 pares, tres relacoes por semana -
   correlacao movel (13 sem), co-movimento `z_i*z_j` e log da razao de
   volatilidade `sigma_i/sigma_j` - testadas por Kruskal-Wallis/epsilon^2
   entre as 4 fases, com FDR. Ranking em `phase4B_relacoes_pares_fases.csv`.
5. **Estrutura multivariada (complexo):** matriz de correlacao 8x8 do conjunto
   Pacifico por fase e distancia de Frobenius entre fases -> cada fase tem uma
   assinatura de covariacao propria (`phase4B_matriz_correlacao_por_fase.csv`,
   `phase4B_estrutura_correlacao_distancias.csv`).

## 4. Fase 4C - Sinal pixel-a-pixel com lags semanais (Brasil -> NEB -> Sul)

Pergunta-chave: **quanto tempo demora para o sinal do El Nino (e da La Nina)
afetar as chuvas do NEB e do Sul?**

- Chuva: CHIRPS 0,25 no Brasil, soma semanal, anomalia padronizada por pixel
  (climatologia harmonica). Cache em `phase4_chirps_weekly_zanom.parquet`.
- Correlacao de Pearson x(t-L) -> chuva(t) por pixel, variavel e lag (0-78,
  passo 2), em tres condicoes: todas as semanas, semanas El Nino e semanas
  La Nina (fases ativas do 4A). `N_eff` por pixel; FDR por condicao/variavel.
- **Organizacao obrigatoria:** primeiro o Brasil inteiro (atlas completo),
  depois o recorte NEB e o recorte Sul com mapas ampliados, resumo por
  variavel e distribuicao dos lags por regiao.
- Resposta direta em `phase4C_lag_resposta_neb_sul.csv` (lag mediano + IQR por
  regiao, variavel e tipo de sinal).
- Atlas auditavel: `phase4C_atlas_pixel.zarr`.
- Ressalva: CHIRPS pode subestimar extremos convectivos locais.

## 5. Fase 4D - Alvos clusterizados, estabilidade e gate G1

- Vetor de resposta por pixel: perfis r(lag) da SSTA Nino 3.4 nas condicoes EN
  e LN (lags 0-52, passo 4); K-means com k por silhueta.
- Clusters ranqueados por |r| maximo medio e fracao FDR-significativa ->
  **alvos mais afetados**; por alvo e tipo de sinal: lag mediano, IQR, sentido
  (umido/seco) e forca.
- Estabilidade por subperiodo (1993-2009 vs 2010-presente) da correlacao do
  alvo no lag otimo, com `N_eff`.
- Gate para ML (`phase4D_gate_ml.csv`): `preditor` (significativo, estavel,
  lag > 6 sem), `preditor_com_ressalva`, `diagnostico` (lag 0-6) ou `excluido`.

## 6. Saidas oficiais

```text
data/processed/parquet/statistics/phase40_inventario_pacifico.csv
data/processed/parquet/statistics/phase40_inventario_alvo.csv
data/processed/figures/fase4/phase40_cobertura_dados.png
data/processed/parquet/statistics/phase4A_eventos_enso.csv
data/processed/parquet/statistics/phase4A_fases_semanais.csv
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
data/processed/parquet/statistics/phase4C_best_lag_pixel.csv
data/processed/parquet/statistics/phase4C_lag_resposta_neb_sul.csv
data/processed/parquet/statistics/phase4D_clusters_pixels.csv
data/processed/parquet/statistics/phase4D_cluster_ranking.csv
data/processed/parquet/statistics/phase4D_cluster_lags_por_sinal.csv
data/processed/parquet/statistics/phase4D_estabilidade.csv
data/processed/parquet/statistics/phase4D_gate_ml.csv
```

## 7. Criterios de aceite

1. 4A separa as 4 fases com criterio estatistico explicito e simetrico EN/LN,
   com avaliacao expandida (duracoes, dispersao, ONI por fase).
2. 4B responde, com testes nao parametricos e FDR, quais variaveis determinam
   a genese, o crescimento, o pico e o decaimento - para EN e LN.
3. 4C publica o sinal pixel-a-pixel do Brasil inteiro ANTES dos recortes; NEB
   e Sul ganham detalhamento proprio; a pergunta do tempo de resposta e
   respondida em semanas (mediana + IQR) por tipo de sinal.
4. 4D deriva os alvos dos dados, mede estabilidade e preenche o gate G1.
5. Escopo estritamente Pacifico -> Brasil.
6. Execucao: `python scripts/run_fase4_all.py` (4A -> 4B -> 4C -> 4D).
