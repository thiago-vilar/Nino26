# NINO-BRASIL - Diretrizes das fases (espinha dorsal)

**Projeto NINO-BRASIL - Oceanografia Fisica UFPE - Thiago Vilar**
**Revisao:** 2026-07-10 - matriz metodologica 2x3 (estatistica / ML / redes) definida pelo pesquisador.

Este e o documento de referencia (fonte de verdade) para o escopo e as diretrizes
de cada fase. Onde outros documentos divergirem, este prevalece.

## Principios transversais (valem para todas as fases)

1. **Fonte de verdade local:** superficie e a SST/SSTA OISST calculada localmente
   na caixa Nino 3.4; nenhum indice ENSO externo entra como metrica. Graficos
   NOAA/PSL sao apenas comparacao visual.
2. **Rastreabilidade reprodutivel:** nenhuma figura analitica sem tabela/numero
   anterior rastreavel. Alem das tabelas em `data/processed/parquet/statistics/`,
   toda figura em `data/processed/figures/` deve ter uma pasta correspondente em
   `data/processed/numeric-tables/`, gerada por
   `scripts/export_numeric_tables_for_figures.py --force --strict`, contendo os
   CSVs congelados e o manifesto figura -> fonte numerica.
3. **Janela real por fonte:** subsuperficie declara sua janela (GLORYS12 desde
   1993); nada e vendido como cobertura homogenea desde 1981 sem ressalva.
4. **Mensal so calibra:** dados mensais (ORAS5, ONI) servem para conferencia/
   calibracao; nunca para estatistica avancada.
5. **Rigor estatistico:** toda significancia usa graus de liberdade efetivos
   (N_eff de Bretherton) e FDR de Benjamini-Hochberg.
6. **Escopo de bacia:** Pacifico -> Brasil; nenhuma covariavel de outra bacia.
7. **Sem breakpoint arbitrario:** 1993 e fronteira de fonte (inicio do GLORYS12),
   nao regime climatico. Nenhum corte tipo 1993-2009/2010+ entra como filtro; uma
   ruptura estrutural exige estudo formal de ponto de mudanca pre-especificado.

---

## Fase 1 - Base local e ingestao bruta  *(CONCLUIDA)*

Garantir dados suficientes, reprodutiveis e com procedencia para diagnosticar o
Pacifico. Superficie/atmosfera sustentam 1981-presente; a subsuperficie e emendada
de forma auditavel (UFS 1981-92 -> GLORYS12 1993+ -> GLO12 cauda; ORAS5 mensal como
memoria independente). Lacunas in situ (CTD/Argo) ficam registradas, nunca imputadas.

## Fase 2 - Padronizacao, anomalias, Zarr e grade comum  *(CONCLUIDA)*

**Diretriz:** tornar disponivel **todos os dados baixados, de todas as variaveis,
em serie temporal 1981-2026** - principalmente os que serao usados em intervalos
**semanais** - deixando-os prontos para as fases posteriores; e **plotar, via
notebook, todos os graficos de sanidade**.

- Matriz-mestre semanal unificada `nino34_master_weekly.csv` (17 oceanicas
  unificadas UFS/GLORYS/GLO12 + 14 atmosfericas ERA5 para Bjerknes; a coluna
  `ocean_source_code` e metadado de fonte, nao variavel fisica), eixo W-SUN,
  1981-2026, sem semanas duplicadas nem imputacao silenciosa.
- Auditoria (`phase2_master_audit.csv`), validacao de integridade
  (`phase2_master_validation.csv`, tudo True) e validacao in situ CTD/WOD
  (`phase2_ctd_validation.csv`).
- Notebook de sanidade `notebooks/fase2/2Z_sanidade_variaveis.ipynb`: todos os
  graficos de todas as variaveis, grandes, com El Nino (vermelho) e La Nina (azul)
  sombreados.
- Construtor: `scripts/build_master_weekly.py --era5-years 1981:2026`.
- Obrigatorio apos gerar/atualizar figuras: exportar `data/processed/numeric-tables/`
  para permitir auditoria independente de cada PNG.

## Matriz metodologica (duas perguntas, tres ferramentas)

O projeto responde **duas** perguntas cientificas, cada uma com **tres**
ferramentas de rigor crescente. Fases na mesma coluna estudam o mesmo mecanismo;
fases na mesma linha usam a mesma ferramenta.

| Ferramenta | Coluna A: mecanismo do ciclo (Pacifico) | Coluna B: distribuicao no Brasil |
|---|---|---|
| Estatistica | **Fase 3** genese/crescimento/pico/decaimento | **Fase 4** teleconexao pixel-a-pixel |
| Machine Learning (RF/XGBoost) | **Fase 5** | **Fase 6** |
| Redes neurais (ConvLSTM) | **Fase 7** | **Fase 8** |

As Fases 3, 5 e 7 caracterizam o **mesmo** ciclo ENSO com estatistica, ML e redes.
As Fases 4, 6 e 8 medem a **mesma** distribuicao espaco-temporal sobre o Brasil com
estatistica, ML e redes. Nada avanca de linha sem vencer a linha anterior e os
baselines de climatologia/persistencia.

## Fase 3 - Ciclo ENSO com estatistica (mecanismo)  *(sem ML/RN)*

**Diretriz:** caracterizacao fisica auditavel do Pacifico, derivada da propria
OISST local. Especificamente:

1. **Separar eventos** de El Nino e La Nina de 1981 a 2026.
2. **Duracao media de cada tipo** (com enfase em fortes e super-fortes/muito
   fortes), por El Nino e La Nina.
3. **Descrever os eventos** com o instrumental fisico: Hovmoller, feedback de
   Bjerknes, ondas de Kelvin, mapas, ciclos de vida, PCA e PCA/EOF por ciclo de
   vida.
4. **Classificar os 4 periodos** de cada evento: genese, crescimento, pico e
   decaimento - para El Nino E La Nina.
5. **Analisar quais variaveis ajudam a classificar/delimitar os 4 periodos**,
   aplicando analises estatisticas exaustivas (niveis, relacoes entre variaveis,
   volatilidade fina semanal de cada variavel, poder discriminante, etc.).

Escala semanal (W-SUN) sobre a matriz-mestre; sem ML, sem redes neurais, sem
rotulo ENSO externo. `WWV` (volume de agua quente = D20 x area) e um **candidato do
bloco de recarga**, colinear com D20/OHC/SSH/tilt; nao e eixo obrigatorio nem
criterio de aceitacao.

Ao final da Fase 3, `scripts/run_fase3_all.py` deve executar
`scripts/export_numeric_tables_for_figures.py --force --strict`; a fase so e
considerada auditavel quando cada figura tiver sua pasta em
`data/processed/numeric-tables/fase3/`.

## Fase 4 - Distribuicao no Brasil com estatistica  *(sem ML/RN)*

**Diretriz:** avaliar **estatisticamente** a teleconexao **pixel-a-pixel**,
medindo se ha influencia do sinal ENSO e sua **distribuicao espaco-temporal**.
O alvo da Fase 4 nao e chuva bruta: e **anomalia padronizada de chuva por pixel
CHIRPS 0,25 grau**, no eixo semanal. Para cada variavel do master, cada lag
testa `Pacifico(t-L) -> anomalia_chuva_pixel(t)`, com tabela de janelas, N_eff
e FDR. O sinal e medido por **pixel**, por **regiao IBGE** e por **bioma** (com os
recortes Caatinga e Mata Atlantica do Nordeste), nunca so no Brasil agregado.
Sem ML, sem redes neurais. A clusterizacao da Fase 4D e apenas agrupamento
descritivo de pixels com perfis estatisticos semelhantes; nao e modelo preditivo
nem classificador treinado.

Hipotese operacional a testar: durante El Nino, ha aumento de probabilidade/
intensidade de **secas no Nordeste do Brasil** e de **chuvas extremas no Sul do
Brasil**, com lag espacialmente variavel por pixel/regiao/bioma.

Ao final da Fase 4, `scripts/run_fase4_all.py` deve executar
`scripts/export_numeric_tables_for_figures.py --force --strict`; a fase so e
considerada auditavel quando cada figura tiver sua pasta em
`data/processed/numeric-tables/fase4/`.

## Fase 5 - Ciclo ENSO com Machine Learning (RF/XGBoost)  *(nao iniciada)*

**Diretriz:** repetir o estudo do **mecanismo do ciclo** da Fase 3, agora com
**Random Forest e XGBoost** e **XAI**. Transformar a matriz semanal multivariada
(1981-2026) por janela deslizante (lags de 4 a 52 semanas), converter precipitacao
`tp` de m/dia para mm acumulados antes do modelo, e projetar o ciclo completo:
`Y_pico` (intensidade maxima) e `Y_tempo_para_pico` (regressao) e `Y_duracao`
(ONI/OISST >= +0,5 C por >=5 estacoes moveis sobrepostas). Selecao por **RFECV**
(sem importancia a priori), validacao cronologica **exclusiva** (TimeSeriesSplit ou
LOO aninhado - proibido split aleatorio), **augmentation** tabular (jitter gaussiano
e SMOGN para regressao) dado o numero pequeno de eventos, e XAI com **SHAP** (summary,
force, waterfall) e **PDP** dos limiares nao-lineares de Bjerknes. O ML so avanca se
superar a caracterizacao estatistica da Fase 3 e os baselines.

## Fase 6 - Distribuicao no Brasil com Machine Learning (RF/XGBoost)  *(nao iniciada)*

**Diretriz:** repetir o estudo espaco-temporal da Fase 4 (Pacifico -> anomalia de
chuva no Brasil) com **RF/XGBoost** e **XAI**, por fase do ciclo, **regiao IBGE** e
**bioma** (com os recortes Caatinga e Mata Atlantica do Nordeste). Series semanais e
diarias quando possivel. O baseline continua sendo climatologia/persistencia; o ML
precisa superar a triagem estatistica da Fase 4.

## Fase 7 - Ciclo ENSO com redes neurais ConvLSTM  *(nao iniciada)*

**Diretriz:** mesmo mecanismo das Fases 3 e 5, agora com **ConvLSTM**. A rede recebe
sequencias espaco-temporais do Pacifico equatorial (campos regriddados em Zarr),
aprende a evolucao genese->decaimento, identifica ciclos EN/LN, mapeia as 4 fases e
ranqueia variaveis por etapa com XAI. So se justifica se vencer climatologia,
persistencia, a Fase 3 e a Fase 5.

## Fase 8 - Distribuicao no Brasil com redes neurais ConvLSTM  *(nao iniciada)*

**Diretriz:** mesmo estudo espaco-temporal das Fases 4 e 6, agora com **ConvLSTM**.
A rede projeta a influencia do El Nino/La Nina sobre a chuva do Brasil no espaco e no
tempo (encoder Pacifico -> decoder chuva Brasil), avaliada por fase, regiao e bioma.
So se justifica se vencer climatologia, persistencia, a Fase 4 e a Fase 6.

## FaseWEB - Publicacao e operacao  *(esqueleto)*

Painel/publicacao web e rotina de recalibracao recorrente, consumindo as saidas
numericas das fases disponiveis.

---

## Gates de decisao

| Gate | Quando | Criterio |
|---|---|---|
| G1 | fim da Fase 4 | hipotese NEB seco / Sul umido em El Nino sustentada por distribuicao pixel-a-pixel, N_eff/FDR, efeito interpretavel, estabilidade temporal e lags defensaveis |
| G2 | fim da Fase 5 | RF/XGBoost com XAI (SHAP/PDP) caracterizam o ciclo melhor que a Fase 3 e os baselines |
| G3 | fim da Fase 6 | RF/XGBoost com XAI superam a triagem estatistica da Fase 4 e os baselines |
| G4 | fim da Fase 7 | ConvLSTM supera climatologia, persistencia, Fase 3 e Fase 5 no mecanismo do ciclo |
| G5 | fim da Fase 8 | ConvLSTM supera climatologia, persistencia, Fase 4 e Fase 6 na distribuicao no Brasil |

A regra de ouro e a mesma em todas: nada avanca sem vencer os baselines simples.
