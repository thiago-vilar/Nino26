# NINO-BRASIL - Diretrizes das fases (espinha dorsal)

**Projeto NINO-BRASIL - Oceanografia Fisica UFPE - Thiago Vilar**
**Revisao:** 2026-07-13 - contratos científicos, execução e gates F1-F8 auditados.

Este e o documento de referencia (fonte de verdade) para o escopo e as diretrizes
de cada fase. Onde outros documentos divergirem, este prevalece.

## Principios transversais (valem para todas as fases)

1. **Fonte de verdade local:** superficie e a SST/SSTA OISST calculada localmente
   na caixa Nino 3.4; nenhum indice ENSO externo entra como metrica. Graficos
   NOAA/PSL sao apenas comparacao visual.
2. **Rastreabilidade reprodutivel:** nenhuma figura analitica sem tabela/numero
   anterior rastreavel. Alem das tabelas em `data/processed/parquet/statistics/`,
   toda figura em `data/processed/figures/` deve ter uma pasta correspondente em
   `data/processed/numeric-tables/`, co-gerada a partir das tabelas semânticas e
   validada por `scripts/validar_figuras.py --strict`. O exportador legado não
   é fonte de verdade e não pode apagar a árvore de tabelas.
3. **Janela real por fonte:** subsuperficie declara sua janela (GLORYS12 desde
   1993); nada e vendido como cobertura homogenea desde 1981 sem ressalva.
4. **Mensal so calibra:** dados mensais (ORAS5, ONI) servem para conferencia/
   calibracao; nunca para estatistica avancada.
5. **Rigor estatistico:** toda significancia usa graus de liberdade efetivos
   (N_eff de Bretherton) e FDR confirmatório de Benjamini-Hochberg com
   `alfa=0,05` lido de `configs/project.yaml`.
6. **Escopo de bacia:** Pacifico -> Brasil; nenhuma covariavel de outra bacia.
7. **Sem breakpoint arbitrario:** 1993 e fronteira de fonte (inicio do GLORYS12),
   nao regime climatico. Nenhum corte tipo 1993-2009/2010+ entra como filtro; uma
   ruptura estrutural exige estudo formal de ponto de mudanca pre-especificado.
8. **Unidade independente:** o evento ENSO completo. Semanas, janelas e dados
   aumentados são dependentes e nunca atravessam treino/teste.
9. **Alvo nativo:** F4/F6/F8 mantêm grid, resolução, coordenadas e `pixel_id`
   CHIRPS originais. Região/bioma são apenas agregações posteriores reversíveis.

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
- Notebook de sanidade `notebooks/fase2/F2Z_sanidade_variaveis.ipynb`: todos os
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
estatistica, ML e redes como comparacao historica, sem impor ordem de execucao.

As Fases 3, 4, 5 e 6 ficam fora do escopo oficial corrente e permanecem apenas
como referencia historica opcional. Novas analises e modelos podem consumir
diretamente os dados das Fases 1 e 2, sem aguardar gates intermediarios.

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

Ao final da Fase 3, o verificador de tabelas semânticas e a compatibilidade
`scripts/validar_figuras.py --strict --allow-render-extraction` devem passar. A
promoção científica exige adicionalmente `scripts/validar_figuras.py --strict`;
enquanto a migração estiver incompleta, esse gate permanece pendente de forma
explícita. Cada tabela promovida declara `run_id`, SHA-256, unidade, parâmetros,
modo diagnóstico/preditivo e família FDR.

## Fase 4 - Distribuicao no Brasil com estatistica  *(sem ML/RN)*

**Diretriz:** avaliar **estatisticamente** a teleconexao **pixel-a-pixel**,
medindo se ha influencia do sinal ENSO e sua **distribuicao espaco-temporal**.
O contrato preserva a chuva bruta e deriva **anomalia/percentil/SPI/extremos por
pixel CHIRPS nativo**, no eixo semanal. Para cada variavel do master, cada lag
testa `Pacifico(t-L) -> anomalia_chuva_pixel(t)`, com tabela de janelas, N_eff
e FDR. O sinal e medido por **pixel**, por **regiao IBGE** e por **bioma** (com os
recortes Caatinga e Mata Atlantica do Nordeste), nunca so no Brasil agregado.
Sem ML, sem redes neurais. A clusterizacao da Fase 4D e apenas agrupamento
descritivo de pixels com perfis estatisticos semelhantes; nao e modelo preditivo
nem classificador treinado.

Hipotese operacional a testar: durante El Nino, ha aumento de probabilidade/
intensidade de **secas no Nordeste do Brasil** e de **chuvas extremas no Sul do
Brasil**, com lag espacialmente variavel por pixel/regiao/bioma.

O builder concatena os dias de todos os anos antes de um único resample W-SUN,
registra dias válidos e produz SPI 1/3/6 meses, Rx1day/Rx5day, R95p/R99p,
CDD/CWD. Fase/filtro são avaliados em `t-lag`, com significância de campo.
No contrato CHIRPS v4, a escala de R95p/R99p é condicional às semanas
positivas (`N+>=20`) e usa fallback físico auditado quando o suporte é menor.
Milímetros, coordenadas, resolução, `pixel_id` e máscaras nativas não são
recortados, interpolados nem substituídos.

## Fase 5 - Ciclo ENSO com Machine Learning (RF/XGBoost)  *(implementada)*

**Diretriz:** repetir o estudo do **mecanismo do ciclo** da Fase 3, agora com
**Random Forest e XGBoost** e **XAI**. Transformar a matriz semanal multivariada
(1981-2026) por janela deslizante (lags de 4 a 52 semanas), converter precipitacao
`tp` de m/dia para mm acumulados antes do modelo, e projetar o ciclo completo:
`Y_pico` (intensidade máxima), `Y_tempo_para_pico` e `Y_duracao`, derivados dos
eventos OISST locais. As 31 variáveis permanecem candidatas no contrato oficial;
não há RFECV promovível. O modelo usa nove estados e validação rolling-origin
por evento inteiro com embargo. Pesos equilibram tipo/evento/fase. Augmentation
conservador é exclusivamente de treino, preserva `original_event_id` e nunca
aumenta o N inferencial. Cada fold oficial exige suporte mínimo separado de
eventos El Nino e La Nina. Importâncias por EN/LN e fase são calculadas por
permutação fora da amostra; SHAP/PDP não são requisitos implementados. O ML só
avança se superar os baselines predeclarados e os gates das dimensões de evento.

## Fase 6 - Distribuicao no Brasil com Machine Learning (RF/XGBoost)  *(implementada; execução por shards)*

**Diretriz:** repetir o estudo espaco-temporal da Fase 4 (Pacifico -> anomalia de
chuva no Brasil) com **RF/XGBoost** e XAI para cada pixel CHIRPS original, usando
31 variáveis, lag e fase em `t-lag`. A transformação da chuva é ajustada no
treino. Shards só liberam o gate após cobertura integral, sem sobreposição e com
skill area-ponderado positivo contra o melhor baseline. Região/bioma vêm depois.

## Fase 7 - Ciclo ENSO com redes neurais ConvLSTM  *(implementada em PyTorch)*

**Diretriz:** mesmo mecanismo das Fases 3 e 5, agora com **ConvLSTM**. A rede recebe
sequências espaço-temporais GLORYS reais e funde a sequência nomeada das 31
variáveis. Prediz estado/tipo/fase e distribuições de pico/tempo/duração. Folds
são por evento com embargo; masking/noise são somente de treino. Existe helper
experimental para mascaramento, mas o pré-treino auto-supervisionado ainda não
está ligado ao runner oficial e não pode ser alegado como resultado. O gate
exige F5 pareado, skill das três dimensões e calibração dos IC90, além da
classificação. Só se justifica se vencer F5 e os baselines.

## Fase 8 - Distribuicao no Brasil com redes neurais ConvLSTM  *(implementada em PyTorch)*

**Diretriz:** mesmo estudo espaco-temporal das Fases 4 e 6, agora com **ConvLSTM**.
A rede projeta a influencia do El Nino/La Nina sobre a chuva do Brasil no espaco e no
tempo por decoder convolucional probabilístico. A saída mantém exatamente a
forma e os `pixel_id` CHIRPS; máscara/fração Brasil e pesos de área afetam apenas
a loss. Campos OOS e métricas por pixel são auditáveis. Só se justifica se
vencer persistência, F4 e F6. O gate probabilístico usa IC90 central e exige
erro absoluto de calibração `<= 0,15` em cada condição e em cada fold-condição.
A cobertura é area-ponderada dentro do evento e event-equal na agregação; pixels
e semanas não são tratados como eventos independentes.

## FaseWEB - Publicacao e operacao  *(esqueleto)*

Painel/publicacao web e rotina de recalibracao recorrente, consumindo as saidas
numericas das fases disponiveis.

---

## Gates de decisao

| Gate | Quando | Criterio |
|---|---|---|
| G1 | fim da Fase 4 | hipotese NEB seco / Sul umido em El Nino sustentada por distribuicao pixel-a-pixel, N_eff/FDR, efeito interpretavel, estabilidade temporal e lags defensaveis |
| G2 | fim da Fase 5 | RF/XGBoost com importância OOS superam os baselines de classificação e dimensões, com suporte independente EN/LN suficiente |
| G3 | fim da Fase 6 | RF/XGBoost com importância OOS superam a Ridge estatística F4 e os demais baselines no grid nativo |
| G4 | fim da Fase 7 | ConvLSTM supera baselines e Fase 5 pareada, dimensiona pico/tempo/duração e calibra IC90 |
| G5 | fim da Fase 8 | ConvLSTM supera climatologia, persistencia, Fase 4 e Fase 6 na distribuicao no Brasil e calibra IC90 por condição/fold com peso igual por evento |

Os gates desta tabela sao historicos e nao bloqueantes. Nenhuma tecnologia
futura exige a conclusao ou aprovacao das Fases 3, 4, 5 ou 6; F1 e F2 podem
alimentar diretamente qualquer modelo analitico ou preditivo.

A regra de ouro e a mesma em todas: nada avanca sem vencer os baselines simples.
