# NINO-BRASIL - Diretrizes das fases (espinha dorsal)

**Projeto NINO-BRASIL - Oceanografia Fisica UFPE - Thiago Vilar**
**Revisao:** 2026-07-09 - definicao canonica das fases pelo pesquisador.

Este e o documento de referencia (fonte de verdade) para o escopo e as diretrizes
de cada fase. Onde outros documentos divergirem, este prevalece.

## Principios transversais (valem para todas as fases)

1. **Fonte de verdade local:** superficie e a SST/SSTA OISST calculada localmente
   na caixa Nino 3.4; nenhum indice ENSO externo entra como metrica. Graficos
   NOAA/PSL sao apenas comparacao visual.
2. **Rastreabilidade:** nenhuma figura sem tabela/numero anterior rastreavel.
3. **Janela real por fonte:** subsuperficie declara sua janela (GLORYS12 desde
   1993); nada e vendido como cobertura homogenea desde 1981 sem ressalva.
4. **Mensal so calibra:** dados mensais (ORAS5, ONI) servem para conferencia/
   calibracao; nunca para estatistica avancada.
5. **Rigor estatistico:** toda significancia usa graus de liberdade efetivos
   (N_eff de Bretherton) e FDR de Benjamini-Hochberg.
6. **Escopo de bacia:** Pacifico -> Brasil; nenhuma covariavel de outra bacia.

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

## Fase 3 - Diagnostico fisico do sinal Nino 3.4  *(sem ML/RN)*

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
rotulo ENSO externo.

## Fase 4 - Teleconexao ENSO -> chuvas extremas/secas no Brasil  *(sem ML/RN)*

**Diretriz:** avaliar **estatisticamente** a teleconexao **pixel-a-pixel**, medindo
se ha influencia do sinal ENSO e sua **distribuicao espaco-temporal**. Adotar a
**metrica do periodo de aquecimento P90** (picos acima do percentil 90 da SSTA) e
identificar **anomalias de chuva** (extremos/secas) nesses periodos, **incluindo
analise de lags semanais**. Chuva = CHIRPS 0,25 grau, eixo semanal; N_eff + FDR.
Sem ML, sem redes neurais.

## Fase 5 - Mesmo estudo da Fase 4, com Machine Learning  *(nao iniciada)*

**Diretriz:** repetir o estudo comparativo da Fase 4, agora com **Machine
Learning - apenas Random Forest e XGBoost** - aplicando tecnicas de **XAI**
(explicabilidade) e trabalhando em **series semanais e diarias** (diarias quando
possivel). O baseline continua sendo climatologia/persistencia; o ML precisa
superar a triagem estatistica da Fase 4 para se justificar.

## Fase 6 - Redes neurais nativas + XAI  *(nao iniciada)*

Complexidade neural nativa (CNN espacial, memoria espaco-temporal, decoder de
teleconexao) com XAI. So se justifica se vencer climatologia, persistencia, a
Fase 4 e a Fase 5.

## FaseWEB - Publicacao e operacao  *(esqueleto)*

Painel/publicacao web e rotina de recalibracao recorrente, consumindo as saidas
numericas das fases disponiveis.

---

## Gates de decisao

| Gate | Quando | Criterio |
|---|---|---|
| G1 | fim da Fase 4 | teleconexao com associacao significativa por campo (N_eff/FDR), efeito interpretavel, estabilidade temporal e lags defensaveis |
| G2 | fim da Fase 5 | RF/XGBoost com XAI superam a triagem estatistica da Fase 4 e os baselines |
| G3 | fim da Fase 6 | redes neurais superam climatologia, persistencia, Fase 4 e Fase 5 |

A regra de ouro e a mesma em todas: nada avanca sem vencer os baselines simples.
