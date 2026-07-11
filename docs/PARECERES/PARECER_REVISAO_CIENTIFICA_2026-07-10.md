# Parecer de revisão científica integral - NINO-BRASIL

**Data:** 2026-07-10
**Escopo:** releitura completa do projeto (docs, código, saídas numéricas das Fases 2-6, esqueletos 7/8), com juízo sobre padrões, métricas e coerência científica.
**Método:** leitura da documentação canônica, inspeção do código em `src/` e `notebooks/`, verificação numérica independente das tabelas em `data/processed/parquet/statistics/` e do master semanal.

## 1. Parecer executivo

O projeto está em estado científico bom e acima da média em engenharia de dados e rigor estatístico declarado, mas a execução ultrapassou a governança: as Fases 5 e 6 rodaram hoje (figuras e tabelas de 2026-07-10, 09h41-10h40) enquanto a fonte canônica (`DIRETRIZES_FASES.md`, revisada hoje) e o painel ainda as declaram "não iniciadas", e o gate G1 da Fase 4 não foi formalmente decidido. Além disso, a auditoria independente exigida pelo próprio projeto (`data/processed/numeric-tables/`) está **vazia**, o que, pelo contrato interno, impede declarar as Fases 3 e 4 auditáveis.

No mérito científico, a Fase 3 é madura e defensável. A Fase 4 tem a melhor máquina estatística do projeto, porém dois confundidores comprometem a leitura do gate G1 (seção 4). Na Fase 5 há **um bug objetivo de unidade no alvo de duração** (seção 3), e a Fase 6 produziu saída truncada e sem skill. Nada disso é fatal: as correções são cirúrgicas e o alicerce (dados, auditoria, testes, honestidade nos números negativos) é sólido.

| Bloco | Estado declarado | Estado real observado | Juízo |
|---|---|---|---|
| Fases 1-2 | concluídas | master semanal 2 372 semanas, validações True, cobertura 98,5-99% | coerente |
| Fase 3 | concluída | notebooks 3A-3L + relatório final (2026-07-10 07:23), nested LOO, bootstrap | madura, mas "auditável" só após exportar numeric-tables |
| Fase 4 | em execução | 4A-4D completos, gate tabulado (`phase4D_gate_hipotese.csv`) | boa máquina, leitura do gate comprometida (seção 4) |
| Fase 5 | "não iniciada" | classificador RF/XGB rodado hoje; regressões Y_pico/Y_duração NÃO treinadas | execução prematura + bug no alvo de duração |
| Fase 6 | "não iniciada" | rodada hoje; `phase6_skill_rf.csv` truncado (1 linha, 7 de 10 colunas) | saída corrompida; sem skill onde mensurado |
| Fases 7-8 | não iniciadas | apenas código/esqueleto, sem saídas | coerente |

## 2. O que está bem feito (e deve ser preservado)

**Fase 3.** O relatório final usa nested leave-one-event-out com separação explícita entre seleção (inner) e avaliação (outer), citando Ambroise & McLachlan e Cawley & Talbot; skill vs climatologia honesto (0,319 no 3I; 0,491 no 3K); bootstrap em blocos com envelope IC95 e fração de mesmo sinal; sensibilidade leave-one-event-out por evento com o delta_r máximo reportado; N_eff realista (cai para ~25-30 nas correlações fortemente autocorrelacionadas do 3D, em vez de fingir n=1746). A projeção condicional 2025/26 é rotulada como exploratória, com IC95 largo declarado. Isso é prática acima do padrão usual de projetos acadêmicos.

**Fase 4B (MLR walk-forward).** Validação por evento inteiro com transforms ajustados só no treino, e resultados negativos publicados sem maquiagem (ex.: la_niña crescimento r2_oos = -1,10; gênese de EN e LN sem skill). Reportar fracasso é sinal de integridade e dá credibilidade aos positivos (crescimento EN r=0,76, r2_oos=0,45; pico LN skill_rmse=0,40).

**Fase 4C (pixel-a-pixel).** Correlação defasada vetorizada com N_eff de Bretherton por par, AR1 calculado apenas em semanas consecutivas do mesmo evento, FDR-BH global sobre todos os testes, IC95 por unidade e inventário de janelas de pareamento (`lag_window_inventory`). O achado agregado é coerente com a literatura: SSTA→Sul/Pampa r≈0,16, FDR-significativo em lags curtos; NEB não significativo no agregado anual (esperado: o sinal do NEB concentra-se na estação chuvosa FMAM e dilui em correlação de ano inteiro).

**Consistência entre fases.** Os eventos EN/LN da Fase 3 (`phase3B_eventos_taxas.csv`) e da Fase 4A (`phase4A_eventos_enso.csv`) coincidem em datas, classes e `duracao_estacoes` (ex.: 82/83 = 11 estações, muito forte, ONI local 2,12). O critério local reproduz a cronologia NOAA de forma verossímil (97/98, 15/16, 23/24 etc.).

## 3. Bug objetivo: alvo de duração da Fase 5 inflado ~3x

`src/nino_brasil/models/phase5_cycle_ml.py`, linha 110:

```python
out["Y_duracao_sem"] = pd.to_numeric(out["duracao_estacoes"], errors="coerce") * 13.0
```

`duracao_estacoes` conta **estações móveis de 3 meses que avançam de 1 mês** — ou seja, é, na prática, o número de meses do evento. Multiplicar por 13 trata cada estação como um trimestre disjunto e infla a duração por fator ~3. Evidência em `phase5_alvos_por_evento.csv`:

| Evento | onset → fim | Duração real | Y_duracao_sem gravado |
|---|---|---:|---:|
| el_nino_1997_1998 | 1997-06 → 1998-04 | ~48 sem | 143 sem (2,75 anos) |
| la_nina_1998_2001 | 1998-06 → 2001-03 | ~147 sem | 442 sem (8,5 anos) |
| el_nino_2014_2016 | 2014-10 → 2016-04 | ~82 sem | 247 sem (4,75 anos) |

A prova interna da inconsistência: `Y_tempo_para_pico_sem` é calculado corretamente por diferença de datas na linha 108 (ex.: 26,3 sem para onset jul/82 → pico jan/83). **Correção recomendada:** `Y_duracao_sem = (fim - onset).dt.days / 7` — a coluna `fim` já existe na tabela de eventos. Como `Y_duracao` é alvo declarado da Fase 5, qualquer regressão treinada nele herdaria o erro; felizmente as regressões por evento ainda não foram treinadas (o script só grava os alvos e conta folds LOO), então o custo da correção agora é zero.

## 4. Confundidores que comprometem a leitura do gate G1

### 4.1 Preditores subsuperficiais crus (não anomalizados)

No master semanal, SSTA e as 14 variáveis ERA5 são anomalias, mas as 15 variáveis oceânicas de subsuperfície são valores físicos crus — a própria Fase 3 documenta isso ("valor físico/índice original; não é anomalia climatológica"). Verificação numérica independente do ciclo anual:

| Variável | Amplitude sazonal (clim semanal) | Desvio-padrão total | Razão |
|---|---:|---:|---:|
| d20_m | 24,9 m | 15,1 m | 1,65 |
| t200m | 1,23 °C | 0,59 °C | 2,08 |
| ohc_0_300 | 1,27e9 J | 1,07e9 J | 1,19 |

Ou seja, na escala semanal a variância dominante dessas variáveis é o **ciclo anual**, não o ENSO. Consequências concretas: (a) nas correlações da Fase 4C/4D condicionadas a semanas de evento, o ciclo anual do preditor interage com o phase-locking sazonal do ENSO e da estação chuvosa, produzindo correlação de calendário que se disfarça de teleconexão interanual; (b) as correlações em lags longos (32-36 sem) entre t500m/t700m e chuva podem carregar também a tendência secular comum (aquecimento de fundo), exatamente o confundidor que o parecer técnico de 26/05/2026 mandou tratar com detrend e que segue sem tratamento; (c) na Fase 5, um classificador de fases com rótulos travados no calendário (picos em DJF) pode usar a sazonalidade crua como atalho, inflando o F1 sem aprender mecanismo.

O resultado mais chamativo do gate — NEB em El Niño com sinal "úmido" e `contraria_hipotese` em D20/OHC/t150-t300 no lag 34 — é suspeito exatamente por esse mecanismo. Com lag de 34 semanas dentro de eventos que duram ~12 meses, o par (preditor em t-34, chuva em t) conecta o início do evento à estação chuvosa seguinte do NEB; com preditor cru e sazonal, isso mede em parte o calendário. Antes de aceitar uma conclusão que contraria a literatura estabelecida (El Niño → seca no NEB), é obrigatório refazer 4C/4D com anomalias (climatologia 1991-2020, a mesma regra harmônica já usada no CHIRPS) e detrend linear, e verificar se o sinal sobrevive.

### 4.2 Restrição de amplitude (range restriction) nas correlações condicionadas

Condicionar a semanas de El Niño restringe a variância do `nino34_ssta` (sempre ≳ +0,5 °C dentro do evento), atenuando mecanicamente seu r — daí o índice canônico aparecer com r≈0,01 (SUL) e -0,03 (NEB) enquanto variáveis não restritas pela definição do evento (t200m etc.) aparecem "melhores". Comparar variáveis dentro da condição favorece sistematicamente as que não definem a condição. A comparação honesta entre variáveis deve ser feita na amostra completa (como em `phase4C_lags_por_unidade.csv`, onde SSTA→Sul é significativo), reservando a análise condicionada para perguntas de modulação dentro do evento.

### 4.3 Um único lag por cluster no gate

Em `phase4D_gate_hipotese.csv`, todas as 31 variáveis de cada cluster são avaliadas no lag da "variável dominante do alvo" (12 sem no SUL, 34 no NEB etc.). Avaliar SSTA no lag ótimo de `t300m` é quase um teste de palha: cada variável deveria ser julgada no seu próprio lag ótimo FDR-sobrevivente (a infraestrutura da 4C já produz isso por pixel).

## 5. Fases 5 e 6: estado real das métricas

**Fase 5 (classificador de 4 fases).** F1-macro médio RF 0,477 e XGB 0,512 em TimeSeriesSplit — acima do acaso (0,25), mas sem nenhum baseline registrado. Dado o phase-locking do ENSO, o baseline mínimo é um classificador ingênuo mês-do-ano/semana-do-ano (mais persistência); sem vencê-lo, o G2 não tem como ser avaliado. Registro também que os rótulos de fase são definidos post hoc (o plateau do pico e a gênese de 26 semanas pré-onset só são conhecidos depois do evento), então o F1 mede capacidade de **caracterização**, não previsão — a linguagem das figuras 5A deve dizer isso. As regressões Y_pico/Y_tempo/Y_duração com LOO por evento não foram treinadas; só os alvos foram gravados (com o bug da seção 3).

**Fase 6.** `phase6_skill_rf.csv` está truncado no meio da escrita (uma linha, 7 de 10 colunas) — a execução deve ser refeita por inteiro. A única linha legível (Amazônia, todas as semanas) diz r fora-da-amostra 0,09 e R²_oos -2: o RF é muito pior que a média — sem skill. O baseline "média expansiva" é fraco demais como único comparador; a diretriz promete persistência/climatologia e o código não as implementa. Com ~91 features (7 preditores × 13 lags) e condições de fase com ~120 observações, o desenho atual convida overfitting; se a triagem estatística da Fase 4 mal encontra r=0,16 no agregado, não é surpresa o ML não achar sinal — mas então o gate G3 deve ser decidido com essa honestidade, não com saída corrompida.

## 6. Pendências de governança e auditabilidade

1. `data/processed/numeric-tables/` está vazio. Pela regra transversal nº 2 das diretrizes, nenhuma fase com figuras é auditável nesse estado. Rodar `scripts/export_numeric_tables_for_figures.py --force --strict` e fazer disso pré-condição de qualquer "concluída".
2. `DIRETRIZES_FASES.md` (revisão de hoje) e o painel dizem Fase 5/6 "não iniciada"; o disco mostra execução de hoje. Ou se declara formalmente "piloto exploratório pré-gate" (com as figuras marcadas como tal), ou não se roda antes do G1 decidido. A decisão do G1, com a tabela 4D em mãos, precisa virar documento (aceita, aceita com ressalva, ou reprovada até re-análise da seção 4).
3. A tabela de cobertura do `PARECER_ORGANIZACAO_2026-07-09.md` já está desatualizada (diz ERA5 até 2026-01-04; o master atual vai a 2026-07-05). Documentos com números datados deveriam apontar para o painel gerado, não congelar valores.
4. O painel acusa 69 mudanças locais no Git; o parecer anterior já vetou `git add .`. Convém rodar `git fsck` local (a leitura do repositório nesta sessão acusou um objeto ilegível, possivelmente artefato do mount — verificar na máquina).
5. `pytest` não pôde ser executado nesta revisão (ambiente sem o pacote); a bateria `tests/` deve ser rodada localmente após as correções.

## 7. Recomendações em ordem de prioridade

**P0 — antes de qualquer nova execução de Fase 5/6.**
Corrigir `Y_duracao_sem` por diferença de datas; anomalizar (clim 1991-2020) e detrend as 15 variáveis oceânicas e rerodar 4C/4D (e só então reler o gate G1, em especial o "contraria_hipotese" do NEB); regenerar `numeric-tables/`; refazer a execução da Fase 6 do zero com CSV íntegro.

**P1 — desenho estatístico.**
Estratificar a Fase 4 pela estação chuvosa de cada unidade (FMAM no NEB, primavera no Sul) antes de concluir ausência de sinal; avaliar cada variável no seu lag ótimo no gate, não no lag do cluster; adicionar baselines fortes (semana-do-ano e persistência) às Fases 5 e 6 como condição dos gates G2/G3; reformular a linguagem das figuras 5A/6A para "caracterização diagnóstica" enquanto não houver protocolo prospectivo (embargo temporal, barreira de primavera — já prometidos no relatório da Fase 3).

**P2 — dívidas do parecer de maio/2026 ainda abertas.**
Climatologia rolling (ou restrita ao treino) para as fases de ML; covariável/detrend de tendência global; distinção Canônico × Modoki na teleconexão; posicionamento frente a NMME/SEAS5/CPTEC no texto final.

## 8. Veredito

A fundação do projeto é forte: dados auditáveis, matriz semanal íntegra, Fase 3 com validação acima do padrão e Fase 4 com a estatística mais cuidadosa do repositório. Os problemas encontrados são de três naturezas — um bug de unidade objetivo e barato de corrigir, dois confundidores metodológicos que exigem re-análise antes de aceitar a conclusão mais surpreendente do gate, e um descompasso de governança entre o que os documentos declaram e o que o disco mostra. Corrigidos os itens P0, o projeto volta a ter coerência integral entre diretriz, execução e evidência — e estará em condições de decidir os gates com números em que se pode confiar.
