# PARECER TÉCNICO
## Projeto "Modelo de Previsão Climática ENOS-Brasil com IA, Satélites e CTDs"
### Avaliação crítica do escopo inicial

**Destinatário:** Thiago Vilar (UFPE)
**Data da revisão:** 26 de maio de 2026
**Natureza:** Revisão técnica independente do escopo de projeto

---

## 1. Avaliação geral

O escopo apresentado tem mérito real: a visão arquitetural híbrida (Modelo ENOS + Modelo Brasil) é defensável e melhora interpretabilidade frente a um modelo monolítico end-to-end; a cobertura de variáveis físicas é ampla e bem informada; o stack tecnológico (xarray, zarr, scikit-learn, PyTorch, FastAPI, Streamlit) segue o padrão Pangeo, que é o ecossistema *de facto* para ciência climática reprodutível; e o reconhecimento de que "dado climático bem organizado vale mais do que arquitetura sofisticada com dados inconsistentes" é uma máxima madura, frequentemente ignorada em projetos análogos.

Há, contudo, três problemas estruturais que comprometem a viabilidade científica e o impacto do projeto na forma atual:

1. **O projeto não se posiciona frente ao estado da arte operacional.** NMME, C3S/SEAS5, CPTEC/INPE já produzem previsões sazonais de ENOS e impactos para o Brasil. Sem definir explicitamente o diferencial, há risco real de reinventar o que centros operacionais já entregam com infraestrutura massivamente maior.
2. **Omite limitações fundamentais de previsibilidade** — em particular a *spring predictability barrier* — que precisam estar embutidas no desenho metodológico desde o início.
3. **Trata o Brasil como entidade homogênea**, quando as teleconexões ENOS variam de sinal, intensidade e tipo de evento (Canônico × Modoki) entre regiões brasileiras. É justamente nesse ponto que um modelo de impacto regional pode agregar valor real, e ele não está suficientemente codificado no desenho atual.

O parecer organiza-se em **(I) correções necessárias**, **(II) adições críticas**, **(III) exclusões e simplificações** e **(IV) recomendação estratégica de reorientação**, fechando com **(V) resumo executivo** e **(VI) referências-chave**.

---

## I. CORREÇÕES NECESSÁRIAS

### 1.1 Barreira de Previsibilidade da Primavera Boreal — AUSENTE

A *spring predictability barrier* é o fenômeno mais bem documentado da literatura de previsibilidade do ENOS: previsões inicializadas em março–maio sofrem queda abrupta de habilidade independentemente do tipo de modelo (estatístico, dinâmico, ML). Isso impacta diretamente o "horizonte 1, 3, 6 e 12 meses" descrito no escopo, especialmente quando a inicialização cai nessa janela.

**Correção mínima:** (a) reportar habilidade *condicional ao mês de inicialização*; (b) discutir abordagens de "predictability of opportunity" (Mariotti et al., 2020); (c) considerar arquiteturas que tratam estado de variabilidade lenta (OHC, D20) com peso aumentado nesse período crítico.

### 1.2 Validação temporal — risco real de vazamento de dados

O escopo menciona "dividir treino, validação e teste por tempo", mas três problemas concretos não estão endereçados:

**Climatologia 1991–2020.** Se essa climatologia for usada para gerar anomalias e parte do período (digamos, 2015–2020) também for usada para teste, há vazamento clássico. A solução é usar **climatologia *rolling*** (climatologia móvel que respeita o cutoff temporal de cada amostra) ou definir a climatologia apenas com dados estritamente anteriores ao período de teste.

**Tendência climática.** Anomalias podem conter componente secular (aquecimento global de fundo) que confunde o sinal ENOS. Necessário detrend explícito ou inclusão de tendência (CO2, temperatura global) como covariável.

**Walk-forward validation.** Não basta um split treino/teste. É necessário validação retrospectiva com re-treinamento periódico (anual ou trianual), o que aproxima do uso operacional real e expõe degradação ao longo do tempo.

### 1.3 Baselines insuficientes — provavelmente o item mais crítico

O escopo lista regressão linear, Random Forest, XGBoost e LightGBM como baselines do modelo. Faltam baselines essenciais para legitimar qualquer claim de habilidade preditiva:

- **Persistência** (assumir que a anomalia atual persiste): trivial, mas surpreendentemente difícil de bater em horizontes curtos.
- **Climatologia** (probabilidade igual entre tercis): baseline mínimo.
- **NMME / C3S / SEAS5**: outputs já operacionais de modelos dinâmicos multi-modelo. Sem comparar contra eles, **não é possível afirmar que o modelo proposto agrega valor.**
- **CPTEC/INPE seasonal forecast**: baseline doméstico obrigatório, especialmente em contexto acadêmico brasileiro.

Sem essas baselines, mesmo um modelo com correlação 0,7 pode ser inferior ao que já existe gratuitamente — situação comum em projetos que pulam essa etapa.

### 1.4 Seção "E as ondas de Hertz" — reescrever ou remover

A seção atual mistura ondas eletromagnéticas usadas por sensoriamento remoto, ondas oceânicas e a unidade Hz. Como redação de proposta técnica, prejudica credibilidade frente a avaliadores. **Recomendação:** remover ou substituir por uma seção breve sobre "Considerações sobre uso de canais radiométricos brutos vs produtos derivados", esclarecendo que o MVP usará produtos geofísicos derivados e que assimilação de canais brutos é trabalho futuro.

### 1.5 Sobre-ênfase em CTDs — incoerência com a prática operacional

CTDs (Conductivity-Temperature-Depth) são instrumentos oceanográficos pontuais, caros de operar e com cobertura espaço-temporal esparsa. Para monitoramento contínuo do Pacífico Tropical, os instrumentos *de fato* relevantes são:

- **Rede TAO/TRITON/PIRATA**: boias fixas que medem temperatura subsuperficial, vento, umidade — é a espinha dorsal observacional do ENOS e **não é mencionada no escopo**. PIRATA é a porção atlântica, com forte participação brasileira (INPE/IFREMER) — ausência ainda mais surpreendente.
- **Argo**: ~4.000 perfiladores autônomos com cobertura global, disponibilizados em produtos gridded como **RG-Argo (Scripps)**, **EN4 (Met Office)**, **Ishii (JMA)**.
- **Reanálises oceânicas**: ORAS5, GODAS, SODA já assimilam Argo, TAO e satélites — caminho prático para variáveis subsuperficiais.

**Correção:** despromover CTDs do título e do MVP (mantê-los apenas como fonte auxiliar para validação local em estudos de caso) e dar destaque a **TAO/TRITON/PIRATA + Argo gridded**. O nome poderia ser "ENOS-Brasil com IA, Satélites e Reanálises Oceânicas".

---

## II. ADIÇÕES CRÍTICAS

### 2.1 Diversidade do ENOS: Canônico (EP) vs Modoki (CP)

O escopo menciona Niño 4 como auxiliar para "Modoki", mas não traz esse ponto para o centro do desenho. Esta é uma omissão crítica para o Brasil:

- **El Niño Canônico (Eastern Pacific)**: aquecimento máximo no Niño 1+2 e Niño 3; teleconexão clássica de seca no Nordeste e chuva no Sul.
- **El Niño Modoki (Central Pacific)**: aquecimento máximo no Niño 4 e Pacífico central; teleconexões para o Brasil **diferem em sinal e geografia** — em vários casos com padrões opostos no Sudeste e Norte/Nordeste.

**Adicionar:** (a) cálculo do **EMI (El Niño Modoki Index)** de Ashok et al. (2007); (b) tratamento do tipo de evento como variável categórica explícita ou índice contínuo no Modelo Brasil; (c) validação separada por tipo de evento (EP, CP, misto) — é provável que a habilidade varie dramaticamente entre eles.

### 2.2 Variabilidade modulante: PDO/IPO, AMO, IOD, SAM

A teleconexão ENOS → Brasil é modulada por oscilações de baixa frequência e por modos de outros oceanos. Ausentes do escopo:

- **PDO (Pacific Decadal Oscillation) / IPO (Interdecadal Pacific Oscillation)**: modula amplitude e padrão espacial dos impactos do ENOS em escala decadal.
- **AMO (Atlantic Multidecadal Oscillation)**: modula a resposta do Atlântico Tropical e os impactos no Nordeste.
- **IOD (Indian Ocean Dipole)**: tem teleconexão documentada com a América do Sul via deslocamento da Célula de Walker.
- **SAM (Southern Annular Mode)**: relevante para o Sul do Brasil.

São índices baratos de calcular e podem melhorar habilidade significativamente, especialmente em horizontes mais longos.

### 2.3 Padrões de teleconexão por região brasileira — codificar explicitamente

O escopo trata o Brasil como entidade uniforme. Para que o produto seja útil, é necessário codificar as teleconexões regionais conhecidas:

- **Nordeste (Sertão)**: El Niño + Atlântico Tropical Norte quente → seca severa (deslocamento da ZCIT para norte). La Niña + gradiente Atlântico favorável → chuva acima da média.
- **Sul (RS/SC/PR)**: El Niño → chuva acima da média na primavera e início do verão; La Niña → seca.
- **Sudeste**: sinal ENOS mais fraco e dependente do tipo (EP/CP); domínio da ZCAS, MJO e umidade transportada da Amazônia.
- **Norte/Amazônia**: El Niño → redução de chuva, especialmente Amazônia oriental; risco amplificado de queimadas. Modoki tem assinatura distinta.
- **Centro-Oeste**: regime de monção, modulado por Jato de Baixos Níveis, MJO e ZCAS.

**Adição arquitetural:** desenhar o Modelo Brasil como família de submodelos regionais ou modelo único com região como feature categórica e termos de interação ENOS × região. Avaliar habilidade separadamente por região e por estação climatológica (DJF, MAM, JJA, SON).

### 2.4 NMME, C3S/SEAS5, CPTEC como entradas E baselines

NMME (North American Multi-Model Ensemble) e C3S/SEAS5 (Copernicus) disponibilizam livremente saídas mensais/sazonais de múltiplos modelos dinâmicos. Esses produtos:

- Devem ser **baselines obrigatórias** para validação (sem eles, é impossível defender valor agregado).
- Podem ser **entradas (predictors) do Modelo Brasil**, eliminando ou complementando a necessidade do Modelo ENOS construído do zero (ver Seção IV).

Ignorar essas fontes equivale a ignorar o estado da arte operacional.

### 2.5 Datasets brasileiros essenciais — ausentes do escopo

Faltam fontes nacionais críticas:

- **Xavier et al. (2016, 2022)**: precipitação diária gridded 0.25° para o Brasil, baseada em estações INMET/ANA — referência para validação no território nacional.
- **ANA (Agência Nacional de Águas)**: vazões de estações fluviométricas — alvo natural para previsão hidrológica.
- **ONS / SIN**: nível de reservatórios do sistema interligado — dado de altíssimo valor operacional (energia).
- **FUNCEME**: monitoramento e previsão climática do Nordeste, com séries longas e expertise regional.
- **INMET**: estações automáticas com dados horários.
- **CEMADEN**: alertas e dados hidrometeorológicos com foco em desastres.

Esses datasets, além de melhorarem o modelo, abrem possibilidade de parcerias institucionais que validam o produto perante usuários reais.

### 2.6 Métricas probabilísticas operacionais

Faltam as métricas que a comunidade operacional efetivamente usa:

- **RPSS (Ranked Probability Skill Score)**: padrão para previsões em tercis (abaixo/normal/acima), que é o formato dos boletins IRI, CPTEC e WMO.
- **Heidke Skill Score** e **Gerrity Skill Score**: para previsões categóricas multi-classe.
- **Reliability diagrams**, **sharpness**, **resolução**: diagnóstico de calibração probabilística (decomposição de Murphy).
- **Forecast verification report cards** no estilo IRI: permitem ao usuário ver habilidade histórica antes de confiar em uma previsão específica.

ROC-AUC, Brier e CRPS estão listados, mas sem RPSS o produto não é comparável aos boletins operacionais.

### 2.7 Bias correction e downscaling estatístico

Não mencionados, mas essenciais para traduzir saídas de modelos globais para escala regional brasileira:

- **BCSD (Bias-Corrected Spatial Disaggregation)**
- **EQM (Empirical Quantile Mapping)** e **QDM (Quantile Delta Mapping)** (este último preserva tendências, importante)
- **Análogos** e **CCA (Canonical Correlation Analysis)** para downscaling estatístico clássico

Estas técnicas são, possivelmente, **onde está o maior valor agregado** do projeto.

### 2.8 Posicionamento institucional explícito

CPTEC/INPE, FUNCEME, INMET e centros universitários (USP, UFCG, INPA, UFPE) já produzem previsões sazonais. O projeto precisa responder: *qual é o diferencial?* Possibilidades realistas:

- **Comunicação de incerteza para usuários setoriais** (energia, agricultura, defesa civil) — gap real no Brasil.
- **Downscaling para bacias hidrográficas específicas** (São Francisco, Tocantins-Araguaia, Paraná).
- **Fusão de múltiplas fontes** (NMME + CPTEC + ML local).
- **Previsão sub-sazonal (S2S)** para a janela 2–6 semanas, onde há gap operacional reconhecido pelo programa WCRP/WWRP S2S.

### 2.9 Aspectos operacionais frequentemente esquecidos

- **Latência de dados**: ERA5 tem ~5 dias de atraso; reanálises oceânicas, semanas a meses; Argo, ~10 dias; produtos satelitais near-real-time vs delayed-mode. O pipeline real precisa lidar com janelas de disponibilidade diferentes e versionamento.
- **Monitoramento de drift**: modelo treinado em 1980–2010 pode degradar pós-2020.
- **Retreinamento periódico**: política explícita (anual? trianual?).
- **Reprodutibilidade**: containerização (Docker), DVC para versionamento de dados, MLflow para experimentos.
- **Comunicação de incerteza**: probabilidades são notoriamente mal interpretadas; UX da página web deve refletir isso.

---

## III. EXCLUSÕES E SIMPLIFICAÇÕES PARA O MVP

### 3.1 Remover do MVP (mover para "trabalho futuro")

**Canais radiométricos brutos (TB 6/10/18/23/36/89 GHz).** Complexidade massiva, ganho marginal no horizonte sazonal. Assimilação direta de TBs é tese de doutorado por si só. Use produtos derivados.

**Graph Neural Networks para municípios/bacias.** Arquitetura promissora, mas requer infraestrutura de dados muito madura e validação cuidadosa. Não para MVP.

**Spatial-Temporal Transformer.** Deixar para fase 2 após validar que ConvLSTM/U-Net têm habilidade competitiva. Transformers em climatologia tendem a *underperform* sem dados suficientes; cuidado com overfitting.

**React/Next.js para frontend.** Prematuro. Streamlit, Panel ou Dash são suficientes para validar produto antes de investir em frontend operacional. Adicionar complexidade web cedo desvia foco do que importa (skill da previsão).

### 3.2 Variáveis que podem ser despromovidas no MVP

- **Salinidade vertical detalhada**: contribuição secundária para skill de ENOS comparado a temperatura/SSHA/D20.
- **Oxigênio dissolvido, fluorescência/clorofila de CTDs**: bioquímico, fora do escopo de ENOS *sensu stricto*.
- **Correntes ADCP de CTDs**: cobertura pontual demais para uso em modelo gridded.
- **Distinção fina entre Niño 3 e Niño 3.4 no MVP**: use Niño 3.4 como índice principal e EMI como secundário; Niño 3 puro é redundante.

### 3.3 Endpoints de API podem ser unificados

Para MVP basta:
- `GET /forecast/{region_type}/{region_id}?horizon=N&variable=X`
- `GET /enso/status`
- `GET /indices/{name}`

A explosão de endpoints individuais (`/risk/drought`, `/maps/precipitation-anomaly`, etc.) pode vir com a maturidade do produto.

---

## IV. RECOMENDAÇÃO ESTRATÉGICA DE REORIENTAÇÃO

A recomendação mais importante deste parecer não é técnica, é de **escopo estratégico**.

### Proposta: arquitetura "fusão + downscaling", não "ENOS do zero"

O desenho atual (Modelo 1 prevê ENOS, Modelo 2 traduz para Brasil) é defensável, mas implica competir com centros operacionais (NCEP, ECMWF, BoM, CPTEC) que têm décadas de investimento em modelos dinâmicos acoplados oceano-atmosfera. Mesmo com IA moderna, **bater o NMME consistentemente é meta de pesquisa de fronteira** — vide os esforços recentes do Google/DeepMind (GraphCast, NeuralGCM) e Nvidia (FourCastNet), que ainda são objeto de debate.

Uma alternativa pragmática, com chance maior de gerar valor mensurável e publicação relevante:

**Arquitetura proposta:**

```
Entradas:
  - NMME + C3S/SEAS5 + CPTEC seasonal forecast (saídas dinâmicas)
  - ERA5 (estado atmosférico recente)
  - Argo/ORAS5/GODAS (estado oceânico recente)
  - Índices climáticos: Niño 3.4, EMI, IOD, AMO, PDO, SAM, NAO
  - Dados regionais Brasil: Xavier, ONS, ANA, FUNCEME, INMET
                    │
                    ▼
  MODELO DE FUSÃO + DOWNSCALING + CALIBRAÇÃO
  (XGBoost por região × estação no MVP;
   U-Net / ConvLSTM em fases posteriores)

  Realiza simultaneamente:
    (a) bias correction das saídas dinâmicas
    (b) calibração probabilística (sharpness + reliability)
    (c) downscaling estatístico para grid/região brasileira
    (d) fusão multi-modelo ponderada por skill regional
                    │
                    ▼
  PRODUTOS:
    - Probabilidades por tercil (abaixo/normal/acima)
    - Intervalos de confiança calibrados
    - Baselines reportadas lado a lado
    - Skill histórico transparente por região × estação × horizonte
```

**Vantagens:**

- **Valor agregado claro**: nenhum centro internacional faz downscaling/calibração específica para regiões brasileiras com a profundidade que um grupo nacional pode fazer.
- **Viabilidade computacional**: dispensa treinar modelos globais de ENOS do zero (que exigem grids de PB de dados e dezenas a centenas de GPUs).
- **Publicação**: papers de calibração/downscaling regional são publicáveis e citáveis (J. of Climate, Climate Dynamics, IJOC, Weather and Forecasting).
- **Comparabilidade**: facilita comparação direta contra NMME/C3S — claim de valor agregado fica defensável.
- **Parcerias institucionais**: alinhamento natural com ONS, ANA, FUNCEME, CPTEC.

O **Modelo 1 (ENOS do zero)** pode continuar como **linha de pesquisa paralela** — talvez como ConvLSTM/Transformer experimental para entender ganhos de IA sobre dinâmico — mas sem ser o gargalo do MVP.

### Próximos passos sugeridos (revisão das Etapas 1–14)

1. **(Semanas 1–4) Definir produto e público-alvo.** Setorial (ONS para energia? CEMADEN para desastres? agricultura?) ou genérico? Esta decisão determina métricas, regiões e horizontes prioritários.
2. **(Mês 2) Pipeline mínimo.** Baixar e regridar ERA5, OISST, CHIRPS, Xavier, NMME hindcast para um período de validação (sugestão: 1991–2024).
3. **(Mês 3) Baselines obrigatórias.** Persistência, climatologia, NMME cru, NMME bias-corrected, CPTEC. Calcular RPSS, Brier, reliability diagrams para tercis de precipitação por região e estação.
4. **(Meses 4–6) Modelo Fusão MVP.** XGBoost por região × estação, com features tabulares (índices climáticos + saídas NMME bias-corrected).
5. **(Mês 7) Dashboard Streamlit** mostrando mapas tercil-probabilidade, comparação com baselines, gráficos de skill histórico transparente.
6. **(Meses 8–12) Refinamentos.** Adicionar ConvLSTM para padrões espaciais; testar climatologias rolling; análise por tipo de evento (EP/CP); incluir IOD/AMO/SAM.
7. **(Ano 2+) Modelo ENOS de pesquisa** com ConvLSTM/Transformer, comparando contra NMME — agora como linha de pesquisa, não bloqueio de produto.

---

## V. RESUMO EXECUTIVO

| Categoria | Item | Prioridade |
|---|---|---|
| **Corrigir** | Endereçar *spring predictability barrier* explicitamente | Alta |
| **Corrigir** | Climatologia rolling para evitar vazamento de dados | Alta |
| **Corrigir** | Adicionar persistência, climatologia, NMME, CPTEC como baselines | Crítica |
| **Corrigir** | Walk-forward validation com re-treinamento periódico | Alta |
| **Corrigir** | Detrend ou inclusão de tendência climática como covariável | Alta |
| **Corrigir** | Despromover CTDs; promover TAO/TRITON/PIRATA + Argo gridded | Média |
| **Corrigir** | Reescrever ou remover a seção "ondas de Hertz" | Média |
| **Adicionar** | Diversidade EP/CP (Canônico/Modoki) com EMI | Alta |
| **Adicionar** | IOD, PDO/IPO, AMO, SAM como covariáveis | Alta |
| **Adicionar** | Teleconexões regionais codificadas (NE/S/SE/N/CO) | Alta |
| **Adicionar** | Xavier, ANA, ONS, FUNCEME, CEMADEN, INMET | Alta |
| **Adicionar** | NMME/C3S como entrada E baseline | Crítica |
| **Adicionar** | RPSS, Heidke, Gerrity, reliability diagrams | Alta |
| **Adicionar** | Bias correction (EQM, QDM, BCSD) e downscaling estatístico | Alta |
| **Adicionar** | Posicionamento institucional explícito vs CPTEC/INPE | Crítica |
| **Adicionar** | Latência, drift, retreinamento, containerização | Média |
| **Excluir do MVP** | Canais radiométricos brutos (TB GHz) | Alta |
| **Excluir do MVP** | GNN, Transformer e React/Next.js | Média |
| **Excluir do MVP** | Salinidade vertical detalhada, oxigênio, fluorescência | Baixa |
| **Reorientar** | Arquitetura "fusão + downscaling" no lugar de "ENOS do zero" no MVP | Crítica |

---

## VI. REFERÊNCIAS-CHAVE PARA APROFUNDAR

- Ashok, K. et al. (2007). "El Niño Modoki and its possible teleconnection." *J. Geophys. Res. Oceans*.
- Cai, W. et al. (2020). "Climate impacts of the El Niño–Southern Oscillation on South America." *Nature Reviews Earth & Environment*.
- Goddard, L. et al. (2013). "A verification framework for interannual-to-decadal predictions experiments." *Climate Dynamics*.
- Kirtman, B. P. et al. (2014). "The North American Multimodel Ensemble." *BAMS*.
- L'Heureux, M. L. et al. (2017). "Observing and predicting the 2015/16 El Niño." *BAMS*.
- Mariotti, A. et al. (2020). "Windows of opportunity for skillful forecasts subseasonal to seasonal and beyond." *BAMS*.
- Xavier, A. C.; King, C. W.; Scanlon, B. R. (2016). "Daily gridded meteorological variables in Brazil (1980–2013)." *Int. J. Climatol.*
- Grimm, A. M. (2003). "The El Niño impact on the summer monsoon in Brazil: regional processes versus remote influences." *J. Climate*.
- Coelho, C. A. S. et al. (2012). "Climate diagnostics of three major drought events in the Amazon and illustrations of their seasonal precipitation predictions." *Meteorol. Appl.*

---

*Fim do parecer.*
