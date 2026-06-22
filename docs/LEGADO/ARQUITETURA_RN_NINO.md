# Sistema de Previsão de El Niño e Impactos no Brasil via Deep Learning

**Arquitetura técnica de referência — documento de projeto**

Sistema em cascata de dois modelos para (A) previsão da formação/intensidade do ENSO e (B) downscaling probabilístico dos impactos de precipitação sobre o Brasil, com publicação operacional em GitHub Pages.

---

## 0. Princípios de projeto (leia antes de codar)

Cinco decisões fundamentais que condicionam todo o resto. Violar qualquer uma compromete a validade científica do sistema.

1. **Dados in situ alimentam o treino, não a inferência.** Perfis CTD/Argo têm latência incompatível com inferência em tempo real, mas a estrutura subsuperficial que eles capturam é o melhor discriminador de regime/estágio de evento que existe. Solução: a subsuperfície entra como *target* e *condicionamento de treino* (gabarito histórico), e a inferência operacional usa apenas superfície (satélite) + reanálise rápida, com a subsuperfície *reconstruída* pelo pré-módulo S2Sub (§3.6). Latência só importa do lado da inferência — rótulos de treino são sempre históricos.

2. **Pré-treino em CMIP, fine-tuning em reanálise.** O registro observacional (~75 anos, ~20-25 eventos) é pequeno demais para treinar uma rede profunda do zero sem overfitting. Pré-treine em milhares de anos de simulações CMIP6 e só então faça fine-tuning nos dados observados. Esta é a prática padrão de todos os modelos SOTA (CTEFNet, ResoNet, 3D-Geoformer).

3. **A barreira de previsibilidade da primavera (SPB) é real e deve ser exposta.** Skill de previsões inicializadas em abril-junho degrada fortemente. Toda métrica reportada é estratificada por mês de inicialização. Skill agregado sem essa estratificação é enganoso.

4. **Previsão probabilística, não pontual.** Saídas pontuais sem incerteza são cientificamente indefensáveis em climatologia. Tanto o Modelo A quanto o B emitem distribuições (média + variância, quantis, ou ensemble).

5. **Bater baselines antes de declarar sucesso.** Persistência e climatologia são o chão. Em leads curtos a persistência é surpreendentemente difícil de superar. Se o modelo não bate persistência + climatologia + uma regressão linear, ele não existe.

---

## 1. Visão geral da cascata

```
                          ┌─────────────────────────────────────────┐
                          │  FONTES DE DADOS (gridded, operacionais) │
                          │  superfície: SST · SSHA · u-stress · OLR  │
                          │  reanálise:  OHC/D20 · SLP                │
                          └────────────────────┬────────────────────┘
                                               │
                          ┌────────────────────▼────────────────────┐
                          │  PRÉ-MÓDULO S2Sub (superfície→subsup.)    │
                          │  reconstrói D20/OHC/θ(z) da superfície    │
                          │  treino: gabarito Argo/CTD (histórico)    │
                          │  inferência: só superfície (sem in situ) │
                          └────────────────────┬────────────────────┘
                                               │  subsuperfície estimada
                          ┌────────────────────▼────────────────────┐
                          │  MODELO A — Preditor de estado ENSO       │
                          │  CNN-Transformer híbrido                  │
                          │  in:  superfície + subsup. estimada × T   │
                          │  out: Niño 1+2/3/3.4/4 + campo SST        │
                          │       + tipo EP/CP + fase (form./pico)    │
                          │       leads τ = 1..18 meses (probabilístico)│
                          └────────────────────┬────────────────────┘
                                               │  estado ENSO previsto
                                               │  + tipo EP/CP + incerteza
                          ┌────────────────────▼────────────────────┐
                          │  MODELO B — Tradutor de teleconexão       │
                          │  U-Net / decoder convolucional            │
                          │  in:  estado ENSO(τ) + mês-alvo + EP/CP   │
                          │  out: campo 2D Brasil — P(chuva extrema), │
                          │       P(seca severa), anomalia esperada   │
                          └────────────────────┬────────────────────┘
                                               │
                          ┌────────────────────▼────────────────────┐
                          │  CAMADA DE PUBLICAÇÃO                      │
                          │  GitHub Action (cron mensal) → inferência │
                          │  → mapas Folium/Plotly → GitHub Pages      │
                          └───────────────────────────────────────────┘
```

A separação em dois modelos é deliberada: a física da formação do El Niño no Pacífico é distinta da física da projeção sobre a América do Sul. Acoplar tudo num único end-to-end sacrifica interpretabilidade e robustez, e impede validar cada estágio isoladamente.

---

## 2. Dados

### 2.1 Regiões Niño — papéis diferenciados

Não é "1+2 ou 3.4": use o campo espacial completo de SST e derive **todos** os índices, porque cada um codifica um tipo distinto de evento, e a distinção EP vs CP é justamente o que define o impacto sobre o Brasil.

| Região | Coordenadas | Codifica | Papel |
|--------|-------------|----------|-------|
| Niño 1+2 | 0-10°S, 80-90°W | El Niño Costeiro / EP. Alta variância | Detector precoce; impacto Nordeste/Norte |
| Niño 3 | 5°N-5°S, 150-90°W | Pacífico leste | Transição EP |
| Niño 3.4 | 5°N-5°S, 170-120°W | Índice operacional (ONI) | **Alvo principal do Modelo A** |
| Niño 4 | 5°N-5°S, 160°E-150°W | El Niño Central (CP/Modoki) | Discrimina CP vs EP |

**Decisão de design:** alimente a rede com o *campo 2D de SST* do Pacífico tropical inteiro, não apenas os índices escalares. A rede convolucional aprende os padrões EP/CP sozinha. Os índices escalares entram como features auxiliares e como targets, não como única entrada.

### 2.2 Variáveis de entrada (multivariadas — não faça SST-only)

O ganho de lead time nos modelos SOTA vem do acoplamento oceano-atmosfera. SST sozinho satura cedo.

**Oceânicas:**
| Variável | Fonte | Resolução | Papel físico |
|----------|-------|-----------|--------------|
| SST / SSTA | OISST v2.1 (satélite); ERSST v5 (série longa) | 0.25° / 2° | Estado superficial |
| SSHA (altura da superfície) | Altimetria CMEMS/AVISO | 0.25° | **Proxy de termoclina e conteúdo de calor** — carrega a memória |
| OHC 0-300m / D20 | Reanálise GODAS, ORAS5, SODA | ~1° | Precursor (Warm Water Volume) |

**Atmosféricas (dão o lead longo):**
| Variável | Fonte | Papel físico |
|----------|-------|--------------|
| Tensão de vento zonal (u-stress) | ERA5 | Westerly wind bursts, feedback de Bjerknes |
| SLP | ERA5 | Oscilação Sul (SOI) |
| OLR | NOAA (satélite) | Convecção tropical profunda |
| Vento 850/200 hPa (opcional) | ERA5 | Circulação de Walker |

**Domínio espacial:** mínimo Pacífico tropical (30°S-30°N, 120°E-80°W). Recomendado: cinturão tropical global (inclui Atlântico e Índico) — interações inter-bacias melhoram o skill, como demonstrado no CTEFNet.

### 2.3 Target de impacto (Modelo B)

| Produto | Fonte | Resolução | Observação |
|---------|-------|-----------|------------|
| Precipitação Brasil | CHIRPS v2 | 0.05° | Satélite + estações, excelente cobertura Brasil |
| Alternativa | MERGE/CPTEC | 0.1° | Produto nacional |
| Alternativa global | GPCP | 1° | Para consistência com literatura |

Defina extremos por percentis locais móveis (P90 para chuva extrema, P10 para seca) calculados por pixel e por estação do ano, sobre a climatologia 1991-2020.

### 2.4 Acesso programático (tudo via Python)

```
ERA5            → cdsapi (Copernicus CDS)
OISST / OLR     → xarray + OPeNDAP (NOAA PSL)
ORAS5 / SODA    → Copernicus Marine Toolbox / FTP
CHIRPS          → wget/HTTP (UCSB) ou Google Earth Engine
WOD (validação) → NCEI THREDDS / netCDF
```

### 2.5 Subsuperfície no aprendizado — distinção crítica: input de inferência vs. fonte de treino

**A regra que não muda:** dados subsuperficiais *in situ* (CTD do WOD, perfis de campanha) **nunca** entram como feature de inferência em tempo real, porque têm latência de anos. Um modelo cuja previsão de amanhã depende de um perfil que só existe com 2-3 anos de atraso não roda operacionalmente.

**O que muda em relação à versão anterior deste documento:** "não serve como input operacional" ≠ "não serve à rede". A subsuperfície é, fisicamente, o melhor discriminador de regime e estágio de evento que existe — e isso a torna valiosa no *treinamento*, não na inferência. A assinatura vertical separa o que a SST de superfície não separa:

- profundidade da termoclina / D20 distingue formação genuína de ruído superficial;
- a estrutura 0-300m discrimina El Niño Leste (EP) de Central (CP/Modoki), que podem ter SST de superfície parecida no Niño 3.4 mas subsuperfícies bem distintas;
- o conteúdo de calor da camada superior (Warm Water Volume) é o precursor físico mais robusto de El Niño com 6+ meses de antecedência — ele lidera a SST.

A física do Exercício 1 é o argumento direto: termoclina afundando de ~29 m (Z0 anterior) para ~103 m (pico, 1983) versus subindo para ~10 m na La Niña de 2013. Essa transição vertical é o sinal de formação/pico — e está na subsuperfície, não na superfície.

#### Quatro usos legítimos da subsuperfície no treino

1. **Subsuperfície como TARGET, não input (o mais forte).** Treine o Modelo A (ou um pré-módulo) para *inferir* a estrutura subsuperficial (D20, OHC 0-300m, perfil de θ) a partir de variáveis que existem em tempo real — SSHA, SST, tensão de vento. O dado subsuperficial é o gabarito durante o treino; em produção a rede reconstrói a subsuperfície só com satélite. A latência desaparece porque rótulos de treino são, por definição, históricos. A rede aprende "esta assinatura de superfície ⟹ esta estrutura vertical".

2. **Discriminação de regime e estágio (formação vs. pico — o seu ponto).** Use a subsuperfície para rotular/condicionar o treino quanto a tipo (EP/CP) e fase (formação, pico, relaxamento), melhorando a capacidade do modelo de reconhecer formação real em vez de aquecimento superficial transitório.

3. **Validação física e correção de bias.** Conferir que o OHC/D20 da reanálise reproduz os perfis observados; corrigir bias da reanálise antes de treinar. Aproveita diretamente o pipeline TEOS-10 já construído.

4. **Regularização física.** Penalizar previsões cuja estrutura vertical implícita viole o observado, mesmo sem usar a subsuperfície como entrada (soft constraint na perda).

#### Ressalva de fonte: CTD vs. Argo

Para os usos 1 e 2 você precisa de **densidade subsuperficial em escala de bacia ao longo de muitas décadas** — domínio do **Argo** (~4000 floats, cobertura global desde ~2005, latência de dias). O CTD histórico do WOD é espacialmente esparso e concentrado em campanhas (vide a Figura 2 do Exercício 1, aglomerada no talude Equador/Peru). Logo:

| Fonte | Papel no treino |
|-------|-----------------|
| **OHC/D20 gridded** (GODAS, ORAS5 — assimilam CTD+Argo+XBT) | Target principal e feature de pré-módulo; cobertura completa |
| **Argo** (perfis 2005+) | Gabarito denso para o esquema "superfície→subsuperfície"; rotulagem de regime |
| **CTD histórico (WOD)** | Validação física; caracterização de regime na **caixa Niño 1+2 especificamente**, onde sua densidade é boa; extensão da série pré-Argo |

Em resumo: a subsuperfície é central ao aprendizado (você estava certo); o que se evita é apenas que dados *in situ* de baixa latência sejam exigidos na hora da inferência. A arquitetura que viabiliza isso está em §3.6.

---

## 3. Modelo A — Preditor de estado ENSO

### 3.1 Arquitetura: CNN-Transformer híbrido

Justificativa da escolha sobre alternativas:
- **CNN pura** (Ham et al. 2019): excelente baseline, lead ~17 meses, mas captura mal dependências temporais longas.
- **ConvLSTM:** bom para sequências, mas custoso e propenso a vanishing gradients em horizontes longos.
- **CNN-Transformer híbrido** (ResoNet, CTEFNet, 3D-Geoformer): SOTA atual, lead 18-22 meses. A CNN extrai features espaciais locais (anomalias de SST, estruturas de Kelvin); o Transformer modela dependências temporais e teleconexões inter-bacias via self-attention.

### 3.2 Especificação de tensores

```
Entrada:  X ∈ ℝ[B, T, C, H, W]
  B = batch
  T = janela temporal de entrada (12 meses recomendado)
  C = canais de variáveis (SST, SSHA, u-stress, OLR, SLP, OHC ≈ 6)
  H, W = grade espacial do domínio tropical (ex: 60 × 360 a 1° de resolução)

Encoder espacial (CNN, aplicado por timestep):
  Conv2D blocks → feature maps por mês → flatten/pool → embedding e_t ∈ ℝ[B, T, D]
  D = dimensão do embedding (ex: 256)

Encoder temporal (Transformer):
  positional encoding sobre T
  multi-head self-attention (heads = 8, layers = 4-6)
  → representação contextualizada h ∈ ℝ[B, T, D]

Decoder / cabeças de saída:
  Cabeça 1 (índices):  MLP → ŷ_idx ∈ ℝ[B, L, 4]   (L leads × 4 regiões Niño)
  Cabeça 2 (campo):    decoder conv → SST_pred ∈ ℝ[B, L, H, W]
  Cabeça de incerteza: prevê (μ, log σ²) por saída  → perda gaussiana negativa
```

### 3.3 Função de perda

```
L = L_idx + λ₁·L_campo + λ₂·L_NLL

L_idx   = MSE ou Huber sobre os índices Niño, ponderado por lead
L_campo = MSE sobre o campo SST previsto (regularização espacial)
L_NLL   = negative log-likelihood gaussiana (calibra a incerteza)
```

Pondere os leads: erros em leads longos importam menos (são intrinsecamente menos previsíveis), mas não os zere.

### 3.4 Estratégia de treino

```
Fase 1 — Pré-treino:
  dados: CMIP6 historical + piControl (múltiplos modelos, milhares de anos)
  objetivo: aprender a dinâmica geral do ENSO
  cuidado: remover bias de cada modelo CMIP antes (anomalias relativas à própria climatologia do modelo)

Fase 2 — Fine-tuning:
  dados: reanálise observada (ERA5/ORAS5, 1958-presente)
  learning rate reduzido, early stopping
  congelar parcialmente o encoder se houver overfitting

Fase 3 — Calibração de incerteza:
  ajustar a cabeça probabilística no conjunto de validação
  verificar reliability (a σ prevista corresponde ao erro real?)
```

### 3.5 Validação (inegociável)

- **Walk-forward temporal**, jamais split aleatório (vazaria autocorrelação serial).
- Métricas **estratificadas por mês de inicialização** (expõe a SPB) e por lead time.
- Métricas: correlação de anomalia (ACC) e RMSE por lead; comparação contra persistência, climatologia e regressão linear/CCA.
- **Backtesting em eventos conhecidos:** 1982-83, 1997-98, 2015-16, 2023. O modelo precisa reproduzir a evolução, não só o pico.
- **Interpretabilidade:** saliency/gradient maps confirmando que o modelo usa precursores fisicamente plausíveis (WWV, ondas de Kelvin, wind bursts). Sem isto, é caixa-preta sem credibilidade científica.

### 3.6 Esquema superfície→subsuperfície (como a subsuperfície entra no treino sem contaminar a inferência)

Objetivo: aproveitar o poder discriminativo da estrutura vertical (§2.5) garantindo que a inferência operacional use **apenas** variáveis de baixa latência (satélite + reanálise rápida). Resolve-se separando treino de inferência por arquitetura.

#### Opção recomendada — Pré-módulo de reconstrução subsuperficial

Um módulo S2Sub aprende a reconstruir a subsuperfície a partir da superfície; o Modelo A consome a *reconstrução*, nunca o dado in situ.

```
TREINO (offline, com gabarito histórico):

  Superfície (baixa latência)            Subsuperfície (gabarito, histórica)
  SSHA, SST, u-stress  ──► S2Sub ──► D20̂, OHĈ, θ̂(z)  ◄── target: Argo / OHC gridded / CTD
       [B,T,Cs,H,W]      (encoder-      [B,T,Ksub,H,W]      (só no treino)
                          decoder
                          conv)
  perda L_sub = MSE(reconstrução, gabarito subsuperficial)

INFERÊNCIA (operacional, tempo quase real):

  SSHA, SST, u-stress ──► S2Sub ──► subsuperfície ESTIMADA ──► Modelo A ──► índices Niño + impacto
  (nenhum CTD/Argo do "hoje" é requerido — a subsuperfície é inferida da superfície)
```

Por que funciona: a relação superfície↔subsuperfície no Pacífico equatorial é fisicamente forte (SSHA é proxy quase direto de termoclina). A rede internaliza essa relação no treino; em produção ela "vê" a subsuperfície através da superfície. O Argo/CTD existe só do lado dos rótulos, onde latência é irrelevante.

#### Treino conjunto vs. estagiado

```
Estagiado (mais estável para começar):
  1. treina S2Sub isoladamente até reconstruir bem D20/OHC
  2. congela S2Sub, treina Modelo A usando as features subsuperficiais reconstruídas
  3. fine-tuning conjunto opcional (descongela S2Sub com lr baixo)

Conjunto (end-to-end, maior teto de performance):
  perda total = L_idx + λ₁·L_campo + λ₂·L_NLL + λ₃·L_sub
  L_sub atua como tarefa auxiliar / regularização física
```

#### Condicionamento por regime (uso 2 da §2.5 — formação vs. pico)

Derive da subsuperfície (observada, no treino) um rótulo de regime e fase e use-o como tarefa auxiliar de classificação:

```
rótulos derivados de D20/OHC + índices:
  tipo  ∈ {EP, CP, costeiro, neutro}
  fase  ∈ {formação (Z0−), pico, relaxamento (Z0+)}

cabeça auxiliar: classificação de (tipo, fase)
  → força o encoder a aprender representações que separam
    formação genuína de aquecimento superficial transitório
```

Isso ataca diretamente sua intuição original: a rede aprende a reconhecer *momentos de formação e pico* porque foi treinada com a assinatura vertical que os distingue — exatamente o contraste 1983 (termoclina a 103 m) vs. 2013 (10 m) do Exercício 1.

#### Tensores adicionais

```
Cs    = canais de superfície (SSHA, SST, u-stress) ≈ 3-4
Ksub  = camadas subsuperficiais reconstruídas (D20, OHC, θ em níveis selecionados) ≈ 3-8
```

#### Degradação graciosa

Se o S2Sub não agregar skill mensurável sobre o uso direto de OHC gridded de reanálise (que já assimila Argo/CTD), prefira o caminho mais simples: usar OHC/D20 da reanálise direto como canal de input (já é de latência aceitável). O S2Sub justifica-se quando você quer (a) resolução/qualidade subsuperficial superior à da reanálise, ou (b) o condicionamento por regime acima. Decida por ablação na Fase 3.



### 4.1 O "lag de distribuição"

A teleconexão ENSO→Brasil tem defasagem sazonal característica e espacialmente heterogênea:

| Região | Sinal típico de El Niño | Lag / janela |
|--------|-------------------------|--------------|
| Norte / Nordeste | Redução de chuva, seca | Estação chuvosa MAM; resposta relativamente rápida |
| Nordeste setentrional | Seca severa | MAM do ano do evento |
| Centro-Oeste | Leve aumento de chuva, calor | Primavera |
| Sudeste | Aquecimento; sinal de chuva inconsistente | Variável por evento |
| Sul | Chuva excessiva, cheias | Primavera/início do verão do 1º ano + outono do ano seguinte |

O Modelo B precisa codificar esse lag explicitamente: recebe o mês-alvo como feature e aprende a resposta defasada por região.

### 4.2 Arquitetura

```
Entrada:
  - estado ENSO previsto pelo Modelo A no lead τ
    (índices + campo SST tropical + classificação EP/CP + incerteza)
  - codificação do mês-alvo (sazonalidade)
  - opcional: estado atmosférico regional concorrente (ERA5 sobre Améria do Sul)

Arquitetura: U-Net ou decoder convolucional (downscaling)
  - faz o "downscaling" do sinal tropical de grande escala
    para a grade fina sobre o Brasil
  - skip connections preservam estrutura espacial regional

Saída: campo 2D sobre o Brasil
  - P(precip > P90)  — probabilidade de chuva extrema
  - P(precip < P10)  — probabilidade de seca severa
  - anomalia esperada de precipitação (mm/mês)
  por pixel, para o mês-alvo
```

### 4.3 Perda e validação

```
Perda: combinação de
  - BCE/focal loss sobre as probabilidades de extremo
  - MSE sobre a anomalia esperada
```

Validação com **Brier Skill Score (BSS)**, ROC e reliability diagrams, contra climatologia e persistência. Estratifique por região (Norte, NE, CO, SE, Sul) — o skill será muito heterogêneo, e o Sul/NE provavelmente terão sinal mais forte que o SE.

### 4.4 Propagação de incerteza

A incerteza do Modelo A deve propagar para o B. Duas opções:
- **Ensemble:** amostre múltiplos estados ENSO da distribuição prevista por A, passe cada um por B, agregue.
- **Monte Carlo dropout** no B condicionado à σ de A.

Não colapse a previsão de A para a média antes de passar para B — isso descarta a informação de incerteza, que é metade do valor científico.

---

## 5. Camada de publicação — GitHub Pages

### 5.1 Restrição fundamental

GitHub Pages serve **apenas conteúdo estático**. Não roda Python no servidor. Duas rotas:

**Rota 1 — Estática com Action agendado (RECOMENDADA).**
A inferência roda offline mensalmente (quando o CPC/reanálises atualizam), gera artefatos estáticos, e o site só os serve.

```
.github/workflows/forecast.yml  (cron mensal)
  ├─ baixa dados novos (cdsapi, OPeNDAP)
  ├─ roda inferência (modelos A + B, checkpoints versionados)
  ├─ gera mapas (Folium/Plotly → HTML standalone)
  ├─ gera JSON de probabilidades + séries
  └─ commit & push → Pages republica automaticamente
```

**Rota 2 — Backend leve (se quiser "rodar previsão agora" interativo).**
Inferência em Hugging Face Spaces (Gradio/Streamlit) ou Render; GitHub Pages como front. Mais complexo, raramente necessário para um produto que atualiza mensalmente.

### 5.2 Stack de visualização

| Componente | Ferramenta | Saída |
|------------|-----------|-------|
| Mapas de impacto Brasil | Folium ou Plotly | HTML interativo standalone |
| Choropleth por estado/município | GeoPandas + Leaflet | Camadas |
| Séries dos índices + leque de incerteza | Plotly | HTML |
| Página | index.html simples, ou Quarto/Jupyter Book | Site completo |

### 5.3 Gestão de dados

Não versione NetCDF bruto no Git. Baixe dentro do Action. Versione apenas: checkpoints dos modelos (Git LFS), código, e os artefatos finais leves (HTML, JSON, PNG).

### 5.4 Estrutura de repositório sugerida

```
enso-brasil-forecast/
├── data/                  # .gitignore (baixado no Action)
├── src/
│   ├── data/              # ingestão, QC, montagem do cubo xarray
│   ├── models/
│   │   ├── model_a.py     # CNN-Transformer
│   │   └── model_b.py     # U-Net downscaling
│   ├── train/
│   ├── eval/              # métricas, walk-forward, SPB
│   └── infer/             # pipeline operacional
├── checkpoints/           # Git LFS
├── notebooks/             # exploração, validação física CTD
├── site/                  # GitHub Pages (HTML gerado)
├── .github/workflows/
│   └── forecast.yml       # cron mensal
└── ARQUITETURA.md         # este documento
```

---

## 6. Roteiro de execução faseado

| Fase | Entregável | Critério de sucesso |
|------|-----------|---------------------|
| 0 | Baselines (persistência, climatologia, regressão linear/CCA) | Métricas de referência estabelecidas |
| 1 | Pipeline de dados + cubo xarray + índices Niño | Reprodução exata do ONI oficial do CPC |
| 1b | Pré-módulo S2Sub (superfície→subsuperfície, gabarito Argo/OHC) | Reconstrói D20/OHC com erro < reanálise vs. CTD observado |
| 2 | Modelo A pré-treinado em CMIP6 | ACC competitivo em dados sintéticos |
| 3 | Fine-tuning observacional + validação walk-forward/SPB | Bate baselines; ACC > 0.5 até ~12 meses |
| 4 | Interpretabilidade (saliency) | Precursores fisicamente plausíveis confirmados |
| 5 | Modelo B (downscaling Brasil, CHIRPS) | BSS > 0 vs climatologia, por região |
| 6 | Pipeline operacional + GitHub Pages + Action | Site atualiza sozinho mensalmente |

---

## 7. Riscos e armadilhas conhecidas

| Risco | Mitigação |
|-------|-----------|
| Overfitting (amostra pequena) | Pré-treino CMIP; regularização; early stopping; walk-forward honesto |
| Vazamento temporal na validação | Nunca split aleatório; sempre walk-forward |
| SPB mascarada | Estratificar todas as métricas por mês de inicialização |
| Confundir EP e CP | Usar campo espacial + Niño 4; classificar tipo de evento |
| Dependência de dados in situ na inferência | Subsuperfície só como target/condicionamento de treino (§3.6); inferência usa apenas superfície + reanálise rápida |
| Subsuperfície subutilizada | Pré-módulo S2Sub reconstrói subsuperfície da superfície; condicionamento por regime EP/CP e fase |
| Bias entre modelos CMIP | Anomalias relativas à climatologia de cada modelo |
| Incerteza descartada na cascata | Propagar distribuição de A→B via ensemble |
| Site sem skill real | Sempre exibir métricas de validação e comparação com baselines no próprio site |

---

## 8. Referências-chave (literatura SOTA)

- Ham, Kim & Luo (2019). *Deep learning for multi-year ENSO forecasts.* Nature. — CNN baseline fundacional.
- Lyu et al. (2024). *ResoNet: robust and explainable ENSO forecasts with hybrid convolution and transformer networks.* Adv. Atmos. Sci. — híbrido CNN-Transformer, lead 19 meses.
- Chen et al. (2025). *Towards Long-Range ENSO Prediction with an Explainable Deep Learning Model (CTEFNet).* npj Clim. Atmos. Sci. — multivariado, lead 20 meses, interações inter-bacias.
- 3D-Geoformer / 3D-STransformer (2023-2025). — previsão 3D de oceano superior, lead 18-22 meses.
- ENSO-PhyNet (2024). npj Clim. Atmos. Sci. — Transformer com física de balanço de calor embutida.
- Takahashi & Dewitte (2016). *Strong and moderate nonlinear El Niño regimes.* Clim. Dyn. — distinção EP/CP.
- Bjerknes (1969); Wyrtki (1975) — fundamentos dinâmicos (já no seu Exercício 1).

---

*Documento de arquitetura. Próximo passo de implementação: Fase 0 (baselines) e Fase 1 (pipeline de dados).*
