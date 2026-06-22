# ESCOPO TÉCNICO — PROJETO NINO
## Sistema de Previsão Pixel-a-Pixel dos Impactos do El Niño no Brasil
### Arquitetura de dados, redes neurais, XAI, análise espacial e análise de impacto

**Responsável técnico:** Thiago Vilar (UFPE)
**Versão:** 1.0 — revisão de escopo
**Data:** 26/05/2026

---

## 1. Visão e objetivos

Construir um sistema de previsão dos **impactos do El Niño/Oscilação Sul (ENOS) sobre o Brasil** com:

1. **Distribuição espaço-temporal pixel-a-pixel** em grade fina (alvo: 0,05° ≈ 5 km, passo temporal diário ou pêntada).
2. **Foco prioritário em dois alvos físicos**: (a) seca no semiárido nordestino, dimensionada por SPI/SPEI multi-escala, anomalia de chuva, resposta vegetal e umidade do solo; (b) chuva extrema no Sul (RS, SC, PR), dimensionada por probabilidade de exceder limiares P90/P95/P99.
3. **Modelagem explícita de ondas de Kelvin** (oceânicas equatoriais, costeiras e atmosféricas convectivamente acopladas) como precursores dinâmicos do ENOS.
4. **Pipeline diário**, sem dependência de agregação mensal como entrada do modelo (agregações mensais aparecem apenas como alvos derivados quando o índice exige — por exemplo, SPI-3).
5. **Métricas operacionais minimizadas até o limite físico**: viés próximo de zero, RMSE próximo do ruído observacional, RPSS positivo em janela útil, distribuições probabilísticas calibradas (ver Seção 9 e 10).
6. **Interpretabilidade**: cada previsão acompanhada de atribuição causal/física, mapa de confiança e identificação dos precursores que dominaram a previsão.

O sistema é desenhado para ser **enxuto, modular e ágil**: cada módulo treinável, testável e substituível sem retreinar o sistema inteiro.

---

## 2. Princípios de desenho

- **Frequência nativa diária.** Toda a pipeline opera em grade temporal diária. Índices climáticos são calculados em janelas móveis diárias, não em médias mensais agregadas a priori.
- **Multi-resolução espacial sem reamostragem agressiva.** Cada variável é mantida na sua resolução nativa até o ponto de fusão pelo modelo, que opera com features pyramidais.
- **Física informada onde possível, dados onde não.** Operadores físicos conhecidos (decomposição modal de ondas, equação de Bjerknes, recharge-discharge oscillator) são embutidos como camadas ou perdas auxiliares; resíduos são aprendidos pela rede.
- **Probabilístico nativo.** Saída do modelo é distribuição, não ponto. CRPS, Brier e RPSS são métricas primárias; RMSE/R² são secundárias e reportados com banda de incerteza.
- **Sem promessas que violem a física.** RMSE literalmente zero é impossível em sistema caótico (ver Seção 10). O alvo é "RMSE no piso do ruído observacional" para nowcasting e "skill probabilístico positivo" para horizontes sazonais.
- **Reprodutibilidade**: tudo versionado (código com Git, dados com DVC ou intake-catalog, experimentos com MLflow ou Weights & Biases, ambientes com Conda+Mamba e Docker).

---

## 3. Catálogo completo de variáveis

Organizado por subsistema físico. Cada variável traz: **nome — o que mede — por que importa — fonte recomendada — resolução nativa**.

### 3.1 Variáveis oceânicas do Pacífico Tropical (núcleo ENOS)

- **SST** — temperatura da superfície do mar — base de todos os índices ENOS — OISST v2.1 (NOAA) — 0,25°/diário.
- **SSTA** — anomalia de SST — núcleo dos índices Niño — derivada com climatologia rolling 30 anos.
- **SSHA** — anomalia da altura do mar — proxy da termoclina e dinâmica de ondas — CMEMS, AVISO — 0,25°/diário.
- **OHC 0–100 m, 0–300 m, 0–700 m** — conteúdo de calor oceânico — memória térmica do sistema — ORAS5, GODAS, RG-Argo, EN4.
- **D20** — profundidade da isoterma de 20°C — proxy direto da termoclina equatorial — ORAS5 ou Argo gridded.
- **D26** — profundidade da isoterma de 26°C — relevante para conveção tropical — mesma fonte.
- **MLD** — profundidade da camada de mistura — controla troca oceano-atmosfera — ORAS5.
- **SSS** — salinidade superficial — afeta estratificação e camada de barreira — SMAP/SMOS satelital, Argo.
- **Perfis verticais T(z), S(z)** — usado para calcular OHC, MLD, estabilidade — Argo gridded (RG-Argo, EN4, Ishii), ORAS5.
- **Correntes u(z), v(z), w** — advecção de calor, ressurgência equatorial — ORAS5, GODAS, OSCAR para superfície.
- **Tensão de vento τ_x, τ_y** — forçante atmosférica do oceano, núcleo do feedback de Bjerknes — ERA5 — 0,25°/horário.
- **Fluxos de calor superficial** — latente, sensível, ondas curtas, ondas longas, líquido — ERA5, J-OFURO, OAFlux.

### 3.2 Variáveis oceânicas do Atlântico Tropical (modulação Brasil)

- **SST TNA** (Tropical North Atlantic, 5°N–25°N, 55°W–15°W) — controla posição da ZCIT, crítico para Nordeste.
- **SST TSA** (Tropical South Atlantic, 0–20°S, 30°W–10°E) — modula ZCIT e umidade.
- **Gradiente Atlântico TNA−TSA (AMM)** — Atlantic Meridional Mode — primeira ordem para chuva no semiárido NE.
- **ATL3** (3°S–3°N, 20°W–0°) — Atlantic Niño — variabilidade equatorial atlântica.
- **South Atlantic Subtropical High (SASH)** — posição e intensidade — controla advecção de umidade para o continente.
- **AMO/AMV** — Atlantic Multidecadal Oscillation/Variability — modulação de baixa frequência.

### 3.3 Ondas equatoriais — módulo dedicado

Este é um eixo central do escopo, com tratamento físico explícito.

**Ondas oceânicas equatoriais** (extraídas via decomposição modal a partir de SSHA, temperatura subsuperficial e correntes ao longo da banda equatorial ±5°):

- **Onda de Kelvin oceânica equatorial** — propaga para leste a ~2,5 m/s, atravessa o Pacífico em ~2–3 meses, carrega anomalias quentes para o leste; é o gatilho dinâmico de El Niño. Filtrada de SSHA pela banda de número de onda zonal positivo e fase apropriada (Roundy & Kiladis 2006).
- **Onda de Rossby oceânica equatorial** — propaga para oeste a ~0,6 m/s (modo baroclínico 1), com máximos fora do equador (~5°N e 5°S); responsável pela "recarga" no oscilador recharge-discharge de Jin (1997).
- **Ondas de Kelvin costeiras** — após reflexão da Kelvin equatorial na costa sul-americana, propagam-se para os polos ao longo do contorno; aquecem a costa peruana e podem alcançar a costa NE brasileira em eventos extremos.
- **Ondas de instabilidade tropical (TIW)** — período 20–30 dias, comprimento ~1000 km no Pacífico equatorial leste; mais ativas em La Niña; modulam SST e fluxos.

**Variáveis derivadas a calcular**:
- Amplitude da Kelvin oceânica em função de longitude e tempo (diagrama Hovmöller longitude × tempo).
- Energia da Kelvin nas janelas de 30, 60 e 90 dias.
- Fluxo de energia zonal.
- Detector de eventos discretos (rajadas de Kelvin) com timestamp e magnitude.

**Ondas atmosféricas equatoriais** (extraídas via espectro número-de-onda/frequência de Wheeler-Kiladis 1999 aplicado a OLR, U200, U850, geopotencial):

- **Kelvin atmosférica convectivamente acoplada** — eastward, ~15 m/s, módulo do MJO de alta frequência; modula conveção tropical.
- **Equatorial Rossby (ER) atmosférica** — westward, modos planetários.
- **MRG (Mixed Rossby-Gravity)** — westward dispersiva.
- **EIG, WIG (Eastward/Westward Inertio-Gravity)** — alta frequência.
- **MJO** — Madden-Julian Oscillation — modo 30–90 dias, modula chuva no Brasil (especialmente Sudeste e Sul) — RMM1, RMM2 diários (Wheeler & Hendon 2004), e adicionalmente RMM-modificado para o hemisfério sul.

### 3.4 Variáveis atmosféricas globais e tropicais

- **Vento u, v em 10 m, 850 hPa, 700 hPa, 500 hPa, 200 hPa** — ERA5.
- **Geopotencial em 850, 500, 200 hPa** — padrões sinóticos, bloqueios.
- **Temperatura do ar em 850, 500, 200 hPa** — estabilidade vertical.
- **Umidade específica e relativa em múltiplos níveis**.
- **SLP** — pressão ao nível do mar — base para SOI.
- **OLR** — Outgoing Longwave Radiation — proxy de convecção profunda — NOAA Interpolated OLR, AIRS, IMERG.
- **TCWV (IWV)** — vapor d'água integrado — combustível para chuva — ERA5, GPS-PWV (rede brasileira RBMC quando disponível).
- **IVT (Integrated Vapor Transport)** — magnitude e componentes u, v — preditor de chuva extrema, atmospheric rivers — calculado de ERA5.
- **Vorticidade relativa em 850, 500, 200 hPa** — ciclones, ondas.
- **Divergência em 200 hPa** — Alta da Bolívia, conveção tropical.
- **Função de corrente Ψ** e **potencial de velocidade χ** em 200 hPa — circulação de Walker e Hadley.
- **Omega (ω = dp/dt)** em 500 hPa — movimento vertical.
- **CAPE, CIN** — instabilidade convectiva — ERA5.
- **Lifted Index, K-Index, Total Totals** — diagnósticos de tempo severo.

### 3.5 Variáveis atmosféricas com foco regional Brasil

- **ZCAS (Zona de Convergência do Atlântico Sul)** — índice de intensidade e posição via OLR ou divergência baixa+convergência média — calculado por critério objetivo (Quadro & Abreu 1994 ou Carvalho et al. 2004).
- **Alta da Bolívia** — posição e intensidade em 200 hPa.
- **Vortex Ciclônico de Altos Níveis (VCAN)** — frequente sobre NE, detectado em 200 hPa.
- **Jato de Baixos Níveis a Leste dos Andes (LLJ)** — transporte de umidade da Amazônia para SE/S.
- **Frequência de frentes frias** — detectada em geopotencial 1000 hPa e gradiente de θ_e.
- **Bloqueios atmosféricos** — índice de Tibaldi-Molteni adaptado para hemisfério sul.
- **Cavados de onda curta** — análise sinótica em 500 hPa.

### 3.6 Variáveis de superfície e hidrologia (Brasil)

- **Precipitação** — alvo central. Fontes:
  - MERGE-INPE (0,1°, diário, calibrado com pluviômetros INMET+ANA, 2000–presente) — **verdade-terreno principal**.
  - CHIRPS v2.0 (0,05°, diário, 1981–presente) — segunda fonte, escala fina.
  - GPM IMERG Final v07 (0,1°, 30 min, 2000–presente) — sub-diário para extremos.
  - Xavier et al. 2022 (0,25°, diário, 1961–presente) — período histórico longo.
  - CPC Unified Gauge (0,5°, diário) — terceira opinião.
- **Temperatura 2 m máxima, mínima, média** — ERA5-Land (0,1°, horário), Xavier 2022 (0,25°).
- **Umidade relativa 2 m** — ERA5-Land.
- **Evapotranspiração** — real (ERA5-Land, MOD16A2, GLEAM v3.7) e potencial (Penman-Monteith calculado).
- **Radiação solar** — global, direta, difusa — ERA5, CERES.
- **Umidade do solo** — superficial (0–10 cm) e zona radicular (0–100 cm) — SMAP L3/L4 (9 km, diário, 2015–presente), ERA5-Land, SMOS.
- **Vazão fluvial** — estações ANA (Hidroweb), pontual.
- **Nível e armazenamento de reservatórios** — ONS para Sistema Interligado Nacional, ANA para reservatórios estaduais.
- **Armazenamento total de água (TWS)** — GRACE/GRACE-FO (~300 km, mensal) — assinatura de seca prolongada.

### 3.7 Vegetação e uso do solo

- **NDVI, EVI** — MODIS MOD13Q1 (250 m, 16 dias), Sentinel-2 (10 m, 5 dias) onde nuvens permitem.
- **VHI (Vegetation Health Index)** = α·VCI + (1−α)·TCI — combina condição vegetal e térmica; clássico para detecção de seca.
- **LAI (Leaf Area Index)** — MODIS MOD15A2H.
- **GPP, NPP** — produtividade primária — MODIS MOD17.
- **Land Surface Temperature** — MODIS MOD11A1, ECOSTRESS.
- **Albedo de superfície** — MODIS MCD43.
- **Cobertura e mudança de uso** — MapBiomas (30 m, anual, 1985–presente, Brasil completo).

### 3.8 Variáveis estáticas de alta resolução

Críticas para o decoder espacial pixel-a-pixel.

- **DEM** — MERIT-DEM (90 m) ou Copernicus GLO-30 (30 m) — derivar: declividade, aspecto, curvatura, TWI (Topographic Wetness Index), TPI (Topographic Position Index).
- **Solo** — SoilGrids 2.0 (250 m): textura (areia/silte/argila), profundidade efetiva, capacidade de retenção de água, classe taxonômica, carbono orgânico.
- **Bioma** — IBGE biomas; subdivisão Caatinga em estepe/savana-estépica para o foco NE.
- **Bacias hidrográficas** — ANA OTTO bacias níveis 1–7.
- **Distância à costa, latitude, longitude** — canais explícitos no modelo.
- **Climatologia local** — média, desvio padrão, percentis 10/50/90 de cada variável-alvo, por pixel × mês.

### 3.9 Índices climáticos — em frequência diária

Todos calculados em janela móvel diária a partir de campos diários, salvo onde indicado.

**Pacífico ENOS:**
- **Niño 1+2, Niño 3, Niño 3.4, Niño 4** — SSTA média em caixas — diário a partir de OISST.
- **ONI** — Niño 3.4 em janela móvel de 90 dias (versão diária do tradicional 3-mês ERSST).
- **EMI (El Niño Modoki Index)** — Ashok et al. 2007: EMI = SSTA[Box C] − 0,5·SSTA[Box E] − 0,5·SSTA[Box W].
- **E-index e C-index (Takahashi et al. 2011)** — rotação dos dois primeiros EOFs de SSTA tropical Pacífico; capturam não-linearidade EP vs CP.
- **TNI (Trans-Niño Index)** = SSTA[Niño 1+2] − SSTA[Niño 4] — discrimina tipos de evento.
- **SOI** — Tahiti − Darwin SLP, padronizado — diário.
- **Equatorial SOI** — alternativa baseada em pressão equatorial.
- **τ-index** — vento zonal médio em 5°N–5°S, 135°E–180°.
- **WWV (Warm Water Volume)** — volume de água acima de 20°C em 5°N–5°S, 120°E–80°W — diário a partir de ORAS5/GODAS.
- **WWV East e WWV West** — versões zonais.

**Atlântico:**
- **TNA, TSA, ATL3** — SSTA média em caixas.
- **AMM index** — primeiro modo conjunto SST+vento via MCA (Chiang & Vimont 2004).
- **AMO index** — SSTA Atlântico Norte 0–60°N, com suavização.

**Outros modos:**
- **IOD (DMI)** — SSTA[Box W: 50°E–70°E, 10°S–10°N] − SSTA[Box E: 90°E–110°E, 10°S–0°].
- **PDO** — primeiro EOF de SSTA Pacífico Norte (> 20°N).
- **IPO** — versão interdecadal tri-polar.
- **SAM/AAO** — modo anular sul — 700 hPa.
- **NAO** — modo anular norte.
- **PSA1, PSA2** — Pacific-South American patterns — EOFs de geopotencial 500 hPa hemisfério sul.
- **QBO** — vento zonal equatorial em 30 hPa e 50 hPa.

**MJO e ondas:**
- **RMM1, RMM2 (Wheeler-Hendon)** — diários.
- **OMI (OLR-based MJO Index)** — alternativa baseada só em OLR.
- **Amplitudes filtradas de Kelvin, ER, MRG, MJO atmosférica** — calculadas via filtro Wheeler-Kiladis em OLR e U850/U200.
- **Amplitude e fase da Kelvin oceânica equatorial** — Hovmöller filtrado de SSHA.

**Brasil (índices regionais):**
- **Índice ZCAS objetivo** — Carvalho et al. 2004.
- **Índice de monção sul-americana** — diferença de chuva entre 5–20°S e 0–5°S, ou índices de Zhou & Lau 1998.
- **Índice de bloqueio** — adaptado para o setor SAM.

**Seca:**
- **SPI** em janelas de 1, 3, 6, 9, 12, 24, 36 meses — atualizado diariamente conforme janela móvel.
- **SPEI** mesmas janelas — usa P − ETP.
- **Palmer Drought Severity Index (PDSI), self-calibrating PDSI**.
- **Reconnaissance Drought Index (RDI)**.
- **Standardized Soil Moisture Index (SSMI)**.
- **VHI** já listado.

---

## 4. Arquitetura de dados

### 4.1 Grades-alvo e resolução temporal

| Camada | Resolução espacial | Resolução temporal | Cobertura |
|---|---|---|---|
| Oceano global | 0,25° (~25 km) | Diário | Global |
| Oceano subsuperficial | 1° | Diário (ORAS5) ou 5-dia (Argo) | Global |
| Atmosfera global | 0,25° | Diário (agregado de horário) | Global |
| Atmosfera Brasil estendido | 0,1° | Diário | 50°S–10°N, 90°W–30°W |
| Superfície Brasil | 0,05° (≈5 km) | Diário | Brasil completo |
| Estáticas Brasil | 0,008° (≈1 km) | Constante | Brasil completo |

A grade de saída do modelo é **0,05° diária**. As demais são reamostradas/agregadas dentro do modelo via downscaling aprendido, não no pré-processamento — preservando informação original.

### 4.2 Pipeline ETL

Estrutura em sete estágios, cada um idempotente e versionado:

```
[1] INGESTÃO            ─ download paralelo via aiohttp / s3fs / cdsapi / earthaccess
        │
[2] VALIDAÇÃO           ─ checksums, esquemas, ranges físicos, gap detection
        │
[3] PADRONIZAÇÃO        ─ unidades SI, calendários alinhados, CF-conventions
        │
[4] ENRIQUECIMENTO      ─ cálculo de anomalias (climatologia rolling),
                          decomposição de ondas, índices derivados
        │
[5] HARMONIZAÇÃO        ─ grades comuns, máscaras de qualidade, missing
        │
[6] PERSISTÊNCIA        ─ Zarr particionado por tempo+variável, com chunks
                          otimizados para acesso espacial vs temporal
        │
[7] CATALOGAÇÃO         ─ intake-catalog ou STAC para descoberta
```

### 4.3 Stack tecnológico (Pangeo+)

- **Storage**: Zarr (cloud-optimized) em filesystem local ou S3-compatível (MinIO local, AWS S3 ou Wasabi).
- **Compute paralelo**: Dask distribuído, com workers em CPU; GPU dedicada para treino.
- **I/O climático**: xarray como API principal, com backend Zarr; rioxarray para raster geoespacial; cfgrib para GRIB.
- **Geoespacial**: geopandas (vetores), regionmask (máscaras de bioma/estado/bacia), shapely.
- **Reamostragem espacial**: xesmf (regridding conservativo, bilinear, nearest).
- **Análise de tempo**: xclim para indicadores climáticos padronizados.
- **Decomposição modal**: eofs (Dawson), pyEOF, scikit-learn para CCA/MCA.
- **Filtros de ondas**: tropycal, mjoindices, ou implementação custom de Wheeler-Kiladis com scipy.signal.
- **Catalogação**: intake, intake-xarray, intake-esm; STAC para metadados.
- **Versionamento de dados**: DVC com remote em S3-compatível.
- **Containers**: Docker + Mamba/Conda para reprodutibilidade exata.

### 4.4 Estratégia de chunking (Zarr)

Decisão crítica para performance. Recomendação:

- **Para acesso temporal (séries em ponto fixo, treino de modelos seq2seq)**: chunks grandes em tempo, pequenos em espaço — `(time=365, lat=64, lon=64)`.
- **Para acesso espacial (mapas em data fixa, visualização)**: chunks pequenos em tempo, grandes em espaço — `(time=1, lat=512, lon=512)`.
- Manter **duas cópias com chunking distinto** quando o custo de armazenamento permitir; ou aplicar rechunker para gerar visão alternativa sob demanda.

### 4.5 Climatologia rolling para anomalias

Para evitar vazamento de dados (data leakage) na validação:

- Para cada ponto temporal *t*, a climatologia usada para calcular anomalia em *t* é construída **apenas com dados estritamente anteriores** a um buffer de 5 anos antes de *t* (para evitar autocorrelação serial).
- Climatologia recalculada a cada ano novo (não rolling diário, que é caro), com janela de 30 anos antes do buffer.
- Esta política respeita o uso operacional real e o split temporal de treino/validação/teste.

### 4.6 Controle de qualidade

- **Detecção de outliers físicos**: limites por variável (SST entre −2°C e 35°C, etc.).
- **Detecção de saltos artificiais**: análise de descontinuidades em séries (mudança de instrumento).
- **Cruzamento entre fontes**: comparação MERGE × CHIRPS × IMERG para detectar discrepâncias regionais.
- **Máscara de qualidade por pixel**: flag binário disponibilizado junto ao dado.
- **Imputação**: para gaps curtos, interpolação temporal + espacial restrita; para gaps longos, máscara de missing propagada ao modelo (que deve saber tratar).

### 4.7 Bias correction camada-base

Aplicada apenas a saídas de modelos globais quando usadas como entrada (NMME, SEAS5):

- **Empirical Quantile Mapping (EQM)** por pixel × mês.
- **Quantile Delta Mapping (QDM)** quando preservar tendências importa.
- Treinada com dados de período-base; aplicada respeitando split temporal.

---

## 5. Arquitetura de redes neurais

### 5.1 Princípio modular hierárquico

O modelo é composto por **sete módulos** treináveis, conectados em grafo de fluxo de dados. Cada módulo tem entrada, saída e perda auxiliar definidas, permitindo treino conjunto (end-to-end) ou em estágios (greedy module-by-module).

```
                  ┌───────────────────────────────────────┐
                  │  ENTRADAS — TENSORES MULTI-RESOLUÇÃO  │
                  └─────────────────┬─────────────────────┘
                                    │
        ┌─────────────────┬─────────┴─────────┬──────────────────┐
        ▼                 ▼                   ▼                  ▼
  ┌─────────┐       ┌─────────┐         ┌──────────┐      ┌──────────┐
  │ Módulo  │       │ Módulo  │         │  Módulo  │      │  Módulo  │
  │ Oceano  │       │  Atmos  │         │  Ondas   │      │ Estática │
  │   3D    │       │  4D     │         │ Kelvin/  │      │ Alta-Res │
  │ encoder │       │ encoder │         │  Rossby  │      │ encoder  │
  └────┬────┘       └────┬────┘         └────┬─────┘      └────┬─────┘
       │                 │                   │                 │
       └──────────┬──────┴───────┬───────────┘                 │
                  ▼              ▼                             │
            ┌──────────┐  ┌─────────────┐                     │
            │  Módulo  │  │   Módulo    │                     │
            │ Dinâmica │  │  Teleconex  │                     │
            │   ENOS   │  │  Atlântico- │                     │
            │ (R-D osc │  │   Brasil    │                     │
            │ informed)│  │ (cross-attn)│                     │
            └────┬─────┘  └──────┬──────┘                     │
                 │               │                            │
                 └───────┬───────┘                            │
                         ▼                                    │
              ┌─────────────────────┐                         │
              │  Estado latente     │◀────────────────────────┘
              │  espaçotemporal     │
              │  do sistema         │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  Decoder super-res  │
              │  (U-Net + FiLM,     │
              │  ou Diffusion       │
              │  CorrDiff-like)     │
              └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       ┌─────────────┐       ┌─────────────┐
       │  Cabeça     │       │  Cabeça     │
       │  Seca NE    │       │  Chuva S    │
       │  (SPI, SPEI,│       │  (P90/95/99,│
       │  VHI, SM)   │       │  IVT, quant)│
       └─────────────┘       └─────────────┘
```

### 5.2 Módulo Oceano (encoder 3D temporal)

**Entrada:** tensor 5D `[batch, tempo, profundidade, lat, lon]` com canais T(z), S(z), u(z), v(z), SSHA, OHC, D20, MLD.

- Janela temporal: últimos 365 dias com pesos decrescentes (ou últimos 180 dias para variantes ágeis).
- Profundidade discretizada em 23 níveis padrão Argo até 2000 m.
- Domínio espacial: Pacífico Tropical estendido (30°S–30°N, 120°E–60°W) + Atlântico Tropical (30°S–30°N, 60°W–10°E).

**Arquitetura:**
- **3D-CNN com convoluções separáveis** (depth × spatial) para reduzir parâmetros — primeiras camadas.
- **Vision Transformer 3D (ViT3D)** com tokens espaçotemporais — camadas profundas — captura dependências de longo alcance (ondas viajam milhares de km).
- Atenção restrita a banda equatorial (±15°) com viés positivo para preservar invariância de propagação zonal.

**Saída:** vetor latente `[batch, D_ocean=512]` + mapas de features intermediárias.

**Perda auxiliar:** reconstrução de SSHA e OHC nos próximos 30 dias (autoencoder preditivo).

### 5.3 Módulo Atmosfera (encoder espaçotemporal)

**Entrada:** tensor 5D `[batch, tempo, nível-pressão, lat, lon]` com canais u, v, T, q, z, OLR, TCWV.

- Níveis: 1000, 850, 700, 500, 300, 200, 100 hPa (7 níveis).
- Janela temporal: 90 dias (atmosfera tem memória mais curta).
- Domínio: global tropical estendido (60°S–60°N).

**Arquitetura:**
- **AFNO (Adaptive Fourier Neural Operator)** ou **SFNO (Spherical FNO)** como base — opera em espaço de Fourier, captura dinâmica de ondas naturalmente. Inspirado em FourCastNet (Pathak et al. 2022).
- **Earth-Specific Transformer** com codificação posicional esférica — inspirado em Pangu-Weather (Bi et al. 2023).
- Camadas de cross-attention temporal entre janelas.

**Saída:** vetor latente `[batch, D_atm=512]` + mapas de features.

**Perda auxiliar:** previsão de OLR e z500 em horizontes 1, 7, 30 dias.

### 5.4 Módulo Ondas — Detector e Propagador

Módulo dedicado à física de ondas equatoriais. Combina pré-processamento espectral com aprendizado.

**Pré-processamento (não-treinável):**
- Filtro de Wheeler-Kiladis 2D no espaço número-de-onda zonal × frequência aplicado a OLR, U850, U200, SSHA equatorial. Extrai amplitudes e fases de cada modo: Kelvin atmosférica, ER, MRG, MJO, Kelvin oceânica equatorial.
- Diagramas de Hovmöller longitude × tempo para cada modo, normalizados.

**Camadas treináveis:**
- **CNN 1D temporal** ao longo de cada longitude para aprender padrões de propagação não-lineares.
- **Transformer com viés posicional de velocidade de fase** — embute o conhecimento de que cada onda tem velocidade característica.
- **Cabeça de detecção de eventos**: classifica rajadas de Kelvin (Kelvin wave bursts) e prevê sua amplitude e tempo de chegada à costa leste do Pacífico — proxy direto para gatilho de El Niño.

**Saída:** vetor latente `[batch, D_waves=256]` + previsão das próximas 4 semanas de cada modo.

**Perda auxiliar:** erro de previsão das fases das ondas (reconstrução de Hovmöller futuro).

### 5.5 Módulo Dinâmica ENOS (recharge-discharge informed)

Módulo pequeno que embute o oscilador de Jin (1997) como restrição física:

- Variáveis de estado: T (anomalia de SST em Niño 3) e h (anomalia de profundidade da termoclina, proxy WWV).
- Equações: dT/dt = R·T − γ·h + ε₁ ; dh/dt = −α·T + ε₂.
- Os parâmetros R, γ, α são **aprendíveis** mas inicializados com valores da literatura; ε₁, ε₂ são resíduos previstos pela rede a partir dos encoders Oceano + Atmosfera + Ondas.
- Rolloutdiferenciável de até 12 meses para previsão de estado ENOS.

**Saída:** trajetória prevista de T_Nino3 e WWV em passos diários, com bandas de incerteza estimadas por dropout Monte-Carlo ou ensemble.

**Vantagem:** interpretabilidade física + robustez fora da distribuição de treino.

### 5.6 Ponte de teleconexão Atlântico-Brasil (cross-attention)

**Entrada:** latentes de Oceano (Pacífico+Atlântico), Atmosfera (global), Ondas, Dinâmica ENOS, mais latente do encoder de estado recente do Brasil.

**Arquitetura:**
- **Cross-attention transformer** onde tokens-query são pontos do Brasil (uma query por região climática: NE-semiárido, NE-litoral, N-Amazônia oriental, N-Amazônia ocidental, CO, SE, S) e tokens-key/value são o estado climático global codificado.
- **Atenção condicional à estação do ano** (DJF, MAM, JJA, SON) via embedding aprendido — teleconexões variam dramaticamente entre estações.
- **Atenção condicional ao tipo de evento ENOS** (EP, CP, Neutro+, Neutro−, La Niña) inferido pelo E-index/C-index — força o modelo a aprender padrões distintos para Modoki vs Canônico.

**Saída:** vetor `[batch, regiões_BR=7, D_teleconex=256]` que codifica o sinal climático global "traduzido" para cada região brasileira.

### 5.7 Decoder de super-resolução pixel-a-pixel

Coração do produto. Mapeia o estado latente para campos físicos em grade 0,05°.

**Arquitetura: U-Net 2D híbrida com diffusion**:

- **Encoder do estado recente Brasil** (chuva, NDVI, SM dos últimos 60 dias em 0,1°) compõe o ramo de skip-connections da U-Net.
- **Latente global** entra via FiLM (Feature-wise Linear Modulation) em cada bloco residual — modula features locais com contexto climático global.
- **Features estáticas alta-resolução** (DEM, solo, MapBiomas, distância à costa) entram nas camadas finais via concatenação multi-escala.
- **Cabeça probabilística**: para cada pixel × variável-alvo, prevê parâmetros de distribuição (média, variância, e parâmetros de forma para distribuições assimétricas como Gama para chuva).
- **Refinamento por difusão** (opcional, fase 2): modelo de difusão condicional estilo CorrDiff (Mardani et al., NVIDIA 2024) que toma a saída U-Net como condicionamento e gera ensemble de realizações pixel-a-pixel calibrado — melhor para extremos.

### 5.8 Cabeças especializadas

**Cabeça Seca-NE:**
- Domínio: máscara do bioma Caatinga + zona semiárida SUDENE.
- Alvos: SPI-1, SPI-3, SPI-6, SPI-12 (probabilidades de tercis e percentis), SPEI mesmas escalas, VHI, anomalia de SM zona radicular, anomalia de NDVI.
- Loss: combinação de CRPS pixel-a-pixel + Brier para "tercil seco" + Focal loss (γ=2) para eventos de seca extrema (SPI < −1,5) para corrigir o desbalanceamento.
- Termo de regularização espacial: penaliza descontinuidades não suportadas por gradientes ambientais (preserva campo suave onde a física pede).

**Cabeça Chuva-Extrema-S:**
- Domínio: RS, SC, PR + máscara do bioma Mata Atlântica/Pampa.
- Alvos: probabilidade pixel-a-pixel de exceder P90, P95, P99 da climatologia local em janelas 1-dia, 5-dia, 30-dia; valor esperado da anomalia; valor esperado do maior evento na próxima janela.
- Loss: Pinball loss (quantile loss) para os quantis-alvo + Brier para exceedances + weighted loss com peso maior em eventos de IVT extremo (> 500 kg·m⁻¹·s⁻¹).
- Termo de penalização para falsos negativos em eventos de catálogo conhecido (julho/1983, set/2023, mai/2024).

### 5.9 Arquiteturas de referência da literatura (estado da arte 2023–2026)

Para fundamentar e fazer comparações:

- **FourCastNet** (Pathak et al. 2022) — FNO para tempo atmosférico, 0,25°, alto desempenho até 7 dias.
- **GraphCast** (Lam et al. DeepMind 2023) — GNN icosaédrica, supera HRES/IFS em métricas determinísticas até 10 dias.
- **Pangu-Weather** (Bi et al. 2023) — 3D Earth-Specific Transformer.
- **NeuralGCM** (Kochkov et al. 2024) — híbrido ML + GCM diferenciável, supera centros operacionais em sub-sazonal.
- **GenCast** (Price et al. DeepMind 2024) — modelo de difusão para previsão de tempo, ensemble probabilístico.
- **AIFS** (ECMWF 2024) — sistema operacional ML do ECMWF.
- **MetNet-3** (Andrychowicz et al. 2023) — alta resolução para precipitação 0–24 h.
- **CorrDiff** (Mardani et al. NVIDIA 2024) — difusão para downscaling km-scale.
- **ClimaX** (Nguyen et al. Microsoft 2023) — modelo fundacional para clima.
- **ENS-10** dataset + métodos relacionados a sub-sazonal.

Não vamos reimplementar nenhum desses, mas **adotaremos seus componentes-chave** (FNO, Earth-Specific Transformer, FiLM, difusão condicional) como blocos da nossa arquitetura modular.

### 5.10 Estratégia de treino

- **Pré-treino dos encoders** em tarefas auto-supervisionadas (reconstrução, predição mascarada) com todo o histórico de reanálise (ERA5+ORAS5+OISST 1979–2020).
- **Treino dos módulos físicos** (Dinâmica ENOS, Ondas) em isolamento com dados de fase neutra para aprender parâmetros básicos.
- **Treino end-to-end** com perdas multitarefa (encoders + decoder + cabeças) em janelas móveis temporais.
- **Fine-tuning operacional** com dados recentes (2020–) para captura de regime climático contemporâneo.
- **Curriculum learning**: começar com horizontes curtos (1 dia) e ir estendendo (7 dias, 30 dias, 90 dias, 180 dias).
- **Adversarial robustness**: aumentar com pequenas perturbações nas entradas para forçar invariância a ruído observacional.

---

## 6. Técnicas de XAI (eXplainable AI)

Cada previsão emitida deve vir com **atribuição interpretável** para uso operacional confiável e para validação científica.

### 6.1 Atribuição de features

- **SHAP (Shapley Additive exPlanations)** — para os componentes baseados em árvore (XGBoost baselines) e para a interface de modelos profundos via DeepSHAP/GradientSHAP.
- **Integrated Gradients (Sundararajan et al. 2017)** — atribuição para redes profundas; aplicada por canal × pixel × tempo.
- **Layer-wise Relevance Propagation (LRP)** — propaga relevância da saída para a entrada.
- **Occlusion sensitivity** — mascarar regiões da entrada (ex.: zerar SSTA do Pacífico Central) e medir mudança na previsão; gera mapas de importância espacial.
- **Saliency maps** clássicos para visualização rápida.

### 6.2 Análise causal

- **PCMCI / PCMCI+ (Runge et al. 2019, 2023)** — causal discovery em séries temporais climáticas; permite construir grafo causal entre índices e quantificar a importância causal de cada modo para o alvo. Implementação: tigramite.
- **Convergent Cross Mapping (CCM, Sugihara et al. 2012)** — detecta acoplamento causal em sistemas dinâmicos não-lineares.
- **Granger causality** — baseline.
- **Counterfactual analysis dirigida**: gerar entradas "contrafactuais" plausíveis (e.g., ENOS neutro com mesmo Atlântico) usando modelos generativos condicionais e medir mudança na previsão.

### 6.3 Composite analysis (XAI clássica do clima)

Antes de qualquer ML, composite analysis é o método estabelecido:

- **Composite por tipo de evento ENOS**: média de campos brasileiros em todos os eventos EP, todos os CP, todas as La Niñas, etc., para gerar "padrão típico" de cada classe.
- **Composite por fase MJO** (oito fases) — clássico para subsazonal.
- **Composite condicional dupla**: ENOS × Atlântico, ENOS × MJO, etc. — revela interações.
- **Significância estatística**: teste de Monte Carlo / bootstrap por pixel.

Esses composites servem como **baseline de interpretabilidade** com o qual comparar as atribuições do modelo neural — se o modelo "vê" os mesmos padrões dominantes que a composite revela, ele aprendeu física correta.

### 6.4 Decomposição modal

- **EOF/PCA** — primeiros modos de variabilidade de cada campo; usados para verificar se o modelo respeita a estrutura modal observada.
- **MCA (Maximum Covariance Analysis)** — modos conjuntos entre dois campos (e.g., SST Pacífico e chuva Brasil) — quantifica fração de variabilidade explicada por cada modo acoplado.
- **CCA (Canonical Correlation Analysis)** — versão normalizada da MCA.
- **EEOF (Extended EOF)** — captura modos com propagação temporal.
- **CEOF (Complex EOF)** — para ondas viajantes.

Para o pixel-a-pixel, EOF e modos derivados servem para **reduzir dimensionalidade da saída** (prever coeficientes dos primeiros modos + resíduo) — pode acelerar treino sem perder qualidade física.

### 6.5 Atenção e mapas internos do modelo

- **Visualização de mapas de atenção** dos transformers (encoder de Atmosfera, ponte de teleconexão) — permite ver "onde o modelo está olhando" para fazer cada previsão.
- **Probing classifiers** — treinar classificadores rasos sobre representações intermediárias para verificar se elas codificam variáveis físicas conhecidas (e.g., a representação contém Niño 3.4? OHC? fase MJO?).
- **Concept Activation Vectors (TCAV, Kim et al. 2018)** — testar se conceitos físicos (e.g., "El Niño Modoki") influenciam decisões.

### 6.6 Contrafactuais físicos

Para cada previsão de alto risco (e.g., probabilidade > 70% de seca severa em sub-bacia), gerar respostas a "e se":

- "E se o Pacífico estivesse neutro?" — substituir SSTA Pacífico por climatologia, manter resto, recomputar previsão.
- "E se o Atlântico Norte estivesse 1°C mais frio?" — perturbação controlada.
- "E se a MJO estivesse na fase 5 em vez de 2?" — substituir RMM.

Quantifica **decomposição causal da previsão** — fundamental para comunicação operacional.

### 6.7 Cartões de explicação operacional

Cada previsão pública vem com um cartão estruturado:

- Fatores que **aumentaram** a previsão de risco (top 3, com magnitude e fonte de dado).
- Fatores que **diminuíram** o risco (top 3).
- **Análogos históricos**: os 3 anos mais similares no histórico, com o que aconteceu de fato.
- **Faixa de incerteza** (intervalo P10–P90).
- **Nível de confiança** baseado na habilidade histórica do modelo para o mês × região × tipo de evento ENOS atual.

---

## 7. Técnicas de análise de distribuição espacial

### 7.1 Estatística espacial clássica

- **Moran's I global e local** — autocorrelação espacial; testa se padrões de seca/chuva são agregados ou aleatórios.
- **Geary's C** — alternativa baseada em diferenças.
- **LISA (Local Indicators of Spatial Association)** — clusters locais.
- **Getis-Ord G\*** — hot spots e cold spots.
- **Variogramas e krigagem** — caracterização da escala de correlação espacial.
- **Geographically Weighted Regression (GWR)** — relações entre variáveis variando no espaço.

Implementação: PySAL, libpysal, esda, mgwr.

### 7.2 Decomposição espaço-temporal

- **EOF / S-mode e T-mode PCA**.
- **POP (Principal Oscillation Patterns)** — modos com dinâmica linear.
- **DMD (Dynamic Mode Decomposition)** — modos espaçotemporais não-lineares.
- **SSA (Singular Spectrum Analysis)** — modos temporais em séries.
- **MSSA (Multichannel SSA)** — versão multivariada.
- **Wavelet 2D espacial** — decomposição por escala em mapas (importante para identificar escala dominante de eventos de chuva extrema).
- **Wavelet temporal (Morlet)** — análise tempo-frequência por pixel para detectar ciclos não-estacionários.

### 7.3 Análise de ondas — Wheeler-Kiladis e variantes

- **Espectro número-de-onda zonal × frequência** aplicado a campos equatoriais (OLR, U200, U850, SSHA).
- Filtros para cada modo de onda (Kelvin, ER, MRG, MJO, ondas de instabilidade).
- **Análise de regressão de onda**: composites condicionados em pico de cada modo de onda.

Implementação: scipy.signal + código adaptado de mjoindices, ou pacote tropycal.

### 7.4 Hovmöller e propagação

- **Hovmöller longitude × tempo** para SSHA equatorial, OLR, U200 — visualiza propagação de Kelvin e Rossby.
- **Hovmöller latitude × tempo** — propagação meridional.
- **Diagramas de fase RMM1-RMM2** para MJO.
- **Diagramas E-C** para tipos de El Niño (Takahashi et al.).
- **Espaço-tempo trajetórias** de centros de ZCAS, frentes frias, bloqueios — análise lagrangeana via algoritmos de tracking (TempestExtremes, MET).

### 7.5 Verificação espacial de previsões

Métricas pontuais (RMSE pixel-a-pixel) não bastam para campos espaciais — uma previsão pode ter padrão certo deslocado e parecer péssima. Métricas espaciais necessárias:

- **FSS (Fractions Skill Score, Roberts & Lean 2008)** — comparação de frações de pixels acima de limiar em janelas crescentes; padrão WMO para chuva.
- **SAL (Structure-Amplitude-Location, Wernli et al. 2008)** — decompõe erro em três componentes interpretáveis.
- **MODE (Method for Object-based Diagnostic Evaluation)** — verificação orientada a objetos (identifica chuvas como objetos e compara).
- **Neighborhood verification** — versões espacializadas de POD, FAR, CSI.
- **Wavelet-based skill scores** — comparam por escala.
- **CRPS espacializado** — pixel-a-pixel + agregação.

### 7.6 Análise de extensão espacial de eventos

Para seca:
- **Área sob seca** por categoria (SPI < −1, < −1,5, < −2) — série temporal.
- **Duração média e desvio** dos eventos pixel-a-pixel.
- **Severidade integrada** = soma temporal de déficit.

Para chuva extrema:
- **Footprint do evento** — área contígua acima de limiar.
- **Duração e progressão espacial** — tracking de objetos chuva.
- **Compound assessment** — concorrência espacial com solo saturado, declividade, uso do solo (para gatilho de deslizamentos).

---

## 8. Análise de impacto

### 8.1 Impacto sobre seca no Nordeste

**Indicadores primários:**
- **SPI** multi-escala (1, 3, 6, 9, 12, 24, 36 meses) — déficit padronizado de precipitação.
- **SPEI** mesmas escalas — incorpora demanda evaporativa (mais relevante em cenário de aquecimento).
- **PDSI / scPDSI** — balanço hidrológico simplificado, integra memória do solo.
- **VHI** — saúde da vegetação.
- **SSMI (Standardized Soil Moisture Index)** — umidade do solo padronizada.
- **Streamflow Drought Index** — para sub-bacias com estação ANA.
- **Storage Drought Index** — para reservatórios ONS.

**Caracterização de eventos:**
- Início (run theory): primeiro mês com SPI abaixo de limiar.
- Fim: primeiro mês com SPI acima de limiar.
- Duração, severidade, intensidade média.
- Área afetada por categoria.

**Análise de impacto setorial:**
- **Agrícola**: cruzamento com calendários agrícolas (sorgo, milho, feijão de sequeiro) — risco de quebra de safra.
- **Pecuário**: VHI + estimativa de oferta forrageira.
- **Hidrológico**: balanço Q-V em reservatórios chave (Sobradinho, Itaparica, Castanhão, Orós, etc.).
- **Energético**: nível Sobradinho × geração hidrelétrica NE.
- **Social**: cruzamento com áreas de vulnerabilidade socioeconômica (CadÚnico, IDH municipal).

### 8.2 Impacto sobre chuva extrema no Sul

**Indicadores primários:**
- **Percentis pixel-a-pixel** P90, P95, P99 em janelas diária, 3-dia, 5-dia.
- **Rx1day, Rx5day** — máximos anuais.
- **R95pTOT, R99pTOT** — fração de chuva total em dias extremos.
- **CDD (Consecutive Dry Days), CWD (Consecutive Wet Days)** — padrões.
- **Índices ETCCDI** completos.

**Análise de extremos com teoria de valores extremos:**
- **GEV (Generalized Extreme Value)** ajustada por pixel para máximos anuais.
- **GPD (Generalized Pareto Distribution)** sobre peak-over-threshold.
- **Períodos de retorno** 10, 25, 50, 100 anos por pixel.
- **Tendências em parâmetros** (non-stationary GEV) — captura mudança climática local.
- **Análise de extremos compostos** (compound events): chuva extrema + solo saturado → enchente; chuva extrema + declividade → deslizamento.

**Análise de impacto setorial:**
- **Defesa civil**: cruzamento com áreas de risco mapeadas (CEMADEN, CPRM).
- **Infraestrutura**: bacias hidrográficas urbanas, redes de drenagem.
- **Agricultura**: perdas por excesso hídrico em soja, milho safrinha, trigo.
- **Energia**: vertedouro em hidrelétricas do Paraná (Itaipu, etc.).

### 8.3 Eventos compostos e cascatas

- **Drought-to-flood transitions** — alternância seca → chuva extrema, comum em El Niño no Sul.
- **Heat-drought compound** — onda de calor + seca, amplificada por feedback solo-atmosfera.
- **Wet-cold compound** — chuva intensa + queda de temperatura, risco para agricultura sensível.

Métodos: análise de dependência (cópulas), análise de exceedance conjunta.

### 8.4 Comunicação e produtos finais

- **Mapas operacionais** atualizados diariamente (probabilidade tercil, percentil de extremos).
- **Boletins regionais** semanais com narrativa baseada em XAI cards.
- **API REST** para integração com sistemas de terceiros (ONS, defesa civil estadual).
- **Alertas** baseados em limiares calibrados — só dispara quando confiança histórica do modelo justifica.
- **Dashboard** interativo (Streamlit/Panel inicial; React+Mapbox para versão operacional).

---

## 9. Métricas de avaliação

### 9.1 Métricas determinísticas (com cuidado)

- **RMSE pixel-a-pixel** — reportado **com piso teórico** estimado pelo desvio padrão entre fontes observacionais (ex.: MERGE × CHIRPS × IMERG dão ideia do ruído observacional irredutível).
- **MAE** — robusto a outliers.
- **Bias** — deve ser próximo de zero por design.
- **R² (coeficiente de determinação)** — útil quando há sinal forte; reportado por região × estação × horizonte.
- **Anomaly Correlation Coefficient (ACC)** — padrão meteorológico.

### 9.2 Métricas probabilísticas (primárias)

- **CRPS (Continuous Ranked Probability Score)** — generaliza MAE para distribuições; pixel-a-pixel agregado.
- **CRPSS (CRPS Skill Score)** vs climatologia, persistência, NMME bias-corrected.
- **Brier Score** e **Brier Skill Score** para eventos binários (acima/abaixo do tercil).
- **RPS (Ranked Probability Score)** e **RPSS** para previsões de tercis — padrão IRI/CPTEC.
- **Logarithmic Score** — penaliza fortemente probabilidades miscalibradas.
- **Quantile loss** para os quantis preditos diretamente.

### 9.3 Calibração probabilística

- **Reliability diagrams** por categoria.
- **Sharpness** (largura média das distribuições preditas).
- **Resolution** (variância das previsões condicionais).
- **Decomposição de Murphy** (Reliability − Resolution + Uncertainty).
- **Spread-skill ratio** para ensembles.

### 9.4 Métricas espaciais

- FSS, SAL, MODE — Seção 7.5.

### 9.5 Métricas de extremos

- **Habilidade condicional em extremos**: ROC-AUC, POD, FAR, CSI, HSS para eventos > P95, P99.
- **Quantile skill** nos quantis extremos.
- **Reliability dos limiares extremos**.

### 9.6 Validação operacional

- **Hindcast retrospectivo 1991–2024** com walk-forward.
- **Eventos de catálogo** avaliados individualmente: El Niños 1982/83, 1997/98, 2015/16, 2023/24; La Niñas 1988/89, 1998–2001, 2010/11, 2020–2023; secas Nordeste 2012–2017; chuvas Sul set/2023, mai/2024.
- **Habilidade reportada por mês de inicialização** — torna visível a spring predictability barrier.
- **Cartões de verificação** estilo IRI atualizados continuamente.

---

## 10. Sobre "RMSE zero" — o que é fisicamente possível

A meta "RMSE zero" é fisicamente inatingível em qualquer sistema preditivo aplicado ao clima, por três razões fundamentais que precisam estar no escopo:

**1. Caos determinístico.** O sistema clima é caótico no sentido de Lorenz: erros infinitesimais nas condições iniciais crescem exponencialmente com taxa dada pelo expoente de Lyapunov positivo. Para a atmosfera, isso impõe um horizonte de previsibilidade determinística da ordem de 2 semanas. Para o ENOS (sistema oceano-atmosfera de variabilidade mais lenta), o horizonte é da ordem de 6–12 meses para a fase, e menos para a amplitude.

**2. Erro observacional irredutível.** A "verdade-terreno" usada para validar não é perfeita. MERGE-INPE tem erro RMSE intrínseco contra pluviômetros não-assimilados estimado em 5–15 mm/dia para o Brasil. Mesmo um modelo perfeito não pode ter RMSE menor que esse piso quando avaliado contra MERGE.

**3. Variabilidade interna não-prevísivel.** Mesmo sob mesmo forçamento de larga escala, eventos individuais de chuva têm componente puramente caótica (turbulência convectiva, micro-fontes de instabilidade) que nenhum modelo de qualquer arquitetura jamais pode prever determinísticamente em horizontes além de horas.

**O que é, portanto, **possível** e **desejável****:

- **RMSE próximo do piso observacional** em nowcasting (0–48 h) — meta tangível.
- **Bias próximo de zero** em todas as escalas — meta tangível.
- **CRPS pequeno** e **calibração probabilística boa** em horizontes sazonais — meta tangível.
- **RPSS positivo e maximizado** — meta tangível e padrão operacional.
- **R² determinístico significativo** apenas onde há sinal previsível (anomalias regionais sazonais condicionadas por ENOS forte) — meta tangível mas regionalmente variável.

O escopo deve **comunicar essa realidade** sem amenizar. Promessas de "RMSE zero" desmoralizam o projeto frente a especialistas e expõem usuários a expectativas que serão violadas. A meta correta é: **erro minimizado até o limite físico, com distribuições probabilísticas honestas que comunicam a incerteza residual**.

---

## 11. Plano de execução em fases

### Fase 0 — Fundação (meses 1–2)
- Decisão de stack final (Pangeo, DVC, MLflow).
- Container Docker reproduzível.
- Catálogo intake com primeiras fontes (OISST, ERA5, MERGE, CHIRPS).
- Pipeline ETL básico para uma variável (chuva MERGE) — prova de conceito.

### Fase 1 — Dados completos (meses 3–4)
- Ingestão e harmonização de **todas** as variáveis da Seção 3.
- Climatologias rolling calculadas.
- Decomposição de ondas (Wheeler-Kiladis) operacional.
- Índices diários (Niño 3.4, EMI, E/C-index, MJO RMM, WWV) calculados.
- Catálogo Zarr completo, com dois chunkings (temporal e espacial).

### Fase 2 — Baselines (mês 5)
- Persistência, climatologia, NMME bias-corrected.
- XGBoost por pixel × variável-alvo com features tabulares.
- Verificação contra eventos de catálogo.
- **Decisão de go/no-go**: se baselines já cobrirem necessidade do usuário, calibrar e entregar como MVP.

### Fase 3 — Modelos neurais base (meses 6–9)
- Treino do encoder Oceano (auto-supervisionado).
- Treino do encoder Atmosfera (auto-supervisionado, base FourCastNet-like).
- Módulo de Ondas integrado.
- Módulo Dinâmica ENOS com R-D oscillator.
- U-Net decoder de super-resolução, primeira versão sem difusão.
- Treino end-to-end com curriculum.

### Fase 4 — Cabeças especializadas e validação (meses 10–12)
- Cabeça Seca-NE com SPI/SPEI multi-escala.
- Cabeça Chuva-Extrema-S com quantis P90/P95/P99.
- Hindcast completo 1991–2024.
- Verificação rigorosa: RPSS, CRPS, FSS, SAL, calibração.
- XAI: composites, atribuição, contrafactuais físicos.

### Fase 5 — Refinamento avançado (ano 2)
- Diffusion decoder (CorrDiff-like) para extremos calibrados.
- Foundation pretraining com ClimaX-like se houver compute.
- Causal discovery PCMCI para validar grafo físico aprendido.
- Integração com NMME/SEAS5 como entrada adicional (após avaliação de ganho).

### Fase 6 — Operacional (ano 2+)
- API FastAPI + Streamlit dashboard.
- Pipeline de atualização diária.
- Cartões de explicação automáticos.
- Parcerias institucionais (ONS, CEMADEN, FUNCEME, ANA).
- Publicação de papers metodológicos e de validação.

---

## 12. Estrutura de pastas proposta

```
NINO26/
├── data/
│   ├── catalog/          # intake catalogs
│   ├── raw/              # dados crus baixados, organizados por fonte
│   ├── interim/          # padronizados, mesma resolução nativa
│   ├── processed/        # zarr stores finais (temporal e espacial)
│   └── static/           # DEM, solo, mascaras, climatologias
├── src/
│   ├── data/
│   │   ├── ingest/       # downloaders por fonte
│   │   ├── validate/
│   │   ├── standardize/
│   │   ├── enrich/       # anomalias, indices, ondas
│   │   └── harmonize/
│   ├── features/
│   │   ├── enso_indices.py
│   │   ├── atlantic_indices.py
│   │   ├── waves_kelvin.py
│   │   ├── waves_atmos.py
│   │   ├── spi_spei.py
│   │   ├── mjo.py
│   │   └── compounds.py
│   ├── models/
│   │   ├── encoders/
│   │   │   ├── ocean3d.py
│   │   │   ├── atmos_fno.py
│   │   │   └── waves.py
│   │   ├── dynamics/
│   │   │   └── recharge_discharge.py
│   │   ├── teleconnection/
│   │   │   └── cross_attention.py
│   │   ├── decoders/
│   │   │   ├── unet_super_res.py
│   │   │   └── diffusion_corrdiff.py
│   │   ├── heads/
│   │   │   ├── drought_ne.py
│   │   │   └── extreme_rain_s.py
│   │   ├── losses.py
│   │   └── training.py
│   ├── xai/
│   │   ├── shap_climate.py
│   │   ├── integrated_gradients.py
│   │   ├── occlusion.py
│   │   ├── composites.py
│   │   ├── causal_pcmci.py
│   │   ├── eof_mca.py
│   │   └── counterfactuals.py
│   ├── spatial/
│   │   ├── pysal_metrics.py
│   │   ├── verification_fss_sal.py
│   │   ├── wheeler_kiladis.py
│   │   ├── hovmoller.py
│   │   └── object_tracking.py
│   ├── impact/
│   │   ├── drought_characterization.py
│   │   ├── extreme_value_gev.py
│   │   ├── compound_events.py
│   │   └── sectoral_overlay.py
│   ├── verification/
│   │   ├── deterministic.py
│   │   ├── probabilistic.py
│   │   ├── spatial.py
│   │   └── case_studies.py
│   ├── api/
│   │   └── main.py
│   └── web/
│       └── dashboard.py
├── notebooks/
│   ├── exploration/
│   ├── validation/
│   └── papers/
├── configs/
│   ├── data_sources.yaml
│   ├── regions.yaml
│   ├── model_arch.yaml
│   ├── training.yaml
│   └── verification.yaml
├── tests/
├── docker/
├── dvc.yaml
├── mlflow_config.yaml
├── environment.yml
└── README.md
```

---

## 13. Referências bibliográficas-chave

- Ashok, K., Behera, S., Rao, A., Weng, H., Yamagata, T. (2007). El Niño Modoki and its possible teleconnection. *JGR Oceans*, 112, C11007.
- Bi, K. et al. (2023). Accurate medium-range global weather forecasting with 3D neural networks. *Nature*, 619, 533–538. (Pangu-Weather)
- Carvalho, L.M.V., Jones, C., Liebmann, B. (2004). The South Atlantic Convergence Zone: intensity, form, persistence, and relationships with intraseasonal to interannual activity and extreme rainfall. *J. Climate*, 17, 88–108.
- Chiang, J.C.H., Vimont, D.J. (2004). Analogous Pacific and Atlantic meridional modes of tropical atmosphere-ocean variability. *J. Climate*, 17, 4143–4158.
- Goddard, L. et al. (2013). A verification framework for interannual-to-decadal predictions experiments. *Climate Dynamics*, 40, 245–272.
- Grimm, A.M. (2003). The El Niño impact on the summer monsoon in Brazil: regional processes versus remote influences. *J. Climate*, 16, 263–280.
- Jin, F.-F. (1997). An equatorial ocean recharge paradigm for ENSO. Part I & II. *J. Atmos. Sci.*, 54, 811–847.
- Kochkov, D. et al. (2024). Neural general circulation models for weather and climate. *Nature*, 632, 1060–1066.
- Lam, R. et al. (2023). Learning skillful medium-range global weather forecasting. *Science*, 382, 1416–1421. (GraphCast)
- Mardani, M. et al. (2024). Residual diffusion modeling for km-scale atmospheric downscaling. *arXiv*. (CorrDiff)
- Mariotti, A. et al. (2020). Windows of opportunity for skillful forecasts subseasonal to seasonal and beyond. *BAMS*, 101, E608–E625.
- McPhaden, M.J. (2003). Tropical Pacific Ocean heat content variations and ENSO persistence barriers. *GRL*, 30, 1480.
- Pathak, J. et al. (2022). FourCastNet: A global data-driven high-resolution weather model. *arXiv*. (FourCastNet)
- Roberts, N.M., Lean, H.W. (2008). Scale-selective verification of rainfall accumulations from high-resolution forecasts of convective events. *Mon. Wea. Rev.*, 136, 78–97. (FSS)
- Roundy, P.E., Kiladis, G.N. (2006). Observed relationships between oceanic Kelvin waves and atmospheric forcing. *J. Climate*, 19, 5253–5272.
- Runge, J. et al. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. *Science Advances*, 5, eaau4996.
- Takahashi, K. et al. (2011). ENSO regimes: Reinterpreting the canonical and Modoki El Niño. *GRL*, 38, L10704.
- Vandal, T. et al. (2017). DeepSD: Generating high resolution climate change projections through single image super-resolution. *KDD*. (DeepSD)
- Wernli, H. et al. (2008). SAL — a novel quality measure for the verification of quantitative precipitation forecasts. *Mon. Wea. Rev.*, 136, 4470–4487. (SAL)
- Wheeler, M., Kiladis, G.N. (1999). Convectively coupled equatorial waves: Analysis of clouds and temperature in the wavenumber-frequency domain. *J. Atmos. Sci.*, 56, 374–399.
- Wheeler, M.C., Hendon, H.H. (2004). An all-season real-time multivariate MJO index. *Mon. Wea. Rev.*, 132, 1917–1932.
- Xavier, A.C. et al. (2016/2022). Daily gridded meteorological variables in Brazil. *Int. J. Climatol.*

---

*Fim do escopo.*
