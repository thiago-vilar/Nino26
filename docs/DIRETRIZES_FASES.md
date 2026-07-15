# NINO26 — Diretrizes canônicas da pesquisa

**Revisão:** 2026-07-15

Este documento é a fonte de verdade do escopo científico e da divisão das fases.
Cada fase responde a uma pergunta própria e pode ser executada de forma
independente. Resultados de uma fase só podem ser herdados ou comparados por outra
quando isso for explicitamente útil e documentado; não há sequência obrigatória,
gate entre fases ou promoção automática de resultados.

## 1. Pergunta central e princípios

O NINO26 investiga El Niño e La Niña no Pacífico, seus ciclos de gênese,
crescimento, faixa de pico e decaimento, e sua relação espacial e temporal com o
clima do Brasil.

1. A fonte oceânica principal de superfície é a SST/SSTA OISST calculada
   localmente na região Niño 3.4.
2. Nenhuma variável recebe importância científica prévia. D20, OHC em qualquer
   camada, WWV, inclinação da termoclina, SSH/SLA, temperatura, salinidade e todas
   as demais começam como candidatas e só ganham relevância após teste adequado ao
   contexto da fase.
3. Sempre escrever **faixa de pico**, nunca tratar o pico como um instante
   isolado quando o objeto analisado for o estágio do evento.
4. Janelas móveis ou deslizantes devem ser definidas separadamente para cada
   contexto, respeitando frequência, horizonte, autocorrelação, disponibilidade
   de dados e risco de vazamento. Não existe uma janela universal do projeto.
5. UFS e GLORYS devem ser apresentados em conjunto como **UFS+GLORYS**, mantendo
   internamente a procedência e a janela real de cada fonte.
6. Figuras analíticas devem nascer de tabelas numéricas. PNG/JPG ficam em
   `data/processed/figures/`, CSV/Parquet em
   `data/processed/numeric-tables/` e JSONs de metadados fora dessas árvores, em
   `data/processed/metadata/` ou `data/audit/`.

## 2. Dados, preparação e sanidade

### Fase 1 — ingestão e base local

Reúne OISST, ERA5, CHIRPS, UFS+GLORYS, ORAS5 e fontes de validação. Cada conjunto
mantém procedência, unidade, cobertura temporal e resolução originais.

A Fase 1 é a única responsável por baixar e atualizar fontes. Para IBGE, deve
baixar e auditar UFs, Grandes Regiões e Biomas, registrando URL oficial, versão,
CRS, componentes do shapefile, hash do bundle, contagem e validade geométrica.

### Fase 2 — disponibilização e sanidade

Padroniza dados, calcula anomalias, organiza matriz semanal e cubos e disponibiliza
os produtos para qualquer fase. Deve produzir testes de sanidade em gráficos de
linha no tempo para as variáveis disponíveis. A Fase 2 não seleciona variáveis
importantes para as fases posteriores.

A Fase 2 não faz downloads. Ao rodar, reconstrói seus produtos com o estado local
mais recente entregue pela Fase 1 e informa, para cada variável, início, último
valor válido, cobertura, lacunas e defasagem até a data de execução. Um eixo
preenchido até o ano atual não basta para declarar os dados atualizados.

### Validação independente: CTD/WOD, TAO/TRITON e Argo

Ao final da Fase 2 deve existir uma seção ou produto exclusivo de validação in
situ, sem misturá-la com o treinamento de modelos:

- **CTD/WOD:** comparação de perfis, profundidades e variáveis compatíveis com os
  campos de UFS+GLORYS, declarando cobertura espacial e temporal.
- **TAO/TRITON:** comparação na faixa equatorial, respeitando localização das
  boias, períodos disponíveis e lacunas reais.
- **Argo:** comparação por perfil e profundidade onde houver cobertura, com
  destaque para a maior disponibilidade a partir dos anos 2000.
- Nenhuma dessas fontes substitui os cubos; elas aferem coerência, viés e
  incerteza.
- Lacunas são registradas, nunca convertidas silenciosamente em observações.
- Os resultados devem mostrar número de observações, período, distância
  espaço-temporal do pareamento, métricas de erro e gráficos comparativos.

## 3. Definição de El Niño e La Niña segundo a NOAA

Para classificação histórica, o projeto segue a definição operacional NOAA/CPC:

- região Niño 3.4: `5°N–5°S, 120°W–170°W`;
- média móvel de SST/SSTA de três meses;
- El Niño oceânico: anomalia `>= +0,5 °C`;
- La Niña oceânica: anomalia `<= -0,5 °C`;
- episódio histórico: limiar mantido por pelo menos cinco estações móveis de três
  meses consecutivas e sobrepostas;
- a caracterização plena do ENSO também requer sinais atmosféricos consistentes
  com o acoplamento oceano–atmosfera, como mudanças nos alísios, pressão,
  convecção e precipitação tropical.

Fontes oficiais: [NOAA Climate.gov — ENSO](https://www.climate.gov/enso) e
[NOAA CPC — ONI](https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php).

O P90 não define El Niño. Qualquer percentil só pode ser usado em análise
complementar após justificativa científica, declaração da variável, período de
referência, método de cálculo e finalidade.

## 4. Faixa de pico

A faixa de pico é um intervalo contínuo ao redor do extremo do evento. Seu limiar
relativo deve ser justificado e submetido a sensibilidade; a configuração inicial
compara 80%, 90% e 95% da magnitude do extremo. Esses percentuais delimitam a
faixa de pico e não substituem a definição NOAA do evento.

## 5. Fases independentes

### Fase 3 — El Niño e La Niña: estatística

Análises puramente estatísticas do ciclo ENSO: gênese, crescimento, faixa de pico
e decaimento. Pode testar qualquer variável disponível, sem importância prévia.
Não usa Machine Learning ou redes neurais.

### Fase 4 — relação estatística ENSO–Brasil

Aferição puramente estatística da relação do ENSO com o Brasil, incluindo lags e
distribuição espacial e temporal do sinal. Esses produtos pertencem a esta fase e
não são obrigação das fases voltadas somente ao ciclo do Pacífico.

### Fase 5 — El Niño e La Niña: Machine Learning

Análises exclusivamente com Random Forest e XGBoost para prever antecipadamente
a evolução das fases e a faixa de pico. Pode usar data augmentation somente no
treino e apenas quando necessário, com procedência e avaliação de ganho real.

### Fase 6 — relação ENSO–Brasil com Machine Learning

Random Forest e XGBoost para estudar a distribuição espaço-temporal do sinal no
Brasil. Lags, mapas e distribuição pertencem ao contexto desta fase. O uso de data
augmentation ainda não está definido e não deve ser presumido.

### Fase 7 — El Niño e La Niña com ConvLSTM

ConvLSTM para prever antecipadamente a evolução das fases e a faixa de pico. Pode
usar data augmentation somente no treino e apenas se necessário e validado.

### Fase 8 — Brasil com ConvLSTM

ConvLSTM para a distribuição espacial e temporal do sinal no Brasil. O uso de
data augmentation ainda não está definido e não deve ser presumido.

### FaseWEB — publicação e operação

Publicação, painel, operação recorrente e comunicação da previsão antecipada da
faixa de pico do El Niño. Deve declarar qual fase/modelo originou cada resultado,
sem fundir metodologias como se fossem uma única sequência.

## 6. CHIRPS e alvos no Brasil

Por enquanto, aplicam-se somente estas regras:

1. A unidade espacial do alvo é o pixel no tamanho original do CHIRPS.
2. Cada pixel apresenta anomalias ou classes relacionadas a extremo de chuva,
   chuva forte, chuva normal, estiagem e seca. As definições numéricas dessas
   categorias devem ser estabelecidas e documentadas antes da análise, sem serem
   presumidas nesta diretriz.
3. As apresentações devem respeitar três níveis: pixel a pixel, regiões do país e
   biomas por região.
4. Regiões e biomas devem ser derivados de shapefiles oficiais do IBGE, mantendo a
   relação reversível com os pixels originais.

Não há outras restrições CHIRPS vigentes neste documento.

## 7. Data augmentation

- Permitido nas Fases 5 e 7, somente se necessário.
- Sempre restrito ao treino e registrado.
- Não pode aumentar artificialmente o número inferencial de eventos.
- Nas Fases 6 e 8, sua adoção permanece em aberto; não implementar como requisito
  sem decisão científica posterior.
- Não se aplica às Fases 1, 2, 3, 4 ou FaseWEB.

## 8. Independência e comparação entre fases

- Nenhuma fase espera, libera, promove ou bloqueia outra fase.
- Uma fase não herda automaticamente variáveis, lags, janelas, alvos ou
  conclusões de outra.
- Herança de dados processados é permitida quando explicitamente declarada.
- Comparações entre estatística, Machine Learning e ConvLSTM são opcionais e
  devem manter protocolos e objetivos identificáveis.
- Lags e distribuição espacial/temporal são centrais apenas nas Fases 4, 6 e 8.
- A escolha de variáveis, janelas móveis e métricas deve ser refeita de acordo com
  a pergunta de cada fase.

## 9. Git e arquivos do projeto

Código, configurações, testes e documentação devem ser versionados. Dados grandes
permanecem fora do Git conforme `.gitignore`. `git add .` é permitido, mas o diff
deve ser revisado para evitar inclusão acidental de dados ou artefatos locais.

## 10. Contrato obrigatório de notebooks

Todo notebook gerado, em qualquer fase, deve começar com uma célula Markdown
explicativa, antes de qualquer célula de código. Cabeçalhos Markdown grandes
(`#`, `##`, `###`) são proibidos nas células; maiúsculas, **negrito** e *itálico*
são suficientes. A primeira célula contém, nessa ordem, as seções:

1. `**TÍTULO**` — identificação clara do objeto estudado e da fase;
2. `**CONTEXTO**` — descrição, delimitação e justificativa científica;
3. `**MOTIVAÇÃO**` — hipótese específica, motivação e função dos testes;
4. `**METODOLOGIA**` — métodos, dados, janelas e referências que fundamentam as escolhas;
5. `**RESULTADOS ESPERADOS**` — produtos esperados e contrato de saída, sem antecipar conclusões.

O notebook deve terminar com uma célula Markdown própria intitulada
`**REFERÊNCIAS BIBLIOGRÁFICAS**`, contendo as fontes efetivamente utilizadas. Essa
célula é sempre a última do notebook. O contrato vale tanto para geradores
automáticos quanto para notebooks criados manualmente.

No primeiro notebook de cada fase, a primeira célula começa com
`**COMANDO WSL2 — EXECUTAR FASE COMPLETA**` e um bloco Bash executável no terminal
WSL2 do VS Code. Esse comando aparece antes de `**TÍTULO**`.
