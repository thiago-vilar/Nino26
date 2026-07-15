# Metodologia NINO26

Esta metodologia complementa as diretrizes canônicas. As fases são estudos
independentes, não degraus de um pipeline único.

## Regras comuns

- OISST local em Niño 3.4 sustenta a identificação oceânica dos eventos.
- A definição NOAA combina limiar térmico, persistência e coerência atmosférica.
- P90 não é definição de El Niño e requer fundamentação específica quando usado.
- Todo estágio chamado pico deve ser tratado como **faixa de pico**.
- UFS+GLORYS é a apresentação oficial da base subsuperficial combinada, mantendo
  indicadores internos de fonte e cobertura.
- Nenhuma variável possui importância prévia; todas são hipóteses a testar.
- Uma janela móvel é um parâmetro do problema local. Tamanho, passo,
  centralização, histórico e horizonte devem ser documentados em cada análise.

## Métodos por fase

| Fase | Pergunta | Família de método | Produtos próprios |
|---|---|---|---|
| F1 | Quais dados locais estão disponíveis? | ingestão | fontes e inventários |
| F2 | Os dados estão padronizados e coerentes? | preparação e sanidade | matriz, cubos e linhas do tempo |
| F3 | Como se comporta o ciclo ENSO? | estatística | gênese, crescimento, faixa de pico e decaimento |
| F4 | Como o sinal ENSO se relaciona com o Brasil? | estatística | lags e distribuição espaço-temporal |
| F5 | É possível antecipar o ciclo e a faixa de pico? | RF/XGBoost | previsões do ciclo |
| F6 | É possível modelar a relação ENSO–Brasil? | RF/XGBoost | lags e distribuição espaço-temporal |
| F7 | É possível antecipar o ciclo e a faixa de pico? | ConvLSTM | previsões do ciclo |
| F8 | É possível modelar a relação ENSO–Brasil? | ConvLSTM | distribuição espaço-temporal |
| WEB | Como publicar e operar previsões? | aplicação | painel e previsão recorrente |

Nenhuma linha da tabela é pré-condição de outra.

## Fase 2 e validação in situ

A sanidade usa séries temporais das variáveis disponíveis. CTD/WOD, TAO/TRITON
e Argo formam um bloco separado de validação de UFS+GLORYS, com cobertura,
pareamento, número de observações, diferenças por profundidade, erro, viés e
gráficos comparativos. Lacunas permanecem explícitas.

## CHIRPS

O alvo brasileiro usa pixels no tamanho original CHIRPS. Os produtos devem
representar extremo de chuva, chuva forte, chuva normal, estiagem e seca, após
definição documentada das categorias. A apresentação ocorre pixel a pixel, por
região do país e por biomas dentro das regiões, usando shapefiles IBGE.

## Data augmentation

Pode ser considerado apenas nas Fases 5 e 7, se necessário, sempre restrito ao
treino. Nas Fases 6 e 8 a decisão está pendente. Não é método das demais fases.

## Janelas e lags

Janelas móveis devem ser adequadas à frequência e ao objetivo de cada análise.
Lags e distribuição espacial/temporal são produtos próprios das Fases 4, 6 e 8;
não devem ser impostos às fases que estudam somente o ciclo do Pacífico.

## Artefatos

Toda figura analítica deve ter tabela numérica correspondente. Imagens, tabelas
e JSONs ficam em árvores separadas conforme `docs/DIRETRIZES_FASES.md`.
