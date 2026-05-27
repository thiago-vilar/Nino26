# Projeto de previsão climática El NINO e seus impactos no Brasil

**Programa de Pós Graduação em Oceanografia UFPE**  
**Área:** Oceanografia Física  
**Responsável:** Thiago Vilar  

## NINO-BRASIL

### Resumo executivo

O Projeto NINO-BRASIL investiga a relação entre o aquecimento do Pacífico e as anomalias de precipitação no Brasil entre 1980 e a data presente.

O projeto organiza dados diários do Pacífico, dados atmosféricos no corredor Pacífico -> Brasil e dados de precipitação no Brasil para treinar modelos estatísticos e de machine learning. As saídas esperadas são mapas pixel-a-pixel e mapas coropléticos de seca, chuva acima do normal e chuva extrema.

Fluxo científico central:

```text
aquecimento do Pacífico -> ponte atmosférica Pacífico-Brasil -> precipitação no Brasil
```

## Pergunta de pesquisa

Como as variáveis oceanográficas e atmosféricas associadas ao aquecimento do Pacífico se relacionam com anomalias de precipitação no Brasil entre 1980 e a data presente?

Pergunta operacional:

```text
O estado do Pacífico no dia t ajuda a explicar ou prever anomalias de precipitação no Brasil em t + lag?
```

## Objetivos

- Construir base local 1980-presente.
- Organizar dados de ERA5, ORAS/ORAS5 e CPC/NOAA.
- Calcular anomalias oceânicas, atmosféricas e de precipitação.
- Avaliar lags entre Pacífico e Brasil.
- Treinar modelos baseline e modelos de machine learning.
- Gerar mapas pixel-a-pixel e coropléticos.
- Publicar resultados no GitHub Pages.

## Etapas

1. Download, armazenamento local, tratamento e disponibilização de dados.
2. Disponibilização de dados:
   - aquecimento do Pacífico: satélite + CTD + ORAS;
   - dados atmosféricos Pacífico -> Brasil;
   - dados de precipitação Brasil.
3. Arquitetura machine learning, validação, mapas, XAI e correção recorrente automatizada.
4. Deploy dos mapas coropléticos de previsão na web com GitHub Pages.

## Variáveis centrais

Oceano:

- SST, SSTA, SSHA;
- T(z), S(z);
- D20, termoclina, camada de mistura;
- conteúdo de calor 0-300 m e 0-700 m;
- correntes oceânicas u e v;
- fluxos de calor na superfície.

Atmosfera:

- SLP, tau_x, tau_y;
- u10, v10;
- u850, v850, q850;
- u500, v500, q500, z500, omega500;
- u200, v200, z200, div200;
- OLR e TCWV.

Precipitação:

- precipitação diária;
- anomalia diária;
- acumulados de 3, 5, 7, 15 e 30 dias;
- eventos P10, P90, P95 e P99.

## Arquivos do projeto

- `PlanoDiretor_NINO-BRASIL.md`: plano técnico por etapas.
- `DATA_SOURCES.md`: fontes, links, cobertura e termos de uso.
- `METHODOLOGY.md`: metodologia matemática e computacional.
- `requirements.txt`: bibliotecas Python.

## Como começar

1. Criar ambiente Python.
2. Instalar dependências de `requirements.txt`.
3. Montar `data/catalog/datasets.yaml`.
4. Implementar os scripts de download.
5. Processar um primeiro recorte pequeno para validar o pipeline.

