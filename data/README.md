# Dados locais NINO26

## Organização

- `raw/`: downloads e caches por fonte;
- `interim/`: produtos intermediários retomáveis;
- `processed/`: matrizes, cubos e saídas científicas;
- `audit/`: relatórios e recibos de execução;
- `catalog/`: inventário de conjuntos de dados.

As fases são independentes e podem consumir os produtos disponibilizados por F1
e F2 sem aguardar qualquer outra fase.

## Produtos visuais

- `processed/figures/`: apenas PNG/JPG;
- `processed/numeric-tables/`: apenas CSV/Parquet correspondentes às figuras;
- `processed/metadata/`: JSONs e manifests.

## Fontes

- OISST identifica a componente oceânica do ENSO segundo a definição NOAA.
- UFS+GLORYS é a denominação conjunta da base subsuperficial; a procedência de
  cada valor continua registrada internamente.
- CHIRPS fornece alvos brasileiros por pixel no tamanho original.
- IBGE fornece regiões e biomas.
- CTD/WOD, TAO/TRITON e Argo validam UFS+GLORYS em seção própria da Fase 2.

P90 não é definição de El Niño. Variáveis e janelas só são selecionadas após
teste e justificativa no contexto de cada fase.

Dados grandes não devem ser adicionados ao Git.
