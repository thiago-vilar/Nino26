# Arquitetura NINO26

O projeto é uma coleção de estudos independentes sobre uma base compartilhada,
e não um pipeline linear com gates entre fases.

```mermaid
flowchart TB
    D["Fontes locais"] --> F1["F1 · ingestão"]
    F1 --> F2["F2 · preparação, cubos e sanidade"]
    F2 -. "dados disponíveis, sem dependência científica" .-> F3["F3 · estatística ENSO"]
    F2 -.-> F4["F4 · estatística ENSO–Brasil"]
    F2 -.-> F5["F5 · RF/XGBoost ENSO"]
    F2 -.-> F6["F6 · RF/XGBoost ENSO–Brasil"]
    F2 -.-> F7["F7 · ConvLSTM ENSO"]
    F2 -.-> F8["F8 · ConvLSTM ENSO–Brasil"]
    F3 -. "comparação opcional" .-> WEB["FaseWEB"]
    F4 -.-> WEB
    F5 -.-> WEB
    F6 -.-> WEB
    F7 -.-> WEB
    F8 -.-> WEB
```

## Fronteiras

- F1 ingere e registra procedência.
- F1 também atualiza e audita as malhas IBGE de UFs, regiões e biomas.
- F2 trata, organiza e disponibiliza o estado mais recente recebido da F1, sem
  baixar fontes nem selecionar variáveis para outras fases.
- F2 publica cobertura, frescor por variável, sanidade temporal e validações no
  notebook F2Z.
- F3, F5 e F7 estudam o ciclo e a faixa de pico por métodos distintos.
- F4, F6 e F8 estudam lags e distribuição espaço-temporal no Brasil por métodos
  distintos.
- FaseWEB publica resultados identificando sua fase de origem.
- Nenhuma fase promove, bloqueia ou valida outra.

## Dados principais

- OISST: superfície e identificação oceânica Niño 3.4.
- UFS+GLORYS: subsuperfície, com fonte e período preservados internamente.
- ERA5: atmosfera.
- CHIRPS: alvos brasileiros em pixels no tamanho original.
- IBGE: regiões e biomas por shapefiles oficiais.
- CTD/WOD, TAO/TRITON e Argo: validação independente de UFS+GLORYS.

## Saídas

| Tipo | Local |
|---|---|
| figuras PNG/JPG | `data/processed/figures/` |
| tabelas CSV/Parquet | `data/processed/numeric-tables/` |
| metadados JSON | `data/processed/metadata/` ou `data/audit/` |
| cubos e matrizes de trabalho | `data/processed/` |

Figuras devem nascer de tabelas, mas JSONs não podem ser misturados às árvores de
figuras e tabelas.

## Execução

Cada fase possui entrada, configuração, runner, validação e outputs próprios.
Reutilização de produtos entre fases é opcional e deve ser declarada no run. O
fingerprint do código é informativo: sua alteração gera aviso, não invalidação
automática de runs históricos.
