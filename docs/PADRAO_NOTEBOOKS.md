# Padrão de notebooks NINO-BRASIL

Toda análise do projeto vive num notebook com **premissa clara** e **saídas
rastreáveis**. Este documento é a fonte de verdade do padrão; ele é aplicado (e
reaplicado) automaticamente por `scripts/padronizar_notebooks.py` e conferido por
`scripts/validar_figuras.py`.

## 1. Estrutura obrigatória de cada notebook

1. **Cabeçalho** (célula markdown, marcada por `<!-- NINO26-CABECALHO v1 -->`):
   - `# <CÓDIGO> — <título>` e o **Código da fase/letra** (ex.: `3C`, `4C`, `5A`) + a **Hipótese** (HIP0–HIP5).
   - **Descritivo** — por que o notebook existe.
   - **Pergunta** — a pergunta específica que ele responde.
   - **Desafio** — a hipótese a testar / a armadilha metodológica a vencer.
   - **Metodologia (com referências)** — método e citações `(Autor, ano)`.
   - **Contrato de saídas** — a tabela `código → figura → numeric-table → descrição`.
2. **Todas as células de código** (o trabalho) — preservadas integralmente.
3. **Rodapé** (célula markdown, `<!-- NINO26-REFERENCIAS v1 -->`) — **Referências Bibliográficas**.

Reexecutar `padronizar_notebooks.py` **sobrescreve apenas** o cabeçalho e o rodapé
(pelas sentinelas); nunca toca no código.

## 2. Código predecessor único: `Fig_<F><B><NN>`

- `F` = fase (2–8), `B` = bloco/letra (0, A, B, C, …), `NN` = sequência (2 dígitos).
- Variantes (`_rf`, `_xgb`, `_en`, `_ln`) mantêm o código **globalmente único**.
- **O mesmo código nomeia a figura e a numeric-table**:

```
data/processed/figures/fase4/Fig_4C01.png
data/processed/numeric-tables/fase4/Fig_4C01/
    ├── <tabela>.csv       (dados congelados que geraram a figura)
    ├── manifest.csv       (linhas, colunas, sha256 de cada tabela)
    └── README.md          (título + descrição + hipótese)
```

## 3. Co-geração e sobreposição (uma única chamada)

Toda figura é salva por `nino_brasil.viz.registrar_figura`, que grava **na mesma
chamada** a figura, as tabelas numéricas, o `manifest.csv`, o `README.md` e uma
linha no manifesto global `data/processed/figuras_manifesto.csv` — sempre por
**sobreposição** (idempotente: corrigir e rodar de novo atualiza tudo por cima,
sem CSV órfão nem duplicata).

```python
from nino_brasil.viz import registrar_figura

registrar_figura(
    fig, "Fig_4C01", fase=4, bloco="C",
    titulo="Sinal pixel-a-pixel no melhor lag",
    descricao="r por pixel entre SSTA e a anomalia de chuva; célula cheia passa FDR.",
    hipotese="HIP1",
    notebook="notebooks/fase4/4C_sinal_pixel_lags.ipynb",
    fontes={"phase4C_best_lag_pixel": df_best, "phase4C_lag_resposta_neb_sul": df_reg},
)
```

Isso substitui o `save_fig(...)` solto e o antigo
`export_numeric_tables_for_figures.py` (dicionário fixo que divergia dos nomes
reais das figuras — causa do `numeric-tables/` vazio).

## 4. Validação

```bash
python scripts/validar_figuras.py --strict
```

Falha (exit 1) se: figura sem numeric-table homônima; código fora do padrão ou
duplicado; PNG fora do manifesto; lixo `_tmp*`/legado. Rode no `pre-commit`/CI e
como último passo de cada `run_faseN_all.py`.

## 5. Mapa Hipótese → Fase

| Hipótese | Fase(s) | Pergunta central |
|---|---|---|
| HIP0 | 3 | Precursores de recarga antecedem o pico e caracterizam as 4 fases? |
| HIP1 | 4 | O Pacífico modula a chuva do Brasil (NEB seco / Sul úmido em El Niño)? |
| HIP2 | 5 | RF/XGBoost caracterizam o ciclo melhor que a Fase 3 e os baselines? |
| HIP3 | 6 | RF/XGBoost preveem a chuva regional melhor que a Fase 4 e os baselines? |
| HIP4 | 7 | ConvLSTM acrescenta skill sobre índices escalares/ML tabular? |
| HIP5 | 8 | ConvLSTM mapeia o Pacífico no campo de chuva do Brasil melhor que 4/6? |

(A Fase 2 não testa hipótese: é a sanidade que autoriza usar o master.)

## 6. Reaplicar o padrão

```bash
python scripts/padronizar_notebooks.py --in-place     # grava nos próprios notebooks
python scripts/padronizar_notebooks.py --out-dir rev  # gera espelho para revisão
```

O conteúdo científico de cada cabeçalho (Descritivo/Pergunta/Desafio/Metodologia/
referências e o contrato de figuras) vive no dicionário `NB` do script — editar lá
mantém os 21 notebooks coerentes de uma vez.
