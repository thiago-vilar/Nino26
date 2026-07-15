# Fase4-zero

Esta pasta restaura a Fase 4 historica exatamente como foi preservada antes da
refatoracao. Ela e uma referencia congelada e nao altera `F4_NINO`, `F4_NINA`
ou qualquer outra fase ativa.

## Conteudo restaurado

- 5 notebooks historicos (`4_0`, `4A`, `4B`, `4C` e `4D`);
- `fase4_utils.py` e o README original;
- 7 figuras;
- 28 arquivos de numeric-tables, incluindo as tabelas pixel-a-pixel;
- 2 arquivos de features F4;
- 6 produtos de modelagem F4;
- 25 tabelas/manifestos estatisticos;
- atlas Zarr F4C com 6.636 arquivos;
- manifesto original de figuras e recibo da quarentena.

Volume restaurado: aproximadamente 8,5 GB.

## Organizacao

| Caminho | Conteudo |
|---|---|
| `notebooks/` | Notebooks, utilitario e documentacao original |
| `outputs/figures/` | Figuras historicas F4 |
| `outputs/numeric-tables/` | Tabelas-fonte das figuras, inclusive pixels CHIRPS |
| `outputs/parquet/` | Features, modelagem e estatisticas F4 |
| `outputs/zarr/statistics/` | Atlas nativo de lags por pixel |
| `audit/` | Manifesto original e recibo da restauracao |
| `RESTORATION_MANIFEST.json` | Hash e tamanho de cada arquivo |

Os notebooks conservam seus outputs historicos e nao foram reexecutados. O
codigo antigo aponta para os caminhos usados na epoca; portanto, nao execute os
notebooks no proprio lugar antes de revisar esses caminhos.

## Auditoria

No WSL, a partir da raiz do projeto:

```bash
./.venv/Scripts/python.exe Fase4-zero/validate.py
```

