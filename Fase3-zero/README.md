# Fase3-zero

Esta pasta restaura, sem reinterpretacao metodologica, a Fase 3 cientifica que
existia antes da refatoracao em `F3Nino`/`F3Nina`.

## Estado

- fonte recuperada: `data/quarantine/output_reset_20260713T220000BRT`;
- ultimo commit anterior a revisao nao commitada: `61ae278` (2026-07-11);
- 11 notebooks historicos com outputs preservados;
- 32 figuras;
- 110 arquivos de suporte numerico das figuras;
- 147 tabelas e manifestos estatisticos;
- nenhum arquivo da F3 atual foi sobrescrito.

## Organizacao

| Caminho | Conteudo |
|---|---|
| `notebooks/` | Notebooks 3A-3L, `fase3_utils.py` e a documentacao cientifica original |
| `outputs/figures/` | Figuras historicas `Fig_3...` |
| `outputs/numeric-tables/` | Tabelas-fonte usadas pelas figuras |
| `outputs/statistics/` | Tabelas estatisticas `phase3...` e seus manifestos |
| `audit/` | Manifesto original de figuras e recibo da quarentena |
| `RESTORATION_MANIFEST.json` | Hash de cada arquivo desta restauracao |

## Leitura recomendada

1. `notebooks/README.md`
2. `notebooks/RELATORIO_FINAL_FASE3.md`
3. `notebooks/INDICE_FIGURAS_FASE3.md`
4. notebooks `3A` a `3L`

Os notebooks foram restaurados com os outputs originais. Eles nao devem ser
executados no proprio lugar: o codigo historico ainda aponta para os caminhos
de saida usados na epoca. A primeira etapa e revisar a narrativa e decidir o
que sera promovido para uma nova F3, sem alterar esta copia congelada.

## Auditoria

No terminal, a partir da raiz do projeto:

```bash
./.venv/Scripts/python.exe scripts/validate_fase3_zero.py
```

