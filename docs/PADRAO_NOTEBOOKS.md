# Padrão dos notebooks NINO26

## Estrutura mínima

1. primeira célula Markdown, sem `#`, `##` ou `###`, com `**TÍTULO**`,
   `**CONTEXTO**`, `**MOTIVAÇÃO**`, `**METODOLOGIA**` e
   `**RESULTADOS ESPERADOS**`, nessa ordem;
2. pergunta da fase;
3. fontes e cobertura;
4. parâmetros específicos, inclusive janela móvel quando houver;
5. método da própria fase;
6. resultados numéricos;
7. figuras derivadas das tabelas;
8. limitações e conclusão restrita à fase;
9. última célula Markdown com `**REFERÊNCIAS BIBLIOGRÁFICAS**`.

A primeira célula deve anteceder qualquer código e explicar detalhadamente o que
será feito. A célula de referências deve ser a última do arquivo.

No primeiro notebook de cada fase, um bloco Bash WSL2 para executar a fase inteira
deve aparecer no início da primeira célula, antes de `**TÍTULO**`.

## Independência

- Não declarar uma fase como pré-requisito de outra.
- Não herdar automaticamente variáveis, janelas, lags ou conclusões.
- Lags e distribuição espacial/temporal pertencem às Fases 4, 6 e 8.
- Data augmentation só pode aparecer em F5/F7 se necessário; F6/F8 aguardam
  decisão; nas demais fases não se aplica.

## Terminologia

- Usar “faixa de pico”.
- Usar “UFS+GLORYS” em apresentações.
- Não declarar importância prévia de variável.
- Não usar P90 como definição de El Niño.

## Artefatos

- `Fig...png` ou `Fig...jpg` em `data/processed/figures/`;
- `Tab...csv` ou Parquet em `data/processed/numeric-tables/`;
- JSONs em `data/processed/metadata/` ou `data/audit/`.
