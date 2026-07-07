# Notebooks NINO-BRASIL

Este diretorio contem os notebooks executaveis do projeto.

## Fase 4

O caminho ativo e limpo da Fase 4, neste momento, e:

| Ordem | Notebook | Uso |
|---:|---|---|
| 0 | `fase4/0_fase4_sanidade_disponibilidade.ipynb` | Pre-flight: sanidade de disponibilidade, cobertura temporal e outliers. |
| I | `fase4/A_fase4_fontes_variaveis_series.ipynb` | Pre-flight: inventario de variaveis processadas; nao conta como 4A cientifico. |
| 4A | `fase4/4A_fase4_regionalizacao_chuva.ipynb` | Clusters/regioes-alvo CHIRPS para Brasil. |
| 4B | `fase4/4B_fase4_correlacao_regressao_defasada.ipynb` | Teleconexao pixel-a-pixel e por cluster. |
| 4C | `fase4/4C_fase4_modos_acoplados.ipynb` | EOF/MCA/SVD/CCA SST(Pacifico+Atlantico) x chuva. |
| 4D | `fase4/4D_fase4_atribuicao_composicoes.ipynb` | Pacifico vs Atlantico, composicoes e gate de variaveis para ML. |

Os notebooks antigos de contrato/status foram removidos porque nao entregavam
metricas cientificas reais. A trilha atual deve permanecer como notebooks
limpos, um por etapa, sempre gravando saidas numericas antes de figuras.

## Como abrir no VS Code

```powershell
cd C:\DEV\NINO26
code .
```

No VS Code, abra:

```text
notebooks\fase4\A_fase4_fontes_variaveis_series.ipynb
```

Selecione o kernel no canto superior direito:

```text
Python 3 (.venv NINO26)
```

Se esse kernel nao aparecer, rode uma vez no terminal integrado do VS Code:

```powershell
.\.venv\Scripts\python -m ipykernel install --user --name nino-brasil --display-name "Python 3 (.venv NINO26)"
```

## Saidas atuais

As tabelas atuais da Fase 4 ficam em:

```text
data\processed\parquet\statistics\phase4_all_processed_variables.csv
data\processed\parquet\statistics\phase4_all_processed_variables_detail.csv
data\processed\parquet\statistics\phase4d_atribuicao_pacifico_atlantico.csv
data\processed\parquet\statistics\phase4d_predictor_gate_for_ml.csv
data\processed\parquet\statistics\phase4d_questions_answers.md
```
