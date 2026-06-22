# Painel descritivo da revisao do parecer

Data: 2026-06-13

## Veredito tecnico

O parecer faz sentido. A leitura e coerente com o estado do repositorio: o projeto tem boa disciplina anti-vazamento e uma arquitetura cientifica promissora, mas carregava riscos reais de reprodutibilidade, custo computacional na Fase 5 e consistencia de escrita Zarr. Apliquei nesta rodada as recomendacoes de maior retorno e menor risco de mudanca metodologica.

## Painel de modificacoes aplicadas

| Pilar | Diagnostico do parecer | Decisao | Modificacao aplicada |
|---|---|---|---|
| Performance Fase 5 | `run_walk_forward` reconstruia X para regressao e para cada evento | Procede | `build_predictor_matrix` constroi X uma vez por `(fold, lag)` e `align_target_to_predictor_matrix` reaproveita X para anomalia e eventos |
| Performance Fase 5 | P10/P25/P75/P90 eram calculados em passadas independentes | Procede | `local_percentile_thresholds` calcula multiplos quantis em uma unica passagem por fold |
| Escala pixel-a-pixel | `flatten` pode tornar permutation importance inviavel | Procede | `permutation_feature_limit=500` pula permutation importance quando o numero de features passa do limite e grava aviso no artifact de importancia |
| Modelo padrao | XGBoost era pesado para hardware pessoal | Procede | Default da Fase 5 mudou para Ridge + Random Forest + LightGBM; XGBoost segue disponivel por CLI |
| Zarr | Escritores podiam gerar formatos mistos com `zarr>=3` | Procede | `ZARR_FORMAT = 2` centralizado e usado por todos os writers Zarr diretos |
| Chunking | `time=31` prejudicava leitura analitica longa | Procede, com escolha conservadora | `chunk_plan` agora usa chunks temporais anuais (`time<=365`) e documenta a politica |
| ETL diario | Dados ja diarios eram resampleados sem necessidade | Procede | `standardize_dataset_to_daily` retorna cedo para calendario diario regular e registra `identity_daily` |
| Seguranca | `zipfile.extractall` aceitava path traversal | Procede | Extracao ZIP agora valida `Path.resolve()` antes de extrair |
| Reprodutibilidade | Pacote nao era instalavel e dependencias nao estavam congeladas | Procede | Adicionados `pyproject.toml`, `requirements.lock.txt`, `pytest` e README com instalacao editavel |
| Documentacao | README tinha numeracao duplicada e default desatualizado | Procede | README atualizado com lock, `pip install -e .`, pytest, LightGBM e numeracao corrigida |

## Backlog mantido

| Prioridade | Item | Motivo para nao fechar nesta rodada |
|---|---|---|
| Alta | Registrar versoes dos pacotes no ledger de cada run | Requer decidir o schema do ledger operacional |
| Alta | Adaptador e teste de contrato para internals do `cdsapi` | Mudanca sensivel de ingestao, melhor isolar em PR proprio |
| Alta | Wrapper formal dos experimentos de ablation A-F | Exige convencao de nomes e saidas comparativas |
| Media | `src/nino_brasil/data/paths.py` como fonte unica de caminhos | Refatoracao ampla dos scripts de download |
| Media | Validacao tipada de `configs/project.yaml` | Requer escolher dataclass puro ou pydantic |
| Media | CI + ruff/pre-commit | Bom proximo passo depois de estabilizar dependencias |
| Media | LICENSE e `CITATION.cff` | Precisa de decisao autoral/licenca pelo pesquisador |
| Cientifica | Marine heatwave OISST e features CP/EP/Modoki | Extensoes cientificas relevantes, mas fora do hardening imediato |

## Validacao executada

- `.\.venv\Scripts\python.exe -m pytest -q`
- `.\.venv\Scripts\python.exe -m compileall -q src scripts tests`

Resultado: 16 testes passaram.
