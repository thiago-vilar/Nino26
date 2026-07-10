# Correcoes aplicadas - revisao cientifica critica NINO26

Este painel registra como o parecer cientifico critico foi incorporado ao
projeto. Ele nao substitui a metodologia; funciona como trilha de auditoria das
correcoes obrigatorias.

## Correcoes obrigatorias incorporadas

| Ponto do parecer | Correcao aplicada |
|---|---|
| Janela "desde 1981" falsa para subsuperficie | `configs/project.yaml`, `README.md`, `DATA_SOURCES.md` e painel agora separam CHIRPS/OISST/ERA5 desde 1981 de GLORYS12 1993+, Argo/TAO validacao irregular e UFS 1981-1992 como ponte/benchmark segregado. |
| FDR em mapas pixel-a-pixel | Criado `nino_brasil.stats.significance.benjamini_hochberg_fdr`; Fase 4 e documentos exigem FDR Benjamini-Hochberg/Wilks. |
| Graus de liberdade efetivos | Criado `effective_sample_size` e `correlation_p_value`; `scripts/fase4_features.py::lagged_correlation` passa a retornar `n_eff` e `p_effective`. |
| Colinearidade D20/OHC/WWV/termoclina | Configuracao e docs tratam esse conjunto como bloco de recarga; ranking deve usar representante fisico, PCA ou regularizacao e reportar importancia por bloco. |
| Vazamento de climatologia/EOF/MCA | Fase 4 e metodologia agora exigem que climatologia, padronizacao, percentis, EOF, MCA/SVD, CCA e selecao de variaveis sejam fitados no treino de cada fold. O modo full/reference ficou apenas descritivo. |
| Atlantico tropical omitido | Adicionados dominios `ATL4`, `ATL3`, `TNA`, `TSA`, IOD oeste/leste; `ATL4` fica prioritario para Nordeste/teleconexao regional, enquanto `ATL3` fica primario para precursores ENSO; criados indices em `features/nino.py`; Fase 4/5B exigem comparacao Pacifico-only vs Pacifico+Atlantico. |
| Skill por mes/estacao e barreira de primavera | Configuracao e metodologia exigem skill por mes de inicializacao e estacao-alvo. |
| Benchmarks obrigatorios | Configuracao exige climatologia, persistencia amortecida, oscilador de recarga simples e plume dinamico/operacional quando disponivel. |
| Fase 3F Kelvin/SLA/vento | A Fase 3 foi reestruturada em 3A-3K; o 3F agora descreve ondas de Kelvin, SLA/SSH equatorial e anomalia zonal de vento, sem metrica acumulada superficial. |
| Fase 4 4A-4D | A Fase 4 foi reorganizada para alvo chuva no Brasil: 4A ciclo ENSO, 4B determinantes estatisticos, 4C distribuicao pixel-a-pixel com lags e 4D clusterizacao descritiva com estabilidade e gate estatistico da hipotese NEB seco / Sul umido em El Nino. |
| Politica temporal | Formalizada a hierarquia diario -> semanal -> mensal: diario para insumo/Kelvin, semanal como eixo canonico de analise, mensal apenas para series nativas/sensibilidade. |

## Arquivos tocados

- `configs/project.yaml`
- `README.md`
- `docs/METODOLOGIA.md`
- `docs/METODOLOGIA_FASE4.md`
- `docs/FASE4_PLANO.md`
- `docs/DATA_SOURCES.md`
- `docs/ARQUITETURA.md`
- `docs/CRONOGRAMA.md`
- `src/nino_brasil/features/nino.py`
- `src/nino_brasil/stats/significance.py`
- `src/nino_brasil/models/walk_forward.py`
- `scripts/fase4_features.py`
- `scripts/model_pipeline.py`
- `scripts/update_painel_executivo.py`
- `tests/test_features.py`
- `tests/test_modeling.py`
- `tests/test_statistics.py`
