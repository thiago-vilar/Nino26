# NINO-BRASIL

Projeto Python para diagnosticar fisicamente o aquecimento do Pacifico equatorial
no Nino 3.4, organizar a matriz oceanografica/atmosferica semanal e avaliar a
teleconexao ENSO -> chuva no Brasil por fases auditaveis.

Este README e a porta de entrada do projeto. Os documentos longos ficam em
`docs/`; o status local vivo fica em `painel_executivo.md`.

## Leia Primeiro

| Necessidade | Arquivo | Uso |
|---|---|---|
| Fonte canonica das fases | [docs/DIRETRIZES_FASES.md](docs/DIRETRIZES_FASES.md) | Espinha dorsal atual: Fases 1-6, FaseWEB, gates e regras cientificas. Se houver divergencia, este documento prevalece. |
| Parecer de organizacao | [docs/PARECER_ORGANIZACAO_2026-07-09.md](docs/PARECER_ORGANIZACAO_2026-07-09.md) | Leitura curta do estado real do projeto, inconsistencias corrigidas e pendencias. |
| Cronograma operacional | [docs/CRONOGRAMA.md](docs/CRONOGRAMA.md) | Estado em disco e ordem de execucao Fase 2 -> Fase 4. |
| Status executivo local | [painel_executivo.md](painel_executivo.md) | Painel gerado automaticamente com cobertura de dados, lacunas e proximo comando recomendado. |
| Fluxo, metodos e outputs | [docs/ARQUITETURA.md](docs/ARQUITETURA.md) | Desenho executivo do pipeline, produtos numericos e fronteiras por fase. |
| Comandos de download | [docs/RUNBOOK_DOWNLOADS.md](docs/RUNBOOK_DOWNLOADS.md) | Sequencia operacional para CHIRPS, OISST, ERA5, oceano diario, CTD/WOD e validacao in situ. |
| Oceano originalmente diario | [docs/RUNBOOK_OCEAN_DAILY.md](docs/RUNBOOK_OCEAN_DAILY.md) | Fontes, contrato cientifico, numero de requisicoes e retomada UFS/GLORYS12. |
| Fechamento da Fase 2 oceanica | [docs/RUNBOOK_FASE2_OCEANO.md](docs/RUNBOOK_FASE2_OCEANO.md) | Execucao completa UFS, GLORYS/GLO12, ORAS5 mensal e auditorias. |
| Fase 3 fisica | [docs/FASE3_RECOMENDACOES.md](docs/FASE3_RECOMENDACOES.md) | Diagnostico fisico Nino 3.4 com OISST local, eventos derivados da propria SST, subsuperficie e Kelvin. |
| Metodologia cientifica | [docs/METODOLOGIA.md](docs/METODOLOGIA.md) | Regras de climatologia, anomalias, diagnosticos fisicos e auditoria. |
| Pico como faixa | [docs/PICO_FAIXA_BIBLIOGRAFIA.md](docs/PICO_FAIXA_BIBLIOGRAFIA.md) | Bibliografia e motivacao pratica para delimitar o pico do El Nino como janela (faixa), definicao adotada e saidas correspondentes. |
| Fontes de dados | [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) | Variaveis, dominios, caminhos de raw/interim/processed e politica de armazenamento. |
| Pareceres e paineis | [docs/PARECERES](docs/PARECERES) | Pareceres tecnicos recebidos e paineis descritivos derivados deles. |
| Documentos historicos | [docs/LEGADO](docs/LEGADO) | Escopo, plano diretor, arquitetura RN anterior e README operacional anterior. |

## Fluxo Em 30 Segundos

```mermaid
flowchart LR
    A["Fontes externas: CHIRPS, OISST, ERA5, UFS, GLORYS/GLO12, ORAS5, CTD/WOD, TAO/TRITON, Argo, IBGE"] --> B["Fase 1: ingestao local auditavel"]
    B --> C["Fase 2: matriz semanal 1981-2026 e Zarr/grade comum"]
    C --> D["Fase 3: diagnostico fisico Nino 3.4"]
    C --> E["Fase 4: teleconexao ENSO -> chuva Brasil, sem ML"]
    E --> F["Fase 5: RF/XGBoost + XAI"]
    F --> G["Fase 6: redes neurais nativas + XAI"]
    C --> H["FaseWEB: publicacao e operacao"]
```

Regra de ouro do projeto: toda saida visual analitica gerada pelo projeto deve
nascer de uma saida numerica anterior, preferencialmente `Zarr` ou `CSV`.
Graficos oficiais espelhados da NOAA/PSL podem ficar em `docs/assets` apenas
como comparativo visual, nunca como metrica, rotulo ou entrada do pipeline.

## Decisoes Fixas

| Tema | Decisao |
|---|---|
| Janela historica | CHIRPS/OISST/ERA5 desde `1981-01-01`; subsuperficie declara janelas reais por fonte e exige sensibilidade `1993+`/`2000+`. |
| Frequencia mestre | Semanal W-SUN para analise integrada; diaria para insumo bruto e diagnosticos especificos; mensal apenas para comparacao/calibracao. |
| Grade comum | `0.25` grau, definida em [configs/project.yaml](configs/project.yaml). |
| Regioes principais | `nino34`, banda equatorial do Pacifico e Brasil/CHIRPS para teleconexao. |
| Chuva oficial | CHIRPS e insumo da Fase 4: teleconexao pixel-a-pixel, extremos/secas, P90 e lags semanais. Nao entra na Fase 3. |
| SST/SSTA principal | NOAA OISST diario baixado/local. |
| Indice ENSO | Eventos sao derivados da propria SST/SSTA OISST baixada com criterio termico NOAA/ONI local: media movel de 3 meses >= +0,5 C por 5 estacoes moveis; intensidade por pico ONI local fraco/moderado/forte/muito forte. |
| Pico dos eventos | O pico e delimitado como FAIXA (meses com ONI local >= pico - 0,1 C), alem do mes central de maximo; ver [docs/PICO_FAIXA_BIBLIOGRAFIA.md](docs/PICO_FAIXA_BIBLIOGRAFIA.md). |
| Memoria subsuperficial | NOAA UFS 1981-1992 como ponte historica, GLORYS12 diario desde 1993 como fonte principal, GLO12 operacional na cauda; ORAS5 mensal independente. |
| Validacao in situ | CTD/WOD, TAO/TRITON e Argo validam D20/OHC/termoclina onde houver cobertura; nao substituem os cubos gridded. |
| Matriz semanal Fase 2 | `nino34_master_weekly.csv`: 17 variaveis oceanicas unificadas + 14 variaveis atmosfericas ERA5 + `ocean_source_code` como metadado de fonte. |
| Modelagem/ML | Fora das Fases 3 e 4. Fase 5 usa apenas Random Forest/XGBoost + XAI; Fase 6 usa redes neurais nativas + XAI se vencer baselines e gates. |
| Publicacao/operacao | **FaseWEB** concentra painel, publicacao e recalibracao recorrente. |

## Comandos Essenciais

Instalacao:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e .
```

Saude e fluxo principal:

```powershell
.\.venv\Scripts\python scripts\data_pipeline.py plan
.\.venv\Scripts\python scripts\data_pipeline.py status
.\.venv\Scripts\python scripts\build_master_weekly.py --era5-years 1981:2026
.\.venv\Scripts\python scripts\fase3_build_inputs.py --force
.\.venv\Scripts\python scripts\run_fase3_all.py
.\.venv\Scripts\python scripts\run_fase4_all.py
.\.venv\Scripts\python scripts\update_painel_executivo.py
.\.venv\Scripts\python -m pytest -q
```

Downloads longos e retomada ficam no runbook: [docs/RUNBOOK_DOWNLOADS.md](docs/RUNBOOK_DOWNLOADS.md).

## Politica De Arquivos

Versione codigo, configs, testes e documentacao. Nao versione dados grandes em
`data/raw/`, `data/interim/` ou `data/processed/`.

`painel_executivo.md` e local, gerado automaticamente e ignorado pelo Git. Ele
pode diferir entre maquinas porque reflete os dados disponiveis em cada
computador.

## Mapa Da Raiz

| Caminho | Papel |
|---|---|
| `src/nino_brasil/` | Biblioteca do projeto. |
| `scripts/` | Entrypoints operacionais de download, curadoria, diagnosticos, Fase 2, Fase 3, Fase 4 e painel. |
| `notebooks/fase2/` | Sanidade da matriz semanal e de todas as variaveis disponiveis. |
| `notebooks/fase3/` | Diagnostico fisico Nino 3.4, sem ML/RN. |
| `notebooks/fase4/` | Teleconexao ENSO -> chuva Brasil, sem ML/RN. |
| `configs/project.yaml` | Fonte de verdade para dominios, grade e parametros principais. |
| `tests/` | Testes de features, saidas numericas e Zarr. |
| `docs/` | Documentacao organizada. |
| `papers/` | Artigos cientificos de apoio. |
| `data/` | Dados locais, geralmente nao versionados. |
