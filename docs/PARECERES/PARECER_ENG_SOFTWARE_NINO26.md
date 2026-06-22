# Parecer Técnico — NINO-BRASIL
**Engenharia de Software & Potencial Científico das Fases 3–5**

Responsável: Thiago Vilar — PPGO/UFPE, Oceanografia Física
Revisor: Arquiteto de Soluções Sênior
Escopo revisado: `src/nino_brasil/` (~30 módulos), `scripts/` (7 entrypoints), `tests/` (3 arquivos), `configs/project.yaml`, `docs/METODOLOGIA.md`, `docs/LEGADO/ARQUITETURA_RN_NINO.md`, `docs/LEGADO/escopo_nino.md`. Total ~8.600 linhas de Python + ~1.200 linhas de documentação técnica.

---

## Sumário executivo

O projeto está acima da média de código científico em dois aspectos críticos: disciplina anti-vazamento (climatologia e percentis ajustados apenas no bloco de treino, com teste de regressão cobrindo isso) e ingestão idempotente com ledger de auditoria. A arquitetura de fases é coerente com o estado da arte em previsão de ENSO e teleconexões. Os maiores riscos são de engenharia de reprodutibilidade — não de ciência — e há um gargalo de performance na Fase 5 que pode multiplicar por cinco o tempo de execução. A seção científica ao final deste parecer aborda o que as Fases 3, 4 e 5 têm condições reais de revelar sobre El Niño e o Brasil.

---

## PILAR 1 — EFICIÊNCIA E PERFORMANCE

### 1.1 `run_walk_forward` reconstrói a matriz X 5× por (fold, lag) — CRÍTICO

**Diagnóstico.** Em `walk_forward.py` (linhas ~573–631), para cada fold e cada lag o código chama `build_feature_matrix` uma vez para regressão e mais uma vez para cada um dos 4 eventos de precipitação (`dry_extreme`, `below_p25`, `above_p75`, `wet_extreme`). O lado X — alinhamento temporal, agregação espacial de todos os preditores — é idêntico nas 5 chamadas. Só o `y` muda.

**Impacto.** Com a configuração oficial (24 lags × ~20 folds em 40 anos × 5 alvos), são ~2.400 reconstruções de matriz, sendo ~80% trabalho redundante. Na estratégia `mean` isso custa horas; na `flatten` custa memória e horas multiplicadas pelo número de pixels. É o gargalo dominante da Fase 5.

**Refatoração sugerida:**

```python
# build_predictor_matrix: constrói X uma única vez por (fold, lag)
X_matrix = build_predictor_matrix(fold_predictors, lag_days=lag, ...)

# align_target: operação O(n), rápida
for name, target in {"anomaly": target_anomaly, **event_targets}.items():
    y = align_target(target, X_matrix.index, lag_days=lag)
    _fit_evaluate(X_matrix, y, ...)
```

### 1.2 Quatro passadas de quantil independentes por fold onde uma resolve

**Diagnóstico.** `local_percentile_threshold` é chamado 4 vezes (P10, P25, P75, P90) por fold, cada uma disparando `rechunk(time=-1)` + `quantile` sobre o bloco de treino completo.

**Impacto.** O rechunk para chunk único é o passo caro — materializa a série inteira por pixel. Fazê-lo 4 vezes quadruplica I/O e memória de pico por fold.

**Sugestão:**

```python
thresholds = train_sample.quantile([0.10, 0.25, 0.75, 0.90], dim=time_name)
# uso: thresholds.sel(quantile=0.10)
```

### 1.3 Estratégia `flatten` não escala para o Brasil 0,25° — sem mitigação no código

**Diagnóstico.** `dataarray_to_frame` com `flatten` materializa um DataFrame denso. Grade Brasil 0,25° ≈ 169×181 ≈ 30 mil colunas × ~15 mil dias × 8 bytes ≈ **3,6 GB por variável preditora**, antes das cópias internas do sklearn. `permutation_importance` com `n_repeats=5` e 30 mil features é computacionalmente inviável mesmo com hardware bom.

**Impacto.** O README corretamente instrui rodar em recorte pequeno antes de escalar, mas o código não tem caminho de escala: vai estourar memória silenciosamente em produção na Fase 5 pixel-a-pixel.

**Sugestão.** Para o caso pixel-a-pixel, inverter o desenho: treinar um modelo por pixel (ou por bloco de pixels com dask), iterando sobre chunks do Zarr. Desabilitar `permutation_importance` automaticamente quando `n_features > limiar` (ex.: 500), com aviso explícito.

### 1.4 `chunk_plan` com `time=31` penaliza o padrão de acesso da modelagem

**Diagnóstico.** Chunks de 31 dias são ótimos para escrita anual, mas a Fase 5 lê séries temporais completas (1981–hoje): ~530 chunks por variável em 45 anos.

**Sugestão.** Gerar cópia analítica rechunkada (`time=365`, espacial menor) no passo de regrid — que já reescreve o cubo. Documentar a política com justificativa no docstring de `chunk_plan`.

### 1.5 `standardize_dataset_to_daily` paga resample em dados já diários

**Diagnóstico.** Para `source_frequency="daily"` executa `ds.resample(time="1D").mean()` mesmo quando o calendário já é diário — um shuffle completo para uma no-op. OISST e CHIRPS, os maiores volumes do projeto, caem nesse ramo.

**Sugestão.** Verificar frequência inferida com `pd.infer_freq` e retornar cedo para dados já diários e regulares.

### 1.6 Pontos menores de eficiência

`AuditLog.read()` carrega o JSONL inteiro em memória a cada chamada — planejável para a Fase 7 (operação recorrente) com rotação por tamanho. `cmd_status` faz `rglob("*")` sobre pastas com dezenas de milhares de arquivos NetCDF — limitar profundidade ou contar lazy. `_all_zarrs_valid` abre cada store via `dataset_summary` em loops anuais — validar apenas existência + `.zmetadata`.

---

## PILAR 2 — PRÓS E CONTRAS

### Prós (raros em código científico)

**1. Disciplina anti-vazamento com teste de regressão.** Climatologia, desvio-padrão, percentis e o índice Niño 3.4 são ajustados apenas no bloco de treino de cada fold. O teste `test_nino34_fold_feature_ignores_post_train_values` verifica isso perturbando +5°C no período pós-treino e confirmando que o bloco de treino permanece inalterado. É o ponto mais forte do projeto e protege a validade científica dos resultados.

**2. Ingestão idempotente e retomável.** Padrão `.part` + `replace()` atômico, `overwrite` explícito, dry-run em tudo, ledger JSONL com hash SHA-256 opcional. Permite matar e retomar downloads que levam semanas sem corromper estado.

**3. Configuração centralizada com decisões metodológicas declaradas.** `configs/project.yaml` declara grade, lags, percentis, latências por fonte. O README amarra cada decisão à fase correspondente. Isso é raro e valioso para reprodutibilidade acadêmica.

**4. Separação limpa biblioteca/CLI.** `src/nino_brasil/` é uma biblioteca testável; `scripts/` são entrypoints. Módulos curtos e coesos (maioria < 110 linhas), dataclasses `frozen`, type hints modernos.

**5. Fit único para validação e teste.** Em `_fit_evaluate_matrix`, o modelo é treinado uma vez e reavaliado nos dois splits — otimização correta e comentada no código.

### Contras / riscos

**1. Reprodutibilidade frágil — risco nº 1 para uma tese.** `requirements.txt` não pinna nenhuma versão. A stack `xesmf/xarray/zarr/cartopy` é notoriamente sensível a versões (zarr 3.x mudou formato default; xarray muda comportamento de `groupby`/`resample` entre releases). Daqui a um ano, `pip install -r requirements.txt` produzirá um ambiente diferente do que gerou os resultados publicados.

*Correção:* `pip freeze > requirements.lock.txt` (ou `pyproject.toml` + `uv lock`; para xesmf, um `environment.yml` conda é o caminho padrão) e registrar versões dos pacotes-chave no ledger a cada run da pipeline.

**2. Acoplamento a internals não públicos do `cdsapi`.** `_download_cds_result` lê `result.location`, `result.session`, `result.retry_max` via `getattr` — atributos sem contrato de API. Um upgrade do `cdsapi` (não pinnado) quebra o download silenciosamente.

*Correção:* pinnar `cdsapi` e isolar esse código num adaptador com teste de contrato e mensagem explícita de versão incompatível.

**3. Dois monólitos com duplicação estrutural.** `scripts/data_pipeline.py` (1.251 linhas) repete o mesmo bloco `run_or_continue(lambda: ingest_...)` ~12 vezes para ERA5/ORAS em três modos de request. `scripts/curate_and_resume_downloads.py` (607 linhas) **redefine localmente** convenções de caminho (`chirps_raw_path`, `oisst_zarr_path`) que pertencem à biblioteca. Se o layout de `data/` mudar, dois lugares quebram de formas diferentes.

*Correção:* extrair `src/nino_brasil/data/paths.py` (fonte única de convenções de caminho) e uma tabela declarativa de tarefas (lista de dicts: fonte → função, modo, args), reduzindo subcomandos a um loop genérico.

**4. Inconsistência de `zarr_format`.** `netcdf_to_daily_zarr`, `dataframe_to_zarr` e o regrid escrevem `zarr_format=2` explícito, mas `netcdf_to_zarr` e `zip_netcdf_to_zarr` não. Com `zarr>=3` não pinnado, esses dois produzirão stores v3 e o projeto terá formatos mistos.

*Correção:* constante `ZARR_FORMAT = 2` em `zarr_store.py` usada em todos os writers.

**5. Config como dict cru.** `cfg["modeling"]["grid"]["resolution_degrees"]` espalhado pela base; um typo no YAML ou no acesso só falha em runtime, possivelmente horas dentro de um download CDS.

*Correção:* validar o YAML na carga com dataclass ou pydantic — `load_config()` retorna objeto tipado; erros de config falham em segundos.

**6. Segurança — dois pontos específicos.** `.env` corretamente ignorado, secrets mascarados em logs, sem credencial hardcoded. Porém: (a) `zipfile.extractall` sem sanitização de caminhos (zip-slip) em `zip_netcdf_to_zarr` — fontes são confiáveis, risco baixo, correção é uma linha de validação de `Path.resolve()`; (b) `load_local_env` é um parser `.env` artesanal que funciona, mas `python-dotenv` elimina manutenção.

---

## PILAR 3 — DOCUMENTAÇÃO E MANUTENÇÃO

**O que está bom.** README com decisões metodológicas explícitas e painel de fases; `docs/METODOLOGIA.md`, `docs/DATA_SOURCES.md`, `docs/LEGADO/ARQUITETURA_RN_NINO.md`, runbook de downloads (`docs/RUNBOOK_DOWNLOADS.md`), guia WSL/GPU (`docs/SETUP_WSL_GPU.md`). Um desenvolvedor entende o que o projeto faz e como rodar sem fricção. A `docs/METODOLOGIA.md` detalha o fluxo CTD/TEOS-10 e as regras de ablation — raridade em projetos de pesquisa.

**Lacunas em ordem de importância:**

**1. Sem mapa da arquitetura do código.** O README explica o fluxo de dados, mas nada diz "qual módulo é a fonte de verdade de caminhos, de chunking, quem chama quem". Um diagrama de módulos de meia página no README ou em `docs/ARQUITETURA.md` elimina horas de onboarding.

**2. Docstrings insuficientes nos módulos científicos centrais.** `anomalies.py`, `walk_forward.py` e `feature_matrix.py` carregam as decisões metodológicas mais delicadas — janela circular de climatologia, semântica de `lag_days` (o X está em `t`, o Y em `t+lag`), convenção `origin_time`/`target_time` — com docstrings de uma frase e sem parâmetros documentados. Quem alterar `window_days` sem esse contexto quebra a ciência sem quebrar os testes.

**3. Higiene da raiz.** Atualização 2026-06-14: documentos de trabalho, pareceres, metodologia e runbook foram movidos para `docs/`; a raiz fica restrita ao `README.md` e ao `painel_executivo.md` local. PDFs de artigos ficam em `papers/`.

**4. Sem LICENSE e sem `CITATION.cff`.** Para um projeto de pós-graduação que mira publicação na Fase 7, ambos são obrigatórios antes de tornar o repositório público.

**5. Detalhes menores.** O README tem duas seções "### 8." (regridding e distribuições). Comandos longos duplicam o runbook de `docs/`. Idioma misto — docstrings em inglês, prints em português ("aguardando CDS", "erro:") — escolher um padrão.

---

## PILAR 4 — BOAS PRÁTICAS

**1. O pacote não é instalável — cada script faz `sys.path.insert`.** Sete scripts e três arquivos de teste repetem o hack. Sem `pyproject.toml`, IDEs, mypy e qualquer execução fora da raiz falham.

*Correção (maior retorno pelo menor esforço):*

```toml
# pyproject.toml
[project]
name = "nino-brasil"
version = "0.1.0"
requires-python = ">=3.12"

[tool.setuptools.packages.find]
where = ["src"]
```

`pip install -e .` e remover todos os blocos `sys.path`.

**2. Sem linter, formatter nem CI.** Sem ruff/black/flake8, sem pre-commit, sem `.github/workflows`. Sintoma visível: `T = TypeVar("T")` no meio do bloco de imports de `data_pipeline.py` (linhas 58–59). Um `ruff.toml` + GitHub Actions rodando lint + testes a cada push custa 30 minutos e é especialmente valioso num projeto multi-máquina.

**3. `print` em vez de `logging`.** Para pipelines que rodam dias, ausência de timestamps e níveis torna diagnóstico pós-falha arqueologia. O ledger de auditoria compensa parcialmente. Migrar para `logging` com `%(asctime)s %(levelname)s %(name)s`, mantendo tqdm para progresso visual.

**4. Cobertura de testes desbalanceada.** ~330 linhas de teste para ~8.600 de código. A escolha de focar no núcleo de modelagem é correta (é onde mora o risco científico). Mas ingestão/ETL (~3.400 linhas) tem zero testes. Adicionar pelo menos: convenções de caminho, `_all_zarrs_valid`, política de retomada com arquivos `.part` simulados, `standardize_dataset_to_daily` para cada frequência. Padronizar o runner: o repo tem `.pytest_cache` mas o README manda `unittest discover` — escolher pytest.

**5. Acertos a registrar.** Nomenclatura PEP 8 consistente. Módulos da biblioteca curtos e coesos. Dataclasses imutáveis. Exceções específicas capturadas. `KeyboardInterrupt`/`SystemExit` re-levantados corretamente em `run_or_continue`. Comentários que explicam *porquês* (pad circular em `_smooth_dayofyear_stat`, fit único reaproveitado para dois splits). SOLID respeitado na biblioteca — é nos scripts que se perde.

---

## SEÇÃO CIENTÍFICA — O QUE AS FASES 3, 4 E 5 TÊM CONDIÇÕES DE REVELAR

Nota metodológica fundamental: esta seção não antecipa resultados nem prescreve o que o projeto vai confirmar. O propósito das Fases 3, 4 e 5 é medir — a partir dos dados, sem hipótese de chegada fixada. As perguntas abaixo são formuladas como problemas de mensuração abertos. Qualquer afirmação sobre lags dominantes, mecanismos ou hierarquias de variáveis só existe depois que os experimentos rodarem.

---

### Q1 — Quais variáveis do Niño 3.4 influenciam a precipitação (seca e extremos de chuva) no Nordeste e no Sul do Brasil?

**O que o projeto vai medir.** A Fase 4 (correlação defasada por lag semanal + ablation de variáveis individuais) e a Fase 5 (walk-forward com permutation importance e SHAP por grupo) produzirão um ranking empírico de quais variáveis — SSTA, OHC 0–300m, OHC 0–700m, D20, termoclina, SLP, u850, v850, u200, z500, TCWV — têm importância mensurável para cada região (NEB, Sul) em cada lag testado (7 a 168 dias, passo semanal) e em cada estação.

**O que não se sabe e o projeto vai descobrir:**
- Qual variável do Niño 3.4 carrega o sinal mais forte para o NEB — se é superfície (SSTA), subsuperfície (OHC/D20), ou uma combinação atmosférica (SLP, vento).
- Se o Sul do Brasil responde a variáveis diferentes do NEB, ou aos mesmos preditores com lags distintos.
- Se há lags em que nenhuma variável tem skill acima do baseline de persistência — o que seria um resultado científico igualmente importante.
- Se o sinal é estável ao longo das décadas (1981–2026) ou se há deriva (mudança climática modificando a teleconexão).

**O que o código entrega para responder isso.** O `walk_forward_group_weights.zarr` contém a importância normalizada por grupo (`ocean`, `atmosphere`, `calendar`) para cada combinação (fold, split, lag, modelo, evento). A Fase 4 adiciona correlação defasada pixel-a-pixel, que é o diagnóstico mais direto sem modelo. Os dois juntos permitem triangulação: se correlação defasada e SHAP concordam em qual lag/variável domina, o sinal é robusto; se discordam, há não-linearidade que a correlação não captura.

**Limitação de design relevante.** O índice Niño 3.4 como feature `mean` colapsa a estrutura espacial do Pacífico equatorial num escalar. Para distinguir eventos do Pacífico Leste (EP) vs Central (CP/Modoki), que podem ter impactos distintos no Brasil, seria necessário incluir Niño 1+2 e Niño 4 como features separadas. Isso está previsto na `docs/LEGADO/ARQUITETURA_RN_NINO.md §2.1` mas não está implementado na Fase 5 ainda — é uma extensão natural após o resultado inicial.

---

### Q2 — Qual é o peso de cada variável independente e o peso das variáveis acopladas (oceano + atmosfera)?

**O que o projeto vai medir.** O design de ablation da `docs/METODOLOGIA.md §7` (Modelos A–F) é o experimento formal para esta pergunta:

- Modelo A: apenas oceânico → skill baseline oceânico
- Modelo B: apenas atmosférico → skill baseline atmosférico
- Modelo C: oceânico + atmosférico → skill do acoplamento
- Modelo D: sem subsuperfície oceânica → ganho/perda do OHC/D20/termoclina
- Modelo E: sem altos níveis atmosféricos → contribuição de u200/v200/z200/div200
- Modelo F: sem umidade atmosférica → contribuição de TCWV

O ganho do Modelo C sobre A e sobre B, por lag e por região, é a medida empírica do que o acoplamento acrescenta. Se Modelo C ≈ Modelo A para todos os lags no NEB, o acoplamento não ajuda lá. Se Modelo C >> A e B separados, há interação não-aditiva entre oceano e atmosfera que só aparece combinados.

**O que o código precisa para executar isso.** O `model_pipeline.py` atual recebe predictor Zarrs via `--predictor-zarr` como lista — já suporta múltiplos Zarrs. O que falta é a convenção para nomear os experimentos A–F e um script wrapper que execute os 6 modelos sistematicamente e gere a tabela comparativa. Isso é uma extensão de ~50 linhas sobre a infraestrutura existente.

**Para variáveis acopladas explícitas.** Se o projeto quiser testar interações (ex.: SSTA × SLP como feature derivada), o `feature_groups` em `feature_matrix.py` suporta declarar um grupo `"coupled"` via o dicionário passado para `build_feature_matrix`. Não há convenção estabelecida ainda — mas a infraestrutura está pronta.

---

### Q3 — O volume das águas aquecidas na termoclina (OHC/D20) é mais importante que apenas a SST do Niño 3.4?

**O que o projeto vai medir.** A comparação Modelo C (completo) vs Modelo D (sem subsuperfície) em walk-forward, estratificada por lag, responde diretamente se OHC e D20 acrescentam skill mensurável além da SST de superfície — e em quais lags isso acontece ou não acontece. O resultado pode ser qualquer coisa: subsuperfície dominante em todos os lags, dominante apenas em lags longos, indiferente, ou até prejudicial por ruído.

**O que o código entrega.** O módulo `ocean_heat.py` calcula `OHC = ρ₀ × cp × ∫T(z)dz` com constantes fixas de referência (`RHO0=1025`, `CP0=3990`). Para comparação de anomalias entre eventos históricos as constantes fixas são válidas. Se o projeto quiser OHC absoluto para balanço de energia, seria necessário TEOS-10 completo via `gsw` — mas para o objetivo de ranking de importância entre variáveis, a aproximação é suficiente.

**Caveat de implementação que deve aparecer nos resultados.** O ORAS5 é mensal, não diário. O código trata isso como memória oceânica com lag explícito via forward-fill (`monthly_ocean_policy` em `configs/project.yaml`). Isso é adequado para capturar inércia da termoclina — que muda em escala de semanas a meses — mas a variação diária do OHC nos stores Zarr é artefato de interpolação, não sinal físico. Os artigos derivados deste projeto devem declarar isso explicitamente na seção de dados.

**Para CTD/WOD.** Os perfis CTD com `max_gradient_thermocline_depth` e `d20_depth` fornecem validação independente do ORAS5 — a pergunta "ORAS5 concorda com observação in situ no Niño 3.4?" é a Fase 3. Se houver divergência sistemática, o peso dado ao ORAS5 nos modelos precisa ser interpretado com cautela.

---

### Q4 — Como funciona a formação dos super El Niño em relação a slope, duração, difusão de calor, e lag de atuação no NEB e no Sul?

Esta pergunta tem duas camadas que o código trata separadamente e que precisam ser mantidas separadas.

**Camada 1 — Diagnóstico físico do Niño 3.4 (Fase 3).** Os stores `nino34_physical_signal.zarr`, `nino34_thermocline_diagnostics.zarr` e `nino34_peak_signal_comparison.zarr` caracterizam *o evento no Pacífico*: como a anomalia de SST cresce (slope longitudinal = derivada espacial da SSTA no eixo leste-oeste), como a termoclina desce ou sobe (slope vertical = `max_gradient_thermocline_depth` ao longo do tempo), por quanto tempo o sinal persiste acima de um limiar, e como os eventos de 1982–83, 1997–98, 2015–16 e 2023–24 se comparam entre si nessas dimensões.

O que o projeto vai medir aqui: a assinatura de cada evento em três dimensões simultâneas — intensidade de superfície (SSTA), profundidade da termoclina (D20), e conteúdo de calor integrado (OHC). A comparação entre super El Niños e eventos fracos pode revelar se há limiar de slope ou de OHC acima do qual o impacto no Brasil muda de caráter, ou se a relação é contínua.

**Camada 2 — Difusão para o Brasil (Fases 4 e 5).** Esta é a parte que o projeto ainda não sabe e vai descobrir. O sinal físico caracterizado na Fase 3 se propaga do Pacífico para o Brasil por alguma combinação de caminhos — atmosféricos, oceânicos, diretos, mediados — com algum conjunto de lags. O projeto não sabe de antemão quais caminhos dominam, quais lags emergem, nem se o comportamento é estável entre décadas ou varia conforme o tipo de evento. A janela de 24 lags semanais (7–168 dias) foi escolhida exatamente para não assumir nada: o dado vai dizer onde o skill é alto.

**O que não deve ser fixado antes de rodar.** Não se deve assumir que o caminho atmosférico é mais rápido que o oceânico, nem que o NEB e o Sul têm lags diferentes, nem que Walker/Hadley é o mecanismo dominante. Essas são hipóteses que entram como variáveis a medir — via distribuição de importâncias por grupo atmosférico vs oceânico em função do lag — e não como premissas de design.

**Implicação para o design do `nino34_signal_slope_duration.zarr`.** O store de slope e duração deve registrar os valores brutos (slope calculado, duração acima de limiar, OHC integrado no período) *sem* rotular qual mecanismo eles implicam. A interpretação física vem depois, cruzando os diagnósticos da Fase 3 com os pesos da Fase 5.

---

### Q5 — El Niño pode ser caracterizado via heatwave OISST?

**A pergunta de mensuração.** Um Marine Heatwave (MHW) no Niño 3.4 — período em que a SST excede o P90 da climatologia local por ≥5 dias consecutivos — é uma forma de caracterizar El Niño em termos operacionais, usando apenas dados de satélite de alta resolução temporal (OISST diário, 0,25°).

**O que o projeto já tem.** Os módulos `daily_anomaly` + `local_percentile_threshold` + `event_mask` são os ingredientes de um detector de MHW: anomalia de SST com climatologia calculada no treino, limiar local (P90), máscara de evento. A lógica é idêntica à usada para os eventos de precipitação no Brasil — só o domínio espacial muda. O código suporta isso sem modificação estrutural.

**O que falta implementar.** Um módulo `src/nino_brasil/features/marine_heatwave.py` adicionando rastreamento de evento (onset, peak, end, duração contínua, intensidade máxima e média), que os módulos atuais não calculam individualmente. Esboço:

```python
def detect_mhw(
    sst: xr.DataArray,
    percentile: float = 90.0,
    min_duration_days: int = 5,
    train_times: xr.DataArray | None = None,
    time_name: str = "time",
) -> xr.Dataset:
    """
    Detecta Marine Heatwaves no OISST (Hobday et al. 2016).
    Retorna onset, peak, end, max_intensity, mean_intensity, duration por pixel.
    Climatologia calculada apenas no train_times (anti-vazamento).
    """
```

**O que o projeto vai descobrir.** MHW no Niño 3.4 e El Niño são conceitos relacionados mas não idênticos: pode haver MHW local sem o padrão de grande escala ativo, e pode haver eventos El Niño moderados sem exceder P90 local. O projeto medirá a sobreposição empírica entre as duas definições na série histórica 1981–2026 — comparando onset/peak do MHW OISST com os anos de pico declarados em `configs/project.yaml` (`el_nino_peak_years: [1982, 1983, 1997, 1998, 2015, 2016, 2023, 2024]`) e com a evolução de OHC e D20 da Fase 3.

Cruzar MHW × D20 × SLP num único análise de co-ocorrência seria uma contribuição metodológica nova e publicável — factível com a infraestrutura atual.

---

### Q6 — Melhores técnicas de ML e Redes Neurais para hardware de baixo custo

O projeto roda em hardware pessoal (Windows + WSL, GPU mencionada no `docs/SETUP_WSL_GPU.md`). As escolhas abaixo priorizam rodar os experimentos completos — todos os 24 lags × 20 folds × ablation — dentro de prazo razoável.

#### ML Clássico (Fases 4 e 5)

| Modelo | Papel | Por que serve | Custo estimado |
|--------|-------|---------------|----------------|
| Ridge Regression | Baseline linear, coeficientes interpretáveis | Coeficientes são pesos físicos diretos; não overfita | Segundos por fold |
| LightGBM | Modelo principal de produção | 3–10× mais rápido que XGBoost, memória menor, GPU opcional | Minutos por fold |
| Random Forest | Ensemble de referência | Paraleliza em CPU; menos sensível a hiperparâmetros | Minutos por fold |
| Correlação defasada | Diagnóstico Fase 4 | Sem treino, gera mapas de lag diretamente, interpretável | Segundos total |

**Mudança recomendada imediata:** mover LightGBM para o topo da lista padrão em `walk_forward.py` em vez de XGBoost. O código já suporta os dois (`_make_regressor`/`_make_classifier`). Com 24 lags × 20 folds × 4 modelos de ablation, LightGBM reduz de ~2 dias para ~5–6 horas em CPU quad-core sem diferença de performance mensurável para este tipo de dado.

#### Diagnóstico de distribuição (já implementado, não mudar)

`distributions.py` com power law vs lognormal vs exponencial via KS é o correto para chuva extrema e OHC. Nenhuma alteração necessária.

#### Redes Neurais (Fase 6) — hierarquia para hardware limitado

**Nível 1 — GPU de consumidor (RTX 3060–4070, 8–12 GB VRAM):**

*ConvLSTM (PyTorch Lightning).* Já está no design. Janela de entrada de 12 meses × variáveis × grade recortada do Niño 3.4 (≈ 60×80 pixels a 0,25°) cabe em 8 GB. Custo de treino: 2–4 horas por experimento. Implementação via `pytorch-lightning` já em `requirements.txt`.

*1D-Transformer sobre série temporal agregada.* Se VRAM for o limitante: série temporal escalar (índice + OHC + SLP) como sequência de tokens. 4 camadas / 8 heads sobre janelas de 365 dias cabe em 4 GB, treina em 1–2 horas. Alternativa ao ConvLSTM quando a memória é o gargalo.

**Nível 2 — GPU de consumidor, mais esforço:**

*U-Net (Encoder-Decoder).* Para o Modelo B da arquitetura em cascata: entrada = estado ENSO previsto; saída = campo 2D de probabilidade de precipitação no Brasil. Grade recortada Brasil 0,25° (≈ 169×181 pixels). U-Net shallow (3 níveis, 32/64/128 filtros): 4–8 horas de treino em RTX 3060. Inferência < 1 segundo.

**Nível 3 — Não recomendado para este hardware:**

*ClimaX / Pangu-Weather / 3D-Geoformer.* Modelos SOTA que exigem pré-treino em CMIP6 em clusters A100. Para publicação acadêmica em hardware próprio, o custo-benefício não fecha. A estratégia correta é citar esses modelos na seção de trabalhos relacionados e comparar skill, não tentar reimplementá-los.

#### XAI para hardware limitado

SHAP TreeExplainer (LightGBM/XGBoost) roda em CPU, é O(n_features × n_samples), e já está implementado em `shap_importance_frame` com `shap_max_rows` controlável. É o método de maior retorno por hora de CPU para este projeto.

Para as redes neurais da Fase 6, **occlusion maps** (mascarar sistematicamente uma sub-região do Pacífico e medir a queda de skill) são baratos de implementar e fisicamente interpretáveis — mostram diretamente quais regiões do Pacífico o modelo usa. Gradient-based saliency (GradCAM, Integrated Gradients) é mais caro e mais difícil de interpretar fisicamente para dados climáticos.

#### Sequência recomendada para hardware limitado

```
Fase 4  →  lag correlation + PCA/EOF              (CPU, sem treino, dias)
Fase 5  →  Ridge + LightGBM + ablation A–F + SHAP (CPU, semanas)
Fase 6a →  ConvLSTM ou 1D-Transformer             (GPU, ~1 semana por experimento)
Fase 6b →  U-Net downscaling Brasil               (GPU, ~1 semana)
Fase 6c →  comparação e publicação                (análise, sem novo treino)
```

A Fase 5 já gera resultados publicáveis sem redes neurais. Publicar os resultados do ablation A–F antes de iniciar a Fase 6 é a estratégia correta para um projeto acadêmico.

---

## Plano de ação priorizado

| # | Ação | Esforço | Retorno |
|---|------|---------|---------|
| 1 | Pinnar dependências (lock/environment.yml) + registrar versões no ledger | baixo | reprodutibilidade da tese |
| 2 | `pyproject.toml` + `pip install -e .`, remover hacks de `sys.path` | baixo | base para tudo abaixo |
| 3 | Refatorar `run_walk_forward`: X construído 1× por (fold, lag); quantis em chamada única | médio | ~5× na Fase 5 |
| 4 | Constante `ZARR_FORMAT=2` em todos os writers + pinnar zarr | trivial | evita corrupção de formato |
| 5 | Substituir XGBoost por LightGBM como padrão na Fase 5 | trivial | ~3× de speedup imediato |
| 6 | Implementar `nino34_physical_signal` completo na Fase 3: slope longitudinal, slope vertical, duração, comparação histórica — sem rotular mecanismo | médio | base para cruzar com pesos da Fase 5 |
| 7 | Formalizar ablation Modelos A–F como experimentos nomeados em `model_pipeline.py`; wrapper que roda todos e gera tabela comparativa | médio | Q2 e Q3 respondidas empiricamente |
| 8 | Extrair `paths.py`, desduplicar `curate_and_resume_downloads.py` | médio | manutenção |
| 9 | ruff + CI (lint + testes a cada push) | baixo | congela qualidade |
| 10 | Guarda de memória/feature-count na estratégia `flatten`; desenho por blocos de pixels com dask | alto | viabiliza Fase 5 pixel-a-pixel |
| 11 | LICENSE + `CITATION.cff` + versionar documentos de trabalho na raiz | baixo | maturidade para publicação |
