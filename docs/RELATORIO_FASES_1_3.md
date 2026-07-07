# Relatório resumido — Fases 1 a 3 (NINO-BRASIL)

**Projeto NINO-BRASIL · Oceanografia Física UFPE · Thiago Vilar**
**Data:** 2026-07-07 · **Escopo deste relatório:** revisão, higienização, reexecução e análise das Fases 1, 2 e 3.

Este relatório consolida a reexecução das Fases 1 a 3 a partir dos dados já baixados localmente, com foco no que cada etapa responde cientificamente e nas justificativas das soluções adotadas na organização dos dados. A Fase 4 (triagem estatística chuva–ENOS) permanece deliberadamente pausada: por decisão de escopo, ela só será retomada quando as Fases 1 a 3 estiverem auditadas sem erros conhecidos — condição que este trabalho verifica.

## 1. Situação executiva

| Fase | O que responde | Estado | Evidência auditável |
|---|---|---|---|
| 1 — Base local e ingestão | Temos dados suficientes, reprodutíveis e com procedência para diagnosticar o Pacífico? | Concluída | CHIRPS/OISST cobrem 1981–2026; ERA5 processado cobre 1981–2025 sem lacunas internas, com disponibilidade operacional indicada para 2026 conforme latência; oceano UFS/GLORYS12/ORAS5 e in situ locais |
| 2 — Padronização, anomalias, Zarr/regrid | Os dados brutos viram cubos comparáveis, sem vazamento de climatologia e com fontes emendadas de forma honesta? | Concluída | Auditoria oceânica `status=complete`, `errors=0`; 51 linhas de auditoria de transição de fonte |
| 3 — Diagnóstico físico do Niño 3.4 | Como o sinal do Niño 3.4 nasce, cresce, atinge pico e decai, e quais relações físicas são defensáveis sem rótulo externo? | Concluída e reexecutada | `phase3_diagnostics_audit.json` com `errors: []`; 16.353 dias, 538 meses, 12 eventos, 11 picos P90 |

Qualidade transversal verificada nesta rodada: **54 testes automatizados passaram** (features, estatística, saídas numéricas, modelagem, oceano diário/mensal, zarr store), auditoria oceânica da Fase 2 recomposta sem erros e auditoria da Fase 3 recomposta sem erros.

## 2. Higienização executada

O repositório estava com o worktree sujo (45 mudanças pendentes, incluindo um `index.lock` preso) e a documentação descrevia a Fase 3 como "etapa ativa/limite do escopo", em conflito com o cronograma real. Foram adotadas as seguintes soluções, organizadas em sete commits temáticos na branch `main` local (sem push):

O primeiro passo foi separar o que é código-fonte do que é dado ou artefato. Literatura (`papers/`, ~100 MB) e temporários (`tmp/`) passaram para o `.gitignore`, assim como a configuração local do editor (`.vscode/`); os dois pareceres `.docx` foram movidos para `docs/PARECERES/`. O código da Fase 3 (diagnósticos físicos, DHW, referência de eventos/P90 e o módulo estatístico com FDR de Benjamini-Hochberg e graus de liberdade efetivos) foi commitado junto da remoção do `download_noaa_psl` do fluxo ativo, formalizando a regra de que rótulos NOAA/PSL entram apenas como comparação visual, nunca como entrada. A documentação foi realinhada num commit próprio (cronograma com gates G1–G4, metodologia da Fase 4, pareceres e o painel refletindo a auditoria da Fase 3). Testes e o andaime da Fase 4 (notebooks e scripts) foram versionados em commits separados, este último rotulado explicitamente como **PAUSADO** para rastreabilidade sem sugerir resultado científico.

Coerência de fases foi corrigida na fonte: `scripts/update_painel_executivo.py` agora lê `data/audit/phase3_diagnostics_audit.json` e marca a Fase 3 como concluída quando a auditoria não tem erros, em vez de texto fixo; `README.md` e o painel deixaram de tratar a Fase 3 como escopo terminal e passaram a descrever as Fases 4–8 como planejadas atrás dos gates.

## 3. Fase 1 — Base local e ingestão

**Pergunta:** a base de dados sustenta um diagnóstico físico do Pacífico equatorial, com procedência e reprodutibilidade?

**O que foi feito e por quê.** A ingestão separa superfície/atmosfera (que sustentam 1981–presente) da subsuperfície (que tem janelas reais por fonte). CHIRPS (chuva) e OISST (SST/SSTA diária) cobrem 1981–2026 sem lacunas internas; o ERA5 processado (superfície e níveis de pressão) cobre 1981–2025 fechado, com disponibilidade operacional de 2026 acompanhada conforme a latência da fonte. A memória subsuperficial é emendada de forma explícita: NOAA UFS como ponte histórica 1981–1992, GLORYS12 diário como fonte principal desde 1993 e uma cauda operacional GLO12 apenas de análise (previsões excluídas); ORAS5 mensal entra como memória independente e nunca é promovido a observação diária. As validações in situ (CTD/WOD, TAO/TRITON, Argo) preservam suas lacunas reais em vez de preenchê-las artificialmente.

**Justificativa da solução.** A decisão central da Fase 1 foi recusar a narrativa de "cobertura homogênea desde 1981". Vender subsuperfície contínua desde 1981 seria cientificamente falso porque GLORYS12 começa em 1993; por isso toda análise subsuperficial deve reportar sensibilidade 1993+ e 2000+. Lacunas do CTD (anos como 1993, 1998, 2004–2005, 2013–2014, 2017, 2020–2022, 2024–2025) ficam registradas, não imputadas.

## 4. Fase 2 — Padronização, anomalias e regrid

**Pergunta:** os dados brutos heterogêneos viram cubos comparáveis na mesma grade e escala temporal, sem vazamento e com as emendas de fonte auditadas?

**O que foi feito e por quê.** Os produtos foram padronizados para grade comum 0,25° e reconciliados em cubos Zarr. A climatologia de referência usa a base 1991–2020 apenas como descrição; qualquer climatologia com uso preditivo é ajustada dentro do treino de cada fold, evitando vazamento. A auditoria oceânica integrada (`audit_ocean_phase2.py`) foi reexecutada nesta rodada e retornou `status=complete` com `errors=0`, cobrindo continuidade das séries, D20/OHC/WWV/tilt e as três transições de fonte (UFS→GLORYS nos anos de sobreposição 1993–1995 e GLORYS multiyear→operacional), com 51 linhas de auditoria de transição registradas.

**Justificativa da solução.** Tratar OHC, temperaturas de 50–700 m, D20, WWV e termoclina como um mesmo bloco físico de recarga — em vez de variáveis independentes — é o que impede dupla contagem quando esses indicadores medem o mesmo estado subsuperficial. A auditoria de transição existe justamente para que a emenda de fontes com resoluções e vieses diferentes não seja lida como um sinal físico espúrio na fronteira entre elas.

## 5. Fase 3 — Diagnóstico físico do Niño 3.4

**Pergunta:** como o sinal do Niño 3.4 se forma, quanto tempo guarda memória e quais relações físicas são defensáveis num parecer auditável — tudo derivado da própria SST OISST baixada, sem rótulo ENSO externo e sem ML?

**Reexecução (2026-07-07).** Os cinco estágios foram regerados na ordem canônica: índice diário Niño 3.4, referência mensal de SST, picos P90, diagnósticos físicos e auditoria. Resultados:

A trajetória diária cobre **16.353 dias (1981-09-01 a 2026-06-09)** e a referência mensal, **538 meses**. Dela derivam **12 eventos El Niño** (limiar de SSTA de 3 meses ≥ 0,5 °C por 5+ meses), classificados a partir da própria base: três *super* (1982–1983, 1997–1998, 2014–2016), três *strong* (1991–1992, 2009–2010, 2023–2024), dois *moderate* e quatro *weak*. O critério de percentil identifica **11 picos P90** (limiar de 0,992 °C), liderados por 2015-11 (2,79 °C), 1982-12 (2,28 °C) e 1997-11 (2,17 °C). Os diagnósticos físicos acompanham cada evento com D20, anomalia de termoclina, OHC 0–300/0–700 m, duração do sinal e a taxonomia de fases (neutral → onset → peak). A auditoria final (`audit-phase3-diagnostics`) retornou **sem erros**, validando os cinco produtos Zarr/CSV e a figura P90.

**Nota metodológica sobre a reexecução.** O estágio de diagnósticos lê ~50 stores Zarr de features oceânicas diárias. Para reexecutar dentro do ambiente de verificação (sistema de arquivos montado, mais lento), os stores foram materializados uma vez em cache Parquet e o cálculo rodou sobre esse cache — o conteúdo numérico é idêntico ao dos Zarr originais, apenas o I/O foi acelerado. Na máquina de origem, o comando `build-phase3-diagnostics` roda diretamente sobre os Zarr sem esse contorno.

### 5.1 Protocolo 3A-3G implementado e executado (07/07, tarde)

A lacuna apontada em revisão — o protocolo 3A-3F existia só como especificação — foi fechada: `notebooks/fase3/` contém sete notebooks **executados com saídas numéricas, gráficos e mapas**, mais o extra 3G. Cada um declara a pergunta que responde e a metodologia. Resultados-chave (janela comum 1993+, N_eff por autocorrelação, FDR α=0,05):

| NB | Pergunta | Resposta numérica |
|---|---|---|
| 3A | Quais séries descrevem o sistema? | Matriz semanal 2.372 × 14 (SSTA, D20, OHC×2, WWV, tilt, SSH, SSS, ATL3/4, TNA, TSA, DHW, τx-proxy), cobertura ≥98% |
| 3B | Como eventos vivem? Quanta memória? | Taxas por evento; **e-folding = 27 semanas (~6,2 meses)**; composto de pico dos super eventos confirma padrão Pacífico-leste |
| 3C | O que antecede o pico? | Tilt/SSH/OHC lideram (r 0,70–0,77 em 0–6 sem); **D20 melhor em 15 sem; WWV em 20 sem**; mapa lon×lag mostra inclinação oeste→leste |
| 3D | O que sobrevive ao rigor? | N_eff cai de ~1.740 para ~22–33; sobrevivem a FDR+IC95: tilt, SSH, OHC 0-300/0-700, DHW, D20, WWV, τx e ATL4 (lag 25, r=−0,27) |
| 3E | O sinal é estável entre regimes? | **Estáveis:** tilt, SSH, OHC, D20, τx, ATL4. **Instáveis:** WWV (p=0,11 pós-2010, coerente com literatura) e DHW no limiar (p=0,077 pré-2010) |
| 3F | DHW agrega além de SSTA/WWV/OHC? | **Sim em +4 semanas** (parcial r=0,478, p=0,015); marginal em +8; redundante em +12. Hovmöller SSH mostra pulsos Kelvin em 1997/2015/2023 e na janela 2025/26 |
| 3G | Ciclo de vida vs DHW °C-week? | DHW **pica 4–11 semanas após** o pico da SSTA (integrador, não precursor); **DHW_max × SSTA_pico: r=0,975**; classes separam-se pelo calor acumulado (super 13,8–19,1; strong 7,5–11,3; moderate ~3; weak ≤1,1 °C-week) |

**Conjunto defensável do parecer (sobrevive a 3D ∩ 3E):** tilt da termoclina, SSH, OHC 0–300/0–700, D20 (lead ~15 sem), τx e ATL4 como controle. WWV entra com ressalva explícita de instabilidade pós-2010; DHW entra como métrica de severidade acumulada e memória curta (+4 sem), não como precursor de longo lead.

**Correção de código no caminho:** a materialização dos índices atlânticos expôs um bug real em `_select_lon_bounds` (`nino.py`): caixas terminando em 0°E (ATL3) quebravam em grades deslocadas como o OISST global bruto. Corrigido com teste de regressão (`e76100f`). Insumos reproduzíveis via `scripts/fase3_build_inputs.py`.

**Justificativa da solução.** A regra de ouro da Fase 3 é derivar eventos, referência e picos da própria SST/SSTA OISST local, e não importar um índice ENSO oficial. Isso torna o parecer internamente consistente e auditável: cada número tem origem rastreável no dado baixado. Rótulos NOAA/PSL ficam permitidos apenas como comparação visual, evitando circularidade (usar um índice externo para "validar" um sinal que deveria ser medido de forma independente). O corte de significância exige sobreviver a rigor estatístico (N_eff, IC95 de Fisher-z, FDR) e a estabilidade entre subperíodos (1993–2009 vs 2010–presente) antes de qualquer afirmação entrar no parecer.

## 6. O que está bem feito, o que corrigir e o que construir

O que está sólido: a base 1981–2026 é completa e auditável nas variáveis de superfície; as emendas de fonte subsuperficial são explícitas e auditadas em vez de mascaradas; a Fase 3 é internamente consistente e reprodutível, com trilha de auditoria sem erros; e a suíte de 54 testes cobre as rotinas críticas.

O que foi corrigido nesta rodada: a incoerência documental que tratava a Fase 3 como escopo terminal, o worktree sujo com `index.lock` preso, a mistura de literatura pesada e artefatos no controle de versão, e a lógica de status do painel, que agora deriva o estado da Fase 3 da auditoria real.

O que ainda será construído (fora deste escopo, atrás dos gates): a Fase 4 permanece pausada por decisão sua até a validação integral das Fases 1–3; suas saídas legadas (notebooks B–E) existem mas foram produzidas antes das correções metodológicas e não devem ser citadas como resultado científico. As Fases 5–8 (progressão até o pico, teleconexão por clusters, redes neurais e bancada Ham2019) seguem condicionadas aos gates G1–G4 descritos em `docs/CRONOGRAMA.md`.

## 7. Reprodutibilidade

Comandos que regeneram a Fase 3 a partir da base local (na máquina de origem, dentro de `C:\DEV\NINO26`):

```cmd
.venv\Scripts\python scripts\data_pipeline.py build-nino34-daily-index
.venv\Scripts\python scripts\data_pipeline.py build-nino34-sst-reference
.venv\Scripts\python scripts\data_pipeline.py build-nino34-p90-peaks
.venv\Scripts\python scripts\data_pipeline.py build-phase3-diagnostics
.venv\Scripts\python scripts\data_pipeline.py audit-phase3-diagnostics
.venv\Scripts\python scripts\audit_ocean_phase2.py --glorys-my-end 2026-05-26 --opera