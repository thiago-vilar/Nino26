# Cronograma e fases revisados — NINO-BRASIL

**Projeto NINO-BRASIL · Oceanografia Física UFPE · Thiago Vilar**
**Revisão:** 2026-06-25 · **Base:** estado real em disco + pareceres de orientação (parcial e da orientadora, 23/06/2026).

Este documento realinha as fases ao que **existe em disco** (não ao que a documentação promete) e fixa um cronograma de execução com dependências e *gates* de decisão. Ele substitui a leitura otimista do `painel_executivo.md`, que ainda chamava a Fase 3 de "próxima etapa".

---

## 1. Estado real por fase (evidência em disco)

| Fase | Título | Estado real | Evidência |
|---|---|---|---|
| 1 | Base local e ingestão bruta | **Concluída** | CHIRPS, OISST, ERA5, oceano UFS/GLORYS12, ORAS5, validação in situ locais. |
| 2 | Padronização, anomalias, lags e regrid | **Concluída** | Cubos Zarr reconciliados; auditoria oceânica `status=complete`. |
| 3 | Diagnóstico físico do sinal Niño 3.4 | **Concluída** (corrigir o painel) | Feature stores físicos do Niño 3.4 + `noaa_psl_nino34_p90_peaks` em disco. |
| 4 | Triagem estatística (pixel→região→multivariado) | **Não iniciada** | `data/processed/zarr/statistics/` vazia. **Prioridade nº 1.** |
| 5A | Progressão diária até o pico ENOS | **Parcial** | `enso_peak_progression.zarr` existe; falta walk-forward fechado. |
| 5B | Niño 3.4 → clusters de pixels no Brasil | **Não iniciada** | sem `nino34_cluster_progression.zarr`. |
| 6A/6B/6C | Redes neurais nativas + XAI | **Não iniciada** | sem PyTorch/Lightning no repo; documentação à frente do código. |
| 7 | Publicação e operação | **Esqueleto** | `docs/` e `web/build_site.py` existem; sem produto fechado. |
| 8 | Exploração Ham2019 (isolada) | **Não iniciada** | pesos externos não baixados. |

**Correção de coerência imediata:** rodar `update_painel_executivo.py` para marcar a Fase 3 como concluída e as Fases 4–8 como planejadas/parciais antes de qualquer relatório de qualificação.

---

## 2. Fases revisadas (definição)

A mudança estrutural desta revisão está na **Fase 4**, que deixa de ser uma triagem de índices em duas caixas à mão e passa a ser uma triagem hierárquica **pixel → região → multivariado**, separando explicitamente as duas perguntas da tese (detalhe completo em [METODOLOGIA_FASE4.md](METODOLOGIA_FASE4.md)). As Fases 4, 5 e 6 são independentes em objetivo, método e conclusão: a sequência abaixo é logística/operacional, não uma dependência científica em que uma fase valida a outra. As demais fases mantêm o desenho já refatorado (5A/5B, 6A/6B/6C, 8 isolada).

| Fase | Foco | Marco |
|---|---|---|
| 4 | **4A** regionalização da chuva · **4B** correlação/regressão defasada pixel-a-pixel e por cluster com FDR/GL efetivos · **4C** modos acoplados EOF/MCA/SVD/CCA · **4D** atribuição Pacífico vs Atlântico, composições, estabilidade e gate para ML | Evidência estatística para a teleconexão El Niño → Brasil, com regiões-alvo, mapas, modos acoplados, atribuição parcial e tabela de variáveis para Fases 5–6. |
| 5A | Progressão diária OISST/SSTA → pico El Niño/Super El Niño | Walk-forward que supera (ou não) climatologia e persistência. |
| 5B | Estado do Niño 3.4 → eventos por clusters de pixels (regiões da 4B) | Store de progressão por cluster + métricas por cluster/lag/estação. |
| 6A | Encoder CNN espacial nativo (baseline neural) | Treino nativo, sem pesos externos. |
| 6B | Memória espaço-temporal da progressão ENOS | ConvLSTM/TCN/1D-Transformer; processo físico vs memorização de anos fortes. |
| 6C | Decoder neural de teleconexões → clusters/P90 Brasil | Métricas P90/P10 por cluster, lag e estação. |
| 7 | Publicação, operação, recalibração recorrente | Painel público + rotina atualizável. |
| 8 | Ham2019 exploratório (bancada isolada) | Inventário de pesos, inferência congelada, skill comparativo. |

---

## 3. Cronograma de execução

Datas indicativas a partir de hoje (2026-06-25), em blocos semanais. A ordem busca reduzir retrabalho e reaproveitar bases comuns, mas as Fases 4, 5 e 6 preservam entregáveis independentes. A regra herdada do projeto continua sendo não publicar figura ou conclusão sem saída numérica rastreável.

| Bloco | Janela | Duração | Depende de | Entregável / gate |
|---|---|---|---|---|
| **Sprint 0 — Higiene** | 25–27 jun 2026 | 3 dias | — | Commit dos 4 arquivos de lógica (`data_pipeline.py`, `download_noaa_psl.py`, `phase3_diagnostics.py`, teste numérico); painel realinhado. |
| **4A** Regionalização da chuva | 29 jun – 10 jul | ~2 sem | Fase 3 + CHIRPS | Clusters/regiões-alvo CHIRPS, EOF/REOF e mapa de regiões. |
| **4B** Correlação/regressão defasada | 13 jul – 31 jul | ~3 sem | 4A | Atlas Zarr/GeoTIFF (r, slope, MLR, R² ajustado, ΔR², VIF, p, FDR, lag/estação) em todo o Brasil 0.25° e por cluster. |
| **4C** Modos acoplados | 3 ago – 14 ago | ~2 sem | 4A, 4B | EOF do campo de chuva, EOF SST Pacífico+Atlântico, MCA/SVD/CCA e seleção de variáveis independentes. |
| **4D** Atribuição/composições/gate | 17 ago – 4 set | ~3 sem | 4A–4C | **Marco Fase 4:** Pacífico vs Atlântico, composições ENSO, estabilidade temporal e tabela de variáveis para ML. *Gate G1: a triagem mostra sinal defensável?* |
| **5A** Progressão ENOS | 7 set – 2 out | ~4 sem | Fase 3; compara com 4D quando existir | Walk-forward fechado. *Gate G2: supera climatologia + persistência?* |
| **5B** Niño 3.4 → clusters | 5 out – 6 nov | ~5 sem | Fase 3/chuva; pode comparar ou reutilizar regiões da 4B | Store de progressão por cluster + métricas por cluster/lag/estação. |
| **Revisão de gate da Fase 5** | 9–13 nov | 1 sem | 5A, 5B | *Gate G3: a Fase 5 supera os baselines? Se não, reavaliar antes da Fase 6.* |
| **Setup neural (WSL/GPU)** | 16–20 nov | 1 sem | G3 ✓ | Ambiente PyTorch/Lightning (`docs/SETUP_WSL_GPU.md`). |
| **6A** CNN espacial | 23 nov – 11 dez | ~3 sem | setup | Baseline neural nativo. |
| **6B** Memória espaço-temporal | 14 dez – 15 jan 2027 | ~4 sem | 6A | Modelo de progressão; processo físico vs memorização. |
| **6C** Decoder teleconexões | 18 jan – 12 fev 2027 | ~4 sem | 6B; compara com 5B e com evidências da 4B/4D | Métricas P90/P10 por cluster. *Gate G4: 6A/6B/6C superam a Fase 5? Senão, fallback CMIP6.* |
| **7** Publicação e operação | parcial desde a Fase 4; consolida 16–27 fev 2027 | ~2 sem | 6C | Painel público + rotina de recalibração. |
| **8** Ham2019 exploratório | oportunístico, mar 2027 | ~2 sem | independente | Bancada isolada; nunca alimenta a Fase 6. |

**Caminho operacional recomendado:** 4A → 4B → 4C → 4D; depois 5A/5B e 6A/6B/6C podem ser avaliadas como linhas independentes, comparadas por métricas comuns. A Fase 7 (painel) começa cedo e em baixa intensidade, usando saídas numéricas das fases disponíveis. A Fase 8 é desacoplada e só ocorre se houver folga.

---

## 4. Gates de decisão (salvaguardas)

| Gate | Quando | Critério | Se falhar |
|---|---|---|---|
| **G1** | fim da 4D | A triagem revela associações significativas por campo (FDR) com GL efetivos, tamanho de efeito físico interpretável, controle Atlântico/IOD incluindo ATL4, colinearidade resolvida por bloco e tabela de gate preenchida. | Revisar variáveis/escala temporal antes de modelar; não avançar para 5 com sinal nulo ou atribuição Pacifico-only instável. |
| **G2** | fim da 5A | Walk-forward com embargo supera climatologia, persistência amortecida e oscilador de recarga simples, com skill por mês de inicialização/estação-alvo. | Revisar features/lead; a 5A é pré-requisito honesto da 6. |
| **G3** | fim da 5B | A Fase 5 (5A+5B) supera os baselines em validação evento-centrada. | **Não iniciar a Fase 6.** Voltar à orientação. |
| **G4** | fim da 6C | 6A/6B/6C superam climatologia, persistência **e** a Fase 5. | Acionar fallback CMIP6 (pré-treino/fine-tuning), com relatório de skill nativo vs fine-tuned separado. |

A regra de ouro contra sofisticação prematura é mantida: a complexidade neural da Fase 6 e o fallback CMIP6 só se justificam se vencerem climatologia, persistência e a Fase 5.

---

## 5. Sequência mínima cobrável (alinhada aos pareceres)

1. **Hoje:** higiene de repositório (commit + painel).
2. **Próximo marco de orientação:** Fase 4 executada — o ranking de variáveis por região é a primeira seção de resultados defensável.
3. Fase 5A no gate contra persistência/climatologia.
4. Fase 5B: a teleconexão Pacífico→Brasil que é o título da tese.
5. Só então a Fase 6, e apenas se passar no gate G3.
