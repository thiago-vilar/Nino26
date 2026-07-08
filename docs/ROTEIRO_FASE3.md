# Roteiro de execução da Fase 3 (3A–3G)

**Projeto NINO-BRASIL · Diagnóstico físico do Niño 3.4 · atualizado em 2026-07-07**

Escopo: executar e interpretar o protocolo 3A–3G no VS Code, a partir dos dados
locais, sem rótulo ENSO externo e sem ML. Regra transversal: **notebook não
materializa dado** — só lê o que o pipeline gravou; **figura ilustra, tabela decide**.

---

## 1. Pré-requisitos (uma única vez)

```cmd
cd /d C:\DEV\NINO26

:: 1. Base da Fase 3 (se os stores ainda nao existem ou apos atualizar OISST)
.venv\Scripts\python scripts\data_pipeline.py build-nino34-daily-index
.venv\Scripts\python scripts\data_pipeline.py build-nino34-sst-reference
.venv\Scripts\python scripts\data_pipeline.py build-phase3-diagnostics
.venv\Scripts\python scripts\data_pipeline.py audit-phase3-diagnostics

:: 2. Insumos dos notebooks (ATL, banda equatorial, SSH eventos, DHW, mapas)
.venv\Scripts\python scripts\fase3_build_inputs.py --force

:: 3. Sanidade
.venv\Scripts\python -m pytest -q
```

Critérios de partida: auditoria da Fase 3 com `errors: []`
(`data\audit\phase3_diagnostics_audit.json`) e testes verdes. Se a auditoria
acusar erro, **não** prossiga para os notebooks.

## 2. Execução dos notebooks (VS Code)

Kernel: **Python 3 (.venv NINO26)**. Em cada notebook: *Run All*; entre
notebooks: *Restart Kernel* (garante que cada um depende só do disco).

| Ordem | Notebook | Depende de | Grava (principais) |
|---|---|---|---|
| 1 | `3A_indices_fisicos_semanais` | pré-requisitos | `features/phase3_indices_semanais.csv`, cobertura |
| 2 | `3B_alvo_eventos_ciclo_vida` | 3A | taxas, trajetórias compostas, persistência, mapa composto |
| 3 | `3C_precursores_lags` | 3A | `phase3C_precursor_ranking.csv`, mapa lon×lag |
| 4 | `3D_rigor_estatistico` | 3A, 3C | `phase3D_ranking_significativo.csv`, mapa FDR |
| 5 | `3E_estabilidade_subperiodos` | 3D | `phase3E_estabilidade.csv` |
| 6 | `3F_dhw_kelvin` | 3A | `phase3F_dhw_redundancia.csv`, Hovmöller SSH |
| 7 | `3G_ciclo_vida_dhw` | 3A | `phase3G_eventos_dhw.csv`, escalonamento |

Se editar o 3A (nova variável, nova janela), reexecute a cadeia inteira 3A→3G.
Se acrescentar variável ou lag "só para testar", ela entra na família do FDR:
ajuste a varredura no 3C/3D e reexecute — nunca teste fora da família.

Tabelas de decisão: `data\processed\parquet\statistics\phase3*.csv`.
Figuras: `data\processed\figures\fase3\`.

## 3. Diretrizes de interpretação

### 3.1 Regra de corte (inegociável)

Uma relação só entra no parecer se sobrevive ao **3D** (FDR α=0,05 com N_eff
+ IC95 excluindo zero) **e** ao **3E** (mesmo sinal e p<0,05 em 1993–2009 e em
2010–presente). DHW tem regra extra (3F): só entra se a correlação parcial,
controlando SSTA/WWV/OHC, permanecer significativa.

### 3.2 Como ler cada saída

**3A — cobertura.** `pct_valido` < 95% numa variável pede investigação antes
de usá-la. Subsuperfície: qualquer conclusão deve citar a ressalva 1993+
(emenda UFS→GLORYS12). O τx é *proxy* (caixa Niño 3.4, não Niño 4): interprete
fase temporal, nunca magnitude.

**3B — ciclo de vida.** O e-folding (≈27 semanas) é o *baseline de
persistência*: qualquer skill futuro só é interessante além desse horizonte.
Assimetria crescimento/decaimento e separação precoce dos super eventos
(~6 meses antes do pico) são as leituras centrais. No mapa composto, confira
que o máximo cai dentro/adjacente à caixa — se não cair, o alvo está mal posto.

**3C — triagem.** É ordenação, não evidência. Use para formar hipóteses:
quem lidera (r alto em lag>0), de onde vem (mapa lon×lag com inclinação
oeste→leste = propagação física). Lag 0 alto significa coincidência, não
precursão — só lags positivos contam como antecedência.

**3D — rigor.** Compare sempre `n` com `n_eff`: quedas de ~1.700 para ~25
são esperadas e explicam por que r=0,5 pode ser fraco. Leia o IC95 antes do
p-valor: intervalo largo que quase toca zero = evidência frágil mesmo se
"significativo". No mapa FDR, importa o *padrão espacial* que sobrevive, não
pixels isolados.

**3E — estabilidade.** Relação instável não é lixo: é *achado* (ex.: WWV
perde lead pós-2010, coerente com a literatura — nossos dados reproduzem isso
de forma independente). Reporte instáveis como limitação de regime, não os
esconda. Diferenças grandes entre os mapas dos subperíodos pedem cautela com
qualquer extrapolação para o presente.

**3F — DHW/Kelvin.** A parcial responde "o DHW é só SSTA reembalada?".
Significativa em +4 semanas e não em +12 = memória curta, não precursor de
longo lead. No Hovmöller SSH, a leitura é direcional e qualitativa (faixas
positivas migrando oeste→leste em semanas = Kelvin downwelling); não estime
velocidades sem método dedicado.

**3G — severidade.** DHW pica *depois* da SSTA (+4 a +11 semanas): é
integrador. Use DHW_max como métrica de severidade acumulada — distingue
eventos de mesmo pico e durações diferentes (r=0,975 com intensidade). O mapa
DHW-lon é aproximação semanal: cite o caveat sempre que usá-lo.

### 3.3 Armadilhas conhecidas

1. **Colinearidade do bloco de recarga**: tilt, SSH, OHC e D20 medem estados
   correlacionados do mesmo bloco físico. Não some suas evidências como se
   fossem independentes; no parecer, reporte o bloco e o melhor representante.
2. **Coincidência ≠ precursão**: r alto em lag 0 (tilt, DHW) descreve o estado
   simultâneo; para antecedência, olhe D20 (~15 sem) e WWV (~20 sem, instável).
3. **Autocorrelação disfarça acaso**: nunca cite r sem o N_eff correspondente.
4. **Janela atual (2025/26)**: o evento em curso aparece nos dados (onset→peak,
   SSTA ~1,4 °C em jun/2026), mas está *incompleto* — não entra em compostos
   nem estatísticas de evento até ter fim registrado.
5. **Não promova a Fase 3 a previsão**: os leads medidos justificam a Fase 5,
   mas nenhuma frase do parecer deve implicar skill preditivo ainda.

### 3.4 Estado defensável atual (referência rápida)

| Conclusão | Suporte |
|---|---|
| Recarga (tilt/SSH/OHC/D20) lidera o Niño 3.4; D20 com ~15 sem de antecedência | 3C+3D+3E |
| WWV tinha lead (~20 sem) mas perdeu significância pós-2010 | 3E |
| Atlântico não explica o Niño 3.4 (só ATL4 fraco/negativo como controle) | 3D+3E |
| DHW tem conteúdo próprio de curto prazo (+4 sem) e mede severidade acumulada | 3F+3G |
| Kelvin visível nos eventos históricos e na janela 2025/26 | 3F |

## 4. Encerramento e atualização

Após qualquer reexecução completa: `.venv\Scripts\python scripts\update_painel_executivo.py`
e commit das mudanças de código/notebooks (dados não são versionados).
Próximo marco além da Fase 3: parecer consolidado (possível "3H") e, somente
após validação integral das Fases 1–3, retomada da Fase 4 conforme
`docs/CRONOGRAMA.md`.
