# Fase 3 - Notebooks 3A-3L

Implementacao executavel do protocolo de `docs/FASE3_RECOMENDACOES.md`.
Todos os notebooks: (i) declaram a pergunta que respondem e a metodologia,
(ii) gravam tabelas numericas em `data/processed/parquet/statistics/`,
(iii) gravam figuras e mapas em `data/processed/figures/fase3/`.
Nenhuma figura sem saida numerica correspondente.

Status em 2026-07-12: o código e o piloto semântico estão fechados; a execução
oficial completa e a migração das figuras legadas para linhagem semântica ainda
são gates separados. Ver `../../docs/PAINEL_REFATORACAO_EXECUCAO_2026-07-12.md`.

Convencao das figuras: `Fig_3A01...png` significa Fase 3, notebook A, figura 1.
Sufixos descritivos são permitidos, preservando o prefixo globalmente único.
O catalogo interpretativo fica em
`INDICE_FIGURAS_FASE3.md` e o relatorio completo em
`RELATORIO_FINAL_FASE3.md`.

## Pre-requisitos

```cmd
.venv\Scripts\python scripts\data_pipeline.py build-nino34-daily-index
.venv\Scripts\python scripts\data_pipeline.py build-nino34-sst-reference
.venv\Scripts\python scripts\data_pipeline.py build-phase3-diagnostics
.venv\Scripts\python scripts\data_pipeline.py audit-phase3-diagnostics
.venv\Scripts\python scripts\fase3_build_inputs.py --force
.venv\Scripts\python -m pytest -q
```

`fase3_build_inputs.py` materializa banda equatorial por longitude, SSH de
eventos e o cache de mapas. Use
`--force` quando OISST/oceano forem atualizados.

Execucao headless completa:

```cmd
.venv\Scripts\python scripts\run_fase3_all.py
```

## Notebooks (ordem de execucao)

| NB | Pergunta | Saidas principais |
|---|---|---|
| 3A | Quais series fisicas descrevem o sistema e em que janelas reais? | `phase3_indices_semanais.csv` (visao derivada W-SUN para consumo F3; a fonte canonica e o master F2), cobertura, series, Hovmoller panorama |
| 3B | Como eventos nascem, crescem, picam e decaem? Quanta memoria ha? | taxas por evento, trajetorias por classe NOAA/ONI local, e-folding, mapa composto de pico |
| 3C | O que antecede o pico do Nino 3.4, e com quantas semanas? | catalogo auditavel de precursores com alvo/aliases excluidos, ranking preliminar de lags, heatmap, mapa lon x lag |
| 3D | O que sobrevive a N_eff + FDR + IC95? | testes somente dos precursores fisicos, familia variavel x lag registrada, ranking significativo, forest plot, mapa FDR |
| 3E | A relacao depende da autocorrelacao ou de um evento ENSO isolado? | bootstrap movel em blocos 26/52/78 semanas, leave-one-event-out, IC95 e influencia por evento; sem breakpoint e sem gate |
| 3F | Kelvin e visivel na dinamica SLA/SSH + vento? | Hovmoller SLA/SSH, resumo por evento e tau_x |
| 3G | (extra) Como SSTA por classe se organiza e como 2025/26 se compara? | composto SSTA por classe, escalonamento termico, mapa SSTA-lon |
| 3H | (extra) Que estado fisico precede o onset? A genese separa classes NOAA? | compostos onset-alinhados por classe, retrato precursor por classe, separacao Spearman |
| 3K | (extra) Quais variaveis explicam o crescimento pre-pico? | PCA, loadings e conjunto indispensavel para crescimento |
| 3I | Qual e a interpretacao integrada da Fase 3? Como ler classes NOAA e 2026? | conclusoes executivas, classificacao NOAA, medias por classe, nested LOO, estado 2026, texto para parecer |
| 3L | O protocolo EN/LN completo fecha a diretriz? | eventos EN/LN, ciclo por evento, duracao por fase, Friedman pareado/Kendall W com BH-FDR separado por tipo e PCA por fase |

**Regra de evidencia do parecer:** o 3D controla N_eff, IC95 e FDR e nunca
inclui o alvo/aliases no ranking de precursores. No 3L, Kendall W sem
`q_friedman_bh <= 0,05` é apenas tamanho de efeito descritivo. O 3E
acompanha o resultado com sensibilidade bootstrap/LOO, mas nao cria corte
binario nem descarta variaveis. O 3F e diagnostico dinamico de Kelvin; o 3G caracteriza o alvo termico SSTA;
o 3I consolida a interpretacao, mas nao cria evidencia nova.

## Referencias oficiais de longitude

Referencias NOAA/CPC usadas na Fase 3:

| Caixa | Referencia oficial | Uso na Fase 3 |
|---|---|---|
| Nino 3.4 | 5N-5S, 170W-120W | alvo das series, eventos NOAA/ONI locais e parecer |
| Nino 4 | 5N-5S, 160E-150W | referencia desejada para anomalias de vento/WWB |
| Banda equatorial 2S-2N, 120E-80W | nao e caixa oficial Nino | diagnostico longitudinal/Hovmoller |

Nas figuras longitudinais, o eixo x segue a leitura oeste->leste:
esquerda = Pacifico oeste (120E), centro = longitudes Nino 3.4 (170W-120W),
direita = Pacifico leste (80W). A notacao interna de arquivos pode usar
longitude numerica continua, mas relatorios e figuras devem mostrar apenas W/E.

Classificacao executiva: evento El Nino exige media movel de 3 meses da SSTA
Nino 3.4 >= +0,5 C por 5 estacoes moveis sobrepostas. A intensidade e o pico
dessa media: `fraco` [0,5;1,0), `moderado` [1,0;1,5), `forte` [1,5;2,0) e
`muito_forte` >= 2,0 C. O acoplamento atmosferico e avaliado por proxies locais,
principalmente `tau_x_anom_nino34_pa`; a declaracao oficial continua sendo
competencia do CPC/NOAA.

As tabelas de referencia ficam em `phase3B_grupos_classes_noaa.csv`,
`phase3G_composto_ssta_classes_noaa.csv`,
`phase3H_grupos_classes_noaa.csv` e `phase3I_classificacao_noaa_oni.csv`.

## Vento

`tau_x` e estresse zonal do vento (friccao na superficie), em Pa. A Fase 3
publica `tau_x_anom_nino34_pa`, anomalia diaria 1991-2020 do proxy de estresse
zonal derivado do `u10` ERA5 na caixa Nino 3.4. Para investigar WWB/Kelvin com
mais rigor, a variavel desejada e `u10_anom` e/ou `tau_x_anom` em Nino 4
(5N-5S, 160E-150W), a ser materializada quando o cache ERA5 Nino 4 existir.

Kernel no VS Code: `Python 3.12 (.venv NINO26)` (`nino-brasil`).
Modulo compartilhado: `fase3_utils.py` (caminhos, matriz semanal, eventos, salvamento padrao).
