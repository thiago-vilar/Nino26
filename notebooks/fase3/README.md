# Fase 3 - Notebooks 3A-3I/3K

Implementacao executavel do protocolo de `docs/FASE3_RECOMENDACOES.md`.
Todos os notebooks: (i) declaram a pergunta que respondem e a metodologia,
(ii) gravam tabelas numericas em `data/processed/parquet/statistics/`,
(iii) gravam figuras e mapas em `data/processed/figures/fase3/`.
Nenhuma figura sem saida numerica correspondente.

Convencao das figuras: `3A1_...png` significa Fase 3, notebook A, figura 1.
Quando um notebook precisa de mais de uma imagem para o mesmo subcontexto, a
sequencia cresce (`3A2`, `3A3` etc.). O catalogo interpretativo fica em
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
eventos, DHW diario canonico e o cache de mapas. Use
`--force` quando OISST/oceano forem atualizados.

Execucao headless completa:

```cmd
.venv\Scripts\python scripts\run_fase3_all.py
```

## Notebooks (ordem de execucao)

| NB | Pergunta | Saidas principais |
|---|---|---|
| 3A | Quais series fisicas descrevem o sistema e em que janelas reais? | `phase3_indices_semanais.csv` (matriz canonica W-SUN), cobertura, series, Hovmoller panorama |
| 3B | Como eventos nascem, crescem, picam e decaem? Quanta memoria ha? | taxas por evento, trajetorias por classe NOAA/ONI local, e-folding, mapa composto de pico |
| 3C | O que antecede o pico do Nino 3.4, e com quantas semanas? | ranking preliminar de lags, heatmap, mapa lon x lag |
| 3D | O que sobrevive a N_eff + FDR + IC95? | testes completos, ranking significativo, forest plot, mapa FDR |
| 3E | O sinal vale em 1993-2009 E 2010-presente? | tabela de estabilidade, scatter r1 x r2, mapas por subperiodo |
| 3F | DHW agrega leitura alem de SSTA/WWV/OHC? Kelvin e visivel? | correlacao parcial por horizonte, Hovmoller SSH |
| 3G | (extra) Como o ciclo de vida se relaciona com o DHW C-week? | metricas por fase de evento, composto SSTA/DHW por classe NOAA, escalonamento, mapa DHW-lon |
| 3H | (extra) Que estado fisico precede o onset? A genese separa classes NOAA? | compostos onset-alinhados por classe, retrato precursor por classe, separacao Spearman |
| 3K | (extra) Quais variaveis explicam o crescimento pre-pico? | PCA, loadings e conjunto indispensavel para crescimento |
| 3I | Qual e a interpretacao integrada da Fase 3? Como ler classes NOAA e 2026? | conclusoes executivas, classificacao NOAA, medias por classe, nested LOO, estado 2026, texto para parecer |

**Regra de corte do parecer:** so entra o que sobrevive a **3D e 3E**.
O 3F tem regra extra de nao-redundancia; o 3G caracteriza severidade acumulada;
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

## DHW

O DHW canonico da Fase 3 e `dhw_cweek_0p5_12w`: soma em C-weeks apenas dos
HotSpots diarios com SSTA Nino 3.4 >= +0,5 C, acumulados em janela movel de 12
semanas. A validacao termica auxiliar exige `oni_12w_mean_c >= +0,5 C` por pelo
menos 20 semanas consecutivas. A Fase 3 nao publica janelas DHW concorrentes.

Classificacao executiva: evento El Nino exige media movel de 3 meses da SSTA
Nino 3.4 >= +0,5 C por 5 estacoes moveis sobrepostas. A intensidade e o pico
dessa media: `fraco` [0,5;1,0), `moderado` [1,0;1,5), `forte` [1,5;2,0) e
`muito_forte` >= 2,0 C. O acoplamento atmosferico e avaliado por proxies locais,
principalmente `tau_x_anom_nino34_pa`; a declaracao oficial continua sendo
competencia do CPC/NOAA.

As tabelas de referencia ficam em `phase3B_grupos_classes_noaa.csv`,
`phase3G_composto_ssta_dhw_classes_noaa.csv`,
`phase3H_grupos_classes_noaa.csv` e `phase3I_classificacao_noaa_oni.csv`.

## Vento

`tau_x` e estresse zonal do vento (friccao na superficie), em Pa. A Fase 3
publica `tau_x_anom_nino34_pa`, anomalia diaria 1991-2020 do proxy de estresse
zonal derivado do `u10` ERA5 na caixa Nino 3.4. Para investigar WWB/Kelvin com
mais rigor, a variavel desejada e `u10_anom` e/ou `tau_x_anom` em Nino 4
(5N-5S, 160E-150W), a ser materializada quando o cache ERA5 Nino 4 existir.

Kernel no VS Code: `Python 3 (.venv NINO26)`.
Modulo compartilhado: `fase3_utils.py` (caminhos, matriz semanal, eventos, salvamento padrao).
