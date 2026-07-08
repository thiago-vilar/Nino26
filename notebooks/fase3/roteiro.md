# Roteiro de execucao e leitura contextualizada - Fase 3

Projeto NINO-BRASIL - Diagnostico fisico do Nino 3.4

## 1. Atualizar 2026 no maximo possivel

Use primeiro o modo seco para conferir comandos:

```cmd
cd /d C:\DEV\NINO26
.venv\Scripts\python scripts\update_2026_nino34.py
```

Para executar downloads/processamentos externos:

```cmd
cd /d C:\DEV\NINO26
.venv\Scripts\python scripts\update_2026_nino34.py --execute
```

O script respeita a latencia por fonte: OISST/ERA5 ~7 dias, GLO12 operacional
~1 dia, ORAS5 mensal ~15 dias e in situ ~3 dias. CTD/WOD nao deve ser usado
como fonte operacional de 2026, pois tem latencia anual longa.

## 2. Reconstruir a Fase 3

```cmd
cd /d C:\DEV\NINO26
.venv\Scripts\python scripts\data_pipeline.py build-nino34-daily-index
.venv\Scripts\python scripts\data_pipeline.py build-nino34-sst-reference
.venv\Scripts\python scripts\data_pipeline.py build-phase3-diagnostics
.venv\Scripts\python scripts\data_pipeline.py audit-phase3-diagnostics
.venv\Scripts\python scripts\update_era5_nino34_atmo_cache.py --start-year 1981 --end-year 2026
.venv\Scripts\python scripts\fase3_build_inputs.py --force
.venv\Scripts\python -m pytest -q
```

Nao avance se `data\audit\phase3_diagnostics_audit.json` tiver `errors`
diferente de `[]`.

## 3. Executar notebooks

No terminal:

```cmd
.venv\Scripts\python scripts\run_fase3_all.py
```

No VS Code, selecione o kernel `Python 3 (.venv NINO26)`, rode na ordem 3A-3H,
3K e 3I
e reinicie o kernel entre notebooks. Os notebooks nao materializam dado bruto:
eles leem produtos do pipeline e gravam tabelas/figuras interpretativas.

## 4. Como ler cada etapa

| Notebook | Pergunta | Decisao |
|---|---|---|
| 3A | Quais series fisicas existem e em que janela real? | Variavel com baixa cobertura ou fonte emendada entra com ressalva. |
| 3B | Como eventos crescem, picam e decaem? | Leia as classes NOAA/ONI locais: fraco, moderado, forte e muito forte; o e-folding da SSTA define o baseline de persistencia. |
| 3C | Quais variaveis parecem liderar o Nino 3.4? | Triagem; nao cite como evidencia final. |
| 3D | O que sobrevive a N_eff, IC95 e FDR? | Primeiro filtro para o parecer. |
| 3E | O que e estavel em 1993-2009 e 2010-presente? | Segundo filtro; instavel vira limite de regime. |
| 3F | DHW tem informacao propria? Kelvin aparece? | DHW so entra se parcial sobreviver; Kelvin e leitura qualitativa. |
| 3G | DHW mede severidade acumulada? | Use somente `dhw_cweek_0p5_12w` e compare as classes NOAA de evento. |
| 3H | A genese separa classes NOAA? | Descritivo por classe e pela media geral dos eventos; prepara hipoteses para Fase 5. |
| 3K | Quais variaveis explicam crescimento pre-pico? | Sintese multivariada; nao substitui 3D/3E. |
| 3I | Qual e o veredito integrado? | Texto e tabelas finais, incluindo classes NOAA, DHW canonico e estado 2026. |

Referencias oficiais NOAA/CPC usadas na leitura:

| Caixa | Longitude/latitude oficial | Papel |
|---|---|---|
| Nino 3.4 | 5N-5S, 170W-120W | alvo da Fase 3 |
| Nino 4 | 5N-5S, 160E-150W | referencia desejada para anomalias de vento/WWB |
| Banda 2S-2N, 120E-80W | faixa diagnostica, nao caixa oficial Nino | Hovmoller e mapas longitude x lag |

Nas figuras 3A/3C/3D/3E/3F/3G, o eixo x segue a leitura oeste->leste. A
esquerda fica o Pacifico oeste (120E), o centro marca as longitudes do Nino
3.4 (170W-120W), e a direita fica o Pacifico leste (80W). Use sempre essa
referencia W/E no texto interpretativo.

## 5. DHW correto para Nino 3.4

`dhw_cweek_0p5_12w` e a unica metrica DHW da Fase 3: soma em C-weeks apenas de
HotSpots diarios com SSTA Nino 3.4 >= +0,5 C, acumulados em 12 semanas. A
validacao temporal auxiliar exige `oni_12w_mean_c >= +0,5 C` por 20 semanas
consecutivas. Nao use DHW sozinho como declaracao oficial de El Nino; ele
materializa o criterio termico local e precisa ser lido junto do acoplamento
atmosferico.

## 6. Classificacao NOAA/ONI local

Evento El Nino = media movel de 3 meses da SSTA Nino 3.4 >= +0,5 C por pelo
menos 5 estacoes moveis sobrepostas. A intensidade e dada pelo pico dessa media:

1. `fraco`: 0,5 C <= pico < 1,0 C.
2. `moderado`: 1,0 C <= pico < 1,5 C.
3. `forte`: 1,5 C <= pico < 2,0 C.
4. `muito_forte`: pico >= 2,0 C.

A Fase 3 usa OISST local para reprodutibilidade e avalia acoplamento por proxy
atmosferico (`tau_x_anom_nino34_pa`). Declaracao oficial operacional continua
sendo responsabilidade do CPC/NOAA.

As tabelas principais ficam em:

```text
data\processed\parquet\statistics\phase3I_classificacao_noaa_oni.csv
data\processed\parquet\statistics\phase3I_medias_classes_noaa.csv
data\processed\parquet\statistics\phase3I_estado_2026.csv
```

## 7. Vento e tau_x

`tau_x` e estresse zonal do vento, ou seja, friccao/forcamento zonal aplicado
na superficie do oceano, em Pa. Ele nao e automaticamente uma anomalia: so vira
anomalia quando removemos a climatologia diaria. A Fase 3 usa
`tau_x_anom_nino34_pa`, anomalia 1991-2020 do proxy de tau_x derivado do `u10`
ERA5 em Nino 3.4. Para investigar deslocamento/anomalia de vento de forma mais
direta, use `u10_anom`; para WWB/Kelvin, a regiao mais adequada e Nino 4
(5N-5S, 160E-150W).

## 8. Estado 2026

A leitura de 2026 deve vir de:

```text
data\processed\parquet\statistics\phase3I_estado_2026.csv
```

Se o mes mais recente estiver incompleto, escreva "aquecimento em curso" e nao
"evento fechado". Um novo El Nino so deve entrar na tabela de eventos quando o
criterio local OISST completar a duracao minima.

## 9. Regra de escrita do parecer

1. Figura ilustra; tabela decide.
2. Cite sempre janela, fonte e `N_eff`.
3. Nao some D20/OHC/WWV/SSH/tilt como evidencias independentes: e um bloco de
   recarga/subsuperficie.
4. Lag 0 e estado simultaneo, nao precursor.
5. Fase 3 nao e previsao; ela justifica hipoteses para Fase 5.
