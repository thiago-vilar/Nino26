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
.venv\Scripts\python scripts\data_pipeline.py build-nino34-p90-peaks
.venv\Scripts\python scripts\data_pipeline.py build-nino34-p95-peaks
.venv\Scripts\python scripts\data_pipeline.py build-phase3-diagnostics
.venv\Scripts\python scripts\data_pipeline.py audit-phase3-diagnostics
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

No VS Code, selecione o kernel `Python 3 (.venv NINO26)`, rode na ordem 3A-3I
e reinicie o kernel entre notebooks. Os notebooks nao materializam dado bruto:
eles leem produtos do pipeline e gravam tabelas/figuras interpretativas.

## 4. Como ler cada etapa

| Notebook | Pergunta | Decisao |
|---|---|---|
| 3A | Quais series fisicas existem e em que janela real? | Variavel com baixa cobertura ou fonte emendada entra com ressalva. |
| 3B | Como eventos crescem, picam e decaem? | O e-folding da SSTA define o baseline de persistencia. |
| 3C | Quais variaveis parecem liderar o Nino 3.4? | Triagem; nao cite como evidencia final. |
| 3D | O que sobrevive a N_eff, IC95 e FDR? | Primeiro filtro para o parecer. |
| 3E | O que e estavel em 1993-2009 e 2010-presente? | Segundo filtro; instavel vira limite de regime. |
| 3F | DHW tem informacao propria? Kelvin aparece? | DHW so entra se parcial sobreviver; Kelvin e leitura qualitativa. |
| 3G | DHW mede severidade acumulada? | Compare `dhw_12w` e `dhw_26w_p90` por evento. |
| 3H | A genese separa fortes de fracos? | Descritivo; prepara hipoteses para Fase 5. |
| 3I | Qual e o veredito integrado? | Texto e tabelas finais para o parecer. |

## 5. DHW correto para Nino 3.4

`dhw_12w` (`>1 C` por 12 semanas) fica como metrica herdada/compatibilidade. A
janela veio da tradicao CRW, que foi pensada para estresse termico de coral, nao
para a memoria de ENOS.

`dhw_26w_p90` e a metrica preferida para severidade ENSO na Fase 3: a janela de
26 semanas aproxima o e-folding observado da SSTA (~27 semanas), e o limiar P90
diario e derivado da propria OISST local. Ela mede carga termica acumulada no
tempo do evento. Nao use DHW como definicao de El Nino nem como skill preditivo.

## 6. P90 e P95

P90 compara todos os picos quentes mensais da OISST local. P95 compara a cauda
extrema/super-eventos. A tabela principal fica em:

```text
data\processed\parquet\statistics\phase3I_picos_p90_p95_comparacao.csv
```

Use P90 para ranking geral e P95 para o conjunto extremo. P95 nao substitui a
classificacao classica por limiares fixos; ele e uma regua interna, auditavel e
sem rotulo externo.

## 7. Estado 2026

A leitura de 2026 deve vir de:

```text
data\processed\parquet\statistics\phase3I_estado_2026.csv
```

Se o mes mais recente estiver incompleto, escreva "aquecimento em curso" e nao
"evento fechado". Um novo El Nino so deve entrar na tabela de eventos quando o
criterio local OISST completar a duracao minima.

## 8. Regra de escrita do parecer

1. Figura ilustra; tabela decide.
2. Cite sempre janela, fonte e `N_eff`.
3. Nao some D20/OHC/WWV/SSH/tilt como evidencias independentes: e um bloco de
   recarga/subsuperficie.
4. Lag 0 e estado simultaneo, nao precursor.
5. Fase 3 nao e previsao; ela justifica hipoteses para Fase 5.
