# Roteiro de execucao e leitura - Fase 3

Projeto NINO-BRASIL - Diagnostico fisico do Nino 3.4  
Atualizado em 2026-07-08.

## 1. Objetivo da Fase 3

A Fase 3 identifica, com dados locais e rastreaveis, quais variaveis fisicas do
Pacifico explicam o aquecimento maximo do Nino 3.4 antes, durante e depois dos
eventos El Nino. Ela ainda nao e uma previsao numerica do pico de 2026: ela
prepara os candidatos, lags e ressalvas para uma futura validacao walk-forward.

Regra transversal: figura ilustra, tabela decide.

## 2. Execucao completa

```cmd
cd /d C:\DEV\NINO26

.venv\Scripts\python scripts\data_pipeline.py build-nino34-daily-index
.venv\Scripts\python scripts\data_pipeline.py build-nino34-sst-reference
.venv\Scripts\python scripts\data_pipeline.py build-phase3-diagnostics
.venv\Scripts\python scripts\data_pipeline.py audit-phase3-diagnostics
.venv\Scripts\python scripts\fase3_build_inputs.py --force

.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3A_indices_fisicos_semanais.ipynb --inplace --ExecutePreprocessor.timeout=1200
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3B_alvo_eventos_ciclo_vida.ipynb --inplace --ExecutePreprocessor.timeout=1200
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3C_precursores_lags.ipynb --inplace --ExecutePreprocessor.timeout=1200
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3D_rigor_estatistico.ipynb --inplace --ExecutePreprocessor.timeout=1200
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3E_estabilidade_subperiodos.ipynb --inplace --ExecutePreprocessor.timeout=1200
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3F_kelvin_sla.ipynb --inplace --ExecutePreprocessor.timeout=1200
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3G_compostos_ssta.ipynb --inplace --ExecutePreprocessor.timeout=1200
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3H_genese_precursores_classe.ipynb --inplace --ExecutePreprocessor.timeout=1200
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3I_interpretacao_integrada.ipynb --inplace --ExecutePreprocessor.timeout=1200
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/fase3/3K_pca_crescimento.ipynb --inplace --ExecutePreprocessor.timeout=1200

.venv\Scripts\python scripts\audit_phase3_temporal_integrity.py
.venv\Scripts\python -m pytest -q
```

## 3. Criterios metodologicos fixos

Evento El Nino local: media movel de 3 meses da SSTA Nino 3.4 >= +0.5 C por
pelo menos 5 estacoes moveis sobrepostas. A classificacao usa o pico dessa
media:

| Classe | Pico ONI local |
|---|---|
| fraco | 0.5 C <= pico < 1.0 C |
| moderado | 1.0 C <= pico < 1.5 C |
| forte | 1.5 C <= pico < 2.0 C |
| muito_forte / super | pico >= 2.0 C |

## 4. Como ler cada notebook

| Notebook | Pergunta | Leitura executiva |
|---|---|---|
| 3A | Quais variaveis existem, quais sao anomalias e qual a cobertura? | SSTA e tau_x sao anomalias; D20/OHC/WWV/tilt/SSH sao indices fisicos originais. |
| 3B | Qual e a memoria da SSTA e o ciclo de vida dos eventos? | A autocorrelacao e o baseline de persistencia que qualquer previsao deve superar. |
| 3C | Quais variaveis lideram a SSTA em lag bruto? | Triagem ordenada por maior abs(r); lag positivo = precursor antecede a SSTA alvo. |
| 3D | O que sobrevive a N_eff, IC95 e FDR? | Primeiro filtro defensavel; forest plot mostra r, lag, N_eff e IC95. |
| 3E | O sinal e estavel antes/depois de 2010? | Estavel entra no parecer; instavel entra como ressalva de regime. |
| 3F | Kelvin e visivel na dinamica equatorial? | SLA/SSH Hovmoller e tau_x dao evidencia qualitativa de propagacao Kelvin. |
| 3G | Como SSTA por classe se organiza e como 2025/26 se compara? | Compara eventos fortes/super com a formacao atual por SSTA longitudinal; atual e alinhado ao ultimo dado. |
| 3H | A genese separa classes NOAA? | Mostra onset e ciclo alinhado ao pico real para todos, moderados, fortes e super. |
| 3I | Qual conjunto antecipa o aquecimento maximo? | Tabela e figura `phase3I_conjunto_antecipacao_pico` separam precursor, estado e severidade. |
| 3K | Quais dimensoes fisicas sao redundantes? | PCA evita contar D20/OHC/SSH/WWV/tilt como evidencias independentes. |

## 5. Eixos e coordenadas

Nos Hovmollers e mapas longitude-lag, o eixo x segue a referencia oficial
oeste para leste: 120E, 140E, 160E, 180, 160W, 140W, 120W, 100W, 80W. A faixa
Nino 3.4 fica sombreada em 170W-120W. O eixo y dos mapas longitude-lag e lag em
semanas antes da SSTA alvo; 0 e simultaneo, 26 e cerca de 6 meses, 52 e cerca
de 1 ano.

## 6. Saidas principais

Figuras: `data\processed\figures\fase3\`.

Tabelas decisorias:

```text
data\processed\parquet\statistics\phase3A_cobertura_variaveis.csv
data\processed\parquet\statistics\phase3C_ranking_lags.csv
data\processed\parquet\statistics\phase3D_ranking_significativo.csv
data\processed\parquet\statistics\phase3E_estabilidade.csv
data\processed\parquet\statistics\phase3G_mapa_ssta_lon_eventos_forte_super.csv
data\processed\parquet\statistics\phase3H_ciclo_vida_classes_pico.csv
data\processed\parquet\statistics\phase3I_conjunto_antecipacao_pico.csv
data\processed\parquet\statistics\phase3I_estado_2026.csv
```

## 7. Parecer defensavel

O bloco de recarga/subsuperficie e o eixo fisico central: D20, SSH, OHC, WWV e
tilt descrevem o reservatorio oceanico que permite amplificar a SSTA. `tau_x`
anomalo representa o acoplamento vento-superficie. A interpretacao final evita
metricas acumuladas auxiliares e prioriza variaveis dinamicas para antecipar o
pico.

Para 2025/26, escreva "formacao/aquecimento em curso" enquanto o pico futuro
nao for observado e validado. A Fase 3 nao deve prometer projecao numerica; ela
entrega a base fisica para a etapa futura de previsao.
