# Painel descritivo - Parecer 3

Data da revisao: 2026-06-14

## Veredito executivo

O Parecer 3 faz sentido e recomenda alteracoes necessarias. O projeto ja tinha variaveis oceanicas e atmosfericas em estado bruto, mas ainda nao tinha os termos dinamicos que expressam o mecanismo fisico do ENSO: recarga de calor, inclinacao da termoclina, tendencias temporais e forcantes mecanicas de vento.

A decisao adotada foi implementar agora o nucleo de baixo risco, sem depender do fechamento do download ERA5/ORAS5:

1. funcoes puras de pre-calculo fisico;
2. classificacao automatica dessas features como `physics_precalc`;
3. ablation `G_with_physics` versus `H_without_physics`;
4. testes unitarios com dados sinteticos.

## Julgamento por item

| Item | Decisao | Motivo |
|---|---|---|
| PC-5 WWV | Necessario; implementado | Representa o ciclo de recarga do ENSO e reduz dependencia de D20 apenas no Nino 3.4 |
| PC-6 Tilt da termoclina | Necessario; implementado | Distingue fase do evento e sinal do feedback de Bjerknes |
| PC-1 dD20/dt | Necessario; implementado | Explicita aprofundamento/rasamento da termoclina |
| PC-2 dOHC/dt | Necessario; implementado | Explicita taxa de acumulacao de calor oceanico |
| PC-3 dSSTA/dt | Necessario; implementado | Separa aquecimento crescente de aquecimento em decaimento |
| PC-4 wind stress curl | Necessario, mas depende de tau_x/tau_y | A funcao foi implementada; falta garantir download/derivacao da tensao de vento |
| PC-7 stress zonal equatorial | Util; implementado como auxiliar | Baixo custo, mas menor prioridade que WWV/tilt/tendencias |
| PC-8 SSHA | Adiar | Nova fonte operacional; nao bloqueia a tese historica |

## Mudancas aplicadas

| Arquivo | Mudanca |
|---|---|
| `src/nino_brasil/features/ocean_heat.py` | Adicionados `warm_water_volume` e `ohc_tendency` |
| `src/nino_brasil/features/thermocline.py` | Adicionados `thermocline_tilt` e `d20_tendency` |
| `src/nino_brasil/features/tendency_features.py` | Novo modulo com `ssta_tendency`, `wind_stress_curl` e `equatorial_zonal_stress` |
| `src/nino_brasil/features/spatial.py` | Helper para selecionar boxes em `0..360` ou `-180..180` |
| `src/nino_brasil/models/feature_matrix.py` | Features fisicas derivadas passam a ser classificadas como `physics_precalc` |
| `src/nino_brasil/models/ablation.py` | Adicionada comparacao `G_with_physics` vs `H_without_physics` |
| `tests/test_features.py` | Testes numericos para WWV, tilt, tendencias e curl |
| `tests/test_modeling.py` | Testes da classificacao `physics_precalc` e da ablation |

## Observacoes tecnicas

- As funcoes de box oceanico aceitam longitudes em `0..360` e `-180..180`.
- O codigo nao inicia downloads nem altera arquivos grandes.
- A geracao de `physics_precalc_timeseries.csv` fica para quando ORAS5 e os campos de tensao de vento estiverem disponiveis.
- ERA5 atual ainda nao baixa `tau_x`/`tau_y`; portanto PC-4 esta pronto em codigo, mas ainda sem fonte local consolidada.

## Proximo passo recomendado

1. Fechar ERA5 pressure-level e ORAS5.
2. Decidir a fonte de `tau_x`/`tau_y`: baixar tensao de vento ERA5, derivar de `u10/v10`, ou usar produto oceano-atmosfera dedicado.
3. Criar o job de Fase 3 que materializa `data/processed/parquet/physics_precalc_timeseries.csv`.
4. Rodar ablation `G_with_physics` contra `H_without_physics` na Fase 5.
