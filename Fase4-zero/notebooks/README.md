# Fase 4 — notebooks canônicos

Teleconexao ENSO -> anomalia de chuva no Brasil. Protocolo completo em
`docs/METODOLOGIA_FASE4.md`. Escopo estritamente Pacifico -> Brasil.
F3 fornece o contrato único de eventos/fases. A execução canônica de F4 contém
somente 4C → 4D: sinal CHIRPS nativo pixel-a-pixel e gate multi-alvo.

| Ordem | Notebook | O que responde |
|---|---|---|
| 4C | `4C_sinal_pixel_lags.ipynb` | Sinal pixel-a-pixel: `Pacifico(t-L)` x anomalia robusta de chuva CHIRPS 0,25(t), lags 0-78, N_eff+BH-FDR α=0,05; Brasil inteiro, NEB e Sul. |
| 4D | `4D_clusters_alvo.ipynb` | Só depois do pixel: clusters descritivos dos perfis nativos e gate multi-alvo com FDR, significância de campo e leave-one-event-out; nenhum corte temporal fixo. |

Os arquivos `4_0`, `4A` e `4B` permanecem apenas para rastreabilidade histórica e
não pertencem a `run_fase4_all.py`: recalculavam fases e/ou selecionavam
preditores para ONI, escopo já coberto por F3 e incompatível com F4 all-31.

Modulo compartilhado: `fase4_utils.py` (apoio histórico); o alvo canônico usa
anomalia robusta CHIRPS por pixel, inventario de lags e correlação vetorizada
com N_eff+FDR, N_eff generico, mapas com recorte regional).

Execucao headless:

```cmd
.venv\Scripts\python scripts\run_fase4_all.py
```

Notas operacionais:

- O alvo oficial é `chirps_native_weekly_targets.zarr`, criado por blocos
  retomáveis sem regrid. O parquet z legado não é fonte científica.
- 4C lê as fases canônicas da F3; 4D depende das saídas auditadas de 4C.

Kernel no VS Code: `Python 3.12 (.venv NINO26)` (`nino-brasil`).
