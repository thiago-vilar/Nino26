# Fase 4 - Plano curto (protocolo)

> Protocolo oficial e completo em `docs/METODOLOGIA_FASE4.md`
> (reconstrucao 2026-07-08, expandida). Escopo estritamente Pacifico -> Brasil.

```
4A  Ciclo ENSO em 4 fases, marcando cada domingo com tipo/fase/event_id,
    semana relativa ao onset e semana relativa ao pico.
4B  Estudo puramente estatistico com todas as variaveis numericas do master:
    Cliff/Mann-Whitney + Kruskal/epsilon2 + Spearman + pares de variaveis.
4C  Sinal por pixel, por regiao IBGE e por bioma: Pacifico(t-L) x anomalia
    padronizada de chuva CHIRPS 0.25(t), lags 0-78, N_eff+FDR, por fase do
    ciclo (genese/crescimento/pico/decaimento); Brasil, regioes e biomas
    (recortes Caatinga e Mata Atlantica do NE).
4D  So depois do pixel: alvos clusterizados por estatistica descritiva,
    lag por tipo de sinal e por unidade espacial, sensibilidade temporal sem
    breakpoint (bootstrap movel + leave-one-event-out) e gate estatistico da
    hipotese NEB seco / Sul umido em El Nino.
```

Notebooks: `notebooks/fase4/4A_ciclo_enso_fases.ipynb`,
`4B_variaveis_determinantes_fases.ipynb`, `4C_sinal_pixel_lags.ipynb`,
`4D_clusters_alvo.ipynb`. Modulo: `notebooks/fase4/fase4_utils.py`.
Execucao: `python scripts/run_fase4_all.py`.

Utilidade legada que permanece valida como referencia tecnica:
`scripts/fase4_harmonize_sync.py` (harmonizacao vertical e correcao
source-aware do degrau UFS->GLORYS).
