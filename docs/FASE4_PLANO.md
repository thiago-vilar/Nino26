# Fase 4 - Plano curto (protocolo)

> Protocolo oficial e completo em `docs/METODOLOGIA_FASE4.md`
> (reconstrucao 2026-07-08, expandida). Escopo estritamente Pacifico -> Brasil.

```
4A  Ciclo ENSO em 4 fases (genese, crescimento/acoplamento, pico, decaimento)
    para El Nino E La Nina + avaliacao expandida (duracoes, ONI por fase).
4B  Estudo puramente estatistico: quais variaveis do Pacifico mais determinam
    cada fase (Mann-Whitney/Cliff + Kruskal/epsilon2 + Spearman genese->pico).
4C  Sinal pixel-a-pixel: conjunto Pacifico x chuva CHIRPS 0.25, lags 0-78,
    N_eff+FDR, por tipo de sinal; Brasil inteiro -> recorte NEB -> recorte Sul.
4D  So depois do pixel: alvos clusterizados, lag por tipo de sinal,
    estabilidade 1993-2009 vs 2010+ e gate G1 para ML.
```

Notebooks: `notebooks/fase4/4A_ciclo_enso_fases.ipynb`,
`4B_variaveis_determinantes_fases.ipynb`, `4C_sinal_pixel_lags.ipynb`,
`4D_clusters_alvo.ipynb`. Modulo: `notebooks/fase4/fase4_utils.py`.
Execucao: `python scripts/run_fase4_all.py`.

Utilidade legada que permanece valida como referencia tecnica:
`scripts/fase4_harmonize_sync.py` (harmonizacao vertical e correcao
source-aware do degrau UFS->GLORYS).
