# Parecer 2 — Saídas Numéricas como Primeira Classe de Resultado
**NINO-BRASIL — Recomendações de Output por Método**

Responsável: Thiago Vilar — PPGO/UFPE, Oceanografia Física
Complementa: `PARECER_ENG_SOFTWARE_NINO26.md`

---

## Motivação

Todo resultado analítico deste projeto — correlações defasadas, importâncias de variáveis, pesos de grupos, diagnóstico de termoclina, comparação de eventos históricos, métricas de walk-forward, saídas neurais — precisa existir primeiro como **texto numérico estruturado** antes de qualquer representação gráfica.

O motivo é operacional e científico ao mesmo tempo: imagens PNG, mapas e gráficos são inspecionáveis visualmente, mas não são analisáveis tecnicamente por uma LLM como ferramenta de revisão científica colaborativa. Se um mapa de lag dominante existir apenas como figura, não é possível perguntar "qual pixel no semiárido tem o maior lag mediano?", "qual variável superou o baseline Ridge em mais de 0,05 de AUC-ROC?", ou "o ganho de OHC sobre SST é consistente entre folds?". Com a tabela numérica correspondente, todas essas perguntas se tornam pesquisáveis em segundos.

A regra geral do projeto a partir deste parecer:

> **Toda saída visual deve ter um Zarr ou CSV numérico correspondente, gerado antes do plot, na mesma função ou script.**

Gráficos e mapas continuam sendo produzidos — são indispensáveis para publicação e para comunicação — mas são derivados das tabelas, não a fonte primária.

---

## Estado atual dos módulos de saída

Antes das recomendações, o diagnóstico do que já existe e o que falta:

| Módulo | Saída atual | Tem numérico? | Tem visual? |
|--------|-------------|---------------|-------------|
| `walk_forward.py` | `walk_forward_metrics.zarr`, `walk_forward_predictions.zarr`, `walk_forward_importances.zarr`, `walk_forward_group_weights.zarr` | ✅ Zarr completo | ❌ Nenhum plot |
| `ablation.py` | `select_feature_groups` retorna FeatureMatrix filtrada | ✅ Implícito via walk_forward | ❌ Sem tabela comparativa consolidada |
| `xai.py` | DataFrames de importância (permutation, SHAP) | ✅ DataFrame | ❌ Nenhum plot |
| `distributions.py` | DataFrame de diagnóstico de cauda | ✅ DataFrame → Zarr | ❌ Nenhum plot |
| `thermocline.py` | DataArrays xarray | ✅ Numérico | ❌ Sem serialização para CSV/Zarr |
| `ocean_heat.py` | DataArray xarray | ✅ Numérico | ❌ Sem serialização |
| `plot_choropleths.py` | Salva PNG via matplotlib | ❌ Sem tabela precedente | ✅ PNG |
| `plot_pixel_maps.py` | Salva PNG via matplotlib | ❌ Sem tabela precedente | ✅ PNG |
| `model_pipeline.py` | Zarrs de métricas/importâncias | ✅ Zarr | ❌ Sem resumo texto |
| `feature_matrix.py` | `build_predictor_matrix` → FeatureMatrix | ✅ DataFrame | ❌ Sem sumário de features |

**Padrão atual:** o lado numérico está bem servido (walk-forward produz Zarr completo); o problema é duplo — os módulos de visualização existentes não têm tabelas precedentes, e os módulos de diagnóstico físico (termoclina, OHC, heatwave) não têm serialização para texto numérico.

---

## Recomendações por método

### M1 — Correlação Defasada (Fase 4)

**Saída atual prevista:** mapa de lag dominante + mapa de correlação por pixel.

**Problema.** Um mapa de lag dominante como PNG diz "o pixel X tem lag 10 semanas", mas não permite perguntar: qual a distribuição de lags por região administrativa? Quantos pixels têm correlação > 0,3? Há descontinuidade espacial suspeita (ruído numérico)?

**Recomendação.** Antes de qualquer mapa, serializar:

```python
# src/nino_brasil/features/lag_correlation.py

def lagged_correlation_table(
    predictor: xr.DataArray,
    target: xr.DataArray,
    lags_days: list[int],
    *,
    time_name: str = "time",
    mask: xr.DataArray | None = None,
) -> pd.DataFrame:
    """
    Retorna DataFrame com colunas:
        lat, lon, lag_days, pearson_r, pearson_p, spearman_r, n_samples
    Uma linha por (pixel, lag).
    Permite filtrar por p-value, rankear lags dominantes e exportar CSV.
    """
```

**Formato de saída numérica obrigatória:**

```
data/processed/zarr/statistics/lag_correlation_full.zarr     ← todos os lags
data/processed/parquet/lag_correlation_dominant.parquet      ← lag de maior |r| por pixel
data/processed/parquet/lag_correlation_summary_by_region.csv ← agregado por bioma/estado
```

O mapa de lag vem depois, plotado a partir de `lag_correlation_dominant.parquet`.

**Para análise colaborativa:** exportar também `lag_correlation_summary_by_region.csv` com medianas por macrorregião (NEB, Sul, Norte, Centro-Oeste, Sudeste) — esse arquivo tem ~50 linhas e pode ser colado diretamente numa conversa para análise.

---

### M2 — PCA / EOF (Fase 4)

**Saída atual prevista:** modos espaciais (mapas) + série temporal dos componentes principais.

**Problema.** Mapa de EOF como PNG não permite verificar se o modo 1 captura realmente a variância esperada, se os loadings físicos fazem sentido numérico, nem comparar autovalores entre experimentos.

**Recomendação.** Serializar antes do plot:

```python
# saída mínima de PCA/EOF
{
    "explained_variance_ratio": [0.32, 0.18, 0.11, ...],  # por modo
    "cumulative_variance":      [0.32, 0.50, 0.61, ...],
    "n_components_95pct": 8,                                # componentes para 95% da variância
    "loadings": DataFrame(shape=(n_pixels, n_components)),  # lat/lon × modo
    "scores": DataFrame(shape=(n_time, n_components)),      # tempo × modo (series temporais)
    "correlation_with_nino34": Series(shape=(n_components,))# correlação de cada modo com Niño 3.4
}
```

**Formato de saída numérica obrigatória:**

```
data/processed/zarr/statistics/phase4_pca_modes.zarr         ← loadings espaciais
data/processed/parquet/phase4_pca_scores.parquet             ← series temporais
data/processed/parquet/phase4_pca_variance_summary.csv       ← tabela de variância explicada
data/processed/parquet/phase4_pca_nino34_correlation.csv     ← correlação modo × Niño 3.4
```

A `phase4_pca_variance_summary.csv` tem ~10 linhas e é o primeiro número a checar colaborativamente: se o modo 1 explica 60% da variância, o projeto simplifica; se nenhum modo domina, a física é mais complexa.

---

### M3 — KNN / Analogos Históricos (Fase 4)

**Saída atual prevista:** tabela de eventos mais similares por similaridade de perfil.

**Recomendação.** A saída já é naturalmente tabular — garantir que inclua:

```
data/processed/parquet/phase4_knn_analogs.csv
```

Com colunas: `query_year`, `query_month`, `analog_year`, `analog_month`, `distance_metric`, `nino34_ssta_query`, `nino34_ssta_analog`, `ohc_query`, `ohc_analog`, `outcome_neb_precip`, `outcome_sul_precip`. Isso permite perguntar: "os análogos do evento de 2015 previram seca no NEB com que acerto?"

---

### M4 — Walk-Forward Metrics (Fase 5) — já bem servido, ajustes menores

**O que já existe e está correto:** `walk_forward_metrics.zarr`, `walk_forward_predictions.zarr`, `walk_forward_importances.zarr`, `walk_forward_group_weights.zarr`. Todos são Zarr de DataFrames via `dataframe_to_zarr` — numéricos e totalmente legíveis.

**O que falta:** um **resumo texto de métricas** gerado automaticamente pelo `model_pipeline.py` ao final de cada run. Hoje o script imprime apenas os caminhos dos Zarrs. Adicionar:

```python
def print_metrics_summary(metrics: pd.DataFrame) -> None:
    """
    Imprime no stdout (e salva em .txt) um resumo das métricas por modelo/lag/região.
    Formato legível por humano e por LLM.
    """
    summary = (
        metrics
        .groupby(["model", "task", "lag_days", "season"])
        [["rmse", "correlation", "brier", "roc_auc", "hit_rate"]]
        .agg(["mean", "std"])
        .round(4)
    )
    print(summary.to_string())
    # salvar também:
    # data/processed/parquet/walk_forward_metrics_summary.csv
```

**Por que isso importa para análise colaborativa.** Com o resumo em texto, é possível colar 50–100 linhas numa conversa e perguntar: "qual modelo teve o menor brier para seca extrema no lag 84 dias no NEB?", "o Ridge supera a persistência em algum lag?", "há degradação de AUC-ROC entre DJF e JJA?".

**Adicionar também:** `data/processed/parquet/walk_forward_baseline_comparison.csv` com colunas `model`, `lag_days`, `region`, `season`, `metric`, `model_value`, `persistence_baseline`, `climatology_baseline`, `delta_vs_persistence`, `delta_vs_climatology`. Esse arquivo é o coração da validade científica — é onde se confirma ou refuta se o modelo bate os baselines.

---

### M5 — Permutation Importance (Fase 5)

**O que já existe:** `walk_forward_importances.zarr` com `importance_mean`, `importance_std`, `feature`, `group`, `fold`, `lag_days`, `model`, `task`. Bem estruturado.

**O que falta para análise colaborativa:**

```
data/processed/parquet/importance_top20_by_lag_model.csv
```

Gerado por:

```python
top20 = (
    importances
    .groupby(["lag_days", "model", "task"])
    .apply(lambda g: g.nlargest(20, "importance_mean"))
    .reset_index(drop=True)
)
```

Esse arquivo tem ~2.400 linhas (24 lags × 3 modelos × 2 tasks × 20 features) e é colável numa conversa para perguntar: "nos lags curtos, quais variáveis oceânicas aparecem consistentemente no top-5?", "o grupo atmosférico sobe em importância depois de 12 semanas?".

**Para o guard de `permutation_feature_limit=500`:** quando o limit é atingido e permutation importance é pulada, registrar no Zarr de importâncias uma linha com `method="skipped"`, `reason="n_features_exceeds_limit"`, `n_features=<valor>`. Assim a ausência de dados de importância é rastreável e não silenciosa.

---

### M6 — SHAP (Fase 5)

**O que já existe:** `shap_importance_frame` retorna `feature`, `mean_abs_shap` — correto.

**O que falta:**

**a) SHAP por grupo com intervalo de confiança entre folds:**

```
data/processed/parquet/shap_group_summary.csv
```

Colunas: `group`, `fold`, `lag_days`, `model`, `mean_abs_shap_sum`, e o agregado entre folds (`mean ± std`). Permite perguntar: "o grupo oceânico tem SHAP consistentemente maior que o atmosférico entre folds, ou há variância grande?".

**b) SHAP de instâncias para eventos históricos específicos:**

Para os anos de pico em `configs/project.yaml` (`el_nino_peak_years`), calcular SHAP por instância (não só o mean_abs global) e serializar:

```
data/processed/parquet/shap_event_instances_1997.csv
data/processed/parquet/shap_event_instances_2015.csv
```

Colunas: `date`, `feature`, `shap_value`, `feature_value`. Isso permite perguntar: "no pico de 1997, o OHC 0–300m teve SHAP positivo ou negativo para seca no NEB?", revelando se o modelo aprendeu a física correta ou um artefato.

**c) Consistência SHAP × Permutation Importance:**

Adicionar ao `model_pipeline.py` uma checagem automática de rank correlation (Spearman) entre o ranking de permutation importance e o ranking de SHAP para cada (lag, modelo). Se Spearman > 0,8, os métodos concordam e o sinal é robusto. Se < 0,5, há não-linearidade relevante.

```
data/processed/parquet/xai_method_agreement.csv
```

---

### M7 — Ablation A–F (Fase 5)

**O que existe:** `ablation.py` com `select_feature_groups` e as constantes `PHASE1_ABLATIONS` e `PHASE2_ABLATIONS` — correto. **Falta** o script wrapper que executa todos os 6 experimentos e produz a tabela comparativa.

**Saída numérica obrigatória:**

```
data/processed/parquet/ablation_comparison.csv
```

Colunas: `experiment` (A–F), `lag_days`, `region` (NEB, Sul), `season`, `model`, `metric`, `value`, `delta_vs_C` (diferença em relação ao modelo completo C). Esse arquivo é o principal produto científico da pergunta Q2 (peso das variáveis acopladas).

**Formato para análise colaborativa:**

```
Experimento  Lag   Região  Estação  Modelo     AUC-ROC  delta_vs_C
A_ocean      56d   NEB     MAM      lightgbm   0.71     -0.08
B_atm        56d   NEB     MAM      lightgbm   0.65     -0.14
C_full       56d   NEB     MAM      lightgbm   0.79      0.00
D_no_sub     56d   NEB     MAM      lightgbm   0.73     -0.06
...
```

Com esse formato, a análise colaborativa fica direta: "em `MAM` no NEB com lag de 56 dias, o modelo atmosférico sozinho (B) perde 14 pontos de AUC-ROC versus o completo (C), enquanto o oceânico sozinho (A) perde apenas 8 — o oceano é mais informativo que a atmosfera nessa janela, mas nenhum sozinho captura o sinal completo."

---

### M8 — Diagnóstico Físico Niño 3.4 (Fase 3)

**Módulos:** `thermocline.py`, `ocean_heat.py`, `features/nino.py`.

**Problema atual.** Esses módulos retornam DataArrays xarray — corretos matematicamente, mas sem serialização padronizada para texto numérico. O diagnóstico da Fase 3 precisa de tabelas de eventos para ser analisável.

**Recomendação: `src/nino_brasil/features/nino34_event_table.py`**

```python
def build_nino34_event_table(
    ssta: xr.DataArray,
    d20: xr.DataArray,
    ohc_300: xr.DataArray,
    ohc_700: xr.DataArray,
    thermocline_depth: xr.DataArray,
    *,
    peak_years: list[int],
    time_name: str = "time",
) -> pd.DataFrame:
    """
    Constrói tabela de diagnóstico por evento histórico.

    Colunas:
        year, month,
        nino34_ssta_mean,          # média espacial do Niño 3.4
        nino34_ssta_max,           # máximo (indica EP vs CP)
        nino34_ssta_slope_lon,     # derivada zonal da SSTA (força de crescimento E-W)
        d20_mean_m,                # profundidade média da isoterma 20°C
        d20_anomaly_m,             # anomalia de D20 vs climatologia do treino
        thermocline_depth_mean_m,  # profundidade da termoclina (gradiente max)
        ohc_300_mean,              # OHC 0-300m médio
        ohc_300_anomaly,           # anomalia de OHC 0-300m
        ohc_700_mean,
        ohc_700_anomaly,
        signal_duration_months,    # meses consecutivos com SSTA > 0.5°C
        event_phase,               # "onset"/"peak"/"decay"/"neutral"
        is_peak_year               # bool: ano em el_nino_peak_years
    """
```

**Saídas:**

```
data/processed/parquet/nino34_event_table_monthly.csv   ← série mensal completa 1981-hoje
data/processed/parquet/nino34_peak_comparison.csv       ← apenas anos de pico
data/processed/zarr/features/nino34_physical_signal.zarr ← cubo espacial (já previsto)
```

A `nino34_peak_comparison.csv` tem ~30 linhas (8 eventos × 12 meses de pico) e é o arquivo para perguntar: "em qual evento o slope longitudinal foi mais abrupto?", "1997 ou 2015 teve maior OHC 0–700m no momento do pico?", "a duração do aquecimento subsuperficial em 1997 precedeu a SSTA de superfície em quanto meses?".

---

### M9 — Marine Heatwave OISST (Fase 3 — a implementar)

*Módulo ainda não existe; ver Q5 do Parecer 1.*

**Saída numérica obrigatória** do módulo `detect_mhw`:

```
data/processed/parquet/mhw_catalog_nino34.csv
```

Colunas por evento detectado:

```
onset_date, peak_date, end_date,
duration_days,
max_intensity_c,   # SSTA máxima acima do P90 local
mean_intensity_c,
cumulative_heat,   # sum(intensity) × duration — "dose" de calor
lat_centroid, lon_centroid,
overlap_with_oni   # bool: ONI ≥ 0.5 durante o evento
neb_precip_anomaly_next_90d,  # anomalia de precipitação no NEB nos 90 dias seguintes
sul_precip_anomaly_next_90d   # anomalia de precipitação no Sul nos 90 dias seguintes
```

As duas últimas colunas são o cruzamento direto MHW → impacto no Brasil — calculáveis após a Fase 1 estar completa. Com esse catálogo, a análise colaborativa pergunta: "qual MHW teve a maior `cumulative_heat` e o que aconteceu com a precipitação nos 90 dias seguintes?".

---

### M10 — Distribuições de Cauda (já implementado — ajustes menores)

**O que existe:** `diagnose_dataset_distributions` retorna DataFrame → `distribution_diagnostics.zarr`. Correto.

**O que adicionar:** exportar também em CSV para inspeção direta:

```
data/processed/parquet/distribution_diagnostics.csv
```

E adicionar ao DataFrame as colunas:

```
recommended_distribution,  # "power_law" | "lognormal" | "exponential"
recommendation_confidence, # "strong" | "weak" | "ambiguous"
physical_note              # string: "chuva extrema: lognormal preferível a power law"
```

Isso transforma o diagnóstico de cauda em orientação acionável para a escolha de função de perda nas redes neurais da Fase 6.

---

### M11 — Redes Neurais (Fase 6) — saídas numéricas por época e por fold

Os módulos de rede neural ainda não estão implementados, mas as saídas numéricas precisam ser planejadas antes da implementação.

**Para ConvLSTM / 1D-Transformer:**

```
data/processed/parquet/neural_training_log.csv
```

Por época: `epoch`, `train_loss`, `val_loss`, `train_rmse`, `val_rmse`, `learning_rate`, `elapsed_seconds`. Essencial para detectar overfitting e comparar convergência entre arquiteturas.

```
data/processed/parquet/neural_skill_comparison.csv
```

Por fold/split/lag: `model` (convlstm/transformer/ridge/lightgbm), `lag_days`, `region`, `season`, `metric`, `value`. Permite a comparação direta ML clássico vs neural na mesma tabela.

**Para U-Net (downscaling):**

```
data/processed/parquet/unet_pixel_metrics.csv
```

Por pixel (lat, lon): `rmse`, `bias`, `correlation`, `brier_p10`, `brier_p90`. Esse arquivo tem ~30 mil linhas (grade Brasil 0,25°) e precisa ser agrupado por região para análise colaborativa — mas o arquivo completo existe para auditoria.

**Para XAI neural (occlusion maps):**

```
data/processed/parquet/occlusion_region_importance.csv
```

Por região mascarada do Pacífico × lag × métrica: quanto o skill cai quando aquela região é zerada. Isso é a versão tabular do mapa de oclusão — e permite perguntar: "mascarar o Pacífico central degradou mais o skill que mascarar o leste?".

---

### M12 — Módulos de mapa (`plot_choropleths.py`, `plot_pixel_maps.py`)

**Problema atual.** `save_choropleth` e `save_pixel_map` recebem dados e salvam PNG diretamente, sem nenhum produto numérico intermediário. Qualquer análise do mapa exige abrir o PNG.

**Recomendação: padrão "numeric-first" para todos os plots.**

Refatorar para que cada função de plot tenha uma função-irmã de exportação numérica:

```python
# plot_pixel_maps.py — ATUAL
def save_pixel_map(da, output_path, title, cmap) -> Path:
    ...salva PNG...

# PROPOSTA: separar em duas responsabilidades
def export_pixel_table(
    da: xr.DataArray,
    output_path: str | Path,
    *,
    value_name: str = "value",
    round_decimals: int = 4,
) -> Path:
    """
    Exporta DataArray lat/lon para CSV com colunas: lat, lon, <value_name>.
    Sem figura, sem matplotlib. Puro dado.
    """
    df = da.to_dataframe(name=value_name).reset_index()[["lat", "lon", value_name]]
    df = df.dropna().round(round_decimals)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out

def save_pixel_map(da, output_path, title, cmap) -> Path:
    """Plota PNG a partir do DataArray. Chame export_pixel_table antes."""
    ...código atual...
```

**Mesmo padrão para coropletas:**

```python
def export_choropleth_table(
    gdf: gpd.GeoDataFrame,
    column: str,
    output_path: str | Path,
) -> Path:
    """
    Exporta GeoDataFrame como CSV sem geometria.
    Colunas: nome_uf | nome_municipio, <column>, rank.
    """
    df = gdf[["NM_UF", column]].copy()
    df["rank"] = df[column].rank(ascending=False).astype(int)
    df.to_csv(output_path, index=False)
    return Path(output_path)
```

**Convenção de nomes de arquivo.**
Para cada `mapa_X.png` deve existir um `tabela_X.csv` no mesmo diretório:

```
docs/assets/maps/lag_dominante_neb.png
docs/assets/maps/lag_dominante_neb.csv    ← obrigatório
docs/assets/maps/shap_group_ocean_sul.png
docs/assets/maps/shap_group_ocean_sul.csv ← obrigatório
```

---

## Convenção global de saídas — resumo

| Tipo de análise | Numérico (obrigatório, primeiro) | Visual (opcional, depois) |
|-----------------|----------------------------------|---------------------------|
| Correlação defasada | `lag_correlation_full.zarr` + `lag_correlation_summary_by_region.csv` | mapa de lag dominante PNG |
| PCA/EOF | `pca_variance_summary.csv` + `pca_scores.parquet` | mapa de loadings PNG |
| KNN/análogos | `knn_analogs.csv` | tabela já é o produto final |
| Walk-forward métricas | `walk_forward_metrics_summary.csv` + `baseline_comparison.csv` | curvas de skill vs lag PNG |
| Permutation importance | `importance_top20_by_lag_model.csv` | barplot de importância PNG |
| SHAP | `shap_group_summary.csv` + `shap_event_instances_<ano>.csv` | beeswarm / waterfall PNG |
| Ablation A–F | `ablation_comparison.csv` | heatmap de ganho/perda PNG |
| Diagnóstico Niño 3.4 | `nino34_event_table_monthly.csv` + `nino34_peak_comparison.csv` | série temporal + mapas PNG |
| Marine Heatwave OISST | `mhw_catalog_nino34.csv` | mapa de frequência PNG |
| Distribuições de cauda | `distribution_diagnostics.csv` | Q-Q plot + histograma PNG |
| Treino neural | `neural_training_log.csv` | curva de loss PNG |
| Comparação ML vs NN | `neural_skill_comparison.csv` | tabela ou barplot PNG |
| Occlusion maps | `occlusion_region_importance.csv` | mapa de sensibilidade PNG |
| Mapas de impacto Brasil | `tabela_X.csv` (por pixel ou por UF) | `mapa_X.png` |

---

## Implementação sugerida — padrão mínimo por script

Todo script de análise ou pipeline deve terminar com um bloco como este, antes de qualquer chamada de plot:

```python
# --- EXPORTAÇÃO NUMÉRICA (obrigatória antes de qualquer plot) ---
summary_path = output_dir / "metrics_summary.csv"
metrics.groupby(["model", "lag_days", "season"])[numeric_cols].mean().round(4).to_csv(summary_path)
print(f"numeric summary: {summary_path}")
# --- PLOTS (opcionais, derivados do CSV acima) ---
if args.plot:
    plot_skill_vs_lag(metrics, output_dir / "skill_vs_lag.png")
```

O argumento `--plot` como flag opcional — desligado por padrão — garante que a pipeline de dados e modelagem não dependa de matplotlib/seaborn instalados e funcionando (o que é um problema comum em ambientes headless como WSL sem display).

---

## Próximos passos

1. Adicionar `export_pixel_table` e `export_choropleth_table` em `plot_pixel_maps.py` e `plot_choropleths.py` — trivial, ~30 linhas cada.
2. Adicionar `print_metrics_summary` e salvar `walk_forward_metrics_summary.csv` ao final do `model_pipeline.py` — ~20 linhas.
3. Criar `src/nino_brasil/features/nino34_event_table.py` com `build_nino34_event_table` — central para a Fase 3.
4. Adicionar `importance_top20_by_lag_model.csv` e `xai_method_agreement.csv` ao `model_pipeline.py` — ~30 linhas.
5. Criar `src/nino_brasil/features/marine_heatwave.py` com `detect_mhw` e saída `mhw_catalog_nino34.csv` — novo módulo, ~80 linhas.
6. Adicionar `ablation_comparison.csv` ao wrapper de ablation (a ser criado, ver Parecer 1 item 7 do plano).
7. Adicionar flag `--plot` como opcional em todos os scripts de análise; padrão `False`.
