#!/usr/bin/env python3
"""Materializa tabelas numericas auditaveis para cada figura processada.

Saida canonica:

    data/processed/numeric-tables/<fase>/<figura_sem_ext>/

Cada pasta de figura contem:
- CSVs congelados com os dados usados na representacao;
- figure_manifest.csv com origem, tipo, dimensoes e hash;
- README.md curto para leitura humana.

O objetivo e cientifico: nenhuma figura analitica deve depender apenas de PNG
para ser auditada por humano ou IA. Este script falha em modo --strict se uma
figura existente em data/processed/figures nao tiver mapeamento numerico.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIG_ROOT = PROCESSED / "figures"
NUM_ROOT = PROCESSED / "numeric-tables"
FEAT = PROCESSED / "parquet" / "features"
STATS = PROCESSED / "parquet" / "statistics"
INTERIM = ROOT / "data" / "interim"

MASTER = FEAT / "nino34_master_weekly.csv"

PHASE2_GROUPS = {
    "phase2_sanidade_oceano_superficie": ["nino34_ssta"],
    "phase2_sanidade_oceano_recarga": [
        "d20_m", "tilt_m", "tilt_slope", "ssh_m", "wwv",
        "ohc_0_100", "ohc_0_300", "ohc_0_700", "ohc_300_700",
    ],
    "phase2_sanidade_oceano_temp_perfil": [
        "t50m", "t100m", "t150m", "t200m", "t300m", "t500m", "t700m",
    ],
    "phase2_sanidade_atmosfera_bjerknes": [
        "tau_x_anom", "u10_anom", "v10_anom", "u850_anom", "u200_anom",
        "mslp_anom", "tcwv_anom", "slhf_anom", "sshf_anom", "ssr_anom",
        "str_anom", "omega850_anom", "omega500_anom", "div850_anom",
    ],
}


@dataclass(frozen=True)
class Source:
    kind: str
    path: Path | None = None
    name: str | None = None
    columns: tuple[str, ...] = ()
    note: str = ""


def csv(path: Path, note: str = "") -> Source:
    return Source("csv", path=path, note=note)


def parquet(path: Path, note: str = "") -> Source:
    return Source("parquet", path=path, note=note)


def npy(path: Path, note: str = "") -> Source:
    return Source("npy", path=path, note=note)


def master_subset(name: str, columns: list[str], note: str = "") -> Source:
    return Source("master_subset", name=name, columns=tuple(columns), note=note)


def master_zscore(note: str = "") -> Source:
    return Source("master_zscore", name="nino34_master_weekly_zscore", note=note)


def custom(name: str, note: str = "") -> Source:
    return Source("custom", name=name, note=note)


def pstat(name: str) -> Path:
    return STATS / name


def pfeat(name: str) -> Path:
    return FEAT / name


FIGURE_SOURCES: dict[str, list[Source]] = {
    # Fase 2 - sanidade visual da matriz-mestre.
    "fase2/phase2_sanidade_oceano_superficie.png": [
        master_subset("nino34_master_weekly_oceano_superficie", PHASE2_GROUPS["phase2_sanidade_oceano_superficie"]),
        csv(pfeat("nino34_monthly_oisst.csv"), "periodos EN/LN sombreados na figura"),
    ],
    "fase2/phase2_sanidade_oceano_recarga.png": [
        master_subset("nino34_master_weekly_oceano_recarga", PHASE2_GROUPS["phase2_sanidade_oceano_recarga"]),
        csv(pfeat("nino34_monthly_oisst.csv"), "periodos EN/LN sombreados na figura"),
    ],
    "fase2/phase2_sanidade_oceano_temp_perfil.png": [
        master_subset("nino34_master_weekly_oceano_temp_perfil", PHASE2_GROUPS["phase2_sanidade_oceano_temp_perfil"]),
        csv(pfeat("nino34_monthly_oisst.csv"), "periodos EN/LN sombreados na figura"),
    ],
    "fase2/phase2_sanidade_atmosfera_bjerknes.png": [
        master_subset("nino34_master_weekly_atmosfera_bjerknes", PHASE2_GROUPS["phase2_sanidade_atmosfera_bjerknes"]),
        csv(pfeat("nino34_monthly_oisst.csv"), "periodos EN/LN sombreados na figura"),
    ],
    "fase2/phase2_sanidade_painel_z.png": [
        master_zscore("todas as variaveis padronizadas como no painel"),
        csv(pfeat("nino34_monthly_oisst.csv"), "periodos EN/LN sombreados na figura"),
        csv(pstat("phase2_master_audit.csv")),
        csv(pstat("phase2_master_validation.csv")),
        csv(pstat("phase2_ctd_validation.csv")),
    ],

    # Fase 3 - diagnostico fisico Nino 3.4.
    "fase3/3A1_series_semanais.png": [
        csv(pfeat("phase3_indices_semanais.csv")),
        csv(pstat("phase3A_cobertura_variaveis.csv")),
        csv(pstat("phase3A_fontes_variaveis.csv")),
    ],
    "fase3/3A2_hovmoller_ssta.png": [
        parquet(pfeat("equatorial_pacific_ssta_weekly_by_lon.parquet")),
        csv(pstat("phase3A_picos_epicentros.csv")),
    ],
    "fase3/3A3_hovmoller_sla_taux.png": [
        parquet(pfeat("ssh_equatorial_daily_by_lon_events.parquet")),
        csv(pfeat("phase3_indices_semanais.csv"), "tau_x_anom usado nas setas"),
        csv(pstat("phase3A_taux_quinzenal_janelas.csv")),
    ],
    "fase3/3B1_trajetorias_compostas.png": [
        csv(pstat("phase3B_trajetorias_compostas.csv")),
        csv(pstat("phase3B_eventos_taxas.csv")),
        csv(pstat("phase3B_grupos_classes_noaa.csv")),
    ],
    "fase3/3B2_autocorrelacao.png": [
        csv(pstat("phase3B_autocorrelacao.csv")),
        csv(pstat("phase3B_memoria_persistencia.csv")),
        csv(pstat("phase3_event_lifecycle_en_ln.csv")),
    ],
    "fase3/3B3_mapa_composto_pico.png": [
        csv(pstat("phase3B_mapa_composto_resumo.csv")),
        custom("phase3B_mapa_composto_pico_grid", "grade lat/lon da SSTA composta que desenha o mapa"),
    ],
    "fase3/3B4_faixa_pico_oni.png": [
        csv(pstat("phase3B_faixa_pico_eventos.csv")),
        csv(pfeat("nino34_daily_oisst.csv"), "serie diaria usada para recompor ONI local mensal"),
    ],
    "fase3/3C1_heatmap_lags.png": [
        csv(pstat("phase3C_lag_correlacoes.csv")),
        csv(pstat("phase3C_ranking_lags.csv")),
    ],
    "fase3/3C2_mapa_lon_lag.png": [
        csv(pstat("phase3C_mapa_lon_lag.csv")),
        parquet(pfeat("equatorial_pacific_ssta_weekly_by_lon.parquet")),
    ],
    "fase3/3D1_forest_ic95.png": [
        csv(pstat("phase3D_ranking_significativo.csv")),
        csv(pstat("phase3D_testes_completos.csv")),
        csv(pstat("phase3D_forest_ic95_legenda.csv")),
    ],
    "fase3/3D2_mapa_lon_lag_fdr.png": [
        csv(pstat("phase3D_testes_completos.csv")),
        csv(pstat("phase3D_mapa_fdr_resumo.csv")),
    ],
    "fase3/Fig_3E1_sensibilidade_bootstrap_loo.png": [
        csv(pstat("phase3E_sensibilidade_resumo.csv")),
        csv(pstat("phase3E_bootstrap_blocos.csv")),
        csv(pstat("phase3E_bootstrap_correlacoes.csv")),
        csv(pstat("phase3E_leave_one_event_out.csv")),
        csv(pstat("phase3D_ranking_significativo.csv")),
    ],
    "fase3/Fig_3E2_influencia_eventos_loo.png": [
        csv(pstat("phase3E_leave_one_event_out.csv")),
        csv(pstat("phase3E_sensibilidade_resumo.csv")),
        csv(pstat("phase3_events_en_ln.csv")),
    ],
    "fase3/3F1_hovmoller_sla_kelvin.png": [
        parquet(pfeat("ssh_equatorial_daily_by_lon_events.parquet")),
        csv(pstat("phase3F_kelvin_eventos_resumo.csv")),
        csv(pstat("phase3F_kelvin_setas.csv")),
    ],
    "fase3/3F2_taux_sla_eventos.png": [
        csv(pstat("phase3F_kelvin_eventos_resumo.csv")),
        csv(pstat("phase3F_kelvin_setas.csv")),
        csv(pfeat("phase3_indices_semanais.csv"), "tau_x e SSTA semanais usados na leitura temporal"),
    ],
    "fase3/3G1_composto_ssta_noaa.png": [
        csv(pstat("phase3G_composto_ssta_classes_noaa.csv")),
        csv(pstat("phase3G_eventos_ssta.csv")),
    ],
    "fase3/3G2_escalonamento_ssta.png": [
        csv(pstat("phase3G_escalonamento_ssta.csv")),
        csv(pstat("phase3G_eventos_ssta.csv")),
    ],
    "fase3/3G3_mapa_ssta_lon.png": [
        csv(pstat("phase3G_mapa_ssta_lon_eventos_forte_super.csv")),
        csv(pstat("phase3G_estado_atual.csv")),
    ],
    "fase3/3H1_compostos_onset.png": [
        csv(pstat("phase3H_compostos_onset.csv")),
        csv(pstat("phase3H_estado_precursor_por_classe.csv")),
        csv(pstat("phase3H_separacao_genese.csv")),
        csv(pstat("phase3H_proveniencia_eventos.csv")),
    ],
    "fase3/3H2_ciclo_vida.png": [
        csv(pstat("phase3H_ciclo_vida_media.csv")),
        csv(pstat("phase3H_fases_ciclo_vida.csv")),
        csv(pstat("phase3H_grupos_classes_noaa.csv")),
    ],
    "fase3/3H3_ciclo_vida_subsuperficie_atmosfera.png": [
        csv(pstat("phase3H_ciclo_vida_media_subsuperficie_atmosfera.csv")),
        csv(pstat("phase3H_fases_ciclo_vida.csv")),
    ],
    "fase3/3I1_sintese_parecer.png": [
        csv(pstat("phase3I_conclusoes_decisao.csv")),
        csv(pstat("phase3I_classificacao_noaa_oni.csv")),
        csv(pstat("phase3I_estado_2026.csv")),
        csv(pstat("phase3I_medias_classes_noaa.csv")),
    ],
    "fase3/3I2_antecipacao_pico.png": [
        csv(pstat("phase3I_conjunto_antecipacao_pico.csv")),
        csv(pstat("phase3I_skill_horizontes.csv")),
        csv(pstat("phase3I_skill_por_variavel.csv")),
        csv(pstat("phase3I_modelos_candidatos.csv")),
    ],
    "fase3/3I3_previsao_condicional_nested.png": [
        csv(pstat("phase3I_nested_loo_metricas.csv")),
        csv(pstat("phase3I_nested_loo_eventos.csv")),
        csv(pstat("phase3I_nested_loo_selecao.csv")),
        csv(pstat("phase3I_projecao_pico_2026.csv")),
    ],
    "fase3/3K1_skill_loo_nested.png": [
        csv(pstat("phase3K_previsao_pico_nested_loo_metricas.csv")),
        csv(pstat("phase3K_previsao_pico_nested_loo_eventos.csv")),
        csv(pstat("phase3K_previsao_pico_nested_loo_selecao.csv")),
        csv(pstat("phase3K_previsao_pico_loo.csv")),
    ],
    "fase3/3K2_scree.png": [
        csv(pstat("phase3K_pca_variancia.csv")),
    ],
    "fase3/3K3_biplot.png": [
        csv(pstat("phase3K_pca_loadings.csv")),
        csv(pstat("phase3K_pcs_explicados.csv")),
        csv(pstat("phase3K_conjunto_indispensavel.csv")),
    ],
    "fase3/3R1_galeria_figuras_fase3.png": [
        csv(pstat("phase3_figuras_catalogo.csv"), "galeria visual, nao analise independente"),
    ],
    "fase3/phase3L_ciclo_vida_en_ln.png": [
        csv(pstat("phase3_events_en_ln.csv")),
        csv(pstat("phase3_fases_semanais_en_ln.csv")),
        csv(pstat("phase3_event_lifecycle_en_ln.csv")),
        csv(pstat("phase3_fase_stats_variaveis.csv")),
    ],
    "fase3/phase3L_duracao_fases_en_ln.png": [
        csv(pstat("phase3_duracao_por_tipo_classe.csv")),
    ],
    "fase3/phase3L_discriminantes_heatmap.png": [
        csv(pstat("phase3_discriminantes_por_periodo.csv")),
        csv(pstat("phase3_fase_stats_variaveis.csv")),
    ],
    "fase3/phase3L_pca_por_fase.png": [
        csv(pstat("phase3_pca_por_fase.csv")),
        csv(pstat("phase3_pca_loadings_por_fase.csv")),
    ],

    # Fase 4 - teleconexao estatistica Brasil.
    "fase4/phase40_cobertura_dados.png": [
        csv(pstat("phase40_inventario_pacifico.csv")),
        csv(pstat("phase40_inventario_alvo.csv")),
    ],
    "fase4/phase4A_ciclo_enso.png": [
        csv(pstat("phase4A_eventos_enso.csv")),
        csv(pstat("phase4A_fases_semanais.csv")),
        csv(pstat("phase4A_evento_fases_cronologia.csv")),
    ],
    "fase4/Fig_4A1_ciclo_enso_fases.png": [
        csv(pstat("phase4A_eventos_enso.csv")),
        csv(pstat("phase4A_fases_semanais.csv")),
        csv(pstat("phase4A_evento_fases_cronologia.csv")),
    ],
    "fase4/phase4A_duracao_fases.png": [
        csv(pstat("phase4A_duracao_fases_por_evento.csv")),
        csv(pstat("phase4A_evento_fases_cronologia.csv")),
        csv(pstat("phase4A_fases_resumo.csv")),
    ],
    "fase4/Fig_4A2_duracao_distribuicao_fases.png": [
        csv(pstat("phase4A_duracao_fases_por_evento.csv")),
        csv(pstat("phase4A_evento_fases_cronologia.csv")),
        csv(pstat("phase4A_fases_resumo.csv")),
    ],
    "fase4/phase4A_plano_fase.png": [
        csv(pstat("phase4A_fases_resumo.csv")),
        csv(pstat("phase4A_fases_semanais.csv")),
        csv(pstat("phase4A_evento_fases_cronologia.csv")),
    ],
    "fase4/Fig_4A3_plano_cronologico_oni_tendencia.png": [
        csv(pstat("phase4A_fases_resumo.csv")),
        csv(pstat("phase4A_fases_semanais.csv")),
        csv(pstat("phase4A_evento_fases_cronologia.csv")),
    ],
    "fase4/phase4B_heatmap_determinantes.png": [
        csv(pstat("phase4B_determinantes_fases.csv")),
        csv(pstat("phase4B_discriminancia_fases.csv")),
    ],
    "fase4/phase4B_ranking_discriminancia.png": [
        csv(pstat("phase4B_discriminancia_fases.csv")),
        csv(pstat("phase4B_fase_intensidade_pico.csv")),
    ],
    "fase4/phase4B_relacoes_pares.png": [
        csv(pstat("phase4B_relacoes_pares_fases.csv")),
        csv(pstat("phase4B_fase_intensidade_pico.csv")),
    ],
    "fase4/phase4B_estrutura_correlacao.png": [
        csv(pstat("phase4B_matriz_correlacao_por_fase.csv")),
        csv(pstat("phase4B_estrutura_correlacao_distancias.csv")),
    ],
    "fase4/phase4C_mapas_brasil.png": [
        csv(pstat("phase4C_best_lag_pixel.csv")),
        csv(pstat("phase4C_lag_resposta_neb_sul.csv")),
        csv(pstat("phase4C_janelas_lag_variavel.csv")),
        csv(pfeat("phase4_chirps_pixels.csv"), "coordenadas dos pixels CHIRPS usados no atlas"),
    ],
    "fase4/phase4C_recorte_NEB.png": [
        csv(pstat("phase4C_best_lag_pixel.csv")),
        csv(pstat("phase4C_lag_resposta_neb_sul.csv")),
        csv(pstat("phase4C_janelas_lag_variavel.csv")),
        csv(pfeat("phase4_chirps_pixels.csv"), "coordenadas dos pixels CHIRPS usados no recorte"),
    ],
    "fase4/phase4C_recorte_SUL.png": [
        csv(pstat("phase4C_best_lag_pixel.csv")),
        csv(pstat("phase4C_lag_resposta_neb_sul.csv")),
        csv(pstat("phase4C_janelas_lag_variavel.csv")),
        csv(pfeat("phase4_chirps_pixels.csv"), "coordenadas dos pixels CHIRPS usados no recorte"),
    ],
    "fase4/phase4C_lags_neb_sul.png": [
        csv(pstat("phase4C_lag_resposta_neb_sul.csv")),
        csv(pstat("phase4C_janelas_lag_variavel.csv")),
    ],
    "fase4/phase4D_mapa_clusters.png": [
        csv(pstat("phase4D_clusters_pixels.csv")),
        csv(pstat("phase4D_cluster_ranking.csv")),
        csv(pfeat("phase4_chirps_pixels.csv"), "coordenadas dos pixels CHIRPS agrupados"),
    ],
    "fase4/phase4D_perfis_clusters.png": [
        csv(pstat("phase4D_cluster_lags_por_sinal.csv")),
        csv(pstat("phase4D_cluster_ranking.csv")),
    ],
    "fase4/phase4D_sintese_gate.png": [
        csv(pstat("phase4D_gate_hipotese.csv")),
        csv(pstat("phase4D_estabilidade.csv")),
    ],
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def slug_from_figure(fig_rel: str) -> tuple[str, str]:
    phase, filename = fig_rel.split("/", 1)
    return phase, Path(filename).stem


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_df(df: pd.DataFrame, out: Path) -> tuple[int, int]:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return int(df.shape[0]), int(df.shape[1])


def read_csv_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_parquet_table(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.RangeIndex):
        index_name = df.index.name or "index"
        df = df.reset_index().rename(columns={index_name: index_name})
    return df


def read_npy_table(path: Path) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim == 1:
        return pd.DataFrame({"index": np.arange(arr.shape[0]), "value": arr})
    if arr.ndim == 2:
        data = pd.DataFrame(arr)
        data.insert(0, "row_index", np.arange(arr.shape[0]))
        data.columns = ["row_index"] + [f"col_{i}" for i in range(arr.shape[1])]
        return data
    flat = pd.DataFrame({"flat_index": np.arange(arr.size), "value": arr.ravel()})
    flat["original_shape"] = "x".join(map(str, arr.shape))
    return flat


def master_subset_table(columns: tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_csv(MASTER, parse_dates=["week_ending_sunday"])
    keep = ["week_ending_sunday"] + [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    out = df[keep].copy()
    if missing:
        out.attrs["missing_columns"] = ",".join(missing)
    return out


def master_zscore_table() -> pd.DataFrame:
    df = pd.read_csv(MASTER, parse_dates=["week_ending_sunday"])
    values = df.drop(columns=[c for c in ["week_ending_sunday", "ocean_source_code"] if c in df.columns])
    values = values.apply(pd.to_numeric, errors="coerce")
    z = (values - values.mean()) / values.std(ddof=1)
    z.insert(0, "week_ending_sunday", df["week_ending_sunday"])
    return z


def phase3b_composite_grid() -> pd.DataFrame:
    """Reconstroi a grade numerica da figura 3B3.

    Usa o mesmo principio do notebook 3B: eventos El Nino muito fortes, campo
    mensal OISST no mes do pico, anomalia contra climatologia 1991-2020 do mesmo
    mes e media composta final.
    """
    import xarray as xr

    events = pd.read_csv(pfeat("nino34_oisst_event_reference.csv"),
                         parse_dates=["event_start", "event_end", "peak_time"])
    if "peak_class" in events.columns:
        strong = events[events["peak_class"].isin(["super_el_nino", "very_strong_el_nino", "muito_forte"])].copy()
    else:
        peak = pd.to_numeric(events.get("peak_oni_local_c", events.get("peak_ssta_c")), errors="coerce")
        strong = events[peak >= 2.0].copy()

    cache = INTERIM / "fase3_map_cache"

    def month_field(year: int, month: int):
        cached = cache / f"sst_month_{year}_{month:02d}.nc"
        if cached.exists():
            return xr.open_dataarray(cached).load()
        zarr = ROOT / f"data/processed/zarr/cpc_noaa/oisst/sst.day.mean.{year}.zarr"
        raw = ROOT / f"data/raw/cpc_noaa/oisst/sst.day.mean.{year}.nc"
        if zarr.exists():
            ds = xr.open_zarr(zarr, consolidated=False)
        else:
            ds = xr.open_dataset(raw)
        try:
            return ds["sst"].sel(time=f"{year}-{month:02d}").mean("time").sel(lat=slice(-30, 30)).load()
        finally:
            ds.close()

    fields = []
    labels = []
    for _, event in strong.iterrows():
        year = int(event["peak_time"].year)
        month = int(event["peak_time"].month)
        clim = xr.concat([month_field(y, month) for y in range(1991, 2021)], "clim_year").mean("clim_year")
        fields.append((month_field(year, month) - clim).load())
        labels.append(f"{year}-{month:02d}")

    if not fields:
        raise RuntimeError("Nenhum evento muito_forte encontrado para 3B3.")

    comp = xr.concat(fields, "event").mean("event")
    df = comp.to_dataframe(name="ssta_anom_c").reset_index()
    df["eventos_compostos"] = ",".join(labels)
    df["n_eventos"] = len(labels)
    return df


CUSTOM_BUILDERS: dict[str, Callable[[], pd.DataFrame]] = {
    "phase3B_mapa_composto_pico_grid": phase3b_composite_grid,
}


def source_to_dataframe(source: Source) -> tuple[pd.DataFrame, str]:
    if source.kind == "csv":
        assert source.path is not None
        if not source.path.exists():
            raise FileNotFoundError(source.path)
        return read_csv_table(source.path), source.path.stem
    if source.kind == "parquet":
        assert source.path is not None
        if not source.path.exists():
            raise FileNotFoundError(source.path)
        return read_parquet_table(source.path), source.path.stem
    if source.kind == "npy":
        assert source.path is not None
        if not source.path.exists():
            raise FileNotFoundError(source.path)
        return read_npy_table(source.path), source.path.stem
    if source.kind == "master_subset":
        assert source.name is not None
        return master_subset_table(source.columns), source.name
    if source.kind == "master_zscore":
        assert source.name is not None
        return master_zscore_table(), source.name
    if source.kind == "custom":
        assert source.name is not None
        return CUSTOM_BUILDERS[source.name](), source.name
    raise ValueError(f"Tipo de fonte desconhecido: {source.kind}")


def write_readme(out_dir: Path, figure_rel: str, records: list[dict[str, object]]) -> None:
    lines = [
        f"# Tabelas numericas - {figure_rel}",
        "",
        "Esta pasta materializa os dados usados para auditar a figura correspondente.",
        "Os CSVs abaixo sao artefatos congelados de reproducao/inspecao.",
        "",
        "| arquivo | origem | tipo | linhas | colunas |",
        "|---|---|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            f"| `{Path(str(r['table_path'])).name}` | `{r['source_path']}` | "
            f"{r['source_kind']} | {r['rows']} | {r['columns']} |"
        )
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def prepare_output_root(force: bool) -> None:
    if NUM_ROOT.exists():
        if not force:
            raise FileExistsError(f"{NUM_ROOT} ja existe; use --force para regenerar.")
        resolved = NUM_ROOT.resolve()
        processed = PROCESSED.resolve()
        if not resolved.is_relative_to(processed):
            raise RuntimeError(f"Recusa remover pasta fora de {processed}: {resolved}")
        shutil.rmtree(NUM_ROOT)
    NUM_ROOT.mkdir(parents=True, exist_ok=True)


def export_all(*, strict: bool, force: bool) -> pd.DataFrame:
    prepare_output_root(force)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    figure_paths = sorted(p.relative_to(FIG_ROOT).as_posix() for p in FIG_ROOT.rglob("*.png"))
    mapped = set(FIGURE_SOURCES)
    existing = set(figure_paths)
    missing_map = sorted(existing - mapped)
    stale_map = sorted(mapped - existing)
    if strict and missing_map:
        raise RuntimeError("Figuras sem mapeamento numerico: " + ", ".join(missing_map))

    global_records: list[dict[str, object]] = []
    for figure_rel in figure_paths:
        sources = FIGURE_SOURCES.get(figure_rel)
        if not sources:
            continue
        phase, stem = slug_from_figure(figure_rel)
        out_dir = NUM_ROOT / phase / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        local_records: list[dict[str, object]] = []

        for idx, source in enumerate(sources, 1):
            df, source_name = source_to_dataframe(source)
            table_name = f"{idx:02d}_{source_name}.csv"
            out_path = out_dir / table_name
            rows, cols = write_df(df, out_path)
            source_path = rel(source.path) if source.path is not None else source.name
            record = {
                "generated_at_utc": generated_at,
                "phase": phase,
                "figure_path": f"data/processed/figures/{figure_rel}",
                "figure_exists": True,
                "figure_numeric_dir": rel(out_dir),
                "table_path": rel(out_path),
                "source_kind": source.kind,
                "source_path": source_path,
                "rows": rows,
                "columns": cols,
                "sha256": sha256(out_path),
                "note": source.note,
            }
            local_records.append(record)
            global_records.append(record)

        pd.DataFrame(local_records).to_csv(out_dir / "figure_manifest.csv", index=False)
        write_readme(out_dir, figure_rel, local_records)

    manifest = pd.DataFrame(global_records)
    manifest.to_csv(NUM_ROOT / "figure_numeric_tables_manifest.csv", index=False)

    root_readme = [
        "# Numeric tables",
        "",
        "Camada obrigatoria de auditoria figura -> tabela numerica.",
        "",
        f"Gerado em UTC: {generated_at}",
        f"Figuras encontradas: {len(figure_paths)}",
        f"Figuras exportadas: {manifest['figure_path'].nunique() if not manifest.empty else 0}",
        f"Tabelas exportadas: {len(manifest)}",
        "",
        "Manifesto global: `figure_numeric_tables_manifest.csv`.",
        "",
        "Regra cientifica: toda figura analitica em `data/processed/figures` deve",
        "ter dados numericos auditaveis nesta pasta. Execute:",
        "",
        "```powershell",
        r".\.venv\Scripts\python scripts\export_numeric_tables_for_figures.py --force --strict",
        "```",
    ]
    if stale_map:
        root_readme.extend(["", "Mapeamentos sem PNG atual:", *[f"- `{item}`" for item in stale_map]])
    (NUM_ROOT / "README.md").write_text("\n".join(root_readme), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="remove e recria data/processed/numeric-tables")
    parser.add_argument("--strict", action="store_true", help="falha se existir PNG sem mapeamento numerico")
    args = parser.parse_args(argv)
    manifest = export_all(strict=args.strict, force=args.force)
    nfig = manifest["figure_path"].nunique() if not manifest.empty else 0
    print(f"numeric-tables: {nfig} figuras, {len(manifest)} tabelas exportadas em {rel(NUM_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
