#!/usr/bin/env python3
"""Compara UFS+GLORYS diário com CTD/WOD, TAO/TRITON e Argo.

As observações são pareadas ao nó diário, horizontal e vertical mais próximo.
Para não inflar o tamanho amostral com dezenas de profundidades do mesmo perfil,
os testes são feitos sobre o resíduo médio por perfil/boia-dia. A saída permanece
separada do master semanal: valida a fonte oceânica, mas não preenche a série.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import gsw
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.stats.significance import (
    benjamini_hochberg_fdr,
    correlation_p_value,
    effective_sample_size,
)


OCEAN_ROOT = ROOT / "data/processed/zarr/ocean_daily"
INSITU_ROOTS = {
    "CTD/WOD": ROOT / "data/processed/zarr/ctd_noaa/wod",
    "TAO/TRITON": ROOT / "data/processed/zarr/validation/tao_triton",
    "Argo": ROOT / "data/processed/zarr/validation/argo",
}
OUT_ROOT = ROOT / "data/processed/parquet/statistics"
PROFILE_OUTPUT = OUT_ROOT / "phase2_insitu_profile_residuals.csv"
SUMMARY_OUTPUT = OUT_ROOT / "phase2_insitu_validation_summary.csv"
CONTRACT_OUTPUT = OUT_ROOT / "phase2_ocean_source_comparison.csv"
ORAS_PAIRS_OUTPUT = OUT_ROOT / "phase2_ufs_glorys_oras_monthly_pairs.csv"
ORAS_SUMMARY_OUTPUT = OUT_ROOT / "phase2_ufs_glorys_oras_comparison_summary.csv"

ORAS_TO_DAILY_FEATURE = {
    "d20_nino34_mean_m": "d20_nino34_mean_m",
    "ohc_0_300_nino34_j_m2": "ohc_0_300_nino34_j_m2",
    "ohc_0_700_nino34_j_m2": "ohc_0_700_nino34_j_m2",
    "ssh_nino34_mean_m": "ssh_nino34_mean_m",
    "sss_nino34_mean": "sss_nino34_mean",
    "temperature_50m_nino34_c": "temperature_50m_nino34_c",
    "temperature_100m_nino34_c": "temperature_100m_nino34_c",
    "temperature_150m_nino34_c": "temperature_150m_nino34_c",
    "temperature_200m_nino34_c": "temperature_200m_nino34_c",
    "temperature_300m_nino34_c": "temperature_300m_nino34_c",
    "temperature_500m_nino34_c": "temperature_500m_nino34_c",
    "temperature_700m_nino34_c": "temperature_700m_nino34_c",
    "thermocline_tilt_east_minus_west_m": "thermocline_tilt_m",
    "wwv_equatorial_pacific_m3": "wwv_equatorial_pacific_m3",
}


def _model_stores() -> list[tuple[pd.Timestamp, pd.Timestamp, Path, str]]:
    stores: list[tuple[pd.Timestamp, pd.Timestamp, Path, str]] = []
    patterns = (
        ("noaa_ufs", "UFS"),
        ("glorys12", "GLORYS12"),
        ("glorys12_operational", "GLO12 operacional"),
    )
    for folder, label in patterns:
        for path in sorted((OCEAN_ROOT / folder).rglob("*.zarr")):
            with xr.open_zarr(path, consolidated=None) as ds:
                if "time" not in ds or not ds.sizes.get("time", 0):
                    continue
                times = pd.DatetimeIndex(ds.time.values).normalize()
                stores.append((times.min(), times.max(), path, label))
    return stores


def _select_store(day: pd.Timestamp, stores) -> tuple[Path, str] | None:
    candidates = [item for item in stores if item[0] <= day <= item[1]]
    if not candidates:
        return None
    # A série oficial usa UFS antes de 1993 e GLORYS depois. Entre stores
    # operacionais sobrepostos, prefira o que alcança a data final mais recente.
    if day.year < 1993:
        candidates = [item for item in candidates if item[3] == "UFS"] or candidates
    else:
        candidates = [item for item in candidates if item[3] != "UFS"] or candidates
    selected = max(candidates, key=lambda item: (item[1], item[0]))
    return selected[2], selected[3]


def _long_frame(
    *, source: str, variable: str, profile_id, time, latitude, longitude,
    depth, observed, qc=None,
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "fonte_insitu": source,
        "variavel": variable,
        "perfil_id": np.asarray(profile_id).astype(str),
        "time": pd.to_datetime(np.asarray(time), errors="coerce"),
        "latitude": np.asarray(latitude, dtype=float),
        "longitude": np.mod(np.asarray(longitude, dtype=float), 360.0),
        "depth_m": np.asarray(depth, dtype=float),
        "observado": np.asarray(observed, dtype=float),
    })
    valid = (
        frame["time"].notna() & np.isfinite(frame["latitude"])
        & np.isfinite(frame["longitude"]) & np.isfinite(frame["depth_m"])
        & np.isfinite(frame["observado"]) & frame["depth_m"].between(0, 800)
    )
    if qc is not None:
        valid &= pd.Series(np.asarray(qc), index=frame.index).isin([1, 2])
    return frame.loc[valid].reset_index(drop=True)


def _ctd_frames() -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in sorted(INSITU_ROOTS["CTD/WOD"].rglob("*.zarr")):
        with xr.open_zarr(path, consolidated=None) as ds:
            n_profile = ds.sizes.get("profile", 0)
            n_depth = ds.sizes.get("depth", 0)
            if not n_profile or not n_depth:
                continue
            prefix = path.stem
            profile = np.repeat([f"{prefix}:{value}" for value in ds.profile.values], n_depth)
            time = np.repeat(ds.time.values, n_depth)
            lat = np.repeat(ds.lat.values, n_depth)
            lon = np.repeat(ds.lon_360.values if "lon_360" in ds else ds.lon.values, n_depth)
            depth = np.tile(ds.depth.values, n_profile)
            mappings = (
                ("temperature", "conservative_temperature"),
                ("salinity", "practical_salinity"),
            )
            for variable, name in mappings:
                if name in ds:
                    frames.append(_long_frame(
                        source="CTD/WOD", variable=variable, profile_id=profile,
                        time=time, latitude=lat, longitude=lon, depth=depth,
                        observed=ds[name].values.reshape(-1),
                    ))
    return frames


def _tao_frames() -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    specs = (
        ("temperature", "T_20", "QT_5020"),
        ("salinity", "S_41", "QS_5041"),
    )
    for variable, value_name, qc_name in specs:
        for path in sorted((INSITU_ROOTS["TAO/TRITON"] / variable).rglob("*.zarr")):
            with xr.open_zarr(path, consolidated=None) as ds:
                day = pd.DatetimeIndex(ds.time.values).strftime("%Y%m%d")
                profile = np.char.add(np.asarray(ds.station.values, dtype=str), np.char.add(":", day))
                frames.append(_long_frame(
                    source="TAO/TRITON", variable=variable, profile_id=profile,
                    time=ds.time.values, latitude=ds.latitude.values,
                    longitude=ds.longitude.values, depth=ds.depth.values,
                    observed=ds[value_name].values, qc=ds[qc_name].values,
                ))
    return frames


def _argo_frames() -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in sorted(INSITU_ROOTS["Argo"].rglob("*.zarr")):
        with xr.open_zarr(path, consolidated=None) as ds:
            day = pd.DatetimeIndex(ds.time.values).strftime("%Y%m%d%H%M")
            profile = np.char.add(np.asarray(ds.platform_number.values, dtype=str), np.char.add(":", day))
            pressure = np.asarray(ds.pres.values, dtype=float)
            salinity = np.asarray(ds.psal.values, dtype=float)
            temperature = np.asarray(ds.temp.values, dtype=float)
            lon = np.asarray(ds.longitude.values, dtype=float)
            lat = np.asarray(ds.latitude.values, dtype=float)
            try:
                absolute_salinity = gsw.SA_from_SP(salinity, pressure, lon, lat)
                potential_temperature = gsw.pt0_from_t(absolute_salinity, temperature, pressure)
            except (TypeError, ValueError, FloatingPointError):
                potential_temperature = temperature
            frames.extend([
                _long_frame(
                    source="Argo", variable="temperature", profile_id=profile,
                    time=ds.time.values, latitude=lat, longitude=lon, depth=pressure,
                    observed=potential_temperature, qc=ds.temp_qc.values,
                ),
                _long_frame(
                    source="Argo", variable="salinity", profile_id=profile,
                    time=ds.time.values, latitude=lat, longitude=lon, depth=pressure,
                    observed=salinity, qc=ds.psal_qc.values,
                ),
            ])
    return frames


def _collocate(observations: pd.DataFrame, stores) -> pd.DataFrame:
    observations = observations.copy()
    observations["dia"] = observations["time"].dt.normalize()
    day_selection = {
        day: _select_store(pd.Timestamp(day), stores)
        for day in observations["dia"].dropna().drop_duplicates()
    }
    observations["modelo_selecao"] = observations["dia"].map(day_selection)
    observations = observations.loc[observations["modelo_selecao"].notna()].copy()
    if observations.empty:
        return pd.DataFrame()
    observations["modelo_path"] = observations["modelo_selecao"].map(lambda value: value[0])
    observations["fonte_modelo"] = observations["modelo_selecao"].map(lambda value: value[1])
    pieces: list[pd.DataFrame] = []
    for (path, model_source), group in observations.groupby(["modelo_path", "fonte_modelo"], sort=True):
        model_variable = "potential_temperature" if group["variavel"].iat[0] == "temperature" else "salinity"
        with xr.open_zarr(path, consolidated=None) as ds:
            if model_variable not in ds:
                continue
            points = group.reset_index(drop=True)
            indexer = {
                "time": xr.DataArray(points["dia"].to_numpy(dtype="datetime64[ns]"), dims="observation"),
                "lat": xr.DataArray(points["latitude"].to_numpy(), dims="observation"),
                "lon": xr.DataArray(points["longitude"].to_numpy(), dims="observation"),
                "depth": xr.DataArray(points["depth_m"].to_numpy(), dims="observation"),
            }
            selected_values = ds[model_variable].sel(indexer, method="nearest")
            points["modelo"] = np.asarray(selected_values.values, dtype=float)
            points["fonte_modelo"] = model_source
            points["modelo_zarr"] = str(path.relative_to(ROOT))
            points["depth_modelo_m"] = np.asarray(selected_values.depth.values, dtype=float)
            points["residuo_modelo_menos_insitu"] = points["modelo"] - points["observado"]
            points = points.drop(columns=["modelo_selecao", "modelo_path"], errors="ignore")
            pieces.append(points)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _profile_residuals(collocations: pd.DataFrame) -> pd.DataFrame:
    if collocations.empty:
        return pd.DataFrame()
    return collocations.groupby(
        ["fonte_insitu", "fonte_modelo", "variavel", "perfil_id"], as_index=False
    ).agg(
        time=("time", "min"), latitude=("latitude", "mean"),
        longitude=("longitude", "mean"), niveis_pareados=("modelo", "count"),
        observado_medio=("observado", "mean"), modelo_medio=("modelo", "mean"),
        residuo_medio=("residuo_modelo_menos_insitu", "mean"),
        residuo_absoluto_medio=("residuo_modelo_menos_insitu", lambda x: np.mean(np.abs(x))),
        rmse_perfil=("residuo_modelo_menos_insitu", lambda x: np.sqrt(np.mean(np.square(x)))),
    )


def _bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def _summary(profiles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for position, (keys, group) in enumerate(profiles.groupby(["fonte_insitu", "variavel"], sort=True)):
        source, variable = keys
        residual = group["residuo_medio"].to_numpy(dtype=float)
        valid = residual[np.isfinite(residual)]
        if len(valid) >= 10 and np.any(valid != 0):
            p_value = float(wilcoxon(valid, alternative="two-sided", zero_method="wilcox").pvalue)
        else:
            p_value = np.nan
        ci_low, ci_high = _bootstrap_mean_ci(valid, 2600 + position)
        correlation = group[["observado_medio", "modelo_medio"]].corr().iloc[0, 1] if len(group) >= 3 else np.nan
        rows.append({
            "fonte_insitu": source, "variavel": variable,
            "n_perfis_independentes": int(len(valid)),
            "n_niveis_pareados": int(group["niveis_pareados"].sum()),
            "inicio": group["time"].min(), "fim": group["time"].max(),
            "bias_modelo_menos_insitu": float(np.mean(valid)) if len(valid) else np.nan,
            "bias_ic95_inf_bootstrap_perfil": ci_low,
            "bias_ic95_sup_bootstrap_perfil": ci_high,
            "mae_medio_perfil": float(group["residuo_absoluto_medio"].mean()),
            "rmse_medio_perfil": float(group["rmse_perfil"].mean()),
            "correlacao_entre_medias_de_perfil": correlation,
            "p_wilcoxon_bias_zero": p_value,
            "unidade": "degC" if variable == "temperature" else "1e-3",
            "unidade_amostral_teste": "perfil/boia-dia; profundidades agregadas antes do teste",
        })
    summary = pd.DataFrame(rows)
    rejected, q_values = benjamini_hochberg_fdr(summary["p_wilcoxon_bias_zero"].to_numpy(), alpha=0.05)
    summary["q_fdr_bh"] = q_values
    summary["diferenca_significativa_fdr_05"] = rejected
    summary["conclusao"] = np.select(
        [summary["n_perfis_independentes"].lt(10), summary["diferenca_significativa_fdr_05"]],
        ["amostra_insuficiente", "vies_sistematico_detectado"],
        default="sem_evidencia_de_vies_sistematico",
    )
    return summary


def _source_contract(stores) -> pd.DataFrame:
    starts = pd.DataFrame(stores, columns=["inicio", "fim", "path", "fonte"])
    coverage = starts.groupby("fonte", as_index=False).agg(inicio=("inicio", "min"), fim=("fim", "max"), zarrs=("path", "nunique"))
    coverage["frequencia_nativa"] = "diaria"
    coverage["frequencia_verificada_nos_timestamps"] = "1 timestamp por dia"
    coverage["grade_processada"] = "41 latitudes x 641 longitudes; 0,25 grau"
    coverage["profundidades_processadas"] = coverage["fonte"].map({"UFS": "48 niveis ate 770,665 m", "GLORYS12": "34 niveis ate 763,333 m", "GLO12 operacional": "34 niveis ate 763,333 m"})
    coverage["variaveis_diretas"] = "temperatura potencial; salinidade; altura da superficie do mar"
    coverage["variaveis_calculadas"] = "D20; OHC 0-100m; OHC 0-300m; OHC 0-700m; OHC 300-700m; WWV; inclinacao da termoclina"
    coverage["destino_fase2"] = "media de dias observados em semanas completas W-SUN"
    oras_root = ROOT / "data/processed/zarr/ocean_monthly/oras5"
    oras_paths = sorted(oras_root.rglob("*.zarr")) if oras_root.exists() else []
    oras_time_paths = sorted(oras_root.rglob("*depth_of_20_c_isotherm*.zarr"))
    oras_times: list[pd.Timestamp] = []
    for path in oras_time_paths:
        with xr.open_zarr(path, consolidated=None) as ds:
            oras_times.extend(pd.DatetimeIndex(ds.time.values).tolist())
    oras = pd.DataFrame([{
        "fonte": "ORAS5 CDS (excluida do fluxo)",
        "inicio": min(oras_times) if oras_times else pd.NaT,
        "fim": max(oras_times) if oras_times else pd.NaT,
        "zarrs": len(oras_paths), "frequencia_nativa": "media mensal",
        "frequencia_verificada_nos_timestamps": "12 timestamps/ano; intervalos de 28 a 31 dias",
        "grade_processada": "41 latitudes x 641 longitudes; 0,25 grau",
        "profundidades_processadas": "44 niveis ate aproximadamente 800 m no recorte local",
        "variaveis_diretas": "D20; OHC 0-300m; OHC 0-700m; SSH; SSS; temperatura potencial; salinidade",
        "variaveis_calculadas": "nao aplicavel ao master semanal",
        "destino_fase2": "nenhum; mensal nao e promovido nem interpolado para semanal",
    }])
    return pd.concat([coverage, oras], ignore_index=True)


def _daily_feature_frame() -> pd.DataFrame:
    root = ROOT / "data/processed/zarr/features/ocean_daily"
    pieces: list[pd.DataFrame] = []
    for folder, label in (("noaa_ufs", "UFS"), ("glorys12", "GLORYS12"), ("glorys12_operational", "GLO12 operacional")):
        for path in sorted((root / folder).rglob("*.zarr")):
            with xr.open_zarr(path, consolidated=None) as ds:
                available = [name for name in set(ORAS_TO_DAILY_FEATURE.values()) if name in ds]
                if not available:
                    continue
                frame = ds[available].to_dataframe().reset_index()
                frame["fonte_diaria"] = label
                frame["store_end"] = pd.DatetimeIndex(ds.time.values).max()
                pieces.append(frame)
    if not pieces:
        return pd.DataFrame()
    daily = pd.concat(pieces, ignore_index=True)
    daily["time"] = pd.to_datetime(daily["time"]).dt.normalize()
    daily = daily.loc[
        ((daily["fonte_diaria"] == "UFS") & daily["time"].lt("1993-01-01"))
        | ((daily["fonte_diaria"] != "UFS") & daily["time"].ge("1993-01-01"))
    ].copy()
    priority = daily["fonte_diaria"].map({"UFS": 0, "GLORYS12": 1, "GLO12 operacional": 2})
    daily["source_priority"] = priority
    daily = daily.sort_values(["time", "source_priority", "store_end"]).drop_duplicates("time", keep="last")
    return daily.drop(columns=["source_priority", "store_end"])


def _oras_feature_frame() -> pd.DataFrame:
    root = ROOT / "data/processed/zarr/features/ocean_monthly/oras5"
    pieces: list[pd.DataFrame] = []
    for path in sorted(root.rglob("*.zarr")):
        with xr.open_zarr(path, consolidated=None) as ds:
            available = [name for name in ORAS_TO_DAILY_FEATURE if name in ds]
            if available:
                pieces.append(ds[available].to_dataframe().reset_index())
    if not pieces:
        return pd.DataFrame()
    frame = pd.concat(pieces, ignore_index=True).drop_duplicates("time", keep="last")
    frame["time"] = pd.to_datetime(frame["time"]).dt.to_period("M").dt.to_timestamp()
    return frame.sort_values("time")


def _monthly_reanalysis_pairs() -> pd.DataFrame:
    daily = _daily_feature_frame()
    oras = _oras_feature_frame()
    if daily.empty or oras.empty:
        return pd.DataFrame()
    value_columns = sorted(set(ORAS_TO_DAILY_FEATURE.values()).intersection(daily.columns))
    monthly_values = daily.set_index("time")[value_columns].resample("MS").mean()
    monthly_source = daily.set_index("time")["fonte_diaria"].resample("MS").agg(
        lambda values: values.mode().iat[0] if not values.mode().empty else values.iloc[-1]
    )
    monthly = monthly_values.join(monthly_source).reset_index()
    rows: list[pd.DataFrame] = []
    for oras_name, daily_name in ORAS_TO_DAILY_FEATURE.items():
        if oras_name not in oras or daily_name not in monthly:
            continue
        left = monthly[["time", "fonte_diaria", daily_name]].rename(columns={daily_name: "ufs_glorys"})
        right = oras[["time", oras_name]].rename(columns={oras_name: "oras5"})
        paired = left.merge(right, on="time", how="inner", validate="one_to_one")
        paired["variavel"] = daily_name
        paired["residuo_ufs_glorys_menos_oras5"] = paired["ufs_glorys"] - paired["oras5"]
        paired["mes_calendario"] = paired["time"].dt.month
        for column in ("ufs_glorys", "oras5"):
            paired[f"{column}_anomalia_mensal"] = paired[column] - paired.groupby("mes_calendario")[column].transform("mean")
        rows.append(paired)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _reanalysis_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expanded = pd.concat([pairs.assign(segmento="UFS+GLORYS total"), pairs.assign(segmento=pairs["fonte_diaria"])], ignore_index=True)
    for (segment, variable), group in expanded.groupby(["segmento", "variavel"], sort=True):
        clean = group.dropna(subset=["ufs_glorys", "oras5", "ufs_glorys_anomalia_mensal", "oras5_anomalia_mensal"])
        if len(clean) < 3:
            continue
        raw_x = clean["ufs_glorys"].to_numpy(dtype=float)
        raw_y = clean["oras5"].to_numpy(dtype=float)
        anom_x = clean["ufs_glorys_anomalia_mensal"].to_numpy(dtype=float)
        anom_y = clean["oras5_anomalia_mensal"].to_numpy(dtype=float)
        r_raw = float(np.corrcoef(raw_x, raw_y)[0, 1])
        r_anom = float(np.corrcoef(anom_x, anom_y)[0, 1])
        n_eff = effective_sample_size(anom_x, anom_y)
        spearman = spearmanr(anom_x, anom_y, nan_policy="omit")
        residual = raw_x - raw_y
        rows.append({
            "segmento": segment, "variavel": variable,
            "n_meses_pareados": len(clean), "inicio": clean["time"].min(), "fim": clean["time"].max(),
            "r_pearson_nivel": r_raw,
            "r_pearson_anomalia_sazonal_removida": r_anom,
            "n_efetivo_autocorrelacao": n_eff,
            "p_pearson_anomalia_efetivo": correlation_p_value(r_anom, n_eff),
            "rho_spearman_anomalia": float(spearman.statistic),
            "p_spearman_anomalia": float(spearman.pvalue),
            "bias_ufs_glorys_menos_oras5": float(np.mean(residual)),
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "metodo": "UFS+GLORYS diario agregado a MS; ORAS5 mensal nativo; anomalias removem climatologia do mes calendario",
        })
    summary = pd.DataFrame(rows)
    rejected, q_values = benjamini_hochberg_fdr(summary["p_pearson_anomalia_efetivo"].to_numpy(), alpha=0.05)
    summary["q_fdr_bh_pearson_anomalia"] = q_values
    summary["correlacao_anomalia_significativa_fdr_05"] = rejected
    summary["forca_correlacao_descritiva"] = pd.cut(
        summary["r_pearson_anomalia_sazonal_removida"].abs(),
        bins=[-np.inf, 0.3, 0.5, 0.7, np.inf],
        labels=["fraca", "moderada", "forte", "muito_forte"],
    ).astype(str)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-observations", type=int, default=0, help="limite por fonte/variavel apenas para teste rapido")
    args = parser.parse_args()
    stores = _model_stores()
    frames = [*_ctd_frames(), *_tao_frames(), *_argo_frames()]
    collocated: list[pd.DataFrame] = []
    for frame in frames:
        if args.max_observations and len(frame) > args.max_observations:
            frame = frame.sample(args.max_observations, random_state=42).sort_values("time")
        result = _collocate(frame, stores)
        if not result.empty:
            collocated.append(result)
    all_collocations = pd.concat(collocated, ignore_index=True) if collocated else pd.DataFrame()
    profiles = _profile_residuals(all_collocations)
    summary = _summary(profiles) if not profiles.empty else pd.DataFrame()
    contract = _source_contract(stores)
    oras_pairs = _monthly_reanalysis_pairs()
    oras_summary = _reanalysis_summary(oras_pairs) if not oras_pairs.empty else pd.DataFrame()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(PROFILE_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    contract.to_csv(CONTRACT_OUTPUT, index=False)
    oras_pairs.to_csv(ORAS_PAIRS_OUTPUT, index=False)
    oras_summary.to_csv(ORAS_SUMMARY_OUTPUT, index=False)
    print(f"F2V perfis pareados: {len(profiles)} -> {PROFILE_OUTPUT}")
    print(f"F2V testes: {len(summary)} -> {SUMMARY_OUTPUT}")
    print(f"F2 fontes: {len(contract)} -> {CONTRACT_OUTPUT}")
    print(f"F2 UFS+GLORYS x ORAS5: {len(oras_pairs)} pares, {len(oras_summary)} resumos")
    if not summary.empty:
        print(summary[["fonte_insitu", "variavel", "n_perfis_independentes", "bias_modelo_menos_insitu", "q_fdr_bh", "conclusao"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
