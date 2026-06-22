from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import xarray as xr
from nino_brasil.data.download_ocean_monthly import align_monthly_features_causally
from nino_brasil.features.nino import nino34_ssta_index


SpatialStrategy = Literal["mean", "flatten"]


@dataclass(frozen=True)
class FeatureMatrix:
    X: pd.DataFrame
    y: pd.DataFrame
    target_time: pd.Series
    lag_days: int
    feature_groups: dict[str, str]


@dataclass(frozen=True)
class PredictorMatrix:
    X: pd.DataFrame
    target_time: pd.Series
    lag_days: int
    feature_groups: dict[str, str]


def open_zarr_dataset(path: str | Path) -> xr.Dataset:
    """Open a Zarr cube for feature matrix construction."""
    return xr.open_zarr(path)


def _spatial_dims(da: xr.DataArray, time_name: str) -> list[str]:
    return [dim for dim in da.dims if dim != time_name]


def infer_feature_group(name: str) -> str:
    lower = name.lower()
    physics_tokens = (
        "wwv",
        "warm_water_volume",
        "tilt",
        "tendency",
        "_tend",
        "curl",
        "thermocline_tilt",
        "d20_tendency",
        "ohc_tendency",
        "ssta_tendency",
        "wind_stress_curl",
        "equatorial_zonal_stress",
        "ekman",
        "physics_precalc",
    )
    ocean_tokens = ("sst", "ssta", "nino", "ohc", "d20", "mld", "thermocline", "salinity", "temperature")
    atmosphere_tokens = ("slp", "u10", "v10", "u850", "v850", "q850", "u500", "v500", "q500", "z500", "omega", "u200", "v200", "z200", "div", "tcwv", "olr", "wind")
    if any(token in lower for token in physics_tokens):
        return "physics_precalc"
    if any(token in lower for token in ocean_tokens):
        return "ocean"
    if any(token in lower for token in atmosphere_tokens):
        return "atmosphere"
    return "other"


def _point_suffix(point: object) -> str:
    if isinstance(point, tuple):
        return "_".join(str(value).replace(" ", "") for value in point)
    return str(point).replace(" ", "")


def dataarray_to_frame(
    da: xr.DataArray,
    *,
    name: str,
    time_name: str = "time",
    spatial_strategy: SpatialStrategy = "mean",
) -> pd.DataFrame:
    """Convert a time-indexed DataArray to model-ready columns."""
    dims = _spatial_dims(da, time_name)
    if spatial_strategy == "mean" and dims:
        series = da.mean(dim=dims, skipna=True).to_pandas()
        return series.to_frame(name)
    if spatial_strategy == "mean":
        return da.to_pandas().to_frame(name)
    if spatial_strategy != "flatten":
        raise ValueError("spatial_strategy must be 'mean' or 'flatten'.")
    if not dims:
        return da.to_pandas().to_frame(name)

    stacked = da.stack(feature_point=dims).transpose(time_name, "feature_point")
    frame = stacked.to_pandas()
    frame.columns = [f"{name}__{_point_suffix(point)}" for point in frame.columns]
    return frame


def dataset_to_feature_frame(
    ds: xr.Dataset,
    *,
    time_name: str = "time",
    spatial_strategy: SpatialStrategy = "mean",
    feature_groups: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Convert all data variables in a Dataset to a feature matrix."""
    frames: list[pd.DataFrame] = []
    groups: dict[str, str] = {}
    for variable in ds.data_vars:
        frame = dataarray_to_frame(
            ds[variable],
            name=variable,
            time_name=time_name,
            spatial_strategy=spatial_strategy,
        )
        frames.append(frame)
        base_group = (feature_groups or {}).get(variable, infer_feature_group(variable))
        groups.update({column: base_group for column in frame.columns})
    if not frames:
        raise ValueError("predictors dataset has no data variables.")
    return pd.concat(frames, axis=1).sort_index(), groups


def target_to_frame(
    target: xr.DataArray,
    *,
    target_name: str = "target",
    time_name: str = "time",
    spatial_strategy: SpatialStrategy = "mean",
) -> pd.DataFrame:
    """Convert a target DataArray to one or more target columns."""
    return dataarray_to_frame(
        target,
        name=target.name or target_name,
        time_name=time_name,
        spatial_strategy=spatial_strategy,
    )


def add_nino34_feature(
    predictors: xr.Dataset,
    *,
    sst_variable: str = "sst",
    time_name: str = "time",
    window_days: int = 15,
) -> xr.Dataset:
    """Add Niño 3.4 daily SSTA as a physical baseline feature when SST is available.

    The climatology here uses the FULL supplied sample, so this helper is only
    suitable for exploratory use. Walk-forward evaluation must pass a fold-safe
    `nino34_ssta` precomputed with a train-only climatology and call
    `build_feature_matrix(..., include_nino34=False)`.
    """
    if sst_variable not in predictors:
        return predictors
    try:
        nino34 = nino34_ssta_index(predictors[sst_variable], window_days=window_days, time_name=time_name)
    except (KeyError, ValueError, IndexError):
        return predictors
    return predictors.assign(nino34_ssta=nino34)


def add_causal_monthly_ocean_features(
    daily_predictors: xr.Dataset,
    monthly_ocean: xr.Dataset,
    *,
    time_name: str = "time",
    publication_lag_days: int = 15,
    prefix: str = "oras5_monthly__",
) -> tuple[xr.Dataset, dict[str, str]]:
    """Attach last-published monthly ORAS5 covariates to daily model rows.

    This is a model-matrix alignment operation, not a conversion of the source
    into daily observations.  Variable names and attributes preserve that
    distinction and all attached columns receive the `ocean_monthly` group.
    """
    if time_name not in daily_predictors.coords:
        raise KeyError(f"daily_predictors must contain {time_name!r}.")
    daily_time = pd.DatetimeIndex(daily_predictors[time_name].values)
    aligned = align_monthly_features_causally(
        monthly_ocean,
        daily_time,
        publication_lag_days=publication_lag_days,
    )
    renamed = aligned.rename({name: f"{prefix}{name}" for name in aligned.data_vars})
    merged = xr.merge([daily_predictors, renamed], join="exact", compat="override")
    groups = {name: "ocean_monthly" for name in renamed.data_vars}
    merged.attrs.update(daily_predictors.attrs)
    merged.attrs["monthly_ocean_alignment"] = "last_published_month_causal"
    return merged, groups


def build_feature_matrix(
    predictors: xr.Dataset,
    target: xr.DataArray,
    *,
    lag_days: int,
    time_name: str = "time",
    predictor_strategy: SpatialStrategy = "mean",
    target_strategy: SpatialStrategy = "mean",
    include_time_features: bool = True,
    include_nino34: bool = True,
    nino34_sst_variable: str = "sst",
    climatology_window_days: int = 15,
    feature_groups: dict[str, str] | None = None,
) -> FeatureMatrix:
    """Build X(t) and Y(t+lag) from regridded Zarr-compatible cubes."""
    predictor_matrix = build_predictor_matrix(
        predictors,
        lag_days=lag_days,
        time_name=time_name,
        predictor_strategy=predictor_strategy,
        include_time_features=include_time_features,
        include_nino34=include_nino34,
        nino34_sst_variable=nino34_sst_variable,
        climatology_window_days=climatology_window_days,
        feature_groups=feature_groups,
    )
    return align_target_to_predictor_matrix(
        predictor_matrix,
        target,
        target_name=target.name or "target",
        time_name=time_name,
        target_strategy=target_strategy,
    )


def build_predictor_matrix(
    predictors: xr.Dataset,
    *,
    lag_days: int,
    time_name: str = "time",
    predictor_strategy: SpatialStrategy = "mean",
    include_time_features: bool = True,
    include_nino34: bool = True,
    nino34_sst_variable: str = "sst",
    climatology_window_days: int = 15,
    feature_groups: dict[str, str] | None = None,
) -> PredictorMatrix:
    """Build reusable X(t) and target-time coordinates for one forecast lag."""
    if lag_days < 0:
        raise ValueError("lag_days must be non-negative.")

    model_predictors = predictors
    groups_override = dict(feature_groups or {})
    if include_nino34:
        model_predictors = add_nino34_feature(
            model_predictors,
            sst_variable=nino34_sst_variable,
            time_name=time_name,
            window_days=climatology_window_days,
        )
        if "nino34_ssta" in model_predictors:
            groups_override.setdefault("nino34_ssta", "ocean")

    X, groups = dataset_to_feature_frame(
        model_predictors,
        time_name=time_name,
        spatial_strategy=predictor_strategy,
        feature_groups=groups_override,
    )
    target_time = pd.Series(
        pd.DatetimeIndex(X.index) + pd.to_timedelta(lag_days, unit="D"),
        index=X.index,
        name="target_time",
    )

    if include_time_features:
        X = X.copy()
        X["month"] = target_time.dt.month.astype(float)
        X["dayofyear"] = target_time.dt.dayofyear.astype(float)
        groups["month"] = "calendar"
        groups["dayofyear"] = "calendar"

    return PredictorMatrix(X=X, target_time=target_time, lag_days=lag_days, feature_groups=groups)


def align_target_to_predictor_matrix(
    predictor_matrix: PredictorMatrix,
    target: xr.DataArray,
    *,
    target_name: str = "target",
    time_name: str = "time",
    target_strategy: SpatialStrategy = "mean",
) -> FeatureMatrix:
    """Align a target Y(t+lag) to a reusable predictor matrix X(t)."""
    if time_name not in target.coords and time_name not in target.dims:
        raise KeyError(f"target must contain {time_name!r}.")

    target_index = pd.DatetimeIndex(target[time_name].values)
    wanted_target_time = pd.DatetimeIndex(predictor_matrix.target_time.values)
    valid_time_mask = wanted_target_time.isin(target_index)
    if not valid_time_mask.any():
        return FeatureMatrix(
            X=predictor_matrix.X.iloc[0:0].copy(),
            y=pd.DataFrame(index=predictor_matrix.X.index[0:0]),
            target_time=predictor_matrix.target_time.iloc[0:0],
            lag_days=predictor_matrix.lag_days,
            feature_groups=predictor_matrix.feature_groups,
        )

    origin_index = predictor_matrix.X.index[valid_time_mask]
    selected_target_time = wanted_target_time[valid_time_mask]
    aligned_y = target.sel({time_name: selected_target_time})
    y = target_to_frame(
        aligned_y,
        target_name=target.name or target_name,
        time_name=time_name,
        spatial_strategy=target_strategy,
    )
    y.index = origin_index

    X = predictor_matrix.X.loc[origin_index]
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    target_time = predictor_matrix.target_time.loc[common_index]

    valid = y.notna().any(axis=1)
    X = X.loc[valid]
    y = y.loc[valid]
    target_time = target_time.loc[valid]

    return FeatureMatrix(
        X=X,
        y=y,
        target_time=target_time,
        lag_days=predictor_matrix.lag_days,
        feature_groups=predictor_matrix.feature_groups,
    )
