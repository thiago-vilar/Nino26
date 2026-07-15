"""Phase 7: event-aware PyTorch ConvLSTM for the ENSO life cycle.

The official target is nine states (neutral plus El Nino/La Nina x genesis,
growth, peak and decay) together with event intensity, time-to-peak and
duration.  All sequence windows from one event stay in the same purged fold.
Synthetic samples are optimisation aids only and retain event provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from nino_brasil.models.event_validation import (
    ENSO_STATES,
    assert_continuous_weekly_index,
    balanced_active_event_test_start,
    canonical_event_ids,
    canonical_state_labels,
    make_event_folds,
    parse_state,
)

try:  # Import remains optional for data preparation and documentation builds.
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - dependency is present in the project env
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[assignment,misc]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def _require_torch():
    if torch is None:
        raise ImportError(
            "PyTorch nao instalado. Instale as dependencias do projeto antes de executar F7/F8."
        )
    return torch


@dataclass(frozen=True)
class PacificCube:
    values: np.ndarray
    times: pd.DatetimeIndex
    lat: np.ndarray
    lon: np.ndarray
    channel_names: tuple[str, ...]
    finite_mask: np.ndarray
    source_path: str


@dataclass(frozen=True)
class SpatialHarmonicNormalizer:
    """Per-grid harmonic climatology and residual scaling fitted on train only."""

    coefficients: np.ndarray
    residual_mean: np.ndarray
    residual_scale: np.ndarray
    fill_value: np.ndarray
    origin: pd.Timestamp
    harmonics: int
    detrend: bool

    def _design(self, times: pd.DatetimeIndex) -> np.ndarray:
        angle = 2.0 * np.pi * (times.dayofyear.to_numpy(dtype=float) - 1.0) / 365.2425
        columns = [np.ones(len(times), dtype=float)]
        for harmonic in range(1, self.harmonics + 1):
            columns.extend([np.sin(harmonic * angle), np.cos(harmonic * angle)])
        if self.detrend:
            columns.append((times - self.origin).days.to_numpy(dtype=float) / 365.2425)
        return np.column_stack(columns)

    def transform(self, values: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray:
        if values.ndim != 4 or len(values) != len(times):
            raise ValueError("values deve ser time,lat,lon,channel e casar com times.")
        design = self._design(times)
        fitted = np.tensordot(design, self.coefficients, axes=(1, 0))
        filled = np.where(np.isfinite(values), values, self.fill_value[None])
        residual = filled.astype("float64") - fitted
        return ((residual - self.residual_mean) / self.residual_scale).astype("float32")


def fit_spatial_harmonic_normalizer(
    values: np.ndarray,
    times: pd.DatetimeIndex,
    train_mask: np.ndarray,
    *,
    harmonics: int = 3,
    detrend: bool = True,
) -> SpatialHarmonicNormalizer:
    """Vectorised per-pixel seasonal/detrend fit using training weeks only."""

    if values.ndim != 4 or len(values) != len(times) or len(train_mask) != len(times):
        raise ValueError("values/times/train_mask incompatíveis.")
    train_mask = np.asarray(train_mask, dtype=bool)
    origin = pd.Timestamp(times[train_mask].min())
    angle = 2.0 * np.pi * (times.dayofyear.to_numpy(dtype=float) - 1.0) / 365.2425
    columns = [np.ones(len(times), dtype=float)]
    for harmonic in range(1, harmonics + 1):
        columns.extend([np.sin(harmonic * angle), np.cos(harmonic * angle)])
    if detrend:
        columns.append((times - origin).days.to_numpy(dtype=float) / 365.2425)
    design = np.column_stack(columns)
    train_values = np.asarray(values[train_mask], dtype="float64")
    # Fill rare ocean-mask gaps with the training temporal mean per grid cell.
    valid_count = np.isfinite(train_values).sum(axis=0)
    mean = np.nansum(train_values, axis=0) / np.maximum(valid_count, 1)
    filled = np.where(np.isfinite(values), values, mean[None])
    coefficients = np.tensordot(
        np.linalg.pinv(design[train_mask]), filled[train_mask], axes=(1, 0)
    )
    fitted_train = np.tensordot(design[train_mask], coefficients, axes=(1, 0))
    residual_train = filled[train_mask] - fitted_train
    residual_mean = np.mean(residual_train, axis=0)
    residual_scale = np.std(residual_train, axis=0)
    residual_scale = np.where(residual_scale > 1e-6, residual_scale, 1.0)
    return SpatialHarmonicNormalizer(
        coefficients=coefficients,
        residual_mean=residual_mean,
        residual_scale=residual_scale,
        fill_value=mean,
        origin=origin,
        harmonics=harmonics,
        detrend=detrend,
    )


def _coordinate_slice(values: np.ndarray, bounds: tuple[float, float]) -> slice:
    start, end = bounds
    return slice(start, end) if values[0] <= values[-1] else slice(end, start)


def load_pacific_dataset(
    zarr_dir: str | Path,
    variables: Sequence[str],
    *,
    lat_bounds: tuple[float, float] = (-15.0, 15.0),
    lon_bounds: tuple[float, float] = (120.0, 290.0),
    time_name: str = "time",
) -> PacificCube:
    """Load real Pacific fields with timestamps and auditable channel names."""

    import xarray as xr

    dataset = xr.open_zarr(zarr_dir, consolidated=False)
    lat_name = "lat" if "lat" in dataset.coords else "latitude"
    lon_name = "lon" if "lon" in dataset.coords else "longitude"
    lat_values = np.asarray(dataset[lat_name].values)
    lon_values = np.asarray(dataset[lon_name].values)
    subset = dataset.sel(
        {
            lat_name: _coordinate_slice(lat_values, lat_bounds),
            lon_name: _coordinate_slice(lon_values, lon_bounds),
        }
    )
    present = [name for name in variables if name in subset.data_vars]
    missing = sorted(set(variables).difference(present))
    if missing:
        raise KeyError(f"Canais ausentes do cubo Pacifico: {missing}")
    arrays = [subset[name].transpose(time_name, lat_name, lon_name) for name in present]
    aligned = xr.align(*arrays, join="inner")
    values = np.stack([np.asarray(array.values, dtype="float32") for array in aligned], axis=-1)
    finite = np.isfinite(values)
    return PacificCube(
        values=values,
        times=pd.DatetimeIndex(aligned[0][time_name].values),
        lat=np.asarray(aligned[0][lat_name].values),
        lon=np.asarray(aligned[0][lon_name].values),
        channel_names=tuple(present),
        finite_mask=finite,
        source_path=str(Path(zarr_dir)),
    )


def load_pacific_cube(
    zarr_dir: str,
    variables: Sequence[str],
    *,
    lat_bounds: tuple[float, float] = (-15.0, 15.0),
    lon_bounds: tuple[float, float] = (120.0, 290.0),
    time_name: str = "time",
) -> np.ndarray:
    """Compatibility wrapper returning ``(time, lat, lon, channel)`` values."""

    return load_pacific_dataset(
        zarr_dir,
        variables,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        time_name=time_name,
    ).values


def sequence_end_indices(n_time: int, *, seq_len: int, horizon: int = 0) -> np.ndarray:
    if seq_len < 1 or horizon < 0:
        raise ValueError("seq_len deve ser >=1 e horizon >=0.")
    count = n_time - seq_len - horizon + 1
    if count <= 0:
        return np.empty(0, dtype=int)
    return np.arange(seq_len - 1, seq_len - 1 + count, dtype=int)


def make_sequences(
    cube: np.ndarray,
    labels: np.ndarray | None,
    *,
    seq_len: int = 24,
    horizon: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build causal channel-last sequences and targets at end+horizon."""

    if cube.ndim != 4:
        raise ValueError("cube deve ter forma (time, lat, lon, channel).")
    ends = sequence_end_indices(len(cube), seq_len=seq_len, horizon=horizon)
    if len(ends):
        sequences = np.stack([cube[end - seq_len + 1 : end + 1] for end in ends]).astype(
            "float32", copy=False
        )
    else:
        sequences = np.empty((0, seq_len, *cube.shape[1:]), dtype="float32")
    if labels is None:
        return sequences, None
    labels = np.asarray(labels)
    if len(labels) != len(cube):
        raise ValueError("labels precisa ter o mesmo eixo temporal do cubo.")
    return sequences, labels[ends + horizon]


@dataclass(frozen=True)
class ENSOSequenceDataset:
    sequences: np.ndarray
    targets: np.ndarray | None
    origin_times: pd.DatetimeIndex
    target_times: pd.DatetimeIndex
    end_indices: np.ndarray
    channel_names: tuple[str, ...]
    event_ids: np.ndarray | None


def build_sequence_event_targets(
    dataset: ENSOSequenceDataset,
    target_phase_table: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Attach completed-event magnitude/timing/duration as output-only labels.

    These retrospective quantities never enter an input sequence.  They are
    used only as supervised targets for active events and are evaluated with
    event-equal aggregation so additional weeks do not create independent
    samples.  The three official regression targets are constants defined once
    per event.  ``weeks_origin_to_peak`` is retained only as a sequence-level
    diagnostic and must not be used as an event-level regression target.
    """

    required_event = {"event_id", "onset", "pico", "fim", "oni_pico_c"}
    if missing := required_event.difference(events.columns):
        raise KeyError(f"Tabela de eventos sem colunas {sorted(missing)}")
    phase = target_phase_table.reindex(dataset.target_times)
    if "event_id" not in phase:
        raise KeyError("target_phase_table precisa de event_id.")
    if events["event_id"].astype(str).duplicated().any():
        raise ValueError("event_id duplicado na tabela de eventos F7.")
    event_meta = events.copy().set_index("event_id")
    event_meta.index = event_meta.index.astype(str)
    for column in ("onset", "pico", "fim"):
        event_meta[column] = pd.to_datetime(event_meta[column])
    invalid_order = ~(
        event_meta["onset"].le(event_meta["pico"])
        & event_meta["pico"].le(event_meta["fim"])
    )
    invalid_peak = ~np.isfinite(pd.to_numeric(event_meta["oni_pico_c"], errors="coerce"))
    if (invalid_order | invalid_peak).any():
        examples = event_meta.index[invalid_order | invalid_peak].astype(str).tolist()[:5]
        raise ValueError(f"Eventos F7 com datas/ONI invalidos: {examples}")
    rows: list[dict[str, object]] = []
    for position, target_time in enumerate(dataset.target_times):
        raw_event_id = phase.iloc[position]["event_id"]
        event_id = "" if pd.isna(raw_event_id) else str(raw_event_id)
        if event_id in {"", "nan", "None"} or event_id not in event_meta.index:
            rows.append(
                {
                    "target_time": target_time,
                    "origin_time": dataset.origin_times[position],
                    "event_id": "",
                    "peak_magnitude_c": np.nan,
                    "event_time_to_peak_weeks": np.nan,
                    "weeks_origin_to_peak": np.nan,
                    "event_duration_weeks": np.nan,
                }
            )
            continue
        event = event_meta.loc[event_id]
        rows.append(
            {
                "target_time": target_time,
                "origin_time": dataset.origin_times[position],
                "event_id": event_id,
                "peak_magnitude_c": abs(float(event["oni_pico_c"])),
                "event_time_to_peak_weeks": (
                    pd.Timestamp(event["pico"]) - pd.Timestamp(event["onset"])
                ).days
                / 7.0,
                "weeks_origin_to_peak": (
                    pd.Timestamp(event["pico"]) - dataset.origin_times[position]
                ).days
                / 7.0,
                "event_duration_weeks": (
                    pd.Timestamp(event["fim"]) - pd.Timestamp(event["onset"])
                ).days
                / 7.0,
            }
        )
    return pd.DataFrame(rows).set_index("target_time")


def event_level_target_table(
    event_targets: pd.DataFrame,
    target_columns: Sequence[str],
) -> pd.DataFrame:
    """Return one auditable row per independent event.

    Event-level targets must be constant within ``event_id``.  Failing loudly
    prevents a time-varying sequence diagnostic from being collapsed with an
    arbitrary ``drop_duplicates`` first row and then used to normalise every
    sample in the fold.
    """

    required = {"event_id", *map(str, target_columns)}
    if missing := required.difference(event_targets.columns):
        raise KeyError(f"event_targets sem colunas {sorted(missing)}")
    active = event_targets.loc[
        event_targets["event_id"].fillna("").astype(str).ne(""),
        ["event_id", *target_columns],
    ].copy()
    if active.empty:
        return pd.DataFrame(columns=list(target_columns), dtype=float).rename_axis("event_id")
    active["event_id"] = active["event_id"].astype(str)
    for column in target_columns:
        active[column] = pd.to_numeric(active[column], errors="coerce")
    if active[list(target_columns)].isna().any().any():
        raise ValueError("Targets definidos por evento contem valores ausentes/nao numericos.")
    unique_counts = active.groupby("event_id", sort=False)[list(target_columns)].nunique(
        dropna=False
    )
    inconsistent = unique_counts.gt(1).any(axis=1)
    if inconsistent.any():
        examples = unique_counts.index[inconsistent].astype(str).tolist()[:5]
        raise ValueError(
            "Targets definidos por evento variam dentro do mesmo event_id: "
            f"{examples}. Use apenas grandezas constantes do evento."
        )
    return active.groupby("event_id", sort=False)[list(target_columns)].first()


EVENT_INTERVAL_NOMINAL_COVERAGE = 0.90
EVENT_INTERVAL_Z = NormalDist().inv_cdf(
    0.5 + EVENT_INTERVAL_NOMINAL_COVERAGE / 2.0
)


def gaussian_event_equal_audit(
    observed: Sequence[float] | np.ndarray,
    predicted_mean: Sequence[float] | np.ndarray,
    predicted_scale: Sequence[float] | np.ndarray,
    event_ids: Sequence[object] | np.ndarray,
    *,
    nominal_coverage: float = EVENT_INTERVAL_NOMINAL_COVERAGE,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Audit Gaussian event predictions without treating weeks as events.

    Row-level NLL, interval coverage and width are first averaged inside each
    event.  The returned summary then gives every independent event equal
    weight, regardless of its duration or the number of eligible sequences.
    """

    if not 0.0 < float(nominal_coverage) < 1.0:
        raise ValueError("nominal_coverage deve estar em (0,1).")
    y = np.asarray(observed, dtype=float)
    mean = np.asarray(predicted_mean, dtype=float)
    scale = np.asarray(predicted_scale, dtype=float)
    ids = np.asarray(event_ids, dtype=object)
    if not (y.ndim == mean.ndim == scale.ndim == ids.ndim == 1):
        raise ValueError("observed/mean/scale/event_ids devem ser vetores.")
    if not (len(y) == len(mean) == len(scale) == len(ids)):
        raise ValueError("observed/mean/scale/event_ids precisam do mesmo tamanho.")
    ids_as_text = pd.Series(ids, dtype="object").fillna("").astype(str).to_numpy()
    valid = (
        np.isfinite(y)
        & np.isfinite(mean)
        & np.isfinite(scale)
        & (scale > 0.0)
        & (ids_as_text != "")
    )
    columns = [
        "event_id",
        "n_sequence_rows",
        "gaussian_nll_mean",
        "interval_coverage",
        "mean_interval_width",
        "mean_absolute_error",
    ]
    if not valid.any():
        empty = pd.DataFrame(columns=columns)
        return empty, {
            "n_events": 0,
            "n_sequence_rows": 0,
            "gaussian_nll_event_equal": np.nan,
            "interval_coverage_event_equal": np.nan,
            "interval_nominal_coverage": float(nominal_coverage),
            "interval_calibration_error": np.nan,
            "interval_absolute_calibration_error": np.nan,
            "mean_interval_width_event_equal": np.nan,
        }
    z_value = NormalDist().inv_cdf(0.5 + float(nominal_coverage) / 2.0)
    selected_y = y[valid]
    selected_mean = mean[valid]
    selected_scale = scale[valid]
    lower = selected_mean - z_value * selected_scale
    upper = selected_mean + z_value * selected_scale
    standardised = (selected_y - selected_mean) / selected_scale
    point = pd.DataFrame(
        {
            "event_id": ids_as_text[valid],
            "gaussian_nll": (
                0.5 * standardised**2
                + np.log(selected_scale)
                + 0.5 * np.log(2.0 * np.pi)
            ),
            "covered": (selected_y >= lower) & (selected_y <= upper),
            "interval_width": upper - lower,
            "absolute_error": np.abs(selected_y - selected_mean),
        }
    )
    by_event = (
        point.groupby("event_id", sort=False)
        .agg(
            n_sequence_rows=("gaussian_nll", "size"),
            gaussian_nll_mean=("gaussian_nll", "mean"),
            interval_coverage=("covered", "mean"),
            mean_interval_width=("interval_width", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
        )
        .reset_index()
    )
    coverage = float(by_event["interval_coverage"].mean())
    calibration_error = coverage - float(nominal_coverage)
    summary: dict[str, float | int] = {
        "n_events": int(len(by_event)),
        "n_sequence_rows": int(valid.sum()),
        "gaussian_nll_event_equal": float(by_event["gaussian_nll_mean"].mean()),
        "interval_coverage_event_equal": coverage,
        "interval_nominal_coverage": float(nominal_coverage),
        "interval_calibration_error": calibration_error,
        "interval_absolute_calibration_error": abs(calibration_error),
        "mean_interval_width_event_equal": float(
            by_event["mean_interval_width"].mean()
        ),
    }
    return by_event, summary


def make_sequence_dataset(
    cube: PacificCube,
    labels: np.ndarray | None,
    *,
    seq_len: int = 24,
    horizon: int = 0,
    event_ids: Sequence[str] | None = None,
) -> ENSOSequenceDataset:
    assert_continuous_weekly_index(cube.times, name="F7 Pacific cube weekly index")
    sequences, targets = make_sequences(cube.values, labels, seq_len=seq_len, horizon=horizon)
    ends = sequence_end_indices(len(cube.times), seq_len=seq_len, horizon=horizon)
    target_positions = ends + horizon
    if event_ids is not None and len(event_ids) != len(cube.times):
        raise ValueError("event_ids precisa ter o mesmo eixo temporal do cubo.")
    return ENSOSequenceDataset(
        sequences=sequences,
        targets=targets,
        origin_times=cube.times[ends],
        target_times=cube.times[target_positions],
        end_indices=ends,
        channel_names=cube.channel_names,
        event_ids=(np.asarray(event_ids, dtype=object)[target_positions] if event_ids is not None else None),
    )


def sequence_event_folds(
    dataset: ENSOSequenceDataset,
    target_phase_table: pd.DataFrame,
    *,
    n_splits: int = 5,
    min_train_groups: int = 8,
    min_train_active_events_per_type_for_start: int = 0,
    seq_len: int = 24,
    horizon: int = 0,
):
    """Create whole-event folds with an embargo covering overlap and horizon."""

    target_phase = target_phase_table.reindex(dataset.target_times)
    groups = canonical_event_ids(target_phase)
    purge_weeks = seq_len + horizon
    earliest_test_start = balanced_active_event_test_start(
        target_phase,
        min_train_active_events_per_type=min_train_active_events_per_type_for_start,
        purge_weeks=purge_weeks,
    )
    return make_event_folds(
        dataset.target_times,
        groups,
        n_splits=n_splits,
        min_train_groups=min_train_groups,
        purge_weeks=purge_weeks,
        earliest_test_start=earliest_test_start,
    )


def sequence_source_phase_table(
    dataset: ENSOSequenceDataset,
    phase_table: pd.DataFrame,
) -> pd.DataFrame:
    """Map ENSO labels at sequence origin onto target-time evaluation rows."""

    source = phase_table.reindex(dataset.origin_times).copy()
    if source[["tipo", "fase"]].isna().any().any():
        raise ValueError("Fase ENSO ausente em uma ou mais origens de sequencia.")
    source.index = dataset.target_times
    source.index.name = "target_time"
    return source


def chronological_split(
    n_samples: int,
    *,
    val_fraction: float = 0.2,
    embargo: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy time split with an explicit sample embargo (event folds are preferred)."""

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction deve estar entre 0 e 1.")
    cut = int(round(n_samples * (1.0 - val_fraction)))
    train_end = max(0, cut - int(embargo))
    return np.arange(train_end), np.arange(cut, n_samples)


if nn is not None:

    class ConvLSTMCell(nn.Module):
        """Convolutional LSTM cell preserving the spatial Pacific layout."""

        def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
            super().__init__()
            padding = kernel_size // 2
            self.hidden_channels = int(hidden_channels)
            self.gates = nn.Conv2d(
                input_channels + hidden_channels,
                4 * hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            )

        def forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
            hidden, cell = state
            input_gate, forget_gate, output_gate, candidate = self.gates(
                torch.cat([x, hidden], dim=1)
            ).chunk(4, dim=1)
            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            output_gate = torch.sigmoid(output_gate)
            candidate = torch.tanh(candidate)
            cell = forget_gate * cell + input_gate * candidate
            hidden = output_gate * torch.tanh(cell)
            return hidden, cell


    class ConvLSTMEncoder(nn.Module):
        def __init__(
            self,
            input_channels: int,
            hidden_channels: Sequence[int] = (32, 32),
            kernel_size: int = 3,
        ):
            super().__init__()
            channels = [input_channels, *hidden_channels]
            self.input_channels = int(input_channels)
            self.cells = nn.ModuleList(
                ConvLSTMCell(channels[layer], channels[layer + 1], kernel_size)
                for layer in range(len(hidden_channels))
            )
            self.output_channels = int(hidden_channels[-1])

        def forward(self, sequence: Tensor, *, channel_last: bool = True) -> Tensor:
            if sequence.ndim != 5:
                raise ValueError("sequence deve ter forma N,T,H,W,C ou N,T,C,H,W.")
            if channel_last:
                sequence = sequence.permute(0, 1, 4, 2, 3).contiguous()
            if sequence.shape[2] != self.input_channels:
                raise ValueError(
                    f"Esperava {self.input_channels} canais; recebi {sequence.shape[2]}."
                )
            layer_input = sequence
            for cell in self.cells:
                batch, steps, _, height, width = layer_input.shape
                hidden = layer_input.new_zeros((batch, cell.hidden_channels, height, width))
                memory = layer_input.new_zeros((batch, cell.hidden_channels, height, width))
                outputs: list[Tensor] = []
                for step in range(steps):
                    hidden, memory = cell(layer_input[:, step], (hidden, memory))
                    outputs.append(hidden)
                layer_input = torch.stack(outputs, dim=1)
            return layer_input[:, -1]


    class ENSOMultiTaskConvLSTM(nn.Module):
        """Nine-state classifier plus probabilistic event regressions."""

        def __init__(
            self,
            input_channels: int,
            hidden_channels: Sequence[int] = (32, 32),
            n_states: int = len(ENSO_STATES),
            dropout: float = 0.25,
            scalar_channels: int = 0,
        ):
            super().__init__()
            self.encoder = ConvLSTMEncoder(input_channels, hidden_channels)
            self.scalar_channels = int(scalar_channels)
            scalar_latent = min(64, max(16, self.scalar_channels * 2)) if self.scalar_channels else 0
            self.scalar_encoder = (
                nn.GRU(self.scalar_channels, scalar_latent, batch_first=True)
                if self.scalar_channels
                else None
            )
            latent = self.encoder.output_channels + scalar_latent
            self.dropout = nn.Dropout(dropout)
            self.state_head = nn.Linear(latent, n_states)
            self.type_head = nn.Linear(latent, 3)  # neutral, El Nino, La Nina
            self.phase_head = nn.Linear(latent, 5)  # neutral + four phases
            self.regression_head = nn.Linear(latent, 6)  # 3 means + 3 log-scales

        def forward(
            self,
            sequence: Tensor,
            *,
            scalar_sequence: Tensor | None = None,
            channel_last: bool = True,
        ) -> dict[str, Tensor]:
            spatial = self.encoder(sequence, channel_last=channel_last)
            pooled = F.adaptive_avg_pool2d(spatial, 1).flatten(1)
            if self.scalar_encoder is not None:
                if scalar_sequence is None:
                    raise ValueError("O modelo foi criado com scalar_channels; forneca scalar_sequence.")
                if scalar_sequence.ndim != 3 or scalar_sequence.shape[-1] != self.scalar_channels:
                    raise ValueError("scalar_sequence deve ter forma N,T,scalar_channels.")
                _, scalar_hidden = self.scalar_encoder(scalar_sequence)
                pooled = torch.cat([pooled, scalar_hidden[-1]], dim=1)
            pooled = self.dropout(pooled)
            regression = self.regression_head(pooled)
            return {
                "state_logits": self.state_head(pooled),
                "type_logits": self.type_head(pooled),
                "phase_logits": self.phase_head(pooled),
                "event_mean": regression[:, :3],
                "event_log_scale": regression[:, 3:].clamp(-6.0, 4.0),
                "spatial_latent": spatial,
            }

else:  # pragma: no cover

    class ConvLSTMCell:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class ConvLSTMEncoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class ENSOMultiTaskConvLSTM:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()


def build_convlstm_classifier(
    input_shape: tuple[int, int, int, int],
    n_classes: int = len(ENSO_STATES),
    *,
    filters: Sequence[int] = (32, 32),
    scalar_channels: int = 0,
):
    """Build the official PyTorch ConvLSTM (input is seq,lat,lon,channel)."""

    _require_torch()
    if len(input_shape) != 4:
        raise ValueError("input_shape deve ser (seq_len, lat, lon, channel).")
    return ENSOMultiTaskConvLSTM(
        input_channels=int(input_shape[-1]),
        hidden_channels=filters,
        n_states=n_classes,
        scalar_channels=scalar_channels,
    )


def multitask_loss(
    output: Mapping[str, Tensor],
    *,
    state_target: Tensor,
    type_target: Tensor | None = None,
    phase_target: Tensor | None = None,
    event_target: Tensor | None = None,
    event_mask: Tensor | None = None,
    sample_weight: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Weighted classification + Gaussian NLL for peak/time/duration."""

    _require_torch()
    losses = F.cross_entropy(output["state_logits"], state_target.long(), reduction="none")
    components: dict[str, Tensor] = {"state": losses}
    if type_target is not None:
        components["type"] = F.cross_entropy(
            output["type_logits"], type_target.long(), reduction="none"
        )
    if phase_target is not None:
        components["phase"] = F.cross_entropy(
            output["phase_logits"], phase_target.long(), reduction="none"
        )
    total_per_sample = sum(components.values())
    if event_target is not None:
        mean = output["event_mean"]
        log_scale = output["event_log_scale"]
        gaussian = 0.5 * (((event_target - mean) / log_scale.exp()) ** 2 + 2.0 * log_scale)
        if event_mask is not None:
            gaussian = gaussian * event_mask
            denom = event_mask.sum(dim=1).clamp_min(1.0)
            gaussian = gaussian.sum(dim=1) / denom
        else:
            gaussian = gaussian.mean(dim=1)
        components["event_nll"] = gaussian
        total_per_sample = total_per_sample + gaussian
    if sample_weight is not None:
        total = (total_per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)
    else:
        total = total_per_sample.mean()
    report = {name: float(value.detach().mean().cpu()) for name, value in components.items()}
    report["total"] = float(total.detach().cpu())
    return total, report


@dataclass(frozen=True)
class SequenceAugmentation:
    values: np.ndarray
    provenance: pd.DataFrame


def augment_sequence_batch(
    sequences: np.ndarray,
    event_ids: Sequence[str],
    *,
    sample_times: Sequence[object],
    origin_times: Sequence[object],
    states: Sequence[str],
    noise_scale: float = 0.01,
    channel_mask_probability: float = 0.05,
    time_mask_probability: float = 0.05,
    random_state: int = 42,
) -> SequenceAugmentation:
    """Train-only masking/noise; every row keeps its original event id."""

    if sequences.ndim != 5:
        raise ValueError("sequences deve ter forma N,T,H,W,C.")
    lengths = {
        "event_ids": len(event_ids),
        "sample_times": len(sample_times),
        "origin_times": len(origin_times),
        "states": len(states),
    }
    if any(length != len(sequences) for length in lengths.values()):
        raise ValueError(f"Metadados de provenance incompatíveis: {lengths}.")
    explicit_ids = np.asarray(event_ids, dtype=object).astype(str)
    if any(not value.strip() for value in explicit_ids):
        raise ValueError(
            "event_ids deve usar event_id ativo ou grupo neutral_YYYYQn explícito."
        )
    state_values = np.asarray(states, dtype=object).astype(str)
    parsed_states = [parse_state(value) for value in state_values]
    rng = np.random.default_rng(random_state)
    augmented = np.asarray(sequences, dtype="float32").copy()
    channel_scale = np.nanstd(augmented, axis=(0, 1, 2, 3), keepdims=True)
    noise = rng.normal(size=augmented.shape).astype("float32")
    augmented += float(noise_scale) * channel_scale * noise
    channel_mask = rng.random((len(augmented), 1, 1, 1, augmented.shape[-1])) < float(
        channel_mask_probability
    )
    time_mask = rng.random((len(augmented), augmented.shape[1], 1, 1, 1)) < float(
        time_mask_probability
    )
    augmented = np.where(channel_mask | time_mask, 0.0, augmented)
    provenance = pd.DataFrame(
        {
            "sample_time": pd.Index(sample_times).astype(str),
            "origin_time": pd.Index(origin_times).astype(str),
            "original_event_id": explicit_ids,
            "state": state_values,
            "event_type": [value[0] for value in parsed_states],
            "phase": [value[1] for value in parsed_states],
            "augmentation_id": [f"sequence_aug_{i:06d}" for i in range(len(augmented))],
            "augmentation_method": "train_only_covscale_noise_channel_time_mask",
            "independent_event": False,
            "noise_scale": float(noise_scale),
            "channel_mask_probability": float(channel_mask_probability),
            "time_mask_probability": float(time_mask_probability),
        }
    )
    return SequenceAugmentation(augmented, provenance)


def masked_pretraining_batch(
    sequences: np.ndarray,
    *,
    mask_fraction: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic masked-reconstruction batch for self-supervision."""

    if not 0 < mask_fraction < 1:
        raise ValueError("mask_fraction deve estar entre 0 e 1.")
    rng = np.random.default_rng(random_state)
    target = np.asarray(sequences, dtype="float32")
    mask = rng.random(target.shape) < mask_fraction
    corrupted = target.copy()
    corrupted[mask] = 0.0
    return corrupted, target.copy(), mask


def channel_occlusion_importance(
    model,
    sequences: np.ndarray,
    *,
    channel_names: Sequence[str] | None = None,
    baseline: float = 0.0,
    target_state_ids: Sequence[int] | np.ndarray | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """OOS channel ablation, including Brier-loss degradation by state."""

    framework = _require_torch()
    names = list(channel_names or [f"channel_{i}" for i in range(sequences.shape[-1])])
    if len(names) != sequences.shape[-1]:
        raise ValueError("channel_names nao corresponde ao numero de canais.")
    model = model.to(device)
    model.eval()
    values = framework.as_tensor(sequences, dtype=framework.float32, device=device)
    target_ids = (
        np.asarray(target_state_ids, dtype=int)
        if target_state_ids is not None
        else None
    )
    if target_ids is not None and len(target_ids) != len(sequences):
        raise ValueError("target_state_ids precisa ter um valor por sequencia.")
    with framework.no_grad():
        reference = framework.softmax(model(values)["state_logits"], dim=1)
        rows: list[dict[str, object]] = []
        for channel, name in enumerate(names):
            changed_values = values.clone()
            changed_values[..., channel] = baseline
            changed = framework.softmax(model(changed_values)["state_logits"], dim=1)
            scopes = [("all", np.ones(len(sequences), dtype=bool))]
            if target_ids is not None:
                scopes.extend(
                    (ENSO_STATES[state_id], target_ids == state_id)
                    for state_id in sorted(set(target_ids.tolist()))
                )
            for scope, selected in scopes:
                if not selected.any():
                    continue
                selected_tensor = framework.as_tensor(
                    selected, dtype=framework.bool, device=device
                )
                row = {
                    "channel": name,
                    "channel_index": channel,
                    "observed_state_scope": scope,
                    "n_test_sequences": int(selected.sum()),
                    "mean_absolute_probability_change": float(
                        (reference[selected_tensor] - changed[selected_tensor]).abs().mean().cpu()
                    ),
                    "delta_brier_occlusion_oos": np.nan,
                    "evaluation_scope": "out_of_sample_only",
                }
                if target_ids is not None:
                    selected_ids = framework.as_tensor(
                        target_ids[selected], dtype=framework.long, device=device
                    )
                    truth = framework.nn.functional.one_hot(
                        selected_ids, num_classes=reference.shape[1]
                    ).float()
                    base_loss = ((reference[selected_tensor] - truth) ** 2).mean()
                    changed_loss = ((changed[selected_tensor] - truth) ** 2).mean()
                    row["delta_brier_occlusion_oos"] = float(
                        (changed_loss - base_loss).cpu()
                    )
                rows.append(row)
    order = "delta_brier_occlusion_oos" if target_ids is not None else "mean_absolute_probability_change"
    return pd.DataFrame(rows).sort_values(order, ascending=False).reset_index(drop=True)


def scalar_occlusion_importance(
    model,
    spatial_sequences: np.ndarray,
    scalar_sequences: np.ndarray,
    scalar_names: Sequence[str],
    *,
    target_state_ids: Sequence[int] | np.ndarray | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Rank all 31 variables by OOS Brier degradation, globally and by state."""

    framework = _require_torch()
    if scalar_sequences.ndim != 3 or scalar_sequences.shape[-1] != len(scalar_names):
        raise ValueError("scalar_sequences deve ser N,T,31 e scalar_names deve corresponder.")
    model = model.to(device)
    model.eval()
    spatial = framework.as_tensor(spatial_sequences, dtype=framework.float32, device=device)
    scalars = framework.as_tensor(scalar_sequences, dtype=framework.float32, device=device)
    target_ids = (
        np.asarray(target_state_ids, dtype=int)
        if target_state_ids is not None
        else None
    )
    if target_ids is not None and len(target_ids) != len(scalar_sequences):
        raise ValueError("target_state_ids precisa ter um valor por sequencia.")
    with framework.no_grad():
        reference = framework.softmax(
            model(spatial, scalar_sequence=scalars)["state_logits"], dim=1
        )
        rows: list[dict[str, object]] = []
        for channel, name in enumerate(scalar_names):
            changed_scalars = scalars.clone()
            changed_scalars[..., channel] = 0.0
            changed = framework.softmax(
                model(spatial, scalar_sequence=changed_scalars)["state_logits"], dim=1
            )
            scopes = [("all", np.ones(len(scalar_sequences), dtype=bool))]
            if target_ids is not None:
                scopes.extend(
                    (ENSO_STATES[state_id], target_ids == state_id)
                    for state_id in sorted(set(target_ids.tolist()))
                )
            for scope, selected in scopes:
                if not selected.any():
                    continue
                selected_tensor = framework.as_tensor(
                    selected, dtype=framework.bool, device=device
                )
                row = {
                    "variable": str(name),
                    "variable_index": channel,
                    "observed_state_scope": scope,
                    "n_test_sequences": int(selected.sum()),
                    "mean_absolute_probability_change": float(
                        (reference[selected_tensor] - changed[selected_tensor]).abs().mean().cpu()
                    ),
                    "delta_brier_occlusion_oos": np.nan,
                    "evaluation_scope": "out_of_sample_only",
                    "n_physical_variables": len(scalar_names),
                }
                if target_ids is not None:
                    selected_ids = framework.as_tensor(
                        target_ids[selected], dtype=framework.long, device=device
                    )
                    truth = framework.nn.functional.one_hot(
                        selected_ids, num_classes=reference.shape[1]
                    ).float()
                    base_loss = ((reference[selected_tensor] - truth) ** 2).mean()
                    changed_loss = ((changed[selected_tensor] - truth) ** 2).mean()
                    row["delta_brier_occlusion_oos"] = float(
                        (changed_loss - base_loss).cpu()
                    )
                rows.append(row)
    order = "delta_brier_occlusion_oos" if target_ids is not None else "mean_absolute_probability_change"
    return pd.DataFrame(rows).sort_values(order, ascending=False, ignore_index=True)


def integrated_gradient_channel_importance(
    model,
    sequences: np.ndarray,
    *,
    target_state: int,
    steps: int = 16,
    channel_names: Sequence[str] | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Integrated gradients aggregated by named input channel."""

    framework = _require_torch()
    model = model.to(device)
    model.eval()
    values = framework.as_tensor(sequences, dtype=framework.float32, device=device)
    baseline = framework.zeros_like(values)
    total_gradient = framework.zeros_like(values)
    for alpha in framework.linspace(0.0, 1.0, steps, device=device):
        interpolated = (baseline + alpha * (values - baseline)).detach().requires_grad_(True)
        score = model(interpolated)["state_logits"][:, int(target_state)].sum()
        gradient = framework.autograd.grad(score, interpolated)[0]
        total_gradient += gradient
    attribution = (values - baseline) * total_gradient / float(steps)
    importance = attribution.abs().mean(dim=(0, 1, 2, 3)).detach().cpu().numpy()
    names = list(channel_names or [f"channel_{i}" for i in range(len(importance))])
    return pd.DataFrame(
        {
            "channel": names,
            "integrated_gradient_abs_mean": importance,
            "target_state": ENSO_STATES[int(target_state)],
            "evaluation_scope": "out_of_sample_only",
        }
    ).sort_values("integrated_gradient_abs_mean", ascending=False, ignore_index=True)
