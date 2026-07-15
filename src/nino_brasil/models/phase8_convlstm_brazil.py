"""Phase 8: Pacific ConvLSTM encoder to the exact native CHIRPS grid.

Targets are never regridded or interpolated.  The neural decoder maps its latent
Pacific representation to the existing CHIRPS grid shape and is evaluated on
the original pixel ids.  A Brazil/land mask and cell-area weights affect the
loss only; they do not alter pixel geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from nino_brasil.models.phase7_convlstm import ConvLSTMEncoder, _require_torch
from nino_brasil.targets.chirps_native import native_grid_hash

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[assignment,misc]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


@dataclass(frozen=True)
class NativeCHIRPSGrid:
    lat: np.ndarray
    lon: np.ndarray
    pixel_ids: np.ndarray
    grid_sha256: str


def native_chirps_grid(
    lat: Sequence[float],
    lon: Sequence[float],
    *,
    pixel_ids: Sequence[object] | None = None,
    expected_grid_sha256: str | None = None,
) -> NativeCHIRPSGrid:
    """Build stable row-major pixel ids and a full SHA256 geometry fingerprint."""

    lat_values = np.asarray(lat, dtype=float)
    lon_values = np.asarray(lon, dtype=float)
    if lat_values.ndim != 1 or lon_values.ndim != 1:
        raise ValueError("lat e lon devem ser vetores unidimensionais do grid nativo.")
    ids = (
        np.asarray(pixel_ids, dtype=object).reshape(-1)
        if pixel_ids is not None
        else np.arange(len(lat_values) * len(lon_values), dtype=np.int64)
    )
    if len(ids) != len(lat_values) * len(lon_values) or pd.Index(ids).duplicated().any():
        raise ValueError("pixel_ids precisa ser unico e cobrir toda a grade latitude x longitude.")
    numeric_ids = pd.to_numeric(pd.Series(ids), errors="coerce")
    canonical_ids = np.arange(len(ids), dtype=np.int64)
    if numeric_ids.isna().any() or not np.array_equal(
        numeric_ids.to_numpy(dtype=np.int64), canonical_ids
    ):
        raise ValueError(
            "pixel_ids deve seguir o contrato row-major: pixel_id[y,x] = y*n_lon+x."
        )
    computed = native_grid_hash(lat_values, lon_values)
    if expected_grid_sha256 and str(expected_grid_sha256) != computed:
        raise ValueError(
            "Hash CHIRPS informado diverge das coordenadas nativas calculadas."
        )
    return NativeCHIRPSGrid(
        lat=lat_values,
        lon=lon_values,
        pixel_ids=ids,
        grid_sha256=computed,
    )


def align_pacific_to_brazil(
    pacific_sequences: np.ndarray,
    brazil_fields: np.ndarray,
    *,
    horizon_weeks: int = 4,
    sequence_end_indices: Sequence[int] | None = None,
    seq_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Align each sequence end ``t`` with the native Brazil field at ``t+h``.

    ``pacific_sequences`` no longer hides its offset.  Supply the exact end
    indices returned by F7, or ``seq_len`` when windows were built consecutively
    from index zero.  This fixes the former ~``seq_len-1`` week target shift.
    """

    if pacific_sequences.ndim != 5 or brazil_fields.ndim != 3:
        raise ValueError("Esperado Pacifico N,T,H,W,C e CHIRPS time,lat,lon.")
    if horizon_weeks < 0:
        raise ValueError("horizon_weeks deve ser >= 0.")
    if sequence_end_indices is None:
        if seq_len is None:
            raise ValueError(
                "Informe sequence_end_indices ou seq_len; sem o offset a associacao temporal e ambigua."
            )
        sequence_end_indices = np.arange(len(pacific_sequences), dtype=int) + int(seq_len) - 1
    ends = np.asarray(sequence_end_indices, dtype=int)
    if len(ends) != len(pacific_sequences):
        raise ValueError("sequence_end_indices precisa ter um valor por sequencia.")
    targets = ends + int(horizon_weeks)
    valid = (targets >= 0) & (targets < len(brazil_fields))
    if not valid.any():
        raise ValueError("Series curtas demais para o horizonte pedido.")
    return pacific_sequences[valid], brazil_fields[targets[valid]]


def align_pacific_to_brazil_times(
    pacific_sequences: np.ndarray,
    sequence_end_times: Sequence[pd.Timestamp],
    brazil_fields: np.ndarray,
    brazil_times: Sequence[pd.Timestamp],
    *,
    horizon_weeks: int = 4,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Timestamp-first alignment used by the official F8 runner."""

    ends = pd.DatetimeIndex(sequence_end_times)
    target_times = ends + pd.Timedelta(weeks=int(horizon_weeks))
    brazil_index = pd.DatetimeIndex(brazil_times)
    positions = brazil_index.get_indexer(target_times)
    valid = positions >= 0
    if not valid.any():
        raise ValueError("Nenhum timestamp Pacifico->CHIRPS alinhou exatamente.")
    return pacific_sequences[valid], brazil_fields[positions[valid]], target_times[valid]


if nn is not None:

    class PacificToBrazilConvLSTM(nn.Module):
        """Convolutional decoder with no flattened dense pixel projection."""

        def __init__(
            self,
            input_channels: int,
            brazil_shape: tuple[int, int],
            *,
            hidden_channels: Sequence[int] = (32, 32),
            decoder_channels: int = 32,
            distribution: str = "gaussian",
            quantiles: Sequence[float] = (0.05, 0.5, 0.95),
            scalar_channels: int = 0,
        ):
            super().__init__()
            if distribution not in {"gaussian", "quantile"}:
                raise ValueError("distribution deve ser 'gaussian' ou 'quantile'.")
            self.encoder = ConvLSTMEncoder(input_channels, hidden_channels)
            self.brazil_shape = (int(brazil_shape[0]), int(brazil_shape[1]))
            self.distribution = distribution
            self.quantiles = tuple(float(value) for value in quantiles)
            if distribution == "quantile":
                if (
                    len(self.quantiles) < 3
                    or len(set(self.quantiles)) != len(self.quantiles)
                    or tuple(sorted(self.quantiles)) != self.quantiles
                    or not all(0.0 < value < 1.0 for value in self.quantiles)
                    or 0.5 not in self.quantiles
                ):
                    raise ValueError(
                        "quantiles devem ser estritamente crescentes em (0,1) e incluir 0.5."
                    )
            self.scalar_channels = int(scalar_channels)
            scalar_latent = min(64, max(16, self.scalar_channels * 2)) if self.scalar_channels else 0
            self.scalar_encoder = (
                nn.GRU(self.scalar_channels, scalar_latent, batch_first=True)
                if self.scalar_channels
                else None
            )
            self.scalar_film = (
                nn.Linear(scalar_latent, 2 * self.encoder.output_channels)
                if self.scalar_channels
                else None
            )
            output_channels = 2 if distribution == "gaussian" else len(self.quantiles)
            # The Pacific and Brazil grids are distinct physical domains.  A
            # direct spatial resize would invent a positional correspondence.
            # Decode instead from global Pacific context plus explicit target
            # coordinates, while keeping the CHIRPS output grid untouched.
            self.context_projection = nn.Conv2d(
                2 * self.encoder.output_channels, decoder_channels, 1
            )
            self.coordinate_projection = nn.Conv2d(2, decoder_channels, 1)
            # Coordinate-aware cross-domain attention retains where a Pacific
            # signal occurred without resizing it onto Brazil.  Keys/values are
            # pooled only within the Pacific domain for tractable memory; every
            # CHIRPS query remains an original target pixel.
            self.attention_query = nn.Conv2d(2, decoder_channels, 1)
            self.attention_key = nn.Conv2d(
                self.encoder.output_channels, decoder_channels, 1
            )
            self.attention_value = nn.Conv2d(
                self.encoder.output_channels, decoder_channels, 1
            )
            self.source_coordinate_projection = nn.Conv2d(
                2, decoder_channels, 1
            )
            self.attention_source_shape = (8, 16)
            target_y = torch.linspace(-1.0, 1.0, self.brazil_shape[0])
            target_x = torch.linspace(-1.0, 1.0, self.brazil_shape[1])
            yy, xx = torch.meshgrid(target_y, target_x, indexing="ij")
            self.register_buffer(
                "target_coordinates", torch.stack([yy, xx], dim=0)[None]
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(decoder_channels, output_channels, 1),
            )

        def forward(
            self,
            sequence: Tensor,
            *,
            scalar_sequence: Tensor | None = None,
            channel_last: bool = True,
        ) -> dict[str, Tensor]:
            latent = self.encoder(sequence, channel_last=channel_last)
            if self.scalar_encoder is not None:
                if scalar_sequence is None:
                    raise ValueError("Forneca scalar_sequence com as 31 variaveis F2.")
                _, scalar_hidden = self.scalar_encoder(scalar_sequence)
                gamma, beta = self.scalar_film(scalar_hidden[-1]).chunk(2, dim=1)
                latent = latent * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
            mean_context = F.adaptive_avg_pool2d(latent, 1)
            max_context = F.adaptive_max_pool2d(latent, 1)
            context = self.context_projection(
                torch.cat([mean_context, max_context], dim=1)
            ).expand(-1, -1, *self.brazil_shape)
            coordinates = self.target_coordinates.expand(len(latent), -1, -1, -1)
            source_shape = (
                min(self.attention_source_shape[0], latent.shape[-2]),
                min(self.attention_source_shape[1], latent.shape[-1]),
            )
            source = F.adaptive_avg_pool2d(latent, source_shape)
            source_y = torch.linspace(
                -1.0, 1.0, source_shape[0], device=latent.device, dtype=latent.dtype
            )
            source_x = torch.linspace(
                -1.0, 1.0, source_shape[1], device=latent.device, dtype=latent.dtype
            )
            source_yy, source_xx = torch.meshgrid(source_y, source_x, indexing="ij")
            source_coordinates = torch.stack([source_yy, source_xx], dim=0)[None].expand(
                len(latent), -1, -1, -1
            )
            query = self.attention_query(coordinates).flatten(2).transpose(1, 2)
            key = (
                self.attention_key(source)
                + self.source_coordinate_projection(source_coordinates)
            ).flatten(2)
            value = self.attention_value(source).flatten(2).transpose(1, 2)
            attention = torch.softmax(
                torch.bmm(query, key) / np.sqrt(float(key.shape[1])), dim=-1
            )
            attended = torch.bmm(attention, value).transpose(1, 2).reshape(
                len(latent), -1, *self.brazil_shape
            )
            decoded_input = (
                context + attended + self.coordinate_projection(coordinates)
            )
            output = self.decoder(decoded_input)
            if self.distribution == "gaussian":
                return {
                    "mean": output[:, 0],
                    "log_scale": output[:, 1].clamp(-6.0, 4.0),
                    "spatial_latent": latent,
                }
            first = output[:, :1]
            increments = F.softplus(output[:, 1:])
            ordered = torch.cat(
                [first, first + torch.cumsum(increments, dim=1)], dim=1
            )
            return {"quantiles": ordered, "spatial_latent": latent}

else:  # pragma: no cover

    class PacificToBrazilConvLSTM:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()


def build_convlstm_encoder_decoder(
    pacific_shape: tuple[int, int, int, int],
    brazil_shape: tuple[int, int],
    *,
    filters: Sequence[int] = (32, 32),
    distribution: str = "gaussian",
    scalar_channels: int = 0,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
):
    """Build the PyTorch spatial encoder-decoder for an exact target shape."""

    _require_torch()
    if len(pacific_shape) != 4:
        raise ValueError("pacific_shape deve ser (seq_len, lat, lon, channel).")
    return PacificToBrazilConvLSTM(
        input_channels=int(pacific_shape[-1]),
        brazil_shape=brazil_shape,
        hidden_channels=filters,
        distribution=distribution,
        scalar_channels=scalar_channels,
        quantiles=quantiles,
    )


def brazil_cell_area_weights(lat: Sequence[float], lon: Sequence[float]) -> np.ndarray:
    """Cos(latitude) cell weights on the unchanged regular CHIRPS grid."""

    lat_values = np.asarray(lat, dtype=float)
    lon_values = np.asarray(lon, dtype=float)
    weights = np.cos(np.deg2rad(lat_values))[:, None] * np.ones((1, len(lon_values)))
    return weights / np.nanmean(weights)


def masked_area_weighted_gaussian_nll(
    mean: Tensor,
    log_scale: Tensor,
    target: Tensor,
    *,
    brazil_mask: Tensor,
    area_weight: Tensor,
    valid_mask: Tensor | None = None,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Gaussian NLL on Brazil cells; original non-Brazil pixels remain in output."""

    _require_torch()
    if mean.shape != target.shape or log_scale.shape != target.shape:
        raise ValueError("mean/log_scale/target precisam ter a mesma grade nativa.")
    weight = brazil_mask.to(mean.dtype) * area_weight.to(mean.dtype)
    if valid_mask is not None:
        weight = weight * valid_mask.to(mean.dtype)
    while weight.ndim < mean.ndim:
        weight = weight.unsqueeze(0)
    nll = 0.5 * (((target - mean) / log_scale.exp()) ** 2 + 2.0 * log_scale)
    weighted = nll * weight
    if weighted.ndim == 3:
        per_sample = weighted.sum(dim=(1, 2)) / weight.sum(dim=(1, 2)).clamp_min(1e-8)
        if sample_weight is not None:
            return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)
        return per_sample.mean()
    return weighted.sum() / weight.sum().clamp_min(1e-8)


def masked_area_weighted_quantile_loss(
    prediction: Tensor,
    target: Tensor,
    quantiles: Sequence[float],
    *,
    brazil_mask: Tensor,
    area_weight: Tensor,
    valid_mask: Tensor | None = None,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Pinball loss for probabilistic wet/dry extremes on native pixels."""

    _require_torch()
    q_values = tuple(float(value) for value in quantiles)
    if (
        tuple(sorted(q_values)) != q_values
        or len(set(q_values)) != len(q_values)
        or not all(0.0 < value < 1.0 for value in q_values)
    ):
        raise ValueError("quantiles devem ser unicos e crescentes em (0,1).")
    if prediction.ndim != 4 or prediction.shape[1] != len(quantiles):
        raise ValueError("prediction deve ser N,Q,lat,lon com Q=len(quantiles).")
    error = target[:, None] - prediction
    q = torch.as_tensor(quantiles, dtype=prediction.dtype, device=prediction.device)[None, :, None, None]
    loss = torch.maximum(q * error, (q - 1.0) * error)
    weight = brazil_mask.to(prediction.dtype) * area_weight.to(prediction.dtype)
    if valid_mask is not None:
        weight = weight * valid_mask.to(prediction.dtype)
    if weight.ndim == 2:
        weight = weight[None, None]
    elif weight.ndim == 3:
        weight = weight[:, None]
    weighted = loss * weight
    if weighted.ndim == 4:
        denominator = weight.sum(dim=(1, 2, 3)).clamp_min(1e-8) * len(quantiles)
        per_sample = weighted.sum(dim=(1, 2, 3)) / denominator
        if sample_weight is not None:
            return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)
        return per_sample.mean()
    return weighted.sum() / (weight.sum().clamp_min(1e-8) * len(quantiles))


def pixel_skill_table(
    observed: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray,
    grid: NativeCHIRPSGrid,
    *,
    brazil_mask: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-original-pixel OOS metrics; regional summaries are downstream only."""

    if observed.shape != predicted.shape or observed.shape != baseline.shape:
        raise ValueError("observed/predicted/baseline precisam da mesma forma time,lat,lon.")
    if observed.shape[1:] != (len(grid.lat), len(grid.lon)):
        raise ValueError("A forma do alvo nao corresponde ao grid CHIRPS fingerprinted.")
    rows: list[dict[str, object]] = []
    mask = np.ones(observed.shape[1:], dtype=bool) if brazil_mask is None else brazil_mask.astype(bool)
    for y, latitude in enumerate(grid.lat):
        for x, longitude in enumerate(grid.lon):
            obs = observed[:, y, x]
            pred = predicted[:, y, x]
            base = baseline[:, y, x]
            valid = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(base)
            pixel_id = grid.pixel_ids[y * len(grid.lon) + x]
            row: dict[str, object] = {
                "pixel_id": pixel_id,
                "lat": float(latitude),
                "lon": float(longitude),
                "is_brazil": bool(mask[y, x]),
                "n_test": int(valid.sum()),
                "grid_sha256": grid.grid_sha256,
            }
            if valid.sum() >= 3:
                rmse = float(np.sqrt(np.mean((pred[valid] - obs[valid]) ** 2)))
                rmse_base = float(np.sqrt(np.mean((base[valid] - obs[valid]) ** 2)))
                row.update(
                    {
                        "rmse": rmse,
                        "mae": float(np.mean(np.abs(pred[valid] - obs[valid]))),
                        "r": (
                            float(np.corrcoef(obs[valid], pred[valid])[0, 1])
                            if obs[valid].std() > 0 and pred[valid].std() > 0
                            else np.nan
                        ),
                        "rmse_baseline": rmse_base,
                        "skill_rmse_vs_baseline": (
                            1.0 - rmse / rmse_base if rmse_base > 0 else np.nan
                        ),
                    }
                )
                if lower is not None and upper is not None:
                    interval_valid = valid & np.isfinite(lower[:, y, x]) & np.isfinite(upper[:, y, x])
                    row["interval_coverage"] = (
                        float(
                            np.mean(
                                (obs[interval_valid] >= lower[:, y, x][interval_valid])
                                & (obs[interval_valid] <= upper[:, y, x][interval_valid])
                            )
                        )
                        if interval_valid.any()
                        else np.nan
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def field_channel_occlusion_importance(
    model,
    spatial_sequences: np.ndarray,
    scalar_sequences: np.ndarray,
    *,
    spatial_names: Sequence[str],
    scalar_names: Sequence[str],
    brazil_mask: np.ndarray,
    observed: np.ndarray | None = None,
    area_weight: np.ndarray | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """OOS ablation for fields and all 31 scalar F2 variables.

    When observations are supplied, importance is the increase in
    area-weighted RMSE.  Field-change sensitivity is retained as a secondary
    diagnostic and is never mislabeled as predictive utility.
    """

    framework = _require_torch()
    model = model.to(device)
    model.eval()
    spatial = framework.as_tensor(spatial_sequences, dtype=framework.float32, device=device)
    scalars = framework.as_tensor(scalar_sequences, dtype=framework.float32, device=device)
    mask = framework.as_tensor(brazil_mask, dtype=framework.bool, device=device)
    target = (
        framework.as_tensor(observed, dtype=framework.float32, device=device)
        if observed is not None
        else None
    )
    weights = framework.as_tensor(
        area_weight if area_weight is not None else np.ones_like(brazil_mask),
        dtype=framework.float32,
        device=device,
    )

    def rmse(prediction: Tensor) -> Tensor:
        if target is None:
            return framework.tensor(float("nan"), device=device)
        valid = framework.isfinite(target) & mask[None]
        weighted = weights[None] * valid
        return framework.sqrt(
            (weighted * (prediction - framework.nan_to_num(target)) ** 2).sum()
            / weighted.sum().clamp_min(1e-8)
        )

    with framework.no_grad():
        reference_output = model(spatial, scalar_sequence=scalars)
        median_index = (
            next(
                index
                for index, value in enumerate(getattr(model, "quantiles", ()))
                if abs(float(value) - 0.5) < 1e-9
            )
            if "quantiles" in reference_output
            else None
        )
        reference = (
            reference_output["mean"]
            if "mean" in reference_output
            else reference_output["quantiles"][:, median_index]
        )
        reference_rmse = rmse(reference)
        rows: list[dict[str, object]] = []
        for channel, name in enumerate(spatial_names):
            changed_input = spatial.clone()
            changed_input[..., channel] = 0.0
            changed_output = model(changed_input, scalar_sequence=scalars)
            changed = (
                changed_output["mean"]
                if "mean" in changed_output
                else changed_output["quantiles"][:, median_index]
            )
            rows.append(
                {
                    "input_kind": "spatial_field",
                    "variable": str(name),
                    "mean_absolute_field_change_brazil": float(
                        (reference - changed).abs()[:, mask].mean().cpu()
                    ),
                    "delta_rmse_occlusion_oos": (
                        float((rmse(changed) - reference_rmse).cpu())
                        if target is not None
                        else np.nan
                    ),
                    "evaluation_scope": "out_of_sample_only",
                }
            )
        for channel, name in enumerate(scalar_names):
            changed_scalar = scalars.clone()
            changed_scalar[..., channel] = 0.0
            changed_output = model(spatial, scalar_sequence=changed_scalar)
            changed = (
                changed_output["mean"]
                if "mean" in changed_output
                else changed_output["quantiles"][:, median_index]
            )
            rows.append(
                {
                    "input_kind": "scalar_f2",
                    "variable": str(name),
                    "mean_absolute_field_change_brazil": float(
                        (reference - changed).abs()[:, mask].mean().cpu()
                    ),
                    "delta_rmse_occlusion_oos": (
                        float((rmse(changed) - reference_rmse).cpu())
                        if target is not None
                        else np.nan
                    ),
                    "evaluation_scope": "out_of_sample_only",
                }
            )
    order = "delta_rmse_occlusion_oos" if target is not None else "mean_absolute_field_change_brazil"
    return pd.DataFrame(rows).sort_values(order, ascending=False, ignore_index=True)
