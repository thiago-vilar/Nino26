"""Fase 7 - ciclo ENSO com redes neurais ConvLSTM.

Mesmo mecanismo das Fases 3 e 5, agora com uma rede espaco-temporal ConvLSTM que
recebe sequencias de campos do Pacifico equatorial (time, lat, lon, canais) e
aprende a evolucao genese -> crescimento -> pico -> decaimento, identificando
ciclos EN/LN e mapeando as 4 fases. A importancia por variavel/etapa e obtida por
oclusao de canais (ablacao espaco-temporal).

TensorFlow/Keras e importado sob guarda: a preparacao de dados (numpy/xarray) roda
sem TF; o treino exige TF+GPU na maquina do usuario.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _require_tf():
    try:
        import tensorflow as tf  # type: ignore
        return tf
    except ImportError as exc:  # pragma: no cover - dependencia pesada opcional
        raise ImportError(
            "TensorFlow nao instalado. Fase 7 (ConvLSTM) requer 'pip install "
            "tensorflow' e, idealmente, GPU. A preparacao de dados funciona sem TF."
        ) from exc


def load_pacific_cube(
    zarr_dir: str,
    variables: Sequence[str],
    *,
    lat_bounds: tuple[float, float] = (-15.0, 15.0),
    lon_bounds: tuple[float, float] = (120.0, 290.0),
    time_name: str = "time",
) -> "np.ndarray":
    """Load regridded Pacific fields into a (time, lat, lon, channel) cube."""

    import xarray as xr

    dataset = xr.open_zarr(zarr_dir, consolidated=False)
    lat = "lat" if "lat" in dataset.coords else "latitude"
    lon = "lon" if "lon" in dataset.coords else "longitude"
    sub = dataset.sel({lat: slice(*lat_bounds), lon: slice(*lon_bounds)})
    channels = [sub[v] for v in variables if v in sub]
    if not channels:
        raise KeyError(f"Nenhuma variavel {variables} no cubo {zarr_dir}.")
    stacked = xr.concat(channels, dim="channel").transpose(time_name, lat, lon, "channel")
    return np.asarray(stacked.values, dtype="float32")


def make_sequences(
    cube: np.ndarray,
    labels: np.ndarray | None,
    *,
    seq_len: int = 24,
    horizon: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build sliding spatiotemporal sequences (samples, seq_len, lat, lon, channel).

    ``labels[t]`` (fase na semana t) e alinhado ao fim de cada janela + ``horizon``.
    Puro numpy: testavel sem TensorFlow.
    """

    if cube.ndim != 4:
        raise ValueError("cube deve ter forma (time, lat, lon, channel).")
    n_time = cube.shape[0]
    last = n_time - horizon
    starts = range(0, last - seq_len + 1)
    sequences = np.stack([cube[s : s + seq_len] for s in starts]) if last - seq_len + 1 > 0 else np.empty((0,))
    if labels is None:
        return sequences, None
    targets = np.array([labels[s + seq_len - 1 + horizon] for s in starts])
    return sequences, targets


def build_convlstm_classifier(
    input_shape: tuple[int, int, int, int],
    n_classes: int = 4,
    *,
    filters: Sequence[int] = (32, 32),
):
    """Keras ConvLSTM2D -> classificacao das 4 fases (requer TensorFlow)."""

    tf = _require_tf()
    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=input_shape)  # (seq_len, lat, lon, channel)
    x = inputs
    for i, f in enumerate(filters):
        x = layers.ConvLSTM2D(
            f, (3, 3), padding="same", return_sequences=i < len(filters) - 1,
        )(x)
        x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="convlstm_cycle_classifier")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def chronological_split(n_samples: int, *, val_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Train on the past, validate on the most recent block (no shuffling)."""

    cut = int(round(n_samples * (1.0 - val_fraction)))
    return np.arange(cut), np.arange(cut, n_samples)


def channel_occlusion_importance(model, sequences: np.ndarray, *, baseline: float = 0.0) -> pd.DataFrame:
    """Rank channels by the accuracy drop when each is occluded (XAI espacial)."""

    _require_tf()
    reference = model.predict(sequences, verbose=0)
    rows: list[dict[str, object]] = []
    for channel in range(sequences.shape[-1]):
        perturbed = sequences.copy()
        perturbed[..., channel] = baseline
        changed = model.predict(perturbed, verbose=0)
        rows.append({
            "canal": channel,
            "delta_softmax_medio": float(np.mean(np.abs(reference - changed))),
        })
    return pd.DataFrame(rows).sort_values("delta_softmax_medio", ascending=False).reset_index(drop=True)
