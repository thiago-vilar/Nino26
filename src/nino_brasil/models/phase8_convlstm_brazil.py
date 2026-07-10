"""Fase 8 - distribuicao no Brasil com redes neurais ConvLSTM.

Mesmo estudo espaco-temporal das Fases 4 e 6, agora com uma ConvLSTM
encoder-decoder: o encoder le sequencias do Pacifico equatorial e o decoder projeta
o campo de anomalia de chuva do Brasil (lat, lon) em t+horizon. A avaliacao e feita
por pixel e agregada por regiao e bioma (reaproveitando maps.spatial_support).

TensorFlow/Keras sob guarda; treino exige TF+GPU na maquina do usuario.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from nino_brasil.models.phase7_convlstm import _require_tf, make_sequences  # noqa: F401


def build_convlstm_encoder_decoder(
    pacific_shape: tuple[int, int, int, int],
    brazil_shape: tuple[int, int],
    *,
    filters: Sequence[int] = (32, 32),
):
    """Encoder ConvLSTM (Pacifico) -> decoder que projeta a chuva do Brasil.

    ``pacific_shape`` = (seq_len, lat_pac, lon_pac, channel);
    ``brazil_shape``  = (lat_br, lon_br) do alvo de anomalia de chuva.
    """

    tf = _require_tf()
    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=pacific_shape)
    x = inputs
    for i, f in enumerate(filters):
        x = layers.ConvLSTM2D(f, (3, 3), padding="same", return_sequences=i < len(filters) - 1)(x)
        x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(brazil_shape[0] * brazil_shape[1] // 4, activation="relu")(x)
    x = layers.Dense(brazil_shape[0] * brazil_shape[1], activation="linear")(x)
    outputs = layers.Reshape(brazil_shape)(x)
    model = tf.keras.Model(inputs, outputs, name="convlstm_brazil_teleconnection")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def align_pacific_to_brazil(
    pacific_sequences: np.ndarray,
    brazil_fields: np.ndarray,
    *,
    horizon_weeks: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair each Pacific sequence with the Brazil rainfall field ``horizon`` ahead."""

    n = min(len(pacific_sequences), len(brazil_fields) - horizon_weeks)
    if n <= 0:
        raise ValueError("Series curtas demais para o horizonte pedido.")
    return pacific_sequences[:n], brazil_fields[horizon_weeks : horizon_weeks + n]
