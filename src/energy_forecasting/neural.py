"""Optional, correctly shaped Time2Vec-LSTM architecture."""

from __future__ import annotations

try:
    from tensorflow import keras
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "The LSTM model requires TensorFlow. Install requirements-deep-learning.txt."
    ) from error

from .layers import Time2Vec


def build_time2vec_lstm(
    *,
    sequence_length: int,
    feature_count: int,
    forecast_horizon: int = 24,
    time_embedding_dim: int = 6,
    lstm_units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
):
    """Build a two-input LSTM that emits the complete forecast horizon.

    Time2Vec receives only a normalized scalar time channel. Historical weather,
    load, and solar features enter through a separate tensor before concatenation.
    This corrects the invalid single-input Sequential design in the proposal.
    """
    time_input = keras.Input((sequence_length, 1), name="normalized_time")
    feature_input = keras.Input((sequence_length, feature_count), name="historical_features")
    time_embedding = Time2Vec(time_embedding_dim, name="time2vec")(time_input)
    merged = keras.layers.Concatenate(name="merge_time_and_features")([time_embedding, feature_input])
    hidden = keras.layers.LSTM(lstm_units, name="sequence_encoder")(merged)
    hidden = keras.layers.Dropout(dropout, name="dropout")(hidden)
    forecast = keras.layers.Dense(forecast_horizon, name="forecast")(hidden)
    model = keras.Model([time_input, feature_input], forecast, name="time2vec_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model
