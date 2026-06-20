"""Optional TensorFlow/Keras layers for sequential experiments.

Install ``requirements-deep-learning.txt`` before importing this module. Keeping
TensorFlow optional allows the fast tabular baseline to run on ordinary laptops.
"""

from __future__ import annotations

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as error:  # pragma: no cover - exercised only without optional deps
    raise ImportError(
        "Time2Vec requires TensorFlow. Install requirements-deep-learning.txt."
    ) from error


@keras.utils.register_keras_serializable(package="energy_forecasting")
class Time2Vec(keras.layers.Layer):
    """Learn one linear and ``output_dim - 1`` periodic time components.

    Input shape is ``(batch, sequence_length, 1)`` and output shape is
    ``(batch, sequence_length, output_dim)``. ``output_dim`` must be at least two;
    this avoids the silent no-periodic-term behaviour in the original proposal.
    """

    def __init__(self, output_dim: int = 6, **kwargs):
        super().__init__(**kwargs)
        if output_dim < 2:
            raise ValueError("output_dim must be at least 2")
        self.output_dim = int(output_dim)

    def build(self, input_shape):
        if input_shape[-1] != 1:
            raise ValueError("Time2Vec expects a single scalar time channel")
        self.linear_weight = self.add_weight(name="linear_weight", shape=(1,), initializer="glorot_uniform")
        self.linear_bias = self.add_weight(name="linear_bias", shape=(1,), initializer="zeros")
        self.periodic_weight = self.add_weight(
            name="periodic_weight", shape=(self.output_dim - 1,), initializer="glorot_uniform"
        )
        self.periodic_bias = self.add_weight(
            name="periodic_bias", shape=(self.output_dim - 1,), initializer="zeros"
        )
        super().build(input_shape)

    def call(self, time_input):
        time_input = tf.cast(time_input, self.compute_dtype)
        linear = time_input * self.linear_weight + self.linear_bias
        periodic = tf.sin(time_input * self.periodic_weight + self.periodic_bias)
        return tf.concat([linear, periodic], axis=-1)

    def get_config(self):
        return {**super().get_config(), "output_dim": self.output_dim}
