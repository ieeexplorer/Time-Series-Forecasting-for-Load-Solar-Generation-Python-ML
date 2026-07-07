import numpy as np
import pandas as pd

from energy_forecasting.models import model_factory


def test_ridge_regression_factory_builds_fresh_trainable_pipeline() -> None:
    X = pd.DataFrame({"x": np.arange(20, dtype=float), "bias_feature": 1.0})
    y = 3 * X["x"] + 2

    factory = model_factory("ridge_regression", {"alpha": 0.1})
    first = factory()
    second = factory()

    assert first is not second
    first.fit(X, y)
    assert np.allclose(first.predict(X).reshape(-1), y, atol=0.2)
