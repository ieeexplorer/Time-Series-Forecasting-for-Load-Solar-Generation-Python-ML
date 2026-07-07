import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from energy_forecasting.evaluation import (
    backtest_model,
    backtest_persistence,
    regression_metrics,
    rolling_origin_splits,
)


def test_splits_respect_purge_gap() -> None:
    for train, test in rolling_origin_splits(500, n_splits=3, test_size=48, gap=24):
        assert test[0] - train[-1] - 1 == 24
        assert np.all(np.diff(train) == 1)
        assert np.all(np.diff(test) == 1)


def test_backtest_accepts_pandas_and_builds_each_fold_fresh() -> None:
    X = pd.DataFrame({"x": np.arange(200, dtype=float)})
    y = pd.Series(2 * X["x"] + 1)
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return LinearRegression()

    metrics = backtest_model(factory, X, y, n_splits=3, test_size=24, gap=1)
    assert calls == 3
    assert len(metrics) == 3
    assert metrics["rmse"].max() < 1e-9


def test_metrics_reject_accidental_broadcasting() -> None:
    try:
        regression_metrics(np.ones(3), np.ones((3, 2)))
    except ValueError as error:
        assert "Shape mismatch" in str(error)
    else:
        raise AssertionError("mismatched prediction shapes should fail")


def test_persistence_uses_configured_lag_values() -> None:
    actual = pd.Series(np.arange(200, dtype=float))
    lagged = actual - 24
    metrics = backtest_persistence(actual, lagged, n_splits=2, test_size=24, gap=24)
    assert np.allclose(metrics["mae"], 24.0)
