"""Model factories for reproducible, fold-independent training."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def model_factory(name: str, parameters: dict[str, Any] | None = None) -> Callable[[], object]:
    """Return a callable that constructs a new unfitted estimator."""
    parameters = dict(parameters or {})
    if name == "linear_regression":
        if parameters:
            raise ValueError("linear_regression does not accept configured parameters")
        return lambda: make_pipeline(StandardScaler(), LinearRegression())
    if name == "ridge_regression":
        defaults = {"alpha": 1.0}
        defaults.update(parameters)
        return lambda: make_pipeline(StandardScaler(), Ridge(**defaults))
    if name == "random_forest":
        defaults = {
            "n_estimators": 150,
            "min_samples_leaf": 2,
            "max_features": 0.8,
            "random_state": 42,
            "n_jobs": 1,
        }
        defaults.update(parameters)
        return lambda: RandomForestRegressor(**defaults)
    raise ValueError(f"Unknown model type: {name}")
