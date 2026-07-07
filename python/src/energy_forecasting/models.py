"""Model factories for reproducible, fold-independent training."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
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
    if name == "knn":
        defaults = {"n_neighbors": 7, "weights": "distance"}
        defaults.update(parameters)
        return lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(**defaults))
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
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("Install requirements-boosting.txt to use xgboost") from error
        defaults = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": 1,
        }
        defaults.update(parameters)
        return lambda: XGBRegressor(**defaults)
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("Install requirements-boosting.txt to use lightgbm") from error
        defaults = {
            "n_estimators": 300,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "n_jobs": 1,
            "verbose": -1,
        }
        defaults.update(parameters)
        return lambda: LGBMRegressor(**defaults)
    raise ValueError(f"Unknown model type: {name}")
