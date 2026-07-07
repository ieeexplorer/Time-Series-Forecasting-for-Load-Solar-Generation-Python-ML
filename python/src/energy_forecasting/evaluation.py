"""Purged rolling-origin evaluation utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    mae: float
    rmse: float
    r2: float


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate metrics after normalizing both inputs to one dimension."""
    actual_1d = np.asarray(actual, dtype=float).reshape(-1)
    predicted_1d = np.asarray(predicted, dtype=float).reshape(-1)
    if actual_1d.shape != predicted_1d.shape:
        raise ValueError(f"Shape mismatch: actual={actual_1d.shape}, predicted={predicted_1d.shape}")
    return {
        "mae": float(mean_absolute_error(actual_1d, predicted_1d)),
        "rmse": float(np.sqrt(mean_squared_error(actual_1d, predicted_1d))),
        "r2": float(r2_score(actual_1d, predicted_1d)),
    }


def rolling_origin_splits(
    n_samples: int,
    *,
    n_splits: int,
    test_size: int,
    gap: int,
    max_train_size: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield chronological folds with a purge gap before every test window."""
    if gap < 0:
        raise ValueError("gap cannot be negative")
    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=test_size,
        gap=gap,
        max_train_size=max_train_size,
    )
    yield from splitter.split(np.arange(n_samples))


def backtest_model(
    model_factory: Callable[[], object],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    test_size: int,
    gap: int,
    max_train_size: int | None = None,
) -> pd.DataFrame:
    """Evaluate a fresh estimator on every rolling-origin fold.

    A factory is required instead of a model instance so neural networks cannot
    accidentally carry fitted weights from one fold into the next. Scalers should
    live inside the estimator pipeline and are consequently refitted per fold.
    """
    rows: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(
        rolling_origin_splits(
            len(X),
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            max_train_size=max_train_size,
        ),
        start=1,
    ):
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predicted = np.maximum(np.asarray(model.predict(X.iloc[test_idx])).reshape(-1), 0)
        rows.append(
            {
                "fold": fold,
                "train_start": int(train_idx[0]),
                "train_end": int(train_idx[-1]),
                "test_start": int(test_idx[0]),
                "test_end": int(test_idx[-1]),
                **regression_metrics(y.iloc[test_idx].to_numpy(), predicted),
            }
        )
    return pd.DataFrame(rows)


def backtest_persistence(
    actual: pd.Series,
    lagged: pd.Series,
    *,
    n_splits: int,
    test_size: int,
    gap: int,
    max_train_size: int | None = None,
) -> pd.DataFrame:
    """Evaluate the same-hour historical value as a no-training benchmark."""
    rows: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(
        rolling_origin_splits(
            len(actual),
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            max_train_size=max_train_size,
        ),
        start=1,
    ):
        rows.append(
            {
                "fold": fold,
                "train_start": int(train_idx[0]),
                "train_end": int(train_idx[-1]),
                "test_start": int(test_idx[0]),
                "test_end": int(test_idx[-1]),
                **regression_metrics(actual.iloc[test_idx], lagged.iloc[test_idx]),
            }
        )
    return pd.DataFrame(rows)
