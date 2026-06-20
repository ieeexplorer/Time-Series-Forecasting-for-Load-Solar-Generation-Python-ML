"""Feature engineering with explicit forecast-availability assumptions."""

from __future__ import annotations

import numpy as np
import pandas as pd


TARGETS = ("load_kw", "solar_kw")
DEFAULT_HORIZON = 24


def build_features(data: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """Create target-time features for a direct day-ahead forecast.

    Calendar values and ``temperature_c`` are treated as future-known covariates.
    On real data, temperature must therefore be a forecast issued no later than
    ``horizon`` hours before the target timestamp. Target lags are restricted to
    information available at that same issue time.
    """
    required = {"timestamp", "load_kw", "solar_kw", "temperature_c"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if horizon < 1:
        raise ValueError("horizon must be positive")

    frame = data.sort_values("timestamp").copy()
    if frame["timestamp"].duplicated().any():
        raise ValueError("timestamp values must be unique")

    ts = pd.to_datetime(frame["timestamp"])
    # Sine/cosine pairs preserve the circular distance between 23:00 and 00:00.
    frame["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    frame["year_sin"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365.25)
    frame["year_cos"] = np.cos(2 * np.pi * ts.dt.dayofyear / 365.25)
    frame["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Net-load history exposes the duck-curve relationship without using the
    # unknown contemporaneous net load, which would leak both target values.
    frame["net_load_kw"] = frame["load_kw"] - frame["solar_kw"]
    for target in (*TARGETS, "net_load_kw"):
        frame[f"{target}_lag_{horizon}"] = frame[target].shift(horizon)
        frame[f"{target}_lag_168"] = frame[target].shift(168)

    return frame.dropna().reset_index(drop=True)


def feature_columns(horizon: int = DEFAULT_HORIZON) -> list[str]:
    """Return the canonical feature set for the tabular baseline."""
    return [
        "temperature_c",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "year_sin",
        "year_cos",
        "is_weekend",
        f"load_kw_lag_{horizon}",
        "load_kw_lag_168",
        f"solar_kw_lag_{horizon}",
        "solar_kw_lag_168",
        f"net_load_kw_lag_{horizon}",
        "net_load_kw_lag_168",
    ]
