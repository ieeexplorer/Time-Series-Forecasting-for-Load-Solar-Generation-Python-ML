"""Feature engineering with explicit forecast-availability assumptions."""

from __future__ import annotations

import numpy as np
import pandas as pd


TARGETS = ("load_kw", "solar_kw")
DEFAULT_HORIZON = 24
EXTRA_LAGS = (48, 168, 336)


def build_features(
    data: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    *,
    extended_history: bool = False,
) -> pd.DataFrame:
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
    lag_hours = sorted({horizon, *EXTRA_LAGS}) if extended_history else [horizon, 168]
    for target in (*TARGETS, "net_load_kw"):
        for lag in lag_hours:
            frame[f"{target}_lag_{lag}"] = frame[target].shift(lag)

    if extended_history:
        # Rolling summaries are also shifted to the forecast issue time. For a
        # 24-hour horizon, the latest value in each window is t-24, never t-1.
        for target in TARGETS:
            issued_history = frame[target].shift(horizon)
            for window in (24, 168):
                frame[f"{target}_rolling_mean_{window}_at_issue"] = issued_history.rolling(window).mean()
                frame[f"{target}_rolling_std_{window}_at_issue"] = issued_history.rolling(window).std(ddof=0)
        frame["net_load_kw_rolling_mean_168_at_issue"] = frame["net_load_kw"].shift(horizon).rolling(168).mean()

    return frame.dropna().reset_index(drop=True)


def feature_columns(horizon: int = DEFAULT_HORIZON, *, extended_history: bool = False) -> list[str]:
    """Return the canonical feature set for the tabular baseline."""
    lag_hours = sorted({horizon, *EXTRA_LAGS}) if extended_history else [horizon, 168]
    lag_columns = [
        f"{target}_lag_{lag}"
        for target in (*TARGETS, "net_load_kw")
        for lag in lag_hours
    ]
    rolling_columns = [
        "load_kw_rolling_mean_24_at_issue",
        "load_kw_rolling_std_24_at_issue",
        "load_kw_rolling_mean_168_at_issue",
        "load_kw_rolling_std_168_at_issue",
        "solar_kw_rolling_mean_24_at_issue",
        "solar_kw_rolling_std_24_at_issue",
        "solar_kw_rolling_mean_168_at_issue",
        "solar_kw_rolling_std_168_at_issue",
        "net_load_kw_rolling_mean_168_at_issue",
    ] if extended_history else []
    return [
        "temperature_c",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "year_sin",
        "year_cos",
        "is_weekend",
        *lag_columns,
        *rolling_columns,
    ]
