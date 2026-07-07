from datetime import timedelta

import numpy as np
import pandas as pd

from energy_forecasting.features import build_features, feature_columns


def test_day_ahead_lag_uses_only_historical_target() -> None:
    timestamps = pd.date_range("2024-01-01", periods=600, freq="h")
    raw = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_kw": np.arange(600, dtype=float),
            "solar_kw": np.arange(600, dtype=float) * 2,
            "temperature_c": np.full(600, 10.0),
        }
    )
    featured = build_features(raw, horizon=24)
    row = featured.iloc[0]
    source_timestamp = pd.Timestamp(row["timestamp"]) - timedelta(hours=24)
    source = raw.loc[raw["timestamp"] == source_timestamp].iloc[0]
    assert row["load_kw_lag_24"] == source["load_kw"]
    assert row["solar_kw_lag_24"] == source["solar_kw"]


def test_rolling_features_end_at_forecast_issue_time() -> None:
    timestamps = pd.date_range("2024-01-01", periods=600, freq="h")
    raw = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_kw": np.arange(600, dtype=float),
            "solar_kw": np.arange(600, dtype=float),
            "temperature_c": np.full(600, 10.0),
        }
    )
    featured = build_features(raw, horizon=24, extended_history=True)
    row = featured.iloc[0]
    raw_index = raw.index[raw["timestamp"] == row["timestamp"]][0]
    expected_window = raw.loc[raw_index - 47 : raw_index - 24, "load_kw"]

    assert row["load_kw_rolling_mean_24_at_issue"] == expected_window.mean()
    assert "load_kw_lag_336" in feature_columns(horizon=24, extended_history=True)
    assert "load_kw_lag_336" not in feature_columns(horizon=24)


def test_duplicate_timestamps_are_rejected() -> None:
    timestamps = pd.date_range("2024-01-01", periods=200, freq="h")
    raw = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_kw": 1.0,
            "solar_kw": 1.0,
            "temperature_c": 1.0,
        }
    )
    raw.loc[1, "timestamp"] = raw.loc[0, "timestamp"]
    try:
        build_features(raw)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate timestamps should fail")
