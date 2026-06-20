import numpy as np
import pandas as pd

from energy_forecasting.features import build_features


def test_day_ahead_lag_uses_only_historical_target() -> None:
    timestamps = pd.date_range("2024-01-01", periods=240, freq="h")
    raw = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_kw": np.arange(240, dtype=float),
            "solar_kw": np.arange(240, dtype=float) * 2,
            "temperature_c": np.full(240, 10.0),
        }
    )
    featured = build_features(raw, horizon=24)
    row = featured.iloc[0]
    source = raw.loc[raw["timestamp"] == row["timestamp"] - pd.Timedelta(hours=24)].iloc[0]
    assert row["load_kw_lag_24"] == source["load_kw"]
    assert row["solar_kw_lag_24"] == source["solar_kw"]


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
