from pathlib import Path

import pandas as pd
import pytest

from energy_forecasting.data import load_hourly_data, validate_hourly_frame


def hourly_frame(periods: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=periods, freq="h"),
            "load_kw": [10.0] * periods,
            "solar_kw": [1.0] * periods,
            "temperature_c": [12.0] * periods,
        }
    )


def test_csv_loader_normalizes_unsorted_hourly_data(tmp_path: Path) -> None:
    path = tmp_path / "meter_data.csv"
    raw = hourly_frame(3).sample(frac=1, random_state=42)
    raw.to_csv(path, index=False)

    loaded = load_hourly_data(path)

    assert loaded["timestamp"].is_monotonic_increasing
    assert list(loaded.columns) == ["timestamp", "load_kw", "solar_kw", "temperature_c"]


def test_validation_rejects_missing_hour() -> None:
    raw = hourly_frame(4).drop(index=2)

    with pytest.raises(ValueError, match="regular hourly"):
        validate_hourly_frame(raw)


def test_validation_rejects_negative_targets() -> None:
    raw = hourly_frame(4)
    raw.loc[1, "solar_kw"] = -1.0

    with pytest.raises(ValueError, match="non-negative"):
        validate_hourly_frame(raw)
