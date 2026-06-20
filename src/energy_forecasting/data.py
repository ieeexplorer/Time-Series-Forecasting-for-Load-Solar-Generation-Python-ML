"""Data generation and, eventually, real-data loading functions."""

from __future__ import annotations

import numpy as np
import pandas as pd


RANDOM_SEED = 42


def generate_synthetic_data(years: int = 2, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate hourly microgrid data with daily, weekly, and annual structure.

    Synthetic observations are useful for exercising the software pipeline, but
    their forecast scores are not evidence of operational performance.
    """
    if years < 1:
        raise ValueError("years must be at least 1")

    rng = np.random.default_rng(seed)
    periods = 24 * 365 * years
    timestamp = pd.date_range("2022-01-01", periods=periods, freq="h")
    hour = timestamp.hour.to_numpy()
    day_of_year = timestamp.dayofyear.to_numpy()
    weekend = (timestamp.dayofweek >= 5).astype(float)

    # Northern-hemisphere temperature is warmest around day 200 and around 15:00.
    annual_temperature = 10 * np.cos(2 * np.pi * (day_of_year - 200) / 365.25)
    daily_temperature = 2.5 * np.cos(2 * np.pi * (hour - 15) / 24)
    temperature = 12 + annual_temperature + daily_temperature + rng.normal(0, 1.8, periods)

    # Gaussian morning/evening peaks and temperature-dependent heating demand.
    morning_peak = 10 * np.exp(-0.5 * ((hour - 8) / 2.2) ** 2)
    evening_peak = 16 * np.exp(-0.5 * ((hour - 19) / 2.8) ** 2)
    heating = 1.1 * np.maximum(15 - temperature, 0)
    load = 35 + morning_peak + evening_peak + heating - 3 * weekend
    load += rng.normal(0, 2.2, periods)

    # Seasonal day length and a half-sine clear-sky production profile.
    daylight_hours = 12 + 4 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    sunrise = 12 - daylight_hours / 2
    solar_angle = np.pi * (hour - sunrise) / daylight_hours
    clear_sky = np.where((solar_angle > 0) & (solar_angle < np.pi), np.sin(solar_angle), 0)

    # Autoregressive clouds are smoother and more plausible than independent noise.
    cloud = np.empty(periods)
    cloud[0] = rng.uniform(0.45, 1.0)
    for index in range(1, periods):
        cloud[index] = 0.88 * cloud[index - 1] + 0.12 * rng.uniform(0.25, 1.0)
    solar = 45 * clear_sky * cloud + rng.normal(0, 0.8, periods) * (clear_sky > 0)

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "load_kw": np.maximum(load, 0),
            "solar_kw": np.maximum(solar, 0),
            "temperature_c": temperature,
        }
    )
