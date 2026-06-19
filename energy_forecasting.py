"""Reproducible day-ahead load and solar forecasting baseline.

The synthetic data generator is deliberately separated from the modelling code so
that a real, hourly data set can later replace it without changing the pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
FORECAST_HORIZON = 24
TARGETS = ("load_kw", "solar_kw")


def generate_synthetic_data(years: int = 2, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate hourly microgrid data with daily, weekly, and annual structure."""
    if years < 1:
        raise ValueError("years must be at least 1")

    rng = np.random.default_rng(seed)
    periods = 24 * 365 * years
    timestamp = pd.date_range("2022-01-01", periods=periods, freq="h")
    hour = timestamp.hour.to_numpy()
    day_of_year = timestamp.dayofyear.to_numpy()
    weekend = (timestamp.dayofweek >= 5).astype(float)

    # Northern-hemisphere temperature: warmest around day 200.
    annual_temperature = 10 * np.cos(2 * np.pi * (day_of_year - 200) / 365.25)
    daily_temperature = 2.5 * np.cos(2 * np.pi * (hour - 15) / 24)
    temperature = 12 + annual_temperature + daily_temperature + rng.normal(0, 1.8, periods)

    # Load contains morning/evening peaks, a weekend effect, and heating demand.
    morning_peak = 10 * np.exp(-0.5 * ((hour - 8) / 2.2) ** 2)
    evening_peak = 16 * np.exp(-0.5 * ((hour - 19) / 2.8) ** 2)
    heating = 1.1 * np.maximum(15 - temperature, 0)
    load = 35 + morning_peak + evening_peak + heating - 3 * weekend
    load += rng.normal(0, 2.2, periods)

    # Solar is exactly zero outside a seasonally varying daylight window.
    daylight_hours = 12 + 4 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    sunrise = 12 - daylight_hours / 2
    solar_angle = np.pi * (hour - sunrise) / daylight_hours
    clear_sky = np.where((solar_angle > 0) & (solar_angle < np.pi), np.sin(solar_angle), 0)

    # Persistent cloud cover is more realistic than independent hourly noise.
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


def build_features(data: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Build leakage-safe features indexed by the hour being predicted.

    At a 24-hour horizon, lag_24 is the newest observed target value. Calendar
    fields and temperature are treated as known future covariates; with real data,
    temperature must therefore come from a weather forecast, not observations.
    """
    required = {"timestamp", "load_kw", "solar_kw", "temperature_c"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = data.sort_values("timestamp").copy()
    ts = frame["timestamp"]
    frame["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    frame["year_sin"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365.25)
    frame["year_cos"] = np.cos(2 * np.pi * ts.dt.dayofyear / 365.25)
    frame["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    for target in TARGETS:
        frame[f"{target}_lag_{horizon}"] = frame[target].shift(horizon)
        frame[f"{target}_lag_168"] = frame[target].shift(168)

    return frame.dropna().reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split without shuffling so evaluation always occurs in the future."""
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test period")
    train_end = int(len(frame) * train_fraction)
    validation_end = int(len(frame) * (train_fraction + validation_fraction))
    return frame.iloc[:train_end], frame.iloc[train_end:validation_end], frame.iloc[validation_end:]


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def train_and_evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Select a model on validation data, refit it, then evaluate once on test."""
    feature_columns = [
        "temperature_c",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "year_sin",
        "year_cos",
        "is_weekend",
        "load_kw_lag_24",
        "load_kw_lag_168",
        "solar_kw_lag_24",
        "solar_kw_lag_168",
    ]
    train, validation, test = chronological_split(frame)
    model_candidates = {
        "linear_regression": make_pipeline(StandardScaler(), LinearRegression()),
        "random_forest": RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_SEED,
            # A single worker is portable in restricted Windows environments.
            n_jobs=1,
        ),
    }

    metric_rows: list[dict] = []
    test_predictions = test[["timestamp", *TARGETS]].copy()
    fitted_models: dict = {}

    for target in TARGETS:
        # The 24-hour persistence forecast is an essential time-series baseline.
        persistence = validation[f"{target}_lag_{FORECAST_HORIZON}"].to_numpy()
        metric_rows.append(
            {"target": target, "split": "validation", "model": "persistence", **regression_metrics(validation[target], persistence)}
        )

        validation_scores: dict[str, float] = {}
        for name, candidate in model_candidates.items():
            model = clone(candidate)
            model.fit(train[feature_columns], train[target])
            prediction = model.predict(validation[feature_columns])
            scores = regression_metrics(validation[target], prediction)
            validation_scores[name] = scores["rmse"]
            metric_rows.append({"target": target, "split": "validation", "model": name, **scores})

        best_name = min(validation_scores, key=validation_scores.get)
        development = pd.concat([train, validation], ignore_index=True)
        best_model = clone(model_candidates[best_name])
        best_model.fit(development[feature_columns], development[target])
        fitted_models[target] = best_model

        test_prediction = np.maximum(best_model.predict(test[feature_columns]), 0)
        test_predictions[f"{target}_prediction"] = test_prediction
        metric_rows.append(
            {"target": target, "split": "test", "model": best_name, **regression_metrics(test[target], test_prediction)}
        )
        persistence_test = test[f"{target}_lag_{FORECAST_HORIZON}"].to_numpy()
        metric_rows.append(
            {"target": target, "split": "test", "model": "persistence", **regression_metrics(test[target], persistence_test)}
        )

    metrics = pd.DataFrame(metric_rows)
    metadata = {
        "forecast_horizon_hours": FORECAST_HORIZON,
        "features": feature_columns,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "selected_models": {
            target: metrics.loc[
                (metrics["target"] == target) & (metrics["split"] == "test") & (metrics["model"] != "persistence"),
                "model",
            ].iloc[0]
            for target in TARGETS
        },
    }
    return metrics, test_predictions, metadata


def save_forecast_plot(predictions: pd.DataFrame, output_path: Path, days: int = 7) -> None:
    sample = predictions.head(24 * days)
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    labels = {"load_kw": "Load (kW)", "solar_kw": "Solar generation (kW)"}
    for axis, target in zip(axes, TARGETS):
        axis.plot(sample["timestamp"], sample[target], label="Actual", color="black", linewidth=1.5)
        axis.plot(
            sample["timestamp"],
            sample[f"{target}_prediction"],
            label="24-hour-ahead forecast",
            color="#2878B5",
            linestyle="--",
        )
        axis.set_ylabel(labels[target])
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("Target timestamp")
    figure.suptitle("Day-ahead forecast: first seven days of the test period")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, default=2, help="Synthetic years to generate")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_data = generate_synthetic_data(years=args.years)
    feature_data = build_features(raw_data)
    metrics, predictions, metadata = train_and_evaluate(feature_data)

    raw_data.to_csv(args.output_dir / "synthetic_energy_data.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_forecast_plot(predictions, args.output_dir / "forecast_preview.png")

    print("\nValidation and held-out test metrics")
    print(metrics.round(3).to_string(index=False))
    print(f"\nArtifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
