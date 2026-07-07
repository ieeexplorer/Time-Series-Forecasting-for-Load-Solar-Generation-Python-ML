"""Configuration-driven tabular forecasting experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from .config import load_config
from .data import generate_synthetic_data, load_hourly_data
from .evaluation import backtest_model, backtest_persistence, regression_metrics
from .features import TARGETS, build_features, feature_columns
from .models import model_factory


def run(config_path: str | Path) -> dict:
    """Backtest candidates, select on mean CV RMSE, and evaluate a final holdout."""
    config = load_config(config_path)
    horizon = int(config["data"].get("forecast_horizon", 24))
    extended_history = bool(config["data"].get("extended_history_features", False))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)

    if config["data"].get("source", "synthetic") == "csv":
        raw = load_hourly_data(config["data"]["path"])
    else:
        raw = generate_synthetic_data(
            years=int(config["data"].get("synthetic_years", 2)),
            seed=int(config["data"].get("seed", 42)),
        )
    frame = build_features(raw, horizon=horizon, extended_history=extended_history)
    columns = feature_columns(horizon, extended_history=extended_history)
    final_test_size = int(config["evaluation"]["final_test_size"])
    if final_test_size >= len(frame):
        raise ValueError(
            "evaluation.final_test_size must be smaller than the number of rows "
            f"available after feature engineering ({len(frame)})"
        )
    development = frame.iloc[:-final_test_size].reset_index(drop=True)
    final_test = frame.iloc[-final_test_size:].reset_index(drop=True)
    minimum_development_rows = (
        int(config["evaluation"]["n_splits"]) * int(config["evaluation"]["test_size"])
        + int(config["evaluation"]["gap"])
        + 1
    )
    if len(development) < minimum_development_rows:
        raise ValueError(
            "not enough development rows for the configured rolling-origin split; "
            f"need at least {minimum_development_rows}, found {len(development)}"
        )

    cv_rows: list[pd.DataFrame] = []
    final_rows: list[dict] = []
    predictions = final_test[["timestamp", *TARGETS]].copy()
    selections: dict[str, str] = {}

    for target in TARGETS:
        mean_rmse: dict[str, float] = {}
        persistence_cv = backtest_persistence(
            development[target],
            development[f"{target}_lag_{horizon}"],
            n_splits=int(config["evaluation"]["n_splits"]),
            test_size=int(config["evaluation"]["test_size"]),
            gap=int(config["evaluation"]["gap"]),
            max_train_size=config["evaluation"].get("max_train_size"),
        )
        persistence_cv.insert(0, "model", "persistence")
        persistence_cv.insert(0, "target", target)
        cv_rows.append(persistence_cv)

        for model_config in config["models"]:
            name = model_config["type"]
            factory = model_factory(name, model_config.get("parameters"))
            fold_metrics = backtest_model(
                factory,
                development[columns],
                development[target],
                n_splits=int(config["evaluation"]["n_splits"]),
                test_size=int(config["evaluation"]["test_size"]),
                gap=int(config["evaluation"]["gap"]),
                max_train_size=config["evaluation"].get("max_train_size"),
            )
            fold_metrics.insert(0, "model", name)
            fold_metrics.insert(0, "target", target)
            cv_rows.append(fold_metrics)
            mean_rmse[name] = float(fold_metrics["rmse"].mean())

        best_name = min(mean_rmse, key=mean_rmse.get)
        selections[target] = best_name
        best_config = next(item for item in config["models"] if item["type"] == best_name)
        final_model = model_factory(best_name, best_config.get("parameters"))()
        final_model.fit(development[columns], development[target])
        predicted = np.maximum(final_model.predict(final_test[columns]), 0)
        predictions[f"{target}_prediction"] = predicted
        final_rows.append({"target": target, "model": best_name, **regression_metrics(final_test[target], predicted)})
        persistence = final_test[f"{target}_lag_{horizon}"].to_numpy()
        predictions[f"{target}_persistence"] = persistence
        final_rows.append(
            {
                "target": target,
                "model": "persistence",
                **regression_metrics(final_test[target], persistence),
            }
        )
        joblib.dump(final_model, output_dir / "models" / f"{target}_{best_name}.joblib")

    cv_metrics = pd.concat(cv_rows, ignore_index=True)
    final_metrics = pd.DataFrame(final_rows)
    cv_metrics.to_csv(output_dir / "cross_validation_metrics.csv", index=False)
    final_metrics.to_csv(output_dir / "final_test_metrics.csv", index=False)
    predictions.to_csv(output_dir / "final_test_predictions.csv", index=False)
    with Path(config_path).open(encoding="utf-8") as stream:
        resolved_config = yaml.safe_load(stream)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )
    summary = {
        "experiment_name": config["experiment_name"],
        "forecast_horizon_hours": horizon,
        "extended_history_features": extended_history,
        "selected_models": selections,
        "final_test_rows": len(final_test),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(final_metrics.round(3).to_string(index=False))
    print(f"Artifacts saved to {output_dir.resolve()}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
