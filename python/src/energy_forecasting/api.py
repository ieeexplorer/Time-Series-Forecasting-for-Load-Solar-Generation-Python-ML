"""FastAPI service for the hybrid dashboard architecture."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from .data import generate_synthetic_data
from .evaluation import backtest_model, backtest_persistence, regression_metrics
from .features import TARGETS, build_features, feature_columns
from .models import model_factory


app = FastAPI(title="Energy Forecasting API", version="0.3.0")


MODEL_ALIASES: dict[str, tuple[str, dict[str, Any]]] = {
    "ridge": ("ridge_regression", {"alpha": 1.0}),
    "ridge_strong": ("ridge_regression", {"alpha": 10.0}),
    "ridge_weak": ("ridge_regression", {"alpha": 0.1}),
    "knn": ("knn", {"n_neighbors": 7, "weights": "distance"}),
    "random_forest": ("random_forest", {}),
    "xgboost": ("xgboost", {}),
    "lightgbm": ("lightgbm", {}),
}


def _metric_payload(actual: pd.Series | np.ndarray, predicted: np.ndarray, persistence_rmse: float | None = None) -> dict:
    metrics = regression_metrics(np.asarray(actual), predicted)
    actual_array = np.asarray(actual, dtype=float).reshape(-1)
    predicted_array = np.asarray(predicted, dtype=float).reshape(-1)
    mask = np.abs(actual_array) > 0.5
    mape = float(np.mean(np.abs((actual_array[mask] - predicted_array[mask]) / actual_array[mask])) * 100) if mask.any() else 0.0
    skill = 0.0 if not persistence_rmse else float(1 - metrics["rmse"] / max(persistence_rmse, 1e-9))
    return {**metrics, "mape": mape, "skillScore": skill}


def _timestamp_ms(timestamp: Any) -> int:
    return int(pd.Timestamp(timestamp).timestamp() * 1000)


def _model_config(model_id: str) -> tuple[str, dict[str, Any]]:
    try:
        return MODEL_ALIASES[model_id]
    except KeyError as error:
        raise ValueError(f"Unknown model option: {model_id}") from error


def run_dashboard_experiment(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the scikit-learn pipeline and return dashboard-compatible JSON."""
    start = perf_counter()
    synthetic_years = int(payload.get("syntheticYears", 2))
    seed = int(payload.get("seed", 42))
    horizon = int(payload.get("forecastHorizon", 24))
    n_splits = int(payload.get("nSplits", 4))
    test_size = int(payload.get("testSize", 168))
    gap = int(payload.get("gap", horizon))
    max_train_size = payload.get("maxTrainSize", 8760)
    final_test_size = int(payload.get("finalTestSize", 720))
    extended_history = bool(payload.get("extendedHistoryFeatures", False))
    requested_models = payload.get("models") or ["ridge", "ridge_strong", "random_forest"]

    if gap < horizon:
        raise ValueError("gap must be at least forecastHorizon")

    raw = generate_synthetic_data(years=synthetic_years, seed=seed)
    frame = build_features(raw, horizon=horizon, extended_history=extended_history)
    columns = feature_columns(horizon, extended_history=extended_history)
    if final_test_size >= len(frame):
        raise ValueError("finalTestSize is too large for the available feature rows")

    development = frame.iloc[:-final_test_size].reset_index(drop=True)
    final_test = frame.iloc[-final_test_size:].reset_index(drop=True)
    if len(development) < n_splits * test_size + gap + 1:
        raise ValueError("not enough development rows for the requested CV settings")

    targets = []
    selected_models: dict[str, str] = {}
    feature_weights: dict[str, np.ndarray] = {}

    for target in TARGETS:
        persistence_values = final_test[f"{target}_lag_{horizon}"].to_numpy()
        persistence_metrics = _metric_payload(final_test[target], persistence_values)
        persistence_rmse = persistence_metrics["rmse"]

        persistence_cv = backtest_persistence(
            development[target],
            development[f"{target}_lag_{horizon}"],
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            max_train_size=max_train_size,
        )
        cv_results = [
            {
                "modelName": "persistence",
                "target": target,
                "folds": _fold_payloads(persistence_cv),
                "meanMetrics": _mean_payload(persistence_cv),
            }
        ]

        best_model_id = "persistence"
        best_rmse = float(persistence_cv["rmse"].mean())
        best_factory = None

        for model_id in requested_models:
            model_name, parameters = _model_config(model_id)
            factory = model_factory(model_name, parameters)
            fold_metrics = backtest_model(
                factory,
                development[columns],
                development[target],
                n_splits=n_splits,
                test_size=test_size,
                gap=gap,
                max_train_size=max_train_size,
            )
            mean_rmse = float(fold_metrics["rmse"].mean())
            cv_results.append(
                {
                    "modelName": model_id,
                    "target": target,
                    "folds": _fold_payloads(fold_metrics),
                    "meanMetrics": _mean_payload(fold_metrics),
                }
            )
            if mean_rmse < best_rmse:
                best_model_id = model_id
                best_rmse = mean_rmse
                best_factory = factory

        if best_factory is None:
            predicted = persistence_values
        else:
            final_model = best_factory()
            final_model.fit(development[columns], development[target])
            predicted = np.maximum(np.asarray(final_model.predict(final_test[columns])).reshape(-1), 0)
            if best_model_id.startswith("ridge"):
                feature_weights[target] = final_model.named_steps["ridge"].coef_

        final_metrics = _metric_payload(final_test[target], predicted, persistence_rmse)
        selected_models[target] = best_model_id
        targets.append(
            {
                "target": target,
                "selectedModel": best_model_id,
                "finalMetrics": final_metrics,
                "persistenceMetrics": persistence_metrics,
                "skillScore": final_metrics["skillScore"],
                "predictions": [
                    {
                        "timestamp": _timestamp_ms(row.timestamp),
                        "actual": float(getattr(row, target)),
                        "predicted": float(predicted[index]),
                        "persistence": float(persistence_values[index]),
                    }
                    for index, row in enumerate(final_test.itertuples(index=False))
                ],
                "cvResults": cv_results,
            }
        )

    raw_step = max(1, len(raw) // 2000)
    raw_data = [
        {
            "timestamp": _timestamp_ms(row.timestamp),
            "load_kw": round(float(row.load_kw), 2),
            "solar_kw": round(float(row.solar_kw), 2),
            "temperature_c": round(float(row.temperature_c), 2),
        }
        for row in raw.iloc[::raw_step].itertuples(index=False)
    ]

    feature_importance = [
        {
            "feature": column.replace("_", " "),
            "loadWeight": float(feature_weights.get("load_kw", np.zeros(len(columns)))[index]),
            "solarWeight": float(feature_weights.get("solar_kw", np.zeros(len(columns)))[index]),
        }
        for index, column in enumerate(columns)
    ]

    return {
        "config": {
            "experimentName": payload.get("experimentName", "hybrid_baseline"),
            "syntheticYears": synthetic_years,
            "seed": seed,
            "forecastHorizon": horizon,
            "models": requested_models,
            "nSplits": n_splits,
            "testSize": test_size,
            "gap": gap,
            "maxTrainSize": max_train_size,
            "finalTestSize": final_test_size,
            "extendedHistoryFeatures": extended_history,
        },
        "targets": targets,
        "rawData": raw_data,
        "featureImportance": feature_importance,
        "summary": {
            "experimentName": payload.get("experimentName", "hybrid_baseline"),
            "forecastHorizon": horizon,
            "totalSamples": len(raw),
            "trainingSamples": len(development),
            "testSamples": final_test_size,
            "selectedModels": selected_models,
            "totalRuntimeMs": round((perf_counter() - start) * 1000),
        },
    }


def _fold_payloads(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "fold": int(row.fold),
            "trainStart": int(row.train_start),
            "trainEnd": int(row.train_end),
            "testStart": int(row.test_start),
            "testEnd": int(row.test_end),
            "mae": float(row.mae),
            "rmse": float(row.rmse),
            "r2": float(row.r2),
            "mape": 0.0,
            "skillScore": 0.0,
        }
        for row in metrics.itertuples(index=False)
    ]


def _mean_payload(metrics: pd.DataFrame) -> dict[str, float]:
    return {
        "mae": float(metrics["mae"].mean()),
        "rmse": float(metrics["rmse"].mean()),
        "r2": float(metrics["r2"].mean()),
        "mape": 0.0,
        "skillScore": 0.0,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/forecast")
def forecast(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_dashboard_experiment(payload)
    except Exception as error:  # pragma: no cover - FastAPI error translation
        raise HTTPException(status_code=400, detail=str(error)) from error
