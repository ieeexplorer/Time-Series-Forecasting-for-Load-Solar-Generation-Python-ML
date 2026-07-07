"""YAML loading with validation for reproducible experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject unsafe day-ahead evaluation settings."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")

    required = {"experiment_name", "data", "models", "evaluation", "output_dir"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    horizon = int(config["data"].get("forecast_horizon", 24))
    gap = int(config["evaluation"].get("gap", horizon))
    source = config["data"].get("source", "synthetic")
    if horizon < 1:
        raise ValueError("data.forecast_horizon must be positive")
    if source not in {"synthetic", "csv"}:
        raise ValueError("data.source must be either 'synthetic' or 'csv'")
    if source == "csv" and not config["data"].get("path"):
        raise ValueError("data.path is required when data.source is 'csv'")
    if gap < horizon:
        raise ValueError(
            "evaluation.gap must be at least the forecast horizon to purge labels "
            "that would not be available at the first test forecast origin"
        )
    if int(config["evaluation"].get("final_test_size", 0)) < 1:
        raise ValueError("evaluation.final_test_size must be positive")
    if not config["models"]:
        raise ValueError("at least one model must be configured")
    return config
