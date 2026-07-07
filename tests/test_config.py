from pathlib import Path

import pytest

from energy_forecasting.config import load_config


def test_baseline_configuration_is_valid() -> None:
    config = load_config(Path("configs/baseline.yaml"))
    assert config["evaluation"]["gap"] >= config["data"]["forecast_horizon"]


def test_gap_shorter_than_horizon_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.yaml"
    path.write_text(
        """experiment_name: unsafe
data: {forecast_horizon: 24}
models: [{type: linear_regression}]
evaluation: {gap: 0}
output_dir: outputs/test
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="gap"):
        load_config(path)


def test_csv_source_requires_path(tmp_path: Path) -> None:
    path = tmp_path / "missing_csv_path.yaml"
    path.write_text(
        """experiment_name: csv_without_path
data: {source: csv, forecast_horizon: 24}
models: [{type: linear_regression}]
evaluation: {gap: 24, final_test_size: 24}
output_dir: outputs/test
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="data.path"):
        load_config(path)


def test_final_test_size_must_be_positive(tmp_path: Path) -> None:
    path = tmp_path / "no_holdout.yaml"
    path.write_text(
        """experiment_name: no_holdout
data: {forecast_horizon: 24}
models: [{type: linear_regression}]
evaluation: {gap: 24, final_test_size: 0}
output_dir: outputs/test
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="final_test_size"):
        load_config(path)
