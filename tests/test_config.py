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
