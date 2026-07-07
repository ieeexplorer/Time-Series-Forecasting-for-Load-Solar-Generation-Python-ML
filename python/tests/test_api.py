from energy_forecasting.api import run_dashboard_experiment


def test_dashboard_experiment_returns_frontend_shape() -> None:
    result = run_dashboard_experiment(
        {
            "syntheticYears": 1,
            "forecastHorizon": 24,
            "models": ["ridge"],
            "nSplits": 2,
            "testSize": 48,
            "gap": 24,
            "maxTrainSize": 1000,
            "finalTestSize": 96,
        }
    )

    assert result["summary"]["selectedModels"]
    assert {target["target"] for target in result["targets"]} == {"load_kw", "solar_kw"}
    assert result["targets"][0]["predictions"]
    assert result["rawData"]
