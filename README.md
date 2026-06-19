# Day-Ahead Load and Solar Forecasting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reproducible machine-learning baseline for **24-hour-ahead microgrid load and
solar-generation forecasting**. The repository currently uses synthetic hourly
data, making it suitable for learning and pipeline development—not yet for
operational grid decisions.

## Why this baseline is useful

- Generates load, solar, and temperature series with daily, weekly, seasonal,
  weather, and cloud-cover effects.
- Defines the forecast horizon explicitly and avoids using measurements that
  would be unavailable when a day-ahead forecast is issued.
- Uses chronological train/validation/test periods rather than random splitting.
- Compares linear regression and random forest against a 24-hour persistence
  baseline.
- Forecasts both load and solar generation and reports MAE, RMSE, and R².
- Saves the generated data, predictions, metrics, metadata, and a forecast plot.

## Quick start

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python energy_forecasting.py
```

Outputs are written to `outputs/`:

- `metrics.csv` — validation and held-out test scores
- `test_predictions.csv` — timestamped actual and predicted values
- `forecast_preview.png` — seven-day visual comparison
- `run_metadata.json` — horizon, features, split sizes, and selected models
- `synthetic_energy_data.csv` — generated input data

Use `python energy_forecasting.py --years 3 --output-dir results` to change the
synthetic period or output location.

## Forecasting design

For a target timestamp `t`, the model predicts load and solar **24 hours in
advance**. It may use:

- calendar fields for `t`;
- a temperature forecast for `t`; and
- load/solar observations from `t-24h` and `t-168h`.

It deliberately does not use a `t-1h` lag, because that value is unavailable when
all 24 hours of the next day are forecast together. With real data,
`temperature_c` must likewise be a weather forecast available at issue time.

## Moving to real data

Replace `generate_synthetic_data()` with a loader that produces these columns:

| Column | Meaning |
|---|---|
| `timestamp` | Unique, regular hourly target timestamp |
| `load_kw` | Mean load during the hour |
| `solar_kw` | Mean PV generation during the hour |
| `temperature_c` | Temperature forecast available 24 hours earlier |

Before modelling, add explicit handling for time zones, daylight-saving changes,
missing intervals, outliers, unit conversions, and forecast-issue timestamps.
Candidate UK sources include NESO/Elexon for power-system data and the Met Office
for weather data; their licences, schemas, and geographic resolution should be
checked before integration.

## Limitations and next steps

Synthetic performance is not evidence of real-world accuracy. Sensible next steps
are rolling-origin backtesting, probabilistic prediction intervals, gradient
boosting, hyperparameter tuning confined to validation folds, and comparison with
operational weather forecasts. LSTM or transformer models should be added only
after strong statistical and tree-based baselines are established.

## License

MIT. See [LICENSE](LICENSE).
