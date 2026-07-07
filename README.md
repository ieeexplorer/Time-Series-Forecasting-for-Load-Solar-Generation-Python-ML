# Energy Forecast Lab

Hybrid day-ahead load and solar forecasting lab.

- `python/` contains the reproducible scikit-learn research baseline.
- `dashboard/` contains the Next.js visualization UI.
- The dashboard calls the Python FastAPI service, so Python remains the source
  of truth for model training, cross-validation, evaluation, and serialization.

## Run Everything

```bash
docker compose up --build
```

Then open `http://localhost:3000`.

## Run Locally Without Docker

Terminal 1:

```bash
cd python
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-service.txt -r requirements-dev.txt
pip install -e .
uvicorn energy_forecasting.api:app --host 0.0.0.0 --port 8000
```

Terminal 2:

```bash
cd dashboard
bun install
$env:PYTHON_API_URL="http://127.0.0.1:8000"
bun run dev
```

## Python Engine

The Python package remains installable and runnable on its own:

```bash
cd python
pip install -r requirements.txt -r requirements-dev.txt
python run_experiment.py --config configs/baseline.yaml
python -m pytest -q
```

Optional model packages:

```bash
pip install -r requirements-boosting.txt
```

That enables the `xgboost` and `lightgbm` model factories.
