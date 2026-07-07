FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements-service.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-service.txt

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "energy_forecasting.api:app", "--host", "0.0.0.0", "--port", "8000"]
