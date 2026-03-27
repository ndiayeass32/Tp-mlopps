# Image officielle Python
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt || true
RUN pip install --no-cache-dir uvicorn fastapi joblib scikit-learn pandas numpy mlflow

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .

RUN mkdir -p /mlflow

EXPOSE 8000
EXPOSE 5000

CMD ["sh", "-c", "uvicorn src.mlops_tp.api:app --host 0.0.0.0 --port ${PORT:-8000}"]