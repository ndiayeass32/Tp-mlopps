# Image officielle Python
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN grep -v "pywin32" requirements.txt | pip install --no-cache-dir -r /dev/stdin

COPY . .

RUN mkdir -p /mlflow

EXPOSE 8000
EXPOSE 5000

CMD ["uvicorn", "src.mlops_tp.api:app", "--host", "0.0.0.0", "--port", "8000"]