# Dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir setuptools==69.5.1 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --upgrade mlflow

COPY . .

RUN mkdir -p data/raw data/processed data/validated models mlruns

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]