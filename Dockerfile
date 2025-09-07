FROM python:3.10.1-slim

WORKDIR /app

# Установка переменных окружения
ENV ARTIFACT_ROOT=/app/artifacts
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Установка пакетов
RUN pip install --upgrade pip && \
    pip install --no-cache-dir 'mlflow==2.8.0' && \
    pip install --no-cache-dir 'protobuf==3.20.3'

# Открытие порта
EXPOSE 5000

# Запуск MLflow server с JSON форматом для CMD
CMD  ["mlflow", "server", \
    "--backend-store-uri", "sqlite:////app/mlruns.db", \
    "--default-artifact-root", "$ARTIFACT_ROOT", \
    "--host", "0.0.0.0"]