FROM python:3.10.1-slim

WORKDIR /app

# Установка переменных окружения
ENV ARTIFACT_ROOT=/app/artifacts
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Установка пакетов
RUN pip install --upgrade pip && \
    pip install --no-cache-dir 'mlflow==3.3.2' && \
    pip install --no-cache-dir 'protobuf==3.20.3'

# Копирование кода проекта
COPY . /app

# Открытие порта
EXPOSE 5000

# Запуск MLflow server
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5001"]