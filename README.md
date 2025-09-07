# mlflow-iris

## Установка среды
---
#### 1. Комманда установки venv
_python -m venv env_
#### 2. Запуск venv
_source env/Scripts/activate_
#### 3. Установка пакетов
_pip install -r requirements.txt_

## Установка MLflow
---
#### Сборка докера
_docker build -t mlflow-server ._
#### Запуск докера
_docker run -p 5000:5000  mlflow-server_