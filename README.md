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

## Примеры использования:
---
#### Обучение с кастомными параметрами
_python src/train.py --model randomforest --scaler standard --features base --seed 45_

## Скрины эксперимента в MLflow:
---
![Alt Docker](images/docker.png)

![Alt Experiments](images/experiments.png)

![Alt Experiment](images/experiment.png)

![Alt Artifacts](images/artifacts-experiment.png)

![Alt Overview](images/overview-experiment.png)

![Alt Models](images/models.png)

![Alt Model](images/model.png)