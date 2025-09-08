import mlflow
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

def test_serving():
    """Тестирование serving модели"""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    # Загрузка последней лучшей модели
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("iris-classification")
    runs = client.search_runs(experiment.experiment_id, order_by=["metrics.accuracy DESC"])
    
    if not runs:
        print("No runs found")
        return
    
    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    
    print(f"Testing model from run: {best_run.info.run_id}")
    print(f"Model: {best_run.data.params.get('model')}")
    print(f"Accuracy: {best_run.data.metrics.get('accuracy'):.4f}")
    
    # Загрузка модели
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Тестовые предсказания
    test_data = X.iloc[:5]
    predictions = model.predict(test_data)
    
    print("\nTest predictions:")
    for i, (features, pred) in enumerate(zip(test_data.values, predictions)):
        print(f"Sample {i+1}: {features} -> {pred}")

if __name__ == "__main__":
    test_serving()