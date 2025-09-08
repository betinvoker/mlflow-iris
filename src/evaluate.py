import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_experiments():
    """Сводная оценка всех экспериментов"""
    client = MlflowClient()
    experiment = client.get_experiment_by_name("iris-classification")
    
    if experiment is None:
        print("Experiment not found")
        return
    
    runs = client.search_runs(experiment.experiment_id)
    
    results = []
    for run in runs:
        results.append({
            'run_id': run.info.run_id,
            'model': run.data.params.get('model', ''),
            'features': run.data.params.get('features', ''),
            'scaler': run.data.params.get('scaler', ''),
            'accuracy': run.data.metrics.get('accuracy', 0),
            'f1_macro': run.data.metrics.get('f1_macro', 0),
            'cv_accuracy_mean': run.data.metrics.get('cv_accuracy_mean', 0),
            'cv_f1_mean': run.data.metrics.get('cv_f1_mean', 0)
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('accuracy', ascending=False)
    
    print("Сводная таблица результатов:")
    print(df.to_string(index=False))
    
    # Сохранение результатов
    df.to_csv("experiment_results.csv", index=False)
    print("\nРезультаты сохранены в experiment_results.csv")
    
    return df

if __name__ == "__main__":
    evaluate_experiments()