import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tempfile
import os
import argparse

#   Парсинг аргументов командной строки
def parse_args():
    parser = argparse.ArgumentParser(description="Training models on the Iris dataset")

    parser.add_argument("--model", type=str, default="randfor", choices=["randfor","logreg","knn"], help="Data training model[randfor|logreg|knn]")
    parser.add_argument("--scaler", type=str, default="standard",  choices=["standard", "minmax", "none"], help="Data normalization[standard|minmax|none]")
    parser.add_argument("--features", type=str, default="base",  choices=["base", "extended"], help="Features[base|extended]")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    return parser.parse_args()

def get_scaler(scaler_type):
    """Возвращает scaler по типу"""
    if scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "minmax":
        return MinMaxScaler()
    elif scaler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

def engineer_features(X, feature_names, feature_set):
    """Создание дополнительных признаков"""
    if feature_set == "base":
        return X, feature_names
    
    # Extended features
    X_extended = np.copy(X)
    extended_names = list(feature_names)
    
    # Добавляем квадраты признаков
    for i in range(X.shape[1]):
        X_extended = np.column_stack([X_extended, X[:, i]**2])
        extended_names.append(f"{feature_names[i]}^2")
    
    # Добавляем произведения признаков
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            X_extended = np.column_stack([X_extended, X[:, i] * X[:, j]])
            extended_names.append(f"{feature_names[i]}*{feature_names[j]}")
    
    return X_extended, extended_names

def get_model(model_type, seed):
    if model_type == "randfor":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=seed,
            n_jobs=-1
        ), {
            'model_type': 'RandomForest',
            'n_estimators': 100,
            'max_depth': 5
        }
    elif model_type == "logreg":
        return LogisticRegression(
            random_state=seed,
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs'
        ), {
            'model_type': 'LogisticRegression',
            'max_iter': 1000,
            'multi_class': 'multinomial',
            'solver': 'lbfgs'
        }
    elif model_type == "knn":
        return KNeighborsClassifier(
            n_neighbors=5
        ), {
            'model_type': 'KNeighbors',
            'n_neighbors': 5
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_model(args):
    """Обучение модели с заданными параметрами"""
    
    # Загрузка данных
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Инженерия признаков
    X, feature_names = engineer_features(X, feature_names, args.features)
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    
    # Масштабирование признаков
    scaler = get_scaler(args.scaler)
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model, model_params = get_model(args.model, args.seed)
    
    experiment_name = f"Iris-{args.model}-{args.scaler}-{args.features}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params({
            'model': args.model,
            'scaler': args.scaler,
            'features': args.features,
            'seed': args.seed,
            'test_size': 0.2
        })
        mlflow.log_params(model_params)
        
        mlflow.log_params({
            'dataset': 'Iris',
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'feature_set': args.features
        })

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        mlflow.log_metrics({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

        create_and_log_plots(y_test, y_pred, target_names)
        create_and_log_classification_report(y_test, y_pred, target_names)
        
        if hasattr(model, 'feature_importances_'):
            create_and_log_feature_importance(model, feature_names)
        
        signature = infer_signature(X_train, model.predict(X_train))

        model_name = f"Iris_{args.model}_{args.scaler}_{args.features}"
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_name,
            input_example=X_train[:5]
        )
        
        if scaler:
            mlflow.sklearn.log_model(
                sk_model=scaler,
                artifact_path="scaler"
            )
        
        print(f"Модель успешно обучена! Accuracy: {accuracy:.4f}")

def create_and_log_plots(y_test, y_pred, target_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=target_names, 
           yticklabels=target_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    mlflow.log_figure(fig, "plots/confusion_matrix.png")
    plt.close(fig)

def create_and_log_classification_report(y_test, y_pred, target_names):
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=target_names)
    mlflow.log_text(report_text, "reports/classification_report.txt")
    
    for class_name in target_names:
        if class_name in report:
            mlflow.log_metric(f"precision_{class_name}", report[class_name]['precision'])
            mlflow.log_metric(f"recall_{class_name}", report[class_name]['recall'])
            mlflow.log_metric(f"f1_{class_name}", report[class_name]['f1-score'])

def create_and_log_feature_importance(model, feature_names):
    fig, ax = plt.subplots(figsize=(12, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    ax.set_title("Feature Importance")
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    
    plt.tight_layout()
    mlflow.log_figure(fig, "plots/feature_importance.png")
    plt.close(fig)
    
    importance_dict = dict(zip(feature_names, importances))
    mlflow.log_dict(importance_dict, "artifacts/feature_importance.json")

if __name__ == "__main__":
    args = parse_args()
    train_model(args)