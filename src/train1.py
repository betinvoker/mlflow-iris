import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, 
                             classification_report, roc_curve, auc, 
                             precision_recall_curve, RocCurveDisplay)
from sklearn.inspection import permutation_importance
from sklearn.multiclass import OneVsRestClassifier
import json
from datetime import datetime
import itertools

# Загрузка данных
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Инициализация MLflow
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Iris_Classification_RandomForest"
mlflow.set_experiment(experiment_name)

# Параметры эксперимента
SEED = 42
N_SPLITS = 5
SCALERS = {
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "none": None
}
FEATURE_VARIANTS = {
    "all_features": [0, 1, 2, 3],
    "only_petal": [2, 3],
    "only_sepal": [0, 1]
}

# Параметры модели
params = {
    "n_estimators": 100,
    "max_depth": 4,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": SEED
}

# Функция для создания confusion matrix
def plot_confusion_matrix(y_true, y_pred, target_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig

# Функция для создания ROC-кривой (исправленная для многоклассовой классификации)
def plot_multiclass_roc(y_true, y_pred_proba, target_names):
    n_classes = len(target_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = itertools.cycle(['blue', 'red', 'green', 'orange', 'purple'])
    
    # One-vs-Rest подход для многоклассовой ROC-кривой
    for i, color in zip(range(n_classes), colors):
        # Бинаризируем метки для текущего класса
        y_true_binary = (y_true == i).astype(int)
        y_score_binary = y_pred_proba[:, i]
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'ROC {target_names[i]} (AUC = {roc_auc:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multiclass ROC Curve (One-vs-Rest)')
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig

# Функция для создания Precision-Recall кривой
def plot_multiclass_pr_curve(y_true, y_pred_proba, target_names):
    n_classes = len(target_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = itertools.cycle(['blue', 'red', 'green', 'orange', 'purple'])
    
    for i, color in zip(range(n_classes), colors):
        y_true_binary = (y_true == i).astype(int)
        y_score_binary = y_pred_proba[:, i]
        
        precision, recall, _ = precision_recall_curve(y_true_binary, y_score_binary)
        pr_auc = auc(recall, precision)
        
        ax.plot(recall, precision, color=color, lw=2,
                label=f'PR {target_names[i]} (AUC = {pr_auc:.2f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Multiclass Precision-Recall Curve')
    ax.legend(loc="lower left")
    plt.tight_layout()
    return fig

# Функция для создания отчета
def create_report(y_true, y_pred, y_pred_proba, target_names, feature_importance, params):
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "RandomForestClassifier",
        "parameters": params,
        "metrics": {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average='macro')
        },
        "classification_report": classification_report(y_true, y_pred, 
                                                     target_names=target_names, output_dict=True),
        "feature_importance": feature_importance.tolist(),
        "notes": "Базовый запуск классификации на Iris dataset"
    }
    return report

# Включение autologging
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    log_datasets=True,
    disable=False,
    exclusive=False,
    max_tuning_runs=5,
    log_post_training_metrics=True
)

# Основной цикл экспериментов
with mlflow.start_run(run_name="main_experiment"):
    # Логируем параметры
    mlflow.log_param("dataset", "iris")
    mlflow.log_param("seed", SEED)
    mlflow.log_param("validation_scheme", "train_test_split")
    mlflow.log_param("model", "RandomForestClassifier")
    
    # Логируем гиперпараметры
    mlflow.log_params(params)
    
    # Вариант: все фичи, стандартный скейлер
    feature_variant = "all_features"
    scaler_name = "standard"
    
    mlflow.log_param("feature_variant", feature_variant)
    mlflow.log_param("scaler", scaler_name)
    
    # Выбор фич
    X_selected = X[:, FEATURE_VARIANTS[feature_variant]]
    selected_feature_names = [feature_names[i] for i in FEATURE_VARIANTS[feature_variant]]
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # Скейлинг
    scaler = SCALERS[scaler_name]
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Обучение модели
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("macro_f1", macro_f1)
    
    # Важность признаков
    feature_importance = model.feature_importances_
    
    # Создание и логирование визуализаций
    fig_cm = plot_confusion_matrix(y_test, y_pred, target_names)
    mlflow.log_figure(fig_cm, "confusion_matrix.png")
    
    fig_roc = plot_multiclass_roc(y_test, y_pred_proba, target_names)
    mlflow.log_figure(fig_roc, "roc_curve.png")
    
    fig_pr = plot_multiclass_pr_curve(y_test, y_pred_proba, target_names)
    mlflow.log_figure(fig_pr, "pr_curve.png")
    
    # Permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, 
                                           random_state=SEED)
    fig_perm, ax = plt.subplots(figsize=(10, 6))
    sorted_idx = perm_importance.importances_mean.argsort()
    ax.boxplot(perm_importance.importances[sorted_idx].T,
              labels=np.array(selected_feature_names)[sorted_idx])
    ax.set_title("Permutation Importance")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    mlflow.log_figure(fig_perm, "permutation_importance.png")
    
    # Создание и логирование отчета
    report = create_report(y_test, y_pred, y_pred_proba, target_names, 
                         feature_importance, params)
    mlflow.log_dict(report, "report.json")
    
    # Текстовый отчет
    report_md = f"""
# Отчет по запуску классификации Iris

## Параметры модели
- Модель: RandomForestClassifier
- Гиперпараметры: {json.dumps(params, indent=2)}
- Скейлер: {scaler_name}
- Вариант фич: {feature_variant}
- Seed: {SEED}

## Метрики
- Accuracy: {accuracy:.4f}
- Macro F1: {macro_f1:.4f}

## Важность признаков
{json.dumps(dict(zip(selected_feature_names, feature_importance.tolist())), indent=2)}

## Выводы
Модель показывает хорошее качество классификации на тестовых данных.
Наиболее важными признаками являются: {selected_feature_names[np.argmax(feature_importance)]}.
"""
    
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    mlflow.log_artifact("report.md")
    
    # Закрываем figures чтобы избежать утечек памяти
    plt.close('all')

print("Эксперимент завершен. Результаты записаны в MLflow.")