import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, 
                             classification_report, roc_curve, auc, 
                             precision_recall_curve, RocCurveDisplay)
from sklearn.inspection import permutation_importance
from sklearn.multiclass import OneVsRestClassifier
import json
from datetime import datetime
import itertools
import argparse
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Создание эксперимента
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Iris_Classification_Comparison"
mlflow.set_experiment(experiment_name)

# Параметры эксперимента
SEED = 42
N_SPLITS = 5

# Конфигурация скейлеров
SCALERS = {
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "none": None
}

# Конфигурация вариантов фич
FEATURE_VARIANTS = {
    "all_features": [0, 1, 2, 3],
    "only_petal": [2, 3],
    "only_sepal": [0, 1],
    "length_only": [0, 2],  # только длины
    "width_only": [1, 3]    # только ширины
}

# Конфигурация моделей с гиперпараметрами
MODELS_CONFIG = {
    "LogisticRegression": {
        "class": LogisticRegression,
        "params": {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": SEED,
            "multi_class": "ovr",
            "solver": "liblinear"
        },
        "requires_scaling": True
    },
    "RandomForest": {
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": 4,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": SEED
        },
        "requires_scaling": False
    },
    "KNN": {
        "class": KNeighborsClassifier,
        "params": {
            "n_neighbors": 5,
            "weights": "uniform",
            "algorithm": "auto"
        },
        "requires_scaling": True
    }
}

# Парсинг аргументов командной строки
def parse_args():
    parser = argparse.ArgumentParser(description='Train ML model on Iris dataset')
    parser.add_argument('--model', type=str, default='RandomForest', 
                       choices=['LogisticRegression', 'RandomForest', 'KNN'],
                       help='Model to train')
    parser.add_argument('--scaler', type=str, default='standard',
                       choices=['standard', 'minmax', 'none'],
                       help='Scaler to use')
    parser.add_argument('--features', type=str, default='all_features',
                       choices=list(FEATURE_VARIANTS.keys()),
                       help='Feature variant to use')
    parser.add_argument('--validation', type=str, default='train_test_split',
                       choices=['train_test_split', 'kfold'],
                       help='Validation scheme')
    parser.add_argument('--seed', type=int, default=SEED,
                       help='Random seed')
    return parser.parse_args()

# Функция для создания confusion matrix
def plot_confusion_matrix(y_true, y_pred, target_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names,
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig

# Функция для создания ROC-кривой
def plot_multiclass_roc(y_true, y_pred_proba, target_names):
    n_classes = len(target_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = itertools.cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])
    
    for i, color in zip(range(n_classes), colors):
        y_true_binary = (y_true == i).astype(int)
        y_score_binary = y_pred_proba[:, i]
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'ROC {target_names[i]} (AUC = {roc_auc:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multiclass ROC Curve (One-vs-Rest)')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# Функция для создания Precision-Recall кривой
def plot_multiclass_pr_curve(y_true, y_pred_proba, target_names):
    n_classes = len(target_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = itertools.cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])
    
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
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# Функция для создания графика важности признаков
def plot_feature_importance(feature_importance, feature_names, model_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = np.argsort(feature_importance)
    
    bars = ax.barh(range(len(indices)), feature_importance[indices], 
                  color='skyblue', edgecolor='black')
    
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Feature Importance - {model_name}')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Добавляем значения на бары
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    return fig

# Функция для создания отчета
def create_report(y_true, y_pred, y_pred_proba, target_names, feature_importance, 
                 model_config, scaler_name, feature_variant, cv_results=None):
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_type": model_config["class"].__name__,
        "model_params": model_config["params"],
        "scaler": scaler_name,
        "feature_variant": feature_variant,
        "metrics": {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average='macro'),
            "weighted_f1": f1_score(y_true, y_pred, average='weighted')
        },
        "classification_report": classification_report(y_true, y_pred, 
                                                     target_names=target_names, output_dict=True),
        "feature_importance": feature_importance.tolist() if hasattr(feature_importance, 'tolist') 
                          else feature_importance,
        "notes": f"Модель {model_config['class'].__name__} на Iris dataset"
    }
    
    if cv_results:
        report["cross_validation"] = {
            "accuracy_mean": cv_results['test_accuracy'].mean(),
            "accuracy_std": cv_results['test_accuracy'].std(),
            "f1_mean": cv_results['test_f1_macro'].mean(),
            "f1_std": cv_results['test_f1_macro'].std()
        }
    
    return report

# Основная функция обучения
def train_model(args):
    # Парсим аргументы
    model_name = args.model
    scaler_name = args.scaler
    feature_variant = args.features
    validation_scheme = args.validation
    seed = args.seed
    
    model_config = MODELS_CONFIG[model_name]
    
    with mlflow.start_run(run_name=f"{model_name}_{feature_variant}_{scaler_name}"):
        # Логируем все параметры
        mlflow.log_param("dataset", "iris")
        mlflow.log_param("seed", seed)
        mlflow.log_param("validation_scheme", validation_scheme)
        mlflow.log_param("model", model_name)
        mlflow.log_param("scaler", scaler_name)
        mlflow.log_param("feature_variant", feature_variant)
        
        # Логируем гиперпараметры модели
        mlflow.log_params(model_config["params"])
        
        # Выбор фич
        X_selected = X[:, FEATURE_VARIANTS[feature_variant]]
        selected_feature_names = [feature_names[i] for i in FEATURE_VARIANTS[feature_variant]]
        
        # Скейлинг (если требуется для модели)
        scaler = SCALERS[scaler_name]
        if scaler is not None and model_config["requires_scaling"]:
            X_scaled = scaler.fit_transform(X_selected)
        else:
            X_scaled = X_selected
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        # Создание и обучение модели
        model_class = model_config["class"]
        model = model_class(**model_config["params"])
        
        # Включение autologging
        mlflow.sklearn.autolog(
            log_input_examples=True,
            log_model_signatures=True,
            log_models=True,
            log_datasets=True,
            max_tuning_runs=5,
            log_post_training_metrics=True
        )
        
        # Обучение модели
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_metric("weighted_f1", weighted_f1)
        
        # Кросс-валидация для дополнительных метрик
        if validation_scheme == "kfold":
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            cv_accuracy = cross_val_score(model, X_scaled, y, cv=kf, scoring='accuracy')
            cv_f1 = cross_val_score(model, X_scaled, y, cv=kf, scoring='f1_macro')
            
            mlflow.log_metric("cv_accuracy_mean", cv_accuracy.mean())
            mlflow.log_metric("cv_accuracy_std", cv_accuracy.std())
            mlflow.log_metric("cv_f1_mean", cv_f1.mean())
            mlflow.log_metric("cv_f1_std", cv_f1.std())
        
        # Важность признаков (если доступно)
        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            # Для линейных моделей берем абсолютные значения коэффициентов
            feature_importance = np.mean(np.abs(model.coef_), axis=0)
        
        # Создание и логирование визуализаций
        fig_cm = plot_confusion_matrix(y_test, y_pred, target_names)
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        
        if y_pred_proba is not None:
            fig_roc = plot_multiclass_roc(y_test, y_pred_proba, target_names)
            mlflow.log_figure(fig_roc, "roc_curve.png")
            
            fig_pr = plot_multiclass_pr_curve(y_test, y_pred_proba, target_names)
            mlflow.log_figure(fig_pr, "pr_curve.png")
        
        if feature_importance is not None:
            fig_fi = plot_feature_importance(feature_importance, selected_feature_names, model_name)
            mlflow.log_figure(fig_fi, "feature_importance.png")
        
        # Permutation importance
        try:
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=seed)
            fig_perm, ax = plt.subplots(figsize=(10, 6))
            sorted_idx = perm_importance.importances_mean.argsort()
            ax.boxplot(perm_importance.importances[sorted_idx].T,
                      labels=np.array(selected_feature_names)[sorted_idx])
            ax.set_title("Permutation Importance")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            mlflow.log_figure(fig_perm, "permutation_importance.png")
        except Exception as e:
            print(f"Permutation importance failed: {e}")
        
        # Создание и логирование отчета
        cv_results = None
        if validation_scheme == "kfold":
            cv_results = {
                'test_accuracy': cv_accuracy,
                'test_f1_macro': cv_f1
            }
        
        report = create_report(y_test, y_pred, y_pred_proba, target_names, 
                             feature_importance, model_config, scaler_name, 
                             feature_variant, cv_results)
        mlflow.log_dict(report, "report.json")
        
        # Текстовый отчет
        report_md = f"""
# Отчет по запуску классификации Iris

## Конфигурация
- Модель: {model_name}
- Скейлер: {scaler_name}
- Вариант фич: {feature_variant}
- Схема валидации: {validation_scheme}
- Seed: {seed}

## Гиперпараметры модели
```json
{json.dumps(model_config["params"], indent=2, ensure_ascii=False)}
Метрики
Accuracy: {accuracy:.4f}
Macro F1: {macro_f1:.4f}
Weighted F1: {weighted_f1:.4f}
"""
    if validation_scheme == "kfold":
        report_md += f"""

Accuracy (mean ± std): {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}
F1 Macro (mean ± std): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}
"""
    if feature_importance is not None:
        report_md += f"""
        
    {json.dumps(dict(zip(selected_feature_names, 
                    [float(f) for f in feature_importance])), indent=2, ensure_ascii=False)}
"""
    report_md += f"""
Модель {model_name} показывает {'хорошее' if accuracy > 0.9 else 'удовлетворительное'} качество классификации.
"""
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    mlflow.log_artifact("report.md")
    
    # Закрываем figures
    plt.close('all')
    
    print(f"Обучение завершено: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

if __name__ == "__main__":
    args = parse_args()
    print(f"Запуск обучения с параметрами:")
    print(f" Модель: {args.model}")
    print(f" Скейлер: {args.scaler}")
    print(f" Фичи: {args.features}")
    print(f" Валидация: {args.validation}")
    print(f" Seed: {args.seed}")

    train_model(args)
    print("Все эксперименты завершены!")