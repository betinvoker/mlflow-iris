import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

def load_data():
    """Загрузка и подготовка данных Iris"""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    return X, y, iris.target_names

def create_features(X, feature_type, scaler_type=None):
    """Создание признаков в зависимости от типа"""
    if feature_type == 'base':
        X_processed = X.copy()
    elif feature_type == 'scaled':
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
        X_processed = pd.DataFrame(X_processed, columns=X.columns)
    elif feature_type == 'poly':
        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        X_processed = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(X.columns)
        X_processed = pd.DataFrame(X_processed, columns=feature_names)
    else:
        X_processed = X.copy()
    
    return X_processed

def get_model(model_name, seed=42):
    """Получение модели по имени"""
    if model_name == 'logreg':
        if seed:
            return LogisticRegression(random_state=seed, max_iter=1000, multi_class='ovr')
        return LogisticRegression(max_iter=1000, multi_class='ovr')
    elif model_name == 'randomforest':
        if seed:
            return RandomForestClassifier(random_state=seed, n_estimators=100)
        return RandomForestClassifier(n_estimators=100)
    elif model_name == 'knn':
        return KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def plot_feature_importance(model, feature_names, model_name):
    """Построение важности признаков"""
    plt.figure(figsize=(10, 6))
    
    if hasattr(model, 'feature_importances_'):
        # Для RandomForest
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.title('Feature Importances')
    
    elif hasattr(model, 'coef_'):
        # Для LogisticRegression
        if len(model.coef_.shape) > 1:
            # Мультиклассовая классификация
            for i in range(model.coef_.shape[0]):
                plt.bar(range(len(feature_names)), model.coef_[i], alpha=0.7, label=f'Class {i}')
            plt.legend()
        else:
            plt.bar(range(len(feature_names)), model.coef_)
        plt.xticks(range(len(feature_names)), feature_names, rotation=45)
        plt.title('Feature Coefficients')
    
    plt.tight_layout()
    return plt.gcf()

def main():
    parser = argparse.ArgumentParser(description='Train classification models on Iris dataset')
    parser.add_argument('--model', type=str, default='logreg', 
                       choices=['logreg', 'randomforest', 'knn'],
                       help='Model type')
    parser.add_argument('--scaler', type=str, default='standard',
                       choices=['standard', 'minmax', 'none'],
                       help='Scaler type')
    parser.add_argument('--features', type=str, default='base',
                       choices=['base', 'scaled', 'poly'],
                       help='Feature engineering type')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    # Загрузка данных
    X, y, class_names = load_data()
    
    # Создание признаков
    X_processed = create_features(X, args.features, args.scaler if args.features == 'scaled' else None)
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    # Настройка MLflow
    experiment_name = "iris-classification"
    mlflow.set_tracking_uri("http://localhost:5001")

    # Создаем эксперимент, если его нет
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise Exception("Experiment not found")
        experiment_id = experiment.experiment_id
    except:
        experiment_id = mlflow.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)
  
    with mlflow.start_run():
        # Логирование параметров
        mlflow.log_params({
            'model': args.model,
            'scaler': args.scaler if args.features == 'scaled' else 'none',
            'features': args.features,
            'seed': args.seed,
            'cv_folds': args.cv_folds,
            'test_size': 0.2
        })
        
        # Добавление тегов
        mlflow.set_tags({
            'model': args.model,
            'purpose': 'ablation',
            'candidate': 'true',
            'dataset': 'iris'
        })
        
        # Получение модели
        model = get_model(args.model, args.seed)
        
        # Включение autolog
        mlflow.sklearn.autolog()
        
        # Кросс-валидация
        kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        cv_accuracy = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
        cv_f1 = cross_val_score(model, X_train, y_train, cv=kf, scoring='f1_macro')
        
        # Обучение модели
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_test)
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        # Логирование метрик
        mlflow.log_metrics({
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'cv_accuracy_mean': cv_accuracy.mean(),
            'cv_accuracy_std': cv_accuracy.std(),
            'cv_f1_mean': cv_f1.mean(),
            'cv_f1_std': cv_f1.std()
        })
        
        # Визуализации
        # Важность признаков
        if args.model in ['logreg', 'randomforest']:
            fi_fig = plot_feature_importance(model, X_processed.columns.tolist(), args.model)
            mlflow.log_figure(fi_fig, "feature_importance.png")
            plt.close(fi_fig)
        
        # Permutation importance
        try:
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=args.seed)
            sorted_idx = perm_importance.importances_mean.argsort()[::-1]
            
            plt.figure(figsize=(10, 10))
            plt.bar(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx])
            plt.xticks(range(len(sorted_idx)), [X_processed.columns[i] for i in sorted_idx], rotation=45)
            plt.title('Permutation Importance')
            plt.tight_layout()
            
            perm_fig = plt.gcf()
            mlflow.log_figure(perm_fig, "permutation_importance.png")
            plt.close(perm_fig)
        except Exception as e:
            print(f"Permutation importance failed: {e}")
        
        # Линейная диаграмма точности предсказаний по эпохам
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                model, X_train, y_train,
                cv=args.cv_folds,
                scoring="accuracy",
                n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10)  # 10 точек от 10% до 100% данных
            )

            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)

            plt.figure(figsize=(6,4))
            plt.plot(train_sizes, train_scores_mean, marker='o', label="Train Accuracy")
            plt.plot(train_sizes, test_scores_mean, marker='s', label="CV Accuracy")
            plt.title(f"Learning Curve (model {args.model})")
            plt.xlabel("Training examples")
            plt.ylabel("Accuracy")
            plt.ylim(0, 1.05)
            plt.grid(True)
            plt.legend()

            perm_fig = plt.gcf()
            mlflow.log_figure(perm_fig, f"learning_curve.png")
            plt.close(perm_fig)
        except Exception as e:
            print(f"Ошибка при построении learning curve: {e}")

        # Classification report
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        mlflow.log_dict(report, "classification_report.json")
        
        # Создание отчета
        report_text = f"""
# Model Report {args.model}

## Parameters
- Model: {args.model}
- Features: {args.features}
- Scaler: {args.scaler if args.features == 'scaled' else 'none'}
- Random seed: {args.seed}
- CV folds: {args.cv_folds}

## Metrics
- Accuracy: {accuracy:.4f}
- Macro F1: {f1_macro:.4f}
- CV Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}
- CV F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}

## Conclusions
The model showed {'good' if accuracy > 0.9 else 'satisfactory'} results.
Recommended {'use in production' if accuracy > 0.95 else 'conduct additional experiments'}.
"""
        
        with open("report.md", "w") as f:
            f.write(report_text)
        mlflow.log_artifact("report.md")
        
        # Логирование модели
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model, 
            "model", 
            signature=signature,
            input_example=X_train.iloc[:5],
            registered_model_name=f"iris-{args.model}-{args.features}"
        )
        
        print(f"Run completed: {mlflow.active_run().info.run_id}")
        print(f"Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}")

if __name__ == "__main__":
    main()