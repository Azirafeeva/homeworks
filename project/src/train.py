#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import time

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.features import get_preprocessor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_and_prepare_data():
    """Загружает данные и разделяет на train/test."""
    df = pd.read_csv(config.data_raw_path, sep=';')
    
    df['y'] = (df['y'] == 'yes').astype(int)
    
    # Разделяем признаки и целевую
    X = df.drop(columns=['y'])
    y = df['y']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, y_train, X_test, y_test, name="Model", sample_weight=None):
    """
    Обучает модель и считает метрики.
    
    Args:
        model: sklearn модель
        X_train, y_train: обучающая выборка
        X_test, y_test: тестовая выборка
        name: имя модели
        sample_weight: веса для обучения (для балансировки)
    
    Returns:
        словарь с метриками и обученная модель
    """
    start_time = time.time()
    
    if sample_weight is not None:
        if hasattr(model, 'named_steps'):
            classifier_name = list(model.named_steps.keys())[-1]  # Получаем имя последнего шага
            model.fit(X_train, y_train, **{f'{classifier_name}__sample_weight': sample_weight})
        else:
            model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'model': name,
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'f1': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc': round(roc_auc_score(y_test, y_proba), 4),
        'train_time_sec': round(train_time, 2)
    }
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    metrics['pr_auc'] = round(auc(recall, precision), 4)
    
    print(f"{name}")
    print(f"Время обучения: {train_time:.2f} сек")
    print(f"Метрики:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['no', 'yes'], zero_division=0))
    
    return metrics, model


def plot_roc_curves(models_dict, X_test, y_test, save_path):
    """Строит и сохраняет ROC-кривые."""
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=2, label='Random')
    
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC-кривые моделей', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pr_curves(models_dict, X_test, y_test, save_path):
    """Строит и сохраняет Precision-Recall кривые."""
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, label=f'{name} (PR-AUC={pr_auc:.3f})', linewidth=2)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall кривые', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(model, X_test, y_test, name, save_path):
    """Строит и сохраняет матрицу ошибок."""
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No', 'Yes'],
                yticklabels=['No', 'Yes'])
    plt.title(f'Матрица ошибок: {name}', fontsize=14, fontweight='bold')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"True Negatives (TN):  {cm[0, 0]}")
    print(f"False Positives (FP): {cm[0, 1]}")
    print(f"False Negatives (FN): {cm[1, 0]}")
    print(f"True Positives (TP):  {cm[1, 1]}")


def main():
    """Основная функция обучения."""
    
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    
    preprocessor = get_preprocessor(cat_cols)
    
    # Baseline Logistic Regression
    logreg = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    pipeline_logreg = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', logreg)
    ])
    metrics_logreg, model_logreg = evaluate_model(
        pipeline_logreg, X_train, y_train, X_test, y_test,
        name="Baseline LogisticRegression"
    )
    
    # Gradient Boosting с sample_weight
    sample_weights = compute_sample_weight('balanced', y_train)
    gb_weighted = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE
    )
    pipeline_gb = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', gb_weighted)
    ])
    metrics_gb, model_gb = evaluate_model(
        pipeline_gb, X_train, y_train, X_test, y_test,
        name="GradientBoosting (sample_weight)",
        sample_weight=sample_weights
    )
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    pipeline_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])
    metrics_rf, model_rf = evaluate_model(
        pipeline_rf, X_train, y_train, X_test, y_test,
        name="RandomForest (balanced)"
    )
    
    results_df = pd.DataFrame([
        metrics_logreg,
        metrics_gb,
        metrics_rf
    ])
    
    results_df = results_df[[
        'model', 'roc_auc', 'pr_auc', 'f1', 'accuracy', 'train_time_sec'
    ]]
    
    print(results_df.to_string(index=False))
    
    results_path = config.reports_dir / 'metrics_comparison.csv'
    results_df.to_csv(results_path, index=False, decimal=',', sep=';')
    
    models_dict = {
        "LogReg": model_logreg,
        "GradientBoosting": model_gb,
        "RandomForest": model_rf,
    }
    
    plot_roc_curves(models_dict, X_test, y_test, config.figures_dir / 'roc_curves.png')
    plot_pr_curves(models_dict, X_test, y_test, config.figures_dir / 'pr_curves.png')
    plot_confusion_matrix(model_gb, X_test, y_test, "GradientBoosting",
                          config.figures_dir / 'confusion_matrix.png')
    
    final_model = model_rf
    final_model_name = "RandomForest (balanced)"
    final_metrics = metrics_rf

    final_model_path = config.models_dir / 'pipeline_v1.pkl'
    joblib.dump(final_model, final_model_path)

if __name__ == "__main__":
    main()