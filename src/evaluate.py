import pandas as pd
import joblib
import json
import sys
import yaml
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_params(path='params.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    train_params = params['train']
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    metrics_path = sys.argv[3]

    print(f" Загрузка данных для оценки: {data_path}")
    df = pd.read_csv(data_path)

    # Загрузка модели и списка признаков, с которыми она обучалась
    model = joblib.load(model_path)
    feature_list_path = model_path.replace('.pkl', '_features.pkl')
    
    if not os.path.exists(feature_list_path):
        raise FileNotFoundError(f" Файл со списком признаков не найден: {feature_list_path}\n"
                                "Убедитесь, что этап train отработал успешно.")
    train_features = joblib.load(feature_list_path)
    print(f"🔍 Модель обучалась на {len(train_features)} признаках")

    # Подготовка target
    target = train_params['target_column']
    if df[target].dtype == 'object':
        df[target] = df[target].astype(str).str.strip().str.lower()
        mapping = {'placed': 1, 'not placed': 0, '1': 1, '0': 0, 'yes': 1, 'no': 0}
        df[target] = df[target].map(mapping)
        df = df.dropna(subset=[target])
        df[target] = df[target].astype(int)

    # Удаление служебных колонок
    drop_cols = [c for c in train_params['drop_columns'] if c in df.columns]
    X = df.drop(drop_cols + [target], axis=1, errors='ignore')
    y = df[target]

    # 🔹 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Приводим X к точному соответствию с train
    # 1. Удаляем лишние столбцы
    extra_cols = set(X.columns) - set(train_features)
    if extra_cols:
        print(f" Удаляю лишние столбцы: {extra_cols}")
        X = X.drop(columns=list(extra_cols))

    # 2. Добавляем отсутствующие (заполняем 0, чтобы не ломать shape)
    missing_cols = set(train_features) - set(X.columns)
    if missing_cols:
        print(f" Добавляю отсутствующие столбцы (0): {missing_cols}")
        for col in missing_cols:
            X[col] = 0

    # 3. Выставляем правильный порядок
    X = X[train_features]
    print(f" X для предсказания: {X.shape} (соответствует обучению)")

    # Предсказание
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

    # Метрики
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
    }
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y, y_proba)

    # Сохранение
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n Метрики сохранены: {metrics_path}")
    print(f"\n Classification Report:\n{classification_report(y, y_pred, target_names=['Not Placed', 'Placed'])}")
    print(f"Summary: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}" + 
          (f", ROC-AUC={metrics['roc_auc']:.4f}" if 'roc_auc' in metrics else ""))

if __name__ == '__main__':
    main()