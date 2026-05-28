import pandas as pd
import joblib
import sys
import yaml
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

def load_params(path='params.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    train_params = params['train']
    input_path = sys.argv[1]
    model_path = sys.argv[2]

    print(f" Загрузка признаков: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  → Форма данных: {df.shape}")

    target = train_params['target_column']
    if target not in df.columns:
        raise ValueError(f" Целевая колонка '{target}' не найдена в данных!")

    # 🔹 Безопасное кодирование target
    if df[target].dtype == 'object':
        df[target] = df[target].astype(str).str.strip().str.lower()
        mapping = {'placed': 1, 'not placed': 0, '1': 1, '0': 0, 'yes': 1, 'no': 0}
        df[target] = df[target].map(mapping)
        
        if df[target].isnull().any():
            print(" Обнаружены нераспознанные значения в target. Удаляю строки с NaN...")
            df = df.dropna(subset=[target])
        df[target] = df[target].astype(int)

    # 🔹 Подготовка X и y
    drop_cols = [c for c in train_params['drop_columns'] if c in df.columns]
    X = df.drop(drop_cols + [target], axis=1, errors='ignore')
    y = df[target]

    # 🔹 Удаляем оставшиеся строковые колонки (если features.py что-то пропустил)
    non_numeric = X.select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        print(f"Удаляю оставшиеся строковые колонки: {non_numeric}")
        X = X.drop(columns=non_numeric)

    print(f"Признаков для обучения: {X.shape[1]}, Строк: {X.shape[0]}")

    # 🔹 Разделение с защитой от stratify-ошибок
    try:
        class_counts = y.value_counts()
        stratify_param = y if (class_counts >= 2).all() and len(class_counts) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=float(train_params['test_size']),
            random_state=int(train_params['random_state']),
            stratify=stratify_param
        )
    except Exception as e:
        print(f" Stratify не сработал ({e}). Разделяю без стратификации...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=float(train_params['test_size']),
            random_state=int(train_params['random_state'])
        )

    # 🔹 Обучение
    print("Запуск обучения RandomForest...")
    model = RandomForestClassifier(
        n_estimators=int(train_params['n_estimators']),
        max_depth=int(train_params['max_depth']),
        random_state=int(train_params['random_state']),
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 🔹 Сохранение
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, model_path)

    feature_list_path = model_path.replace('.pkl', '_features.pkl')
    joblib.dump(X.columns.tolist(), feature_list_path)

    print(f"Модель сохранена: {model_path}")
    print(f"Список признаков: {feature_list_path}")

if __name__ == '__main__':
    main()