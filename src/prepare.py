import pandas as pd
import numpy as np
import sys
import yaml
import os

def load_params(path='params.yaml'):
    """Загрузка параметров из YAML-файла"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def validate_columns(df, required_cols):
    """Проверка наличия обязательных колонок"""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing}")
    print(f" Все {len(required_cols)} обязательных колонки присутствуют")

def handle_missing_numeric(df, cols, strategy='median'):
    """Обработка пропусков в числовых колонках"""
    for col in cols:
        if col in df.columns and df[col].isnull().any():
            if strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'drop':
                df.dropna(subset=[col], inplace=True)
            print(f"  → {col}: заполнено {df[col].isnull().sum()} пропусков (стратегия: {strategy})")
    return df

def handle_missing_categorical(df, cols, strategy='mode'):
    """Обработка пропусков в категориальных колонках"""
    for col in cols:
        if col in df.columns and df[col].isnull().any():
            if strategy == 'mode':
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
            elif strategy == 'constant':
                df[col].fillna('Unknown', inplace=True)
            elif strategy == 'drop':
                df.dropna(subset=[col], inplace=True)
            print(f"  → {col}: заполнено {df[col].isnull().sum()} пропусков (стратегия: {strategy})")
    return df

def remove_outliers_iqr(df, cols, multiplier=3.0):
    """Удаление выбросов методом IQR"""
    initial_len = len(df)
    for col in cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    removed = initial_len - len(df)
    if removed > 0:
        print(f"  → Удалено {removed} строк с выбросами ({removed/initial_len*100:.2f}%)")
    return df

def normalize_column_names(df):
    """Приведение имён колонок к единому стилю (опционально)"""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def main():
    # === Загрузка параметров ===
    params = load_params()
    prepare_params = params.get('prepare', {})
    
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/student_placement_prediction_dataset_2026.csv'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'data/processed/cleaned.csv'
    
    print(f" Загрузка данных: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  → Загружено {len(df)} строк, {len(df.columns)} колонок")
    
    # === Нормализация имён колонок ===
    if prepare_params.get('normalize_names', True):
        df = normalize_column_names(df)
    
    # === Валидация обязательных колонок ===
    required_cols = [
        'student_id', 'age', 'gender', 'cgpa', 'branch', 'college_tier',
        'internships_count', 'projects_count', 'certifications_count',
        'coding_skill_score', 'aptitude_score', 'communication_skill_score',
        'logical_reasoning_score', 'placement_status'
    ]
    validate_columns(df, required_cols)
    
    # === Удаление дубликатов ===
    if prepare_params.get('remove_duplicates', True):
        initial = len(df)
        df = df.drop_duplicates(subset=prepare_params.get('duplicate_subset', ['student_id']))
        removed = initial - len(df)
        if removed > 0:
            print(f"✓ Удалено {removed} дубликатов")
    
    # === Обработка пропусков ===
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = ['gender', 'branch', 'college_tier', 'placement_status']  # явный список
    
    missing_strategy_num = prepare_params.get('missing_numeric_strategy', 'median')
    missing_strategy_cat = prepare_params.get('missing_categorical_strategy', 'mode')
    
    print("🔧 Обработка пропусков:")
    df = handle_missing_numeric(df, numeric_cols, strategy=missing_strategy_num)
    df = handle_missing_categorical(df, categorical_cols, strategy=missing_strategy_cat)
    
    # === Удаление выбросов (опционально) ===
    if prepare_params.get('remove_outliers', False):
        outlier_cols = prepare_params.get('outlier_columns', [])
        multiplier = prepare_params.get('outlier_multiplier', 3.0)
        print(f"🔍 Удаление выбросов в колонках: {outlier_cols} (множитель IQR: {multiplier})")
        df = remove_outliers_iqr(df, outlier_cols, multiplier=multiplier)
    
    # === Приведение типов данных (опционально) ===
    if prepare_params.get('optimize_dtypes', True):
        # Уменьшаем потребление памяти для целочисленных колонок
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].max() < 255:
                df[col] = df[col].astype('int8')
            elif df[col].max() < 65535:
                df[col] = df[col].astype('int16')
        # Для float
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        print("✓ Оптимизированы типы данных")
    
    # === Сохранение результата ===
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n Данные очищены и сохранены: {output_path}")
    print(f"  → Итоговый размер: {len(df)} строк × {len(df.columns)} колонок")
    print(f"  → Категориальные колонки: {[c for c in categorical_cols if c in df.columns]}")
    print(f"  → Числовые колонки: {len(df.select_dtypes(include=[np.number]).columns)}")

if __name__ == '__main__':
    main()