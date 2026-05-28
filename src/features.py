import pandas as pd
import sys
import yaml
import joblib
import os

def load_params(path='params.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    encoder_path = os.path.join(os.path.dirname(output_path), 'encoder.pkl')
    
    df = pd.read_csv(input_path)
    
    # === 1. Генерация новых признаков ===
    df['skill_ratio'] = df['coding_skill_score'] / (df['aptitude_score'] + 1)
    df['experience_score'] = (
        df['internships_count'] * params['features']['internship_weight'] +
        df['projects_count'] * params['features']['project_weight'] +
        df['certifications_count'] * params['features']['cert_weight']
    )
    df['academic_strength'] = df['cgpa'] * df['attendance_percentage'] / 100
    df['engagement_score'] = (
        df['hackathons_participated'] + 
        df['github_repos'] * 0.5 + 
        df['linkedin_connections'] * 0.1
    )
    
    # === 2. Кодирование категориальных признаков ===
    categorical_cols = params['features']['categorical_columns']  # ['gender', 'branch', 'college_tier']
    
    # Используем Label Encoding для простоты (или One-Hot, если нужно)
    encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            if params['features'].get('one_hot_encode', False):
                # One-Hot Encoding
                df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
            else:
                # Label Encoding
                df[col] = df[col].astype('category')
                encoders[col] = df[col].cat.codes
                df[col] = df[col].cat.codes
    
    # Сохраняем энкодеры для инференса (опционально)
    if params['features'].get('save_encoders', True):
        joblib.dump(encoders, encoder_path)
        print(f"Энкодеры сохранены: {encoder_path}")
    
    # === 3. Сохранение результата ===
    df.to_csv(output_path, index=False)
    print(f"Признаки сгенерированы: {output_path}")
    print(f"Колонок после обработки: {len(df.columns)}")

if __name__ == '__main__':
    main()