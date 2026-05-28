python -m venv venv && source venv/bin/activate
pip install dvc pandas scikit-learn numpy seaborn matplotlib joblib

git init
dvc init

dvc repro
dvc dag
