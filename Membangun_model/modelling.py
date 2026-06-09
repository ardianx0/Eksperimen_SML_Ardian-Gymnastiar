import mlflow
import mlflow.sklearn  # Sesuaikan jika menggunakan framework lain seperti xgboost / tensorflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 1. Inisialisasi Experiment MLflow
mlflow.set_experiment("Diabetes_Prediction_Experiment")

# 2. WAJIB: Aktifkan Fitur Autolog Sebelum Training Dimulai
mlflow.autolog()

# Load Data
data = pd.read_csv("diabetes.csv")
X = data.drop(columns=["Outcome"])
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Jalankan MLflow Run Context
with mlflow.start_run():
    # Model secara otomatis dicatat parameternya oleh autolog saat fitting
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluasi model (Metriks pengujian juga otomatis tercatat)
    accuracy = model.score(X_test, y_test)
    print(f"Model Training Selesai. Accuracy: {accuracy}")