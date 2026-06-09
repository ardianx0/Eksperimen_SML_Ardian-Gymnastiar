import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import dagshub

def main():
    repo_owner = "ardianx0"
    repo_name = "Eksperimen_SML_Ardian-Gymnastiar"
    
    print("Menghubungkan ke DagsHub Tracking URI...")
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_experiment("Diabetes_Baseline_Model")
    
    # Membaca dataset dari folder kriteria 1 kamu
    data_path = os.path.join("preprocessing", "diabetes_preprocessing", "diabetes_clean.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data bersih tidak ditemukan di {data_path}!")
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    n_estimators = 50
    max_depth = 5
    
    with mlflow.start_run(run_name="Random_Forest_Baseline"):
        print("Melatih model Random Forest Baseline...")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Hasil Evaluasi -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")
        
        # Logging parameter & metrik secara manual sesuai kriteria
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # Pembuatan Artefak 1: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sehat', 'Diabetes'], yticklabels=['Sehat', 'Diabetes'])
        plt.title('Confusion Matrix - Baseline Model')
        plt.ylabel('Aktual')
        plt.xlabel('Prediksi')
        cm_path = "confusion_matrix_baseline.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)
        
        # Artefak Tambahan 1: Grafik Distribusi Probabilitas Prediksi
        plt.figure(figsize=(7, 4))
        sns.histplot(y_pred_proba, kde=True, color='purple', bins=15)
        plt.title('Distribusi Probabilitas Prediksi Diabetes')
        plt.xlabel('Probabilitas Prediksi')
        plt.ylabel('Frekuensi')
        prob_path = "prediction_probabilities.png"
        plt.savefig(prob_path)
        plt.close()
        mlflow.log_artifact(prob_path)
        os.remove(prob_path)
        
        # Artefak Tambahan 2: File Ringkasan Teks
        summary_path = "summary_report.txt"
        with open(summary_path, "w") as f:
            f.write("=== LAPORAN EVALUASI MODEL DIABETES ===\n")
            f.write(f"Model: Random Forest Baseline\n")
            f.write(f"Akurasi Pengujian: {acc:.4f}\n")
            f.write(f"F1-Score Pengujian: {f1:.4f}\n")
        mlflow.log_artifact(summary_path)
        os.remove(summary_path)
        
        mlflow.sklearn.log_model(model, "model")
        print("Eksperimen model baseline berhasil dicatat di DagsHub!")

if __name__ == "__main__":
    main()