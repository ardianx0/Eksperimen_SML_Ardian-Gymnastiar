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
    
    print("Menghubungkan ke DagsHub Tracking URI untuk Tuning...")
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_experiment("Diabetes_Hyperparameter_Tuning")
    
    mlflow.autolog(log_model_signatures=True, log_input_examples=True, disable=False)
    
    data_path = os.path.join("preprocessing", "diabetes_preprocessing", "diabetes_clean.csv")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Ruang uji coba parameter (Grid Search Manual Looping)
    tuning_space = [
        {"n_estimators": 100, "max_depth": 8},
        {"n_estimators": 150, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 12},
        {"n_estimators": 80,  "max_depth": 6}
    ]
    
    for i, params in enumerate(tuning_space):
        run_name = f"RF_Tuning_Kombinasi_{i+1}"
        with mlflow.start_run(run_name=run_name):
            print(f"\n--- Menjalankan {run_name} ---")
            
            model = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            print(f"Hasil -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
            
            # Logging parameter dan metrik tambahan secara manual
            mlflow.log_param("model_type", "RandomForestClassifier_Tuned")
            mlflow.log_param("n_estimators", params['n_estimators'])
            mlflow.log_param("max_depth", params['max_depth'])
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            
            # Artefak Grafik 1: Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))  # Diperbaiki menggunakan subplots agar state bersih
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                        xticklabels=['Sehat', 'Diabetes'], yticklabels=['Sehat', 'Diabetes'], ax=ax)
            ax.set_title(f'Confusion Matrix - Tuning {i+1}')
            cm_path = f"confusion_matrix_tuning_{i+1}.png"
            fig.savefig(cm_path)
            plt.close(fig)
            mlflow.log_artifact(cm_path)
            os.remove(cm_path)
            
            # Artefak Grafik 2: Feature Importance Plot
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            fig2, ax2 = plt.subplots(figsize=(8, 4))  # Diperbaiki menggunakan instance gambar baru
            sns.barplot(x=importances[indices], y=X.columns[indices], palette="viridis", ax=ax2)
            ax2.set_title(f'Fitur Paling Berpengaruh - Tuning {i+1}')
            feat_path = f"feature_importance_tuning_{i+1}.png"
            fig2.savefig(feat_path)
            plt.close(fig2)
            mlflow.log_artifact(feat_path)
            os.remove(feat_path)
            
            # Artefak Tambahan 3: Ringkasan Laporan .txt
            summary_path = f"tuning_report_{i+1}.txt"
            with open(summary_path, "w") as f:
                f.write(f"=== LAPORAN TUNING JALUR {i+1} ===\n")
                f.write(f"Config: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}\n")
                f.write(f"Akurasi: {acc:.4f}\n")
                f.write(f"F1-Score: {f1:.4f}\n")
            mlflow.log_artifact(summary_path)
            os.remove(summary_path)
            
            # Registrasi model final ke dalam komponen artifacts MLflow
            mlflow.sklearn.log_model(model, "model")
            
    print("\nSemua proses Hyperparameter Tuning sukses dicatat di DagsHub!")

if __name__ == "__main__":
    main()