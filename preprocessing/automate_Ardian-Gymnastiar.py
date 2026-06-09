import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def run_automation():
    # Mengambil base directory secara dinamis
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'preprocessing' in os.getcwd() else os.getcwd()
    input_path = os.path.join(base_dir, 'diabetes.csv')
    output_dir = os.path.join(base_dir, 'preprocessing', 'diabetes_preprocessing')

    if not os.path.exists(input_path):
        print(f"Error: {input_path} tidak ditemukan!")
        return

    df = pd.read_csv(input_path)
    df_clean = df.copy()
    
    # 1. Handling Missing Values (Nilai 0 tidak logis)
    cols_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_check:
        df_clean[col] = df_clean[col].replace(0, np.nan)
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # 2. Feature Scaling untuk semua kolom kecuali Target
    X = df_clean.drop(columns=['Outcome'])
    y = df_clean['Outcome']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    df_preprocessed = pd.DataFrame(X_scaled, columns=X.columns)
    df_preprocessed['Outcome'] = y.reset_index(drop=True)
    
    # 3. Simpan ke folder tujuan kriteria
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'diabetes_clean.csv')
    df_preprocessed.to_csv(output_file, index=False)
    print(f"Otomatisasi Preprocessing Sukses! File disimpan di: {output_file}")

if __name__ == "__main__":
    run_automation()