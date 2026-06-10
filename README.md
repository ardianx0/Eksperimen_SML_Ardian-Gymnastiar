# 🩺 Diabetes Monitoring App

**Proyek Submission Akhir — Kelas "Membangun Sistem Machine Learning"**  
*PIJAK in Collaboration with IBM SkillsBuild and Dicoding*

**Username Dicoding:** `ardian_g8`  
**Developer:** Ardian Gymnastiar  
**Repository GitHub:** [ardianx0/Eksperimen_SML_Ardian-Gymnastiar](https://github.com/ardianx0/Eksperimen_SML_Ardian-Gymnastiar)

---

## 📖 Tentang Proyek

Proyek **Diabetes Monitoring App** merupakan submission akhir dari kelas **"Membangun Sistem Machine Learning"** yang diselenggarakan oleh **PIJAK** bekerja sama dengan **IBM SkillsBuild** dan **Dicoding**. Proyek ini membangun sistem *end-to-end machine learning* untuk memprediksi diabetes pada pasien berdasarkan delapan fitur klinis dari dataset **Pima Indians Diabetes Database**.

Sistem mencakup seluruh pipeline ML:
1. **Preprocessing** data otomatis
2. **Pembangunan model** dengan **Random Forest Classifier**
3. **Experiment tracking** menggunakan **MLflow** yang diintegrasikan dengan **DagsHub**
4. **Hyperparameter tuning** untuk optimalisasi performa model
5. **Monitoring dan logging** menggunakan **Prometheus** dan **Grafana**

---

## 🎯 Tujuan

- Menerapkan *end-to-end machine learning pipeline* mulai dari data mentah hingga model siap *deploy*
- Melakukan *experiment tracking* dan *hyperparameter tuning* menggunakan MLflow
- Membangun sistem *monitoring* untuk metrik performa model dan infrastruktur
- Membuat *alerting* untuk mendeteksi anomali pada sistem

---

## 🗂️ Struktur Proyek

```
Eksperimen_SML_Ardian-Gymnastiar/
├── .gitignore
├── diabetes.csv                      # Dataset mentah
├── requirements.txt                  # Dependencies utama
├── Workflow-CI.txt                   # Link workflow CI
├── README.md                         # Dokumentasi proyek
│
├── preprocessing/
│   ├── Eksperimen_Ardian-Gymnastiar.ipynb   # Notebook eksperimen preprocessing
│   ├── Eksperimen_SML_Ardian-Gymnastiar.txt # Link repository GitHub
│   ├── automate_Ardian-Gymnastiar.py        # Script otomatisasi preprocessing
│   └── diabetes_preprocessing/
│       └── diabetes_clean.csv               # Dataset hasil preprocessing
│
├── Membangun_model/
│   ├── requirements.txt              # Dependencies modelling & MLflow
│   ├── DagsHub.txt                   # Link DagsHub
│   ├── modelling.py                  # Script training model (autolog MLflow)
│   ├── modelling_tuning.py           # Script hyperparameter tuning
│   ├── screenshoot_artifak.jpg       # Screenshot artifacts MLflow
│   └── screenshoot_dashboard.jpg     # Screenshot dashboard MLflow
│
└── Monitoring dan Logging/
    ├── inference.py                  # Flask API endpoint inference
    ├── prometheus_exporter.py        # Eksportir metrik Prometheus
    ├── prometheus.yml                 # Konfigurasi Prometheus
    ├── bukti_serving.jpg             # Screenshot serving berjalan
    ├── bukti_serving2.jpg            # Screenshot serving berjalan
    ├── bukti alerting Grafana/       # Screenshot alerting Grafana
    │   ├── rules_active_sessions.jpg
    │   ├── rules_cpu_usage_high.jpg
    │   └── rules_inference_errors.jpg
    ├── bukti monitoring Grafana/     # Screenshot dashboard monitoring Grafana
    │   ├── monitoring_inference_latency_seconds_bucket.jpg
    │   ├── monitoring_inference_requests_total.jpg
    │   ├── monitoring_predictions_diabetes_total.jpg
    │   ├── monitoring_predictions_sehat_total.jpg
    │   ├── monitoring_system_cpu_usage_percentage.png
    │   └── monitoring_system_memory_usage_bytes.png
    └── bukti monitoring Prometheus/  # Screenshot monitoring Prometheus
        ├── monitoring_cpu_graph.jpg
        ├── monitoring_requests_graph.jpg
        └── monitoring_target_up.jpg
```

---

## 📊 Dataset

Dataset yang digunakan adalah **Pima Indians Diabetes Database** yang berisi data medis pasien wanita keturunan Suku Indian Pima. Dataset ini tersedia secara publik di Kaggle.

### Fitur-fitur Dataset:

| Fitur | Deskripsi |
|-------|-----------|
| **Pregnancies** | Jumlah kehamilan |
| **Glucose** | Konsentrasi glukosa plasma (mg/dL) |
| **BloodPressure** | Tekanan darah diastolik (mm Hg) |
| **SkinThickness** | Ketebalan lipatan kulit trisep (mm) |
| **Insulin** | Insulin serum 2 jam (mu U/ml) |
| **BMI** | Indeks massa tubuh (kg/m²) |
| **DiabetesPedigreeFunction** | Fungsi silsilah diabetes |
| **Age** | Usia (tahun) |
| **Outcome** | Target: 1 = Diabetes, 0 = Tidak Diabetes |

- **Jumlah data:** 768 sampel
- **Jumlah fitur:** 8 fitur numerik + 1 target biner
- **Ketidakseimbangan kelas:** 268 positif (34.9%) dan 500 negatif (65.1%)

---

## ⚙️ Tahapan Proyek

### 1. Data Preprocessing

Proses preprocessing dilakukan secara otomatis melalui script `preprocessing/automate_Ardian-Gymnastiar.py` yang mencakup:

1. **Handling Missing Values** — Nilai `0` yang tidak logis pada kolom `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` diganti dengan `NaN`, kemudian diisi dengan nilai median masing-masing kolom.
2. **Feature Scaling** — Semua fitur (kecuali target) distandarisasi menggunakan **StandardScaler** (mean=0, std=1).
3. **Penyimpanan Data** — Hasil preprocessing disimpan ke `preprocessing/diabetes_preprocessing/diabetes_clean.csv`.

### 2. Pembangunan Model

Model dibangun menggunakan **Random Forest Classifier** dengan experiment tracking melalui **MLflow**:

- **Script:** `Membangun_model/modelling.py`
- **Algoritma:** Random Forest Classifier
  - `n_estimators=100`
  - `max_depth=5`
  - `random_state=42`
- **Split data:** 80% training, 20% testing (random_state=42)
- **Experiment tracking:** MLflow autolog dengan experiment name `Diabetes_Prediction_Experiment`
- **Integrasi:** DagsHub sebagai remote tracking URI

### 3. Hyperparameter Tuning

Hyperparameter tuning dilakukan secara manual (*grid search*) untuk menemukan kombinasi parameter terbaik:

| Kombinasi | n_estimators | max_depth |
|-----------|:---:|:-----:|
| 1 | 100 | 8 |
| 2 | 150 | 10 |
| 3 | 200 | 12 |
| 4 | 80  | 6 |

**Metrik yang dievaluasi:**
- Accuracy
- Precision
- Recall
- F1-Score

**Artefak yang dilogging:**
- Confusion Matrix (heatmap)
- Feature Importance Plot
- Laporan tuning (ringkasan `.txt`)
- Model terdaftar di MLflow artifacts

Semua hasil tuning dicatat di **DagsHub**: [Eksperimen_SML_Ardian-Gymnastiar](https://dagshub.com/ardianx0/Eksperimen_SML_Ardian-Gymnastiar)

### 4. Monitoring dan Logging

Sistem monitoring dibangun menggunakan **Prometheus** dan **Grafana** dengan metrik real-time.

#### Metrik yang Dipantau:
| Metrik | Tipe | Deskripsi |
|--------|------|-----------|
| `inference_requests_total` | Counter | Total jumlah request ke model |
| `predictions_diabetes_total` | Counter | Total prediksi positif diabetes |
| `predictions_sehat_total` | Counter | Total prediksi negatif (sehat) |
| `inference_errors_total` | Counter | Total error pada sistem inference |
| `system_cpu_usage_percentage` | Gauge | Persentase penggunaan CPU |
| `system_memory_usage_bytes` | Gauge | Penggunaan memori dalam bytes |
| `deployed_model_version` | Gauge | Versi model yang aktif |
| `active_inference_sessions` | Gauge | Jumlah sesi inferensi aktif |
| `inference_latency_seconds` | Histogram | Distribusi lama waktu inferensi |
| `data_drift_p_value` | Summary | Nilai p-value untuk deteksi data drift |

#### Alerting di Grafana:
- **CPU Usage High** — Peringatan saat penggunaan CPU > 85%
- **Active Sessions High** — Peringatan saat sesi aktif > 15
- **Inference Errors** — Peringatan saat terjadi error pada inference

---

## 🛠️ Teknologi yang Digunakan

| Teknologi | Kegunaan |
|-----------|----------|
| **Python 3** | Bahasa pemrograman utama |
| **Pandas & NumPy** | Manipulasi dan analisis data |
| **Scikit-learn** | Preprocessing & Machine Learning |
| **Matplotlib & Seaborn** | Visualisasi data |
| **MLflow** | Experiment tracking dan model registry |
| **DagsHub** | Remote tracking dan kolaborasi MLflow |
| **Flask** | REST API untuk serving model |
| **Prometheus** | Monitoring dan pengumpulan metrik |
| **Prometheus Client** | Eksportir metrik dari aplikasi Python |
| **Grafana** | Dashboard visualisasi dan alerting |
| **Docker** | Containerisasi aplikasi |

---

## 🚀 Cara Menjalankan Proyek

### Prasyarat
- Python 3.8+
- pip (Python package manager)
- (Opsional) Docker & Docker Compose untuk monitoring stack

### 1. Clone Repository
```bash
git clone https://github.com/ardianx0/Eksperimen_SML_Ardian-Gymnastiar.git
cd Eksperimen_SML_Ardian-Gymnastiar
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Preprocessing Data
```bash
python preprocessing/automate_Ardian-Gymnastiar.py
```

### 4. Training Model
```bash
cd Membangun_model
pip install -r requirements.txt
python modelling.py
```

### 5. Hyperparameter Tuning
```bash
python modelling_tuning.py
```

### 6. Monitoring Stack

#### Jalankan Prometheus Exporter
```bash
cd "Monitoring dan Logging"
pip install prometheus_client flask
python prometheus_exporter.py
```

#### Jalankan Flask API
```bash
python inference.py
```

#### Prometheus
Pastikan `prometheus.yml` sudah terkonfigurasi, lalu jalankan:
```bash
prometheus --config.file=prometheus.yml
```

#### Grafana
Akses Grafana di `http://localhost:3000` dan konfigurasikan:
- **Data Source:** Prometheus (`http://localhost:9090`)
- **Import Dashboard** atau buat panel sesuai metrik yang tersedia

---

## 📜 Sertifikat

🔗 **[Lihat Sertifikat Kelulusan Dicoding](https://www.dicoding.com/certificates/NVP7NV93VZR0)**

> **Nama:** Ardian Gymnastiar  
> **Username Dicoding:** `ardian_g8`  
> **Program:** PIJAK in Collaboration with IBM SkillsBuild and Dicoding  
> **Kelas:** Membangun Sistem Machine Learning

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan submission akhir kelas **"Membangun Sistem Machine Learning"** — PIJAK × IBM SkillsBuild × Dicoding.

---

<p align="center">
  Creating by <strong>Ardian Gymnastiar</strong> — <code>ardian_g8</code>
</p>