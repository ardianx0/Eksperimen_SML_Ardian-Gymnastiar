import time
import random
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary

REQUEST_COUNT = Counter('inference_requests_total', 'Total jumlah request ke model')
PREDICTION_DIABETES = Counter('predictions_diabetes_total', 'Total prediksi positif diabetes')
PREDICTION_SEHAT = Counter('predictions_sehat_total', 'Total prediksi negatif (sehat)')
ERROR_COUNT = Counter('inference_errors_total', 'Total error pada sistem inference')

CPU_USAGE = Gauge('system_cpu_usage_percentage', 'Persentase penggunaan CPU')
MEMORY_USAGE = Gauge('system_memory_usage_bytes', 'Penggunaan memori dalam bytes')
MODEL_VERSION = Gauge('deployed_model_version', 'Versi model yang aktif', ['version'])
ACTIVE_SESSIONS = Gauge('active_inference_sessions', 'Jumlah sesi inferensi aktif')

REQUEST_LATENCY = Histogram('inference_latency_seconds', 'Distribusi lama waktu inferensi')
DATA_DRIFT_SCORE = Summary('data_drift_p_value', 'Nilai p-value untuk deteksi data drift')

def simulate_metrics():
    MODEL_VERSION.labels(version='1.0.0').set(1)
    while True:
        REQUEST_COUNT.inc()
        if random.random() > 0.35:
            PREDICTION_SEHAT.inc()
        else:
            PREDICTION_DIABETES.inc()
            
        if random.random() < 0.02:
            ERROR_COUNT.inc()

        # Simulasi angka fluktuatif naik-turun agar memicu alert
        CPU_USAGE.set(random.uniform(20.0, 95.0)) 
        MEMORY_USAGE.set(random.randint(500000000, 1500000000))
        ACTIVE_SESSIONS.set(random.randint(1, 20)) 
        DATA_DRIFT_SCORE.observe(random.uniform(0.01, 0.99))
        
        with REQUEST_LATENCY.time():
            time.sleep(random.uniform(0.05, 0.5))

if __name__ == '__main__':
    print("Memulai Prometheus Exporter di port 8000...")
    start_http_server(8000)
    simulate_metrics()