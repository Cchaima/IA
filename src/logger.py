import csv
import os
from datetime import datetime

# Chemin du fichier de logs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, "logs", "inference_history.csv")

def log_prediction(filename, defect_found, confidence, drift_status):
    """Enregistre une prédiction dans le CSV"""
    
    # Créer le dossier logs s'il n'existe pas
    if not os.path.exists(os.path.dirname(LOG_FILE)):
        os.makedirs(os.path.dirname(LOG_FILE))

    # Si le fichier n'existe pas, on crée l'en-tête
    file_exists = os.path.isfile(LOG_FILE)
    
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Image", "Defect", "Confidence", "Drift_Status"])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename,
            defect_found,
            confidence,
            drift_status
        ])
