import os
from ultralytics import YOLO
import shutil
import torch
from drift import create_reference

# --- CONFIGURATION DES CHEMINS (Relatifs pour que ça marche partout) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(BASE_DIR, "data", "dataset.yaml")
MODEL_DIR = os.path.join(BASE_DIR, "models")
EXPERIMENT_NAME = "mlops_textile_run"

# --- CONFIGURATION MATÉRIELLE ---
# Si CUDA (GPU) est dispo, on l'utilise, sinon CPU (pour GitHub Actions)
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
WORKERS = 0 # Important pour Windows et pour éviter les erreurs de mémoire
BATCH_SIZE = 2
EPOCHS = 2 # On met peu d'époques pour le test MLOps (sinon GitHub va timeout)

def train():
    print(f"\n⚡ Lancement MLOps sur {DEVICE} (Batch: {BATCH_SIZE}, Epochs: {EPOCHS})...")

    # 1. Charger le modèle
    model = YOLO("yolov8n.pt")

    # 2. Entraînement
    model.train(
        data=DATA_YAML,
        imgsz=640,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        project=MODEL_DIR,
        name=EXPERIMENT_NAME,
        device=DEVICE,
        workers=WORKERS,
        exist_ok=True # Écrase l'ancien test pour économiser de la place
    )

    # 3. Récupérer le modèle entraîné (best.pt) et le mettre à la racine de models/
    trained_model_path = os.path.join(MODEL_DIR, EXPERIMENT_NAME, "weights", "best.pt")
    final_model_path = os.path.join(MODEL_DIR, "best.pt")

    if os.path.exists(trained_model_path):
        shutil.copy(trained_model_path, final_model_path)
        print(f"✅ Modèle sauvegardé et prêt pour le déploiement : {final_model_path}")
    else:
        print("❌ Erreur : L'entraînement n'a pas généré de fichier best.pt")
        exit(1)
        # On met à jour la référence Drift
    create_reference()

if __name__ == "__main__":
    train()
