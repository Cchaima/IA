import os
from ultralytics import YOLO
import shutil
import torch
import mlflow
from drift import create_reference

# --- CONFIGURATION DES CHEMINS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(BASE_DIR, "data", "dataset.yaml")
MODEL_DIR = os.path.join(BASE_DIR, "models")
EXPERIMENT_NAME = "mlops_textile_run"

# --- CONFIGURATION MATÉRIELLE ---
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
WORKERS = 0 
BATCH_SIZE = 2
EPOCHS = 200

def train():
    print(f"\n⚡ Lancement MLOps sur {DEVICE} (Batch: {BATCH_SIZE}, Epochs: {EPOCHS})...")

    mlflow.set_experiment("Projet_Textile_IA")
    
    with mlflow.start_run() as run:
        print(f"✅ MLflow Run ID: {run.info.run_id}")
        
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("device", DEVICE)
        mlflow.log_param("model_type", "YOLOv8n")

        # 1. Charger le modèle
        model = YOLO("yolov8n.pt")

        # 2. Entraînement
        results = model.train(
            data=DATA_YAML,
            imgsz=640,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            project=MODEL_DIR,
            name=EXPERIMENT_NAME,
            device=DEVICE,
            workers=WORKERS,
            exist_ok=True 
        )

        # 3. Sauvegarde du modèle
        trained_model_path = os.path.join(MODEL_DIR, EXPERIMENT_NAME, "weights", "best.pt")
        final_model_path = os.path.join(MODEL_DIR, "best.pt")

        if os.path.exists(trained_model_path):
            shutil.copy(trained_model_path, final_model_path)
            print(f"✅ Modèle sauvegardé : {final_model_path}")
            
            # Envoi du modèle vers MLflow
            mlflow.log_artifact(final_model_path)
            
            # --- CORRECTION ICI : Nettoyage des noms de métriques ---
            # On enlève les parenthèses (B) qui font planter MLflow
            if hasattr(results, 'results_dict'):
                clean_metrics = {}
                for key, value in results.results_dict.items():
                    # On remplace '(' par '_' et on enlève ')'
                    new_key = key.replace('(', '_').replace(')', '')
                    clean_metrics[new_key] = value
                
                mlflow.log_metrics(clean_metrics)
                print("✅ Métriques nettoyées et envoyées vers MLflow")

        else:
            print("❌ Erreur : L'entraînement n'a pas généré de fichier best.pt")
            exit(1)
        
        # Mise à jour drift
        create_reference()

if __name__ == "__main__":
    train()