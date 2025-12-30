import optuna
import mlflow
from ultralytics import YOLO

# Configuration MLflow
mlflow.set_experiment("Optimisation_Optuna_Avancee")

def objective(trial):
    """
    Fonction d'optimisation améliorée.
    Teste plus de paramètres pour trouver la recette parfaite.
    """
    
    # --- 1. LES NOUVEAUX PARAMÈTRES À TESTER ---
    
    # Learning rate initial (Vitesse de départ)
    lr0 = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)
    
    # Learning rate final (Vitesse à la fin, fraction de lr0)
    lrf = trial.suggest_float("lrf", 0.01, 1.0)
    
    # Momentum (L'élan : aide à ne pas rester bloqué)
    momentum = trial.suggest_float("momentum", 0.6, 0.98)
    
    # Weight decay (Pénalité pour éviter d'apprendre par cœur)
    weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.001)
    
    # Choix de l'optimiseur
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "AdamW"])

    print(f"\n🚀 Essai {trial.number} démarré avec : lr={lr0:.4f}, mom={momentum:.2f}, opt={optimizer}")

    # --- 2. CHARGEMENT ET ENTRAÎNEMENT ---
    
    model = YOLO("yolov8n.pt") 

    results = model.train(
        data="/app/data/dataset.yaml",  # Chemin Docker corrigé
        epochs=30,             # 30 suffit souvent pour comparer (50 c'est long sur CPU)
        patience=10,           # STOP si pas d'amélioration pendant 10 époques (Gain de temps !)
        
        # Injection des paramètres choisis par Optuna
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        optimizer=optimizer,
        
        project="runs/optuna_v2",
        name=f"trial_{trial.number}",
        imgsz=640,
        batch=3,               # Augmenté un peu si possible, sinon remets 2
        verbose=False
    )

    # --- 3. RÉCUPÉRATION DU SCORE ---
    
    # On vise la précision moyenne globale (mAP50-95) car c'est plus robuste
    metrics = results.box.map  # map50-95
    
    # --- 4. ENREGISTREMENT MLFLOW ---
    with mlflow.start_run(nested=True):
        mlflow.log_params(trial.params)
        mlflow.log_metric("mAP50-95", metrics)
        mlflow.log_metric("mAP50", results.box.map50)
    
    return metrics

if __name__ == "__main__":
    # Création de l'étude
    study = optuna.create_study(direction="maximize")
    
    print("⏳ Démarrage de l'optimisation avancée...")
    # On lance 10 essais (Prépare-toi, ça va prendre du temps sur CPU !)
    study.optimize(objective, n_trials=10)

    print("\n------------------------------------------------")
    print("🏆 LES PARAMÈTRES GAGNANTS SONT :")
    print(study.best_params)
    print("------------------------------------------------")