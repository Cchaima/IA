import cv2
import numpy as np
import json
import os
import glob

# Chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIFT_FILE = os.path.join(BASE_DIR, "drift", "reference.json")
DATA_DIR = os.path.join(BASE_DIR, "data", "images")

def get_image_properties(image_path):
    """Calcule la luminosité et le flou d'une image"""
    img = cv2.imread(image_path)
    if img is None: return None
    
    # 1. Luminosité (Moyenne des pixels gris)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    # 2. Flou (Variance du Laplacien)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return brightness, blur

def create_reference():
    """Appelé par train.py : Crée la 'Carte d'Identité' des données saines"""
    print("📊 Calcul du Drift de référence...")
    
    images = glob.glob(os.path.join(DATA_DIR, "*.jpg")) + glob.glob(os.path.join(DATA_DIR, "*.png"))
    
    if not images:
        print("⚠️ Pas d'images pour le drift !")
        return

    bright_vals = []
    blur_vals = []

    for img_path in images:
        res = get_image_properties(img_path)
        if res:
            bright_vals.append(res[0])
            blur_vals.append(res[1])

    # On sauvegarde les stats "Normales"
    reference_data = {
        "mean_brightness": float(np.mean(bright_vals)),
        "min_brightness": float(np.min(bright_vals)) - 20, # Marge de tolérance
        "max_brightness": float(np.max(bright_vals)) + 20,
        "mean_blur": float(np.mean(blur_vals)),
        "blur_threshold": float(np.min(blur_vals)) - 10 # Si c'est moins net que le pire des cas
    }

    if not os.path.exists(os.path.dirname(DRIFT_FILE)):
        os.makedirs(os.path.dirname(DRIFT_FILE))

    with open(DRIFT_FILE, "w") as f:
        json.dump(reference_data, f)
    
    print(f"✅ Référence Drift sauvegardée dans {DRIFT_FILE}")

def check_drift(image_path):
    """Appelé par predict_api.py : Vérifie si l'image est OK"""
    if not os.path.exists(DRIFT_FILE):
        return {"status": "OK", "message": "Pas de référence Drift"}

    with open(DRIFT_FILE, "r") as f:
        ref = json.load(f)

    b_val, blur_val = get_image_properties(image_path)

    # Vérification
    drift_report = []
    
    # Test Luminosité
    if b_val < ref["min_brightness"]:
        drift_report.append("DRIFT: Image trop sombre")
    elif b_val > ref["max_brightness"]:
        drift_report.append("DRIFT: Image trop claire")

    # Test Flou
    if blur_val < ref["blur_threshold"]:
        drift_report.append("DRIFT: Image floue")

    if drift_report:
        return {"status": "DRIFT_DETECTED", "details": drift_report}
    
    return {"status": "OK"}

if __name__ == "__main__":
    # Pour tester, lance ce script seul : il va créer le fichier json
    create_reference()
