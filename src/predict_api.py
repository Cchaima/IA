import os
import cv2
from ultralytics import YOLO
from gtts import gTTS
from collections import Counter

# --- BLOC DE CORRECTION DES IMPORTS (Azure vs Local) ---
try:
    # Si on lance depuis la racine du projet
    from src.drift import check_drift
    from src.logger import log_prediction
except ModuleNotFoundError:
    # Si on lance depuis le dossier src (comme sur Azure/Docker)
    from drift import check_drift
    from logger import log_prediction
# -------------------------------------------------------

# Chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

# Ta base de connaissances experte
RECOMMANDATIONS_EXPERTES = {
    "Trou": "Arrêtez la machine immédiatement. Vérifiez aiguille cassée.",
    "Tache": "Marquez la zone pour nettoyage. Vérifiez fuite d'huile.",
    "Couture": "Défaut d'assemblage. Vérifiez l'alignement.",
    "Pli": "Problème tension tissu. Vérifiez rouleaux.",
    "fil": "Fil cassé. Vérifiez bobines.",
    "Dformation": "Distorsion thermique. Vérifiez température.",
    "default": "Inspection manuelle requise."
}

# Chargement global du modèle
try:
    print(f"⏳ Chargement du modèle : {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✅ Modèle chargé !")
except Exception as e:
    model = None
    print(f"❌ Erreur modèle : {e}")

def analyze_image(image_path):
    # 1. Vérification du DRIFT (Dérive)
    drift_result = check_drift(image_path)
    
    if drift_result["status"] == "DRIFT_DETECTED":
        print(f"⚠️ ALERTE DRIFT : {drift_result['details']}")

    if model is None:
        return {"error": "Modèle non chargé"}

    # 2. Prédiction IA
    results = model.predict(image_path, conf=0.30, save=False)
    
    # 3. Traitement de l'image
    result = results[0]
    img_with_boxes = result.plot()
    
    output_dir = os.path.dirname(image_path)
    filename = "analyzed_" + os.path.basename(image_path)
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, img_with_boxes)

    # 4. Analyse des résultats
    detected_objects = []
    defects_details = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]
        conf = float(box.conf[0])
        detected_objects.append(name)
        
        defects_details.append({
            "name": name,
            "confidence": round(conf * 100, 1),
            "severity": "CRITIQUE" if name in ["Trou", "Dformation"] else "MAJEUR",
            "location": "Détecté par IA"
        })

    # 5. Génération du texte du rapport
    if not detected_objects:
        report_text = "Analyse terminée. Tissu conforme."
    else:
        counts = Counter(detected_objects)
        report_text = "Attention. Anomalies détectées : "
        conseils = []
        for defect, count in counts.items():
            report_text += f"{count} {defect}. "
            rec = RECOMMANDATIONS_EXPERTES.get(defect, RECOMMANDATIONS_EXPERTES["default"])
            if rec not in conseils:
                conseils.append(rec)
        report_text += " Conseil : " + " ".join(conseils)

    # 6. Génération Audio
    audio_filename = "rapport.mp3"
    audio_path = os.path.join(output_dir, audio_filename)
    try:
        tts = gTTS(text=report_text, lang='fr', slow=False)
        tts.save(audio_path)
    except:
        pass

    # =================================================================
    # 🚨 MONITORING MLOps
    # =================================================================
    
    # Déterminer le statut Drift pour le log
    drift_status_log = "ALERTE" if drift_result["status"] == "DRIFT_DETECTED" else "OK"
    
    # Déterminer le défaut principal et sa confiance
    main_defect = "R.A.S"
    main_conf = 0.0
    
    if defects_details:
        main_defect = defects_details[0]['name']
        main_conf = defects_details[0]['confidence']
        
    # Enregistrement dans le fichier CSV via logger.py
    log_prediction(
        filename=os.path.basename(image_path),
        defect_found=main_defect,
        confidence=main_conf,
        drift_status=drift_status_log
    )
    # =================================================================

    return {
        "analyzed_image": filename,
        "audio_file": audio_filename,
        "text": report_text,
        "defects": defects_details,
        "pdf_file": "#"
    }
