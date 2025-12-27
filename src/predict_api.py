import os
import cv2
from ultralytics import YOLO
from gtts import gTTS
from collections import Counter
from src.drift import check_drift

# Chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

# Ta base de connaissances experte (C'est top !)
RECOMMANDATIONS_EXPERTES = {
    "Trou": "Arrêtez la machine immédiatement. Vérifiez aiguille cassée.",
    "Tache": "Marquez la zone pour nettoyage. Vérifiez fuite d'huile.",
    "Couture": "Défaut d'assemblage. Vérifiez l'alignement.",
    "Pli": "Problème tension tissu. Vérifiez rouleaux.",
    "fil": "Fil cassé. Vérifiez bobines.",
    "Dformation": "Distorsion thermique. Vérifiez température.",
    "default": "Inspection manuelle requise."
}

# Chargement global du modèle (une seule fois au démarrage)
try:
    print(f"⏳ Chargement du modèle : {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✅ Modèle chargé !")
except Exception as e:
    model = None
    print(f"❌ Erreur modèle : {e}")

def analyze_image(image_path):
    drift_result = check_drift(image_path)
    
    if drift_result["status"] == "DRIFT_DETECTED":
        # On peut choisir de bloquer ou juste d'avertir
        print(f"⚠️ ALERTE DRIFT : {drift_result['details']}")
    if model is None:
        return {"error": "Modèle non chargé"}

    # 1. Prédiction
    results = model.predict(image_path, conf=0.30, save=False)
    
    # 2. Dessiner les boîtes sur l'image
    result = results[0]
    img_with_boxes = result.plot()
    
    # Sauvegarder l'image analysée
    output_dir = os.path.dirname(image_path)
    filename = "analyzed_" + os.path.basename(image_path)
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, img_with_boxes)

    # 3. Analyse des défauts pour le rapport
    detected_objects = []
    defects_details = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]
        conf = float(box.conf[0])
        detected_objects.append(name)
        
        # Pour l'affichage HTML
        defects_details.append({
            "name": name,
            "confidence": round(conf * 100, 1),
            "severity": "CRITIQUE" if name in ["Trou", "Dformation"] else "MAJEUR",
            "location": "Détecté par IA"
        })

    # 4. Génération du texte (Ton algo)
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

    # 5. Génération Audio
    audio_filename = "rapport.mp3"
    audio_path = os.path.join(output_dir, audio_filename)
    try:
        tts = gTTS(text=report_text, lang='fr', slow=False)
        tts.save(audio_path)
    except:
        pass

    return {
        "analyzed_image": filename,
        "audio_file": audio_filename,
        "text": report_text,
        "defects": defects_details,
        "pdf_file": "#" # À faire plus tard si tu veux
    }
