# Dans src/predict_ai.py
import os
import cv2
import time
import random
from ultralytics import YOLO
from gtts import gTTS
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF

# --- CONFIGURATION DU CHEMIN DU MODÈLE ---
# On remonte d'un niveau (..) pour trouver le dossier 'model'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")

print(f"⏳ Chargement du modèle depuis : {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    print("✅ Modèle YOLO chargé !")
except Exception as e:
    print(f"❌ ERREUR CRITIQUE : Impossible de charger le modèle au chemin {MODEL_PATH}")
    print(f"❌ Erreur détail : {e}")
    # On ne quitte pas brutalement ici pour éviter de tuer le conteneur, mais ça ne marchera pas.
    model = None

# --- BASE DE CONNAISSANCES ---
KNOWLEDGE_BASE = {
    "Trou": {"cause": "aiguille cassée", "action": "Changez l'aiguille", "severity": "CRITIQUE", "conseil": "Vérifiez plaque aiguille."},
    "Tache": {"cause": "huile", "action": "Nettoyez au solvant", "severity": "MAJEUR", "conseil": "Vérifiez réservoir."},
    "Couture": {"cause": "tension fil", "action": "Réglez tension", "severity": "MINEUR", "conseil": "Testez sur chute."},
    "Pli": {"cause": "alimentation tissu", "action": "Ajustez rouleaux", "severity": "MAJEUR", "conseil": "Vérifiez planéité."},
    "fil": {"cause": "rupture", "action": "Renouez fil", "severity": "MINEUR", "conseil": "Contrôlez guide-fils."},
    "Dformation": {"cause": "surchauffe", "action": "Réduisez temp", "severity": "CRITIQUE", "conseil": "Vérifiez calandre."}
}

def generate_expert_text(defects):
    """Génère le texte pour l'audio"""
    if not defects:
        return "Analyse terminée. Tissu 100% conforme. Production optimale."
    
    # Tri par gravité
    sev_order = {"CRITIQUE": 3, "MAJEUR": 2, "MINEUR": 1, "MOYEN": 1}
    sorted_defects = sorted(defects, key=lambda x: sev_order.get(x['severity'], 0), reverse=True)
    main_defect = sorted_defects[0]
    
    count = len(defects)
    info = KNOWLEDGE_BASE.get(main_defect['name'], {"cause": "inconnue", "action": "inspecter", "conseil": ""})
    
    intro = "Alerte qualité." if main_defect['severity'] == "CRITIQUE" else "Défaut détecté."
    constat = f"J'ai trouvé {count} défauts, principalement {main_defect['name']}."
    action = f"Cause probable : {info['cause']}. Action : {info['action']}."
    
    return f"{intro} {constat} {action}"

def analyze_image(image_path):
    """Fonction principale appelée par app.py"""
    if model is None:
        raise Exception("Le modèle n'est pas chargé.")

    # 1. Prédiction
    results = model.predict(image_path, conf=0.30, save=False, verbose=False)
    
    output_dir = os.path.dirname(image_path)
    base_name = os.path.basename(image_path)
    analyzed_name = f"analyzed_{base_name}"
    save_path_img = os.path.join(output_dir, analyzed_name)
    
    detected_defects = []

    # 2. Traitement des résultats
    for result in results:
        # Sauvegarde image avec cadres
        img_with_boxes = result.plot()
        cv2.imwrite(save_path_img, img_with_boxes)
        
        for box in result.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            conf = float(box.conf[0])
            
            # Info expert
            kb_info = KNOWLEDGE_BASE.get(name, {"severity": "MOYEN"})
            
            # Position
            x_center = box.xywh[0][0]
            w = result.orig_shape[1]
            loc = "Gauche" if x_center < w/3 else "Droite" if x_center > 2*w/3 else "Centre"

            detected_defects.append({
                "name": name,
                "confidence": round(conf*100, 2),
                "severity": kb_info['severity'],
                "location": f"Zone {loc}"
            })

    # Si pas de défaut, on sauvegarde quand même l'image
    if not os.path.exists(save_path_img):
        img = cv2.imread(image_path)
        if img is not None: cv2.imwrite(save_path_img, img)

    # 3. Génération Audio & PDF
    report_text = generate_expert_text(detected_defects)
    
    # Audio
    audio_name = os.path.splitext(base_name)[0] + ".mp3"
    audio_path = os.path.join(output_dir, audio_name)
    try:
        tts = gTTS(text=report_text, lang='fr', slow=False)
        tts.save(audio_path)
    except Exception as e:
        print(f"⚠️ Erreur Audio: {e}")

    # PDF (Version simplifiée pour éviter les erreurs de code long)
    pdf_name = os.path.splitext(base_name)[0] + ".pdf"
    pdf_path = os.path.join(output_dir, pdf_name)
    try:
        c = canvas.Canvas(pdf_path, pagesize=A4)
        c.drawString(100, 800, f"Rapport: {base_name}")
        c.drawString(100, 780, f"Résultat: {report_text}")
        if os.path.exists(save_path_img):
            c.drawImage(save_path_img, 100, 500, width=400, height=300, preserveAspectRatio=True)
        c.save()
    except Exception as e:
        print(f"⚠️ Erreur PDF: {e}")

    return {
        "defects": detected_defects,
        "analyzed_image": analyzed_name,
        "audio_file": audio_name,
        "pdf_file": pdf_name,
        "text": report_text
    }