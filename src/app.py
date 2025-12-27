import os
import sys
from flask import Flask, request, jsonify, render_template, send_from_directory

# --- CONFIGURATION DU "GPS" (Chemins absolus) ---
# 1. On trouve où est ce fichier (src/app.py)
base_dir = os.path.dirname(os.path.abspath(__file__))

# 2. On ajoute ce dossier au chemin système pour trouver predict_api.py
sys.path.append(base_dir)

# 3. On pointe vers le dossier templates situé DANS src
template_dir = os.path.join(base_dir, 'templates')

# --- IMPORTS LOCAUX ---
# Importation de ton cerveau (doit être après le sys.path.append si besoin)
import predict_api

# --- CRÉATION DE L'APP ---
# On force Flask à utiliser le dossier templates qu'on a défini juste au-dessus
app = Flask(__name__, template_folder=template_dir)

# Dossier temporaire pour stocker les images reçues
UPLOAD_FOLDER = '/tmp' if os.name != 'nt' else 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    # --- DEBUG : ON REGARDE CE QUI ARRIVE ---
    print("\n📩 REQUÊTE REÇUE !")
    print(f"📁 Fichiers reçus : {request.files}")
    if request.files:
        print(f"🔑 Clés détectées : {list(request.files.keys())}")
    
    # 1. Vérification souple (On accepte n'importe quel nom de fichier)
    if not request.files:
        print("❌ ERREUR : Aucun fichier dans la requête")
        return jsonify({"success": False, "error": "Aucun fichier reçu (vérifiez l'enctype du form)"}), 400
    
    # 2. On prend le PREMIER fichier, quel que soit son nom ('image', 'file', etc.)
    first_key = next(iter(request.files))
    file = request.files[first_key]
    print(f"✅ Fichier trouvé : {file.filename} (sous la clé '{first_key}')")

    if file.filename == '':
        return jsonify({"success": False, "error": "Nom de fichier vide"}), 400

    try:
        # 1. Sauvegarde temporaire
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # 2. Appel au cerveau
        result = predict_api.analyze_image(file_path)

        # 3. Retour JSON
        return jsonify({
            "success": True,
            "data": result,
            "files": {
                "image": result.get('analyzed_image'),
                "audio": result.get('audio_file'),
                "pdf": result.get('pdf_file')
            }
        })

    except Exception as e:
        print(f"❌ Erreur Serveur: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# Route pour télécharger les fichiers générés (images, pdf, audio)
@app.route('/files/<filename>')
def get_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    # Écoute sur le port 80 pour Azure/Docker
    app.run(host='0.0.0.0', port=80, debug=True)