# Dans src/app.py
from flask import Flask, request, jsonify, send_from_directory
import os
import sys

# Ajout du chemin courant pour trouver predict_ai
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importation de ton cerveau
import predict_ai

app = Flask(__name__)

# Dossier temporaire pour stocker les images reçues
UPLOAD_FOLDER = '/tmp' if os.name != 'nt' else 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return "Service IA Textile en ligne 🟢"

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Aucune image envoyée"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "Nom de fichier vide"}), 400

    try:
        # 1. Sauvegarde temporaire de l'image
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # 2. Appel au cerveau (predict_ai.py)
        result = predict_ai.analyze_image(file_path)

        # 3. Retourne le JSON
        return jsonify({
            "success": True,
            "data": result,
            # Pour récupérer les fichiers générés, on renverra les URLs plus tard
            # Pour l'instant, on renvoie les noms
            "files": {
                "image": result['analyzed_image'],
                "audio": result['audio_file'],
                "pdf": result['pdf_file']
            }
        })

    except Exception as e:
        print(f"❌ Erreur Serveur: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# Route pour télécharger les fichiers générés (images, pdf, audio)
@app.route('/files/<filename>')
def get_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    # Écoute sur le port 80 pour Azure/Docker
    app.run(host='0.0.0.0', port=80, debug=True)