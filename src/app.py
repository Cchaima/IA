import os
import sys
import pandas as pd  # <--- NOUVEAU : Pour lire les logs
from flask import Flask, request, jsonify, render_template, send_from_directory, render_template_string

# --- CONFIGURATION DU "GPS" (Chemins absolus) ---
# 1. On trouve où est ce fichier (src/app.py)
base_dir = os.path.dirname(os.path.abspath(__file__))

# 2. On ajoute ce dossier au chemin système pour trouver predict_api.py
sys.path.append(base_dir)

# 3. On pointe vers le dossier templates situé DANS src
template_dir = os.path.join(base_dir, 'templates')

# --- IMPORTS LOCAUX ---
import predict_api

# --- CRÉATION DE L'APP ---
app = Flask(__name__, template_folder=template_dir)

# Dossier temporaire pour stocker les images reçues
UPLOAD_FOLDER = '/tmp' if os.name != 'nt' else 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# =========================================================
# 🏠 ROUTE D'ACCUEIL
# =========================================================
@app.route('/')
def home():
    return render_template('index.html')

# =========================================================
# 📊 ROUTE MONITORING (DASHBOARD MLOps)
# =========================================================
@app.route('/dashboard')
def dashboard():
    # Le fichier de logs est dans un dossier frère de 'src'
    log_path = os.path.join(base_dir, '..', 'logs', 'inference_history.csv')
    
    stats = {
        "total": 0,
        "top_defect": "En attente...",
        "avg_conf": 0,
        "drift_alerts": 0
    }
    recent_logs = []

    if os.path.exists(log_path):
        try:
            # Lecture du CSV avec Pandas
            df = pd.read_csv(log_path)
            
            # Calcul des KPIs
            stats["total"] = len(df)
            
            if not df.empty:
                # Défaut le plus fréquent
                if 'Defect' in df.columns:
                    stats["top_defect"] = df['Defect'].mode()[0]
                
                # Confiance moyenne
                if 'Confidence' in df.columns:
                    stats["avg_conf"] = round(df['Confidence'].mean(), 1)
                
                # Nombre d'alertes Drift
                if 'Drift_Status' in df.columns:
                    stats["drift_alerts"] = len(df[df['Drift_Status'] == "ALERTE"])
                
                # Récupérer les 10 dernières lignes pour le tableau
                recent_logs = df.tail(10).to_dict(orient='records')
                recent_logs.reverse() # Du plus récent au plus vieux
        except Exception as e:
            print(f"Erreur lecture logs: {e}")

    # Template HTML intégré (pour éviter de créer un fichier dashboard.html séparé)
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Monitoring MLOps - Texelai</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; padding: 40px; background: #f0f2f5; color: #333; }
            h1 { color: #2c3e50; margin-bottom: 30px; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            
            .kpi-container { display: flex; gap: 20px; margin-bottom: 40px; flex-wrap: wrap; }
            .kpi-card { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); flex: 1; min-width: 200px; text-align: center; transition: transform 0.2s; }
            .kpi-card:hover { transform: translateY(-5px); }
            .kpi-value { font-size: 2.5em; font-weight: bold; color: #2c3e50; margin-bottom: 5px; }
            .kpi-label { color: #7f8c8d; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }
            
            table { width: 100%; border-collapse: separate; border-spacing: 0; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
            th, td { padding: 15px 20px; text-align: left; border-bottom: 1px solid #eee; }
            th { background-color: #34495e; color: white; font-weight: 600; text-transform: uppercase; font-size: 0.85em; }
            tr:last-child td { border-bottom: none; }
            tr:hover { background-color: #f8f9fa; }
            
            .status-badge { padding: 5px 10px; border-radius: 20px; font-size: 0.85em; font-weight: bold; }
            .status-ok { background-color: #d4edda; color: #155724; }
            .status-alert { background-color: #f8d7da; color: #721c24; }
            
            .nav-link { display: inline-block; margin-top: 20px; text-decoration: none; color: #3498db; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>🎛️ Dashboard de Monitoring - Texelai</h1>
        
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-value">{{ stats.total }}</div>
                <div class="kpi-label">Images Analysées</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{{ stats.top_defect }}</div>
                <div class="kpi-label">Défaut Fréquent</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{{ stats.avg_conf }}%</div>
                <div class="kpi-label">Confiance Moyenne</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value" style="color: {{ 'red' if stats.drift_alerts > 0 else '#27ae60' }}">
                    {{ stats.drift_alerts }}
                </div>
                <div class="kpi-label">Alertes Drift</div>
            </div>
        </div>

        <h2>🕵️ Historique des Inférences (Temps Réel)</h2>
        <table>
            <thead>
                <tr>
                    <th>Date & Heure</th>
                    <th>Image</th>
                    <th>Résultat IA</th>
                    <th>Confiance</th>
                    <th>Statut Drift</th>
                </tr>
            </thead>
            <tbody>
                {% for log in logs %}
                <tr>
                    <td>{{ log.Timestamp }}</td>
                    <td>{{ log.Image }}</td>
                    <td><strong>{{ log.Defect }}</strong></td>
                    <td>{{ log.Confidence }}%</td>
                    <td>
                        <span class="status-badge {{ 'status-alert' if log.Drift_Status == 'ALERTE' else 'status-ok' }}">
                            {{ log.Drift_Status }}
                        </span>
                    </td>
                </tr>
                {% else %}
                <tr>
                    <td colspan="5" style="text-align:center; padding: 20px;">Aucune donnée pour le moment. Lancez une prédiction !</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        <br>
        <a href="/" class="nav-link">← Retour à l'analyse</a>
    </body>
    </html>
    """
    
    return render_template_string(html_template, stats=stats, logs=recent_logs)

# =========================================================
# 🧠 ROUTE PRÉDICTION
# =========================================================
@app.route('/predict', methods=['POST'])
def predict_route():
    # --- DEBUG : ON REGARDE CE QUI ARRIVE ---
    print("\n📩 REQUÊTE REÇUE !")
    
    if not request.files:
        return jsonify({"success": False, "error": "Aucun fichier reçu"}), 400
    
    # On prend le PREMIER fichier, quel que soit son nom
    first_key = next(iter(request.files))
    file = request.files[first_key]
    print(f"✅ Fichier trouvé : {file.filename}")

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

# Route pour télécharger les fichiers générés
@app.route('/files/<filename>')
def get_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    # Écoute sur le port 80 pour Azure/Docker
    app.run(host='0.0.0.0', port=80, debug=True)
