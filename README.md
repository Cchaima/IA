🏭 Texelai - Contrôle Qualité Textile Intelligent 

Texelai est une solution de vision par ordinateur automatisée destinée à l'industrie textile. Ce projet implémente un pipeline MLOps complet (CI/CD/CT) pour détecter les défauts de fabrication (trous, taches, déchirures) en utilisant YOLOv8.

L'architecture repose sur l'intégration continue, le versioning des données, l'optimisation automatique des hyperparamètres et le déploiement Cloud.

🚀 Fonctionnalités Clés

👁️ Vision par Ordinateur : Détection d'objets en temps réel via YOLOv8 (custom trained).

🧠 Optimisation Automatique : Recherche des meilleurs hyperparamètres avec Optuna.

📊 Suivi d'Expériences : Tracking complet des métriques et artefacts via MLflow.

🗂️ Versioning des Données : Gestion du dataset lourd avec DVC sur Azure Blob Storage.

☁️ Déploiement Continu (CI/CD) : Pipeline GitHub Actions automatisé vers Azure App Service.

🐳 Conteneurisation : Architecture Dockerisée reproductible.

🔊 Alerte Vocale : Synthèse vocale immédiate pour l'opérateur (gTTS).

📈 Monitoring : Dashboard intégré pour surveiller la dérive des données (Data Drift).

📂 Architecture du Projet

Texelai/
├── .dvc/                  # Configuration DVC (lien vers Azure Storage)
├── .github/workflows/     # Pipelines CI/CD/CT pour l'automatisation
├── data/                  # Dossier des données (géré par DVC, ignoré par Git)
├── drift/                 # Scripts et références pour la détection de dérive
├── logs/                  # Historique des inférences
├── models/                # Stockage des modèles entraînés (.pt)
├── runs/                  # Logs d'entraînement MLflow et Optuna
├── src/
│   ├── app.py             # API Flask (Backend)
│   ├── predict_api.py     # Logique d'inférence et rapport expert
│   ├── train.py           # Script d'entraînement MLOps
│   ├── optimize.py        # Script d'optimisation Optuna
│   ├── drift.py           # Algorithme de détection de Drift
│   ├── logger.py          # Système de logging
│   └── templates/         # Interface Web (Frontend)
├── Dockerfile             # Configuration de l'image Docker de production
├── docker-compose.yml     # Orchestration locale
├── requirements.txt       # Dépendances Python
└── README.md              # Documentation du projet


🛠️ Installation et Exécution

1. Prérequis

Docker & Docker Compose

Compte Azure (pour le déploiement)

Python 3.9+ (pour le développement local)

2. Récupération des Données (DVC)

Les images lourdes ne sont pas sur GitHub. Nous utilisons DVC pour les télécharger depuis le stockage distant Azure.

# Installer DVC et le plugin Azure
pip install dvc dvc-azure

# Récupérer les données et le modèle
dvc pull


3. Lancer l'application (Docker)

La méthode recommandée pour lancer tout l'environnement :

docker-compose up --build


L'application sera accessible sur : http://localhost:5000

Le Dashboard de monitoring : http://localhost:5000/dashboard

🧠 Pipeline MLOps & Entraînement

Ce projet utilise une approche moderne pour garantir la performance du modèle.

🔹 1. Optimisation des Hyperparamètres (Optuna)

Plutôt que de choisir les paramètres au hasard, nous utilisons Optuna pour rechercher la meilleure configuration (Learning Rate, Momentum, Epochs).

# Lancer la recherche d'hyperparamètres dans le conteneur
docker exec -it le_conteneur_texelai python src/optimize.py


Optuna va lancer plusieurs "Trials", enregistrer les résultats dans MLflow et sélectionner le modèle champion.

🔹 2. Suivi des Expériences (MLflow)

Chaque entraînement est loggué dans MLflow. Cela permet de visualiser les courbes de perte, la matrice de confusion et de comparer les modèles (Parallel Coordinates Plot).

Pour visualiser l'interface MLflow (si lancée localement) :

mlflow ui


Accès via : http://localhost:5000

☁️ Déploiement Automatisé (CI/CD)

Le déploiement est entièrement géré par GitHub Actions.

Trigger : Un git push sur la branche main.

Build : GitHub construit la nouvelle image Docker.

Registry : L'image est poussée sur Azure Container Registry (ACR) (textilegarage).

Deploy : L'API est mise à jour sur Azure App Service (Texelai).

📊 Résultats et Performance

Le meilleur modèle obtenu via l'optimisation Optuna (Experience ID: optuna_v2) a été sélectionné pour la production.

Modèle : YOLOv8 Nano (optimisé)

Hyperparamètres gagnants :

Learning Rate : 0.00037

Momentum : 0.92

Optimizer : AdamW

Performance (mAP50) : ~8.2% (sur dataset réduit pour démonstration)

