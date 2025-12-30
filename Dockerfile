FROM python:3.9-slim

# Installe les outils pour que OpenCV fonctionne
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Installe les dépendances
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# Copie ton organisation propre
COPY model/ model/
COPY src/ src/

# Ouvre le port
EXPOSE 80

# Lance l'application (Note le chemin src/app.py)
CMD ["python", "src/app.py"]