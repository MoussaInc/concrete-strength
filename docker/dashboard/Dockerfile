# Dockerfile pour Streamlit
FROM python:3.11

# --- Répertoire de travail ---
WORKDIR /app

# --- Installer les outils de build (pour xgboost si besoin) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# --- Mise à jour de pip / setuptools / wheel ---
RUN python -m pip install --upgrade pip setuptools wheel

# --- Copie du fichier de dépendances ---
COPY docker/dashboard/requirements.txt ./docker/dashboard/requirements.txt

# --- Installation des dépendances ---
RUN pip install --no-cache-dir -r docker/dashboard/requirements.txt

# --- Copie des fichiers sources ---
COPY src/dashboard /app/src/dashboard
COPY src/utils /app/src/utils
COPY models /app/models

# --- Définir PYTHONPATH ---
ENV PYTHONPATH=/app

# --- Expose port Streamlit ---
EXPOSE 8501

# --- Commande de démarrage ---
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
