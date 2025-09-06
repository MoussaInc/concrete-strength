FROM python:3.11-slim

# Crée et définit le répertoire de travail pour l'application
WORKDIR /app

# Copie le fichier de dépendances en premier pour optimiser le cache Docker
# Cela garantit que la couche d'installation de pip est mise en cache et réutilisée
COPY docker/dashboard/requirements.txt ./docker/dashboard/requirements.txt

# Installe toutes les dépendances nécessaires
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r docker/dashboard/requirements.txt

# Copie uniquement les fichiers essentiels à l'exécution du dashboard
# C'est la partie clé pour réduire la taille de l'image
COPY src/dashboard /app/src/dashboard
COPY src/utils /app/src/utils
COPY models /app/models

# Définit le PYTHONPATH pour que Python puisse trouver les modules
ENV PYTHONPATH=/app

# Expose le port par défaut de Streamlit
EXPOSE 8501

# Commande de démarrage pour l'application Streamlit
# --server.address=0.0.0.0 permet au conteneur de servir les requêtes en dehors de localhost
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
