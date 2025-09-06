FROM python:3.11-slim

# Crée et définit le répertoire de travail pour l'application
WORKDIR /app

# Copie le fichier de dépendances en premier pour optimiser le cache Docker
COPY docker/api/requirements.txt ./docker/api/requirements.txt

# Installe toutes les dépendances
# Le cache sera réutilisé tant que le requirements.txt ne change pas
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r docker/api/requirements.txt

# Copie uniquement les fichiers essentiels à l'API pour réduire la taille de l'image
COPY src/api /app/src/api
COPY src/utils /app/src/utils
COPY models /app/models
COPY config /app/config

# Définit le PYTHONPATH pour que Python puisse trouver les modules
ENV PYTHONPATH=/app

# Expose le port sur lequel l'API va s'exécuter
EXPOSE 8000

# Commande de démarrage pour l'application
# --host 0.0.0.0 permet d'écouter les requêtes depuis n'importe quelle adresse IP
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]