# docker/api/Dockerfile
FROM python:3.12-slim

# --- OS deps légers (curl pour healthcheck, build-essential si jamais une lib tombe en sdist) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# --- Répertoire de travail ---
WORKDIR /app

# --- Pip: uniquement des wheels, upgrade outils de build (évite compilations) ---
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ONLY_BINARY=:all:

# On installe pip/setuptools/wheel d'abord pour profiter du cache
RUN pip install --upgrade pip setuptools wheel

# --- Dépendances Python (utilise le cache docker) ---
COPY docker/api/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# --- Code application ---
COPY src/api /app/src/api
COPY src/utils /app/src/utils
# Si le dossier models peut être absent en dev, décommente la ligne suivante au lieu du COPY:
RUN mkdir -p /app/models
COPY models /app/models
COPY config /app/config

# --- Pour que Python voie `src/...` ---
ENV PYTHONPATH=/app

# --- Exposition du port ---
EXPOSE 8000

# --- Commande de lancement (prod) ---
# Gunicorn + UvicornWorker = robuste en prod, 2 workers en CPU limité (ajuste selon besoin)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "src.api.main:app"]

