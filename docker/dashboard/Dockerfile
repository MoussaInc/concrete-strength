# docker/dashboard/Dockerfile
FROM python:3.12-slim

# ---- OS deps minimalistes (pas de build toolchain) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Pip: wheels only (évite toute compilation), outils à jour ----
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ONLY_BINARY=:all:

RUN pip install --upgrade pip setuptools wheel

# ---- Dépendances (cache-friendly) ----
COPY docker/dashboard/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# ---- Code applicatif ----
COPY src /app/src
# Crée le dossier models même s'il est vide pour éviter un échec de COPY
RUN mkdir -p /app/models /app/config
# Si tu as vraiment des fichiers à copier, décommente ces lignes (sinon laisse le dossier vide)
# COPY models /app/models
# COPY config /app/config

# ---- Environnement ----
ENV PYTHONPATH=/app \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=auto \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

EXPOSE 8501

# (Optionnel) healthcheck Streamlit (fonctionne sur versions récentes)
HEALTHCHECK --interval=30s --timeout=3s --retries=5 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# ---- Démarrage ----
CMD ["sh", "-c", "streamlit run src/dashboard/app.py --server.address 0.0.0.0 --server.port ${PORT:-8501}"]

