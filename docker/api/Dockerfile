FROM python:3.11

WORKDIR /app

# Copie le fichier de dépendances
COPY docker/api/requirements.txt ./docker/api/requirements.txt

# Installe les dépendances
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r docker/api/requirements.txt

# Copie les fichiers de l'application
COPY src/api /app/src/api
COPY src/utils /app/src/utils
COPY models /app/models
COPY config /app/config

# Définit le PYTHONPATH
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]