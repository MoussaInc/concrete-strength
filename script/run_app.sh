#!/bin/bash

# Dossier racine du projet
PROJECT_ROOT=$(dirname "$0")/..

# Ajouter le projet au PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Charger les variables d'environnement depuis le .env universel
if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
fi

echo "Lancement de l'API FastAPI avec Uvicorn..."
# Lance FastAPI en arrière-plan
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
FASTAPI_PID=$!

# Petite pause pour laisser le temps à FastAPI de démarrer
sleep 5

echo "Lancement de l'interface Streamlit..."
# Définit l'URL de l'API pour Streamlit (local ou Render)
export API_URL="${API_URL:-http://127.0.0.1:8000}"

streamlit run src/dashboard/app.py

# Optionnel : tuer FastAPI à la fin si tu fermes Streamlit (Ctrl+C)
echo "Fermeture du serveur FastAPI (PID: $FASTAPI_PID)..."
kill $FASTAPI_PID
