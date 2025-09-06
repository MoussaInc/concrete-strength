#!/bin/bash

# Dossier racine du projet
PROJECT_ROOT=$(dirname "$0")

echo "Lancement de l'API FastAPI avec Uvicorn..."
# Lance FastAPI en arrière-plan
uvicorn src.api.main:app --reload &
FASTAPI_PID=$!

# Petite pause pour laisser le temps à FastAPI de démarrer
sleep 3

echo "Lancement de l'interface Streamlit..."
# Lance Streamlit
streamlit run src/dashboard/app.py

# Optionnel : tuer FastAPI à la fin si tu fermes Streamlit (Ctrl+C)
echo "Fermeture du serveur FastAPI (PID: $FASTAPI_PID)..."
kill $FASTAPI_PID
