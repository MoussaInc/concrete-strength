# src/api/model_loader.py

import os
import joblib
from pathlib import Path
from xgboost import XGBRegressor

# --- Définition du chemin Racine du Projet ---
# Remonter de src/api/ vers la racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2] 

# --- Chemins Absolus vers les Modèles ---
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_JOBLIB = MODELS_DIR / "best_model.joblib"
MODEL_JSON = MODELS_DIR / "best_model.json"

def load_model():
    """
    Charge et retourne le modèle ML sauvegardé (Pipeline scikit-learn).
    Priorise le format joblib ou JSON s'il est utilisé.
    Returns:
        model: Objet modèle (généralement un Pipeline) chargé.
    """
    
    # Tentative de chargement du modèle Joblib (Standard pour les Pipelines sklearn)
    if MODEL_JOBLIB.exists():
        try:
            model = joblib.load(MODEL_JOBLIB)
            print(f"Modèle chargé (Joblib) depuis : {MODEL_JOBLIB}")
            return model
        except Exception as e:
            # Soulever une erreur interne si le fichier est corrompu
            raise RuntimeError(f"Erreur lors du chargement du modèle joblib : {e}")

    # Tentative de chargement du modèle JSON (si vous sauvegardez XGBoost directement)
    elif MODEL_JSON.exists():
        try:
            # Note : Le modèle XGBoost chargé seul ne contiendra pas le Scaler/Imputer.
            # Assurez-vous d'avoir sauvegardé le Pipeline complet en joblib pour la cohérence.
            # Ce bloc est conservé pour la compatibilité, mais joblib est préféré pour les Pipelines.
            model = XGBRegressor()
            model.load_model(MODEL_JSON)
            print(f"Modèle chargé (JSON XGBoost) depuis : {MODEL_JSON}")
            return model
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle JSON XGBoost : {e}")

    else:
        # Erreur si aucun modèle n'est trouvé
        raise FileNotFoundError(
            f"Aucun modèle trouvé. Veuillez exécuter l'entraînement. Chemins vérifiés : {MODEL_JOBLIB} ou {MODEL_JSON}"
        )