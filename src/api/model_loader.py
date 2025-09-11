# src/api/model_loader.py

import os
import sys
import joblib
from xgboost import XGBRegressor

# Ajouter le chemin vers src/ml pour pouvoir retrouver utils.py
ML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ml"))
if ML_PATH not in sys.path:
    sys.path.append(ML_PATH)

# --- chemins possibles pour les modèles ---
MODEL_JOBLIB = os.path.join("models", "best_model.joblib")
MODEL_JSON = os.path.join("models", "best_model.json")

def load_model():
    """
    Charge et retourne le modèle ML sauvegardé.

    - Si un fichier JSON existe → chargement avec XGBoost (CPU-compatible).
    - Sinon, fallback sur joblib (pour compatibilité avec d'anciens modèles).

    Returns:
        model: Objet modèle chargé.
    """
    if os.path.exists(MODEL_JSON):
        try:
            model = XGBRegressor()
            model.load_model(MODEL_JSON)
            return model
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle JSON XGBoost : {e}")

    elif os.path.exists(MODEL_JOBLIB):
        try:
            return joblib.load(MODEL_JOBLIB)
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle joblib : {e}")

    else:
        raise FileNotFoundError(
            f"Aucun modèle trouvé. Attendu : {MODEL_JSON} ou {MODEL_JOBLIB}"
        )
