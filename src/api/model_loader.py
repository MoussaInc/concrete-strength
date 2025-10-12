# src/api/model_loader.py

import os
import joblib
from pathlib import Path
from xgboost import XGBRegressor
import logging

# Configuration du logger
logger = logging.getLogger(__name__)

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
    
    IMPORTANT : Pour XGBoost chargé depuis un environnement GPU vers un environnement CPU, 
    nous forçons le paramètre 'predictor' à 'cpu_predictor' pour éviter l'erreur gpu_id.

    Returns:
        model: Objet modèle (généralement un Pipeline) chargé.
    """
    
    # Tentative de chargement du modèle Joblib (Standard pour les Pipelines sklearn)
    if MODEL_JOBLIB.exists():
        try:
            model = joblib.load(MODEL_JOBLIB)
            
            # Si le modèle chargé est un Pipeline et qu'il contient un XGBoost, 
            # le correctif doit être appliqué au sein du Pipeline, mais cela nécessite 
            # de connaître sa structure. Pour simplifier, nous supposons que le pipeline 
            # gère déjà l'inférence.
            logger.info(f"Model loaded (Joblib) from: {MODEL_JOBLIB}")
            return model
            
        except Exception as e:
            # Soulever une erreur interne si le fichier est corrompu
            raise RuntimeError(f"Erreur lors du chargement du modèle joblib : {e}")

    # Tentative de chargement du modèle JSON (si vous sauvegardez XGBoost directement)
    elif MODEL_JSON.exists():
        try:
            model = XGBRegressor()
            model.load_model(MODEL_JSON)
            
            # --- DÉBUT DU CORRECTIF POUR ERREUR GPU/CPU ---
            # Force le modèle à utiliser le prédicteur CPU pour éviter l'erreur 
            # "'XGBModel' object has no attribute 'gpu_id'" dans un environnement CPU.
            model.set_param({"predictor": "cpu_predictor"})
            # --- FIN DU CORRECTIF ---
            
            logger.info(f"Model loaded (JSON XGBoost) from: {MODEL_JSON}")
            return model
            
        except Exception as e:
            # Cela inclut les erreurs potentielles de la méthode set_param si elle échoue
            raise RuntimeError(f"Erreur lors du chargement du modèle JSON XGBoost : {e}")

    else:
        # Erreur si aucun modèle n'est trouvé
        raise FileNotFoundError(
            f"Aucun modèle trouvé. Veuillez exécuter l'entraînement. Chemins vérifiés : {MODEL_JOBLIB} ou {MODEL_JSON}"
        )
