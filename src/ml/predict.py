# src/ml/predict.py

import os
import numpy as np
import pandas as pd
import joblib
import argparse
import sys
from pathlib import Path
from typing import Union
import logging

# --- Configuration et Chemins ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_ROOT / "logs"

# --- Constantes ---
MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREDICTION_PATH = PREDICTIONS_DIR / "predicted_strength.csv"

ALL_FEATURES = [
    'Cement', 'Slag', 'FlyAsh', 'Water', 'Superplasticizer', 
    'CoarseAggregate', 'FineAggregate', 'Age', 
    'Water_Cement_Ratio', 'Binder', 'Fine_to_Coarse_Ratio', 'Total_Material_Mass',
    'Source'
]

# --- Configuration du Logging ---
def setup_logging(level=logging.INFO):
    """Configure le système de logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__) 
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger

logger = setup_logging(logging.INFO)

# --- Fonctions de Chargement et Préparation ---
def load_model(path: Path):
    """Charge le pipeline de modèle entraîné."""
    if not path.exists():
        logger.error(f"Modèle introuvable : {path}")
        sys.exit(1)
    try:
        model = joblib.load(path)
        logger.info(f"Modèle (Pipeline) chargé : {path.name}")
        return model
    except Exception as e:
        logger.critical(f"Erreur lors du chargement du modèle : {e}")
        sys.exit(1)

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vérifie et aligne le DataFrame d'entrée avec les features attendues par le modèle.
    
    IMPORTANT : Les calculs de ratios et l'imputation des NaN DOIVENT être faits ICI
    avant la prédiction, car le modèle (pipeline) s'attend à ces features.
    """
    
    # Calculs des Features Engineering (doit être fait avant le modèle)
    
    # Éviter les divisions par zéro : remplacer 0 par NaN pour générer la bonne valeur de ratio
    cement_safe = df['Cement'].replace(0, np.nan) 
    coarse_agg_safe = df['CoarseAggregate'].replace(0, np.nan)
    
    # Water / Cement Ratio
    df["Water_Cement_Ratio"] = df["Water"] / cement_safe
    df["Water_Cement_Ratio"] = df["Water_Cement_Ratio"].replace([np.inf, -np.inf], np.nan)
    
    # Binder
    df["Binder"] = df["Cement"] + df["Slag"] + df["FlyAsh"]
    
    # Fine to Coarse Ratio
    df["Fine_to_Coarse_Ratio"] = df["FineAggregate"] / coarse_agg_safe
    df["Fine_to_Coarse_Ratio"] = df["Fine_to_Coarse_Ratio"].replace([np.inf, -np.inf], np.nan)
    
    # Total Material Mass (pour l'alignement des colonnes)
    component_cols = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplasticizer', 'CoarseAggregate', 'FineAggregate']
    df['Total_Material_Mass'] = df[[c for c in component_cols if c in df.columns]].sum(axis=1)

    # Alignement des colonnes
    missing_cols = [col for col in ALL_FEATURES if col not in df.columns]
    if missing_cols:
        logger.warning(f"Colonnes manquantes dans le fichier d'entrée (ajoutées avec NaN) : {missing_cols}")
        for col in missing_cols:
            df[col] = np.nan # Ajouter NaN pour que l'Imputer le gère
    
    # Réordonner et ne conserver que les features nécessaires
    df_aligned = df[ALL_FEATURES]
    
    logger.info(f"DataFrame préparé avec {len(df_aligned)} lignes et {len(ALL_FEATURES)} features.")

    return df_aligned


def predict_strength(df: pd.DataFrame, model) -> pd.Series:
    """
    Effectue la prédiction avec le pipeline de modèle chargé.
    """
    try:
        # Le pipeline contient : Imputer -> Scaler -> Modèle (gestion des NaN incluse)
        predictions = model.predict(df)
        
        # S'assurer que les prédictions négatives (physiquement impossibles) sont clippées à zéro
        predictions = np.clip(predictions, a_min=0.0, a_max=None)
        
        return pd.Series(predictions, name="Predicted_Strength")
        
    except AttributeError as e:
        # Gestion du changement GPU -> CPU pour XGBoost
        if "'gpu_id'" in str(e) and hasattr(model.named_steps['model'], 'get_booster'):
            booster = model.named_steps['model'].get_booster()
            booster.set_param({'gpu_id': -1})
            logger.warning("Correction : Définit XGBoost pour fonctionner sur CPU.")
            predictions = model.predict(df)
            return pd.Series(np.clip(predictions, a_min=0.0, a_max=None), name="Predicted_Strength")
        else:
            logger.error(f"Erreur lors de la prédiction : {e}")
            raise

def main(input_path: Path) -> None:
    if not input_path.exists():
        logger.error(f"Fichier d'entrée non trouvé : {input_path}")
        sys.exit(1)

    logger.info(f"Chargement des données brutes depuis : {input_path.name}")
    df_raw = pd.read_csv(input_path)
    
    # Le calcul des features engineering DOIT se faire ici pour le fichier brut d'entrée
    # avant le passage au pipeline du modèle.
    df_prepared = prepare_dataframe(df_raw.copy())

    model = load_model(MODEL_PATH)

    logger.info("Prédictions en cours...")
    predictions = predict_strength(df_prepared, model)
    
    # Ajouter la colonne de prédiction au DataFrame initial pour la sortie
    df_raw["Predicted_Strength"] = predictions

    # --- Sauvegarde ---
    df_raw.to_csv(PREDICTION_PATH, index=False)
    logger.info(f"Prédictions sauvegardées dans : {PREDICTION_PATH.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prédiction de la résistance du béton à partir d'un fichier CSV."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Chemin vers le fichier CSV contenant les caractéristiques du béton (e.g., data/raw/new_data.csv)."
    )
    args = parser.parse_args()
    
    # Utiliser Path pour gérer les chemins correctement
    main(Path(args.input))