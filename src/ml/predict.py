# src/ml/predict.py - Version avec calcul conditionnel des features

import numpy as np
import pandas as pd
import joblib
import argparse
import sys
from pathlib import Path
import logging
from src.utils.data_utils import (
    apply_constraints_and_warnings, 
    calculate_derived_features, 
    ALL_FEATURES, 
    BASE_COLS 
)

# --- Configuration et Chemins ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_ROOT / "logs"

# --- Constantes ---
MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREDICTION_PATH = PREDICTIONS_DIR / "predicted_strength.csv"

# --- Configuration du Logging (Identique) ---
def setup_logging(level=logging.INFO):
    """
    Configure le système de logging.
    """

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

def load_model(path: Path):
    """
    Chargement du pipeline du modèle entraîné.
    """

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
    Vérifie la présence des features dérivées. Si elles sont absentes, les calcule.
    Sinon, les conserve, puis aligne sur les 12 features.
    """
    
    # Standardisation des noms de colonnes
    def to_camel_case(s):
        if '_' in s:
            s = "".join(x.capitalize() for x in s.lower().split('_'))
        return s[:1].upper() + s[1:] if s and s[:1].islower() else s

    df.columns = [to_camel_case(col) for col in df.columns]
    logger.info("Noms de colonnes standardisés (CamelCase).")

    # Vérification des colonnes de base (les 8 nécessaires)
    for col in BASE_COLS:
        if col not in df.columns:
            logger.warning(f"Colonne de base '{col}' est manquante. Ajoutée avec NaN.")
            df[col] = np.nan
    
    df = calculate_derived_features(df)
    logger.info(f"Features dérivées gérées par le module utils.")

    # Suppression si necessaire des colonnes non désirées et alignement
    cols_to_drop = [col for col in df.columns if col not in ALL_FEATURES and col != 'Strength']
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors='ignore')
        logger.info(f"Colonnes supprimées car non utilisées à l'entraînement: {cols_to_drop}")

    # Réordonner et ne conserver que les 12 features nécessaires
    # reindex garantit que les 12 colonnes sont présentes, les manquantes étant NaN (gérées par l'Imputer)
    df_aligned = df.reindex(columns=ALL_FEATURES, fill_value=np.nan)
    
    # Vérification finale
    if len(df_aligned.columns) != 12:
        logger.error(f"ERREUR D'ALIGNEMENT: Attendu 12 features, obtenu {len(df_aligned.columns)}. Vérifiez ALL_FEATURES.")
        sys.exit(1)
            
    logger.info(f"DataFrame préparé avec {len(df_aligned)} lignes et 12 features.")

    return df_aligned


def predict_strength(df: pd.DataFrame, model) -> pd.Series:
    """
    Prédiction avec le pipeline de modèle chargé.
    """

    try:
        X_array = df.values 
        predictions = model.predict(X_array)
        #predictions = model.predict(df)
        predictions = np.clip(predictions, a_min=0.0, a_max=None)
        return pd.Series(predictions, name="Predicted_Strength")
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise

def main(input_path: Path) -> None:
    if not input_path.exists():
        logger.error(f"Fichier d'entrée non trouvé : {input_path}")
        sys.exit(1)

    logger.info(f"\n--- Démarrage du script de Prédiction ---")
    logger.info(f"Chargement des données brutes depuis : {input_path.name}")
    df_raw = pd.read_csv(input_path)
    
    df_prepared = prepare_dataframe(df_raw.copy())

    model = load_model(MODEL_PATH)

    logger.info("Prédictions en cours...")
    predictions = predict_strength(df_prepared, model)
    
    df_raw["Predicted_Strength"] = predictions

    # --- Sauvegarde ---
    df_raw.to_csv(PREDICTION_PATH, index=False)
    logger.info(f"Prédictions sauvegardées dans : {PREDICTION_PATH.name}")
    logger.info(f"--- Fin du script de Prédiction ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prédiction de la résistance du béton à partir d'un fichier CSV."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Chemin vers le fichier CSV contenant les caractéristiques du béton (e.g., data/to_predict/test_batch.csv)."
    )
    args = parser.parse_args()
    main(Path(args.input))