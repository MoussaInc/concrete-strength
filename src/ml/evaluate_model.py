# src/ml/evaluate_model.py

import os
import sys
import argparse
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils.data_utils import apply_constraints

"""
Évaluation du modèle Concrete Strength Predictor.
Calcule RMSE, MAE et R², et sauvegarde les prédictions avec erreurs absolues.
Compatible CPU et petit dataset pour Render.
"""

# --- Constantes ---
MODEL_PATH = "models/best_model.joblib"
DEFAULT_OUTPUT_PATH = "data/predictions/evaluation_predictions.csv"

BASE_COLS = ["cement", "slag", "fly_ash", "water", "superplasticizer",
             "coarse_aggregate", "fine_aggregate", "age"]
DERIVED_COLS = ["water_cement_ratio", "binder", "fine_to_coarse_ratio"]
ALL_FEATURES = BASE_COLS + DERIVED_COLS

def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique les contraintes et prépare le DataFrame pour l'évaluation.
    """
    df = apply_constraints(df)

    # Compléter les colonnes manquantes
    for col in ALL_FEATURES:
        if col not in df.columns:
            print(f"Colonne manquante '{col}' ajoutée avec 0.0")
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # Réordonner
    df = df[ALL_FEATURES]

    return df

def evaluate(model, X, y):
    """
    Évalue le modèle et retourne RMSE, MAE, R² et prédictions.
    """
    y_pred = model.predict(X)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return rmse, mae, r2, y_pred

def main(input_path: str, output_path: str = DEFAULT_OUTPUT_PATH):
    # Charger le modèle
    try:
        model = load_model(MODEL_PATH)
        print(f"Modèle chargé depuis : {MODEL_PATH}")
    except Exception as e:
        print(f"Erreur : {e}")
        sys.exit(1)

    # Charger les données
    if not os.path.exists(input_path):
        print(f"Fichier non trouvé : {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    if 'strength' not in df.columns:
        print("Erreur : le fichier doit contenir la colonne 'strength'.")
        sys.exit(1)

    df_prepared = prepare_dataframe(df)
    X = df_prepared
    y = df['strength']

    # Évaluation
    print("Évaluation en cours...")
    rmse, mae, r2, y_pred = evaluate(model, X, y)

    print("\nRésultats de l'évaluation :")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"R²   : {r2:.2f}\n")

    # Sauvegarde des prédictions
    df_results = df.copy()
    df_results['predicted_strength'] = y_pred
    df_results['abs_error'] = (df_results['strength'] - y_pred).abs()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"Prédictions sauvegardées dans : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluer un modèle de prédiction de résistance du béton.")
    parser.add_argument("--input", required=True, help="Chemin vers le fichier CSV contenant les données d'évaluation.")
    parser.add_argument("--output", required=False, help="Chemin de sortie pour sauvegarder les prédictions.", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    main(args.input, args.output)
