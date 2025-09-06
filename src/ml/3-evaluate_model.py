# src/ml/3-evaluate_model.py

import argparse
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils.data_utils import apply_constraints

# --- Constantes ---
MODEL_PATH = "models/best_model.joblib"
DEFAULT_PREDICTIONS_PATH = "data/predictions/evaluation_predictions.csv"

# Colonnes de base et dérivées
BASE_COLS = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_aggregate", "fine_aggregate", "age"]
DERIVED_COLS = ["water_cement_ratio", "binder", "fine_to_coarse_ratio"]
ALL_FEATURES = BASE_COLS + DERIVED_COLS


def load_data(file_path: str):
    """
    Charge les données CSV et applique les contraintes physiques.
    Retourne X et y.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier non trouvé : {file_path}")

    df = pd.read_csv(file_path)
    if 'strength' not in df.columns:
        raise ValueError("Le fichier doit contenir la colonne 'strength'.")

    df = apply_constraints(df)

    # Compléter les colonnes manquantes
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    X = df[ALL_FEATURES]
    y = df['strength']
    return X, y, df


def evaluate(model, X, y):
    y_pred = model.predict(X)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return rmse, mae, r2, y_pred

def main(input_path: str, output_path: str = DEFAULT_PREDICTIONS_PATH):
    # Charger le modèle
    if not os.path.exists(MODEL_PATH):
        print(f"Erreur : modèle non trouvé à {MODEL_PATH}")
        sys.exit(1)
    model = joblib.load(MODEL_PATH)
    print(f"Modèle chargé depuis : {MODEL_PATH}")

    # Charger les données
    try:
        X, y, df_original = load_data(input_path)
        print(f"Données chargées depuis : {input_path}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Erreur lors du chargement des données : {e}")
        sys.exit(1)

    # Évaluation
    print("Évaluation en cours...")
    rmse, mae, r2, y_pred = evaluate(model, X, y)

    print("\nRésultats de l'évaluation :")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"R²   : {r2:.2f}\n")

    # Sauvegarde des prédictions
    df_results = df_original.copy()
    df_results['predicted_strength'] = y_pred
    df_results['abs_error'] = (df_results['strength'] - y_pred).abs()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"Prédictions sauvegardées dans : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluer un modèle de prédiction de résistance du béton.")
    parser.add_argument("--input", required=True, help="Chemin vers le fichier CSV contenant les données d'évaluation.")
    parser.add_argument("--output", required=False, help="Chemin de sortie pour sauvegarder les prédictions.", default=DEFAULT_PREDICTIONS_PATH)
    args = parser.parse_args()

    main(args.input, args.output)
