# src/ml/predict.py

import os
import numpy as np
import pandas as pd
import joblib
import argparse
from typing import Union

from src.utils.data_utils import apply_constraints

"""
Script de prédiction de la résistance du béton à l'aide du modèle ML entraîné.
Il applique les contraintes physiques et préserve l'ordre exact des colonnes.
"""

# --- Constantes ---
MODEL_PATH = "models/best_model.joblib"
PREDICTION_PATH = "data/predictions/predicted_strength.csv"

# Colonnes attendues
BASE_COLS = ['cement', 'slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age']
# Colonnes dérivées
DERIVED_COLS = ['water_cement_ratio', 'binder', 'fine_to_coarse_ratio']
ALL_FEATURES = BASE_COLS + DERIVED_COLS

def main(input_path: Union[str, os.PathLike]) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Fichier d'entrée non trouvé : {input_path}")

    print(f"Chargement des données depuis : {input_path}")
    df = pd.read_csv(input_path)
    df = apply_constraints(df)

    # --- Vérifier et compléter les colonnes de base ---
    for col in BASE_COLS:
        if col not in df.columns:
            print(f"Colonne manquante '{col}' ajoutée avec des zéros.")
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)

    # Vérifier que toutes les colonnes attendues sont présentes
    missing_cols = set(ALL_FEATURES) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Colonnes manquantes après génération : {missing_cols}")

    # Réordonner
    df = df[ALL_FEATURES]

    # --- Charger le modèle ---
    print(f"Chargement du modèle depuis : {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)

    # --- Prédictions ---
    print("Prédictions en cours...")
    predictions = pipeline.predict(df)
    df["predicted_strength"] = predictions

    # --- Sauvegarde ---
    os.makedirs(os.path.dirname(PREDICTION_PATH), exist_ok=True)
    df.to_csv(PREDICTION_PATH, index=False)
    print(f"Prédictions sauvegardées dans : {PREDICTION_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédiction de la résistance du béton à partir d'un fichier CSV.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Chemin vers le fichier CSV contenant les caractéristiques du béton."
    )
    args = parser.parse_args()
    main(args.input)
