# src/ml/evaluate_model.py - Cohérence totale du pipeline

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
from src.utils.data_utils import (
    calculate_derived_features, 
    apply_constraints_and_warnings, 
    ALL_FEATURES
) 

# --- Constantes ---
MODEL_PATH = "models/best_model.joblib"
DEFAULT_OUTPUT_PATH = "data/predictions/evaluation_predictions.csv"

def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")

# Fonction utilitaire pour standardiser les noms (pour réutilisation)
def _standardize_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise les noms de colonnes en CamelCase."""
    def to_camel_case(s):
        if '_' in s:
            s = "".join(x.capitalize() for x in s.lower().split('_'))
        return s[:1].upper() + s[1:] if s and s[:1].islower() else s

    df.columns = [to_camel_case(col) for col in df.columns]
    
    # Renommage de la cible (si nécessaire)
    if 'Strength' not in df.columns and 'strength' in df.columns:
        df = df.rename(columns={'strength': 'Strength'})
        
    return df

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique la FE, l'audit, et retourne le DataFrame X (12 features) pour la prédiction.
    """
    
    df = df.copy()
    # Standardisation des noms de colonnes (effectuée avant FE/Audit)
    df = _standardize_names(df)
    # Calcul/Vérification des features dérivées
    df = calculate_derived_features(df)
    # Application des contraintes et ajout de la colonne 'Warnings' (pour l'audit)
    df = apply_constraints_and_warnings(df)
    # ALIGNEMENT de X (les features pour le modèle) ne doit contenir que les 12 features numériques, excluant 'Strength' et 'Warnings'
    df_X = df.reindex(columns=ALL_FEATURES, fill_value=np.nan)

    return df_X

def evaluate(model, X, y):
    """
    Évalue le modèle, supprime le UserWarning et retourne RMSE, MAE, R² et prédictions.
    """

    X_array = X.values 
    y_pred = model.predict(X_array)
    # Assurer que les prédictions négatives sont clippées (sécurité)
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
    # Calcul des métriques
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
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

    df_raw = pd.read_csv(input_path)
    
    # Standardisation des noms de colonnes de df_raw AU DÉBUT pour faciliter l'extraction de Y
    df_raw = _standardize_names(df_raw)
    
    # La cible doit être extraite APRÈS la standardisation
    if 'Strength' not in df_raw.columns:
        print("Erreur : le fichier doit contenir la colonne 'Strength' (cible).")
        sys.exit(1)

    y = df_raw['Strength']
    
    # Préparation du DataFrame pour le modèle (FE, Audit, Alignement)
    X = prepare_dataframe(df_raw) 

    # Évaluation
    print("Évaluation en cours...")
    rmse, mae, r2, y_pred = evaluate(model, X, y)

    # --- Affichage des Résultats ---
    print("\n==============================")
    print("  Résultats de l'évaluation")
    print("==============================")
    print(f"RMSE (Root Mean Squared Error) : {rmse:.2f}")
    print(f"MAE (Mean Absolute Error)      : {mae:.2f}")
    print(f"R² (Coefficient de Détermination): {r2:.2f}\n")

    # --- Sauvegarde des prédictions (avec les warnings) ---
    df_results = df_raw.copy()
    
    # Ré-application de la FE et de l'Audit pour s'assurer que les colonnes dérivées/Warnings sont présentes pour la sauvegarde
    df_results = calculate_derived_features(df_results)
    df_results = apply_constraints_and_warnings(df_results)
    
    # Ajout des métriques
    df_results['Predicted_Strength'] = y_pred
    df_results['Absolute_Error'] = (df_results['Strength'] - y_pred).abs()

    # Sauvegarde
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"Prédictions sauvegardées dans : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluer un modèle de prédiction de résistance du béton.")
    parser.add_argument("--input", required=True, help="Chemin vers le fichier CSV contenant les données d'évaluation (incluant la colonne 'Strength').")
    parser.add_argument("--output", required=False, help="Chemin de sortie pour sauvegarder les prédictions.", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    main(args.input, args.output)