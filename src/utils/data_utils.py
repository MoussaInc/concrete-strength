# src/utils/data_utils.py

import numpy as np
import pandas as pd

BASE_COLS = ["cement", "slag", "fly_ash", "water", "superplasticizer",
             "coarse_aggregate", "fine_aggregate", "age"]
DERIVED_COLS = ["water_cement_ratio", "binder", "fine_to_coarse_ratio"]
ALL_FEATURES = BASE_COLS + DERIVED_COLS

def apply_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique contraintes physiques & métier, calcule les features dérivées et
    renvoie une colonne 'warnings' contenant une liste de messages par ligne.
    Ne supprime jamais la colonne 'strength' si elle est présente.
    """
    df = df.copy()

    # Remplir les valeurs manquantes des colonnes de base
    df[BASE_COLS] = df[BASE_COLS].fillna(0)

    # Calculs vectoriels des features dérivées
    df["binder"] = df["cement"] + df["slag"] + df["fly_ash"]
    df["water_cement_ratio"] = df["water"] / (df["cement"] + 1e-6)
    df["fine_to_coarse_ratio"] = df["fine_aggregate"] / (df["coarse_aggregate"] + 1e-6)

    # Construire la liste de warnings ligne par ligne
    warnings_list = []
    for idx, row in df.iterrows():
        row_warnings = []

        cement = row["cement"]
        slag = row["slag"]
        fly_ash = row["fly_ash"]
        water = row["water"]
        superplasticizer = row["superplasticizer"]
        coarse = row["coarse_aggregate"]
        fine = row["fine_aggregate"]
        age = row["age"]
        binder = row["binder"]
        wcr = row["water_cement_ratio"]
        ratio = row["fine_to_coarse_ratio"]
        total = cement + slag + fly_ash + water + superplasticizer + coarse + fine

        # --- Règles essentielles ---
        if cement <= 0:
            row_warnings.append("Absence de ciment, formulation invalide")
        if water <= 0:
            row_warnings.append("Absence d’eau, formulation invalide")
        if binder <= 0:
            row_warnings.append("Aucun liant, formulation invalide")

        if wcr < 0.2:
            row_warnings.append("Rapport E/C trop bas (<0.2), mélange trop ferme")
        if wcr > 0.8:
            row_warnings.append("Rapport E/C trop haut (>0.8), béton trop faible")

        if age < 1:
            row_warnings.append("Âge < 1 jour, pas de résistance mesurable")
        if age > 365:
            row_warnings.append("Âge > 365 jours → extrapolation du modèle")

        if (total < 1000) or (total > 4000):
            row_warnings.append("Masse totale hors plage réaliste (1000–4000 kg/m³)")

        if (ratio < 0.3) or (ratio > 1.5):
            row_warnings.append("Rapport fines/gros hors plage [0.3–1.5]")

        if (superplasticizer < 0) or (superplasticizer > 50):
            row_warnings.append("Superplasticizer hors plage réaliste (0–50)")

        warnings_list.append(row_warnings)

    df["warnings"] = warnings_list

    return df
