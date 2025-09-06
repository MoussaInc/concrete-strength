# src/utils/data_utils.py

import numpy as np
import pandas as pd

BASE_COLS = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_aggregate", "fine_aggregate", "age"]
DERIVED_COLS = ["water_cement_ratio", "binder", "fine_to_coarse_ratio"]
ALL_FEATURES = BASE_COLS + DERIVED_COLS

def apply_constraints(X: pd.DataFrame) -> pd.DataFrame:
    """
    Applique contraintes physiques & métier, calcule features dérivées et
    renvoie une colonne 'warnings' contenant une LISTE de messages par ligne.
    """
    # Travailler sur une copie pour éviter effets de bord
    df = X.copy()
    # Remplir les valeurs manquantes des colonnes de base
    df[BASE_COLS] = df[BASE_COLS].fillna(0)

    # Calculs vectoriels des features dérivées
    binder_series = df["cement"] + df["slag"] + df["fly_ash"]
    wcr_series = df["water"] / (df["cement"] + 1e-6)
    ratio_series = df["fine_aggregate"] / (df["coarse_aggregate"] + 1e-6)

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

        binder = cement + slag + fly_ash
        wcr = water / (cement + 1e-6)
        ratio = fine / (coarse + 1e-6)
        total = cement + slag + fly_ash + water + superplasticizer + coarse + fine

        # --- Règles essentielles ---
        if cement <= 0:
            row_warnings.append("Absence de ciment, formulation invalide")
        if water <= 0:
            row_warnings.append("Absence d’eau, formulation invalide")
        if binder <= 0:
            row_warnings.append("Aucun liant, formulation invalide")

        # Rapport eau/ciment
        if wcr < 0.2:
            row_warnings.append("Rapport E/C trop bas (<0.2), mélange trop ferme")
        if wcr > 0.8:
            row_warnings.append("Rapport E/C trop haut (>0.8), béton trop faible")

        # Âge
        if age < 1:
            row_warnings.append("Âge < 1 jour, pas de résistance mesurable")
        if age > 365:
            row_warnings.append("Âge > 365 jours → extrapolation du modèle")

        # Masse totale
        if (total < 1000) or (total > 4000):
            row_warnings.append("Masse totale hors plage réaliste (1000–4000 kg/m³)")

        # Rapport granulats fins/gros
        if (ratio < 0.3) or (ratio > 1.5):
            row_warnings.append("Rapport fines/gros hors plage [0.3–1.5]")

        # Superplasticizer
        if (superplasticizer < 0) or (superplasticizer > 50):
            row_warnings.append("Superplasticizer hors plage réaliste (0–50)")

        warnings_list.append(row_warnings)

    # Assigner les derived features (vectoriels)
    df["water_cement_ratio"] = wcr_series
    df["binder"] = binder_series
    df["fine_to_coarse_ratio"] = ratio_series

    # Assigner la colonne warnings (liste de strings par ligne)
    df["warnings"] = warnings_list

    # Réordonner les colonnes pour le modèle + warnings
    df = df[ALL_FEATURES + ["warnings"]]

    return df
