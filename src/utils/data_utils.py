# src/utils/data_utils.py

import numpy as np
import pandas as pd
from typing import List

# --- Constantes (Alignées sur la convention ML) ---
BASE_COLS = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplasticizer', 'CoarseAggregate', 'FineAggregate', 'Age']
DERIVED_COLS = ["Water_Cement_Ratio", "Binder", "Fine_to_Coarse_Ratio", "Total_Material_Mass"]
ALL_FEATURES = BASE_COLS + DERIVED_COLS


def calculate_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les 4 features dérivées si elles sont manquantes.
    Le DataFrame d'entrée DOIT utiliser des noms de colonnes CamelCase (e.g., 'Cement').
    """
    
    df = df.copy()

    # Détecter les features dérivées manquantes
    features_to_calculate = [col for col in DERIVED_COLS if col not in df.columns]
    
    if not features_to_calculate:
        # Aucune feature dérivée à calculer, on renvoie le DataFrame tel quel
        return df

    # --- Calculs robustes (Utilisation de np.nan pour les diviseurs nuls) ---
    cement_safe = df['Cement'].replace(0, np.nan) 
    coarse_agg_safe = df['CoarseAggregate'].replace(0, np.nan)
    component_cols = BASE_COLS[:-1] # Les 7 composants sans 'Age'

    # Water / Cement Ratio
    if "Water_Cement_Ratio" in features_to_calculate:
        # Calcule le ratio E/C, gère Inf si C=0
        df["Water_Cement_Ratio"] = df["Water"] / cement_safe
        df["Water_Cement_Ratio"] = df["Water_Cement_Ratio"].replace([np.inf, -np.inf], np.nan)
    
    # Binder (Liant)
    if "Binder" in features_to_calculate:
        # Utilise fillna(0) si un composant est NaN
        df["Binder"] = df["Cement"].fillna(0) + df["Slag"].fillna(0) + df["FlyAsh"].fillna(0)
    
    # Fine to Coarse Ratio
    if "Fine_to_Coarse_Ratio" in features_to_calculate:
        df["Fine_to_Coarse_Ratio"] = df["FineAggregate"] / coarse_agg_safe
        df["Fine_to_Coarse_Ratio"] = df["Fine_to_Coarse_Ratio"].replace([np.inf, -np.inf], np.nan)
    
    # Total Material Mass
    if "Total_Material_Mass" in features_to_calculate:
        # Somme des 7 composants (sans l'Age)
        df['Total_Material_Mass'] = df[[c for c in component_cols if c in df.columns]].sum(axis=1)

    return df


def apply_constraints_and_warnings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique contraintes physiques & métier et ajoute une colonne 'Warnings'
    contenant une liste de messages par ligne.
    Le DataFrame doit avoir les 12 features en CamelCase (après 'calculate_derived_features').
    """

    df = df.copy()
    df_rules = df[BASE_COLS + DERIVED_COLS].fillna(0)
    
    # Construire la liste de warnings ligne par ligne
    warnings_list = []
    for _, row in df_rules.iterrows():
        row_warnings = []

        # Extraction des valeurs (maintenant en CamelCase)
        Cement = row["Cement"]
        Water = row["Water"]
        Age = row["Age"]
        Binder = row["Binder"]
        WCR = row["Water_Cement_Ratio"]
        Ratio = row["Fine_to_Coarse_Ratio"]
        Superplasticizer = row["Superplasticizer"]
        
        # Calcul de la masse totale pour la règle (doit être fait même si Total_Material_Mass existe)
        Total_Mass = row["Total_Material_Mass"]

        # --- Règles essentielles ---
        if Cement <= 0:
            row_warnings.append("Absence de Ciment, formulation invalide")
        if Water <= 0:
            row_warnings.append("Absence d’Eau, formulation invalide")
        if Binder <= 0:
            row_warnings.append("Aucun Liant, formulation invalide")

        # --- Contraintes métier (WCR) ---
        if WCR < 0.2:
            row_warnings.append("Rapport E/C trop bas (<0.2), mélange trop ferme")
        if WCR > 0.8:
            row_warnings.append("Rapport E/C trop haut (>0.8), béton trop faible")

        # --- Contraintes métier (Age) ---
        if Age < 1:
            row_warnings.append("Âge < 1 jour, résistance non significative")
        if Age > 365:
            row_warnings.append("Âge > 365 jours → extrapolation du modèle")

        # --- Contraintes métier (Masse et Ratios granulaires) ---
        if (Total_Mass < 1000) or (Total_Mass > 4000):
            row_warnings.append(f"Masse totale {Total_Mass:.0f} hors plage (1000–4000 kg/m³)")

        if (Ratio < 0.3) or (Ratio > 1.5):
            row_warnings.append(f"Rapport fines/gros {Ratio:.2f} hors plage [0.3–1.5]")

        # --- Contraintes métier (Additifs) ---
        if (Superplasticizer < 0) or (Superplasticizer > 50):
            row_warnings.append(f"Superplasticizer {Superplasticizer:.1f} hors plage réaliste (0–50)")

        warnings_list.append(row_warnings)

    df["Warnings"] = warnings_list

    return df