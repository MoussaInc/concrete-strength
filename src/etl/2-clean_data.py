# src/etl/2-clean_data.py

import os
import pandas as pd

"""
Script de nettoyage et de création de nouvelles features pour le dataset de résistance en compression simple du béton.

Fonctions principales :
- Chargement du fichier CSV brut.
- Imputation des valeurs manquantes selon une stratégie paramétrable : moyenne (par défaut) ou médiane.
- Détection et traitement des outliers (valeurs extrêmes) par la méthode de l'IQR (Interquartile Range).
- Création de nouvelles features utiles pour le modèle ML :
    * Rapport eau/ciment
    * Quantité totale de liants (binder)
    * Rapport sables/graviers

Usage :
- Pour une imputation avec la médiane, lancer clean_and_engineer(df, impute_strategy='median')
- Le jeu de données nettoyé est sauvegardé dans : data/processed/concrete_data_clean.csv
"""

RAW_CSV = os.path.join("data", "raw", "concrete_data.csv")
PROCESSED_CSV = os.path.join("data", "processed", "concrete_data_clean.csv")


def load_data(path):
    print(f"Lecture du fichier brut : {path}")
    return pd.read_csv(path)


def impute_missing_values(df, strategy='mean'):
    """
    Impute les valeurs manquantes avec la moyenne ou la médiane.

    Args:
        df (pd.DataFrame): Données d'entrée
        strategy (str): 'mean' ou 'median'

    Returns:
        df imputé
    """

    print(f"Imputation des valeurs manquantes avec la stratégie : {strategy}")

    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if df[col].isnull().sum() > 0:
            if strategy == 'median':
                val = df[col].median()
            else:
                val = df[col].mean()
            df[col].fillna(val, inplace=True)
            print(f" - {col}: imputée avec {strategy} ({val:.2f})")

    return df


def treat_outliers_iqr(df):
    """
    Traite les outliers en les limitant aux bornes IQR [Q1 - 1.5*IQR, Q3 + 1.5*IQR]

    Args:
        df (pd.DataFrame): Données d'entrée

    Returns:
        df (pd.DataFrame): Données avec outliers traités
    """

    print("Traitement des outliers (méthode IQR)...")

    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > 0:
            print(f" - {col}: {outliers} outliers traités")
            df[col] = df[col].clip(lower_bound, upper_bound)

    return df


def clean_and_engineer(df, impute_strategy='mean'):
    print("Nettoyage des colonnes et ingénierie de features...")

    df.columns = [
        "cement", "slag", "fly_ash", "water",
        "superplasticizer", "coarse_aggregate", "fine_aggregate",
        "age", "strength"
    ]

    # Suppression des doublons
    # num_duplicates = df.duplicated().sum()
    # print(f"Nombre de doublons supprimés: {num_duplicates}")
    # df = df.drop_duplicates()

    df = impute_missing_values(df, strategy=impute_strategy)
    df = treat_outliers_iqr(df)

    df["water_cement_ratio"] = df["water"] / df["cement"]
    df["binder"] = df["cement"] + df["slag"] + df["fly_ash"]
    df["fine_to_coarse_ratio"] = df["fine_aggregate"] / df["coarse_aggregate"]

    return df


def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Données nettoyées sauvegardées dans : {path}")


def main():
    df = load_data(RAW_CSV)
    impute_strategy="mean"

    print("Souhaitez-vous impute les valeurs manquantes avec la moyenne ou avec la médiane?")
    print(f"[1] pour la moyenne ('mean': choix par défaut)")
    print("[2] pour la mediane 'median'")
    choice = input("Entrez 1 ou 2 : ").strip()

    if choice == "2":
        impute_strategy="median"

    df_clean = clean_and_engineer(df, impute_strategy)
    save_data(df_clean, PROCESSED_CSV)


if __name__ == "__main__":
    main()
