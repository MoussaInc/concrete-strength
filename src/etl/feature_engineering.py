# src/features/feature_engineering.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# --- Configuration et Chemins ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "final"
OUTPUT_DIR = PROJECT_ROOT / "data" / "engineered"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_ROOT / "logs"

# --- Configuration du Logging ---
def setup_logging(level=logging.INFO):
    """
    Configure le système de logging pour afficher les messages en console.

    Args:
        level (int): Niveau de logging (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: L'objet logger configuré.
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

# --- Colonnes numériques essentielles pour le traitement ---
# Note : Ces colonnes doivent exister après l'étape clean_data.py
NUMERIC_COLS = [
    'Cement', 'Slag', 'FlyAsh', 'Water', 'Superplasticizer', 
    'CoarseAggregate', 'FineAggregate', 'Age', 'Strength'
]
# Liste des colonnes de composants pour le calcul du Binder et de la masse totale
COMPONENT_COLS = [
    'Cement', 'Slag', 'FlyAsh', 'Water', 'Superplasticizer', 
    'CoarseAggregate', 'FineAggregate'
]


def load_data(path: Path) -> pd.DataFrame:
    """
    Charge le dataset final propre (après nettoyage des duplicats/négatifs).

    Args:
        path (Path): Chemin vers le fichier CSV.

    Returns:
        pd.DataFrame: Le jeu de données chargé.
    """
    logger.info(f"Lecture du fichier d'entrée propre : {path.name}")
    if not path.exists():
        logger.error(f"Fichier non trouvé. Assurez-vous que '3-clean_data.py' a été exécuté.")
        sys.exit(1)
        
    return pd.read_csv(path)


def treat_outliers_iqr_clipping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Traite les outliers des colonnes numériques par 'clipping' (plafonnement)
    aux bornes IQR [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    
    Cette méthode est préférée à la suppression des lignes pour préserver les données.
    Les bornes inférieures sont plafonnées à 0 pour les matériaux.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.

    Returns:
        pd.DataFrame: Le DataFrame avec outliers modérés.
    """
    logger.info("Traitement des outliers par 'clipping' (méthode IQR 1.5)...")

    for col in [c for c in NUMERIC_COLS if c in df.columns]:
        # On exclut Strength du clipping car les 0.0 sont des valeurs physiques réelles
        if col == 'Strength':
             continue
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # S'assurer que les quantités de matériaux et l'âge ne sont pas clippés sous zéro
        lower_bound = max(0, lower_bound)
        
        outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        if outliers_count > 0:
            df[col] = df[col].clip(lower_bound, upper_bound)
            logger.info(f" - {col}: {outliers_count} outliers clippés.")
    
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les features d'ingénierie essentielles pour la prédiction de la résistance
    du béton.

    1. Ratio Eau/Ciment (Water_Cement_Ratio)
    2. Liant Total (Binder)
    3. Ratio Granulats Fin/Gros (Fine_to_Coarse_Ratio)
    4. Masse Totale des Matériaux (Total_Material_Mass)
    
    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.

    Returns:
        pd.DataFrame: Le DataFrame enrichi de nouvelles features.
    """
    logger.info("Ingénierie de features (ratios et sommes)...")

    # --- Ratio Eau / Ciment ---
    # Remplacement de 0 par pd.NA pour éviter les divisions par zéro/infini
    df["Water_Cement_Ratio"] = df["Water"] / df["Cement"].replace(0, pd.NA) 
    df["Water_Cement_Ratio"] = df["Water_Cement_Ratio"].replace([float('inf'), -float('inf')], pd.NA)
    logger.info(f" - Feature 'Water_Cement_Ratio' créée (NaNs: {df['Water_Cement_Ratio'].isna().sum()}).")

    # --- Liant total (Binder) ---
    df["Binder"] = df["Cement"] + df["Slag"] + df["FlyAsh"]
    logger.info(" - Feature 'Binder' (Liant total) créée.")

    # --- Ratio Granulats fins / Granulats grossiers ---
    df["Fine_to_Coarse_Ratio"] = df["FineAggregate"] / df["CoarseAggregate"].replace(0, pd.NA)
    df["Fine_to_Coarse_Ratio"] = df["Fine_to_Coarse_Ratio"].replace([float('inf'), -float('inf')], pd.NA)
    logger.info(f" - Feature 'Fine_to_Coarse_Ratio' créée (NaNs: {df['Fine_to_Coarse_Ratio'].isna().sum()}).")
    
    # --- Total des intrants (Masse Totale) ---
    # Utile pour vérifier les densités ou les proportions globales
    df['Total_Material_Mass'] = df[[c for c in COMPONENT_COLS if c in df.columns]].sum(axis=1)
    logger.info(" - Feature 'Total_Material_Mass' créée.")

    return df


def save_data(df: pd.DataFrame, path: Path):
    """
    Sauvegarde le dataset d'ingénierie de features au format CSV.

    Args:
        df (pd.DataFrame): Le DataFrame final.
        path (Path): Chemin de sortie du CSV.
    """
    df.to_csv(path, index=False)
    logger.info(f"Données avec features sauvegardées dans : {path}")


def main():
    """Fonction principale du script d'ingénierie de features."""
    
    input_path = INPUT_DIR / "final_concrete_dataset.csv"
    output_path = OUTPUT_DIR / "engineered_concrete_dataset.csv"
    
    df = load_data(input_path)

    # 1. Traitement des Outliers par clipping
    df = treat_outliers_iqr_clipping(df)
    
    # 2. Ingénierie des Features
    df_engineered = create_features(df)
    
    # 3. Sauvegarde CSV intermédiaire (prêt pour la DB)
    save_data(df_engineered, output_path)
    
    logger.info(f"\nTotal de {len(df_engineered)} lignes prêtes pour la sauvegarde finale en DB.")


if __name__ == "__main__":
    main()