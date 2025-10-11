# src/etl/process_data.py

import pandas as pd
import numpy as np
import re
from pathlib import Path
import logging
import sys

# --- Configuration et Chemins ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_ROOT / "logs"

# --- Configuration du Logging ---
def setup_logging(level=logging.INFO):
    """
    Configure le système de logging (copié/adapté de download_data.py)
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    # Évite les handlers multiples si le script est rechargé
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # Stream Handler (console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    return logger

# Initialisation du logger global
logger = setup_logging(logging.INFO) # Par défaut INFO

# --- Dictionnaire de Mappage Global (Robuste) ---
# Clé : Nom de colonne simplifié après nettoyage (en minuscules, sans espaces ni unités)
# Valeur : Nom de colonne final (Standardisé)
STANDARD_MAPPING = {
    # Variables de base (communes aux 3 sources)
    "cement": "Cement",
    "blastfurnaceslag": "Slag", # UCI/Figshare
    "ggbs": "Slag",              # Mendeley (Laitier)
    "flyash": "FlyAsh",
    "water": "Water",
    "superplasticizer": "Superplasticizer", # UCI/Figshare
    "sp": "Superplasticizer",              # Mendeley
    "coarseaggregate": "CoarseAggregate",
    "fineaggregate": "FineAggregate",
    "sand": "FineAggregate",      # Mendeley
    "age": "Age",
    "concretecompressivestrength": "Strength", # UCI/Figshare
    "cs": "Strength",                         # Mendeley
    
    # Composants supplémentaires de Mendeley
    "mk": "MK",
    "tcm": "TCM",
    "watertcm": "Water_TCM_Ratio",
    "vma": "VMA",
    "nca20down": "NCA_20_DOWN",
    "nca10down": "NCA_10_DOWN",
    "rca20down": "RCA_20_Down",
    "rca10down": "RCA_10_DOWN",
    "serialno": "Serial_No",
}

def clean_column_name(col_name: str) -> str:
    """
    Nettoyage des noms de colonne :
    1. Supprime sauts de ligne, espaces multiples, et ponctuation.
    2. Retire le contenu entre parenthèses (unités, numéros).
    3. Retire les unités courantes.
    4. Supprime tous les espaces restants et convertit en minuscules.
    """
    name = col_name.replace('\n', ' ').replace('\r', ' ')
    name = re.sub(r'\([^)]*\)', '', name)
    name = re.sub(r'\(kg/m3\)|\[\w+\]|MPa|megapascals|\(day\)|,|;|\.', '', name, flags=re.IGNORECASE)
    name = ''.join(name.strip().split()).lower()
    
    return name

def rename_dataframe_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Renomme les colonnes du DataFrame en utilisant le mapping standardisé.
    """
    
    rename_dict = {}
    logger.debug(f"Colonnes brutes de {source_name}: {list(df.columns)}")
    for col in df.columns:
        # Nettoyage de bas niveau
        cleaned_key = clean_column_name(col)
        # Mappage au nom standardisé
        if cleaned_key in STANDARD_MAPPING:
            final_name = STANDARD_MAPPING[cleaned_key]
            rename_dict[col] = final_name
            logger.debug(f"  '{col}' -> '{cleaned_key}' -> '{final_name}'")
        else:
            logger.warning(f"  Colonne '{col}' ({cleaned_key}) de {source_name} non mappée et sera supprimée.")
            rename_dict[col] = None
    
    # Supprimer les colonnes non mappées
    cols_to_drop = [k for k, v in rename_dict.items() if v is None or not v.strip()]
    df.drop(columns=cols_to_drop, inplace=True)
    rename_dict = {k: v for k, v in rename_dict.items() if v is not None}
    
    # Application du renommage
    df.rename(columns=rename_dict, inplace=True)
    return df

def load_and_process_dataset(file_path: Path, source_name: str) -> pd.DataFrame:
    """
    Charge et traite un dataset (UCI, Figshare, ou Mendeley).
    """

    logger.info(f"Chargement et traitement de {source_name}...")
    df = pd.read_csv(file_path)

    # Nettoyage et renommage des colonnes (robuste)
    df = rename_dataframe_columns(df, source_name)
    
    # Traitement spécifique Mendeley (agrégation des agrégats grossiers)
    if source_name == 'Mendeley':
        if 'NCA_20_DOWN' in df.columns and 'NCA_10_DOWN' in df.columns:
            # Agrégation des Natural Coarse Aggregates (NCA) en CoarseAggregate
            df['CoarseAggregate'] = df[['NCA_20_DOWN', 'NCA_10_DOWN']].sum(axis=1, skipna=True)
            df.drop(columns=['NCA_20_DOWN', 'NCA_10_DOWN'], inplace=True)
            logger.info("  Agrégation des NCA en 'CoarseAggregate'.")

        # Supprimer les colonnes RCA si on ne veut pas les considérer comme une feature supplémentaire
        # Pour le moment, nous les gardons si elles existent, elles seront NaN dans UCI/Figshare.
        
    # Conversion des types
    for col in df.columns:
        if col not in ['Source', 'Serial_No']:
            # Forcer la conversion numérique, laissant NaN si impossible (pour les entêtes mal gérés par exemple)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Ajout de la colonne Source
    df['Source'] = source_name
    
    # Suppression des lignes vides (où la force est nulle ou NaN)
    df.dropna(subset=['Strength'], inplace=True) 
    
    logger.info(f"{source_name} prêt: {len(df)} lignes, {len(df.columns)} colonnes.")
    return df

def merge_datasets(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Fusionne la liste de DataFrames après harmonisation.
    """

    logger.info("Début de la fusion des datasets...")
    
    # Fusion des DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)

    # Suppression des lignes sans données utiles
    combined_df.dropna(subset=['Cement', 'Strength'], how='any', inplace=True)
    
    # Remplacer les colonnes qui n'existent que dans un dataset par 0 ou la valeur appropriée (Imputation simple par 0)
    # Ceci est crucial pour les colonnes comme 'MK', 'TCM', etc., qui sont nulles dans UCI/Figshare.
    cols_to_fill_zero = [col for col in combined_df.columns if col not in ['Source', 'Serial_No', 'Age', 'Strength']]
    combined_df[cols_to_fill_zero] = combined_df[cols_to_fill_zero].fillna(0)
    
    # Réordonner les colonnes (mettre les 9 variables clés au début)
    main_cols = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplasticizer', 
                 'CoarseAggregate', 'FineAggregate', 'Age', 'Strength', 'Source']
    
    # Ajouter toutes les autres colonnes à la fin (MK, TCM, etc.)
    other_cols = [c for c in combined_df.columns if c not in main_cols]
    final_cols = [c for c in main_cols if c in combined_df.columns] + sorted(other_cols)
    
    combined_df = combined_df[final_cols]
    
    logger.info(f"Fusion réussie. Dataset final: {len(combined_df)} lignes, {len(combined_df.columns)} colonnes.")
    return combined_df

def main():
    """
    Fonction principale du pipeline de traitement.
    """
    
    # Définition des chemins des fichiers bruts
    datasets_info = {
        'UCI': RAW_DIR / "concrete_data_uci.csv",
        'Figshare': RAW_DIR / "concrete_data_figshare.csv",
        'Mendeley': RAW_DIR / "concrete_data_mendeley_from_pdf.csv"
    }

    if not all(path.exists() for path in datasets_info.values()):
        logger.error("Fichiers bruts manquants. Assurez-vous d'avoir exécuté 'download_data.py --dataset all'.")
        return

    try:
        # Chargement et traitement des 3 datasets
        dfs = []
        for name, path in datasets_info.items():
            dfs.append(load_and_process_dataset(path, name))
        
        # Fusion des datasets
        final_df = merge_datasets(dfs)
        
        # Sauvegarde
        output_path = PROCESSED_DIR / "combined_concrete_strength_data.csv"
        final_df.to_csv(output_path, index=False)
        logger.info(f"Nettoyage et fusion terminés. Fichier sauvegardé dans : {output_path}")
        
    except Exception as e:
        logger.error(f"Une erreur critique s'est produite lors du traitement : {e}", exc_info=True)


if __name__ == "__main__":
    # Définir le log-level à DEBUG pour voir les détails de nettoyage
    logger = setup_logging(logging.DEBUG)
    main()