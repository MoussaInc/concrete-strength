# src/etl/clean_data.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# --- Configuration et Chemins ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_DIR = PROJECT_ROOT / "data" / "final" 
FINAL_DIR.mkdir(parents=True, exist_ok=True)
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

# --- Liste des colonnes de composants (kg/m³) ---
COMPONENT_COLS = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplasticizer', 'CoarseAggregate', 'FineAggregate']

# --- Fonctions de Nettoyage ---
def load_data() -> pd.DataFrame:
    """
    Charge le dataset combiné (post-fusion) à partir du dossier processed.

    Returns:
        pd.DataFrame: Le jeu de données combiné.
    """

    input_path = PROCESSED_DIR / "combined_concrete_strength_data.csv"
    if not input_path.exists():
        logger.error(f"Fichier d'entrée non trouvé: {input_path}")
        logger.error("Assurez-vous d'avoir exécuté 'process_data.py' avant.")
        sys.exit(1)
        
    df = pd.read_csv(input_path)
    logger.info(f"Dataset chargé : {len(df)} lignes, {len(df.columns)} colonnes.")
    return df

def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identification et suppression des lignes entièrement dupliquées basées sur les ingrédients
    et l'âge de test et la résistance mesurée. Conserve la première occurrence.

    Args:
        df (pd.DataFrame): Le jeu de données combiné.

    Returns:
        pd.DataFrame: Le DataFrame sans duplicats.
    """

    initial_rows = len(df)
    
    # On définit les duplicats comme ayant les mêmes quantités d'intrants et le même âge
    duplicate_subset = [c for c in COMPONENT_COLS if c in df.columns] + ['Age'] + ['Strength']
    
    df.drop_duplicates(subset=duplicate_subset, keep='first', inplace=True)
    
    duplicates_removed = initial_rows - len(df)
    logger.info(f"Duplicats traités : {duplicates_removed} lignes dupliquées supprimées.")
    return df

def validate_physical_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valide les contraintes physiques en gérant les valeurs négatives/manquantes
    (suppression de lignes) et les valeurs nulles (mise à zéro de 'Strength').

    Args:
        df (pd.DataFrame): Le jeu de données actuel.

    Returns:
        pd.DataFrame: Le DataFrame après validation et nettoyage.
    """

    initial_rows = len(df)
    essential_cols = ['Cement', 'Water', 'Age']
    
    # Remplacer les valeurs négatives par NaN pour les cibler
    for col in [c for c in COMPONENT_COLS + ['Age'] if c in df.columns]:
        df[col] = np.where(df[col] < 0, np.nan, df[col])
        
    # Suppression des lignes où des intrants essentiels sont NaN après la conversion des négatifs
    pre_zero_rows = len(df)
    df.dropna(subset=essential_cols, inplace=True)
    rows_deleted_nan_neg = pre_zero_rows - len(df)

    logger.info(f"Contraintes physiques validées (Suppression): {rows_deleted_nan_neg} lignes supprimées (NaN ou Négatif dans {', '.join(essential_cols)}).")
    
    # Crée un masque pour les lignes où AU MOINS UN des composants essentiels est ÉGAL à ZÉRO
    mask_zero_component = (df[essential_cols] == 0).any(axis=1)
    
    rows_modified_zero = mask_zero_component.sum()
    
    if rows_modified_zero > 0:
        # Met Strength à zéro pour toutes les lignes correspondant au masque
        df.loc[mask_zero_component, 'Strength'] = 0.0
        logger.warning(f"Traitement des valeurs nulles : {rows_modified_zero} lignes avec composant essentiel à zéro ont vu leur 'Strength' fixée à ZÉRO (0.0).")
    
    # Supprime toute ligne où Strength est encore NaN (cas non gérés ou corrompus)
    df.dropna(subset=['Strength'], inplace=True)
    
    rows_deleted_total = initial_rows - len(df)
    logger.info(f"Résumé de la validation : {rows_deleted_total} lignes ont été supprimées au total.")
    
    return df

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vérifie l'état des valeurs manquantes et effectue une imputation simple
    pour les agrégats recyclés (RCA) si elles sont toujours manquantes.

    Args:
        df (pd.DataFrame): Le jeu de données actuel.

    Returns:
        pd.DataFrame: Le DataFrame avec RCA imputé.
    """
    
    # Affichage des NaN restants
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if not missing_data.empty:
        logger.warning(f"Valeurs manquantes restantes (non essentielles) :\n{missing_data.head(5)}")
    else:
        logger.info("Aucune valeur manquante restante dans les colonnes numériques principales.")
        
    # Imputation: Remplir les colonnes des agrégats recyclés (RCA) par 0, car NaN
    # signifie probablement absence de ce composant dans ces mélanges.
    rca_cols = [c for c in df.columns if 'RCA' in c]
    if rca_cols:
        df[rca_cols] = df[rca_cols].fillna(0)
        logger.info(f"  > Imputation par 0 appliquée aux colonnes RCA.")
    
    return df

def handle_outliers_iqr_removal(df: pd.DataFrame, col: str, factor: float = 3.0) -> pd.DataFrame:
    """
    Supprime les outliers d'une colonne en utilisant la méthode IQR (Interquartile Range).
    Les valeurs en dehors de [Q1 - factor*IQR, Q3 + factor*IQR] sont supprimées.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.
        col (str): La colonne à vérifier.
        factor (float): Facteur multiplicateur pour l'IQR (3.0 est moins agressif que 1.5).

    Returns:
        pd.DataFrame: Le DataFrame après suppression des outliers pour cette colonne.
    """
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    initial_count = len(df)
    # Filtrer le DataFrame pour ne conserver que les valeurs dans la plage
    df_filtered = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)].copy()
    
    outliers_removed = initial_count - len(df_filtered)
    if outliers_removed > 0:
        logger.debug(f"  > Outliers pour {col}: {outliers_removed} lignes supprimées (IQR factor={factor})")
    
    return df_filtered

def clean_data_pipeline(df: pd.DataFrame, remove_outliers: bool = True) -> pd.DataFrame:
    """
    Exécute le pipeline de nettoyage des données (Duplicats, Contraintes, Outliers).

    Args:
        df (pd.DataFrame): Le jeu de données à nettoyer.
        remove_outliers (bool): Si True, supprime les lignes considérées comme outliers.

    Returns:
        pd.DataFrame: Le jeu de données final, propre.
    """
    
    logger.info("\n--- Début du pipeline de nettoyage final (clean_data) ---")
    
    # 1. Duplicats
    df = handle_duplicates(df)
    
    # 2. Contraintes Physiques (valeurs négatives/essentielles manquantes)
    df = validate_physical_constraints(df)
    
    # 3. Valeurs Manquantes (Vérification et imputation secondaire)
    df = check_missing_values(df)
    
    # 4. Outliers (Optionnel)
    if remove_outliers:
        pre_outlier_count = len(df)
        
        # Colonnes clés pour la détection d'outliers
        outlier_cols = [c for c in COMPONENT_COLS + ['Age', 'Strength'] if c in df.columns]
        
        # Nettoyage séquentiel par suppression de lignes (méthode IQR modérée)
        for col in outlier_cols:
             df = handle_outliers_iqr_removal(df, col, factor=3.0) 
             
        outliers_removed = pre_outlier_count - len(df)
        logger.info(f"Outliers traités : {outliers_removed} lignes supprimées au total.")

    logger.info("--- Nettoyage terminé ---")
    return df

def main():
    """Fonction principale du script de nettoyage final."""
    
    df = load_data()
    
    # Exécuter le pipeline de nettoyage
    final_df = clean_data_pipeline(df, remove_outliers=True)
    
    # Sauvegarde du fichier final (prêt pour l'ingénierie de features)
    output_path = FINAL_DIR / "final_concrete_dataset.csv"
    final_df.to_csv(output_path, index=False)
    
    logger.info(f"\n Dataset intermédiaire propre sauvegardé : {len(final_df)} lignes.")
    logger.info(f"Fichier sauvegardé dans : {output_path}")

if __name__ == "__main__":
    main()