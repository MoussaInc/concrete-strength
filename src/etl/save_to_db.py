# src/db/save_to_db.py

import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path
import logging
import sys
import yaml
import os # Nécessaire pour os.remove

# --- Configuration et Chemins ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "engineered"
CONFIG_PATH = PROJECT_ROOT / "config" / "database.yaml"
LOG_DIR = PROJECT_ROOT / "logs"

# Nom de la table cible dans PostgreSQL
TABLE_NAME = "engineered_concrete_data" 
TEMP_CSV_PATH = PROJECT_ROOT / "data" / "temp_concrete_upload.csv"

# --- Configuration du Logging ---
def setup_logging(level=logging.INFO):
    """Configure le système de logging."""
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

def load_db_config(config_path: Path) -> dict:
    """Charge les identifiants de connexion PostgreSQL à partir du fichier YAML."""
    if not config_path.exists():
        logger.error(f"Fichier de configuration DB non trouvé: {config_path}")
        logger.error("Veuillez créer 'config/database.yaml' avec les identifiants.")
        sys.exit(1)
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)['postgres']
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration YAML : {e}")
        sys.exit(1)


def get_db_connection(db_params: dict) -> psycopg2.extensions.connection:
    """
    Établit et retourne une connexion à la base de données PostgreSQL.
    """
    try:
        conn = psycopg2.connect(
            host=db_params["db_host"],
            port=db_params["db_port"],
            user=db_params["db_user"],
            password=db_params["db_password"],
            dbname=db_params["db_name"]
        )
        logger.info("Connexion à la base de données PostgreSQL réussie.")
        return conn
    except psycopg2.Error as e:
        logger.error(f"Erreur de connexion PostgreSQL : {e}")
        sys.exit(1)


def load_to_postgres_copy(csv_path: Path, table_name: str, conn: psycopg2.extensions.connection):
    """
    Charge un fichier CSV dans une table PostgreSQL en utilisant COPY FROM STDIN.

    Args:
        csv_path (Path): Chemin du fichier CSV à charger.
        table_name (str): Nom de la table PostgreSQL cible.
        conn (psycopg2.extensions.connection): Objet de connexion PostgreSQL.
    """
    
    # Lire le DataFrame et extraire la liste exacte des colonnes
    logger.info("Chargement du DataFrame et vérification des colonnes...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"Fichier de données non trouvé à : {csv_path}")
        sys.exit(1)

    # Noms des colonnes du DataFrame (qui sont les noms des colonnes SQL)
    df_columns = df.columns.tolist() 
    # Le schéma SQL utilisera la même liste (sauf la colonne 'id' auto-incrémentée)
    sql_columns_list = ', '.join(f'"{col}"' for col in df_columns) 

    # Création du DDL (Data Definition Language)
    # On définit les types pour les colonnes numériques comme REAL (float), sauf pour 'Serial_No' et 'Source'
    
    # Création d'une structure de colonnes avec les types appropriés (REAL ou INTEGER)
    column_defs = []
    for col in df_columns:
        # Tenter d'inférer le type SQL
        if col in ['Source', 'Serial_No']: # 'Source' contient 'UCI', 'Figshare', 'Mendeley'
            sql_type = 'TEXT'
        elif df[col].dtype in (np.float64, np.float32) or col in ['Water_Cement_Ratio', 'Binder', 'Fine_to_Coarse_Ratio']:
            sql_type = 'REAL'
        else:
            sql_type = 'REAL'

        column_defs.append(f'"{col}" {sql_type}')
        
    # Ajouter la clé primaire si elle n'est pas déjà dans les données
    ddl = f"""
        CREATE TABLE "{table_name}" (
            id SERIAL PRIMARY KEY,
            {', '.join(column_defs)}
        );
    """
    
    cur = conn.cursor()

    try:
        logger.info(f"Suppression de la table '{table_name}' si elle existe...")
        cur.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE;') # CASCADE permet de supprimer les dépendances

        logger.info(f"Création de la table '{table_name}' avec schéma adapté...")
        cur.execute(ddl)

        # Préparation du fichier CSV temporaire (SANS en-tête pour COPY)
        logger.info("Création d'un fichier CSV temporaire (sans en-têtes)...")
        df.to_csv(TEMP_CSV_PATH, index=False, header=False, na_rep='') # na_rep='' pour COPY FROM (gère mieux les NaN)

        # Insertion des données via COPY FROM (EXPERT)
        logger.info("Insertion des données via COPY FROM STDIN...")
        
        copy_command = f"""
            COPY "{table_name}" (
                {sql_columns_list}
            )
            FROM STDIN WITH (FORMAT CSV, NULL '', HEADER FALSE)
        """
        
        with open(TEMP_CSV_PATH, 'r') as f:
            cur.copy_expert(copy_command, f)

        conn.commit()
        logger.info(f"Données ({len(df)} lignes) chargées avec succès dans PostgreSQL.")

    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Échec de l'insertion COPY FROM : {e}")
        sys.exit(1)
    finally:
        cur.close()
        conn.close()
        
    # Nettoyage
    if os.path.exists(TEMP_CSV_PATH):
        os.remove(TEMP_CSV_PATH)
        logger.info("Fichier temporaire supprimé.")


def main():
    """Fonction principale du script de sauvegarde dans la base de données."""
    
    csv_path = INPUT_DIR / "engineered_concrete_dataset.csv"
    
    # 1. Configuration et Connexion
    db_config = load_db_config(CONFIG_PATH)
    conn = get_db_connection(db_config)
    
    # 2. Sauvegarde
    load_to_postgres_copy(csv_path, TABLE_NAME, conn)


if __name__ == "__main__":
    main()