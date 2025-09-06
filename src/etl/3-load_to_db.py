# src/etl/3-load_to_db.py

import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

"""
Script de chargement des données nettoyées dans une base PostgreSQL.

Ce script lit un fichier CSV contenant les données prétraitées de résistance du béton,
et les insère dans une table PostgreSQL en utilisant la commande COPY pour des performances optimales.

Fonctionnalités :
- Connexion à la base de données via des variables d'environnement (fichier .env)
- Suppression de la table si elle existe
- Création d'une nouvelle table avec les bons types de colonnes
- Insertion rapide des données (COPY FROM)
- Nettoyage automatique du fichier temporaire utilisé pour l'insertion

Assurez-vous que le serveur PostgreSQL est lancé et que les variables suivantes sont définies :
- PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE
"""

# Charger les variables d'environnement
load_dotenv()

# Chemin vers les données nettoyées
CSV_PATH = os.path.join("data", "processed", "concrete_data_clean.csv")

# Paramètres DB
DB_PARAMS = {
    "host": os.getenv("PG_HOST"),
    "port": os.getenv("PG_PORT"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD"),
    "dbname": os.getenv("PG_DATABASE")
}

TABLE_NAME = "concrete_strength"
TEMP_CSV_PATH = "temp_concrete.csv"

def load_to_postgres_copy(csv_path, table_name):
    """
    Charge un fichier CSV dans une table PostgreSQL en utilisant COPY FROM STDIN.

    Args:
        csv_path (str): Chemin du fichier CSV à charger.
        table_name (str): Nom de la table PostgreSQL cible.
    """

    print("Chargement du DataFrame...")
    df = pd.read_csv(csv_path)

    print("Création d'un fichier CSV temporaire (sans en-têtes)...")
    df.to_csv(TEMP_CSV_PATH, index=False, header=False)

    print("Connexion à PostgreSQL...")
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    print(f"Suppression de la table '{table_name}' si elle existe...")
    cur.execute(f"DROP TABLE IF EXISTS {table_name}")

    print(f"Création de la table '{table_name}' avec schéma adapté...")
    cur.execute(f"""
        CREATE TABLE {table_name} (
            id SERIAL PRIMARY KEY,
            cement REAL,
            slag REAL,
            fly_ash REAL,
            water REAL,
            superplasticizer REAL,
            coarse_aggregate REAL,
            fine_aggregate REAL,
            age REAL,
            strength REAL,
            water_cement_ratio REAL,
            binder REAL,
            fine_to_coarse_ratio REAL
        );
    """)

    print("Insertion des données via COPY FROM...")
    with open(TEMP_CSV_PATH, 'r') as f:
        cur.copy_expert(f"""
            COPY {table_name} (
                cement, slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age, strength,
                water_cement_ratio, binder, fine_to_coarse_ratio
            )
            FROM STDIN WITH CSV
        """, f)

    conn.commit()
    cur.close()
    conn.close()
    os.remove(TEMP_CSV_PATH)

    print("Données chargées avec succès dans PostgreSQL.")

def main():
    load_to_postgres_copy(CSV_PATH, TABLE_NAME)

if __name__ == "__main__":
    main()

