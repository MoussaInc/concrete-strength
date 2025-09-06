# src/etl/1-download_data.py

import os
import pandas as pd
import requests

"""
Script de téléchargement de données brutes pour l'entraînement d'un modèle de Machine Learning sur la résistance en compresion simple du béton.

Fonctionnalités :
- Téléchargement du jeu de données par défaut depuis l'UCI Machine Learning Repository (format Excel).
- Possibilité pour l'utilisateur de fournir une URL personnalisée pointant vers un fichier au format `.xls`, `.xlsx` ou `.csv`.
- Détection automatique du type de fichier téléchargé.
- Conversion automatique des fichiers Excel en CSV pour faciliter le traitement en aval.
- Sauvegarde des fichiers dans le dossier 'data/raw/'.

Usage :
Lancer le script et suivre les instructions à l'écran pour choisir entre l'URL par défaut ou une URL personnalisée.
"""

# URL par défaut du dataset Excel
DEFAULT_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"

# Chemins
RAW_DIR = os.path.join("data", "raw")
XLS_PATH = os.path.join(RAW_DIR, "concrete_data.xls")
CSV_PATH = os.path.join(RAW_DIR, "concrete_data.csv")

def download_file(url: str, dest_path: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(response.content)
        print(f"Fichier téléchargé avec succès : {dest_path}")
    except Exception as e:
        print(f"Erreur lors du téléchargement : {e}")
        raise

def convert_excel_to_csv(xls_path: str, csv_path: str):
    try:
        df = pd.read_excel(xls_path)
        df.to_csv(csv_path, index=False)
        print(f"Conversion Excel --> CSV terminée : {csv_path}")
    except Exception as e:
        print(f"Erreur lors de la conversion Excel : {e}")
        raise

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Souhaitez-vous utiliser l'URL par défaut ou une URL personnalisée ?")
    print(f"[1] Oui (URL par défaut : {DEFAULT_DATA_URL})")
    print("[2] Non, je veux entrer mon URL (Excel ou CSV)")
    choice = input("Entrez 1 ou 2 : ").strip()

    if choice == "2":
        user_url = input("Entrez l'URL de votre fichier (.xls, .xlsx ou .csv) : ").strip()
        if not (user_url.lower().endswith(".xls") or user_url.lower().endswith(".xlsx") or user_url.lower().endswith(".csv")):
            print("L'URL ne pointe pas vers un fichier .xls, .xlsx ou .csv.")
            return
        data_url = user_url
    else:
        data_url = DEFAULT_DATA_URL

    file_extension = data_url.split(".")[-1].lower()

    print("Téléchargement du fichier...")

    if file_extension == "csv":
        download_file(data_url, CSV_PATH)
    elif file_extension in ["xls", "xlsx"]:
        download_file(data_url, XLS_PATH)
        print("Conversion du fichier Excel en CSV...")
        convert_excel_to_csv(XLS_PATH, CSV_PATH)
    else:
        print("Format de fichier non pris en charge.")

if __name__ == "__main__":
    main()
