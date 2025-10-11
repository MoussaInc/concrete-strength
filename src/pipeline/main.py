# src/pipeline/main.py

import subprocess
import sys
from pathlib import Path
import logging

# --- Configuration du Logging ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"

def setup_logging(level=logging.INFO):
    """
    Configuration du système de logging.
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


# --- Définition des Étapes du Pipeline ---
PIPELINE_STEPS = [
    # Etape 1: Téléchargement des données brutes
    {"name": "Téléchargement des données (download_data.py)", 
     "command": [sys.executable, str(PROJECT_ROOT / "src" / "etl" / "download_data.py"), "--dataset", "all"]},
    
    # Etape 2: Harmonisation et Fusion des 3 datasets
    {"name": "Harmonisation et Fusion (process_data.py)", 
     "command": [sys.executable, str(PROJECT_ROOT / "src" / "etl" / "process_data.py")]},
     
    # Etape 3: Nettoyage et Validation (Duplicats, Négatifs, Outliers par suppression)
    {"name": "Nettoyage final des données (clean_data.py)", 
     "command": [sys.executable, str(PROJECT_ROOT / "src" / "etl" / "clean_data.py")]},
     
    # Etape 4: Ingénierie des Features (Ratios, Clipping des Outliers)
    {"name": "Ingénierie de Features (feature_engineering.py)", 
     "command": [sys.executable, str(PROJECT_ROOT / "src" / "etl" / "feature_engineering.py")]},
     
    # Etape 5: Sauvegarde dans PostgreSQL
    {"name": "Sauvegarde dans PostgreSQL (save_to_db.py)", 
     "command": [sys.executable, str(PROJECT_ROOT / "src" / "etl" / "save_to_db.py")]},
]

def execute_step(step_info: dict) -> bool:
    """
    Exécute une commande de pipeline et gère les erreurs.
    """

    name = step_info["name"]
    command = step_info["command"]
    
    logger.info(f"\n=======================================================")
    logger.info(f"Début de l'étape : {name}")
    logger.info(f"   Commande: {' '.join(command)}")
    logger.info(f"=======================================================")
    
    try:
        # Exécute la commande et attend la fin
        # capture_output=False pour que les logs de chaque script s'affichent directement
        process = subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        
        if process.returncode == 0:
            logger.info(f"Succès de l'étape : {name}")
            return True
        else:
            # Ceci ne devrait pas être atteint avec check=True, mais c'est une sécurité
            logger.error(f"Échec de l'étape : {name}. Code de retour : {process.returncode}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.critical(f"ERREUR CRITIQUE dans l'étape : {name}")
        logger.critical(f"   La commande s'est terminée avec un code d'erreur : {e.returncode}")
        logger.critical("   Le pipeline est interrompu.")
        return False
    except FileNotFoundError:
        logger.critical(f"ERREUR CRITIQUE: Le script pour l'étape '{name}' est introuvable.")
        logger.critical("   Vérifiez les chemins dans PIPELINE_STEPS.")
        return False

def main():
    """
    Fonction principale qui exécute toutes les étapes du pipeline.
    """
    
    logger.info("\n=== DÉMARRAGE DU PIPELINE ETL COMPLET ===")
    
    all_success = True
    
    for step in PIPELINE_STEPS:
        if not execute_step(step):
            all_success = False
            break

    logger.info("\n=== RÉSULTAT FINAL DU PIPELINE ===")
    if all_success:
        logger.info("Le pipeline ETL complet s'est terminé avec SUCCÈS !")
        logger.info("Les données sont prêtes dans la base PostgreSQL.")
    else:
        logger.error("Le pipeline ETL s'est arrêté en raison d'une ERREUR. Veuillez vérifier les logs ci-dessus.")

if __name__ == "__main__":
    main()