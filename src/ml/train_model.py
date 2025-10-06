# src/ml/train_model.py

import numpy as np
import pandas as pd
from time import time
import joblib
import sys
import yaml
from pathlib import Path
import logging

# --- ML Frameworks ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# --- Configuration et Chemins ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = PROJECT_ROOT / "config" / "database.yaml"
LOG_DIR = PROJECT_ROOT / "logs"

# --- Constantes ---
TABLE_NAME = "engineered_concrete_data" # Nom de la table finale dans PostgreSQL
MODEL_PATH = MODELS_DIR / "best_model.joblib"

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

# --- Fonctions de Data Access ---

def load_db_config(config_path: Path):
    """Charge les identifiants de connexion PostgreSQL."""
    if not config_path.exists():
        logger.error(f"Fichier de configuration DB non trouvé: {config_path}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)['postgres']

def load_data_from_db(table_name: str) -> tuple:
    """Charge le dataset nettoyé et enrichi depuis PostgreSQL."""
    db_config = load_db_config(CONFIG_PATH)
    
    # Construction de l'URL de connexion SQLAlchemy
    db_url = (
        f"postgresql://{db_config['db_user']}:{db_config['db_password']}@"
        f"{db_config['db_host']}:{db_config['db_port']}/{db_config['db_name']}"
    )

    try:
        logger.info(f"Chargement des données depuis la table PostgreSQL : {table_name}")
        df = pd.read_sql_table(table_name, con=db_url)
    except Exception as e:
        logger.error(f"Erreur lors de la lecture de la base de données : {e}")
        sys.exit(1)

    # Nettoyage des colonnes non nécessaires pour le ML
    cols_to_drop = [col for col in ['id', 'Serial_No', 'Source'] if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Séparation des features (X) et de la cible (y)
    X = df.drop("Strength", axis=1)
    y = df["Strength"]
    
    logger.info(f"Dataset chargé : {len(df)} lignes, {len(X.columns)} features.")
    
    # Augmentation du test_size à 0.25 (25%) pour plus de robustesse à l'évaluation
    return train_test_split(X, y, test_size=0.25, random_state=42)

# --- Fonctions de Modélisation ---

def evaluate_model(model, X_test, y_test):
    """Évalue le modèle avec RMSE et MAE."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    return rmse, mae

def make_pipeline(model):
    """
    Crée un pipeline de ML incluant l'imputation des NaN (par la médiane) 
    et la standardisation (scaling).
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')), # Stable pour les NaN (ratios)
        ('scaler', StandardScaler()),                  # Essentiel pour la Régression Linéaire
        ('model', model)
    ])

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Entraîne et évalue les modèles avec une priorité donnée à la généralisation 
    (espace de recherche réduit et validation croisée).
    """
    results = {}

    # 1. Linear Regression (Baseline)
    logger.info("\nEntraînement de Linear Regression (Baseline)...")
    lr_pipe = make_pipeline(LinearRegression())
    lr_pipe.fit(X_train, y_train)
    rmse, mae = evaluate_model(lr_pipe, X_test, y_test)
    results['LinearRegression'] = {"model": lr_pipe, "rmse": rmse, "mae": mae }

    # 2. Random Forest (Optimisation avec max_depth réduit)
    # Réduction de l'espace de recherche pour un dataset de 900 lignes
    rf_params = {'model__n_estimators': [100, 200], 
                 'model__max_depth': [10, 15]} # max_depth limité pour éviter l'overfitting
    rf_pipe = make_pipeline(RandomForestRegressor(random_state=42, n_jobs=-1))
    logger.info("\nOptimisation de RandomForest (GridSearch modéré)...")
    t_start = time()
    # Utilisation de cv=5 pour une meilleure estimation des performances sur le petit dataset
    rf_grid = GridSearchCV(rf_pipe, rf_params, cv=5, n_jobs=-1, scoring='neg_root_mean_squared_error', verbose=0)
    rf_grid.fit(X_train, y_train)
    t_end = time()
    logger.info(f"RandomForest entraîné en {round(t_end - t_start, 2)} s")
    rmse, mae = evaluate_model(rf_grid.best_estimator_, X_test, y_test)
    results['RandomForest'] = {
        "model": rf_grid.best_estimator_,
        "rmse": rmse,
        "mae": mae,
        "best_params": rf_grid.best_params_
    }

    # 3. XGBoost (Optimisation avec max_depth très réduit)
    # Utilisation d'un learning_rate légèrement plus élevé pour une convergence rapide
    xgb_params = {'model__n_estimators': [100, 200],
                  'model__max_depth': [3, 5], # max_depth très faible (robustesse)
                  'model__learning_rate': [0.1, 0.2]}
    xgb_pipe = make_pipeline(XGBRegressor(random_state=42, eval_metric='rmse', tree_method='hist', n_jobs=-1))
    logger.info("\nOptimisation de XGBoost (GridSearch modéré)...")
    t_start = time()
    xgb_grid = GridSearchCV(xgb_pipe, xgb_params, cv=5, n_jobs=-1, scoring='neg_root_mean_squared_error', verbose=0)
    xgb_grid.fit(X_train, y_train)
    t_end = time()
    logger.info(f"XGBoost entraîné en {round(t_end - t_start, 2)} s")
    rmse, mae = evaluate_model(xgb_grid.best_estimator_, X_test, y_test)
    results['XGBoost'] = {
        "model": xgb_grid.best_estimator_,
        "rmse": rmse,
        "mae": mae,
        "best_params": xgb_grid.best_params_
    }

    return results

def main():
    logger.info("\n--- Démarrage de l'entraînement ML (Priorité Robustesse) ---")
    
    X_train, X_test, y_train, y_test = load_data_from_db(TABLE_NAME)

    logger.info("Entraînement des modèles et optimisation des hyperparamètres...")
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    logger.info("\n=======================")
    logger.info("  RÉSULTATS FINAUX")
    logger.info("=======================")
    
    # Affichage des résultats
    for name, info in results.items():
        logger.info(f"Modèle: {name} | RMSE: {info['rmse']:.2f} | MAE: {info['mae']:.2f}")
        if "best_params" in info:
            logger.info(f"  Meilleurs Params: {info['best_params']}")

    # --- Sauvegarde du meilleur modèle ---
    # Le modèle avec le plus petit RMSE est le meilleur (plus petit écart quadratique moyen)
    best_model_name = min(results, key=lambda k: results[k]['rmse'])
    best_model = results[best_model_name]['model']
    
    logger.info(f"\nMeilleur modèle sélectionné : {best_model_name} (RMSE={results[best_model_name]['rmse']:.2f})")
    
    joblib.dump(best_model, MODEL_PATH)
    logger.info(f"Pipeline de modèle sauvegardé dans : {MODEL_PATH}")

    # Sauvegarde JSON (pour les déploiements spécifiques à XGBoost)
    if best_model_name == 'XGBoost':
        MODEL_JSON = MODELS_DIR / "best_model.json"
        best_model.named_steps['model'].save_model(str(MODEL_JSON))
        logger.info(f"Modèle XGBoost sauvegardé en JSON dans : {MODEL_JSON}")

if __name__ == "__main__":
    main()