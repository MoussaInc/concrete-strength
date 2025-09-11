import os
import pandas as pd
from time import time
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.data_utils import apply_constraints



"""
Ce script entraîne et évalue plusieurs modèles de régression (Régression Linéaire, Forêt Aléatoire, XGBoost)
pour prédire la résistance à la compression du béton à partir de ses caractéristiques.

Il utilise GridSearchCV pour optimiser les hyperparamètres de RandomForestRegressor et XGBRegressor,
et sélectionne automatiquement le meilleur modèle selon le RMSE.

Le modèle final est sauvegardé sous forme de fichier .joblib.

Fichier attendu : data/processed/concrete_data_clean.csv
Variable cible : 'strength'
"""

# --- Constantes ---
DATA_PATH = "data/processed/concrete_data_clean.csv"
MODEL_PATH = "models/best_model.joblib"

BASE_COLS = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_aggregate", "fine_aggregate", "age"]
DERIVED_COLS = ["water_cement_ratio", "binder", "fine_to_coarse_ratio"]
ALL_FEATURES = BASE_COLS + DERIVED_COLS

def load_data(path):
    """
    Charge les données depuis un fichier CSV et divise le jeu de données en ensembles d'entraînement et de test.

    Args:
        path (str): Chemin vers le fichier CSV contenant les données prétraitées.

    Returns:
        X_train, X_test, y_train, y_test: Jeux de données séparés pour l'entraînement et le test.
    """

    df = pd.read_csv(path)
    df = apply_constraints(df)

    # Exclure la colonne 'warnings' si présente
    if 'warnings' in df.columns:
        df = df.drop(columns=['warnings'])

    # Séparer features et target
    X = df.drop("strength", axis=1)
    y = df["strength"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_model(model, X_test, y_test):
    """
    Évalue un modèle de régression à l'aide du RMSE et du MAE.

    Args:
        model: Modèle entraîné.
        X_test (DataFrame): Données de test.
        y_test (Series): Cible réelle.

    Returns:
        tuple: RMSE et MAE du modèle.
    """
    
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    return rmse, mae

def make_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Entraîne plusieurs modèles (Régression Linéaire, Random Forest, XGBoost),
    optimise les hyperparamètres pour RandomForest et XGBoost,
    et retourne leurs performances.

    Args:
        X_train, X_test, y_train, y_test: Jeux de données.

    Returns:
        dict: Résultats pour chaque modèle avec le modèle entraîné, RMSE, MAE et meilleurs paramètres le cas échéant.
    """

    results = {}

    # --- Régression Linéaire ---
    lr_pipe = make_pipeline(LinearRegression())
    lr_pipe.fit(X_train, y_train)
    rmse, mae = evaluate_model(lr_pipe, X_test, y_test)
    results['LinearRegression'] = {"model": lr_pipe, "rmse": rmse, "mae": mae }

    # --- Random Forest ---
    rf_params = {'model__n_estimators': [50, 100, 200], 'model__max_depth': [None, 10, 20]}
    rf_pipe = make_pipeline(RandomForestRegressor(random_state=42))
    print("\nOptimisation de RandomForest...")
    t_start = time()
    rf_grid = GridSearchCV(rf_pipe, rf_params, cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error')
    rf_grid.fit(X_train, y_train)
    t_end = time()
    print(f"RandomForest entraîné en {round(t_end - t_start, 2)} secondes.")
    rmse, mae = evaluate_model(rf_grid.best_estimator_, X_test, y_test)
    results['RandomForest'] = {
        "model": rf_grid.best_estimator_,
        "rmse": rmse,
        "mae": mae,
        "best_params": rf_grid.best_params_
    }

    # --- XGBoost ---
    # Ajout du paramètre tree_method='hist' pour éviter l'utilisation du GPU
    xgb_params = {'model__n_estimators': [50, 100, 200], 'model__max_depth': [3, 5], 'model__learning_rate': [0.05, 0.1, 0.2]}
    xgb_pipe = make_pipeline(XGBRegressor(random_state=42, eval_metric='rmse', tree_method='hist'))
    print("\nOptimisation de XGBoost...")
    t_start = time()
    xgb_grid = GridSearchCV(xgb_pipe, xgb_params, cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error')
    xgb_grid.fit(X_train, y_train)
    t_end = time()
    print(f"XGBoost entraîné en {round(t_end - t_start, 2)} secondes.")
    rmse, mae = evaluate_model(xgb_grid.best_estimator_, X_test, y_test)
    results['XGBoost'] = {
        "model": xgb_grid.best_estimator_,
        "rmse": rmse,
        "mae": mae,
        "best_params": xgb_grid.best_params_
    }

    return results

def main():
    print("\nChargement des données...")
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)

    print("\nEntraînement des modèles...")
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\nRésultats des modèles :")
    for name, info in results.items():
        print(f"{name} | RMSE: {info['rmse']:.2f} | MAE: {info['mae']:.2f}")
        if "best_params" in info:
            print(f"Best Params: {info['best_params']} \n")

    # --- Sauvegarde du meilleur modèle ---
    best_model_name = min(results, key=lambda k: results[k]['rmse'])
    best_model = results[best_model_name]['model']
    print(f"\nMeilleur modèle : {best_model_name} (RMSE = {results[best_model_name]['rmse']:.2f})")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Modèle sauvegardé dans : {MODEL_PATH}\n")

if __name__ == "__main__":
    main()
