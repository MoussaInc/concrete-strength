# src/api/main.py

import datetime
import traceback
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from fastapi import FastAPI, Request, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func
from sklearn.metrics import mean_absolute_error, r2_score

from src.api.model_loader import load_model
from src.api.schemas import (
    ConcreteFeatures, PredictionInput, PredictionOutput, 
    EvaluationInput, EvaluationSample, EvaluationOutput
)
from src.utils.database import get_db, UsageLog, create_tables
from src.utils.data_utils import (
    calculate_derived_features, 
    apply_constraints_and_warnings, 
    ALL_FEATURES
)


# -------------------------
# Configuration
# -------------------------
# Création des tables si nécessaire
create_tables()

# Initialisation FastAPI
app = FastAPI(
    title="Concrete Strength Prediction API",
    description="API pour la prédiction et l'évaluation de la résistance à la compression du béton.",
    version="1.2"
)

# Chargement du modèle ML
model = load_model()  # charge best_model.joblib (pipeline avec scaler)

# ====================================
# 🔹 Fonctions d'Audit et de Conversion
# ====================================

def process_batch_data(samples: List[ConcreteFeatures], has_target: bool = False) -> tuple[pd.DataFrame, List[List[str]]]:
    """
    Convertit les objets Pydantic en DataFrame, applique la FE (FEATURE ENGINEERING ), et l'audit.
    Retourne le DataFrame prêt pour le modèle (X) et les warnings.
    """
    
    # Convertir la liste de schémas en DataFrame Pandas
    data_dicts = [sample.model_dump() for sample in samples]
    df = pd.DataFrame(data_dicts)
    # FEATURE ENGINEERING
    df = calculate_derived_features(df)
    # AUDIT (Ajoute la colonne 'Warnings')
    df_audited = apply_constraints_and_warnings(df)
    # ALIGNEMENT pour le modèle en réordonnant et ne conservant que les 12 features
    df_X = df_audited.reindex(columns=ALL_FEATURES, fill_value=np.nan)
    # Extraction des warnings
    warnings_list = df_audited.get('Warnings', pd.Series([[]]*len(df_audited))).tolist()
    
    # Vérification
    if len(df_X.columns) != 12:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                             detail=f"Erreur interne : Alignement des features échoué (attendu 12, obtenu {len(df_X.columns)}).")
    
    # Conversion en NumPy pour supprimer le UserWarning du SimpleImputer
    X_array = df_X.values

    return X_array, warnings_list


# ====================================
# 🔹 Logging centralisé (Simplifié)
# ====================================
class DashboardLog(BaseModel):
    user_id: str

def log_api_request(endpoint: str, user_id: str, request: Request, db: Session):
    """
    Fonction centralisée pour logger l'usage de l'API/Dashboard.
    """

    try:
        log_entry = UsageLog(
            timestamp=datetime.datetime.utcnow(),
            endpoint=endpoint,
            user_type="API" if user_id == "API" else "Dashboard",
            ip_address=request.client.host if request and request.client else "Unknown",
            user_id=user_id
        )
        db.add(log_entry)
        db.commit()
    except Exception:
        # Afficher la trace en cas d'erreur de DB, mais ne pas bloquer l'API
        print(f"Erreur de logging pour l'endpoint {endpoint}: {traceback.format_exc()}")
        pass # Ignorer l'erreur de logging pour ne pas impacter la prédiction

# -------------------------
# Endpoints Logging/Santé
# -------------------------
@app.post("/log_dashboard_usage")
async def log_dashboard_usage_endpoint(data: DashboardLog, request: Request, db: Session = Depends(get_db)):
    """
    Logger les utilisateurs du dashboard.
    """
    log_api_request("/log_dashboard_usage", data.user_id, request, db)
    return {"status": "success", "message": "Dashboard usage logged."}

@app.get("/get_usage_count")
async def get_usage_count(db: Session = Depends(get_db)):
    """
    Retourne le nombre d'utilisateurs uniques (Dashboard + API).
    """
    try:
        count = db.query(func.count(UsageLog.user_id.distinct())).scalar()
        return {"unique_users_count": count}
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Erreur lors du comptage des utilisateurs.")

@app.get("/health")
async def health_check():
    """
    Vérification de l'état de l'API.
    """
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow()}

# ====================================
# 🔹 Endpoint de Prédiction (Batch JSON)
# ====================================
@app.post("/predict", response_model=PredictionOutput)
async def predict_batch_json(input_data: PredictionInput, db: Session = Depends(get_db), request: Request = None):
    """
    Effectue une prédiction (simple ou batch) sur un corps de requête JSON.
    Utilise la logique FE centralisée.
    """
    log_api_request("/predict", "API", request, db)
    
    try:
        X_array, warnings_list = process_batch_data(input_data.samples)
        y_pred = model.predict(X_array)
        y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
        preds_rounded = [round(float(p), 3) for p in y_pred]

        return {
            "predicted_strengths_MPa": preds_rounded, 
            "warnings": warnings_list
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Erreur inattendue dans /predict: {traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Erreur lors de la prédiction : {e}")

# ====================================
# 🔹 Endpoint d'Évaluation (Batch JSON)
# ====================================
@app.post("/evaluate", response_model=EvaluationOutput)
async def evaluate_batch_json( input_data: EvaluationInput, db: Session = Depends(get_db), request: Request = None):
    """
    Évalue le modèle sur des données d'entrée (simple ou batch).
    """
    log_api_request("/evaluate", "API", request, db)
    
    try:
        samples_features_only = [
            ConcreteFeatures.model_validate(s.model_dump(exclude={'true_strength'})) 
            for s in input_data.samples
        ]
        y_true = np.array([s.true_strength for s in input_data.samples])

        X_array, warnings_list = process_batch_data(samples_features_only)
        y_pred = model.predict(X_array)
        y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
        rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred)) 

        return {
            "rmse": round(rmse, 3), 
            "mae": round(mae, 3), 
            "r2": round(r2, 3), 
            "n_samples": len(y_true),
            "warnings": warnings_list
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Erreur inattendue dans /evaluate: {traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Erreur lors de l'évaluation : {e}")
