# src/api/main.py

import datetime
import traceback
import pandas as pd
import numpy as np

from fastapi import FastAPI, Request, File, UploadFile, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.api.model_loader import load_model
from src.api.schemas import (
    PredictionInput, PredictionOutput, BatchPredictionOutput,
    EvaluationInput, EvaluationOutput
)
from src.utils.data_utils import apply_constraints
from src.utils.database import get_db, UsageLog, create_tables

# -------------------------
# Création des tables si nécessaire
# -------------------------
create_tables()

# -------------------------
# Initialisation FastAPI
# -------------------------
app = FastAPI(title="Concrete Strength Prediction API")

# -------------------------
# Chargement du modèle ML
# -------------------------
model = load_model()  # charge best_model.joblib (pipeline avec scaler)

# -------------------------
# Colonnes
# -------------------------
BASE_COLS = ["cement", "slag", "fly_ash", "water", "superplasticizer",
             "coarse_aggregate", "fine_aggregate", "age"]
DERIVED_COLS = ["water_cement_ratio", "binder", "fine_to_coarse_ratio"]
ALL_FEATURES = BASE_COLS + DERIVED_COLS

# ====================================
# 🔹 Logging centralisé
# ====================================
class DashboardLog(BaseModel):
    user_id: str

def log_usage(endpoint: str, user_type: str, user_id: str, request: Request = None, db: Session = None):
    """Fonction centralisée pour logger API et Dashboard."""
    try:
        ip_address = request.client.host if request else "Unknown"
        log_entry = UsageLog(
            timestamp=datetime.datetime.utcnow(),
            endpoint=endpoint,
            user_type=user_type,
            ip_address=ip_address,
            user_id=user_id
        )
        db.add(log_entry)
        db.commit()
        return True
    except Exception:
        traceback.print_exc()
        return False

def log_api_request(endpoint: str, request: Request = None, db: Session = None):
    """Logger simplifié pour les appels API."""
    return log_usage(endpoint=endpoint, user_type="API", user_id="API", request=request, db=db)

# -------------------------
# Endpoints Dashboard
# -------------------------
@app.post("/log_dashboard_usage")
async def log_dashboard_usage_endpoint(data: DashboardLog, request: Request, db: Session = Depends(get_db)):
    """Logger les utilisateurs du dashboard (retry automatique)."""
    success = False
    for _ in range(3):
        success = log_usage("/dashboard_usage", "Dashboard", data.user_id, request=request, db=db)
        if success:
            break
    if success:
        return {"status": "success", "message": "Dashboard usage logged."}
    raise HTTPException(status_code=500, detail="Erreur de logging du dashboard après plusieurs tentatives.")

@app.get("/get_usage_count")
async def get_usage_count(db: Session = Depends(get_db)):
    """Retourne le nombre d'utilisateurs uniques (Dashboard + API)."""
    try:
        count = db.query(func.count(UsageLog.user_id.distinct())).scalar()
        return {"unique_users_count": count}
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Erreur lors du comptage des utilisateurs.")

# -------------------------
# Endpoints généraux
# -------------------------
@app.get("/")
async def root():
    return {"message": "Concrete Strength Prediction API est en ligne 🚀"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow()}

# -------------------------
# Endpoint prédiction individuelle
# -------------------------
@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput, db: Session = Depends(get_db), request: Request = None):
    try:
        log_api_request("/predict", request, db)
        df = pd.DataFrame([input_data.features], columns=BASE_COLS)
        df = apply_constraints(df)
        # Préparer DataFrame avec toutes les features
        for col in DERIVED_COLS:
            if col not in df.columns:
                df[col] = 0.0
        df_ml = df[ALL_FEATURES]
        pred = float(model.predict(df_ml)[0])
        warnings = df.get("warnings", pd.Series([[]]))[0]
        if not isinstance(warnings, list):
            warnings = []
        return {"predicted_strength_MPa": round(pred,3), "warnings": warnings}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {e}")

# -------------------------
# Endpoint prédiction batch
# -------------------------
@app.post("/predict-batch", response_model=BatchPredictionOutput)
async def predict_batch(file: UploadFile = File(...), db: Session = Depends(get_db), request: Request = None):
    try:
        log_api_request("/predict-batch", request, db)
        df = pd.read_csv(file.file)
        missing_cols = [col for col in BASE_COLS if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400,
                                detail=f"Colonnes manquantes dans le fichier uploadé : {', '.join(missing_cols)}")
        df_final = apply_constraints(df)
        # Compléter colonnes dérivées
        for col in DERIVED_COLS:
            if col not in df_final.columns:
                df_final[col] = 0.0
        df_ml = df_final[ALL_FEATURES]
        preds = [round(float(p),3) for p in model.predict(df_ml)]
        warnings_list = df_final.get('warnings', pd.Series([[]]*len(df_final))).tolist()
        return {"predicted_strengths_MPa": preds, "warnings": warnings_list}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction batch : {e}")

# -------------------------
# Endpoint évaluation
# -------------------------
@app.post("/evaluate", response_model=EvaluationOutput)
async def evaluate(data: list[EvaluationInput], db: Session = Depends(get_db), request: Request = None):
    try:
        log_api_request("/evaluate", request, db)
        df = pd.DataFrame([d.features for d in data], columns=BASE_COLS)
        df = apply_constraints(df)
        for col in DERIVED_COLS:
            if col not in df.columns:
                df[col] = 0.0
        df_ml = df[ALL_FEATURES]
        y_true = [d.true_strength for d in data]
        y_pred = model.predict(df_ml)
        rmse = float(np.sqrt(np.mean((np.array(y_true)-y_pred)**2)))
        mae = float(np.mean(np.abs(np.array(y_true)-y_pred)))
        r2 = float(np.corrcoef(y_true, y_pred)[0,1]**2)
        return {"rmse": round(rmse,3), "mae": round(mae,3), "r2": round(r2,3), "n_samples": len(y_true)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de l'évaluation : {e}")
