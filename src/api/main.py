from fastapi import (
    FastAPI, File, HTTPException, UploadFile, Depends, Request
)
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy import func
from sqlalchemy.orm import Session
import traceback
import pandas as pd
import numpy as np
import datetime
import os

# Import des scripts existants
from src.utils.data_utils import apply_constraints
from src.api.model_loader import load_model
from src.api.schemas import (
    PredictionInput,
    PredictionOutput,
    BatchPredictionOutput,
    EvaluationInput,
    EvaluationOutput,
)

# Import des dépendances de BDD
from src.utils.database import get_db, UsageLog 

app = FastAPI(title="Concrete Strength Prediction API")

# Chargement le modèle ML au démarrage
model = load_model()

# Colonnes de base et dérivées
BASE_COLS = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_aggregate", "fine_aggregate", "age"]
DERIVED_COLS = ["water_cement_ratio", "binder", "fine_to_coarse_ratio"]
ALL_FEATURES = BASE_COLS + DERIVED_COLS

# ====================================
# 🔹 NOUVEL ENDPOINT pour le dashboard
# ====================================

class DashboardLog(BaseModel):
    user_id: str

@app.post("/log_dashboard_usage")
async def log_dashboard_usage(data: DashboardLog, request: Request, db: Session = Depends(get_db)):
    try:
        log_entry = UsageLog(
            timestamp=datetime.datetime.utcnow(),
            endpoint="/dashboard_usage",
            user_type="Dashboard",
            ip_address=request.client.host,
            user_id=data.user_id
        )
        db.add(log_entry)
        db.commit()
        return {"status": "success", "message": "Dashboard usage logged."}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur de logging: {e}")

@app.get("/get_usage_count")
async def get_usage_count(db: Session = Depends(get_db)):
    try:
        unique_users_count = db.query(func.count(UsageLog.user_id.distinct())).scalar()
        return {"unique_users_count": unique_users_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erreur interne lors du comptage des utilisateurs.")

# ========================
# 🔹 ENDPOINTS
# ========================
@app.get("/")
async def root():
    return {"message": "Concrete Strength Prediction API est en ligne 🚀"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput, db: Session = Depends(get_db), request: Request = None):
    """
    Prédiction unique à partir des features de base.
    """
    try:
        if request:
            client_ip = request.headers.get("X-Forwarded-For", request.client.host)
            log_entry = UsageLog(
                timestamp=datetime.datetime.utcnow(),
                endpoint="/predict",
                user_type="API",
                ip_address=client_ip
            )
            db.add(log_entry)
            db.commit()

        df = pd.DataFrame([input_data.features], columns=BASE_COLS)
        df = apply_constraints(df)
        df_ml = df[ALL_FEATURES]
        prediction = model.predict(df_ml)[0]
        
        warnings = df["warnings"].iloc[0]
        if not isinstance(warnings, list):
            warnings = []

        return {
            "predicted_strength_MPa": round(float(prediction), 3),
            "warnings": warnings
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {e}")


@app.post("/predict-batch", response_model=BatchPredictionOutput)
async def predict_batch(file: UploadFile = File(...), db: Session = Depends(get_db), request: Request = None):
    """
    Prédiction batch à partir d'un fichier CSV uploadé.
    Retourne les prédictions et les warnings par ligne.
    """
    try:
        client_ip = request.headers.get("X-Forwarded-For", request.client.host)
        log_entry = UsageLog(
            timestamp=datetime.datetime.utcnow(),
            endpoint="/predict-batch",
            user_type="API",
            ip_address=client_ip
        )
        db.add(log_entry)
        db.commit()
        df = pd.read_csv(file.file)
        missing_cols = [col for col in BASE_COLS if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Colonnes manquantes dans le fichier uploadé : {', '.join(missing_cols)}"
            )

        df_final = apply_constraints(df)
        df_ml = df_final[ALL_FEATURES]
        predictions = model.predict(df_ml)
        preds = [round(float(p), 3) for p in predictions]
        warnings_list = df_final['warnings'].tolist()
        return {
            "predicted_strengths_MPa": preds,
            "warnings": warnings_list
        }

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Le fichier uploadé est vide ou invalide.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction batch : {e}")

@app.post("/evaluate", response_model=EvaluationOutput)
async def evaluate(data: list[EvaluationInput], db: Session = Depends(get_db), request: Request = None):
    """
    Évalue le modèle sur un batch de données avec valeurs réelles.
    """
    try:
        client_ip = request.headers.get("X-Forwarded-For", request.client.host)
        log_entry = UsageLog(
            timestamp=datetime.datetime.utcnow(),
            endpoint="/evaluate",
            user_type="API",
            ip_address=client_ip
        )
        db.add(log_entry)
        db.commit()
        
        df = pd.DataFrame([d.features for d in data], columns=BASE_COLS)
        df = apply_constraints(df)
        df_ml = df[ALL_FEATURES]
        y_true = [d.true_strength for d in data]

        y_pred = model.predict(df_ml)

        rmse = float(np.sqrt(np.mean((np.array(y_true) - y_pred) ** 2)))
        mae = float(np.mean(np.abs(np.array(y_true) - y_pred)))
        r2 = float(np.corrcoef(y_true, y_pred)[0, 1] ** 2)

        return {
            "rmse": round(rmse, 3),
            "mae": round(mae, 3),
            "r2": round(r2, 3),
            "n_samples": len(y_true),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de l'évaluation : {e}")
