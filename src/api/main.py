# src/api/main.py

import os
import datetime
import traceback
from contextlib import asynccontextmanager
from typing import List, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func
from sklearn.metrics import mean_absolute_error, r2_score

from src.api.model_loader import load_model
from src.api.schemas import (
    ConcreteFeatures, PredictionInput, PredictionOutput,
    EvaluationInput, EvaluationOutput
)
from src.utils.database import get_db, UsageLog, create_tables
from src.utils.data_utils import (
    calculate_derived_features,
    apply_constraints_and_warnings,
    ALL_FEATURES,
)

# ==============================
# Helpers
# ==============================

model = None
model_load_error = None

def get_model():
    global model, model_load_error
    if model is None and model_load_error is None:
        try:
            print("[MODEL] Loading (lazy)...")
            model = load_model()
            print("[MODEL] Loaded")
        except Exception as e:
            model_load_error = e
            print(f"[MODEL] Load failed (lazy): {e}")
            raise
    return model


def _render_public_origin_from_host(host: str) -> str:
    """
    Convertit un host Render éventuel comme 'concrete-dashboard' en
    'https://concrete-dashboard.onrender.com'. Si 'host' a déjà un point,
    on le considère comme pleinement qualifié.
    """
    host = (host or "").strip()
    if not host:
        return ""
    if "." not in host:
        host = f"{host}.onrender.com"
    return f"https://{host}"

# ==============================
# Lifespan (init/shutdown)
# ==============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init DB
    try:
        create_tables()
        print("[DB] Tables ready")
    except Exception as e:
        print(f"[DB] Init skipped/failed: {e}")

    yield


app = FastAPI(
    title="Concrete Strength Prediction API",
    description="Prédiction & évaluation de la résistance à la compression du béton (avec règles métier).",
    version="2.0",
    lifespan=lifespan,
)

# ==============================
# CORS (Render + local)
# ==============================
dash_host = os.getenv("DASH_ORIGIN_HOST", "").strip()
origins = set()

if dash_host:
    origin = _render_public_origin_from_host(dash_host)
    origins.update({origin, origin.replace("https://", "http://")})
else:
    origins.update({"http://localhost:8501", "http://127.0.0.1:8501"})

origins.update({"http://localhost", "http://127.0.0.1"})

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print(f"[CORS] Allowed origins -> {list(origins)}")


# ==============================
# Modèles Pydantic utilitaires
# ==============================
class DashboardLog(BaseModel):
    user_id: str


# ==============================
# Règles métier
# ==============================
def apply_business_rules(samples: List[ConcreteFeatures]) -> Tuple[List[bool], List[List[str]]]:
    """
    Règles:
      - Cement <= 0  => béton invalide (résistance forcée à 0)
      - Water  <= 0  => béton invalide (résistance forcée à 0)
    """
    is_valid_list: List[bool] = []
    business_warnings_list: List[List[str]] = []

    for s in samples:
        ok = True
        warn: List[str] = []
        if s.Cement <= 0:
            ok = False
            warn.append("⛔ ERREUR CRITIQUE: Cement = 0 kg/m³. Un béton sans ciment n'existe pas (résistance = 0 MPa).")
        if s.Water <= 0:
            ok = False
            warn.append("⛔ ERREUR CRITIQUE: Water = 0 kg/m³. Un béton sans eau n'existe pas (résistance = 0 MPa).")
        is_valid_list.append(ok)
        business_warnings_list.append(warn)

    return is_valid_list, business_warnings_list


# ==============================
# Préparation des données
# ==============================
def process_batch_data(samples: List[ConcreteFeatures]) -> Tuple[np.ndarray, List[List[str]], List[bool]]:
    """
    Convertit les objets Pydantic en DataFrame, applique FE + audit, aligne sur ALL_FEATURES.
    Retourne (X_array, warnings_list, is_valid_flags)
    """
    # Règles métier
    is_valid_list, biz_warnings = apply_business_rules(samples)
    # En DataFrame
    df = pd.DataFrame([s.model_dump() for s in samples])
    # Feature engineering
    df = calculate_derived_features(df)
    # Audit (ajoute 'Warnings')
    df_audited = apply_constraints_and_warnings(df)
    # Alignement
    df_X = df_audited.reindex(columns=ALL_FEATURES, fill_value=np.nan)
    # Warnings audit
    audit_w = df_audited.get("Warnings", pd.Series([[]] * len(df_audited))).tolist()
    # Fusion des warnings
    warnings_list: List[List[str]] = []
    for w_biz, w_aud in zip(biz_warnings, audit_w):
        combined = (w_biz or []) + (w_aud if isinstance(w_aud, list) else [])
        warnings_list.append(combined)

    # Sanity check
    if len(df_X.columns) != len(ALL_FEATURES):
        raise HTTPException(
            status_code=500,
            detail={
                "msg": f"Alignement features échoué (attendu {len(ALL_FEATURES)}, obtenu {len(df_X.columns)})",
                "expected": ALL_FEATURES,
                "got": list(df_X.columns),
            },
        )

    return df_X.values, warnings_list, is_valid_list


# ==============================
# Logging usage
# ==============================
def log_api_request(endpoint: str, user_id: str, request: Request, db: Session):
    try:
        log_entry = UsageLog(
            timestamp=datetime.datetime.utcnow(),
            endpoint=endpoint,
            user_type="API" if user_id == "API" else "Dashboard",
            ip_address=(request.client.host if request and request.client else "Unknown"),
            user_id=user_id,
        )
        db.add(log_entry)
        db.commit()
    except Exception:
        print(f"[LOG] Error for {endpoint}: {traceback.format_exc()}")


# ==============================
# Endpoints utilitaires
# ==============================
@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", "health": "/health", "version": "2.0"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow(),
        "model_loaded": (model is not None),
        "model_error": str(model_load_error) if model_load_error else None,
    }

@app.post("/log_dashboard_usage")
def log_dashboard_usage_endpoint(data: DashboardLog, request: Request, db: Session = Depends(get_db)):
    log_api_request("/log_dashboard_usage", data.user_id, request, db)
    return {"status": "success", "message": "Dashboard usage logged."}

@app.get("/get_usage_count")
def get_usage_count(db: Session = Depends(get_db)):
    try:
        count = db.query(func.count(UsageLog.user_id.distinct())).scalar()
        return {"unique_users_count": count}
    except Exception:
        raise HTTPException(status_code=500, detail="Erreur lors du comptage des utilisateurs.")


# ==============================
# Endpoints ML
# ==============================
@app.post("/predict", response_model=PredictionOutput)
def predict_batch_json(input_data: PredictionInput, db: Session = Depends(get_db), request: Request = None):
    log_api_request("/predict", "API", request, db)

    try:
        m = get_model()
    except Exception:
        raise HTTPException(status_code=503, detail="Modèle indisponible sur le serveur.")

    try:
        X, warnings_list, is_valid = process_batch_data(input_data.samples)

        # Prédiction
        y_pred = np.array(m.predict(X), dtype=float)
        y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

        # Application règle métier (force 0 si invalide)
        final_preds = [0.0 if not ok else round(float(p), 3) for p, ok in zip(y_pred, is_valid)]

        return {"predicted_strengths_MPa": final_preds, "warnings": warnings_list}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[PREDICT] Unexpected error: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {e}")


@app.post("/evaluate", response_model=EvaluationOutput)
def evaluate_batch_json(input_data: EvaluationInput, db: Session = Depends(get_db), request: Request = None):
    log_api_request("/evaluate", "API", request, db)

    try:
        m = get_model()
    except Exception:
        raise HTTPException(status_code=503, detail="Modèle indisponible sur le serveur.")
  
    try:
        # Extraire features sans la cible
        samples_features: List[ConcreteFeatures] = [
            ConcreteFeatures.model_validate(s.model_dump(exclude={"true_strength"}))
            for s in input_data.samples
        ]
        y_true = np.array([s.true_strength for s in input_data.samples], dtype=float)

        X, warnings_list, is_valid = process_batch_data(samples_features)

        y_pred = np.array(m.predict(X), dtype=float)
        y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

        final_preds = np.array([0.0 if not ok else float(p) for p, ok in zip(y_pred, is_valid)], dtype=float)

        # Metrics
        rmse = float(np.sqrt(np.mean((y_true - final_preds) ** 2)))
        mae = float(mean_absolute_error(y_true, final_preds))
        r2 = float(r2_score(y_true, final_preds))

        preds_rounded = [round(float(p), 3) for p in final_preds]

        return {
            "rmse": round(rmse, 3),
            "mae": round(mae, 3),
            "r2": round(r2, 3),
            "n_samples": int(len(y_true)),
            "predicted_strengths_MPa": preds_rounded,
            "warnings": warnings_list,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[EVALUATE] Unexpected error: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Erreur lors de l'évaluation : {e}")
