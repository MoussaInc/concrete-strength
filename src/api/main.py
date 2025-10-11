# src/api/main.py - Avec Logique Métier Béton

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
create_tables()

app = FastAPI(
    title="Concrete Strength Prediction API",
    description="API pour la prédiction et l'évaluation de la résistance à la compression du béton avec logique métier.",
    version="2.0"
)

model = load_model()

# ====================================
# 🔹 LOGIQUE MÉTIER BÉTON
# ====================================

def apply_business_rules(samples: List[ConcreteFeatures]) -> tuple[List[bool], List[List[str]]]:
    """
    Applique les règles métier du béton sur les échantillons.
    
    Règles :
    - Si Cement = 0 OU Water = 0 → Béton invalide (résistance = 0 MPa)
    
    Args:
        samples: Liste d'échantillons ConcreteFeatures
    
    Returns:
        (is_valid_list, business_warnings_list)
        - is_valid_list: [True/False] pour chaque échantillon
        - business_warnings_list: [[warnings]] pour chaque échantillon
    """
    is_valid_list = []
    business_warnings_list = []
    
    for sample in samples:
        sample_warnings = []
        is_valid = True
        
        # Règle 1: Pas de ciment → Pas de béton
        if sample.Cement <= 0:
            is_valid = False
            sample_warnings.append("⛔ ERREUR CRITIQUE: Cement = 0 kg/m³. Un béton sans ciment n'existe pas (résistance = 0 MPa).")
        
        # Règle 2: Pas d'eau → Pas de béton
        if sample.Water <= 0:
            is_valid = False
            sample_warnings.append("⛔ ERREUR CRITIQUE: Water = 0 kg/m³. Un béton sans eau n'existe pas (résistance = 0 MPa).")
        
        is_valid_list.append(is_valid)
        business_warnings_list.append(sample_warnings)
    
    return is_valid_list, business_warnings_list


# ====================================
# 🔹 Fonctions d'Audit et de Conversion
# ====================================

def process_batch_data(samples: List[ConcreteFeatures], has_target: bool = False) -> tuple[pd.DataFrame, List[List[str]], List[bool]]:
    """
    Convertit les objets Pydantic en DataFrame, applique la FE et l'audit.
    Retourne le DataFrame prêt pour le modèle (X), les warnings et les flags de validité métier.
    """
    # Appliquer les règles métier AVANT tout traitement
    is_valid_list, business_warnings = apply_business_rules(samples)
    
    # Convertir en DataFrame
    data_dicts = [sample.model_dump() for sample in samples]
    df = pd.DataFrame(data_dicts)
    
    # FEATURE ENGINEERING
    df = calculate_derived_features(df)
    
    # AUDIT (Ajoute la colonne 'Warnings')
    df_audited = apply_constraints_and_warnings(df)
    
    # ALIGNEMENT pour le modèle
    df_X = df_audited.reindex(columns=ALL_FEATURES, fill_value=np.nan)
    
    # Extraction des warnings d'audit
    audit_warnings = df_audited.get('Warnings', pd.Series([[]]*len(df_audited))).tolist()
    
    # Fusion des warnings métier + audit
    combined_warnings = []
    for biz_warn, audit_warn in zip(business_warnings, audit_warnings):
        combined = biz_warn + (audit_warn if isinstance(audit_warn, list) else [])
        combined_warnings.append(combined)
    
    # Vérification
    if len(df_X.columns) != 12:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Erreur interne : Alignement des features échoué (attendu 12, obtenu {len(df_X.columns)})."
        )
    
    X_array = df_X.values
    return X_array, combined_warnings, is_valid_list


# ====================================
# 🔹 Logging centralisé
# ====================================
class DashboardLog(BaseModel):
    user_id: str

def log_api_request(endpoint: str, user_id: str, request: Request, db: Session):
    """Fonction centralisée pour logger l'usage de l'API/Dashboard."""
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
        print(f"Erreur de logging pour l'endpoint {endpoint}: {traceback.format_exc()}")
        pass

# -------------------------
# Endpoints Logging/Santé
# -------------------------
@app.post("/log_dashboard_usage")
async def log_dashboard_usage_endpoint(data: DashboardLog, request: Request, db: Session = Depends(get_db)):
    """Logger les utilisateurs du dashboard."""
    log_api_request("/log_dashboard_usage", data.user_id, request, db)
    return {"status": "success", "message": "Dashboard usage logged."}

@app.get("/get_usage_count")
async def get_usage_count(db: Session = Depends(get_db)):
    """Retourne le nombre d'utilisateurs uniques (Dashboard + API)."""
    try:
        count = db.query(func.count(UsageLog.user_id.distinct())).scalar()
        return {"unique_users_count": count}
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Erreur lors du comptage des utilisateurs."
        )

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API."""
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow()}

# ====================================
# 🔹 Endpoint de Prédiction (Batch JSON)
# ====================================
@app.post("/predict", response_model=PredictionOutput)
async def predict_batch_json(input_data: PredictionInput, db: Session = Depends(get_db), request: Request = None):
    """
    Effectue une prédiction (simple ou batch) sur un corps de requête JSON.
    Applique la logique métier : Si Cement=0 OU Water=0 → Prédiction=0 MPa + Warning.
    """
    log_api_request("/predict", "API", request, db)
    
    try:
        X_array, warnings_list, is_valid_list = process_batch_data(input_data.samples)
        
        # Prédiction par le modèle
        y_pred = model.predict(X_array)
        y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
        
        # Application de la logique métier : Forcer à 0 si échantillon invalide
        final_predictions = []
        for i, (pred, is_valid) in enumerate(zip(y_pred, is_valid_list)):
            if not is_valid:
                # Béton invalide (pas de ciment ou pas d'eau) → Résistance = 0
                final_predictions.append(0.0)
            else:
                final_predictions.append(round(float(pred), 3))
        
        return {
            "predicted_strengths_MPa": final_predictions, 
            "warnings": warnings_list
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Erreur inattendue dans /predict: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Erreur lors de la prédiction : {e}"
        )

# ====================================
# 🔹 Endpoint d'Évaluation (Batch JSON)
# ====================================
@app.post("/evaluate", response_model=EvaluationOutput)
async def evaluate_batch_json(input_data: EvaluationInput, db: Session = Depends(get_db), request: Request = None):
    """
    Évalue le modèle sur des données d'entrée (simple ou batch).
    Applique la même logique métier que /predict.
    """
    log_api_request("/evaluate", "API", request, db)
    
    try:
        # Extraction des features uniquement (sans true_strength)
        samples_features_only = [
            ConcreteFeatures.model_validate(s.model_dump(exclude={'true_strength'})) 
            for s in input_data.samples
        ]
        y_true = np.array([s.true_strength for s in input_data.samples])

        X_array, warnings_list, is_valid_list = process_batch_data(samples_features_only)
        
        # Prédiction par le modèle
        y_pred = model.predict(X_array)
        y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
        
        # Application de la logique métier
        final_predictions = []
        for pred, is_valid in zip(y_pred, is_valid_list):
            if not is_valid:
                final_predictions.append(0.0)
            else:
                final_predictions.append(float(pred))
        
        final_predictions = np.array(final_predictions)
        
        # Calcul des métriques
        rmse = float(np.sqrt(np.mean((y_true - final_predictions)**2)))
        mae = float(mean_absolute_error(y_true, final_predictions))
        r2 = float(r2_score(y_true, final_predictions))
        
        # Préparation des prédictions arrondies pour la sortie
        preds_rounded = [round(float(p), 3) for p in final_predictions]

        return {
            "rmse": round(rmse, 3), 
            "mae": round(mae, 3), 
            "r2": round(r2, 3), 
            "n_samples": len(y_true),
            "predicted_strengths_MPa": preds_rounded,
            "warnings": warnings_list
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Erreur inattendue dans /evaluate: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Erreur lors de l'évaluation : {e}"
        )