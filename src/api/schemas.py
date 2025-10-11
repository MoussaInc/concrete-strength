from pydantic import BaseModel, Field, conlist
from typing import List, Optional

# --- Constantes pour la validation ---
MIN_AGE = 1.0  # Âge minimal typique pour la mesure (jours)
MIN_NON_NEGATIVE = 0.0  # Tous les composants doivent être >= 0 (logique métier appliquée côté API)

# ========================
# 🔹 FEATURES DE BASE
# ========================

class ConcreteFeatures(BaseModel):
    """
    Schéma des 8 features de base du béton. 
    Les noms sont en CamelCase pour la cohérence avec le modèle ML.
    
    Validation technique :
    - Tous les composants : >= 0 (pas de valeurs négatives)
    - Age : >= 1.0 jour
    
    Logique métier (appliquée dans l'API) :
    - Si Cement = 0 OU Water = 0 → Prédiction = 0 MPa + Warning
    """
    # Tous les composants acceptent >= 0 (la logique métier est dans l'API)
    Cement: float = Field(
        ..., 
        ge=MIN_NON_NEGATIVE,
        description="Quantité de ciment (kg/m³). Si = 0, la résistance sera nulle (logique métier)."
    )
    
    Slag: float = Field(
        MIN_NON_NEGATIVE, 
        ge=MIN_NON_NEGATIVE,
        description="Quantité de laitier de haut-fourneau (Slag) (kg/m³)."
    )
    
    FlyAsh: float = Field(
        MIN_NON_NEGATIVE, 
        ge=MIN_NON_NEGATIVE, 
        description="Quantité de cendres volantes (Fly Ash) (kg/m³)."
    )
    
    Water: float = Field(
        ..., 
        ge=MIN_NON_NEGATIVE,
        description="Quantité d'eau (kg/m³). Si = 0, la résistance sera nulle (logique métier)."
    )
    
    Superplasticizer: float = Field(
        MIN_NON_NEGATIVE, 
        ge=MIN_NON_NEGATIVE, 
        description="Quantité de superplastifiant (kg/m³)."
    )
    
    CoarseAggregate: float = Field(
        MIN_NON_NEGATIVE, 
        ge=MIN_NON_NEGATIVE, 
        description="Quantité de gros agrégats (kg/m³)."
    )
    
    FineAggregate: float = Field(
        MIN_NON_NEGATIVE, 
        ge=MIN_NON_NEGATIVE, 
        description="Quantité de fines agrégats (kg/m³)."
    )
    
    Age: float = Field(
        ..., 
        ge=MIN_AGE, 
        description=f"Âge du béton (jours), supérieur ou égal à {MIN_AGE} jour."
    )

# ========================
# 🔹 PREDICTION
# ========================

class PredictionInput(BaseModel):
    """
    Schéma d'entrée pour une prédiction. Accepte une liste d'échantillons ConcreteFeatures.
    """
    samples: List[ConcreteFeatures]
    
    class Config:
        schema_extra = {
            "example": {
                "samples": [
                    {
                        "Cement": 540.0, 
                        "Slag": 0.0, 
                        "FlyAsh": 0.0, 
                        "Water": 162.0, 
                        "Superplasticizer": 2.5, 
                        "CoarseAggregate": 1040.0, 
                        "FineAggregate": 676.0, 
                        "Age": 28.0
                    }
                ]
            }
        }


class PredictionOutput(BaseModel):
    """
    Schéma de sortie pour une prédiction batch.
    """
    predicted_strengths_MPa: List[float] = Field(
        ..., 
        description="Liste des prédictions de résistance en MPa."
    )
    warnings: List[List[str]] = Field(
        ..., 
        description="Liste des avertissements (audit métier) pour chaque échantillon."
    )


# ========================
# 🔹 EVALUATION
# ========================

class EvaluationSample(ConcreteFeatures):
    """
    Schéma d'entrée pour l'évaluation, hérite des features et ajoute la valeur réelle.
    """
    true_strength: float = Field(
        ..., 
        ge=MIN_NON_NEGATIVE,
        description="Résistance réelle mesurée (MPa)."
    )


class EvaluationInput(BaseModel):
    """
    Schéma d'entrée pour l'évaluation du modèle (accepte un batch).
    """
    samples: List[EvaluationSample]

    class Config:
        schema_extra = {
            "example": {
                "samples": [
                    {
                        "Cement": 540.0, 
                        "Slag": 0.0, 
                        "FlyAsh": 0.0, 
                        "Water": 162.0, 
                        "Superplasticizer": 2.5, 
                        "CoarseAggregate": 1040.0, 
                        "FineAggregate": 676.0, 
                        "Age": 28.0, 
                        "true_strength": 79.99 
                    }
                ]
            }
        }


class EvaluationOutput(BaseModel):
    """
    Schéma de sortie pour l'évaluation globale du modèle.
    """
    rmse: float = Field(..., description="Root Mean Squared Error (RMSE) global.")
    mae: float = Field(..., description="Mean Absolute Error (MAE) global.")
    r2: float = Field(..., description="Coefficient de détermination R² global.")
    n_samples: int = Field(..., ge=1, description="Nombre d'échantillons évalués.")
    
    predicted_strengths_MPa: Optional[List[float]] = Field(
        None, 
        description="Liste des prédictions individuelles (MPa) pour chaque échantillon."
    )
    
    warnings: Optional[List[List[str]]] = Field(
        None, 
        description="Liste des avertissements pour chaque échantillon d'entrée."
    )