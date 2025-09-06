from pydantic import BaseModel, conlist
from typing import List


# ========================
# 🔹 PREDICTION
# ========================

class PredictionInput(BaseModel):
    """
    Schéma d'entrée pour une prédiction unique ou batch.

    Attributs:
        features (List[float]): Liste de 8 valeurs numériques correspondant aux features de base :
                                cement, slag, fly_ash, water, superplasticizer,
                                coarse_aggregate, fine_aggregate, age.
                                Les features dérivées seront calculées automatiquement.
    """
    features: conlist(float, min_length=8, max_length=8)


class PredictionOutput(BaseModel):
    """
    Schéma de sortie pour une prédiction unique.

    Attributs:
        predicted_strength_MPa (float): Résultat de la prédiction en MPa.
        warnings (List[str]): Liste de messages d’avertissement pour la formulation.
    """
    predicted_strength_MPa: float
    warnings: List[str] = []


class BatchPredictionOutput(BaseModel):
    """
    Schéma de sortie pour une prédiction batch.

    Attributs:
        predicted_strengths_MPa (List[float]): Liste des prédictions en MPa pour chaque entrée.
        warnings (List[List[str]]): Liste des avertissements par ligne.
    """
    predicted_strengths_MPa: List[float]
    warnings: List[List[str]] = []


# ========================
# 🔹 EVALUATION
# ========================

class EvaluationInput(BaseModel):
    """
    Schéma d'entrée pour l'évaluation du modèle.

    Attributs:
        features (List[float]): Liste de 8 features de base.
        true_strength (float): Résistance réelle (MPa).
    """
    features: conlist(float, min_length=8, max_length=8)
    true_strength: float


class EvaluationOutput(BaseModel):
    """
    Schéma de sortie pour l'évaluation du modèle.

    Attributs:
        rmse (float): Root Mean Squared Error global.
        mae (float): Mean Absolute Error global.
        r2 (float): Coefficient de détermination R².
        n_samples (int): Nombre d'échantillons évalués.
    """
    rmse: float
    mae: float
    r2: float
    n_samples: int
