## src/dashboard/components.py

import streamlit as st
import pandas as pd
import requests
import os
from typing import List, Dict, Any
import base64
from pathlib import Path
import json

INPUT_NAMES_CAMEL = [
    "Cement", "Slag", "FlyAsh", "Water", "Superplasticizer",
    "CoarseAggregate", "FineAggregate", "Age"
]

# --- Chemins statiques ---
STATIC_DIR = Path("src/dashboard/static")
CSS_FILE = STATIC_DIR / "style.css"
LOGO_PATH = STATIC_DIR / "images" / "logo.png"

# =================================
# Fonctions de Renommage et Préparation
# =================================

def normalize_column_name(name: str) -> str:
    """
    Normalise un nom de colonne en supprimant espaces, underscores, tirets
    et en mettant en minuscules pour comparaison robuste.
    
    Examples:
        'Fly Ash' -> 'flyash'
        'fly_ash' -> 'flyash'
        'FlyAsh' -> 'flyash'
        'coarse-aggregate' -> 'coarseaggregate'
    """
    return name.replace(" ", "").replace("_", "").replace("-", "").lower()


def ensure_camel_case(df: pd.DataFrame, required_names: list) -> pd.DataFrame:
    """
    Renomme les colonnes du DataFrame pour correspondre aux noms CamelCase requis, 
    en ignorant la casse, espaces, underscores et tirets.
    
    Args:
        df: DataFrame à renommer
        required_names: Liste des noms CamelCase attendus
    
    Returns:
        DataFrame avec colonnes renommées (copie)
    
    Examples:
        Colonne CSV 'fly ash' ou 'Fly_Ash' -> renommée en 'FlyAsh'
    """
    df = df.copy()
    name_map = {}
    
    # Créer le mappage normalisé (insensible à la casse, espaces, underscores)
    for required_name_camel in required_names:
        normalized_required = normalize_column_name(required_name_camel)
        for col in df.columns:
            normalized_col = normalize_column_name(col)
            if normalized_col == normalized_required:
                name_map[col] = required_name_camel
                break

    df.rename(columns=name_map, inplace=True)
    return df


def convert_df_to_api_json(df: pd.DataFrame) -> Dict[str, List[Dict[str, float]]]:
    """
    Convertit un DataFrame CSV en structure JSON compatible avec l'API.
    Applique automatiquement ensure_camel_case pour garantir la cohérence.
    
    Args:
        df: DataFrame contenant au minimum les 8 features (n'importe quelle casse)
    
    Returns:
        Dict au format {"samples": [{"Cement": 540.0, ...}, ...]}
    
    Raises:
        ValueError: Si des colonnes requises sont manquantes après conversion
    """
    # Renommage automatique pour supporter différentes casses/formats
    df = ensure_camel_case(df, INPUT_NAMES_CAMEL)
    
    # Vérification des colonnes manquantes
    missing_cols = [name for name in INPUT_NAMES_CAMEL if name not in df.columns]
    if missing_cols:
        # Message d'erreur détaillé pour aider l'utilisateur
        available_cols = list(df.columns)
        raise ValueError(
            f"❌ Colonnes manquantes après conversion : {', '.join(missing_cols)}\n\n"
            f"📋 Colonnes détectées dans votre CSV : {', '.join(available_cols)}\n\n"
            f"✅ Colonnes attendues (8 features) : {', '.join(INPUT_NAMES_CAMEL)}\n\n"
            f"💡 Formats acceptés : 'FlyAsh', 'Fly Ash', 'fly_ash', 'fly-ash', etc."
        )

    # Sélection uniquement des 8 features attendues par l'API
    df_filtered = df[INPUT_NAMES_CAMEL]
    
    # Création de la structure JSON de l'API
    json_payload = {
        "samples": df_filtered.to_dict(orient='records')
    }
    
    return json_payload


# =================================
# Fonctions Visuelles & Utilitaires
# =================================

def load_custom_css(css_file: Path = CSS_FILE):
    """Charge le fichier CSS personnalisé."""
    try:
        with open(css_file, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Fichier CSS non trouvé, styles par défaut appliqués.")


def get_base64_of_bin_file(bin_file_path: Path) -> str:
    """Encode un fichier binaire en base64."""
    with open(bin_file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def display_logo(path: Path = LOGO_PATH, width=100, cv_url="https://www.poussaim.org"):
    """Affiche le logo avec lien cliquable."""
    if path.exists():
        logo_html = f"""
        <a href="{cv_url}" target="_blank">
            <img src="data:image/png;base64,{get_base64_of_bin_file(path)}" width="{width}">
        </a>
        """
        st.markdown(logo_html, unsafe_allow_html=True)
    else:
        st.warning("Logo non trouvé.")


def display_header(title: str = "Concrete Strength Predictor"):
    """Affiche le titre centré de l'application."""
    st.markdown(f"<h1 style='text-align:center'>{title}</h1>", unsafe_allow_html=True)


def display_footer():
    """Affiche le footer de l'application."""
    st.markdown("""
    <hr style="border-top: 2px solid #2E3B4E"/>
    <div style="text-align:center; color:gray; font-size: 0.9rem;">
        <a href="https://poussaim.org" target="_blank">Moussa.Inc</a> — © 2025 Concrete Strength Predictor - Tous droits réservés
    </div>
    """, unsafe_allow_html=True)


def show_input_instructions(batch_cols: List[str]):
    """
    Affiche les instructions pour l'utilisation du dashboard.
    FIX: Message mis à jour pour refléter la flexibilité du système.
    """
    st.info(
        "🔹 Entrez les valeurs numériques des 8 caractéristiques du béton pour une prédiction unique.\n"
    )
    
    st.markdown("### Prédiction par fichier CSV (Batch)")
    st.markdown(f"""
    Pour réaliser une prédiction en batch, veuillez charger un fichier CSV contenant les colonnes suivantes :

    **{', '.join(batch_cols)}**

    ✅ **Le système accepte différents formats** :
    - Casses variées : `cement`, `Cement`, `CEMENT`
    - Avec espaces : `Fly Ash`, `Coarse Aggregate`, `Fine Aggregate`
    - Avec underscores : `fly_ash`, `coarse_aggregate`, `fine_aggregate`
    - Avec tirets : `fly-ash`, `coarse-aggregate`
    
    ⚠️ Assurez-vous que les 8 features sont présentes avec des noms reconnaissables.
    """)


def display_warnings(warnings: List):
    """
    Affiche les avertissements d'audit de manière formatée.
    Gère l'aplatissement des listes imbriquées.
    """
    # Aplatir les warnings (peut être une liste de listes)
    flat_warnings = []
    for item in warnings:
        if isinstance(item, list):
            flat_warnings.extend([w for w in item if w and isinstance(w, str)])
        elif item and isinstance(item, str):
            flat_warnings.append(item)
    
    if flat_warnings:
        st.markdown("---")
        st.error(f"**{len(flat_warnings)} Avertissement(s) d'Audit Détecté(s) :**")
        for w in flat_warnings:
            st.markdown(f"- {w}", unsafe_allow_html=True)
        st.markdown("---")


def display_warnings_by_line(warnings_list: List, title: str = "⚠️ Warnings par ligne"):
    """
    Affiche les warnings ligne par ligne de manière structurée.
    
    Args:
        warnings_list: Liste des warnings (un élément par ligne du dataset)
        title: Titre de la section
    """
    # Filtrer les lignes qui ont des warnings
    warnings_per_line = []
    for i, warning in enumerate(warnings_list):
        if warning:  # Si warning n'est pas None, vide ou []
            warnings_per_line.append((i + 1, warning))
    
    if warnings_per_line:
        st.markdown(f"### {title}")
        st.info(f"📋 {len(warnings_per_line)} ligne(s) avec des avertissements sur {len(warnings_list)} au total")
        
        # Affichage détaillé par ligne
        for line_num, warning in warnings_per_line:
            if isinstance(warning, list):
                # Si c'est une liste de warnings
                warning_text = " | ".join([str(w) for w in warning if w])
                if warning_text:
                    st.warning(f"**Ligne {line_num}** : {warning_text}")
            elif isinstance(warning, str) and warning.strip():
                st.warning(f"**Ligne {line_num}** : {warning}")
        
        st.markdown("---")


# ==================================
# Formulaire d'Entrée
# ==================================

def create_input_form(input_names: List[str]) -> Dict[str, float]:
    """
    Crée un formulaire d'entrée pour les 8 features du béton.
    La validation métier est gérée côté API.
    
    Args:
        input_names: Liste des noms de features (CamelCase)
    
    Returns:
        Dict avec les valeurs saisies {feature_name: value}
    """
    default_values = {
        "Cement": 540.0, 
        "Slag": 0.0, 
        "FlyAsh": 0.0, 
        "Water": 162.0, 
        "Superplasticizer": 2.5, 
        "CoarseAggregate": 1040.0, 
        "FineAggregate": 676.0, 
        "Age": 28.0
    }
    
    cols = st.columns(4)
    input_data = {}
    
    for i, name in enumerate(input_names):
        default_val = default_values.get(name, 0.0)
        min_val = 1.0 if name == "Age" else 0.0
        
        # Help text informatif
        help_text = None
        if name == "Cement":
            help_text = "⚠️ Si = 0, la prédiction sera 0 MPa (logique métier)"
        elif name == "Water":
            help_text = "⚠️ Si = 0, la prédiction sera 0 MPa (logique métier)"
        elif name == "Age":
            help_text = "⚠️ Minimum 1 jour"
        
        val = cols[i % 4].number_input(
            name, 
            value=st.session_state.get(name, default_val), 
            key=f"input_{name}", 
            format="%.2f", 
            min_value=min_val,
            help=help_text
        )
        input_data[name] = float(val)

    return input_data


# ========================
# Logging usage
# ========================

def validate_dataframe_batch(df: pd.DataFrame) -> tuple[bool, List[str]]:
    """
    Valide un DataFrame avant envoi batch à l'API.
    Vérifie les contraintes sur toutes les lignes.
    
    Args:
        df: DataFrame avec colonnes en CamelCase
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # Vérifier les contraintes strictement positives
    if "Cement" in df.columns:
        invalid_cement = df[df["Cement"] <= 0]
        if not invalid_cement.empty:
            errors.append(f"❌ **Cement** : {len(invalid_cement)} ligne(s) avec valeur ≤ 0")
    
    if "Water" in df.columns:
        invalid_water = df[df["Water"] <= 0]
        if not invalid_water.empty:
            errors.append(f"❌ **Water** : {len(invalid_water)} ligne(s) avec valeur ≤ 0")
    
    # Vérifier l'âge
    if "Age" in df.columns:
        invalid_age = df[df["Age"] < 1]
        if not invalid_age.empty:
            errors.append(f"❌ **Age** : {len(invalid_age)} ligne(s) avec valeur < 1 jour")
    
    # Vérifier les valeurs négatives pour les autres champs
    non_negative = ["Slag", "FlyAsh", "Superplasticizer", "CoarseAggregate", "FineAggregate"]
    for field in non_negative:
        if field in df.columns:
            invalid = df[df[field] < 0]
            if not invalid.empty:
                errors.append(f"❌ **{field}** : {len(invalid)} ligne(s) avec valeur < 0")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_input_features(features: Dict[str, float]) -> tuple[bool, List[str]]:
    """
    Valide les valeurs des features avant envoi à l'API.
    
    Args:
        features: Dictionnaire {feature_name: value}
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # Règles de validation (strictement positif pour tous sauf Age qui peut être >= 1)
    strictly_positive = ["Cement", "Water"]
    non_negative = ["Slag", "FlyAsh", "Superplasticizer", "CoarseAggregate", "FineAggregate"]
    min_age = 1.0
    
    # Validation des champs strictement positifs
    for field in strictly_positive:
        if field in features and features[field] <= 0:
            errors.append(f"❌ **{field}** doit être > 0 (actuellement: {features[field]})")
    
    # Validation des champs non négatifs
    for field in non_negative:
        if field in features and features[field] < 0:
            errors.append(f"❌ **{field}** doit être ≥ 0 (actuellement: {features[field]})")
    
    # Validation de l'âge
    if "Age" in features:
        if features["Age"] < min_age:
            errors.append(f"❌ **Age** doit être ≥ {min_age} jour(s) (actuellement: {features['Age']})")
    
    # Validations supplémentaires (règles métier)
    if "Water" in features and "Cement" in features and features["Cement"] > 0:
        water_cement_ratio = features["Water"] / features["Cement"]
        if water_cement_ratio > 1.0:
            errors.append(f"⚠️ **Ratio Eau/Ciment très élevé** : {water_cement_ratio:.2f} (> 1.0). Cela peut indiquer une erreur.")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def display_validation_errors(errors: List[str]):
    """Affiche les erreurs de validation de manière formatée."""
    if errors:
        st.error("### ⚠️ Erreurs de Validation")
        for error in errors:
            st.markdown(error)
        st.markdown("---")
        st.info("💡 **Conseil** : Corrigez les valeurs ci-dessus avant de relancer la prédiction.")


def log_dashboard_usage(api_url: str, user_id: str):
    """
    Log l'utilisation du dashboard auprès de l'API.
    Silencieux en cas d'échec (non critique).
    """
    try:
        payload = {"user_id": user_id}
        requests.post(f"{api_url}/log_dashboard_usage", json=payload, timeout=5) 
    except requests.exceptions.RequestException:
        pass  # Logging non critique, on ignore les erreurs
