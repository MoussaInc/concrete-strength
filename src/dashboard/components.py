# src/dashboard/components.py

import streamlit as st
import pandas as pd
import requests
import os
from typing import List
import base64

# -----------------------
# Fonctions visuelles
# -----------------------

def load_custom_css(css_file: str = "src/dashboard/static/style.css"):
    """Charge un fichier CSS personnalisé pour le dashboard Streamlit."""
    try:
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Fichier CSS non trouvé, styles par défaut appliqués.")

def get_base64_of_bin_file(bin_file_path: str) -> str:
    """Convertit un fichier binaire en string Base64."""
    with open(bin_file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def display_logo(path="src/dashboard/static/images/logo.png", width=100, cv_url="https://www.poussaim.org"):
    if path and os.path.exists(path):
        logo_html = f"""
        <a href="{cv_url}" target="_blank">
            <img src="data:image/png;base64,{get_base64_of_bin_file(path)}" width="{width}">
        </a>
        """
        st.markdown(logo_html, unsafe_allow_html=True)
    else:
        st.warning("Logo non trouvé.")

def display_header(title: str = "Concrete Strength Predictor"):
    st.markdown(f"<h1 style='text-align:center'>{title}</h1>", unsafe_allow_html=True)

def display_footer():
    st.markdown("""
    <hr style="border-top: 2px solid #2E3B4E"/>
    <div style="text-align:center; color:gray; font-size: 0.9rem;">
        <a href="https://poussaim.org" target="_blank">Moussa.Inc</a> — © 2025 Concrete Strength Predictor - Tous droits réservés
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# Formulaire d'entrée
# -----------------------

def create_input_form(input_names):
    cols = st.columns(4)
    inputs = []
    for i, name in enumerate(input_names):
        val = cols[i % 4].number_input(
            name.replace("_", " ").capitalize(),
            value=st.session_state.get(name, 0.0),
            key=name,
            format="%.2f"
        )
        inputs.append(val)
    return inputs

def show_input_instructions():
    st.info(
        "🔹 Entrez les valeurs numériques pour chaque caractéristique du béton.\n"
        "🔹 Les valeurs doivent être réalistes selon les contraintes physiques."
    )
    st.markdown("""

    ### Pour réaliser une prédiction en batch, veuillez charger un fichier CSV contenant les colonnes suivantes :

    - **cement** : quantité de ciment (kg/m³)
    - **slag** : quantité de laitier (kg/m³)
    - **fly_ash** : quantité de cendre volante (kg/m³)
    - **water** : quantité d’eau (kg/m³)
    - **superplasticizer** : adjuvant superplastifiant (kg/m³)
    - **coarse_aggregate** : granulats grossiers (kg/m³)
    - **fine_aggregate** : sable ou granulats fins (kg/m³)
    - **age** : âge du béton (en jours)

    ⚠️ Les noms de colonnes doivent être strictement identiques à ceux listés ci-dessus.

    """)

def display_warnings(warnings: List):
    """
    Affiche les warnings dans Streamlit.
    - Si c'est une liste simple (str), affiche chaque warning.
    - Si c'est une liste de listes (batch), aplati et affiche tout.
    """
    if not warnings:
        return
    # Si c'est une liste de listes (batch)
    if any(isinstance(w, list) for w in warnings):
        flat = [w for sub in warnings for w in (sub if isinstance(sub, list) else [sub])]
    else:
        flat = warnings
    
    for w in flat:
        if isinstance(w, str) and w.strip():
            st.warning(w)

# -----------------------
# Logging usage
# -----------------------

def log_dashboard_usage(api_url: str, user_id: str):
    """
    Log l'utilisation du dashboard auprès de l'API.
    """
    try:
        payload = {"user_id": user_id}
        requests.post(f"{api_url}/log_dashboard_usage", json=payload, timeout=5)
    except Exception:
        st.warning("⚠️ Impossible de logger l'utilisation du dashboard.")


# -----------------------
# Préparation DataFrame pour batch ou évaluation
# -----------------------

BASE_COLS = ['cement', 'slag', 'fly_ash', 'water', 'superplasticizer',
             'coarse_aggregate', 'fine_aggregate', 'age']
DERIVED_COLS = ['water_cement_ratio', 'binder', 'fine_to_coarse_ratio']
ALL_FEATURES = BASE_COLS + DERIVED_COLS

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare un DataFrame pour la prédiction ML :
    - applique les contraintes physiques
    - remplit les colonnes manquantes avec 0
    - réordonne les colonnes
    """
    from src.utils.data_utils import apply_constraints
    df = apply_constraints(df.copy())

    # Compléter colonnes manquantes
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    df = df[ALL_FEATURES]
    return df
