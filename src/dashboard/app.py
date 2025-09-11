# src/dashboard/app.py

import os
import requests
import pandas as pd
import streamlit as st
import time
import requests
from requests.exceptions import RequestException
from dotenv import load_dotenv
from components import (
    load_custom_css,
    display_logo,
    display_header,
    display_footer,
    create_input_form,
    call_prediction_api,
    call_batch_prediction_api,
    call_evaluation_api,
    show_input_instructions,
    display_warnings,
    log_dashboard_usage, 
)

# --- Configuration de la page ---
st.set_page_config(
    page_title="Concrete Strength Predictor",
    layout="wide",
    page_icon="🧱"
)

# --- Initialisation ---
load_custom_css()
display_logo()
display_header()

def wait_for_api(api_url, max_retries=30, delay=2):
    """Attend que l'API soit disponible"""
    for i in range(max_retries):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                return True
        except RequestException:
            if i == 0:  # Premier essai seulement
                st.info("🔄 Connexion à l'API en cours...")
            time.sleep(delay)
    return False

# Charger les variables d'environnement
load_dotenv()
# URL par défaut = Render (production)
DEFAULT_API_URL = "https://concrete-strength-rz69.onrender.com"

API_URL = os.getenv("API_URL", DEFAULT_API_URL)

st.sidebar.markdown(f"🌐 **API en cours d'utilisation :** `{API_URL}`")

if not wait_for_api(API_URL):
    st.error("Impossible de se connecter à l'API. Veuillez réessayer plus tard.")
    st.stop()

# Utilisation de st.session_state pour s'assurer que le log n'est fait qu'une seule fois par session
if "logged" not in st.session_state:
    log_dashboard_usage(API_URL)
    st.session_state["logged"] = True

# Affichage du nombre d'utilisateurs avec retry
try:
    response = requests.get(f"{API_URL}/get_usage_count", timeout=10)
    if response.status_code == 200:
        data = response.json()
        user_count = data.get("unique_users_count", "N/A")
        st.sidebar.markdown(f"👥 **Utilisateurs uniques :** `{user_count}`")
    else:
        st.sidebar.markdown(f"👥 **Utilisateurs uniques :** `Service en cours...`")
except requests.exceptions.RequestException:
    st.sidebar.markdown(f"👥 **Utilisateurs uniques :** `Non disponible`")

st.sidebar.markdown(f"🌐 API_URL détectée : `{API_URL}`")

INPUT_NAMES = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_aggregate", "fine_aggregate", "age"]

# Instructions pour batch et évaluation
show_input_instructions()

# --- Onglets principaux ---
tab1, tab2, tab3 = st.tabs(["Prédiction individuelle", "Prédiction en batch", "Évaluation modèle"])


# --------------------------------------
# 🔹 Onglet 1 : Prédiction individuelle
# --------------------------------------
with tab1:
    st.subheader("Entrez les paramètres du béton")
    features = create_input_form(INPUT_NAMES)

    if st.button("Prédire la résistance"):
        result = call_prediction_api(API_URL, features)
        if result["success"]:
            st.success(f"Résistance prédite : {result['value']:.3f} MPa")
            display_warnings(result.get("warnings", []))
        else:
            st.error(result["message"])

# -------------------------------
# 🔹 Onglet 2 : Prédiction batch
# -------------------------------
with tab2:
    st.subheader("Import d’un fichier CSV pour prédiction en lot")
    uploaded = st.file_uploader("Chargez un fichier CSV", type="csv")

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Erreur lecture CSV: {e}")
        else:
            if st.button("Lancer la prédiction batch"):
                result = call_batch_prediction_api(API_URL, uploaded)
                if result["success"]:
                    df["predicted_strength_MPa"] = result.get("predictions", [])
                    df["warnings"] = result.get("warnings", [])
                    st.success(f"{len(result['predictions'])} prédictions générées")
                    st.dataframe(df)
                    st.download_button(
                        "Télécharger les résultats",
                        df.to_csv(index=False),
                        "predictions_batch.csv",
                        "text/csv"
                    )
                    for idx, row_warnings in enumerate(result.get("warnings", [])):
                        if row_warnings:
                            st.warning(f"Ligne {idx+1} : {row_warnings}")
                else:
                    st.error(result["message"])


# -----------------------------------
# 🔹 Onglet 3 : Évaluation du modèle
# -----------------------------------
with tab3:
    st.subheader("Évaluer le modèle avec un fichier CSV")
    st.markdown("Le CSV doit contenir les colonnes de base + une colonne **`true_strength`**")
    eval_file = st.file_uploader("Chargez le fichier CSV d'évaluation", type="csv")

    if eval_file:
        try:
            df_eval = pd.read_csv(eval_file)
            st.dataframe(df_eval.head())
        except Exception as e:
            st.error(f"Erreur lecture CSV: {e}")
        else:
            if st.button("Lancer l'évaluation"):
                eval_list = [
                    {"features": row[INPUT_NAMES].tolist(), "true_strength": row["true_strength"]}
                    for _, row in df_eval.iterrows()
                ]
                result = call_evaluation_api(API_URL, eval_list)
                if result["success"]:
                    metrics = result["metrics"]
                    st.success("Évaluation terminée")
                    st.metric("RMSE", metrics["rmse"])
                    st.metric("MAE", metrics["mae"])
                    st.metric("R²", metrics["r2"])
                    st.write(f"Nombre d'échantillons évalués : {metrics['n_samples']}")
                else:
                    st.error(result["message"])


# -------------------------------
# 🔹 Footer
# -------------------------------
display_footer()
