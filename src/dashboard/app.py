# src/dashboard/app.py - Version Corrigée

import os
import uuid
import pandas as pd
import streamlit as st
import requests
from dotenv import load_dotenv
import numpy as np

from components import (
    load_custom_css,
    display_logo,
    display_header,
    display_footer,
    create_input_form,
    show_input_instructions,
    display_warnings,
    display_warnings_by_line,
    log_dashboard_usage,
    INPUT_NAMES_CAMEL, 
    convert_df_to_api_json,
    ensure_camel_case
)

# --- Configuration de la page ---
st.set_page_config(
    page_title="Concrete Strength Predictor",
    layout="wide",
    page_icon="🧱"
)

# --- Initialisation visuelle ---
load_custom_css()
display_logo()
display_header()

# --- Configuration API ---
load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
st.sidebar.markdown(f"🌐 **API utilisée :** `{API_URL}`")

# --- Identifiant utilisateur et Logging ---
if "user_uuid" not in st.session_state:
    st.session_state["user_uuid"] = str(uuid.uuid4())
USER_ID = st.session_state["user_uuid"]

if "logged" not in st.session_state:
    log_dashboard_usage(API_URL, USER_ID)
    st.session_state["logged"] = True

# --- Affichage du nombre d'utilisateurs ---
try:
    response = requests.get(f"{API_URL}/get_usage_count", timeout=5)
    if response.status_code == 200:
        user_count = response.json().get("unique_users_count", "N/A")
        st.sidebar.markdown(f"👥 **Utilisateurs uniques :** `{user_count}`")
    else:
        st.sidebar.markdown("👥 **Utilisateurs uniques :** `Service en cours...`")
except requests.exceptions.RequestException:
    st.sidebar.markdown("👥 **Utilisateurs uniques :** `Non disponible`")

# --- Instructions ---
show_input_instructions(INPUT_NAMES_CAMEL)

# --- Onglets ---
tab1, tab2, tab3 = st.tabs(["Prédiction unique", "Prédiction batch (CSV)", "Évaluation batch"])

# =======================================
# Onglet 1 : Prédiction unique
# =======================================
with tab1:
    st.subheader("Entrez les paramètres du béton (8 features)")
    input_features_dict = create_input_form(INPUT_NAMES_CAMEL)

    if st.button("Prédire la résistance"):
        try:
            payload = {"samples": [input_features_dict]}
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            if response.status_code == 200:
                result = response.json()
                pred = result['predicted_strengths_MPa'][0]
                warnings = result.get("warnings", [[]])[0]
                st.success(f"Résistance prédite : **{pred:.3f} MPa**")
                display_warnings(warnings) 
            else:
                st.error(f"❌ Erreur API ({response.status_code}): {response.text}")
        except Exception as e:
            st.error(f"❌ Erreur lors de l'appel API : {e}")

# =======================================
# Onglet 2 : Prédiction batch (CSV)
# =======================================
with tab2:
    st.subheader("Import d'un fichier CSV pour prédiction en lot")
    uploaded_file = st.file_uploader("Chargez un fichier CSV", type="csv", key="batch_upload")

    if uploaded_file:
        df_batch_original = pd.read_csv(uploaded_file)
        st.dataframe(df_batch_original.head())

        if st.button("Lancer la prédiction batch"):
            try:
                # FIX: convert_df_to_api_json applique déjà ensure_camel_case
                # On travaille sur une copie pour ne pas modifier l'original
                payload = convert_df_to_api_json(df_batch_original.copy())
                
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    preds = result["predicted_strengths_MPa"]
                    warnings_list = result["warnings"]
                    
                    # FIX: On crée un nouveau DataFrame avec TOUTES les colonnes originales
                    # + les résultats, sans dupliquer le renommage
                    df_results = df_batch_original.copy()
                    df_results["Predicted_Strength_MPa"] = preds
                    df_results["Warnings_Audit"] = [str(w) for w in warnings_list]

                    st.success(f"✅ {len(df_results)} prédictions générées.")
                    st.dataframe(df_results)
                    display_warnings(warnings_list)
                    
                    st.download_button(
                        "⬇️ Télécharger les résultats", 
                        df_results.to_csv(index=False), 
                        "predictions_batch_results.csv", 
                        "text/csv"
                    )
                else:
                    st.error(f"❌ Erreur API ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"❌ Erreur lors de l'appel API : {e}")

# =======================================
# Onglet 3 : Évaluation modèle
# =======================================
with tab3:
    st.subheader("Évaluer le modèle sur des données labellisées")
    st.markdown(
        "Le CSV doit contenir les 8 colonnes de base (n'importe quelle casse acceptée) "
        "**+** la colonne de cible (`true_strength`, `strength` ou `Strength`)."
    )
    eval_file = st.file_uploader("Chargez le fichier CSV d'évaluation", type="csv", key="eval_upload")
    
    if eval_file:
        df_eval_original = pd.read_csv(eval_file)
        st.dataframe(df_eval_original.head())
        
        if st.button("Lancer l'évaluation"):
            try:
                # Copie pour travailler sans modifier l'original
                df_eval = df_eval_original.copy()
                
                # Renommage des 8 features
                df_eval = ensure_camel_case(df_eval, INPUT_NAMES_CAMEL)
                
                # Normalisation de la colonne cible
                if 'true_strength' not in df_eval.columns:
                    if 'strength' in df_eval.columns:
                        df_eval.rename(columns={'strength': 'true_strength'}, inplace=True)
                    elif 'Strength' in df_eval.columns:
                        df_eval.rename(columns={'Strength': 'true_strength'}, inplace=True)
                    else:
                        st.error(
                            "❌ Le fichier doit contenir la colonne de résistance cible, "
                            "nommée 'true_strength', 'strength' ou 'Strength'."
                        )
                        st.stop()

                # Sélection des colonnes nécessaires
                df_temp = df_eval[INPUT_NAMES_CAMEL + ['true_strength']].copy()
                
                # Création du payload
                payload_samples = df_temp.to_dict(orient='records')
                payload = {"samples": payload_samples}
                
                # Appel à l'API /evaluate
                response = requests.post(f"{API_URL}/evaluate", json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    
                    # Affichage des métriques
                    st.success(f"✅ Évaluation réussie sur {result['n_samples']} échantillons.")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("RMSE", f"{result['rmse']:.3f}")
                    col2.metric("MAE", f"{result['mae']:.3f}")
                    col3.metric("R²", f"{result['r2']:.3f}")
                    
                    # Récupération des prédictions et warnings
                    predictions = result.get("predicted_strengths_MPa", [])
                    warnings_list = result.get("warnings", [])
                    
                    # Création du DataFrame de résultats avec toutes les colonnes originales
                    df_results = df_eval_original.copy()
                    
                    # Ajout des prédictions (si disponibles)
                    if predictions:
                        df_results["Predicted_Strength_MPa"] = predictions
                    
                    # Ajout de la colonne Warnings_Audit
                    df_results["Warnings_Audit"] = [str(w) if w else "" for w in warnings_list]
                    
                    # Affichage du DataFrame avec les résultats
                    st.markdown("### 📊 Résultats détaillés par ligne")
                    st.dataframe(df_results)
                    
                    # Affichage des warnings par ligne avec la fonction dédiée
                    display_warnings_by_line(warnings_list, "⚠️ Avertissements détectés par ligne")
                    
                    # Bouton de téléchargement
                    st.download_button(
                        "⬇️ Télécharger les résultats d'évaluation",
                        df_results.to_csv(index=False),
                        "evaluation_results.csv",
                        "text/csv"
                    )
                    
                else:
                    st.error(f"❌ Erreur API ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"❌ Erreur lors de l'appel API : {e}")

# --- Footer ---
display_footer()