# src/dashboard/app.py

import os
import uuid
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from components import (
    load_custom_css,
    display_logo,
    display_header,
    display_footer,
    create_input_form,
    show_input_instructions,
    display_warnings,
    log_dashboard_usage
)
from src.ml.predict import load_model, prepare_dataframe

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

# --- Charger les variables d'environnement ---
load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")
st.sidebar.markdown(f"🌐 **API utilisée :** `{API_URL}`")

# --- Génération d'un UUID unique par session ---
if "user_uuid" not in st.session_state:
    st.session_state["user_uuid"] = str(uuid.uuid4())
USER_ID = st.session_state["user_uuid"]

# --- Log d'utilisation (une seule fois par session) ---
if "logged" not in st.session_state:
    try:
        log_dashboard_usage(API_URL, USER_ID)
        st.session_state["logged"] = True
    except Exception:
        st.warning("⚠️ Impossible de logger l'utilisation du dashboard.")

# --- Affichage du nombre d'utilisateurs ---
try:
    import requests
    response = requests.get(f"{API_URL}/get_usage_count", timeout=10)
    if response.status_code == 200:
        data = response.json()
        user_count = data.get("unique_users_count", "N/A")
        st.sidebar.markdown(f"👥 **Utilisateurs uniques :** `{user_count}`")
    else:
        st.sidebar.markdown("👥 **Utilisateurs uniques :** `Service en cours...`")
except requests.exceptions.RequestException:
    st.sidebar.markdown("👥 **Utilisateurs uniques :** `Non disponible`")

# --- Champs d'entrée pour le modèle ---
BASE_COLS = ['cement', 'slag', 'fly_ash', 'water', 'superplasticizer',
             'coarse_aggregate', 'fine_aggregate', 'age']
DERIVED_COLS = ['water_cement_ratio', 'binder', 'fine_to_coarse_ratio']
ALL_FEATURES = BASE_COLS + DERIVED_COLS
show_input_instructions()

# --- Onglets ---
tab1, tab2, tab3 = st.tabs(["Prédiction individuelle", "Prédiction batch", "Évaluation modèle"])

# --- Charger le modèle globalement ---
MODEL_PATH = "models/best_model.joblib"
model = load_model(MODEL_PATH)

# --- Onglet 1 : Prédiction individuelle ---
with tab1:
    st.subheader("Entrez les paramètres du béton")
    input_features = create_input_form(BASE_COLS)

    if st.button("Prédire la résistance"):
        try:
            #payload = {"features": [input_features[col] for col in BASE_COLS]}
            payload = {"features": input_features}
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Résistance prédite : {result['predicted_strength_MPa']:.3f} MPa")
                display_warnings(result.get("warnings", [])) 
            else:
                st.error(f"Erreur API : {response.text}")
        except Exception as e:
            st.error(f"Erreur lors de l'appel API : {e}")

# --- Onglet 2 : Prédiction batch ---
with tab2:
    st.subheader("Import d’un fichier CSV pour prédiction en lot")
    uploaded = st.file_uploader("Chargez un fichier CSV", type="csv")

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.dataframe(df_batch.head())

        if st.button("Lancer la prédiction batch"):
            try:
                files = {"file": uploaded.getvalue()}
                response = requests.post(f"{API_URL}/predict-batch", files=files, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    preds = result["predicted_strengths_MPa"]
                    warnings_list = result["warnings"]

                    df_batch["predicted_strength"] = preds
                    df_batch["warnings"] = warnings_list

                    st.success(f"{len(df_batch)} prédictions générées")
                    st.dataframe(df_batch)
                    display_warnings(warnings_list)
                    st.download_button(
                        "Télécharger les résultats",
                        df_batch.to_csv(index=False),
                        "predictions_batch.csv",
                        "text/csv"
                    )
                else:
                    st.error(f"Erreur API : {response.text}")
            except Exception as e:
                st.error(f"Erreur lors de l'appel API : {e}")


# --- Onglet 3 : Évaluation modèle ---
with tab3:
    st.subheader("Évaluer le modèle avec un fichier CSV")
    st.markdown("Le CSV doit contenir les colonnes de base + une colonne **`strength`**")
    eval_file = st.file_uploader("Chargez le fichier CSV d'évaluation", type="csv")
    
    if eval_file:
        df_eval = pd.read_csv(eval_file)
        st.dataframe(df_eval.head())
        
        if st.button("Lancer l'évaluation"):
            try:
                if 'strength' not in df_eval.columns:
                    st.error("Le fichier doit contenir la colonne 'strength'.")
                else:
                    y_true = df_eval['strength']
                    df_prepared = prepare_dataframe(df_eval)
                    y_pred = model.predict(df_prepared)
                    
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    rmse = mean_squared_error(y_true, y_pred, squared=False)
                    mae = mean_absolute_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)
                    
                    st.metric("RMSE", round(rmse, 3))
                    st.metric("MAE", round(mae, 3))
                    st.metric("R²", round(r2, 3))
                    
                    df_eval['predicted_strength'] = y_pred
                    df_eval['abs_error'] = (y_true - y_pred).abs()
                    
                    st.download_button(
                        "Télécharger les résultats d'évaluation",
                        df_eval.to_csv(index=False),
                        "evaluation_results.csv",
                        "text/csv"
                    )
            except Exception as e:
                st.error(f"Erreur lors de l'évaluation : {e}")

# --- Footer ---
display_footer()
