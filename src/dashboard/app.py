# src/dashboard/app.py - Version Corrigée (Render + Local)

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
    ensure_camel_case,
)

# ===================== Page config =====================
st.set_page_config(page_title="Concrete Strength Predictor", layout="wide", page_icon="🧱")

# ===================== Styles & Header =====================
load_custom_css()
display_logo()
display_header()

# ===================== ENV / API URL =====================
# Charge .env uniquement en local (évite d'écraser les variables Render)
if not os.getenv("RENDER"):  # Render définit des variables d'env spécifiques
    load_dotenv()

# Priorité Render: API_HOST -> API_URL -> fallback local
_api_host = os.getenv("API_HOST", "").strip()
_api_url_env = os.getenv("API_URL", "").strip()
if _api_host:
    API_URL = f"https://{_api_host}"
elif _api_url_env:
    API_URL = _api_url_env
else:
    API_URL = "http://127.0.0.1:8000"

st.sidebar.markdown(f"🌐 **API utilisée :** `{API_URL}`")

# Session HTTP réutilisable
SESSION = requests.Session()

# ===================== User ID & logging =====================
if "user_uuid" not in st.session_state:
    st.session_state["user_uuid"] = str(uuid.uuid4())
USER_ID = st.session_state["user_uuid"]

if "logged" not in st.session_state:
    try:
        SESSION.post(f"{API_URL}/log_dashboard_usage", json={"user_id": USER_ID}, timeout=5)
        st.session_state["logged"] = True
    except requests.RequestException:
        st.session_state["logged"] = False  # silencieux

# ===================== Compteur d'utilisateurs =====================
try:
    r = SESSION.get(f"{API_URL}/get_usage_count", timeout=5)
    if r.ok:
        user_count = r.json().get("unique_users_count", "N/A")
        st.sidebar.markdown(f"👥 **Utilisateurs uniques :** `{user_count}`")
except requests.RequestException:
    st.sidebar.markdown("👥 **Utilisateurs uniques :** `Non disponible`")

# ===================== Instructions =====================
show_input_instructions(INPUT_NAMES_CAMEL)

# ===================== Tabs =====================
tab1, tab2, tab3 = st.tabs(["Prédiction unique", "Prédiction batch (CSV)", "Évaluation batch"])

# ============ Tab 1 : Prédiction unique ============
with tab1:
    st.subheader("Entrez les paramètres du béton (8 features)")
    input_features_dict = create_input_form(INPUT_NAMES_CAMEL)

    if st.button("Prédire la résistance"):
        try:
            payload = {"samples": [input_features_dict]}
            with st.spinner("Prédiction en cours..."):
                resp = SESSION.post(f"{API_URL}/predict", json=payload, timeout=15)
            if resp.ok:
                result = resp.json()
                pred = result["predicted_strengths_MPa"][0]
                warnings = result.get("warnings", [[]])[0]
                st.success(f"Résistance prédite : **{pred:.3f} MPa**")
                display_warnings(warnings)  # accepte liste plate
            else:
                st.error(f"❌ Erreur API ({resp.status_code}): {resp.text}")
        except Exception as e:
            st.error(f"❌ Erreur lors de l'appel API : {e}")

# ============ Tab 2 : Prédiction batch ============
with tab2:
    st.subheader("Import d’un fichier CSV pour prédiction en lot")
    uploaded_file = st.file_uploader("Chargez un fichier CSV", type="csv", key="batch_upload")

    if uploaded_file:
        # Lecture robuste (BOM + détection ; ou ; )
        df_batch_original = pd.read_csv(uploaded_file, encoding="utf-8-sig", sep=None, engine="python")
        st.dataframe(df_batch_original.head())

        if st.button("Lancer la prédiction batch"):
            try:
                payload = convert_df_to_api_json(df_batch_original.copy())
                with st.spinner("Prédictions en cours..."):
                    resp = SESSION.post(f"{API_URL}/predict", json=payload, timeout=30)
                if resp.ok:
                    result = resp.json()
                    preds = result["predicted_strengths_MPa"]
                    warnings_list = result.get("warnings", [])

                    df_results = df_batch_original.copy()
                    df_results["Predicted_Strength_MPa"] = preds
                    df_results["Warnings_Audit"] = [str(w) if w else "" for w in warnings_list]

                    st.success(f"✅ {len(df_results)} prédictions générées.")
                    st.dataframe(df_results)
                    # Affiche un récap (la fonction gère liste ou liste de listes)
                    display_warnings(warnings_list)

                    st.download_button(
                        "⬇️ Télécharger les résultats",
                        df_results.to_csv(index=False).encode("utf-8-sig"),
                        "predictions_batch_results.csv",
                        "text/csv",
                    )
                else:
                    st.error(f"❌ Erreur API ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"❌ Erreur lors de l'appel API : {e}")

# ============ Tab 3 : Évaluation ============
with tab3:
    st.subheader("Évaluer le modèle sur des données labellisées")
    st.markdown(
        "Le CSV doit contenir les 8 colonnes de base (casse libre) **+** la cible "
        "(`true_strength`, `strength` ou `Strength`)."
    )
    eval_file = st.file_uploader("Chargez le fichier CSV d'évaluation", type="csv", key="eval_upload")

    if eval_file:
        df_eval_original = pd.read_csv(eval_file, encoding="utf-8-sig", sep=None, engine="python")
        st.dataframe(df_eval_original.head())

        if st.button("Lancer l'évaluation"):
            try:
                # Copie pour travail
                df_eval = ensure_camel_case(df_eval_original.copy(), INPUT_NAMES_CAMEL)

                # Normalise la cible
                if "true_strength" not in df_eval.columns:
                    if "strength" in df_eval.columns:
                        df_eval.rename(columns={"strength": "true_strength"}, inplace=True)
                    elif "Strength" in df_eval.columns:
                        df_eval.rename(columns={"Strength": "true_strength"}, inplace=True)
                    else:
                        st.error(
                            "❌ Le fichier doit contenir la colonne cible 'true_strength' (ou 'strength' / 'Strength')."
                        )
                        st.stop()

                # Cols requises pour l'API /evaluate
                df_temp = df_eval[INPUT_NAMES_CAMEL + ["true_strength"]].copy()

                payload = {"samples": df_temp.to_dict(orient="records")}
                with st.spinner("Évaluation en cours..."):
                    resp = SESSION.post(f"{API_URL}/evaluate", json=payload, timeout=30)

                if resp.ok:
                    result = resp.json()

                    st.success(f"✅ Évaluation réussie sur {result['n_samples']} échantillons.")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("RMSE", f"{result['rmse']:.3f}")
                    c2.metric("MAE", f"{result['mae']:.3f}")
                    c3.metric("R²", f"{result['r2']:.3f}")

                    predictions = result.get("predicted_strengths_MPa", [])
                    warnings_list = result.get("warnings", [])

                    df_results = df_eval_original.copy()
                    if predictions:
                        df_results["Predicted_Strength_MPa"] = predictions
                    df_results["Warnings_Audit"] = [str(w) if w else "" for w in warnings_list]

                    st.markdown("### 📊 Résultats détaillés par ligne")
                    st.dataframe(df_results)

                    # Affichage warnings ligne par ligne
                    display_warnings_by_line(warnings_list, "⚠️ Avertissements détectés par ligne")

                    st.download_button(
                        "⬇️ Télécharger les résultats d'évaluation",
                        df_results.to_csv(index=False).encode("utf-8-sig"),
                        "evaluation_results.csv",
                        "text/csv",
                    )
                else:
                    st.error(f"❌ Erreur API ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"❌ Erreur lors de l'appel API : {e}")

# ===================== Footer =====================
display_footer()
