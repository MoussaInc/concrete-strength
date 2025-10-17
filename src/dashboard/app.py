# src/dashboard/app.py

import os
import uuid
import time
import pandas as pd
import streamlit as st
import requests
from dotenv import load_dotenv
import numpy as np

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

# ===================== Helpers =====================
def is_full_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

def build_render_url_from_host(host_or_url: str) -> str:
    """
    Avec render.yaml (property: url), API_HOST peut déjà être une URL complète.
    Sinon, si c'est un hostname (ex: 'concrete-api'), on génère https://<host>.onrender.com
    """
    host_or_url = (host_or_url or "").strip()
    if not host_or_url:
        return ""
    if is_full_url(host_or_url):
        return host_or_url
    if "." not in host_or_url:
        host_or_url = f"{host_or_url}.onrender.com"
    return f"https://{host_or_url}"

def resolve_api_url() -> str:
    """
    Priorité:
      1) API_URL (override manuel complet ou hostname)
      2) API_HOST (injection Render via render.yaml -> fromService.url ou host)
      3) fallback local
    """
    # En local (pas sur Render), on lit .env si présent
    if not os.getenv("RENDER"):
        load_dotenv()

    api_url_env = os.getenv("API_URL", "").strip()
    if api_url_env:
        return api_url_env if is_full_url(api_url_env) else build_render_url_from_host(api_url_env)

    api_host = os.getenv("API_HOST", "").strip()
    if api_host:
        return build_render_url_from_host(api_host)

    return "http://127.0.0.1:8000"

def make_session_with_retries(total=5, backoff=0.6) -> requests.Session:
    """
    Session requests avec retries sur 502/503/504 pour absorber le cold start Render.
    """
    retry = Retry(
        total=total,
        connect=total,
        read=total,
        status=total,
        backoff_factor=backoff,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"]),
        raise_on_status=False,
    )
    s = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def warm_api(api_url: str, session: requests.Session):
    """
    Réveille l'API: GET /health puis POST /warmup (si exposé côté API).
    Silencieux en cas d'échec.
    """
    try:
        session.get(f"{api_url}/health", timeout=5)
    except requests.RequestException:
        pass
    try:
        session.post(f"{api_url}/warmup", timeout=15)
    except requests.RequestException:
        pass

# ===================== API URL & HTTP session =====================
API_URL = resolve_api_url()
SESSION = make_session_with_retries()

# Affiche l'URL utilisée en sidebar
st.sidebar.markdown(f"🌐 **API utilisée :** `{API_URL}`")

# Warmup discret (cold start)
warm_api(API_URL, SESSION)

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
def fetch_usage_count(api_url: str, tries: int = 3, timeout: float = 6.0):
    last_err = None
    for i in range(tries):
        try:
            r = SESSION.get(f"{api_url}/get_usage_count", timeout=timeout)
            if r.ok:
                return r.json().get("unique_users_count", "N/A")
            last_err = f"HTTP {r.status_code} - {r.text[:120]}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(0.6 * (i + 1))
    return f"ERR: {last_err}"

count_or_err = fetch_usage_count(API_URL)
if isinstance(count_or_err, (int, float)) or (isinstance(count_or_err, str) and count_or_err.isdigit()):
    st.sidebar.markdown(f"👥 **Utilisateurs uniques :** `{count_or_err}`")
elif isinstance(count_or_err, str) and count_or_err.startswith("ERR:"):
    st.sidebar.markdown("👥 **Utilisateurs uniques :** `Non disponible`")
else:
    st.sidebar.markdown("👥 **Utilisateurs uniques :** `N/A`")

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
                resp = SESSION.post(f"{API_URL}/predict", json=payload, timeout=30)
            if resp.ok:
                result = resp.json()
                pred = result["predicted_strengths_MPa"][0]
                warnings = result.get("warnings", [[]])[0]
                st.success(f"Résistance prédite : **{pred:.3f} MPa**")
                display_warnings(warnings)
            else:
                st.error(f"❌ Erreur API ({resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            st.error(f"❌ Erreur lors de l'appel API : {e}")

# ============ Tab 2 : Prédiction batch ============
with tab2:
    st.subheader("Import d’un fichier CSV pour prédiction en lot")
    uploaded_file = st.file_uploader("Chargez un fichier CSV", type="csv", key="batch_upload")

    if uploaded_file:
        df_batch_original = pd.read_csv(uploaded_file, encoding="utf-8-sig", sep=None, engine="python")
        st.dataframe(df_batch_original.head())

        if st.button("Lancer la prédiction batch"):
            try:
                payload = convert_df_to_api_json(df_batch_original.copy())
                with st.spinner("Prédictions en cours..."):
                    resp = SESSION.post(f"{API_URL}/predict", json=payload, timeout=60)
                if resp.ok:
                    result = resp.json()
                    preds = result["predicted_strengths_MPa"]
                    warnings_list = result.get("warnings", [])

                    df_results = df_batch_original.copy()
                    df_results["Predicted_Strength_MPa"] = preds
                    df_results["Warnings_Audit"] = [str(w) if w else "" for w in warnings_list]

                    st.success(f"✅ {len(df_results)} prédictions générées.")
                    st.dataframe(df_results)
                    display_warnings(warnings_list)

                    st.download_button(
                        "⬇️ Télécharger les résultats",
                        df_results.to_csv(index=False).encode("utf-8-sig"),
                        "predictions_batch_results.csv",
                        "text/csv",
                    )
                else:
                    st.error(f"❌ Erreur API ({resp.status_code}): {resp.text[:200]}")
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
                df_eval = ensure_camel_case(df_eval_original.copy(), INPUT_NAMES_CAMEL)
                if "true_strength" not in df_eval.columns:
                    if "strength" in df_eval.columns:
                        df_eval.rename(columns={"strength": "true_strength"}, inplace=True)
                    elif "Strength" in df_eval.columns:
                        df_eval.rename(columns={"Strength": "true_strength"}, inplace=True)
                    else:
                        st.error("❌ Le fichier doit contenir la colonne cible 'true_strength' (ou 'strength' / 'Strength').")
                        st.stop()

                df_temp = df_eval[INPUT_NAMES_CAMEL + ["true_strength"]].copy()
                payload = {"samples": df_temp.to_dict(orient="records")}
                with st.spinner("Évaluation en cours..."):
                    resp = SESSION.post(f"{API_URL}/evaluate", json=payload, timeout=60)

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

                    display_warnings_by_line(warnings_list, "⚠️ Avertissements détectés par ligne")

                    st.download_button(
                        "⬇️ Télécharger les résultats d'évaluation",
                        df_results.to_csv(index=False).encode("utf-8-sig"),
                        "evaluation_results.csv",
                        "text/csv",
                    )
                else:
                    st.error(f"❌ Erreur API ({resp.status_code}): {resp.text[:200]}")
            except Exception as e:
                st.error(f"❌ Erreur lors de l'appel API : {e}")

# ===================== Footer =====================
display_footer()
