# 🔹 src/dashboard/components.py

import os
import streamlit as st
import requests
import base64
from dotenv import load_dotenv

def log_dashboard_usage(api_url):
    """
    Envoie une requête à l'API pour logguer l'utilisation du dashboard.
    """
    log_endpoint = f"{api_url}/log_dashboard_usage"
    try:
        # L'adresse IP sera capturée par le middleware de l'API.
        requests.post(log_endpoint, timeout=5)
    except requests.exceptions.RequestException as e:
        # Affiche l'erreur dans la console sans l'afficher à l'utilisateur
        print(f"Erreur lors de la journalisation de l'utilisation du dashboard : {e}")

def load_custom_css(css_path="src/dashboard/static/style.css"):
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"Fichier CSS non trouvé à : {css_path}")

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

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def display_header():
    st.markdown("""
    <div style="text-align: center; max-width: 700px; margin: auto;">
        <h1 style="color: #2E3B4E;">Concrete Strength Predictor</h1>
        <h4 style="color: #4a5a70; font-style: italic;">Prédiction de la résistance du béton (MPa)</h4>
    </div>
    <hr style="border-top: 3px solid #f68b1e; width: 50%; margin: 1rem auto;" />
    """, unsafe_allow_html=True)

def display_footer():
    st.markdown("""
    <hr style="border-top: 2px solid #2E3B4E"/>
    <div style="text-align:center; color:gray; font-size: 0.9rem;">
        © 2025 <a href="https://poussaim.org" target="_blank">Moussa MBALLO</a> — Tous droits réservés.
    </div>
    """, unsafe_allow_html=True)


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


# -------------------------------
# 🔹 Fonctions utilitaires warnings
# -------------------------------
def display_warnings(warnings):
    """
    Affiche joliment les warnings remontés par l'API.
    """
    if warnings:
        st.warning("⚠️ Attention :")
        for w in warnings:
            st.markdown(f"- {w}")


# -------------------------------
# 🔹 Fonctions API mises à jour
# -------------------------------

def call_prediction_api(api_url, features):
    """
    Requête POST pour prédiction individuelle. Récupère également les warnings.
    """
    try:
        response = requests.post(f"{api_url}/predict", json={"features": features})
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "value": data.get("predicted_strength_MPa"),
                "warnings": data.get("warnings", [])
            }
        else:
            return {
                "success": False,
                "message": response.json().get("detail", f"Erreur API ({response.status_code})")
            }
    except Exception as e:
        return {"success": False, "message": str(e)}

def call_batch_prediction_api(api_url, file):
    """
    Requête POST pour prédiction batch. Récupère également les warnings.
    """
    try:
        files = {"file": (file.name, file.getvalue(), "text/csv")}
        response = requests.post(f"{api_url}/predict-batch", files=files)
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "predictions": data.get("predicted_strengths_MPa", []),
                "warnings": data.get("warnings", [])
            }
        else:
            return {
                "success": False,
                "message": response.json().get("detail", f"Erreur API ({response.status_code})")
            }
    except Exception as e:
        return {"success": False, "message": str(e)}

def call_evaluation_api(api_url, eval_list):
    try:
        response = requests.post(f"{api_url}/evaluate", json=eval_list)
        if response.status_code == 200:
            return {"success": True, "metrics": response.json(), "message": None}
        else:
            return {
                "success": False,
                "metrics": None,
                "message": response.json().get("detail", f"Erreur API ({response.status_code})")
            }
    except Exception as e:
        return {"success": False, "metrics": None, "message": str(e)}
