import streamlit as st
import requests
import os

# Configuration de la page
st.set_page_config(page_title="Educational AI", layout="wide")

st.title("🎓 Educational AI Platform")
st.markdown("---")

# Zone de statut pour vérifier la connexion avec le backend
api_url = os.getenv("API_URL", "http://localhost:8000")

with st.sidebar:
    st.header("Status Système")
    if st.button("Test Connexion API"):
        try:
            response = requests.get(f"{api_url}/health")
            if response.status_code == 200:
                st.success("Backend Connecté ✅")
            else:
                st.error("Backend Erreur ❌")
        except:
            st.error("Backend Inaccessible ⚠️")

st.info("Bienvenue sur la plateforme. Le système est prêt.")