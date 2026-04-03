import streamlit as st
import requests
import os

st.set_page_config(page_title="Educational AI", layout="wide")

api_url = os.getenv("API_URL", "http://localhost:8000")

st.title("Générateur de Questions")
st.markdown("---")

# Sidebar — system status
with st.sidebar:
    st.header("Statut Système")
    if st.button("Tester Connexion API"):
        try:
            response = requests.get(f"{api_url}/health")
            if response.status_code == 200:
                st.success("Backend Connecté ✅")
            else:
                st.error("Backend Erreur ❌")
        except Exception:
            st.error("Backend Inaccessible ⚠️")

# --- Document upload ---
st.header("Importer des documents PDF")
uploaded_files = st.file_uploader("Choisissez un ou plusieurs fichiers PDF", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Générer les questions"):
        with st.spinner("Génération en cours..."):
            try:
                files_payload = [
                    ("files", (f.name, f.getvalue(), "application/pdf"))
                    for f in uploaded_files
                ]
                response = requests.post(f"{api_url}/generate", files=files_payload)
                if response.status_code == 200:
                    data = response.json()
                    for item in data["results"]:
                        st.success(f"Questions générées pour : **{item['filename']}**")
                        st.markdown("---")
                        st.subheader(f"Questions / Réponses — {item['filename']}")
                        for i, qa in enumerate(item["questions"], 1):
                            st.markdown(f"**Q{i} : {qa['question']}**")
                            st.markdown(f"*R : {qa['reponse']}*")
                            st.markdown("")
                else:
                    try:
                        detail = response.json().get("detail", response.text)
                    except Exception:
                        detail = response.text or f"Code HTTP {response.status_code}"
                    st.error(f"Erreur {response.status_code}: {detail}")
            except Exception as e:
                st.error(f"Impossible de contacter l'API: {e}")
