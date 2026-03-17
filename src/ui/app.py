import streamlit as st
import requests
import os

st.set_page_config(page_title="Educational AI", layout="wide")

api_url = os.getenv("API_URL", "http://localhost:8000")

st.title("🎓 Educational AI Platform")
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
st.header("📄 Importer un document")
uploaded_file = st.file_uploader("Choisissez un fichier PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("Indexer le document"):
        with st.spinner("Indexation en cours..."):
            try:
                response = requests.post(
                    f"{api_url}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"{data['message']} ({data['chunks']} chunks indexés)")
                else:
                    st.error(f"Erreur: {response.json().get('detail', 'Inconnue')}")
            except Exception as e:
                st.error(f"Impossible de contacter l'API: {e}")

st.markdown("---")

# --- Question answering ---
st.header("💬 Poser une question")
question = st.text_input("Votre question")

if st.button("Envoyer") and question:
    with st.spinner("Recherche en cours..."):
        try:
            response = requests.post(f"{api_url}/ask", json={"question": question})
            if response.status_code == 200:
                data = response.json()
                st.subheader("Réponse")
                st.write(data["answer"])

                if data.get("sources"):
                    with st.expander("Sources"):
                        for src in data["sources"]:
                            st.write(f"- **{src['source']}** — page {src['page']}")
            else:
                st.error(f"Erreur: {response.json().get('detail', 'Inconnue')}")
        except Exception as e:
            st.error(f"Impossible de contacter l'API: {e}")
