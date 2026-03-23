import streamlit as st
import requests
import os

st.set_page_config(page_title="Educational AI", layout="wide")

api_url = os.getenv("API_URL", "http://localhost:8000")

st.title("Educational AI Platform")
st.markdown("---")

# --- Role selection ---
role = st.sidebar.selectbox("I am a...", ["Select a role", "Teacher", "Student"])

with st.sidebar:
    st.markdown("---")
    st.header("System Status")
    if st.button("Test API Connection"):
        try:
            response = requests.get(f"{api_url}/health")
            if response.status_code == 200:
                st.success("Backend Connected")
            else:
                st.error("Backend Error")
        except Exception:
            st.error("Backend Unreachable")

# =============================================================================
# TEACHER VIEW
# =============================================================================
if role == "Teacher":
    st.header("Teacher — Generate Q&A from a document")

    uploaded_file = st.file_uploader("Upload a PDF course document", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Generate 5 Questions & Answers"):
            with st.spinner("Generating Q&A pairs... this may take a moment"):
                try:
                    response = requests.post(
                        f"{api_url}/generate-qa",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Generated {len(data['pairs'])} Q&A pairs for '{data['filename']}'")
                        st.session_state["last_generated"] = data
                    else:
                        try:
                            detail = response.json().get('detail', 'Unknown error')
                        except Exception:
                            detail = response.text or f"HTTP {response.status_code}"
                        st.error(f"Error: {detail}")
                except Exception as e:
                    st.error(f"Could not reach API: {e}")

    # Show last generated Q&A
    if "last_generated" in st.session_state:
        st.markdown("---")
        st.subheader("Generated Q&A pairs")
        for i, pair in enumerate(st.session_state["last_generated"]["pairs"], 1):
            with st.expander(f"Question {i}"):
                st.markdown(f"**Q:** {pair['question']}")
                st.markdown(f"**A:** {pair['answer']}")

    # Show all past Q&A sets
    st.markdown("---")
    st.subheader("All Q&A sets")
    try:
        sets = requests.get(f"{api_url}/qa-sets").json()
        if sets:
            for qa_set in sets:
                with st.expander(f"{qa_set['filename']} — {qa_set['created_at']}"):
                    pairs = requests.get(f"{api_url}/qa-sets/{qa_set['id']}").json()["pairs"]
                    for i, pair in enumerate(pairs, 1):
                        st.markdown(f"**Q{i}:** {pair['question']}")
                        st.markdown(f"**A{i}:** {pair['answer']}")
                        st.markdown("---")
        else:
            st.info("No Q&A sets yet. Upload a document to generate some.")
    except Exception as e:
        st.error(f"Could not load Q&A sets: {e}")

# =============================================================================
# STUDENT VIEW (placeholder for next phase)
# =============================================================================
elif role == "Student":
    st.header("Student — Answer Questions")
    st.info("Student interface coming soon. Q&A grading and AI detection will be available here.")

# =============================================================================
# NO ROLE SELECTED
# =============================================================================
else:
    st.info("Please select your role from the sidebar to continue.")
