import os
import json
import tempfile

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")


def ingest_and_generate_questions(file_bytes: bytes, filename: str) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    full_text = "\n\n".join(chunk.page_content for chunk in chunks[:10])

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    prompt = f"""Tu es un expert pédagogique. À partir du texte suivant extrait du document "{filename}", génère 10 couples question-réponse pertinents.

Retourne uniquement un tableau JSON valide, sans markdown ni explication, sous ce format exact :
[
  {{"question": "...", "reponse": "..."}},
  ...
]

Texte :
{full_text}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    # Enlever les blocs markdown si présents (```json ... ```)
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    return json.loads(content)
