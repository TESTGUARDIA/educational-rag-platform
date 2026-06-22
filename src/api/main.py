import os
import uuid
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.api.rag import ingest_and_generate_questions
from src.api.database import (
    init_db, save_document, save_questions,
    list_documents, get_document, get_questions, delete_document,
)

UPLOADS_DIR = os.getenv("UPLOADS_DIR", "./data/uploads")

app = FastAPI(title="Educational AI Platform", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()
    os.makedirs(UPLOADS_DIR, exist_ok=True)


@app.get("/app", include_in_schema=False)
def serve_frontend():
    return FileResponse("src/web/web.html")


@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is running correctly"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


def _save_file(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """Sauvegarde le PDF sur disque, retourne (doc_id, file_path)."""
    doc_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOADS_DIR, f"{doc_id}.pdf")
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    return doc_id, file_path


# --- Sauvegarde de cours (sans génération) ---

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """Sauvegarde les PDF pour que les étudiants puissent les télécharger."""
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} : seuls les PDF sont supportés.")

    results = []
    for file in files:
        file_bytes = await file.read()
        _, file_path = _save_file(file_bytes, file.filename)
        doc_id = save_document(file.filename, file_path)
        results.append({"document_id": doc_id, "filename": file.filename})

    return {"results": results}


# --- Génération de questions ---

@app.post("/generate")
async def generate(files: List[UploadFile] = File(...)):
    """Sauvegarde les PDF et génère les questions d'évaluation."""
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} : seuls les PDF sont supportés.")

    results = []
    for file in files:
        file_bytes = await file.read()
        _, file_path = _save_file(file_bytes, file.filename)
        questions = ingest_and_generate_questions(file_bytes, file.filename)
        doc_id = save_questions(file.filename, questions, file_path)
        results.append({"document_id": doc_id, "filename": file.filename, "questions": questions})

    return {"results": results}


# --- Endpoints documents ---

@app.get("/documents")
def documents():
    return {"documents": list_documents()}


@app.get("/documents/{document_id}/questions")
def questions_by_document(document_id: str):
    items = get_questions(document_id)
    if not items:
        raise HTTPException(status_code=404, detail="Document introuvable ou sans questions.")
    return {"document_id": document_id, "questions": items}


@app.get("/documents/{document_id}/file")
def download_document(document_id: str):
    """Télécharge le PDF original."""
    doc = get_document(document_id)
    if not doc or not doc.get("file_path") or not os.path.exists(doc["file_path"]):
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    return FileResponse(doc["file_path"], media_type="application/pdf", filename=doc["filename"])


@app.delete("/documents/{document_id}")
def remove_document(document_id: str):
    doc = get_document(document_id)
    if not delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document introuvable.")
    if doc and doc.get("file_path") and os.path.exists(doc["file_path"]):
        os.remove(doc["file_path"])
    return {"status": "deleted", "document_id": document_id}
