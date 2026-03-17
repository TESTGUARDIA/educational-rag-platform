from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.api.rag import ingest_pdf, ask_question

app = FastAPI(title="Educational AI Platform", version="1.0")


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is running correctly"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    chunks = ingest_pdf(file_bytes, file.filename)

    return {"message": f"'{file.filename}' ingested successfully.", "chunks": chunks}


@app.post("/ask")
def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = ask_question(request.question)
    return result
