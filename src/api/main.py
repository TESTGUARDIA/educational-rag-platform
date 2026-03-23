from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.api.rag import ingest_pdf, ask_question
from src.api.qa_generator import generate_qa, get_all_qa_sets, get_qa_pairs

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


@app.post("/generate-qa")
async def generate_qa_endpoint(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    result = generate_qa(file_bytes, file.filename)
    return result


@app.get("/qa-sets")
def list_qa_sets():
    return get_all_qa_sets()


@app.get("/qa-sets/{set_id}")
def get_qa_set(set_id: int):
    pairs = get_qa_pairs(set_id)
    if not pairs:
        raise HTTPException(status_code=404, detail="Q&A set not found.")
    return {"set_id": set_id, "pairs": pairs}
