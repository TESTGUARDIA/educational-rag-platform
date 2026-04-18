from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException

from src.api.rag import ingest_and_generate_questions

app = FastAPI(title="Educational AI Platform", version="1.0")


@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is running correctly"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/generate")
async def generate(files: List[UploadFile] = File(...)):
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} : seuls les fichiers PDF sont supportés.")

    results = []
    for file in files:
        file_bytes = await file.read()
        questions = ingest_and_generate_questions(file_bytes, file.filename)
        results.append({"filename": file.filename, "questions": questions})

    return {"results": results}
