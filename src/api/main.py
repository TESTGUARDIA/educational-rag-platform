from fastapi import FastAPI

app = FastAPI(title="Educational AI Platform", version="1.0")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is running correctly"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}