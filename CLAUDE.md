# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An Educational RAG (Retrieval-Augmented Generation) platform with a FastAPI backend and Streamlit frontend, orchestrated via Docker Compose. The stack uses LangChain + ChromaDB + OpenAI for RAG functionality and PyPDF for document ingestion.

## Commands

### Running the platform

```bash
# Start all services
docker-compose up

# Rebuild and start
docker-compose up --build

# Stop services
docker-compose down
```

### Local development (without Docker)

```bash
pip install -r requirements.txt

# API (hot-reload)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (separate terminal)
streamlit run src/ui/app.py --server.port=8501
```

## Architecture

Two-service Docker Compose setup:

- **API** (`src/api/main.py`) — FastAPI on port 8000. Entry point: `src.api.main:app`
- **Frontend** (`src/ui/app.py`) — Streamlit on port 8501. Calls the API via `API_URL` env var (defaults to `http://localhost:8000` locally, `http://api:8000` in Docker)

The frontend depends on the API service. Both containers mount `./src` as a volume for hot-reload during development. The API also mounts `./data` for document/vector storage persistence.

**RAG pipeline** (`src/api/rag.py`):
- `ingest_pdf()` — loads PDF bytes via PyPDF, splits with `RecursiveCharacterTextSplitter` (1000 chars / 200 overlap), embeds with `OpenAIEmbeddings`, persists to ChromaDB at `CHROMA_PATH` (default `./data/chroma_db/`)
- `ask_question()` — retrieves top-4 chunks from ChromaDB, runs `RetrievalQA` chain with `gpt-4o-mini`, returns answer + source metadata

API endpoints: `POST /upload` (PDF ingest) and `POST /ask` (question answering).

## Environment

Create a `.env` file at the project root (referenced by `docker-compose.yml`):

```env
OPENAI_API_KEY=sk-...
```
