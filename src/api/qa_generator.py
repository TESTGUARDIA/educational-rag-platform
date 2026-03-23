import os
import json
import sqlite3
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

DB_PATH = os.getenv("DB_PATH", "./data/qa.db")


def _get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS qa_sets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS qa_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            set_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            FOREIGN KEY (set_id) REFERENCES qa_sets(id)
        )
    """)
    conn.commit()
    return conn


def generate_qa(file_bytes: bytes, filename: str, num_questions: int = 5) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Use the first few chunks as context (avoid exceeding token limits)
    context = "\n\n".join(c.page_content for c in chunks[:5])

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

    prompt = f"""You are an educational assistant. Based on the course content below, generate exactly {num_questions} open-ended questions and their detailed reference answers.

Rules:
- Questions must require written, analytical answers (no yes/no, no multiple choice)
- Answers must be thorough and based strictly on the provided content
- Return ONLY a valid JSON array, no extra text

Format:
[
  {{"question": "...", "answer": "..."}},
  ...
]

Course content:
{context}
"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Strip markdown code blocks if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    pairs = json.loads(raw)

    conn = _get_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO qa_sets (filename) VALUES (?)", (filename,))
    set_id = cursor.lastrowid
    for pair in pairs:
        cursor.execute(
            "INSERT INTO qa_pairs (set_id, question, answer) VALUES (?, ?, ?)",
            (set_id, pair["question"], pair["answer"]),
        )
    conn.commit()
    conn.close()

    return {"set_id": set_id, "filename": filename, "pairs": pairs}


def get_all_qa_sets() -> list:
    conn = _get_db()
    cursor = conn.execute("SELECT id, filename, created_at FROM qa_sets ORDER BY created_at DESC")
    sets = [{"id": r[0], "filename": r[1], "created_at": r[2]} for r in cursor.fetchall()]
    conn.close()
    return sets


def get_qa_pairs(set_id: int) -> list:
    conn = _get_db()
    cursor = conn.execute(
        "SELECT id, question, answer FROM qa_pairs WHERE set_id = ?", (set_id,)
    )
    pairs = [{"id": r[0], "question": r[1], "answer": r[2]} for r in cursor.fetchall()]
    conn.close()
    return pairs
