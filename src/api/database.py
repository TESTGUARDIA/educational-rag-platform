import os
import sqlite3
import uuid
from datetime import datetime

DB_PATH = os.getenv("DB_PATH", "./data/questions.db")


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                file_path TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                question TEXT NOT NULL,
                reponse TEXT NOT NULL,
                source TEXT,
                raisonnement TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        # Migrations : ajoute les colonnes si les tables existaient déjà sans elles
        try:
            conn.execute("ALTER TABLE documents ADD COLUMN file_path TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE questions ADD COLUMN source TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE questions ADD COLUMN raisonnement TEXT")
        except Exception:
            pass
        conn.commit()


def save_document(filename: str, file_path: str = None) -> str:
    """Sauvegarde un document sans questions (juste le fichier)."""
    doc_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO documents (id, filename, created_at, file_path) VALUES (?, ?, ?, ?)",
            (doc_id, filename, created_at, file_path),
        )
        conn.commit()
    return doc_id


def save_questions(filename: str, qa_pairs: list[dict], file_path: str = None) -> str:
    doc_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO documents (id, filename, created_at, file_path) VALUES (?, ?, ?, ?)",
            (doc_id, filename, created_at, file_path),
        )
        conn.executemany(
            "INSERT INTO questions (id, document_id, question, reponse, source, raisonnement) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (str(uuid.uuid4()), doc_id, qa["question"], qa["reponse"], qa.get("source"), qa.get("raisonnement"))
                for qa in qa_pairs
            ],
        )
        conn.commit()
    return doc_id


def list_documents() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("""
            SELECT d.id, d.filename, d.created_at, d.file_path, COUNT(q.id) as question_count
            FROM documents d
            LEFT JOIN questions q ON q.document_id = d.id
            GROUP BY d.id
            ORDER BY d.created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_document(document_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, filename, file_path FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
    return dict(row) if row else None


def get_questions(document_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, question, reponse, source, raisonnement FROM questions WHERE document_id = ?",
            (document_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_document(document_id: str) -> bool:
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM documents WHERE id = ?", (document_id,)
        )
        conn.commit()
    return cursor.rowcount > 0
